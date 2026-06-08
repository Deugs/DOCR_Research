"""
marl_agents.py
──────────────
Independent PPO (IPPO) with Centralised Value Baseline.

Phase 1 Refactor: Enhanced for dynamic network conditions
─────────────────────────────────────────────────────────
- Supports curriculum learning (6 PV stages)
- Handles topology changes in observations
- Sensor noise robustness
- Multi-fault-type generalization
- Enhanced critic with global state [TMS_all || PS_all || PV_level || topology_flag]

Architecture
────────────
- Each relay agent owns a GaussianActor (decentralised policy).
- A shared CentralCritic conditions on global state for variance reduction (CTDE).
- Policy gradient via OpenAI-ES finite-difference surrogate (pure numpy).
- Maps 1-to-1 onto a PyTorch/JAX implementation for production use.
"""

import numpy as np
from typing import Tuple, List, Dict, Optional
import time

from network.network_data import NetworkData
from environment.dynamic_network import DistributionNetwork


# ── Tiny numpy MLP ────────────────────────────────────────────────────────────
def relu(x):  return np.maximum(0, x)
def tanh(x):  return np.tanh(x)


class MLP:
    """2-hidden-layer MLP: in → h1 → h2 → out."""

    def __init__(self, in_dim, hidden, out_dim, seed=0):
        rng  = np.random.default_rng(seed)
        s    = lambda a: np.sqrt(2.0 / a)
        self.W1 = rng.normal(0, s(in_dim),  (hidden, in_dim))
        self.b1 = np.zeros(hidden)
        self.W2 = rng.normal(0, s(hidden),  (hidden, hidden))
        self.b2 = np.zeros(hidden)
        self.W3 = rng.normal(0, s(hidden),  (out_dim, hidden))
        self.b3 = np.zeros(out_dim)

    def forward(self, x):
        h1 = relu(self.W1 @ x + self.b1)
        h2 = relu(self.W2 @ h1 + self.b2)
        return self.W3 @ h2 + self.b3

    def get_params(self):
        return np.concatenate([
            self.W1.ravel(), self.b1,
            self.W2.ravel(), self.b2,
            self.W3.ravel(), self.b3,
        ])

    def set_params(self, p):
        sizes  = [self.W1.size, self.b1.size,
                  self.W2.size, self.b2.size,
                  self.W3.size, self.b3.size]
        chunks = np.split(p, np.cumsum(sizes[:-1]))
        self.W1 = chunks[0].reshape(self.W1.shape)
        self.b1 = chunks[1]
        self.W2 = chunks[2].reshape(self.W2.shape)
        self.b2 = chunks[3]
        self.W3 = chunks[4].reshape(self.W3.shape)
        self.b3 = chunks[5]

    def n_params(self):
        return (self.W1.size + self.b1.size + self.W2.size
                + self.b2.size + self.W3.size + self.b3.size)


# ── Gaussian Actor ────────────────────────────────────────────────────────────
class GaussianActor:
    def __init__(self, obs_dim, act_dim, hidden=32, seed=0):
        self.act_dim = act_dim
        self.net     = MLP(obs_dim, hidden, act_dim, seed)
        self.log_std = np.full(act_dim, -1.0)

    def get_mean(self, obs):
        return tanh(self.net.forward(obs))

    def sample(self, obs, rng) -> Tuple[np.ndarray, float]:
        mean  = self.get_mean(obs)
        std   = np.exp(np.clip(self.log_std, -3, 0))
        noise = rng.normal(0, 1, self.act_dim)
        action = np.clip(mean + std * noise, -1.0, 1.0)
        log_prob = float(np.sum(
            -0.5 * (noise**2 + 2*self.log_std + np.log(2*np.pi))))
        return action, log_prob

    def log_prob(self, obs, action) -> float:
        mean = self.get_mean(obs)
        std  = np.exp(np.clip(self.log_std, -3, 0))
        z    = (action - mean) / (std + 1e-8)
        return float(np.sum(
            -0.5 * (z**2 + 2*self.log_std + np.log(2*np.pi))))

    def get_all_params(self):
        return np.concatenate([self.net.get_params(), self.log_std])

    def set_all_params(self, p):
        n = self.net.n_params()
        self.net.set_params(p[:n])
        self.log_std = p[n:n + self.act_dim]


# ── Centralised Critic ────────────────────────────────────────────────────────
class CentralCritic:
    """
    Value network on global state [TMS_all || PS_all || PV_level || topology || fault_type].
    Phase 1: Enhanced to handle dynamic network conditions.
    """
    def __init__(self, n_relays, hidden=64, seed=0, use_pv_features=True):
        # Global state: TMS (n_relays) + PS (n_relays) + PV_level (1) + topology_flag (1) + fault_type_enc (4)
        global_dim = 2 * n_relays + (6 if use_pv_features else 0)
        self.net = MLP(global_dim, hidden, 1, seed)
        self.use_pv_features = use_pv_features
        self.n_relays = n_relays

    def value(self, TMS, PS, pv_level=0.0, topology_flag=0.0, fault_type_idx=0) -> float:
        global_state = np.concatenate([TMS, PS])
        if self.use_pv_features:
            # One-hot encode fault type (4 types: 3ph, LL, 1ph, 2ph-g)
            fault_onehot = np.zeros(4)
            if 0 <= fault_type_idx < 4:
                fault_onehot[fault_type_idx] = 1.0
            global_state = np.concatenate([
                global_state,
                [pv_level, topology_flag],
                fault_onehot,
            ])
        return float(self.net.forward(global_state)[0])


# ── Rollout Buffer ────────────────────────────────────────────────────────────
class RolloutBuffer:
    def __init__(self, n_agents, obs_dim, act_dim, capacity):
        self.obs       = np.zeros((capacity, n_agents, obs_dim))
        self.actions   = np.zeros((capacity, n_agents, act_dim))
        self.rewards   = np.zeros((capacity, n_agents))
        self.log_probs = np.zeros((capacity, n_agents))
        self.values    = np.zeros(capacity)
        self.dones     = np.zeros(capacity, dtype=bool)
        self.pv_levels = np.zeros(capacity)  # Phase 1: track PV level
        self.top_flags = np.zeros(capacity)  # Phase 1: track topology
        self.ptr = self.size = 0
        self.capacity  = capacity
        self.n_agents  = n_agents

    def add(self, obs, actions, rewards, log_probs, value, done, pv_level=0.0, topology_flag=0.0):
        i = self.ptr % self.capacity
        self.obs[i]       = obs
        self.actions[i]   = actions
        self.rewards[i]   = rewards
        self.log_probs[i] = log_probs
        self.values[i]    = value
        self.dones[i]     = done
        self.pv_levels[i] = pv_level
        self.top_flags[i] = topology_flag
        self.ptr  += 1
        self.size  = min(self.ptr, self.capacity)

    def compute_gae(self, gamma=0.99, lam=0.95,
                    last_value=0.0) -> Tuple[np.ndarray, np.ndarray]:
        """Generalised Advantage Estimation."""
        T   = self.size
        adv = np.zeros((T, self.n_agents))
        ret = np.zeros((T, self.n_agents))
        for ag in range(self.n_agents):
            gae = 0.0
            for t in reversed(range(T)):
                Vn    = (last_value if (t == T-1 or self.dones[t])
                         else self.values[t+1])
                delta = (self.rewards[t, ag]
                         + gamma * Vn * (1 - self.dones[t])
                         - self.values[t])
                gae   = delta + gamma * lam * (1 - self.dones[t]) * gae
                adv[t, ag] = gae
                ret[t, ag] = gae + self.values[t]
            adv[:, ag] = ((adv[:, ag] - adv[:, ag].mean())
                          / (adv[:, ag].std() + 1e-8))
        return adv, ret

    def clear(self):
        self.ptr = self.size = 0


# ── MARL Trainer (IPPO + centralised value baseline) ─────────────────────────
class MARLTrainer:
    """
    Phase 1 Enhanced IPPO Trainer.

    Parameters
    ──────────
    env           DynamicDOCREnv instance
    hidden_actor  hidden units per actor MLP
    hidden_critic hidden units for shared critic
    lr_actor      ES step size for actor update
    clip_eps      PPO clip ratio ε
    gamma / lam   discount and GAE parameters
    rollout_len   steps collected before each update
    n_epochs      ES perturbation rounds per rollout
    curriculum    Enable curriculum learning (6 stages)
    """

    def __init__(
        self,
        env,
        hidden_actor  = 32,
        hidden_critic = 64,
        lr_actor      = 0.02,
        clip_eps      = 0.2,
        gamma         = 0.99,
        lam           = 0.95,
        rollout_len   = 64,
        n_epochs      = 4,
        seed          = 0,
        curriculum    = False,    # Phase 1: curriculum learning
        curriculum_stages = 6,      # Phase 1: 6 PV levels
    ):
        self.env         = env
        self.n_agents    = env.n_agents
        self.clip_eps    = clip_eps
        self.gamma       = gamma
        self.lam         = lam
        self.rollout_len = rollout_len
        self.n_epochs    = n_epochs
        self.lr_actor    = lr_actor
        self.rng         = np.random.default_rng(seed)

        # Phase 1: Curriculum learning
        self.curriculum = curriculum
        self.curriculum_stages = curriculum_stages
        self.current_stage = 0
        self.stage_episodes = 0
        self.stage_success_threshold = 0.95  # 95% CTI satisfaction to advance

        self.actors = [
            GaussianActor(env.OBS_DIM, env.ACT_DIM,
                          hidden_actor, seed + i)
            for i in range(self.n_agents)
        ]
        self.critic = CentralCritic(
            self.n_agents, hidden_critic, seed+999,
            use_pv_features=curriculum
        )
        self.buffer = RolloutBuffer(
            self.n_agents, env.OBS_DIM, env.ACT_DIM, rollout_len)

    def _update_actor_es(
        self,
        agent_idx,
        obs_batch,
        act_batch,
        adv_batch,
        old_lp,
        n_pert = 10,
        sigma  = 0.02,
    ):
        """OpenAI-ES surrogate policy gradient with PPO-clip objective."""
        actor  = self.actors[agent_idx]
        theta  = actor.get_all_params()
        noises, rewards = [], []

        for _ in range(n_pert):
            eps = self.rng.normal(0, sigma, len(theta))
            for sign in (+1, -1):
                actor.set_all_params(theta + sign * eps)
                new_lp = np.array([
                    actor.log_prob(obs_batch[t], act_batch[t])
                    for t in range(len(obs_batch))
                ])
                ratio  = np.exp(np.clip(new_lp - old_lp, -5, 5))
                clip_r = np.clip(ratio, 1-self.clip_eps, 1+self.clip_eps)
                surr   = float(np.mean(
                    np.minimum(ratio*adv_batch, clip_r*adv_batch)))
                noises.append(sign * eps)
                rewards.append(surr)

        r_arr  = np.array(rewards)
        r_norm = (r_arr - r_arr.mean()) / (r_arr.std() + 1e-8)
        grad   = sum(r*n for r, n in zip(r_norm, noises))
        grad  /= (2 * n_pert * sigma)
        actor.set_all_params(theta + self.lr_actor * grad)

    def _collect_rollout(self, obs):
        self.buffer.clear()
        for _ in range(self.rollout_len):
            actions   = np.zeros((self.n_agents, self.env.ACT_DIM))
            log_probs = np.zeros(self.n_agents)
            for i, actor in enumerate(self.actors):
                actions[i], log_probs[i] = actor.sample(obs[i], self.rng)

            TMS, PS = self.env.get_state()

            # Phase 1: Get PV level and topology flag from environment
            pv_level = getattr(self.env, 'current_pv_level', 0.0)
            topology_flag = 0.0
            if hasattr(self.env, 'current_state') and self.env.current_state:
                topology_flag = 1.0 if any(
                    not v for v in self.env.current_state.topology_state.line_status.values()
                ) else 0.0

            value   = self.critic.value(TMS, PS, pv_level, topology_flag)
            next_obs, rewards, done, info = self.env.step(actions)

            self.buffer.add(
                obs, actions, rewards, log_probs, value, done,
                pv_level=pv_level, topology_flag=topology_flag
            )
            obs = next_obs
            if done:
                obs = self.env.reset()
        return obs

    def train(self, total_steps=30000, verbose=True, log_every=2000):
        obs      = self.env.reset()
        step     = 0
        history  = []
        best_OF  = np.inf
        best_TMS = self.env.TMS.copy()
        best_PS  = self.env.PS.copy()
        t0       = time.perf_counter()

        # Phase 1: Curriculum tracking
        stage_history = {i: [] for i in range(self.curriculum_stages)}
        cti_satisfaction_history = []

        if verbose:
            print(f"\n{'─'*55}")
            print(f"  MARL-IPPO  ·  {self.env.net.name}  ·  "
                  f"{self.n_agents} agents  ·  {self.env.net.n_pairs} pairs")
            if self.curriculum:
                print(f"  Curriculum Learning: {self.curriculum_stages} stages")
            print(f"{'─'*55}")

        while step < total_steps:
            obs  = self._collect_rollout(obs)
            T    = self.buffer.size
            step += T

            TMS, PS   = self.env.get_state()

            # Phase 1: Get dynamic features for critic
            pv_level = getattr(self.env, 'current_pv_level', 0.0)
            topology_flag = 0.0
            if hasattr(self.env, 'current_state') and self.env.current_state:
                topology_flag = 1.0 if any(
                    not v for v in self.env.current_state.topology_state.line_status.values()
                ) else 0.0

            last_val  = self.critic.value(TMS, PS, pv_level, topology_flag)
            advs, _   = self.buffer.compute_gae(
                self.gamma, self.lam, last_val)

            for ag in range(self.n_agents):
                for _ in range(self.n_epochs):
                    self._update_actor_es(
                        ag,
                        self.buffer.obs[:T, ag],
                        self.buffer.actions[:T, ag],
                        advs[:T, ag],
                        self.buffer.log_probs[:T, ag],
                    )

            from network.relay_physics import objective_function as OF_fn

            # Phase 1: Use dynamic network for evaluation
            if hasattr(self.env, 'dist_net'):
                eval_result = self.env.evaluate_current_settings()
                OF_now = eval_result["OF"]
                cti_ok = eval_result["constraints"]["cti_satisfied"]
            else:
                OF_now, _, _ = OF_fn(self.env.net, TMS, PS)
                cti_ok = False

            if OF_now < best_OF:
                best_OF  = OF_now
                best_TMS = TMS.copy()
                best_PS  = PS.copy()
            history.append(OF_now)
            cti_satisfaction_history.append(1.0 if cti_ok else 0.0)

            # Phase 1: Curriculum stage advancement
            if self.curriculum:
                stage_history[self.current_stage].append(OF_now)
                recent_cti_rate = np.mean(cti_satisfaction_history[-100:]) if len(cti_satisfaction_history) >= 100 else 0.0

                if (recent_cti_rate > self.stage_success_threshold and 
                    self.current_stage < self.curriculum_stages - 1 and
                    step > (self.current_stage + 1) * total_steps // self.curriculum_stages):
                    self.current_stage += 1
                    self.env.set_curriculum_stage(self.current_stage)
                    if verbose:
                        print(f"  ↑ Advanced to curriculum stage {self.current_stage} "
                              f"(PV {self.env.pv_levels[self.current_stage]*100:.0f}%)")

            if verbose and step % log_every < T:
                stage_info = f" stage={self.current_stage}" if self.curriculum else ""
                print(f"  step {step:6d}/{total_steps}  "
                      f"OF={OF_now:.4f}  best={best_OF:.4f}  "
                      f"CTI_ok={cti_ok}{stage_info}  "
                      f"t={time.perf_counter()-t0:.1f}s")

        elapsed = time.perf_counter() - t0
        if verbose:
            print(f"  Done. Best OF = {best_OF:.4f}  [{elapsed:.1f}s]")
            if self.curriculum:
                print(f"  Final curriculum stage: {self.current_stage}")

        return {
            "method":  "MARL-IPPO",
            "network": self.env.net.name,
            "TMS":     best_TMS,
            "PS":      best_PS,
            "OF":      best_OF,
            "history": history,
            "elapsed": elapsed,
            "curriculum_stage": self.current_stage if self.curriculum else 0,
            "stage_history": stage_history if self.curriculum else {},
        }