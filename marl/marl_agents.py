"""
marl_agents.py
──────────────
Multi-Agent PPO (MAPPO) with GNN-Based Learned Communication.

Phase 2: Enhanced decentralized agent architecture with:
- MAPPO (Multi-Agent PPO) with parameter sharing
- GNN communication graph for coordination
- Centralized training, decentralized execution (CTDE)
- Support for neighbor observations and message passing

Architecture
────────────
- Shared actor network (parameter sharing across agents)
- Shared critic network conditioning on global state
- GNN communication module (comm_graph.py)
- Each agent: o_i -> [encode -> communicate -> decode] -> enhanced_o_i -> policy

Design Decisions
────────────────
- MAPPO over IPPO: centralized value function reduces variance
- GNN over fixed topology: learned attention adapts to network conditions
- 8-float messages (~32 bytes): bandwidth-efficient
- Parameter sharing: reduces sample complexity, enables scaling

References
──────────
- Yu et al., "The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games", NeurIPS 2022
- Jiang & Lu, "Learning Attentional Communication for Multi-Agent Cooperation", NeurIPS 2018
"""

import numpy as np
from typing import Tuple, List, Dict, Optional
import time

from network.network_data import NetworkData
from environment.dynamic_network import DistributionNetwork
from network.relay_physics import (
    objective_function, check_constraints,
    compute_operating_times,
    TMS_MIN, TMS_MAX, CTI, TOP_MIN, TOP_MAX,
)


def relu(x): return np.maximum(0, x)
def tanh(x): return np.tanh(x)


class MLP:
    """2-hidden-layer MLP: in -> h1 -> h2 -> out."""

    def __init__(self, in_dim, hidden, out_dim, seed=0):
        rng = np.random.default_rng(seed)
        s = lambda a: np.sqrt(2.0 / a)
        self.W1 = rng.normal(0, s(in_dim), (hidden, in_dim))
        self.b1 = np.zeros(hidden)
        self.W2 = rng.normal(0, s(hidden), (hidden, hidden))
        self.b2 = np.zeros(hidden)
        self.W3 = rng.normal(0, s(hidden), (out_dim, hidden))
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
        sizes = [self.W1.size, self.b1.size, self.W2.size, self.b2.size, self.W3.size, self.b3.size]
        chunks = np.split(p, np.cumsum(sizes[:-1]))
        self.W1 = chunks[0].reshape(self.W1.shape)
        self.b1 = chunks[1]
        self.W2 = chunks[2].reshape(self.W2.shape)
        self.b2 = chunks[3]
        self.W3 = chunks[4].reshape(self.W3.shape)
        self.b3 = chunks[5]

    def n_params(self):
        return sum(p.size for p in [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3])


class SharedGaussianActor:
    """
    Shared Gaussian policy for all agents (parameter sharing).

    All agents use the same network weights but receive different observations.
    This reduces sample complexity and enables scaling to larger networks.
    """

    def __init__(self, obs_dim, act_dim, hidden=64, seed=0):
        self.act_dim = act_dim
        self.net = MLP(obs_dim, hidden, act_dim, seed)
        self.log_std = np.full(act_dim, -1.0)

    def get_mean(self, obs):
        """Get mean action for single or batch observations."""
        if obs.ndim == 1:
            return tanh(self.net.forward(obs))
        else:
            return tanh(np.array([self.net.forward(o) for o in obs]))

    def sample(self, obs, rng) -> Tuple[np.ndarray, float]:
        """Sample action and compute log probability."""
        mean = self.get_mean(obs)
        std = np.exp(np.clip(self.log_std, -3, 0))
        noise = rng.normal(0, 1, self.act_dim)
        action = np.clip(mean + std * noise, -1.0, 1.0)
        log_prob = float(np.sum(
            -0.5 * (noise**2 + 2 * self.log_std + np.log(2 * np.pi))))
        return action, log_prob

    def log_prob(self, obs, action) -> float:
        """Compute log probability of action."""
        mean = self.get_mean(obs)
        std = np.exp(np.clip(self.log_std, -3, 0))
        z = (action - mean) / (std + 1e-8)
        return float(np.sum(
            -0.5 * (z**2 + 2 * self.log_std + np.log(2 * np.pi))))

    def get_all_params(self):
        return np.concatenate([self.net.get_params(), self.log_std])

    def set_all_params(self, p):
        n = self.net.n_params()
        self.net.set_params(p[:n])
        self.log_std = p[n:n + self.act_dim]


class MAPPOCentralCritic:
    """
    MAPPO Centralized Critic with global state.

    Global state: [TMS_all || PS_all || PV_level || topology_flag || fault_type_enc || neighbor_stats]

    Unlike IPPO where each agent has its own critic, MAPPO uses a single
    shared critic that conditions on the global state, reducing variance.
    """

    def __init__(self, n_relays, hidden=128, seed=0, use_global_state=True):
        self.n_relays = n_relays
        self.use_global_state = use_global_state

        # Global state dimension
        if use_global_state:
            # TMS (n_relays) + PS (n_relays) + PV_level (1) + topology_flag (1) + 
            # fault_type_onehot (4) + neighbor_stats (6)
            global_dim = 2 * n_relays + 12
        else:
            global_dim = 2 * n_relays + 6

        self.net = MLP(global_dim, hidden, 1, seed)

    def value(self, TMS, PS, pv_level=0.0, topology_flag=0.0,
              fault_type_idx=0, neighbor_stats=None) -> float:
        """Compute value given global state."""
        global_state = np.concatenate([TMS, PS])

        # Fault type one-hot
        fault_onehot = np.zeros(4)
        if 0 <= fault_type_idx < 4:
            fault_onehot[fault_type_idx] = 1.0

        if self.use_global_state and neighbor_stats is not None:
            global_state = np.concatenate([
                global_state,
                [pv_level, topology_flag],
                fault_onehot,
                neighbor_stats,
            ])
        else:
            global_state = np.concatenate([
                global_state,
                [pv_level, topology_flag],
                fault_onehot,
            ])

        return float(self.net.forward(global_state)[0])

    def get_params(self):
        return self.net.get_params()

    def set_params(self, p):
        self.net.set_params(p)

    def n_params(self):
        return self.net.n_params()


class MAPPOBuffer:
    """
    Rollout buffer for MAPPO with communication tracking.

    Stores: observations, enhanced_obs (with comm), actions, rewards,
            log_probs, values, communication info, done flags
    """

    def __init__(self, n_agents, obs_dim, act_dim, capacity, use_comm=False, msg_dim=8):
        self.obs = np.zeros((capacity, n_agents, obs_dim))
        self.actions = np.zeros((capacity, n_agents, act_dim))
        self.rewards = np.zeros((capacity, n_agents))
        self.log_probs = np.zeros((capacity, n_agents))
        self.values = np.zeros(capacity)
        self.dones = np.zeros(capacity, dtype=bool)
        self.pv_levels = np.zeros(capacity)
        self.top_flags = np.zeros(capacity)

        # Communication tracking
        self.use_comm = use_comm
        if use_comm:
            self.messages = np.zeros((capacity, n_agents, msg_dim))
            self.attention = np.zeros((capacity, n_agents, n_agents))

        self.ptr = self.size = 0
        self.capacity = capacity
        self.n_agents = n_agents

    def add(self, obs, actions, rewards, log_probs, value, done,
            pv_level=0.0, topology_flag=0.0, messages=None, attention=None):
        i = self.ptr % self.capacity
        self.obs[i] = obs
        self.actions[i] = actions
        self.rewards[i] = rewards
        self.log_probs[i] = log_probs
        self.values[i] = value
        self.dones[i] = done
        self.pv_levels[i] = pv_level
        self.top_flags[i] = topology_flag

        if self.use_comm and messages is not None:
            self.messages[i] = messages
        if self.use_comm and attention is not None:
            self.attention[i] = attention

        self.ptr += 1
        self.size = min(self.ptr, self.capacity)

    def compute_gae(self, gamma=0.99, lam=0.95, last_value=0.0) -> Tuple[np.ndarray, np.ndarray]:
        """Generalised Advantage Estimation (single shared value)."""
        T = self.size
        adv = np.zeros((T, self.n_agents))
        ret = np.zeros((T, self.n_agents))

        for ag in range(self.n_agents):
            gae = 0.0
            for t in reversed(range(T)):
                Vn = (last_value if (t == T - 1 or self.dones[t])
                      else self.values[t + 1])
                delta = (self.rewards[t, ag]
                         + gamma * Vn * (1 - self.dones[t])
                         - self.values[t])
                gae = delta + gamma * lam * (1 - self.dones[t]) * gae
                adv[t, ag] = gae
                ret[t, ag] = gae + self.values[t]

            adv[:, ag] = ((adv[:, ag] - adv[:, ag].mean())
                          / (adv[:, ag].std() + 1e-8))
        return adv, ret

    def clear(self):
        self.ptr = self.size = 0


class MAPPOMARLTrainer:
    """
    MAPPO Trainer with optional GNN Communication.

    Parameters
    ──────────
    env           DynamicDOCREnv instance
    use_comm      Enable GNN communication
    hidden_actor  hidden units per shared actor
    hidden_critic hidden units for centralized critic
    lr_actor      ES step size for actor update
    lr_critic     ES step size for critic update
    clip_eps      PPO clip ratio epsilon
    gamma / lam   discount and GAE parameters
    rollout_len   steps collected before each update
    n_epochs      ES perturbation rounds per rollout
    curriculum    Enable curriculum learning
    """

    def __init__(
        self,
        env,
        use_comm=False,
        hidden_actor=64,
        hidden_critic=128,
        lr_actor=0.02,
        lr_critic=0.01,
        clip_eps=0.2,
        gamma=0.99,
        lam=0.95,
        rollout_len=64,
        n_epochs=4,
        seed=0,
        curriculum=False,
        curriculum_stages=6,
    ):
        self.env = env
        self.n_agents = env.n_agents
        self.use_comm = use_comm
        self.clip_eps = clip_eps
        self.gamma = gamma
        self.lam = lam
        self.rollout_len = rollout_len
        self.n_epochs = n_epochs
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.rng = np.random.default_rng(seed)

        # Curriculum learning
        self.curriculum = curriculum
        self.curriculum_stages = curriculum_stages
        self.current_stage = 0
        self.stage_episodes = 0
        self.stage_success_threshold = 0.95

        # Communication module (must be created before actor to know enhanced_dim)
        self.comm_graph = None
        if use_comm:
            try:
                from marl.comm_graph import CommGraph, build_relay_adjacency
                adj = build_relay_adjacency(env.net)
                self.comm_graph = CommGraph(
                    n_agents=self.n_agents,
                    obs_dim=env.OBS_DIM,
                    enhanced_dim=env.OBS_DIM + 8,  # Original + message features
                    adjacency=adj,
                    seed=seed + 100,
                )
            except ImportError:
                print("  Warning: comm_graph not available, running without communication")
                self.use_comm = False

        # Shared actor (parameter sharing)
        # When using communication, actor receives enhanced observations
        actor_obs_dim = env.OBS_DIM
        if self.use_comm and self.comm_graph is not None:
            actor_obs_dim = self.comm_graph.decoder.enhanced_dim
        self.actor = SharedGaussianActor(actor_obs_dim, env.ACT_DIM, hidden_actor, seed)

        # Centralized critic
        self.critic = MAPPOCentralCritic(
            self.n_agents, hidden_critic, seed + 999,
            use_global_state=True
        )

        # Buffer - use enhanced observation dimension if using communication
        buffer_obs_dim = env.OBS_DIM
        if self.use_comm and self.comm_graph is not None:
            buffer_obs_dim = self.comm_graph.decoder.enhanced_dim
        self.buffer = MAPPOBuffer(
            self.n_agents, buffer_obs_dim, env.ACT_DIM, rollout_len,
            use_comm=use_comm,
            msg_dim=8 if use_comm else 0,
        )

        # Statistics
        self.comm_bandwidth_history = []
        self.comm_attention_history = []

    def _get_neighbor_stats(self, obs):
        """Compute neighbor statistics for global critic input."""
        # Mean and std of neighbor TMS/PS from observations
        # obs[:, 0] = tms_norm, obs[:, 1] = ps_norm
        mean_tms = float(np.mean(obs[:, 0]))
        mean_ps = float(np.mean(obs[:, 1]))
        std_tms = float(np.std(obs[:, 0]))
        std_ps = float(np.std(obs[:, 1]))
        max_viol = float(np.max(obs[:, 4]))  # viol_flag
        mean_slack = float(np.mean(obs[:, 3]))  # cti_slack_norm
        return np.array([mean_tms, mean_ps, std_tms, std_ps, max_viol, mean_slack])

    def _update_actor_es(self, agent_idx, obs_batch, act_batch, adv_batch, old_lp, n_pert=10, sigma=0.02):
        """OpenAI-ES surrogate policy gradient with PPO-clip."""
        theta = self.actor.get_all_params()
        noises, rewards = [], []

        for _ in range(n_pert):
            eps = self.rng.normal(0, sigma, len(theta))
            for sign in (+1, -1):
                self.actor.set_all_params(theta + sign * eps)
                new_lp = np.array([
                    self.actor.log_prob(obs_batch[t], act_batch[t])
                    for t in range(len(obs_batch))
                ])
                ratio = np.exp(np.clip(new_lp - old_lp, -5, 5))
                clip_r = np.clip(ratio, 1 - self.clip_eps, 1 + self.clip_eps)
                surr = float(np.mean(
                    np.minimum(ratio * adv_batch, clip_r * adv_batch)))
                noises.append(sign * eps)
                rewards.append(surr)

        r_arr = np.array(rewards)
        r_norm = (r_arr - r_arr.mean()) / (r_arr.std() + 1e-8)
        grad = sum(r * n for r, n in zip(r_norm, noises))
        grad /= (2 * n_pert * sigma)
        self.actor.set_all_params(theta + self.lr_actor * grad)

    def _update_critic_es(self, obs_batch, ret_batch, n_pert=10, sigma=0.02):
        """Update centralized critic with ES."""
        theta = self.critic.get_params()
        noises, losses = [], []

        for _ in range(n_pert):
            eps = self.rng.normal(0, sigma, len(theta))
            for sign in (+1, -1):
                self.critic.set_params(theta + sign * eps)
                # We need global state for each timestep - simplified: use mean return
                pred = np.array([
                    self.critic.value(
                        self.env.TMS, self.env.PS,
                        pv_level=getattr(self.env, 'current_pv_level', 0.0),
                        topology_flag=0.0,
                        fault_type_idx=0,
                        neighbor_stats=self._get_neighbor_stats(obs_batch[t])
                    ) for t in range(len(obs_batch))
                ])
                loss = -float(np.mean([(ret_batch[t] - pred[t])**2 for t in range(len(ret_batch))]))
                noises.append(sign * eps)
                losses.append(loss)

        l_arr = np.array(losses)
        l_norm = (l_arr - l_arr.mean()) / (l_arr.std() + 1e-8)
        grad = sum(l * n for l, n in zip(l_norm, noises))
        grad /= (2 * n_pert * sigma)
        self.critic.set_params(theta + self.lr_critic * grad)

    def _collect_rollout(self, obs):
        """Collect one rollout with optional communication."""
        self.buffer.clear()

        for _ in range(self.rollout_len):
            # Apply communication if enabled
            if self.use_comm and self.comm_graph is not None:
                enhanced_obs, messages, comm_info = self.comm_graph.forward(obs, self.rng)
                current_obs = enhanced_obs
            else:
                current_obs = obs
                messages = None
                comm_info = {}

            # Sample actions from shared actor
            actions = np.zeros((self.n_agents, self.env.ACT_DIM))
            log_probs = np.zeros(self.n_agents)
            for i in range(self.n_agents):
                actions[i], log_probs[i] = self.actor.sample(current_obs[i], self.rng)

            TMS, PS = self.env.get_state()

            # Compute value with global state
            pv_level = getattr(self.env, 'current_pv_level', 0.0)
            topology_flag = 0.0
            if hasattr(self.env, 'current_state') and self.env.current_state:
                topology_flag = 1.0 if any(
                    not v for v in self.env.current_state.topology_state.line_status.values()
                ) else 0.0

            neighbor_stats = self._get_neighbor_stats(current_obs)
            value = self.critic.value(
                TMS, PS, pv_level, topology_flag,
                fault_type_idx=0, neighbor_stats=neighbor_stats
            )

            next_obs, rewards, done, info = self.env.step(actions)

            # Store experience
            self.buffer.add(
                current_obs, actions, rewards, log_probs, value, done,
                pv_level=pv_level, topology_flag=topology_flag,
                messages=messages,
                attention=comm_info.get("attention") if comm_info else None,
            )

            # Track communication stats
            if self.use_comm and comm_info:
                bw = self.comm_graph.get_bandwidth_usage(steps_per_second=1.0)
                self.comm_bandwidth_history.append(bw["kbps"])

            obs = next_obs
            if done:
                obs = self.env.reset()

        return obs

    def train(self, total_steps=30000, verbose=True, log_every=2000):
        """Train MAPPO with optional GNN communication."""
        obs = self.env.reset()
        step = 0
        history = []
        best_OF = np.inf
        best_TMS = self.env.TMS.copy()
        best_PS = self.env.PS.copy()
        t0 = time.perf_counter()

        stage_history = {i: [] for i in range(self.curriculum_stages)}
        cti_satisfaction_history = []

        if verbose:
            print(f"\n{'─' * 55}")
            print(f"  MAPPO  ·  {self.env.net.name}  ·  "
                  f"{self.n_agents} agents  ·  {self.env.net.n_pairs} pairs")
            if self.use_comm:
                print(f"  GNN Communication: ON  (msg_dim={self.comm_graph.msg_dim if self.comm_graph else 8})")
            else:
                print(f"  GNN Communication: OFF")
            if self.curriculum:
                print(f"  Curriculum Learning: {self.curriculum_stages} stages")
            print(f"{'─' * 55}")

        while step < total_steps:
            obs = self._collect_rollout(obs)
            T = self.buffer.size
            step += T

            TMS, PS = self.env.get_state()

            # Compute value for GAE
            pv_level = getattr(self.env, 'current_pv_level', 0.0)
            topology_flag = 0.0
            if hasattr(self.env, 'current_state') and self.env.current_state:
                topology_flag = 1.0 if any(
                    not v for v in self.env.current_state.topology_state.line_status.values()
                ) else 0.0

            neighbor_stats = self._get_neighbor_stats(obs)
            last_val = self.critic.value(
                TMS, PS, pv_level, topology_flag,
                fault_type_idx=0, neighbor_stats=neighbor_stats
            )
            advs, rets = self.buffer.compute_gae(self.gamma, self.lam, last_val)

            # Update shared actor
            for ag in range(self.n_agents):
                for _ in range(self.n_epochs):
                    self._update_actor_es(
                        ag,
                        self.buffer.obs[:T, ag],
                        self.buffer.actions[:T, ag],
                        advs[:T, ag],
                        self.buffer.log_probs[:T, ag],
                    )

            # Update centralized critic
            for _ in range(self.n_epochs):
                self._update_critic_es(
                    self.buffer.obs[:T],
                    rets[:T],
                )

            # Evaluate
            from network.relay_physics import objective_function as OF_fn
            if hasattr(self.env, 'dist_net'):
                eval_result = self.env.evaluate_current_settings()
                OF_now = eval_result["OF"]
                cti_ok = eval_result["constraints"]["cti_satisfied"]
            else:
                OF_now, _, _ = OF_fn(self.env.net, TMS, PS)
                cti_ok = False

            if OF_now < best_OF:
                best_OF = OF_now
                best_TMS = TMS.copy()
                best_PS = PS.copy()

            history.append(OF_now)
            cti_satisfaction_history.append(1.0 if cti_ok else 0.0)

            # Curriculum stage advancement
            if self.curriculum:
                stage_history[self.current_stage].append(OF_now)
                recent_cti_rate = np.mean(cti_satisfaction_history[-100:]) if len(cti_satisfaction_history) >= 100 else 0.0

                if (recent_cti_rate > self.stage_success_threshold and
                    self.current_stage < self.curriculum_stages - 1 and
                    step > (self.current_stage + 1) * total_steps // self.curriculum_stages):
                    self.current_stage += 1
                    if hasattr(self.env, 'set_curriculum_stage'):
                        self.env.set_curriculum_stage(self.current_stage)
                    if verbose:
                        print(f"  ↑ Advanced to curriculum stage {self.current_stage}")

            if verbose and step % log_every < T:
                stage_info = f" stage={self.current_stage}" if self.curriculum else ""
                comm_info = ""
                if self.use_comm and self.comm_bandwidth_history:
                    avg_bw = np.mean(self.comm_bandwidth_history[-100:])
                    comm_info = f" bw={avg_bw:.2f}kbps"
                print(f"  step {step:6d}/{total_steps}  "
                      f"OF={OF_now:.4f}  best={best_OF:.4f}  "
                      f"CTI_ok={cti_ok}{stage_info}{comm_info}  "
                      f"t={time.perf_counter() - t0:.1f}s")

        elapsed = time.perf_counter() - t0
        if verbose:
            print(f"  Done. Best OF = {best_OF:.4f}  [{elapsed:.1f}s]")
            if self.curriculum:
                print(f"  Final curriculum stage: {self.current_stage}")
            if self.use_comm and self.comm_bandwidth_history:
                avg_bw = np.mean(self.comm_bandwidth_history)
                print(f"  Avg comm bandwidth: {avg_bw:.2f} kbps")

        return {
            "method": "MAPPO" + ("-Comm" if self.use_comm else ""),
            "network": self.env.net.name,
            "TMS": best_TMS,
            "PS": best_PS,
            "OF": best_OF,
            "history": history,
            "elapsed": elapsed,
            "curriculum_stage": self.current_stage if self.curriculum else 0,
            "stage_history": stage_history if self.curriculum else {},
            "use_comm": self.use_comm,
            "comm_bandwidth": np.mean(self.comm_bandwidth_history) if self.comm_bandwidth_history else 0.0,
        }


def run_mappo(
    net,
    total_steps=30000,
    seed=0,
    verbose=True,
    use_comm=False,
    curriculum=False,
    curriculum_stage=0,
    fault_type="3ph",
    use_pandapower=False,
    hidden_actor=64,
    hidden_critic=128,
):
    """Convenience function to run MAPPO."""
    from marl.marl_env import DynamicDOCREnv

    env = DynamicDOCREnv(
        net,
        action_scale=0.05,
        max_steps=300,
        seed=seed,
        use_pandapower=use_pandapower,
        curriculum_stage=curriculum_stage,
        fault_type=fault_type,
    )

    trainer = MAPPOMARLTrainer(
        env,
        use_comm=use_comm,
        hidden_actor=hidden_actor,
        hidden_critic=hidden_critic,
        seed=seed,
        curriculum=curriculum,
    )

    return trainer.train(total_steps=total_steps, verbose=verbose)
