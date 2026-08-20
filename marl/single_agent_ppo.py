"""
single_agent_ppo.py
───────────────────
Single-Agent PPO (Centralized) Baseline for DOCR Coordination.

Phase 2: Centralized PPO baseline to prove decentralization (MARL) 
beats centralization even with RL.

Design:
- Single agent controls ALL relay settings simultaneously
- Centralized observation: [TMS_1..N || PS_1..N || global_features]
- Continuous action space: [TMS_1..N || PS_1..N] directly
- PPO with clipped surrogate objective
- Comparison against IPPO/MAPPO demonstrates value of decentralization

Architecture:
- Actor-Critic with shared feature extractor
- Gaussian policy for continuous actions
- Value function estimates return for variance reduction

References:
- Schulman et al., "Proximal Policy Optimization Algorithms", arXiv 2017
- Yu et al., "The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games", 2022
"""

import numpy as np
from typing import Tuple, Dict, Optional, List
import time

from network.network_data import NetworkData
from network.relay_physics import (
    objective_function, check_constraints,
    TMS_MIN, TMS_MAX, CTI, TOP_MAX,
)
from environment.dynamic_network import DistributionNetwork


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
        """x: (in_dim,) single input, or (T, in_dim) batch — see
        marl/marl_agents.py's MLP.forward for why batching this matters
        (ES perturbation evaluation re-scores a whole rollout at once)."""
        x = np.asarray(x)
        if x.ndim == 1:
            h1 = relu(self.W1 @ x + self.b1)
            h2 = relu(self.W2 @ h1 + self.b2)
            return self.W3 @ h2 + self.b3
        else:
            h1 = relu(x @ self.W1.T + self.b1)
            h2 = relu(h1 @ self.W2.T + self.b2)
            return h2 @ self.W3.T + self.b3

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


class CentralizedGaussianActor:
    """
    Centralized Gaussian policy controlling all relays simultaneously.

    Action: [TMS_1..N, PS_1..N] -> mapped to valid ranges
    """

    def __init__(self, obs_dim, n_relays, hidden=64, seed=0):
        self.n_relays = n_relays
        self.act_dim = n_relays * 2  # TMS + PS for each relay
        self.net = MLP(obs_dim, hidden, self.act_dim, seed)
        self.log_std = np.full(self.act_dim, -1.5)

        # Action bounds
        self.tms_min = TMS_MIN
        self.tms_max = TMS_MAX

    def get_mean(self, obs):
        """Get mean action in normalized space [-1, 1]."""
        return tanh(self.net.forward(obs))

    def sample(self, obs, rng) -> Tuple[np.ndarray, float]:
        """Sample action and compute log probability."""
        mean = self.get_mean(obs)
        std = np.exp(np.clip(self.log_std, -4, 0))
        noise = rng.normal(0, 1, self.act_dim)
        action_norm = np.clip(mean + std * noise, -1.0, 1.0)

        # Log prob
        log_prob = float(np.sum(
            -0.5 * (noise**2 + 2*self.log_std + np.log(2*np.pi))))

        return action_norm, log_prob

    def log_prob(self, obs, action_norm):
        """Compute log probability of action(s).

        obs/action_norm of shape (obs_dim,)/(act_dim,) -> scalar float.
        obs/action_norm of shape (T, obs_dim)/(T, act_dim) -> (T,) array,
        replacing what used to require a Python loop of T individual
        log_prob() calls at every ES perturbation evaluation.
        """
        obs = np.asarray(obs)
        mean = self.get_mean(obs)
        std = np.exp(np.clip(self.log_std, -4, 0))
        z = (action_norm - mean) / (std + 1e-8)
        terms = -0.5 * (z**2 + 2 * self.log_std + np.log(2 * np.pi))
        if obs.ndim == 1:
            return float(np.sum(terms))
        return np.sum(terms, axis=-1)

    def decode_action(self, action_norm, ps_min, ps_max):
        """
        Decode normalized action to actual TMS and PS values.

        action_norm: (2*n_relays,) in [-1, 1]
        Returns: TMS (n_relays,), PS (n_relays,)
        """
        n = self.n_relays
        tms_norm = action_norm[:n]
        ps_norm = action_norm[n:]

        # Map [-1, 1] to [TMS_MIN, TMS_MAX]
        TMS = self.tms_min + (tms_norm + 1) / 2 * (self.tms_max - self.tms_min)
        TMS = np.clip(TMS, self.tms_min, self.tms_max)

        # Map [-1, 1] to [ps_min, ps_max]
        PS = ps_min + (ps_norm + 1) / 2 * (ps_max - ps_min)
        PS = np.clip(PS, ps_min, ps_max)

        return TMS, PS

    def encode_action(self, TMS, PS, ps_min, ps_max):
        """Encode actual TMS/PS to normalized action."""
        n = self.n_relays
        tms_norm = 2 * (TMS - self.tms_min) / (self.tms_max - self.tms_min) - 1
        ps_norm = 2 * (PS - ps_min) / (ps_max - ps_min) - 1
        return np.concatenate([tms_norm, ps_norm])

    def get_all_params(self):
        return np.concatenate([self.net.get_params(), self.log_std])

    def set_all_params(self, p):
        n = self.net.n_params()
        self.net.set_params(p[:n])
        self.log_std = p[n:n + self.act_dim]


class CentralizedCritic:
    """Centralized value function on global state."""

    def __init__(self, obs_dim, hidden=64, seed=0):
        self.net = MLP(obs_dim, hidden, 1, seed)

    def value(self, obs) -> float:
        return float(self.net.forward(obs)[0])

    def value_batch(self, obs_batch: np.ndarray) -> np.ndarray:
        """Batched form of value(): obs_batch (T, obs_dim) -> (T,) predictions
        via one vectorized MLP.forward call instead of a Python loop over T
        individual value() calls (used by the ES critic update)."""
        return self.net.forward(obs_batch).ravel()

    def get_params(self):
        return self.net.get_params()

    def set_params(self, p):
        self.net.set_params(p)

    def n_params(self):
        return self.net.n_params()


class CentralizedPPOEnvironment:
    """
    Environment wrapper for centralized PPO.

    Global observation: [TMS_all || PS_all || If_pri_mean || If_bkp_mean || 
                        PV_level || topology_flag || cti_slack || time]
    """

    def __init__(self, net: NetworkData, use_pandapower: bool = False, seed: int = 0):
        self.net = net
        self.n_relays = net.n_relays
        self.ps_min, self.ps_max = net.ps_bounds()
        self.ps_range = np.maximum(self.ps_max - self.ps_min, 1e-3)
        self.tms_range = TMS_MAX - TMS_MIN

        # Create distribution network
        self.dist_net = DistributionNetwork(net_data=net, use_pandapower=use_pandapower)

        # Observation dimension
        self.OBS_DIM = 2 * self.n_relays + 6  # TMS, PS, If stats, PV, topo, cti, time

        self.rng = np.random.default_rng(seed)
        self.TMS = np.zeros(self.n_relays)
        self.PS = np.zeros(self.n_relays)
        self.step_count = 0
        self.max_steps = 300
        self.current_pv_level = 0.0
        self.fault_type = "3ph"

        self.reset()

    def reset(self, pv_level: float = 0.0, fault_type: str = "3ph", seed: Optional[int] = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.current_pv_level = pv_level
        self.fault_type = fault_type

        # Random initialization
        self.TMS = self.rng.uniform(TMS_MIN + 0.02, TMS_MAX * 0.5, self.n_relays)
        self.PS = self.rng.uniform(0, 1, self.n_relays) * self.ps_range + self.ps_min
        self.PS = np.clip(self.PS, self.ps_min, self.ps_max)
        self.step_count = 0

        # Reset distribution network
        time_of_day = self.rng.uniform(6, 18)
        self.dist_net.reset(
            pv_level=pv_level,
            fault_type=fault_type,
            time_of_day=time_of_day,
        )
        self.dist_net.set_relay_settings(self.TMS, self.PS)

        return self._get_obs()

    def _get_obs(self) -> np.ndarray:
        """Build global observation."""
        from network.relay_physics import compute_operating_times

        t_pri, t_bkp = compute_operating_times(self.net, self.TMS, self.PS, self.fault_type)
        delta = t_bkp - t_pri

        # Fault current statistics
        if_pri_mean = np.mean([p.If_pri for p in self.net.pairs]) / 10000.0
        if_bkp_mean = np.mean([p.If_bkp for p in self.net.pairs]) / 1000.0

        # CTI slack
        cti_slack = float(np.mean(delta - CTI)) / 2.0

        # Topology flag
        topology_flag = 0.0

        # Normalize TMS and PS
        tms_norm = (self.TMS - TMS_MIN) / self.tms_range
        ps_norm = (self.PS - self.ps_min) / self.ps_range

        obs = np.concatenate([
            tms_norm,
            ps_norm,
            [if_pri_mean, if_bkp_mean, self.current_pv_level, topology_flag, cti_slack, self.step_count / self.max_steps]
        ])

        return obs

    def step(self, action_norm: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Step environment with normalized action.

        Returns:
            next_obs, reward, done, info
        """
        # Decode action
        self.TMS, self.PS = self.actor.decode_action(action_norm, self.ps_min, self.ps_max) if hasattr(self, 'actor') else (self.TMS, self.PS)

        # Actually we need the actor to decode - let me fix this
        # For now, assume action_norm is already decoded by the trainer

        self.dist_net.set_relay_settings(self.TMS, self.PS)

        # Compute objective
        from network.relay_physics import objective_function, compute_operating_times

        OF, sum_top, penalty = objective_function(
            self.net, self.TMS, self.PS, fault_type=self.fault_type
        )
        t_pri, t_bkp = compute_operating_times(self.net, self.TMS, self.PS, self.fault_type)
        delta_cti = t_bkp - t_pri

        # Reward
        cti_ok = np.all(delta_cti >= CTI - 1e-6)
        reward = -OF / (self.n_relays * TOP_MAX)
        if cti_ok:
            reward += 1.0

        self.step_count += 1
        done = self.step_count >= self.max_steps

        info = {
            "OF": OF,
            "sum_top": sum_top,
            "penalty": penalty,
            "cti_min": float(np.min(delta_cti)),
            "cti_ok": cti_ok,
            "step": self.step_count,
        }

        return self._get_obs(), reward, done, info

    def set_relay_settings(self, TMS, PS):
        self.TMS = TMS.copy()
        self.PS = PS.copy()
        self.dist_net.set_relay_settings(TMS, PS)


class SinglePPOTrainer:
    """
    Centralized PPO Trainer for DOCR coordination.

    Single agent controls all relays simultaneously.
    """

    def __init__(
        self,
        env: CentralizedPPOEnvironment,
        hidden_actor: int = 128,
        hidden_critic: int = 128,
        lr_actor: float = 0.01,
        lr_critic: float = 0.01,
        clip_eps: float = 0.2,
        gamma: float = 0.99,
        lam: float = 0.95,
        rollout_len: int = 128,
        n_epochs: int = 4,
        seed: int = 0,
    ):
        self.env = env
        self.n_relays = env.n_relays
        self.clip_eps = clip_eps
        self.gamma = gamma
        self.lam = lam
        self.rollout_len = rollout_len
        self.n_epochs = n_epochs
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.rng = np.random.default_rng(seed)

        self.actor = CentralizedGaussianActor(
            env.OBS_DIM, self.n_relays, hidden_actor, seed
        )
        self.critic = CentralizedCritic(env.OBS_DIM, hidden_critic, seed + 999)

        # Buffers
        self.obs_buf = np.zeros((rollout_len, env.OBS_DIM))
        self.act_buf = np.zeros((rollout_len, self.actor.act_dim))
        self.rew_buf = np.zeros(rollout_len)
        self.val_buf = np.zeros(rollout_len)
        self.lp_buf = np.zeros(rollout_len)
        self.done_buf = np.zeros(rollout_len, dtype=bool)

        # Attach actor to env for decoding
        env.actor = self.actor

    def _update_actor_es(self, obs_batch, act_batch, adv_batch, old_lp, n_pert=10, sigma=0.02):
        """OpenAI-ES surrogate policy gradient with PPO-clip.

        `eps` below is already sigma-scaled (~N(0, sigma^2)), not unit
        noise, so the standard antithetic ES gradient estimate
        (1/(2*n*sigma)) * sum[(r+ - r-) * unit_eps_i] normalizes to
        sigma**2 in the denominator here, not sigma — dividing by plain
        sigma left the gradient magnitude inflated by 1/sigma (50x at
        the default sigma=0.02). marl/comm_graph.py's ES update already
        uses the correct sigma**2 form; this now matches it.
        """
        theta = self.actor.get_all_params()
        noises, rewards = [], []

        for _ in range(n_pert):
            eps = self.rng.normal(0, sigma, len(theta))
            for sign in (+1, -1):
                self.actor.set_all_params(theta + sign * eps)
                new_lp = self.actor.log_prob(obs_batch, act_batch)
                ratio = np.exp(np.clip(new_lp - old_lp, -5, 5))
                clip_r = np.clip(ratio, 1 - self.clip_eps, 1 + self.clip_eps)
                surr = float(np.mean(np.minimum(ratio * adv_batch, clip_r * adv_batch)))
                noises.append(sign * eps)
                rewards.append(surr)

        r_arr = np.array(rewards)
        r_norm = (r_arr - r_arr.mean()) / (r_arr.std() + 1e-8)
        grad = sum(r * n for r, n in zip(r_norm, noises))
        grad /= (2 * n_pert * sigma ** 2)
        self.actor.set_all_params(theta + self.lr_actor * grad)

    def _update_critic(self, obs_batch, ret_batch, n_pert=10, sigma=0.02):
        """Update critic with ES."""
        theta = self.critic.get_params()
        noises, losses = [], []

        for _ in range(n_pert):
            eps = self.rng.normal(0, sigma, len(theta))
            for sign in (+1, -1):
                self.critic.set_params(theta + sign * eps)
                pred = self.critic.value_batch(obs_batch)
                loss = -float(np.mean((ret_batch - pred) ** 2))
                noises.append(sign * eps)
                losses.append(loss)

        l_arr = np.array(losses)
        l_norm = (l_arr - l_arr.mean()) / (l_arr.std() + 1e-8)
        grad = sum(l * n for l, n in zip(l_norm, noises))
        grad /= (2 * n_pert * sigma ** 2)
        self.critic.set_params(theta + self.lr_critic * grad)

    def _collect_rollout(self, obs):
        """Collect one rollout of experience."""
        for t in range(self.rollout_len):
            action_norm, log_prob = self.actor.sample(obs, self.rng)
            value = self.critic.value(obs)

            # Decode action for environment step
            TMS, PS = self.actor.decode_action(action_norm, self.env.ps_min, self.env.ps_max)
            self.env.set_relay_settings(TMS, PS)

            next_obs, reward, done, info = self.env.step(action_norm)

            self.obs_buf[t] = obs
            self.act_buf[t] = action_norm
            self.rew_buf[t] = reward
            self.val_buf[t] = value
            self.lp_buf[t] = log_prob
            self.done_buf[t] = done

            obs = next_obs
            if done:
                obs = self.env.reset()

        return obs

    def _compute_gae(self, last_value):
        """Generalized Advantage Estimation."""
        T = self.rollout_len
        adv = np.zeros(T)
        ret = np.zeros(T)

        gae = 0.0
        for t in reversed(range(T)):
            Vn = last_value if (t == T - 1 or self.done_buf[t]) else self.val_buf[t + 1]
            delta = self.rew_buf[t] + self.gamma * Vn * (1 - self.done_buf[t]) - self.val_buf[t]
            gae = delta + self.gamma * self.lam * (1 - self.done_buf[t]) * gae
            adv[t] = gae
            ret[t] = gae + self.val_buf[t]

        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        return adv, ret

    def train(self, total_steps=30000, verbose=True, log_every=2000,
              early_stop_patience=None, early_stop_min_delta=1e-4):
        """Train centralized PPO.

        early_stop_patience: if set, stop once this many consecutive
        rollout evaluations pass without the best objective improving by
        more than early_stop_min_delta. Disabled by default (None) — see
        MAPPOMARLTrainer.train() in marl/marl_agents.py for the same
        opt-in pattern and rationale.
        """
        obs = self.env.reset()
        step = 0
        history = []
        best_OF = np.inf
        best_TMS = self.env.TMS.copy()
        best_PS = self.env.PS.copy()
        no_improve_count = 0
        early_stopped = False
        t0 = time.perf_counter()

        if verbose:
            print(f"\n{'─' * 55}")
            print(f"  Single-Agent PPO  ·  {self.env.net.name}  ·  "
                  f"{self.n_relays} relays")
            print(f"{'─' * 55}")

        while step < total_steps:
            obs = self._collect_rollout(obs)
            step += self.rollout_len

            last_val = self.critic.value(obs)
            advs, rets = self._compute_gae(last_val)

            # Update actor
            for _ in range(self.n_epochs):
                self._update_actor_es(
                    self.obs_buf, self.act_buf, advs, self.lp_buf
                )

            # Update critic
            for _ in range(self.n_epochs):
                self._update_critic(self.obs_buf, rets)

            # Evaluate
            TMS, PS = self.env.TMS, self.env.PS
            from network.relay_physics import objective_function
            OF_now, _, _ = objective_function(self.env.net, TMS, PS)

            improved = OF_now < best_OF - early_stop_min_delta
            if OF_now < best_OF:
                best_OF = OF_now
                best_TMS = TMS.copy()
                best_PS = PS.copy()
            no_improve_count = 0 if improved else no_improve_count + 1

            history.append(OF_now)

            if verbose and step % log_every < self.rollout_len:
                print(f"  step {step:6d}/{total_steps}  OF={OF_now:.4f}  best={best_OF:.4f}  "
                      f"t={time.perf_counter() - t0:.1f}s")

            if early_stop_patience is not None and no_improve_count >= early_stop_patience:
                early_stopped = True
                if verbose:
                    print(f"  Early stop at step {step}/{total_steps}: "
                          f"no improvement > {early_stop_min_delta} for {no_improve_count} rollouts")
                break

        elapsed = time.perf_counter() - t0
        if verbose:
            print(f"  Done. Best OF = {best_OF:.4f}  [{elapsed:.1f}s]")

        return {
            "method": "Single-PPO",
            "network": self.env.net.name,
            "TMS": best_TMS,
            "PS": best_PS,
            "OF": best_OF,
            "history": history,
            "elapsed": elapsed,
            "early_stopped": early_stopped,
            "stopped_at_step": step,
        }


def run_single_ppo(
    net,
    total_steps=30000,
    seed=0,
    verbose=True,
    use_pandapower=False,
    hidden_actor=128,
    hidden_critic=128,
):
    """Convenience function to run centralized PPO."""
    env = CentralizedPPOEnvironment(net, use_pandapower=use_pandapower, seed=seed)
    trainer = SinglePPOTrainer(
        env,
        hidden_actor=hidden_actor,
        hidden_critic=hidden_critic,
        seed=seed,
    )
    return trainer.train(total_steps=total_steps, verbose=verbose)
