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

# torch required as of 2026-10-22 -- see marl/marl_agents.py's module-level
# comment for the full ES-to-backprop rewrite this file is part of,
# including the import-order warning (import something from network/
# environment/marl before torch, not after -- satisfied here already by
# the NetworkData/DistributionNetwork imports above).
import torch


def relu(x): return np.maximum(0, x)
def tanh(x): return np.tanh(x)


def _default_device():
    """Mirrors marl/marl_agents.py's _default_device -- resolves at call
    time so a CUDA-disabling monkeypatch (e.g.
    run_core_training.py's _disable_gpu_in_worker) still works."""
    import marl.marl_agents as _ma
    return _ma._GPU_DEVICE if _ma._GPU_AVAILABLE else torch.device("cpu")


class _TorchMLP(torch.nn.Module):
    """2-hidden-layer MLP: in -> h1 -> h2 -> out. See
    marl/marl_agents.py's _TorchMLP for the rationale (real backprop +
    Adam replacing OpenAI-ES, 2026-10-22)."""

    def __init__(self, in_dim, hidden, out_dim, seed=0):
        super().__init__()
        gen = torch.Generator().manual_seed(seed)
        self.fc1 = torch.nn.Linear(in_dim, hidden)
        self.fc2 = torch.nn.Linear(hidden, hidden)
        self.fc3 = torch.nn.Linear(hidden, out_dim)
        for layer, fan_in in ((self.fc1, in_dim), (self.fc2, hidden), (self.fc3, hidden)):
            torch.nn.init.normal_(layer.weight, mean=0.0, std=(2.0 / fan_in) ** 0.5, generator=gen)
            torch.nn.init.zeros_(layer.bias)

    def forward(self, x):
        h1 = torch.relu(self.fc1(x))
        h2 = torch.relu(self.fc2(h1))
        return self.fc3(h2)


class CentralizedGaussianActor:
    """
    Centralized Gaussian policy controlling all relays simultaneously.

    Action: [TMS_1..N, PS_1..N] -> mapped to valid ranges
    """

    def __init__(self, obs_dim, n_relays, hidden=64, seed=0, device=None, lr=3e-4):
        self.n_relays = n_relays
        self.act_dim = n_relays * 2  # TMS + PS for each relay
        self.device = torch.device(device) if device is not None else _default_device()
        self.net = _TorchMLP(obs_dim, hidden, self.act_dim, seed).to(self.device)
        self.log_std = torch.nn.Parameter(torch.full((self.act_dim,), -1.5, device=self.device))
        self.optimizer = torch.optim.Adam(list(self.net.parameters()) + [self.log_std], lr=lr)

        # Action bounds
        self.tms_min = TMS_MIN
        self.tms_max = TMS_MAX

    def get_mean(self, obs):
        """Get mean action in normalized space [-1, 1]. NumPy in/out,
        no-grad -- the rollout-collection hot path."""
        obs_t = torch.as_tensor(np.asarray(obs), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            mean = torch.tanh(self.net(obs_t))
        return mean.cpu().numpy()

    def sample(self, obs, rng) -> Tuple[np.ndarray, float]:
        """Sample action and compute log probability."""
        mean = self.get_mean(obs)
        log_std_np = self.log_std.detach().cpu().numpy()
        std = np.exp(np.clip(log_std_np, -4, 0))
        noise = rng.normal(0, 1, self.act_dim)
        action_norm = np.clip(mean + std * noise, -1.0, 1.0)
        log_prob = float(np.sum(
            -0.5 * (noise**2 + 2*log_std_np + np.log(2*np.pi))))
        return action_norm, log_prob

    def _log_prob_t(self, obs_t, act_t):
        """Differentiable log-prob, used by both log_prob() (no_grad) and
        update() (with grad tracking)."""
        mean = torch.tanh(self.net(obs_t))
        std = torch.exp(torch.clamp(self.log_std, -4, 0))
        z = (act_t - mean) / (std + 1e-8)
        terms = -0.5 * (z**2 + 2 * self.log_std + np.log(2 * np.pi))
        return terms.sum(dim=-1)

    def log_prob(self, obs, action_norm):
        """Compute log probability of action(s). NumPy in, NumPy/float
        out, no-grad.

        obs/action_norm of shape (obs_dim,)/(act_dim,) -> scalar float.
        obs/action_norm of shape (T, obs_dim)/(T, act_dim) -> (T,) array.
        """
        obs_t = torch.as_tensor(np.asarray(obs), dtype=torch.float32, device=self.device)
        act_t = torch.as_tensor(np.asarray(action_norm), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            lp = self._log_prob_t(obs_t, act_t)
        return float(lp.item()) if obs_t.ndim == 1 else lp.cpu().numpy()

    def update(self, obs_batch, act_batch, adv_batch, old_lp_batch, clip_eps=0.2):
        """One real PPO-clip gradient step via backprop + Adam, replacing
        _update_actor_es's one ES generation."""
        obs_t = torch.as_tensor(np.asarray(obs_batch), dtype=torch.float32, device=self.device)
        act_t = torch.as_tensor(np.asarray(act_batch), dtype=torch.float32, device=self.device)
        adv_t = torch.as_tensor(np.asarray(adv_batch), dtype=torch.float32, device=self.device)
        old_lp_t = torch.as_tensor(np.asarray(old_lp_batch), dtype=torch.float32, device=self.device)

        self.optimizer.zero_grad()
        new_lp = self._log_prob_t(obs_t, act_t)
        ratio = torch.exp(torch.clamp(new_lp - old_lp_t, -5, 5))
        clip_r = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)
        loss = -torch.minimum(ratio * adv_t, clip_r * adv_t).mean()
        loss.backward()
        self.optimizer.step()
        return float(loss.item())

    def get_all_params(self):
        """Flat parameter vector -- kept for backward compatibility with
        ES-era tests/fixtures, see marl_agents.SharedGaussianActor's
        equivalent note."""
        with torch.no_grad():
            flat = torch.cat([p.reshape(-1) for p in self.net.parameters()] + [self.log_std.reshape(-1)])
        return flat.cpu().numpy()

    def set_all_params(self, p):
        p_t = torch.as_tensor(np.asarray(p), dtype=torch.float32, device=self.device)
        torch.nn.utils.vector_to_parameters(p_t[:-self.act_dim], self.net.parameters())
        with torch.no_grad():
            self.log_std.copy_(p_t[-self.act_dim:])

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


class CentralizedCritic:
    """Centralized value function on global state."""

    def __init__(self, obs_dim, hidden=64, seed=0, device=None, lr=1e-3):
        self.device = torch.device(device) if device is not None else _default_device()
        self.net = _TorchMLP(obs_dim, hidden, 1, seed).to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)

    def value(self, obs) -> float:
        """NumPy in, float out, no-grad -- the rollout-collection hot path."""
        obs_t = torch.as_tensor(np.asarray(obs), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            return float(self.net(obs_t)[0].item())

    def value_batch(self, obs_batch: np.ndarray) -> np.ndarray:
        """Batched form of value(). NumPy in, NumPy out, no-grad (see
        update() for the differentiable path)."""
        obs_t = torch.as_tensor(np.asarray(obs_batch), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            return self.net(obs_t).cpu().numpy().ravel()

    def update(self, obs_batch, ret_batch):
        """One real MSE gradient step via backprop + Adam, replacing the
        old ES generation."""
        obs_t = torch.as_tensor(np.asarray(obs_batch), dtype=torch.float32, device=self.device)
        ret_t = torch.as_tensor(np.asarray(ret_batch), dtype=torch.float32, device=self.device)

        self.optimizer.zero_grad()
        pred = self.net(obs_t)[:, 0]
        loss = torch.mean((ret_t - pred) ** 2)
        loss.backward()
        self.optimizer.step()
        return float(loss.item())

    def get_params(self):
        """Flat parameter vector -- kept for backward compatibility with
        ES-era tests/fixtures."""
        with torch.no_grad():
            flat = torch.cat([p.reshape(-1) for p in self.net.parameters()])
        return flat.cpu().numpy()

    def set_params(self, p):
        p_t = torch.as_tensor(np.asarray(p), dtype=torch.float32, device=self.device)
        torch.nn.utils.vector_to_parameters(p_t, self.net.parameters())

    def n_params(self):
        return sum(p.numel() for p in self.net.parameters())


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
        # self.TMS/self.PS are already decoded and applied by the caller
        # via set_relay_settings() before step() runs (see
        # SinglePPOTrainer._collect_rollout: decode_action() +
        # set_relay_settings() precede env.step()) — action_norm is
        # accepted here only to match the env.step(action) interface.
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
        lr_actor: float = 3e-4,
        lr_critic: float = 1e-3,
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
            env.OBS_DIM, self.n_relays, hidden_actor, seed, lr=lr_actor
        )
        self.critic = CentralizedCritic(env.OBS_DIM, hidden_critic, seed + 999, lr=lr_critic)

        # Buffers
        self.obs_buf = np.zeros((rollout_len, env.OBS_DIM))
        self.act_buf = np.zeros((rollout_len, self.actor.act_dim))
        self.rew_buf = np.zeros(rollout_len)
        self.val_buf = np.zeros(rollout_len)
        self.lp_buf = np.zeros(rollout_len)
        self.done_buf = np.zeros(rollout_len, dtype=bool)

    def _update_actor(self, obs_batch, act_batch, adv_batch, old_lp):
        """One real PPO-clip gradient step via backprop + Adam, replacing
        the old ES generation (_update_actor_es)."""
        self.actor.update(obs_batch, act_batch, adv_batch, old_lp, clip_eps=self.clip_eps)

    def _update_critic(self, obs_batch, ret_batch):
        """One real MSE gradient step via backprop + Adam, replacing the
        old ES generation."""
        self.critic.update(obs_batch, ret_batch)

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
        # reset()'s pv_level/fault_type default to 0.0/"3ph" regardless
        # of what the caller already configured on env (via __init__ or
        # an explicit reset() beforehand) -- calling it bare here silently
        # reverted every training run back to 0% PV no matter what the
        # caller intended, since this is the only reset() call train()
        # ever makes. Reuse whatever the env is currently configured
        # with instead of re-defaulting it.
        obs = self.env.reset(pv_level=self.env.current_pv_level, fault_type=self.env.fault_type)
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
                self._update_actor(
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
    pv_level=0.0,
    fault_type="3ph",
):
    """Convenience function to run centralized PPO."""
    env = CentralizedPPOEnvironment(net, use_pandapower=use_pandapower, seed=seed)
    env.reset(pv_level=pv_level, fault_type=fault_type, seed=seed)
    trainer = SinglePPOTrainer(
        env,
        hidden_actor=hidden_actor,
        hidden_critic=hidden_critic,
        seed=seed,
    )
    return trainer.train(total_steps=total_steps, verbose=verbose)
