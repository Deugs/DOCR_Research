"""
vectorized_marl_agents.py
───────────────────────────
VectorizedMAPPOTrainer: Phase V9 of the GPU-vectorized-environment redesign
(see elegant-rolling-map.md for full context) -- wires Phases V1-V8 together
into one trainer that steps `n_envs` parallel environments as a single
batched computation per rollout step, replacing what was `n_envs` separate
Python-level `_collect_rollout` calls (today's `--n-workers` CPU-process
axis) with one batched call across all `n_envs` simultaneously.

Reuses `SharedGaussianActor.update()`/`MAPPOCentralCritic.update()`/
`.value_batch()` completely unchanged -- confirmed during planning (see
plan's "Context" section) that both already generalize to an arbitrary
leading batch axis for free (their internal `_global_state_batch_np` and
backprop paths treat the leading axis generically, with no assumption about
what it represents). This trainer's own new work is flattening the
`(T, n_envs)` axes into one batch dimension before each such call (`(T *
n_envs, ...)`), and the actually-new batched calls this redesign built
specifically for this: `SharedGaussianActor.sample_batch()` (Phase V5),
`VectorizedMAPPOBuffer`/its `compute_gae()` (Phase V5), and
`VectorizedDynamicDOCREnv.step()`/`.reset()` (Phase V6).

Phase V7 (RNG strategy) is realized concretely here: this trainer owns
exactly ONE `torch.Generator` (matching its device), used for
`actor.sample_batch()`'s noise draw -- not one NumPy `Generator` per
environment, keeping randomness GPU-resident for the one genuinely
per-step stochastic draw in the rollout loop. Reproducibility *across* a
given `n_envs` holds (same seed -> same batched draws); reproducibility
*against* the single-env `MAPPOMARLTrainer`'s NumPy-RNG-based rollout is
explicitly not a goal (same "fine to diverge on RNG order" precedent as
every other vectorization in this project).

Scope, documented rather than silently incomplete: GNN communication
(`use_comm=True`) is NOT wired into this first cut of the trainer. Phase V4
built and verified the batched attention primitive
(`GNNAttention.compute_attention_batched_t`) this would need, but
integrating `CommGraph`'s full encoder/decoder/update pipeline with an added
`n_envs` axis (including its own `update()`'s backprop path and bandwidth/
attention-snapshot bookkeeping) is a substantial separate piece of work, not
yet done. `VectorizedMAPPOTrainer` only supports the `use_comm=False`
configuration today -- a documented limitation, not an oversight, the same
category as `use_pandapower=True`/`sensor_noise=True` being out of scope for
the vectorized path elsewhere in this redesign. Similarly, `VectorizedSafeRLWrapper`
(a Phase-V8-powered analogue of `SafeRLWrapper`) is also not implemented here
yet -- Safe RL integration is deferred to a follow-up pass, the same
documented-scope-cut pattern.
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np

from marl.marl_agents import (
    SharedGaussianActor, MAPPOCentralCritic, VectorizedMAPPOBuffer,
    _default_device, _fault_type_idx,
)
from marl.vectorized_marl_env import VectorizedDynamicDOCREnv
from network.relay_physics import objective_function_batch, compute_operating_times_batch, CTI


class VectorizedMAPPOTrainer:
    """
    MAPPO trainer stepping `n_envs` parallel copies of
    `VectorizedDynamicDOCREnv` as one batched computation per rollout step.

    Parameters mirror `MAPPOMARLTrainer`'s (see that class's docstring for
    the shared ones); `use_comm` is deliberately not exposed (see module
    docstring).
    """

    def __init__(
        self,
        env: VectorizedDynamicDOCREnv,
        hidden_actor: int = 64,
        hidden_critic: int = 128,
        lr_actor: float = 3e-4,
        lr_critic: float = 1e-3,
        clip_eps: float = 0.2,
        gamma: float = 0.99,
        lam: float = 0.95,
        rollout_len: int = 64,
        n_epochs: int = 4,
        seed: int = 0,
        device=None,
    ):
        self.env = env
        self.n_envs = env.n_envs
        self.n_agents = env.n_agents
        self.clip_eps = clip_eps
        self.gamma = gamma
        self.lam = lam
        self.rollout_len = rollout_len
        self.n_epochs = n_epochs
        self.device = device if device is not None else _default_device()

        import torch
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(seed)

        self.actor = SharedGaussianActor(env.OBS_DIM, env.ACT_DIM, hidden_actor, seed,
                                          lr=lr_actor, device=self.device)
        self.critic = MAPPOCentralCritic(self.n_agents, hidden_critic, seed + 999,
                                          use_global_state=True, lr=lr_critic, device=self.device)
        self.buffer = VectorizedMAPPOBuffer(self.n_envs, self.n_agents, env.OBS_DIM, env.ACT_DIM,
                                             rollout_len, device=self.device, use_comm=False)

        self._global_step = 0

    def _topology_flag_batch(self) -> np.ndarray:
        """(n_envs,) float -- 1.0 where that env has at least one outaged
        line, matching MAPPOMARLTrainer's scalar topology_flag computation
        but reduced per-environment instead of per-call."""
        return (~self.env.topology.line_status).any(dim=1).cpu().numpy().astype(float)

    @staticmethod
    def _get_neighbor_stats_batch_envs(obs: np.ndarray) -> np.ndarray:
        """Batched counterpart of MAPPOMARLTrainer._get_neighbor_stats(),
        reduced over the agent axis per environment: obs (n_envs, n_agents,
        obs_dim) -> (n_envs, 6), same formulas/feature columns."""
        mean_tms = obs[:, :, 0].mean(axis=1)
        mean_ps = obs[:, :, 1].mean(axis=1)
        std_tms = obs[:, :, 0].std(axis=1)
        std_ps = obs[:, :, 1].std(axis=1)
        max_viol = obs[:, :, 4].max(axis=1)
        mean_slack = obs[:, :, 3].mean(axis=1)
        return np.stack([mean_tms, mean_ps, std_tms, std_ps, max_viol, mean_slack], axis=1)

    def _collect_rollout(self, obs: np.ndarray) -> np.ndarray:
        """obs: (n_envs, n_agents, OBS_DIM). One batched call per rollout
        step across all n_envs -- the actual throughput win this redesign
        exists to deliver, replacing what the single-env trainer does via
        `n_workers` separate CPU processes each running their own Python
        rollout loop."""
        self.buffer.clear()

        for _ in range(self.rollout_len):
            obs_flat = obs.reshape(self.n_envs * self.n_agents, self.env.OBS_DIM)
            actions_flat, log_probs_flat = self.actor.sample_batch(obs_flat, generator=self.generator)
            actions = actions_flat.reshape(self.n_envs, self.n_agents, self.env.ACT_DIM)
            log_probs = log_probs_flat.reshape(self.n_envs, self.n_agents)

            TMS, PS = self.env.get_state()  # each (n_envs, n_agents)
            pv_level = self.env.current_pv_level  # (n_envs,)
            topology_flag = self._topology_flag_batch()  # (n_envs,)
            neighbor_stats = self._get_neighbor_stats_batch_envs(obs)  # (n_envs, 6)
            value = self.critic.value_batch(
                TMS, PS, pv_level, topology_flag, neighbor_stats,
                fault_type_idx=_fault_type_idx(self.env.fault_type),
            )  # (n_envs,)

            next_obs, rewards, done, info = self.env.step(actions)

            self.buffer.add(
                obs, actions, rewards, log_probs, value, np.full(self.n_envs, done),
                pv_level=pv_level, topology_flag=topology_flag, tms=TMS, ps=PS,
            )

            self._global_step += self.n_envs
            obs = next_obs
            if done:
                obs = self.env.reset()

        return obs

    def train(self, total_steps: int = 30000, verbose: bool = True, log_every: int = 2000):
        """Train for `total_steps` total environment steps (summed across
        all n_envs, matching MAPPOMARLTrainer's `step` accounting
        convention of counting real env.step() calls, now n_envs at a
        time).

        Return schema is deliberately aligned with `MAPPOMARLTrainer.
        train()`'s (`"history"` = list of per-rollout OF values, mean across
        envs; `"cti_satisfaction_history"` = list of per-rollout CTI
        satisfaction rates; `"TMS"`/`"PS"`/`"OF"` = the single best
        (environment, rollout) found across the whole run) so the two
        trainers' outputs can be compared directly, e.g. by
        `run_vectorized_validation.py`. `"actor_loss_history"`/
        `"critic_loss_history"`/`"mean_reward_history"` are additional
        diagnostics `MAPPOMARLTrainer.train()` doesn't expose per rollout.
        """
        obs = self.env.reset()
        step = 0
        of_history = []
        cti_satisfaction_history = []
        actor_loss_history = []
        critic_loss_history = []
        mean_reward_history = []
        best_OF = np.inf
        best_TMS = self.env.TMS[0].copy()
        best_PS = self.env.PS[0].copy()
        t0 = time.perf_counter()

        if verbose:
            print(f"\n{'-' * 55}")
            print(f"  VectorizedMAPPO  ·  {self.env.net.name}  ·  n_envs={self.n_envs}  ·  "
                  f"{self.n_agents} agents")
            print(f"{'-' * 55}")

        while step < total_steps:
            obs = self._collect_rollout(obs)
            T = self.buffer.size
            step += T * self.n_envs

            TMS, PS = self.env.get_state()
            pv_level = self.env.current_pv_level
            topology_flag = self._topology_flag_batch()
            neighbor_stats = self._get_neighbor_stats_batch_envs(obs)
            last_val = self.critic.value_batch(
                TMS, PS, pv_level, topology_flag, neighbor_stats,
                fault_type_idx=_fault_type_idx(self.env.fault_type),
            )  # (n_envs,)

            advs, rets = self.buffer.compute_gae(self.gamma, self.lam, last_val)
            advs_np = advs.cpu().numpy()
            rets_np = rets.cpu().numpy()

            obs_buf = self.buffer.obs[:T].cpu().numpy()      # (T, n_envs, n_agents, OBS_DIM)
            act_buf = self.buffer.actions[:T].cpu().numpy()
            lp_buf = self.buffer.log_probs[:T].cpu().numpy()
            tms_buf = self.buffer.tms[:T].cpu().numpy()
            ps_buf = self.buffer.ps[:T].cpu().numpy()
            pv_buf = self.buffer.pv_levels[:T].cpu().numpy()
            topo_buf = self.buffer.top_flags[:T].cpu().numpy()

            TN = T * self.n_envs
            actor_losses = []
            for ag in range(self.n_agents):
                obs_ag = obs_buf[:, :, ag, :].reshape(TN, self.env.OBS_DIM)
                act_ag = act_buf[:, :, ag, :].reshape(TN, self.env.ACT_DIM)
                adv_ag = advs_np[:, :, ag].reshape(TN)
                lp_ag = lp_buf[:, :, ag].reshape(TN)
                for _ in range(self.n_epochs):
                    loss = self.actor.update(obs_ag, act_ag, adv_ag, lp_ag, clip_eps=self.clip_eps)
                    actor_losses.append(loss)

            neighbor_stats_buf = np.stack(
                [self._get_neighbor_stats_batch_envs(obs_buf[t]) for t in range(T)], axis=0
            )  # (T, n_envs, 6)

            critic_losses = []
            for _ in range(self.n_epochs):
                closs = self.critic.update(
                    tms_buf.reshape(TN, self.n_agents),
                    ps_buf.reshape(TN, self.n_agents),
                    pv_buf.reshape(TN),
                    topo_buf.reshape(TN),
                    neighbor_stats_buf.reshape(TN, 6),
                    rets_np.reshape(TN, self.n_agents),
                    fault_type_idx=_fault_type_idx(self.env.fault_type),
                )
                critic_losses.append(closs)

            mean_reward = float(self.buffer.rewards[:T].mean().item())
            actor_loss = float(np.mean(actor_losses)) if actor_losses else 0.0
            critic_loss = float(np.mean(critic_losses)) if critic_losses else 0.0

            # Per-rollout OF/CTI-satisfaction snapshot, reusing the same
            # cached per-env dynamic fault currents the env itself computed
            # at reset() (Phase V6) -- matching check_constraints()'s single-
            # env CTI-satisfaction definition (delta_cti >= CTI - 1e-6 for
            # every pair), evaluated batched across all n_envs at once.
            OF_batch = objective_function_batch(
                self.env.net, TMS, PS, fault_type=self.env.fault_type,
                fault_currents_batch=(self.env._If_pri_dyn, self.env._If_bkp_dyn),
            )
            t_pri_e, t_bkp_e = compute_operating_times_batch(
                self.env.net, TMS, PS, fault_type=self.env.fault_type,
                fault_currents_batch=(self.env._If_pri_dyn, self.env._If_bkp_dyn),
            )
            cti_ok_batch = np.all(t_bkp_e - t_pri_e >= CTI - 1e-6, axis=1)  # (n_envs,)
            OF_mean = float(np.mean(OF_batch))
            cti_rate = float(np.mean(cti_ok_batch))

            best_idx = int(np.argmin(OF_batch))
            if OF_batch[best_idx] < best_OF:
                best_OF = float(OF_batch[best_idx])
                best_TMS = TMS[best_idx].copy()
                best_PS = PS[best_idx].copy()

            of_history.append(OF_mean)
            cti_satisfaction_history.append(cti_rate)
            actor_loss_history.append(actor_loss)
            critic_loss_history.append(critic_loss)
            mean_reward_history.append(mean_reward)

            if verbose and step % log_every < T * self.n_envs:
                print(f"  step={step:>8d}  OF_mean={OF_mean:.4f}  best_OF={best_OF:.4f}  "
                      f"cti_rate={cti_rate:.2f}  mean_reward={mean_reward:+.4f}  "
                      f"actor_loss={actor_loss:.4f}  critic_loss={critic_loss:.4f}")

        elapsed = time.perf_counter() - t0
        if verbose:
            print(f"  Done. Best OF = {best_OF:.4f}  [{elapsed:.1f}s]")

        return {
            "method": "VectorizedMAPPO",
            "network": self.env.net.name,
            "n_envs": self.n_envs,
            "TMS": best_TMS,
            "PS": best_PS,
            "OF": best_OF,
            "history": of_history,
            "cti_satisfaction_history": cti_satisfaction_history,
            "actor_loss_history": actor_loss_history,
            "critic_loss_history": critic_loss_history,
            "mean_reward_history": mean_reward_history,
            "elapsed": elapsed,
            "stopped_at_step": step,
        }
