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

Phase V11 wires GNN communication (`use_comm=True`) into this trainer, via
`CommGraph.forward_batched()`/`.forward_batched_t()`/`.update_batched()`
(the full degradation model included -- packet loss, latency, quantization,
top-k sparsification -- not just the clean training pipeline), mirroring
`MAPPOMARLTrainer`'s use_comm wiring exactly (same enhanced-vs-raw obs split,
same once-per-rollout GNN update call). `VectorizedSafeRLWrapper` (a
Phase-V8-powered analogue of `SafeRLWrapper`) is still not implemented here
-- Safe RL integration is deferred to a follow-up pass (Phase V12), the same
documented-scope-cut pattern used throughout this redesign.
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
        use_comm: bool = False,
        hidden_actor: int = 64,
        hidden_critic: int = 128,
        lr_actor: float = 3e-4,
        lr_critic: float = 1e-3,
        lr_comm: float = 1e-3,
        clip_eps: float = 0.2,
        gamma: float = 0.99,
        lam: float = 0.95,
        rollout_len: int = 64,
        n_epochs: int = 4,
        seed: int = 0,
        device=None,
        curriculum: bool = False,
        curriculum_stages: int = 6,
    ):
        self.env = env
        self.n_envs = env.n_envs
        self.n_agents = env.n_agents
        self.use_comm = use_comm
        self.clip_eps = clip_eps
        self.gamma = gamma
        self.lam = lam
        self.rollout_len = rollout_len
        self.n_epochs = n_epochs
        self.device = device if device is not None else _default_device()

        # Curriculum learning (Phase V13): same auto-advance trigger as
        # MAPPOMARLTrainer.train() -- recent CTI-satisfaction rate clearing
        # a threshold past a step checkpoint -- wiring
        # VectorizedDynamicDOCREnv.set_curriculum_stage() (already present,
        # previously a dead stub with zero callers in this trainer).
        self.curriculum = curriculum
        self.curriculum_stages = curriculum_stages
        self.current_stage = 0
        self.stage_success_threshold = 0.95

        import torch
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(seed)

        # Communication module (must be created before the actor to know
        # enhanced_dim), mirroring MAPPOMARLTrainer.__init__'s ordering
        # exactly. See CommGraph.forward_batched()/.update_batched() (Phase
        # V11) for the batched comm pipeline this wires in.
        self.comm_graph = None
        if use_comm:
            from marl.comm_graph import CommGraph, build_relay_adjacency
            adj = build_relay_adjacency(env.net)
            self.comm_graph = CommGraph(
                n_agents=self.n_agents,
                obs_dim=env.OBS_DIM,
                enhanced_dim=env.OBS_DIM + 8,  # Original + message features
                adjacency=adj,
                seed=seed + 100,
                device=self.device,
                lr=lr_comm,
            )

        actor_obs_dim = env.OBS_DIM
        buffer_obs_dim = env.OBS_DIM
        if self.use_comm and self.comm_graph is not None:
            actor_obs_dim = self.comm_graph.decoder.enhanced_dim
            buffer_obs_dim = self.comm_graph.decoder.enhanced_dim

        self.actor = SharedGaussianActor(actor_obs_dim, env.ACT_DIM, hidden_actor, seed,
                                          lr=lr_actor, device=self.device)
        self.critic = MAPPOCentralCritic(self.n_agents, hidden_critic, seed + 999,
                                          use_global_state=True, lr=lr_critic, device=self.device)
        self.buffer = VectorizedMAPPOBuffer(self.n_envs, self.n_agents, buffer_obs_dim, env.ACT_DIM,
                                             rollout_len, device=self.device, use_comm=use_comm,
                                             msg_dim=8 if use_comm else 0, raw_obs_dim=env.OBS_DIM)

        # Communication statistics (mirrors MAPPOMARLTrainer's Tier-2
        # bookkeeping for Figs 8/13/22/38, reduced per-rollout-step instead
        # of per-agent since this trainer never calls the GNN per-agent).
        self.comm_bandwidth_history: list = []
        self.attention_checkpoint_interval = 500
        self.attention_snapshots: list = []

        # CommGraph.forward_batched()'s degradation model (dropout, packet
        # loss) needs a np.random.Generator, not the torch.Generator this
        # trainer otherwise uses exclusively for Phase V7's GPU-resident
        # actor noise draw -- kept as a separate, dedicated stream.
        self.comm_rng = np.random.default_rng(seed + 300)

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
        actor_obs_dim = self.comm_graph.decoder.enhanced_dim if self.use_comm else self.env.OBS_DIM

        for _ in range(self.rollout_len):
            if self.use_comm and self.comm_graph is not None:
                enhanced_obs, messages, comm_info = self.comm_graph.forward_batched(obs, self.comm_rng)
                current_obs = enhanced_obs
            else:
                current_obs = obs
                messages = None
                comm_info = {}

            obs_flat = current_obs.reshape(self.n_envs * self.n_agents, actor_obs_dim)
            actions_flat, log_probs_flat = self.actor.sample_batch(obs_flat, generator=self.generator)
            actions = actions_flat.reshape(self.n_envs, self.n_agents, self.env.ACT_DIM)
            log_probs = log_probs_flat.reshape(self.n_envs, self.n_agents)

            TMS, PS = self.env.get_state()  # each (n_envs, n_agents)
            pv_level = self.env.current_pv_level  # (n_envs,)
            topology_flag = self._topology_flag_batch()  # (n_envs,)
            # Neighbor stats always come from the RAW observation, never the
            # GNN-enhanced one -- it indexes raw feature columns (tms_norm,
            # ps_norm, cti_slack_norm, viol_flag) that don't exist in the
            # decoded enhanced representation, same reasoning as
            # MAPPOMARLTrainer._collect_rollout()'s identical comment.
            neighbor_stats = self._get_neighbor_stats_batch_envs(obs)  # (n_envs, 6)
            value = self.critic.value_batch(
                TMS, PS, pv_level, topology_flag, neighbor_stats,
                fault_type_idx=_fault_type_idx(self.env.fault_type),
            )  # (n_envs,)

            next_obs, rewards, done, info = self.env.step(actions)

            self.buffer.add(
                current_obs, actions, rewards, log_probs, value, np.full(self.n_envs, done),
                pv_level=pv_level, topology_flag=topology_flag, tms=TMS, ps=PS,
                messages=messages, attention=comm_info.get("attention") if comm_info else None,
                raw_obs=obs if self.use_comm else None,
            )

            if self.use_comm and comm_info:
                bw = self.comm_graph.get_bandwidth_usage(steps_per_second=1.0)
                self.comm_bandwidth_history.append(bw["kbps"])

            prev_global_step = self._global_step
            self._global_step += self.n_envs
            if (self.use_comm and comm_info and
                    self._global_step // self.attention_checkpoint_interval >
                    prev_global_step // self.attention_checkpoint_interval):
                attn_raw = comm_info.get("attention")  # (n_envs, n_agents, n_agents) or None
                if attn_raw is not None:
                    # Snapshot env 0's attention matrix -- a representative
                    # single-graph sample, matching Figs 8/38's existing
                    # single-env attention-snapshot shape/consumer exactly.
                    attn_mat = np.array(attn_raw[0])
                    eps_e = 1e-9
                    p = np.clip(attn_mat, eps_e, 1.0)
                    p /= p.sum(axis=1, keepdims=True) + eps_e
                    entropy = float(-np.mean(np.sum(p * np.log(p + eps_e), axis=1)))
                    self.attention_snapshots.append({
                        "step": self._global_step,
                        "attn_matrix": attn_mat.copy(),
                        "entropy": entropy,
                    })

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
            print(f"  GNN Communication: {'ON' if self.use_comm else 'OFF'}")
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

            obs_buf = self.buffer.obs[:T].cpu().numpy()      # (T, n_envs, n_agents, actor_obs_dim)
            act_buf = self.buffer.actions[:T].cpu().numpy()
            lp_buf = self.buffer.log_probs[:T].cpu().numpy()
            tms_buf = self.buffer.tms[:T].cpu().numpy()
            ps_buf = self.buffer.ps[:T].cpu().numpy()
            pv_buf = self.buffer.pv_levels[:T].cpu().numpy()
            topo_buf = self.buffer.top_flags[:T].cpu().numpy()
            # buffer.obs holds the GNN-enhanced observation when use_comm is
            # on (same split as MAPPOBuffer) -- neighbor_stats/the critic
            # need the true raw obs instead, since they index raw feature
            # columns (tms_norm, ps_norm, ...) that don't exist in the
            # decoded enhanced representation.
            raw_obs_buf = self.buffer.raw_obs[:T].cpu().numpy() if self.use_comm else obs_buf

            actor_obs_dim = self.comm_graph.decoder.enhanced_dim if self.use_comm else self.env.OBS_DIM
            TN = T * self.n_envs
            actor_losses = []
            for ag in range(self.n_agents):
                obs_ag = obs_buf[:, :, ag, :].reshape(TN, actor_obs_dim)
                act_ag = act_buf[:, :, ag, :].reshape(TN, self.env.ACT_DIM)
                adv_ag = advs_np[:, :, ag].reshape(TN)
                lp_ag = lp_buf[:, :, ag].reshape(TN)
                for _ in range(self.n_epochs):
                    loss = self.actor.update(obs_ag, act_ag, adv_ag, lp_ag, clip_eps=self.clip_eps)
                    actor_losses.append(loss)

            neighbor_stats_buf = np.stack(
                [self._get_neighbor_stats_batch_envs(raw_obs_buf[t]) for t in range(T)], axis=0
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

            # Update GNN communication weights (encoder/attention/decoder) --
            # real PPO-clip backprop through the actual actor objective (see
            # CommGraph.update_batched()'s docstring), once per rollout,
            # mirroring MAPPOMARLTrainer.train()'s identical call. Without
            # this, the GNN's params never move from random initialization.
            if self.use_comm and self.comm_graph is not None:
                self.comm_graph.update_batched(
                    raw_obs_buf, act_buf, advs_np, lp_buf, self.actor, clip_eps=self.clip_eps,
                )

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

            # Curriculum stage advancement -- identical trigger formula to
            # MAPPOMARLTrainer.train()'s (recent_cti_rate here is already a
            # per-rollout MEAN satisfaction RATE across n_envs, not a 0/1
            # flag like the single-env version's per-rollout value, but the
            # same np.mean(...[-100:]) threshold check applies unchanged).
            if self.curriculum:
                recent_cti_rate = (np.mean(cti_satisfaction_history[-100:])
                                    if len(cti_satisfaction_history) >= 100 else 0.0)
                if (recent_cti_rate > self.stage_success_threshold and
                        self.current_stage < self.curriculum_stages - 1 and
                        step > (self.current_stage + 1) * total_steps // self.curriculum_stages):
                    self.current_stage += 1
                    if hasattr(self.env, 'set_curriculum_stage'):
                        self.env.set_curriculum_stage(self.current_stage)
                    if verbose:
                        print(f"  ^ Advanced to curriculum stage {self.current_stage}")

            if verbose and step % log_every < T * self.n_envs:
                stage_info = f"  stage={self.current_stage}" if self.curriculum else ""
                print(f"  step={step:>8d}  OF_mean={OF_mean:.4f}  best_OF={best_OF:.4f}  "
                      f"cti_rate={cti_rate:.2f}  mean_reward={mean_reward:+.4f}  "
                      f"actor_loss={actor_loss:.4f}  critic_loss={critic_loss:.4f}{stage_info}")

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
            "use_comm": self.use_comm,
            "comm_bandwidth_history": self.comm_bandwidth_history,
            "curriculum_stage": self.current_stage if self.curriculum else 0,
        }

    def get_fisher_diagonal_from_buffer(self, n_samples: Optional[int] = None) -> np.ndarray:
        """EWC (Phase V13) Fisher-diagonal estimation via the real
        per-sample torch.autograd.grad-based SharedGaussianActor.
        estimate_fisher_diagonal() -- already actor-generic and already
        more faithful than run_core_training.py's own finite-difference
        approximation -- fed from this trainer's own last rollout buffer
        instead of porting that single-env-structured trajectory-collection
        loop. Works unchanged with
        marl.continual_trainer.ContinualMAPPOTrainer(base_trainer=self, ...),
        which only ever touches `base_trainer.actor.get_all_params()`/
        `.set_all_params()` -- both already satisfied by this trainer's
        actor (the same SharedGaussianActor class MAPPOMARLTrainer uses);
        feed its result straight into
        `ewc_wrapper.ewc_regularizer.update_fisher(task_name, fisher)`.

        Uses `self.buffer.obs` -- what the actor actually consumes each
        step (the GNN-enhanced observation when use_comm, matching the
        actor's real input dimension; raw_obs would be the wrong shape).
        Must be called after at least one `_collect_rollout()`/`train()`
        call has populated the buffer.

        n_samples: optional cap on how many (obs, action) rows to use (a
        random subset of the buffer, without replacement). None uses every
        sample currently in the buffer.
        """
        T = self.buffer.size
        if T == 0:
            raise RuntimeError("buffer is empty -- call train() or _collect_rollout() first")
        actor_obs_dim = self.comm_graph.decoder.enhanced_dim if self.use_comm else self.env.OBS_DIM
        obs_buf = self.buffer.obs[:T].cpu().numpy().reshape(-1, actor_obs_dim)
        act_buf = self.buffer.actions[:T].cpu().numpy().reshape(-1, self.env.ACT_DIM)

        if n_samples is not None and n_samples < obs_buf.shape[0]:
            idx = np.random.default_rng(0).choice(obs_buf.shape[0], size=n_samples, replace=False)
            obs_buf, act_buf = obs_buf[idx], act_buf[idx]

        return self.actor.estimate_fisher_diagonal(obs_buf, act_buf)
