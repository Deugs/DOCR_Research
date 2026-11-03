#!/usr/bin/env python3
"""
run_core_training_vectorized.py
--------------------------------
Phase V14.1 of the GPU-vectorized-environment redesign (see
.claude/plans/elegant-rolling-map.md -- the plan file is outside this repo,
kept in the user's Claude Code plans directory): a from-scratch replication
of Core A's real training+evaluation recipe
(run_core_training.py's _run_core_A_experimental()/run_core_A()),
dispatching to VectorizedMAPPOTrainer/VectorizedSafeRLWrapper (Phases
V9-V13: batched env, GNN comm, Safe RL, EWC) instead of
MAPPOMARLTrainer/SafeRLWrapper, and reproducing _aggregate_runs()'s exact
output schema so CacheManager.put()/run_post_processing.py's load_cache()
and every Core-A-consuming figure script would work unmodified against
this new cache entry.

Does NOT modify run_core_training.py or any existing results/core_cache/
entry. Writes to a DIFFERENT cache name
("core_A_mappo_safe_ewc_8bus_60pv_vectorized") than the real Core A's
("core_A_mappo_safe_ewc_8bus_60pv") -- a new, additional, clearly-labeled
cache entry, never colliding with or overwriting the real one.

Deliberate design choice, not an oversight -- the post-training per-episode
evaluation loop below runs n_envs=1, SEQUENTIALLY, min(n_episodes, 200)
times, NOT n_envs=200 in parallel. Core A's real eval episodes are
independent, sequential single-env episodes where the shared Lagrangian's
dual-ascent update fires once per episode (enforce_cti=True + done=True
every episode), so `lagrange_norms` is a genuinely 200-long EVOLVING
sequence. Running n_envs=200 in parallel would instead produce exactly ONE
lockstep episode boundary for the whole batch, collapsing that 200-long
evolving sequence into one value repeated 200 times -- silently changing
what the cached lagrange_norms field actually measures (Figs 9 and 10 plot
this trajectory directly). Since this eval phase is cheap relative to
training (200 x 300 = 60,000 env steps vs. training's 1,500,000+ for the
real full-scale budget), there is no reason to risk that for a speed win
that has not even been demonstrated on real hardware yet (see the plan's
standing non-goal section -- nothing in this whole redesign should be
treated as a performance win before scripts/benchmark_vectorized_env.py
runs on real CUDA hardware).

Reuses run_core_training.py's UnifiedTrainingLogger, CacheManager,
topology_fp(), and _aggregate_runs() directly (imported, not duplicated) --
importing that module is safe (it is guarded by `if __name__ == "__main__"`,
confirmed before writing this) and keeps this script's output schema
byte-for-byte tied to the real one instead of a hand-copied, driftable
duplicate.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from network.network_data import get_8bus
from network.relay_physics import check_constraints
from marl.vectorized_marl_env import VectorizedDynamicDOCREnv
from marl.vectorized_marl_agents import VectorizedMAPPOTrainer
from marl.safe_rl import create_vectorized_safe_rl_env, SafeRLConfig
from marl.continual_trainer import create_ewc_trainer, EWCConfig

from run_core_training import (
    UnifiedTrainingLogger, CacheManager, topology_fp, _aggregate_runs,
    CACHE_DIR, CHECKPOINT_DIR, FAULT_TYPE, SEED,
    N_RUNS_DEFAULT, N_RUNS_FAST, EPISODES_DEFAULT, EPISODES_FAST, FAST_MODE,
)


def _run_core_A_vectorized_single(n_episodes: int, seed: int) -> Dict:
    """Vectorized-path counterpart of run_core_training.py's
    _run_core_A_experimental() -- same real recipe (8bus, curriculum_stage=3,
    use_comm=True, the identical literal SafeRLConfig, EWCConfig()
    defaults), same UnifiedTrainingLogger-compatible output schema. See
    this module's docstring for why the eval loop stays n_envs=1/sequential.
    """
    net = get_8bus()
    env = VectorizedDynamicDOCREnv(net, n_envs=1, seed=seed, fault_type=FAULT_TYPE, curriculum_stage=3)
    n_agents = env.n_agents
    n_pairs = net.n_pairs

    safe_env = create_vectorized_safe_rl_env(env, SafeRLConfig(
        project_actions=True, enforce_cti=True,
        lambda_init=1.0, lambda_lr=0.01, rho_penalty=10.0,
        use_pid=True, adaptive_rho=True,
    ))
    trainer = VectorizedMAPPOTrainer(safe_env, use_comm=True, seed=seed)
    ewc_trainer = create_ewc_trainer(trainer, EWCConfig())

    logger_obj = UnifiedTrainingLogger(
        n_agents=n_agents, n_pairs=n_pairs,
        snapshot_episodes=[0, 100, 500, 1000, 5000] if n_episodes >= 5000 else [0, 50, 100, 250, 500],
    )
    rng = np.random.default_rng(seed)

    total_steps = n_episodes * env.max_steps
    history = trainer.train(total_steps=total_steps, verbose=False)

    # Attention snapshots: same subsampling as the real Core A (avoids
    # Fig 38's gridspec crash on hundreds of raw per-checkpoint snapshots).
    max_steps_local = env.max_steps
    all_snaps = getattr(trainer, "attention_snapshots", [])
    for snap in all_snaps:
        logger_obj.attention_entropies.append(snap["entropy"])
    max_snapshot_panels = 5
    snap_indices = (np.linspace(0, len(all_snaps) - 1, min(max_snapshot_panels, len(all_snaps)), dtype=int)
                    if all_snaps else [])
    for idx in snap_indices:
        snap = all_snaps[idx]
        pseudo_ep = snap["step"] // max_steps_local
        logger_obj.attention_snapshots[pseudo_ep] = snap["attn_matrix"]

    # EWC: real per-sample Fisher estimator (Phase V13), fed from this
    # trainer's own last rollout buffer -- deliberately NOT replicating
    # Core A's manual short-rollout + finite-difference approximation,
    # since the real estimator is already built, already more faithful,
    # and already actor-generic (see get_fisher_diagonal_from_buffer()'s
    # own docstring in marl/vectorized_marl_agents.py).
    cur_params = trainer.actor.get_all_params()
    fisher = trainer.get_fisher_diagonal_from_buffer()
    ewc_trainer.ewc_regularizer.register_task("PV060", cur_params)
    ewc_trainer.ewc_regularizer.update_fisher("PV060", fisher)
    ewc_trainer.ewc_regularizer.finalize()

    # Post-training evaluation rollout, n_envs=1, sequential (see module
    # docstring for why) -- uses safe_env so action projection and the
    # Lagrangian CTI penalty are exercised, matching how the policy trained.
    action_projection_samples = {"raw": [], "projected": []}
    max_projection_samples = 200

    for ep in range(min(n_episodes, 200)):
        obs = safe_env.reset(seed=seed + ep)[0]  # (1, n_agents, OBS_DIM) -> (n_agents, OBS_DIM)
        done = False
        ep_reward = 0.0
        ep_safety_acts = 0
        ep_projections = 0
        while not done:
            if trainer.use_comm and trainer.comm_graph is not None:
                enhanced_obs, _, _ = trainer.comm_graph.forward(obs, rng)
            else:
                enhanced_obs = obs

            actions = np.zeros((n_agents, env.ACT_DIM))
            for i in range(n_agents):
                a, _ = trainer.actor.sample(enhanced_obs[i], rng)
                actions[i] = a

            if len(action_projection_samples["raw"]) < max_projection_samples:
                TMS_cur, PS_cur = safe_env.get_state()
                TMS_cur, PS_cur = TMS_cur[0], PS_cur[0]
                proj_dt, proj_dp = safe_env.projector.project(
                    TMS_cur, PS_cur, actions[:, 0], actions[:, 1],
                    action_scale=safe_env.action_scale)
                scale = safe_env.action_scale
                raw_tms = TMS_cur + actions[:, 0] * scale * safe_env.projector.tms_range
                raw_ps = PS_cur + actions[:, 1] * scale * safe_env.projector.ps_range
                proj_tms = TMS_cur + proj_dt * scale * safe_env.projector.tms_range
                proj_ps = PS_cur + proj_dp * scale * safe_env.projector.ps_range
                for i in range(n_agents):
                    action_projection_samples["raw"].append((float(raw_tms[i]), float(raw_ps[i])))
                    action_projection_samples["projected"].append((float(proj_tms[i]), float(proj_ps[i])))

            next_obs, reward, done, info = safe_env.step(actions[None, :, :])
            obs = next_obs[0]
            ep_reward += float(np.mean(reward[0]))
            if bool(info["safe_rl"]["projected_actions"][0]):
                ep_projections += 1
            ep_safety_acts += int(info["safe_rl"]["cti_violations"][0])

        TMS, PS = safe_env.get_state()
        c = check_constraints(net, TMS[0], PS[0])
        lag_norm = float(np.mean(safe_env.get_safe_rl_stats().get("lambda_mean", 0.0)))
        # Real attention snapshots (Figs 8, 38) are populated above from
        # training itself -- this eval rollout doesn't expose per-step
        # attention, same reasoning as the real Core A's identical note.
        logger_obj.log_episode(
            ep=ep, global_return=ep_reward,
            per_agent_rewards=np.ones(n_agents) * ep_reward / n_agents,
            cti_violations=c["cti_violations"], lagrange_norm=lag_norm,
            safety_acts=ep_safety_acts, attention_matrix=None,
            action_projections=ep_projections,
        )

    result = logger_obj.to_dict()
    result["action_projection_samples"] = action_projection_samples
    return result


def run_core_A_vectorized(n_runs: Optional[int] = None, n_episodes: Optional[int] = None,
                           seed: Optional[int] = None, cache: Optional[CacheManager] = None) -> Dict:
    """Vectorized-path counterpart of run_core_training.py's run_core_A().

    Writes to a cache entry under a DIFFERENT name
    ("core_A_mappo_safe_ewc_8bus_60pv_vectorized") so it never collides
    with or overwrites the real Core A cache entry -- see module docstring.
    """
    n_runs = n_runs if n_runs is not None else (N_RUNS_FAST if FAST_MODE else N_RUNS_DEFAULT)
    n_episodes = n_episodes if n_episodes is not None else (EPISODES_FAST if FAST_MODE else EPISODES_DEFAULT)
    seed = seed if seed is not None else SEED
    cache = cache if cache is not None else CacheManager(CACHE_DIR)

    config = {
        "name": "core_A_mappo_safe_ewc_8bus_60pv_vectorized",
        "method": "MAPPO-Safe-EWC-Vectorized",
        "system": "8bus",
        "pv": 0.6,
        "n_runs": n_runs,
        "n_episodes": n_episodes,
        "seed": seed,
        "fault_type": FAULT_TYPE,
        "vectorized_path_v14_1": True,
        "topology_fp": topology_fp("8bus"),
    }
    cached = cache.get(config)
    if cached is not None:
        return cached

    all_runs_data: List[Optional[Dict]] = [None] * n_runs
    for run_idx in range(n_runs):
        run_seed = seed + run_idx
        all_runs_data[run_idx] = _run_core_A_vectorized_single(n_episodes, run_seed)

    aggregated = _aggregate_runs(all_runs_data)
    aggregated["config"] = config
    aggregated["checkpoint_path"] = str(CHECKPOINT_DIR / "core_A_vectorized_best.pt")
    cache.put(config, aggregated)
    return aggregated


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-runs", type=int, default=None)
    parser.add_argument("--n-episodes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    result = run_core_A_vectorized(n_runs=args.n_runs, n_episodes=args.n_episodes, seed=args.seed)
    print(f"Core A (vectorized): n_runs={result['n_runs']}  "
          f"final global_return_mean={result['global_returns_mean'][-1]:.4f}  "
          f"final cti_violation_rate_mean={result['cti_violation_rates_mean'][-1]:.2f}%")
