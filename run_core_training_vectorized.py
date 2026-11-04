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
import itertools
import time
from typing import Dict, List, Optional

import numpy as np

from network.network_data import get_8bus, NETWORKS, load_network
from network.relay_physics import check_constraints, objective_function
from marl.vectorized_marl_env import VectorizedDynamicDOCREnv
from marl.vectorized_marl_agents import VectorizedMAPPOTrainer
from marl.safe_rl import create_vectorized_safe_rl_env, SafeRLConfig
from marl.continual_trainer import create_ewc_trainer, EWCConfig, TaskSimilarityEstimator

from run_core_training import (
    UnifiedTrainingLogger, CacheManager, topology_fp, _aggregate_runs,
    CACHE_DIR, CHECKPOINT_DIR, FAULT_TYPE, SEED,
    N_RUNS_DEFAULT, N_RUNS_FAST, EPISODES_DEFAULT, EPISODES_FAST, FAST_MODE,
    N_RUNS_BASELINE, N_RUNS_BASELINE_FAST, _method_seed_offset,
    N_RUNS_HPARAM, N_RUNS_HPARAM_FAST, EPISODES_HPARAM, EPISODES_HPARAM_FAST,
)

# Phase V14.7 (Cores G-K): TrainingConfig/TrainingResult/TrainingPipeline are
# already generic, trainer-agnostic containers (confirmed by reading
# experiments/training_pipeline.py directly -- every _eval_* method reads
# only dataclass fields, never anything MAPPOMARLTrainer-specific), and
# core_eval_bridge.py already imports them the same way -- reused directly
# rather than duplicated, same precedent as reusing run_core_training.py's
# own UnifiedTrainingLogger/CacheManager/_aggregate_runs above.
from experiments.training_pipeline import TrainingConfig, TrainingResult, TrainingPipeline


def _run_eval_episode(safe_env, trainer, rng, seed, noise_std: float = 0.0):
    """Run one full episode with the current policy against an n_envs=1
    VectorizedDynamicDOCREnv/VectorizedSafeRLWrapper pair, sequentially --
    see module docstring for why eval episodes across this whole script
    stay n_envs=1/sequential rather than batched. Mirrors the
    structurally-repeated single-env eval loop pattern throughout
    run_core_training.py's real Cores (act_fn + safe_env.step() in a
    while-not-done loop). Returns (ep_reward, final_TMS, final_PS) --
    enough for every Core B/C eval loop's needs (return-only or
    final-state-only); Core A's own eval loop needed extra per-step
    info["safe_rl"] tracking and action-projection sampling, so it stays
    hand-written rather than forced through this shared helper.

    noise_std, when >0, adds fresh IID Gaussian noise to the policy's
    observation input each step (NOT to the env's own evolving state) --
    mirrors Core C's real "Robustness" eval (run_core_training.py:
    1925-1939), which perturbs only what the policy sees, not what
    actually happens physically, each step independently."""
    n_agents = trainer.n_agents
    obs = safe_env.reset(seed=seed)[0]
    done = False
    ep_reward = 0.0
    while not done:
        obs_in = obs + rng.normal(0, noise_std, size=obs.shape) if noise_std > 0 else obs
        if trainer.use_comm and trainer.comm_graph is not None:
            enhanced_obs, _, _ = trainer.comm_graph.forward(obs_in, rng)
        else:
            enhanced_obs = obs_in
        actions = np.zeros((n_agents, trainer.env.ACT_DIM))
        for i in range(n_agents):
            a, _ = trainer.actor.sample(enhanced_obs[i], rng)
            actions[i] = a
        next_obs, reward, done, _info = safe_env.step(actions[None, :, :])
        obs = next_obs[0]
        ep_reward += float(np.mean(reward[0]))
    TMS, PS = safe_env.get_state()
    return ep_reward, TMS[0], PS[0]


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
    trainer.train(total_steps=total_steps, verbose=False)

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


# ═════════════════════════════════════════════════════════════════
# Core B (Phase V14.2) -- Curriculum training (0% -> 60% PV)
# ═════════════════════════════════════════════════════════════════
def _run_curriculum_single_vectorized(use_ewc: bool, seed: int, episodes_per_task: int,
                                       system: str = "8bus",
                                       n_eval_episodes: int = 30) -> Dict:
    """Vectorized-path counterpart of run_core_training.py's
    _run_curriculum_single() (Core B). Same real recipe: the 4-stage PV
    curriculum [0.0, 0.20, 0.40, 0.60], use_comm=True, the identical
    literal SafeRLConfig, EWCConfig() defaults with real per-task-
    transition Fisher + TaskSimilarityEstimator-driven adapt_lambda.
    Deliberately NOT replicating Core B's manual short-rollout +
    finite-difference Fisher estimate -- same Phase V13 "use the real
    per-sample estimator instead" decision as Core A's vectorized
    counterpart. Eval loops use _run_eval_episode() (n_envs=1, sequential
    -- see module docstring).

    n_eval_episodes defaults to 30 to faithfully match the real Core B's
    hardcoded range(30) eval loops (run_core_training.py:1551,1601 -- NOT
    reduced by FAST_MODE there either). Tests override this down (e.g. 2)
    purely for runtime -- real Core B's per-task-transition eval sweep is
    4 tasks x 4 eval_pv x 30 episodes = 480 (task_perf) + 120 (pv_evaluation)
    = 600 sequential full episodes per use_ewc arm, which is correct
    fidelity but far too slow for a schema/correctness smoke test."""
    TASKS = [0.0, 0.20, 0.40, 0.60]
    net = NETWORKS[system]()
    env = VectorizedDynamicDOCREnv(net, n_envs=1, seed=seed, fault_type=FAULT_TYPE, curriculum_stage=0)
    safe_env = create_vectorized_safe_rl_env(env, SafeRLConfig(
        project_actions=True, enforce_cti=True,
        lambda_init=1.0, lambda_lr=0.01, rho_penalty=10.0,
        use_pid=True, adaptive_rho=True,
    ))
    base_trainer = VectorizedMAPPOTrainer(safe_env, use_comm=True, seed=seed)
    ewc_wrapper = create_ewc_trainer(base_trainer, EWCConfig()) if use_ewc else None
    similarity_estimator = TaskSimilarityEstimator() if use_ewc else None

    rng = np.random.default_rng(seed)
    steps_per_task = episodes_per_task * env.max_steps

    task_performances = []
    curriculum_history = []
    prev_task_name = None
    prev_fisher = None
    prev_conditions = None
    prev_params = None

    for pv in TASKS:
        env.set_curriculum_stage(int(pv * 5))  # 0.0->0, 0.2->1, 0.4->2, 0.6->3

        if use_ewc and ewc_wrapper is not None and prev_task_name is not None:
            cur = base_trainer.actor.get_all_params()
            ewc_grad = ewc_wrapper.apply_ewc_to_gradient(cur, np.zeros_like(cur))
            base_trainer.actor.set_all_params(cur - 0.01 * ewc_grad)

        stage_result = base_trainer.train(total_steps=steps_per_task, verbose=False)
        curriculum_history.extend(stage_result.get("history", []))
        task_name = f"PV{int(pv * 100):03d}"

        if use_ewc and ewc_wrapper is not None:
            cur = base_trainer.actor.get_all_params()
            fisher = base_trainer.get_fisher_diagonal_from_buffer()

            cur_conditions = {"pv_level": pv}
            ewc_wrapper.ewc_regularizer.register_task(task_name, cur)
            if prev_conditions is not None:
                similarity = similarity_estimator.estimate_similarity(
                    prev_conditions, prev_params, cur_conditions, cur
                )
                ewc_wrapper.ewc_regularizer.adapt_lambda(similarity)
            if prev_task_name is not None:
                ewc_wrapper.ewc_regularizer.update_fisher(prev_task_name, prev_fisher)
            prev_task_name = task_name
            prev_fisher = fisher
            prev_conditions = cur_conditions
            prev_params = cur.copy()

        task_perf = []
        for eval_pv in TASKS:
            env.set_curriculum_stage(int(eval_pv * 5))
            rewards = []
            for ep in range(n_eval_episodes):
                ep_reward, _, _ = _run_eval_episode(
                    safe_env, base_trainer, rng, seed=seed + 50000 + int(eval_pv * 1000) + ep)
                rewards.append(ep_reward)
            task_perf.append(float(np.mean(rewards)))
        task_performances.append(task_perf)

    if use_ewc and ewc_wrapper is not None:
        # TASKS[-1] is only consolidated via the *next* task's register_task()
        # call, which doesn't exist for the last task -- finalize() alone
        # would otherwise consolidate it with a zero-information placeholder
        # Fisher instead of the real one just computed. Same ordering fix
        # the real Core B applies.
        ewc_wrapper.ewc_regularizer.update_fisher(prev_task_name, prev_fisher)
        ewc_wrapper.ewc_regularizer.finalize()

    pv_evaluation = {}
    for eval_pv in TASKS:
        env.set_curriculum_stage(int(eval_pv * 5))
        of_vals, cti_rates = [], []
        for ep in range(n_eval_episodes):
            _, TMS, PS = _run_eval_episode(
                safe_env, base_trainer, rng, seed=seed + 70000 + int(eval_pv * 1000) + ep)
            of_val, _, _ = objective_function(net, TMS, PS, fault_type=FAULT_TYPE)
            c = check_constraints(net, TMS, PS, fault_type=FAULT_TYPE)
            of_vals.append(of_val)
            cti_rates.append(c["cti_violations"] / net.n_pairs)
        pv_evaluation[eval_pv] = {
            "objective": float(np.mean(of_vals)),
            "cti_rate": float(np.mean(cti_rates)),
        }

    return {
        "task_performances": np.array(task_performances),
        "use_ewc": use_ewc,
        "final_checkpoint": "saved",
        "pv_evaluation": pv_evaluation,
        "fisher": prev_fisher if use_ewc else None,
        "history": curriculum_history,
    }


def run_core_B_vectorized(n_runs: Optional[int] = None, episodes_per_task: Optional[int] = None,
                           seed: Optional[int] = None, system: str = "8bus",
                           cache: Optional[CacheManager] = None,
                           n_eval_episodes: int = 30) -> Dict:
    """Vectorized-path counterpart of run_core_training.py's run_core_B().

    Writes to a cache entry under a DIFFERENT name
    (f"{real_name}_vectorized") so it never collides with or overwrites the
    real Core B cache entry -- see module docstring. n_eval_episodes
    defaults to 30 (faithful to the real Core B); tests override it down
    for runtime -- see _run_curriculum_single_vectorized's docstring.
    """
    n_runs = n_runs if n_runs is not None else (N_RUNS_BASELINE_FAST if FAST_MODE else N_RUNS_BASELINE)
    episodes_per_task = episodes_per_task if episodes_per_task is not None else (
        (EPISODES_FAST if FAST_MODE else EPISODES_DEFAULT) // 10)
    seed = seed if seed is not None else SEED
    cache = cache if cache is not None else CacheManager(CACHE_DIR)

    TASKS = [0.0, 0.20, 0.40, 0.60]
    base_name = "core_B_curriculum" if system == "8bus" else f"core_B_curriculum_{system}"
    config = {
        "name": f"{base_name}_vectorized",
        "system": system,
        "tasks": TASKS,
        "episodes_per_task": episodes_per_task,
        "n_runs": n_runs,
        "seed": seed,
        "fault_type": FAULT_TYPE,
        "vectorized_path_v14_2": True,
        "topology_fp": topology_fp(system),
        "n_eval_episodes": n_eval_episodes,
    }
    cached = cache.get(config)
    if cached is not None:
        return cached

    with_ewc: List[Optional[Dict]] = [None] * n_runs
    without_ewc: List[Optional[Dict]] = [None] * n_runs
    task_evaluations: List[Optional[Dict]] = [None] * n_runs

    for run_idx in range(n_runs):
        result_ewc = _run_curriculum_single_vectorized(
            True, seed + run_idx, episodes_per_task, system=system,
            n_eval_episodes=n_eval_episodes)
        with_ewc[run_idx] = result_ewc
        task_evaluations[run_idx] = result_ewc.get("pv_evaluation")

        without_ewc[run_idx] = _run_curriculum_single_vectorized(
            False, seed + run_idx + 10000, episodes_per_task, system=system,
            n_eval_episodes=n_eval_episodes)

    all_results = {"with_ewc": with_ewc, "without_ewc": without_ewc, "task_evaluations": task_evaluations}
    cache.put(config, all_results)
    return all_results


# ═════════════════════════════════════════════════════════════════
# Core C (Phase V14.3) -- Baseline comparison (5 methods)
# ═════════════════════════════════════════════════════════════════
def _evaluate_traditional_vectorized(net, n_episodes: int, seed: int) -> "tuple[np.ndarray, Dict]":
    """Vectorized-path counterpart of run_core_training.py's
    _evaluate_traditional() -- zero-action baseline, no trainer at all.
    n_envs=1, sequential (see module docstring)."""
    env = VectorizedDynamicDOCREnv(net, n_envs=1, seed=seed, fault_type=FAULT_TYPE, curriculum_stage=3)
    returns: List[float] = []
    cti_satisfied_flags: List[bool] = []
    TMS = PS = None
    for ep in range(n_episodes):
        env.reset(seed=seed + ep)
        done = False
        ep_return = 0.0
        zero_actions = np.zeros((1, env.n_agents, env.ACT_DIM))
        while not done:
            _, reward, done, _info = env.step(zero_actions)
            ep_return += float(np.mean(reward[0]))
        returns.append(ep_return)
        TMS, PS = env.get_state()
        TMS, PS = TMS[0], PS[0]
        cti_satisfied_flags.append(check_constraints(net, TMS, PS, fault_type=FAULT_TYPE)["cti_satisfied"])

    returns_arr = np.array(returns)
    final_constraints = check_constraints(net, TMS, PS, fault_type=FAULT_TYPE)
    metrics = {
        "cti_satisfaction_rate": float(np.mean(cti_satisfied_flags)) if cti_satisfied_flags else None,
        "convergence_ep": 0,
        "sample_efficiency": None,
        "robustness": 1.0,
        "top_pri_mean": final_constraints["top_pri_mean"],
        "delta_cti_mean": float(np.mean(final_constraints["delta_cti"])),
        "final_of": None,
    }
    return returns_arr, metrics


def _run_baseline_single_vectorized(method: str, seed: int, n_episodes: int,
                                     system: str = "8bus") -> "tuple[np.ndarray, Optional[Dict]]":
    """Vectorized-path counterpart of run_core_training.py's
    _run_baseline_single() (Core C). Confirmed from the real code (not
    assumed): "IPPO" here is NOT a separate single-agent-PPO trainer --
    MAPPO-Safe-EWC/MAPPO-Safe/MAPPO/IPPO all route through the same
    MAPPOMARLTrainer class, differing only in use_comm/use_safe/use_ewc;
    only "Traditional" (handled by _evaluate_traditional_vectorized) has
    no trainer at all. use_comm/use_safe/use_ewc per method mirror the
    real function's exact boolean table
    (run_core_training.py:1826-1828). EWC (when use_ewc) uses the real
    per-sample Fisher estimator from the trainer's own rollout buffer
    (Phase V13's get_fisher_diagonal_from_buffer), not the real
    function's manual finite-difference estimate -- same established
    decision as Core A/B."""
    net = load_network(system)
    if method == "Traditional":
        return _evaluate_traditional_vectorized(net, n_episodes, seed)

    use_safe = method in ["MAPPO-Safe-EWC", "MAPPO-Safe"]
    use_ewc = method == "MAPPO-Safe-EWC"
    use_comm = method in ["MAPPO-Safe-EWC", "MAPPO-Safe", "MAPPO"]

    env = VectorizedDynamicDOCREnv(net, n_envs=1, seed=seed, fault_type=FAULT_TYPE, curriculum_stage=3)
    train_env = env
    if use_safe:
        train_env = create_vectorized_safe_rl_env(env, SafeRLConfig(
            project_actions=True, enforce_cti=True,
            lambda_init=1.0, lambda_lr=0.01, rho_penalty=10.0,
            use_pid=True, adaptive_rho=True,
        ))
    trainer = VectorizedMAPPOTrainer(train_env, use_comm=use_comm, seed=seed)

    total_steps = n_episodes * env.max_steps
    trainer.train(total_steps=total_steps, verbose=False)

    if use_ewc:
        ewc_trainer = create_ewc_trainer(trainer, EWCConfig())
        cur = trainer.actor.get_all_params()
        fisher = trainer.get_fisher_diagonal_from_buffer()
        ewc_trainer.ewc_regularizer.register_task(method, cur)
        ewc_trainer.ewc_regularizer.update_fisher(method, fisher)
        ewc_trainer.ewc_regularizer.finalize()

    rng = np.random.default_rng(seed + 9999)
    returns: List[float] = []
    cti_satisfied_flags: List[bool] = []
    final_TMS = final_PS = None
    for ep in range(min(n_episodes, 200)):
        ep_reward, TMS, PS = _run_eval_episode(train_env, trainer, rng, seed=seed + ep)
        returns.append(ep_reward)
        final_TMS, final_PS = TMS, PS
        cti_satisfied_flags.append(
            check_constraints(net, TMS, PS, fault_type=FAULT_TYPE)["cti_satisfied"])
    returns_arr = np.array(returns)

    # Fig 33: real "Robustness" -- relative performance under observation
    # noise (see _run_eval_episode's noise_std docstring for exactly what
    # is and isn't perturbed).
    noisy_rng = np.random.default_rng(seed + 20000)
    noisy_returns = []
    for ep in range(min(10, n_episodes)):
        ep_reward, _, _ = _run_eval_episode(
            train_env, trainer, noisy_rng, seed=seed + 30000 + ep, noise_std=0.1)
        noisy_returns.append(ep_reward)
    clean_mean = float(np.mean(returns_arr[-100:] if len(returns_arr) > 100 else returns_arr))
    noisy_mean = float(np.mean(noisy_returns))
    robustness = float(noisy_mean / clean_mean) if clean_mean else None

    n_eps = len(returns_arr)
    plateau = float(np.mean(returns_arr[-max(1, n_eps // 5):]))
    tolerance = 0.1 * abs(plateau) if plateau else 1e-6
    converged = np.where(np.abs(returns_arr - plateau) <= tolerance)[0]
    convergence_ep = int(converged[0]) if len(converged) else n_eps
    first_half_mean = float(np.mean(returns_arr[:max(1, n_eps // 2)]))
    sample_efficiency = float(first_half_mean / plateau) if plateau else None

    metrics = {
        "cti_satisfaction_rate": float(np.mean(cti_satisfied_flags)) if cti_satisfied_flags else None,
        "convergence_ep": convergence_ep,
        "sample_efficiency": sample_efficiency,
        "robustness": robustness,
        "top_pri_mean": None,
        "delta_cti_mean": None,
        "final_of": None,
    }
    if final_TMS is not None:
        final_constraints = check_constraints(net, final_TMS, final_PS, fault_type=FAULT_TYPE)
        metrics["top_pri_mean"] = final_constraints["top_pri_mean"]
        metrics["delta_cti_mean"] = float(np.mean(final_constraints["delta_cti"]))
        of_val, _, _ = objective_function(net, final_TMS, final_PS, fault_type=FAULT_TYPE)
        metrics["final_of"] = float(of_val)

    return returns_arr, metrics


_CORE_C_METHODS = ["MAPPO-Safe-EWC", "MAPPO-Safe", "MAPPO", "IPPO", "Traditional"]


def run_core_C_vectorized(n_runs: Optional[int] = None, n_episodes: Optional[int] = None,
                           seed: Optional[int] = None, system: str = "8bus",
                           cache: Optional[CacheManager] = None) -> Dict:
    """Vectorized-path counterpart of run_core_training.py's run_core_C()
    (5-method baseline comparison). Writes to a cache entry under a
    DIFFERENT name (f"core_C_baselines_{system}_vectorized") so it never
    collides with the real Core C cache entry -- see module docstring.
    Mirrors the real function's TWO-cache-entry pattern exactly: the
    returned {method: np.ndarray} dict is cached under this function's own
    name, and real per-method metrics (Figs 33/55/56) are cached
    separately under f"core_C_metrics_{system}_vectorized" (only for
    methods with real -- not None -- metrics), matching
    run_core_training.py:1796-1802's reasoning: every existing consumer
    iterating `results.keys()` expecting only method names keeps working
    unchanged."""
    n_runs = n_runs if n_runs is not None else (N_RUNS_FAST if FAST_MODE else N_RUNS_DEFAULT)
    n_episodes = n_episodes if n_episodes is not None else (EPISODES_FAST if FAST_MODE else EPISODES_DEFAULT)
    seed = seed if seed is not None else SEED
    cache = cache if cache is not None else CacheManager(CACHE_DIR)

    config = {
        "name": f"core_C_baselines_{system}_vectorized",
        "system": system,
        "methods": _CORE_C_METHODS,
        "n_runs": n_runs,
        "n_episodes": n_episodes,
        "seed": seed,
        "fault_type": FAULT_TYPE,
        "vectorized_path_v14_3": True,
        "topology_fp": topology_fp(system),
    }
    cached = cache.get(config)
    if cached is not None:
        return cached

    results: Dict[str, List[Optional[np.ndarray]]] = {method: [None] * n_runs for method in _CORE_C_METHODS}
    metrics_by_method: Dict[str, List[Dict]] = {method: [] for method in _CORE_C_METHODS}
    for method in _CORE_C_METHODS:
        for run_idx in range(n_runs):
            run_seed = seed + run_idx + _method_seed_offset(method)
            returns, run_metrics = _run_baseline_single_vectorized(method, run_seed, n_episodes, system=system)
            results[method][run_idx] = returns
            if run_metrics is not None:
                metrics_by_method[method].append(run_metrics)

    final_results: Dict = {method: np.array(arr) for method, arr in results.items()}

    metric_keys = ["cti_satisfaction_rate", "convergence_ep", "sample_efficiency",
                   "robustness", "top_pri_mean", "delta_cti_mean", "final_of"]
    real_metrics = {}
    for method, runs_metrics in metrics_by_method.items():
        if not runs_metrics:
            continue
        real_metrics[method] = {
            key: float(np.mean([m[key] for m in runs_metrics if m.get(key) is not None]))
                 if any(m.get(key) is not None for m in runs_metrics) else None
            for key in metric_keys
        }
        real_metrics[method]["per_run"] = runs_metrics
    if real_metrics:
        cache.put({**config, "name": f"core_C_metrics_{system}_vectorized"}, real_metrics)

    cache.put(config, final_results)
    return final_results


# ═════════════════════════════════════════════════════════════════
# Core D (Phase V14.4) -- 5-arm ablation study
# ═════════════════════════════════════════════════════════════════
_CORE_D_CONFIGS = ["MAPPO", "MAPPO+GNN", "MAPPO+GNN+Safe", "MAPPO+GNN+Safe+EWC", "MAPPO+GNN+Safe+EWC+Curr"]


def _run_ablation_single_vectorized(config_name: str, seed: int, n_episodes: int) -> "tuple[float, float]":
    """Vectorized-path counterpart of run_core_training.py's
    _run_ablation_single() (Core D). use_comm/use_safe/use_ewc/use_curr
    are parsed from config_name exactly as the real function does
    ("GNN"/"Safe"/"EWC"/"Curr" substring checks, run_core_training.py:
    2138-2141). EWC (when use_ewc) uses the real per-sample Fisher
    estimator from the trainer's own rollout buffer
    (get_fisher_diagonal_from_buffer, Phase V13), not the real
    function's manual finite-difference estimate -- same established
    decision as Core A/B/C. The use_curr arm's 4-stage curriculum +
    per-task-transition EWC mirrors Core B's vectorized counterpart's
    structure exactly. Eval is a fixed 50 episodes (NOT
    min(n_episodes, 200) like Core A/C -- confirmed from the real
    function's hardcoded `for ep in range(50)`), tracking the real
    per-step info["OF"] minimum over the episode (not a post-episode
    objective_function() call) and check_constraints()'s raw
    cti_violations COUNT (not a rate, unlike Core A/B/C) -- both
    faithfully reproduced as-is, not "corrected" to match other Cores'
    conventions."""
    use_comm = "GNN" in config_name
    use_safe = "Safe" in config_name
    use_ewc = "EWC" in config_name
    use_curr = "Curr" in config_name

    net = get_8bus()
    env = VectorizedDynamicDOCREnv(net, n_envs=1, seed=seed, fault_type=FAULT_TYPE,
                                    curriculum_stage=0 if use_curr else 3)
    train_env = env
    if use_safe:
        train_env = create_vectorized_safe_rl_env(env, SafeRLConfig(
            project_actions=True, enforce_cti=True,
            lambda_init=1.0, lambda_lr=0.01, rho_penalty=10.0,
            use_pid=True, adaptive_rho=True,
        ))
    trainer = VectorizedMAPPOTrainer(train_env, use_comm=use_comm, seed=seed)
    ewc_trainer = create_ewc_trainer(trainer, EWCConfig()) if use_ewc else None

    total_steps = n_episodes * env.max_steps
    rng = np.random.default_rng(seed + 9999)

    if use_curr:
        steps_per_task = total_steps // 4
        TASKS = [0.0, 0.20, 0.40, 0.60]
        prev_task_name = prev_fisher = None
        prev_conditions = prev_params = None
        similarity_estimator = TaskSimilarityEstimator() if use_ewc else None
        for pv in TASKS:
            env.set_curriculum_stage(int(pv * 5))

            if use_ewc and prev_task_name is not None:
                cur = trainer.actor.get_all_params()
                ewc_grad = ewc_trainer.apply_ewc_to_gradient(cur, np.zeros_like(cur))
                trainer.actor.set_all_params(cur - 0.01 * ewc_grad)

            trainer.train(total_steps=steps_per_task, verbose=False)
            task_name = f"PV{int(pv * 100):03d}"

            if use_ewc:
                cur = trainer.actor.get_all_params()
                fisher = trainer.get_fisher_diagonal_from_buffer()
                cur_conditions = {"pv_level": pv}
                ewc_trainer.ewc_regularizer.register_task(task_name, cur)
                if prev_conditions is not None:
                    similarity = similarity_estimator.estimate_similarity(
                        prev_conditions, prev_params, cur_conditions, cur)
                    ewc_trainer.ewc_regularizer.adapt_lambda(similarity)
                if prev_task_name is not None:
                    ewc_trainer.ewc_regularizer.update_fisher(prev_task_name, prev_fisher)
                prev_task_name, prev_fisher = task_name, fisher
                prev_conditions, prev_params = cur_conditions, cur.copy()
        if use_ewc:
            ewc_trainer.ewc_regularizer.update_fisher(prev_task_name, prev_fisher)
            ewc_trainer.ewc_regularizer.finalize()
    else:
        trainer.train(total_steps=total_steps, verbose=False)
        if use_ewc:
            cur = trainer.actor.get_all_params()
            fisher = trainer.get_fisher_diagonal_from_buffer()
            ewc_trainer.ewc_regularizer.register_task(config_name, cur)
            ewc_trainer.ewc_regularizer.update_fisher(config_name, fisher)
            ewc_trainer.ewc_regularizer.finalize()

    of_values, cti_values = [], []
    for ep in range(50):
        obs = train_env.reset(seed=seed + 999 + ep)[0]
        done = False
        ep_of = float("inf")
        while not done:
            if trainer.use_comm and trainer.comm_graph is not None:
                enhanced_obs, _, _ = trainer.comm_graph.forward(obs, rng)
            else:
                enhanced_obs = obs
            actions = np.zeros((env.n_agents, env.ACT_DIM))
            for i in range(env.n_agents):
                a, _ = trainer.actor.sample(enhanced_obs[i], rng)
                actions[i] = a
            next_obs, _, done, info = train_env.step(actions[None, :, :])
            obs = next_obs[0]
            ep_of = min(ep_of, float(np.asarray(info.get("OF", float("inf"))).reshape(-1)[0]))
        of_values.append(ep_of)
        TMS, PS = train_env.get_state()
        TMS, PS = TMS[0], PS[0]
        c = check_constraints(net, TMS, PS, fault_type=FAULT_TYPE)
        cti_values.append(c["cti_violations"])

    return float(np.mean(of_values)), float(np.mean(cti_values))


def run_core_D_vectorized(n_runs: Optional[int] = None, n_episodes: Optional[int] = None,
                           seed: Optional[int] = None, cache: Optional[CacheManager] = None) -> Dict:
    """Vectorized-path counterpart of run_core_training.py's run_core_D()
    (5-arm ablation study). Writes to a cache entry under a DIFFERENT
    name ("core_D_ablation_vectorized") so it never collides with the
    real Core D cache entry -- see module docstring."""
    n_runs = n_runs if n_runs is not None else (N_RUNS_BASELINE_FAST if FAST_MODE else N_RUNS_BASELINE)
    n_episodes = n_episodes if n_episodes is not None else (EPISODES_FAST if FAST_MODE else EPISODES_DEFAULT)
    seed = seed if seed is not None else SEED
    cache = cache if cache is not None else CacheManager(CACHE_DIR)

    config = {
        "name": "core_D_ablation_vectorized",
        "configs": _CORE_D_CONFIGS,
        "n_runs": n_runs,
        "n_episodes": n_episodes,
        "seed": seed,
        "fault_type": FAULT_TYPE,
        "vectorized_path_v14_4": True,
        "topology_fp": topology_fp("8bus"),
    }
    cached = cache.get(config)
    if cached is not None:
        return cached

    results: Dict[str, List[Optional[Dict]]] = {name: [None] * n_runs for name in _CORE_D_CONFIGS}
    for cfg_name in _CORE_D_CONFIGS:
        for run_idx in range(n_runs):
            run_seed = seed + run_idx + _method_seed_offset(cfg_name, 1000)
            of_val, cti_val = _run_ablation_single_vectorized(cfg_name, run_seed, n_episodes)
            results[cfg_name][run_idx] = {"OF": of_val, "CTI": cti_val}

    cache.put(config, results)
    return results


# ═════════════════════════════════════════════════════════════════
# Core E (Phase V14.5) -- Hyperparameter grid (lambda_ewc x alpha_lambda x hidden_dim)
# ═════════════════════════════════════════════════════════════════
_CORE_E_LAMBDA_EWC_GRID = [1e2, 1e3, 1e4, 1e5, 1e6]
_CORE_E_ALPHA_LAMBDA_GRID = [1e-5, 1e-4, 1e-3, 1e-2]
_CORE_E_HIDDEN_DIM_GRID = [16, 32, 64, 128]


def _run_hparam_single_vectorized(lambda_ewc: float, alpha_lambda: float, hidden_dim: int,
                                   seed: int, n_steps: int) -> "tuple[float, float]":
    """Vectorized-path counterpart of run_core_training.py's
    _run_hparam_single() (Core E). use_comm=False/use_pid=False are
    REQUIRED here, not incidental -- the real function's own comment
    (run_core_training.py:2429-2432) explains AugmentedLagrangian's PID
    update branch never reads lambda_lr at all, so use_pid=True would
    silently void alpha_lambda's entire sweep. hidden_dim sweeps BOTH
    hidden_actor and hidden_critic to the same value (confirmed from the
    real call, not just hidden_actor). lambda_ewc only has a real effect
    across a task transition -- replicates the real function's
    warm-start-then-switch-with-EWC-correction structure exactly (train
    on 0% PV for warm_steps, register+finalize a single "warmstart" EWC
    task, apply one gradient correction, then train on 60% PV for the
    remainder), reusing get_fisher_diagonal_from_buffer() (Phase V13's
    real per-sample estimator) instead of the real function's manual
    finite-difference Fisher estimate -- same established decision as
    every other vectorized Core. Eval is a fixed 30 episodes tracking
    the real per-step info["OF"] minimum and the SUM (not a rate or a
    final-state count) of info["safe_rl"]["cti_violations"] across each
    episode -- a third distinct CTI-metric convention alongside Core
    A/B/C's rate and Core D's final-state count, faithfully reproduced
    as-is."""
    net = get_8bus()
    env = VectorizedDynamicDOCREnv(net, n_envs=1, seed=seed, fault_type=FAULT_TYPE, curriculum_stage=0)
    safe_env = create_vectorized_safe_rl_env(env, SafeRLConfig(
        project_actions=True, enforce_cti=True,
        lambda_init=1.0, lambda_lr=alpha_lambda, rho_penalty=10.0,
        use_pid=False, adaptive_rho=True,
    ))
    trainer = VectorizedMAPPOTrainer(safe_env, use_comm=False,
                                      hidden_actor=hidden_dim, hidden_critic=hidden_dim, seed=seed)
    ewc_trainer = create_ewc_trainer(trainer, EWCConfig(lambda_ewc=lambda_ewc))
    rng = np.random.default_rng(seed + 9999)

    warm_steps = max(n_steps // 4, 300)
    env.set_curriculum_stage(0)
    trainer.train(total_steps=warm_steps, verbose=False)

    cur = trainer.actor.get_all_params()
    fisher = trainer.get_fisher_diagonal_from_buffer()
    ewc_trainer.ewc_regularizer.register_task("warmstart", cur)
    ewc_trainer.ewc_regularizer.update_fisher("warmstart", fisher)
    ewc_trainer.ewc_regularizer.finalize()
    ewc_grad = ewc_trainer.apply_ewc_to_gradient(cur, np.zeros_like(cur))
    trainer.actor.set_all_params(cur - 0.01 * ewc_grad)

    env.set_curriculum_stage(3)
    trainer.train(total_steps=max(n_steps - warm_steps, 300), verbose=False)

    of_values, cti_values = [], []
    for ep in range(30):
        obs = safe_env.reset(seed=seed + 999 + ep)[0]
        done = False
        ep_of = float("inf")
        ep_cti = 0
        while not done:
            actions = np.zeros((env.n_agents, env.ACT_DIM))
            for i in range(env.n_agents):
                a, _ = trainer.actor.sample(obs[i], rng)
                actions[i] = a
            next_obs, _, done, info = safe_env.step(actions[None, :, :])
            obs = next_obs[0]
            ep_of = min(ep_of, float(np.asarray(info.get("OF", float("inf"))).reshape(-1)[0]))
            safe_rl_info = info.get("safe_rl", {})
            ep_cti += int(np.asarray(safe_rl_info.get("cti_violations", 0)).reshape(-1)[0])
        of_values.append(ep_of)
        cti_values.append(ep_cti)

    return float(np.mean(of_values)), float(np.mean(cti_values))


def run_core_E_vectorized(n_seeds: Optional[int] = None, n_steps: Optional[int] = None,
                           seed: Optional[int] = None, cache: Optional[CacheManager] = None,
                           lambda_ewc_grid: Optional[List[float]] = None,
                           alpha_lambda_grid: Optional[List[float]] = None,
                           hidden_dim_grid: Optional[List[int]] = None) -> List[Dict]:
    """Vectorized-path counterpart of run_core_training.py's run_core_E()
    (80-point hyperparameter grid: 5 lambda_ewc x 4 alpha_lambda x 4
    hidden_dim by default). Writes to a cache entry under a DIFFERENT
    name ("core_E_hparam_vectorized") so it never collides with the real
    Core E cache entry -- see module docstring.

    lambda_ewc_grid/alpha_lambda_grid/hidden_dim_grid default to the
    real 80-point grid; tests override them down to a tiny grid purely
    for runtime -- each grid point is a full two-phase training run plus
    a fixed 30-episode eval loop, so the real 80-point x n_seeds grid is
    far too slow for a schema/correctness smoke test -- same
    test-ergonomics pattern as Core B's n_eval_episodes override."""
    n_seeds = n_seeds if n_seeds is not None else (N_RUNS_HPARAM_FAST if FAST_MODE else N_RUNS_HPARAM)
    n_steps = n_steps if n_steps is not None else (EPISODES_HPARAM_FAST if FAST_MODE else EPISODES_HPARAM)
    seed = seed if seed is not None else SEED
    cache = cache if cache is not None else CacheManager(CACHE_DIR)
    lambda_ewc_grid = lambda_ewc_grid if lambda_ewc_grid is not None else _CORE_E_LAMBDA_EWC_GRID
    alpha_lambda_grid = alpha_lambda_grid if alpha_lambda_grid is not None else _CORE_E_ALPHA_LAMBDA_GRID
    hidden_dim_grid = hidden_dim_grid if hidden_dim_grid is not None else _CORE_E_HIDDEN_DIM_GRID

    config = {
        "name": "core_E_hparam_vectorized",
        "lambda_ewc": lambda_ewc_grid,
        "alpha_lambda": alpha_lambda_grid,
        "hidden_dim": hidden_dim_grid,
        "n_seeds": n_seeds,
        "n_steps": n_steps,
        "seed": seed,
        "fault_type": FAULT_TYPE,
        "vectorized_path_v14_5": True,
        "topology_fp": topology_fp("8bus"),
    }
    cached = cache.get(config)
    if cached is not None:
        return cached

    grid = list(itertools.product(lambda_ewc_grid, alpha_lambda_grid, hidden_dim_grid))
    results_grid: List[List[Optional[Dict]]] = [[None] * len(grid) for _ in range(n_seeds)]
    for seed_idx in range(n_seeds):
        for idx, (lamb, alpha, hid) in enumerate(grid):
            run_seed = seed + seed_idx
            of_val, cti_val = _run_hparam_single_vectorized(lamb, alpha, hid, run_seed, n_steps)
            results_grid[seed_idx][idx] = {
                "lambda_ewc": lamb, "alpha_lambda": alpha, "hidden_dim": hid,
                "seed": seed_idx, "OF": of_val, "CTI": cti_val,
            }

    results = [r for row in results_grid for r in row]
    cache.put(config, results)
    return results


# ═════════════════════════════════════════════════════════════════
# Core F (Phase V14.6) -- Scalability analysis
# ═════════════════════════════════════════════════════════════════
def _run_scalability_single_vectorized(sys_name: str, seed: int) -> Dict:
    """Vectorized-path counterpart of run_core_training.py's
    _run_scalability_single() (Core F). The simplest recipe of any Core
    in this plan -- no Safe RL, no EWC, no curriculum: use_comm=True,
    curriculum_stage=3 fixed, one trainer.train(total_steps=100*300)
    call (the real function's own comment: this Core measures
    training-time SCALING across network sizes, not final performance,
    so the budget is deliberately short and NOT parametrized by
    n_episodes the way every other Core's training call is), wall-clock
    timed. Eval is a fixed 10 episodes tracking the real per-step
    info["OF"] minimum (same convention as Core D/E)."""
    net = load_network(sys_name)
    n_relays = net.n_relays
    env = VectorizedDynamicDOCREnv(net, n_envs=1, seed=seed, fault_type=FAULT_TYPE, curriculum_stage=3)
    trainer = VectorizedMAPPOTrainer(env, use_comm=True, seed=seed)
    rng = np.random.default_rng(seed)

    t0 = time.time()
    trainer.train(total_steps=100 * env.max_steps, verbose=False)
    train_time = time.time() - t0

    of_values = []
    for ep in range(10):
        obs = env.reset(seed=seed + 999 + ep)[0]
        done = False
        ep_of = float("inf")
        while not done:
            if trainer.use_comm and trainer.comm_graph is not None:
                enhanced_obs, _, _ = trainer.comm_graph.forward(obs, rng)
            else:
                enhanced_obs = obs
            actions = np.zeros((env.n_agents, env.ACT_DIM))
            for i in range(env.n_agents):
                a, _ = trainer.actor.sample(enhanced_obs[i], rng)
                actions[i] = a
            next_obs, _, done, info = env.step(actions[None, :, :])
            obs = next_obs[0]
            ep_of = min(ep_of, float(np.asarray(info.get("OF", float("inf"))).reshape(-1)[0]))
        of_values.append(ep_of)

    return {
        "n_relays": n_relays,
        "train_time": train_time,
        "final_performance": float(np.mean(of_values)),
    }


#: Real finding, confirmed empirically (not assumed): VectorizedPowerFlow-
#: Solver (Phase V-PF) requires real per-branch impedance data
#: (NetworkData.line_data["lines"]) and only 8bus/14bus/39bus have it --
#: 3bus/30bus/57bus/118bus raise a ValueError on construction. This is a
#: pre-existing Phase V-PF scope boundary (not a bug introduced here, and
#: not fixable within V14's "pipeline integration" scope -- it would mean
#: extending the power-flow solver itself to systems with no real
#: impedance data at all). Real Core F's own system lists (both the
#: fast-mode default ["8bus","14bus","30bus"] and the full 7-system
#: suite) each include at least one unsupported system, so NEITHER of
#: the real function's default system lists can run end-to-end on this
#: vectorized path as-is -- the default here is narrowed to exactly the
#: 3 genuinely-supported systems instead of silently reusing either real
#: default and crashing.
_CORE_F_SUPPORTED_SYSTEMS = ["8bus", "14bus", "39bus"]


def run_core_F_vectorized(systems: Optional[List[str]] = None, seed: Optional[int] = None,
                           cache: Optional[CacheManager] = None) -> Dict:
    """Vectorized-path counterpart of run_core_training.py's run_core_F()
    (scalability analysis across network sizes). Writes to a cache entry
    under a DIFFERENT name ("core_F_scalability_vectorized") so it never
    collides with the real Core F cache entry -- see module docstring.

    `systems` defaults to _CORE_F_SUPPORTED_SYSTEMS (8bus/14bus/39bus),
    NOT the real function's fast/full system lists -- see that
    constant's comment for why neither real default can run on this
    vectorized path. Passing an unsupported system raises the same
    ValueError VectorizedPowerFlowSolver already raises
    ("...requires real per-branch impedance data..."), not a silent
    fallback -- this is a genuine capability gap for Core F's
    full-scalability-sweep claim, not something to paper over."""
    seed = seed if seed is not None else SEED
    cache = cache if cache is not None else CacheManager(CACHE_DIR)
    if systems is None:
        systems = list(_CORE_F_SUPPORTED_SYSTEMS)

    config = {
        "name": "core_F_scalability_vectorized",
        "systems": systems, "seed": seed, "fault_type": FAULT_TYPE,
        "vectorized_path_v14_6": True,
        "topology_fp": topology_fp(systems),
    }
    cached = cache.get(config)
    if cached is not None:
        return cached

    results: Dict[str, Dict] = {}
    for sys_name in systems:
        results[sys_name] = _run_scalability_single_vectorized(sys_name, seed)

    cache.put(config, results)
    return results


#: ═══════════════════════════════════════════════════════════════════════
#: Phase V14.7 — Cores G-K (CoreEvalBridge-based figure evaluations)
#: ═══════════════════════════════════════════════════════════════════════
#: Real finding, confirmed by reading core_eval_bridge.py and
#: experiments/training_pipeline.py directly (not assumed): EVERY real
#: Core G-K evaluation is a single-env (n_envs=1), SEQUENTIAL rollout
#: against a FROZEN, reconstructed actor -- none of the 13 _eval_*
#: methods below train further; all are pure inference. This is the
#: exact same shape every eval loop elsewhere in this module already
#: uses. Unlike Cores A-F, the real _eval_* methods (training_pipeline.py)
#: hardcode `from marl.marl_env import DynamicDOCREnv` internally, so a
#: genuinely vectorized-path counterpart needs its own small rewrite per
#: method swapping in VectorizedDynamicDOCREnv(n_envs=1) -- EXCEPT two
#: methods (_eval_fault_contributions, _eval_relay_curves) which never
#: touch any env class at all (pure net/result-only math) and are reused
#: completely unmodified below, no vectorized counterpart needed.
#:
#: TrainingPipeline._get_actor_for_eval(env, result, net) is reused
#: UNMODIFIED throughout (not reimplemented): confirmed VectorizedDynamicDOCREnv
#: exposes the exact same OBS_DIM/ACT_DIM/net/n_agents attributes
#: _get_actor_for_eval expects (marl/vectorized_marl_env.py:124-141), and
#: it reconstructs SharedGaussianActor/CommGraph -- the SAME classes
#: VectorizedMAPPOTrainer uses -- from flat parameter vectors, so nothing
#: about the reconstruction itself is single-env-specific.
def _get_eval_pipeline() -> TrainingPipeline:
    """One shared TrainingPipeline instance purely for its _get_actor_for_eval/
    _apply_seasonal_temperature/_eval_fault_contributions/_eval_relay_curves
    methods (all four are stateless w.r.t. the dummy config list passed at
    construction) -- mirrors core_eval_bridge.py's CoreEvalBridge._get_pipeline()
    exactly, including its (pre-existing, unmodified) outputs/checkpoints
    and outputs/results directory side effect."""
    dummy_configs = [TrainingConfig(method="MAPPO-Safe-EWC", system="8bus", pv_level=0.6, fault_type=FAULT_TYPE)]
    return TrainingPipeline(dummy_configs, n_workers=1, verbose=False)


def _train_base_policy_vectorized(system: str = "8bus", pv_level: float = 0.6,
                                   fault_type: Optional[str] = None, seed: Optional[int] = None,
                                   fast_mode: Optional[bool] = None) -> TrainingResult:
    """Vectorized-path counterpart of core_eval_bridge.py's
    CoreEvalBridge._train_base_policy() (core_eval_bridge.py:168-388) --
    the shared base-policy recipe every Core G-K evaluation builds on.
    Structurally identical to Core A's recipe (both trace back to the
    same MAPPO-Safe-EWC recipe): 8bus default, use_comm=True, the same
    literal SafeRLConfig, EWCConfig() defaults. curriculum_stage is
    derived from pv_level via the real bridge's exact
    _PV_LEVELS.index(...) mapping (letting the env's own PV machinery do
    the scaling once, not double-scaling via apply_pv_penetration() --
    see the real bridge's own bug-fix comment on this exact point).

    Returns a real experiments.training_pipeline.TrainingResult so every
    _eval_*_vectorized function below can pass it straight into
    TrainingPipeline._get_actor_for_eval() unmodified.

    EWC uses the real per-sample Fisher estimator from the trainer's own
    rollout buffer (Phase V13's get_fisher_diagonal_from_buffer) rather
    than the real bridge's manual finite-difference loop -- same
    established "use the better real estimator" decision as every
    Core A-F before this."""
    fault_type = fault_type if fault_type is not None else FAULT_TYPE
    fast_mode = fast_mode if fast_mode is not None else FAST_MODE
    seed = seed if seed is not None else SEED

    net = load_network(system)
    _pv_levels = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    curriculum_stage = _pv_levels.index(pv_level) if pv_level in _pv_levels else 5

    env = VectorizedDynamicDOCREnv(net, n_envs=1, seed=seed, fault_type=fault_type,
                                    curriculum_stage=curriculum_stage)
    safe_env = create_vectorized_safe_rl_env(env, SafeRLConfig(
        project_actions=True, enforce_cti=True,
        lambda_init=1.0, lambda_lr=0.01, rho_penalty=10.0,
        use_pid=True, adaptive_rho=True,
    ))
    trainer = VectorizedMAPPOTrainer(safe_env, use_comm=True, seed=seed)

    total_steps = 5000 if fast_mode else 50000
    train_out = trainer.train(total_steps=total_steps, verbose=False)
    history = list(train_out.get("history", [])) if train_out else []

    ewc_trainer = create_ewc_trainer(trainer, EWCConfig())
    task_name = f"{system}_{int(pv_level * 100)}pv"
    cur_params = trainer.actor.get_all_params()
    fisher = trainer.get_fisher_diagonal_from_buffer()
    ewc_trainer.ewc_regularizer.register_task(task_name, cur_params)
    ewc_trainer.ewc_regularizer.update_fisher(task_name, fisher)
    ewc_trainer.ewc_regularizer.finalize()

    # Post-training eval, n_envs=1 sequential (same convention as every
    # other eval loop in this module) -- find the best observed (TMS, PS),
    # mirroring the real bridge's eval loop exactly.
    rng = np.random.default_rng(seed + 999)
    n_eval_episodes = 20 if fast_mode else 100
    best_of = float("inf")
    best_tms = best_ps = None
    for ep in range(n_eval_episodes):
        _ep_reward, TMS, PS = _run_eval_episode(safe_env, trainer, rng, seed=seed + 10_000 + ep)
        try:
            of_val, _, _ = objective_function(net, TMS, PS, fault_type=fault_type)
            if of_val < best_of:
                best_of = float(of_val)
                best_tms, best_ps = TMS.copy(), PS.copy()
            history.append(float(of_val))
        except Exception:
            pass

    if best_tms is None:
        TMS, PS = safe_env.get_state()
        best_tms, best_ps = TMS[0].copy(), PS[0].copy()
        best_of = float(objective_function(net, best_tms, best_ps, fault_type=fault_type)[0])

    constraints = check_constraints(net, best_tms, best_ps, fault_type=fault_type)
    actor_params = trainer.actor.get_all_params()
    comm_graph_params = trainer.comm_graph.get_all_params() if trainer.comm_graph is not None else None
    comm_msg_dim = getattr(trainer.comm_graph, "msg_dim", 8) if trainer.comm_graph is not None else 8

    return TrainingResult(
        config=TrainingConfig(method="MAPPO-Safe-EWC", system=system, pv_level=pv_level, fault_type=fault_type),
        run_id=0,
        best_of=best_of, best_tms=best_tms, best_ps=best_ps,
        cti_satisfied=constraints["cti_satisfied"],
        cti_violations=constraints["cti_violations"],
        cti_min=constraints["cti_min"],
        elapsed_seconds=0.0, history=history,
        actor_params=actor_params,
        obs_dim=env.OBS_DIM, act_dim=env.ACT_DIM, actor_hidden=64,
        use_comm=True, comm_graph_params=comm_graph_params,
        comm_n_agents=env.n_agents, comm_msg_dim=comm_msg_dim,
    )


# ── Core G: Robustness evaluation ────────────────────────────────────────

def _eval_comm_stress_vectorized(net, result: TrainingResult, packet_loss_rate: float = 0.0,
                                  latency_ms: float = 0.0, quantization_bits: int = 32,
                                  fault_type: Optional[str] = None, seed: int = 0) -> Dict:
    """Vectorized counterpart of training_pipeline.py's _eval_comm_stress
    (Fig 13). Reconstructs the real trained CommGraph via
    _get_actor_for_eval, perturbs its packet_loss_rate/latency_steps/
    quantization_bits in place (same attributes Phase V11 already
    verified), then rolls out 300 steps on a fresh
    VectorizedDynamicDOCREnv(n_envs=1)."""
    fault_type = fault_type if fault_type is not None else FAULT_TYPE
    env = VectorizedDynamicDOCREnv(net, n_envs=1, seed=seed + 999, fault_type=fault_type, curriculum_stage=3)
    pipeline = _get_eval_pipeline()
    act_fn, _mean_fn, comm_graph, _actor = pipeline._get_actor_for_eval(env, result, net=net)

    if packet_loss_rate > 0:
        comm_graph.packet_loss_rate = packet_loss_rate
    if latency_ms > 0:
        comm_graph.latency_steps = max(1, int(latency_ms / 100))
    if quantization_bits < 32:
        comm_graph.quantization_bits = quantization_bits

    rng = np.random.default_rng(seed + 999)
    obs = env.reset(seed=seed + 999)[0]
    done = False
    steps = 0
    while not done and steps < 300:
        actions = np.zeros((env.n_agents, env.ACT_DIM))
        for i in range(env.n_agents):
            a, _ = act_fn(obs, i, rng)
            actions[i] = a
        next_obs, _, done, _info = env.step(actions[None, :, :])
        obs = next_obs[0]
        steps += 1

    TMS, PS = env.get_state()
    TMS, PS = TMS[0], PS[0]
    of_stress, _, _ = objective_function(net, TMS, PS, fault_type=fault_type)
    constraints = check_constraints(net, TMS, PS, fault_type=fault_type)
    return {"packet_loss": float(of_stress), "cti_ok": constraints["cti_satisfied"], "latency": latency_ms}


def _eval_n1_contingency_vectorized(net, result: TrainingResult, line_idx: int = 0,
                                     recovery: bool = True, fault_type: Optional[str] = None,
                                     seed: int = 0) -> Dict:
    """Vectorized counterpart of training_pipeline.py's _eval_n1_contingency
    (Fig 14). net.apply_n1_outage() is trainer-agnostic (NetworkData
    method, Phase V1's own TopologyManager outage-factor physics) --
    reused unmodified. Seeds the policy from result.best_tms/best_ps post-
    reset() (reset() unconditionally re-randomizes TMS/PS, so set_state()
    must come after), then lets it actually act on the outaged topology."""
    fault_type = fault_type if fault_type is not None else FAULT_TYPE
    out = {"n1_of": None, "n1_cti_ok": None, "n1_recovery_of": None, "n1_line_index": line_idx}
    net_n1, outage_factor = net.apply_n1_outage(line_idx)
    env_n1 = VectorizedDynamicDOCREnv(net_n1, n_envs=1, seed=seed + 500, fault_type=fault_type, curriculum_stage=3)
    pipeline = _get_eval_pipeline()
    act_fn, _mean_fn, _comm_graph, _actor = pipeline._get_actor_for_eval(env_n1, result, net=net_n1)
    rng = np.random.default_rng(seed + 500)

    env_n1.reset(seed=seed + 500)
    env_n1.set_state(result.best_tms.copy()[None, :], result.best_ps.copy()[None, :])
    obs = env_n1._get_obs()[0]
    done = False
    steps = 0
    while not done and steps < env_n1.max_steps:
        actions = np.zeros((env_n1.n_agents, env_n1.ACT_DIM))
        for i in range(env_n1.n_agents):
            a, _ = act_fn(obs, i, rng)
            actions[i] = a
        next_obs, _, done, _info = env_n1.step(actions[None, :, :])
        obs = next_obs[0]
        steps += 1

    TMS, PS = env_n1.get_state()
    TMS, PS = TMS[0], PS[0]
    of_n1, _, _ = objective_function(net_n1, TMS, PS, fault_type=fault_type)
    constraints_n1 = check_constraints(net_n1, TMS, PS, fault_type=fault_type)
    out["n1_of"] = float(of_n1)
    out["n1_cti_ok"] = constraints_n1["cti_satisfied"]
    out["n1_outage_factor"] = outage_factor

    if recovery:
        env_n1.reset(seed=seed + 600)
        env_n1.set_state(TMS[None, :].copy(), PS[None, :].copy())
        obs = env_n1._get_obs()[0]
        done, steps = False, 0
        recovery_steps = 100
        while not done and steps < recovery_steps:
            actions = np.zeros((env_n1.n_agents, env_n1.ACT_DIM))
            for i in range(env_n1.n_agents):
                a, _ = act_fn(obs, i, rng)
                actions[i] = a
            next_obs, _, done, _info = env_n1.step(actions[None, :, :])
            obs = next_obs[0]
            steps += 1
        TMS_rec, PS_rec = env_n1.get_state()
        of_rec, _, _ = objective_function(net_n1, TMS_rec[0], PS_rec[0], fault_type=fault_type)
        out["n1_recovery_of"] = float(of_rec)

    return out


def _eval_byzantine_vectorized(net, result: TrainingResult, byzantine_agents: List[int],
                                strategy: str = "random", fault_type: Optional[str] = None,
                                seed: int = 0) -> Dict:
    """Vectorized counterpart of training_pipeline.py's _eval_byzantine
    (Fig 20). Honest agents act first via act_fn so "reverse" Byzantine
    agents invert a real honest action, matching the real method's fixed
    ordering bug-fix exactly. Pure action-array post-processing after a
    normal rollout step -- no physics change at all."""
    fault_type = fault_type if fault_type is not None else FAULT_TYPE
    env = VectorizedDynamicDOCREnv(net, n_envs=1, seed=seed + 700, fault_type=fault_type, curriculum_stage=3)
    pipeline = _get_eval_pipeline()
    act_fn, _mean_fn, _comm_graph, _actor = pipeline._get_actor_for_eval(env, result, net=net)
    rng = np.random.default_rng(seed + 700)

    obs = env.reset(seed=seed + 700)[0]
    trust_scores = np.ones(env.n_agents)
    done = False
    steps = 0
    while not done and steps < 300:
        actions = np.zeros((env.n_agents, env.ACT_DIM))
        for i in range(env.n_agents):
            if i not in byzantine_agents:
                a, _ = act_fn(obs, i, rng)
                actions[i] = a
        for i in byzantine_agents:
            if strategy == "random":
                actions[i] = rng.uniform(-1, 1, env.ACT_DIM)
            elif strategy == "reverse":
                honest_a, _ = act_fn(obs, i, rng)
                actions[i] = -honest_a
            elif strategy in ("colluding", "collusion"):
                actions[i] = np.ones(env.ACT_DIM) * 0.9
            elif strategy == "fdi":
                actions[i] = np.full(env.ACT_DIM, 0.3)
            else:
                raise ValueError(f"Unknown byzantine_strategy {strategy!r}")

        next_obs, _reward, done, info = env.step(actions[None, :, :])
        obs = next_obs[0]

        per_agent_reward = info.get("per_agent_reward") if isinstance(info, dict) else None
        if per_agent_reward is not None:
            par = np.asarray(per_agent_reward).reshape(env.n_envs, env.n_agents)[0]
            for i in range(env.n_agents):
                if par[i] < -0.5:
                    trust_scores[i] *= 0.9
        steps += 1

    TMS, PS = env.get_state()
    TMS, PS = TMS[0], PS[0]
    of_byz, _, _ = objective_function(net, TMS, PS, fault_type=fault_type)
    return {
        "byzantine_of": float(of_byz),
        "byzantine_trust_scores": trust_scores,
        "byzantine_detected": bool(np.any(trust_scores[byzantine_agents] < 0.5)) if byzantine_agents else False,
    }


def _eval_bandwidth_vectorized(net, result: TrainingResult, compression_ratio: float = 1.0,
                                fault_type: Optional[str] = None, seed: int = 0) -> Dict:
    """Vectorized counterpart of training_pipeline.py's _eval_bandwidth
    (Fig 22). gnn_depth is accepted by the real config but never actually
    consumed inside the real method's body -- confirmed by reading it
    directly -- so it is correctly omitted here too, not a gap. Degrades
    the message the policy receives by zeroing a drop_fraction of the
    enhanced-obs vector each step, same technique the real method uses."""
    fault_type = fault_type if fault_type is not None else FAULT_TYPE
    env = VectorizedDynamicDOCREnv(net, n_envs=1, seed=seed + 800, fault_type=fault_type, curriculum_stage=3)
    pipeline = _get_eval_pipeline()
    act_fn, _mean_fn, comm_graph, _actor = pipeline._get_actor_for_eval(env, result, net=net)
    rng = np.random.default_rng(seed + 800)

    msg_dim = result.comm_msg_dim
    bytes_per_agent_per_step = msg_dim * 4 * compression_ratio
    kbps = bytes_per_agent_per_step * 8 / 1000.0
    total_params = comm_graph.n_params() if hasattr(comm_graph, "n_params") else 0
    compressed_params = int(total_params * compression_ratio)

    original_forward = comm_graph.forward
    drop_fraction = max(0.0, 1.0 - compression_ratio)

    def degraded_forward(o, rng_):
        enhanced_obs, messages, info = original_forward(o, rng_)
        if drop_fraction > 0:
            mask = rng_.random(enhanced_obs.shape) < drop_fraction
            enhanced_obs = enhanced_obs.copy()
            enhanced_obs[mask] = 0.0
        return enhanced_obs, messages, info

    comm_graph.forward = degraded_forward
    try:
        obs = env.reset(seed=seed + 800)[0]
        done, steps = False, 0
        while not done and steps < 300:
            actions = np.zeros((env.n_agents, env.ACT_DIM))
            for i in range(env.n_agents):
                a, _ = act_fn(obs, i, rng)
                actions[i] = a
            next_obs, _, done, _info = env.step(actions[None, :, :])
            obs = next_obs[0]
            steps += 1
    finally:
        comm_graph.forward = original_forward

    TMS, PS = env.get_state()
    of_compressed, _, _ = objective_function(net, TMS[0], PS[0], fault_type=fault_type)
    return {"bandwidth": float(kbps), "of": float(of_compressed), "params": compressed_params}


def _eval_sensor_robustness_vectorized(net, result: TrainingResult, noise_std: float = 0.0,
                                        bias: float = 0.0, missing_rate: float = 0.0,
                                        fault_type: Optional[str] = None, seed: int = 0) -> Dict:
    """Vectorized counterpart of training_pipeline.py's
    _eval_sensor_robustness (Fig 24). Corrupts EVERY agent's raw
    observation before the comm-graph forward pass each step (not just
    the subject agent's own slice), matching the real method's exact
    rationale: a noisy agent's corrupted sensor reading would otherwise
    flow cleanly to neighbors through an uncorrupted GNN message."""
    fault_type = fault_type if fault_type is not None else FAULT_TYPE
    env = VectorizedDynamicDOCREnv(net, n_envs=1, seed=seed + 1000, fault_type=fault_type, curriculum_stage=3)
    pipeline = _get_eval_pipeline()
    act_fn, _mean_fn, _comm_graph, _actor = pipeline._get_actor_for_eval(env, result, net=net)
    rng = np.random.default_rng(seed + 1000)

    def _corrupt(o):
        o = o.copy()
        if noise_std > 0:
            o = o + rng.normal(0, noise_std, o.shape)
        if bias != 0:
            o = o + bias
        if missing_rate > 0:
            mask = rng.random(o.shape) < missing_rate
            o[mask] = 0.0
        return o

    obs = env.reset(seed=seed + 1000)[0]
    done, steps = False, 0
    while not done and steps < 300:
        noisy_obs = np.array([_corrupt(obs[i]) for i in range(env.n_agents)])
        actions = np.zeros((env.n_agents, env.ACT_DIM))
        for i in range(env.n_agents):
            a, _ = act_fn(noisy_obs, i, rng)
            actions[i] = a
        next_obs, _, done, _info = env.step(actions[None, :, :])
        obs = next_obs[0]
        steps += 1

    TMS, PS = env.get_state()
    of_noisy, _, _ = objective_function(net, TMS[0], PS[0], fault_type=fault_type)
    constraints = check_constraints(net, TMS[0], PS[0], fault_type=fault_type)
    signal_power = float(np.mean(obs ** 2))
    snr = float(10 * np.log10(signal_power / (noise_std ** 2 + 1e-10)))
    return {"noisy_of": float(of_noisy), "noisy_cti_ok": constraints["cti_satisfied"], "sensor_snr": snr}


def _eval_adversarial_vectorized(net, result: TrainingResult, attack: str = "FGSM",
                                  epsilon: float = 0.01, fault_type: Optional[str] = None,
                                  seed: int = 0) -> Dict:
    """Vectorized counterpart of training_pipeline.py's _eval_adversarial
    (Fig 54). A real finite-difference gradient attack, not vectorized-
    physics work at all -- purely actor-forward-pass math (mean_fn), so
    it works identically whether the actor came from MAPPOMARLTrainer or
    VectorizedMAPPOTrainer (same SharedGaussianActor class)."""
    fault_type = fault_type if fault_type is not None else FAULT_TYPE
    env = VectorizedDynamicDOCREnv(net, n_envs=1, seed=seed + 1700, fault_type=fault_type, curriculum_stage=3)
    pipeline = _get_eval_pipeline()
    act_fn, mean_fn, _comm_graph, _actor = pipeline._get_actor_for_eval(env, result, net=net)
    rng = np.random.default_rng(seed + 1700)

    def _grad_sign(obs_i, h=1e-3):
        base = mean_fn(obs_i)
        base_loss = np.linalg.norm(base)
        grad = np.zeros_like(obs_i)
        for k in range(len(obs_i)):
            bump = obs_i.copy()
            bump[k] += h
            loss_k = np.linalg.norm(mean_fn(bump))
            grad[k] = (loss_k - base_loss) / h
        return np.sign(grad)

    obs = env.reset(seed=seed + 1700)[0]
    n_success, n_total = 0, 10
    perturbation_norms = []

    for _ in range(n_total):
        clean_obs = obs.copy()
        adv_obs = clean_obs.copy()
        for i in range(env.n_agents):
            if attack == "FGSM":
                perturbation_i = epsilon * _grad_sign(clean_obs[i])
            elif attack == "PGD":
                perturbed = clean_obs[i].copy()
                step_size = epsilon / 4.0
                for _pgd_step in range(4):
                    grad_sign = _grad_sign(perturbed)
                    perturbed = perturbed + step_size * grad_sign
                    delta = np.clip(perturbed - clean_obs[i], -epsilon, epsilon)
                    perturbed = clean_obs[i] + delta
                perturbation_i = perturbed - clean_obs[i]
            else:
                perturbation_i = epsilon * rng.standard_normal(clean_obs[i].shape)
            adv_obs[i] = clean_obs[i] + perturbation_i
            perturbation_norms.append(float(np.linalg.norm(perturbation_i)))

        actions = np.zeros((env.n_agents, env.ACT_DIM))
        for i in range(env.n_agents):
            a, _ = act_fn(adv_obs, i, rng)
            actions[i] = a
        next_obs, _, done, _info = env.step(actions[None, :, :])
        obs = next_obs[0]
        if done:
            obs = env.reset(seed=seed + 1700)[0]

        TMS, PS = env.get_state()
        try:
            of_adv, _, _ = objective_function(net, TMS[0], PS[0], fault_type=fault_type)
            constraints = check_constraints(net, TMS[0], PS[0], fault_type=fault_type)
            if not constraints["cti_satisfied"] or of_adv > result.best_of * 1.5:
                n_success += 1
        except Exception:
            pass

    TMS, PS = env.get_state()
    of_adv, _, _ = objective_function(net, TMS[0], PS[0], fault_type=fault_type)
    return {
        "of": float(of_adv),
        "success": n_success / n_total,
        "norm": float(np.mean(perturbation_norms)) if perturbation_norms else 0.0,
    }


def run_core_G_vectorized(seed: Optional[int] = None, cache: Optional[CacheManager] = None,
                           fast_mode: Optional[bool] = None) -> Dict:
    """Vectorized-path counterpart of run_core_training.py's run_core_G()
    (robustness evaluation). Writes to a DIFFERENT cache name
    ("core_G_robustness_vectorized") -- see module docstring. Unlike the
    real Core G's ~120-point sweep across 3 base policies (8bus/14bus/
    39bus), this is narrowed to ONE base policy (8bus) and a small,
    representative sweep per stress type -- a correctness/schema smoke-test
    budget (same test-ergonomics narrowing precedent as Core F's system
    list), not a real-scale replication. Every individual
    _eval_*_vectorized function above is faithful to the real method's
    exact mechanics regardless of how many sweep points this orchestrator
    requests. `fast_mode` (defaults to the module FAST_MODE) is threaded
    into the base-policy training budget only -- a test-ergonomics
    override, same precedent as Core B's n_eval_episodes/Core E's grid
    overrides, included in the cache config so different budgets never
    collide in the cache."""
    seed = seed if seed is not None else SEED
    cache = cache if cache is not None else CacheManager(CACHE_DIR)
    fast_mode = fast_mode if fast_mode is not None else FAST_MODE

    config = {
        "name": "core_G_robustness_vectorized",
        "stress_types": ["communication", "byzantine", "bandwidth", "sensor", "adversarial", "n1_contingency"],
        "seed": seed, "fault_type": FAULT_TYPE, "fast_mode": fast_mode,
        "vectorized_path_v14_7": True,
        "topology_fp": topology_fp("8bus"),
    }
    cached = cache.get(config)
    if cached is not None:
        return cached

    result = _train_base_policy_vectorized(system="8bus", seed=seed, fast_mode=fast_mode)
    net = load_network("8bus")
    base_of = result.best_of

    def _rel_perf(of_value):
        return float(base_of / max(of_value, 1e-9))

    results: Dict = {}
    packet_loss_sweep = {pl: _rel_perf(_eval_comm_stress_vectorized(
        net, result, packet_loss_rate=pl, seed=seed)["packet_loss"]) for pl in [0.0, 0.2]}
    latency_sweep = {lat: _rel_perf(_eval_comm_stress_vectorized(
        net, result, latency_ms=lat, seed=seed)["packet_loss"]) for lat in [0, 200]}
    results["communication"] = {"packet_loss": packet_loss_sweep, "latency": latency_sweep}

    results["byzantine"] = {s: _rel_perf(_eval_byzantine_vectorized(
        net, result, byzantine_agents=[1], strategy=s, seed=seed)["byzantine_of"])
        for s in ["random", "fdi"]}

    bandwidth_sweep = {}
    for ratio in [1.0, 0.5]:
        d = _eval_bandwidth_vectorized(net, result, compression_ratio=ratio, seed=seed)
        bandwidth_sweep[ratio] = {"performance": _rel_perf(d["of"]), "bandwidth_kbps": d["bandwidth"]}
    results["bandwidth"] = bandwidth_sweep

    noise_sweep = {sigma: _rel_perf(_eval_sensor_robustness_vectorized(
        net, result, noise_std=sigma, seed=seed)["noisy_of"]) for sigma in [0.0, 0.1]}
    results["sensor"] = {"noise": noise_sweep}

    fgsm_sweep = {eps: _rel_perf(_eval_adversarial_vectorized(
        net, result, attack="FGSM", epsilon=eps, seed=seed)["of"]) for eps in [0.01, 0.1]}
    results["adversarial"] = {"fgsm": fgsm_sweep}

    n1_sweep = {}
    for line_idx in range(min(2, net.n_lines)):
        d = _eval_n1_contingency_vectorized(net, result, line_idx=line_idx, recovery=True, seed=seed)
        n1_sweep[line_idx] = {
            "performance": _rel_perf(d["n1_of"]), "cti_ok": bool(d["n1_cti_ok"]),
            "recovery_performance": _rel_perf(d["n1_recovery_of"]) if d["n1_recovery_of"] is not None else None,
        }
    results["n1_contingency"] = n1_sweep

    cache.put(config, results)
    return results


# ── Core H: Transfer learning ────────────────────────────────────────────

def _eval_unseen_faults_vectorized(net, result: TrainingResult, unseen_fault_type: str,
                                    seed: int = 0) -> Dict:
    """Vectorized counterpart of training_pipeline.py's _eval_unseen_faults
    (Fig 23). The env's real generalization axis is fault TYPE (not
    config.fault_impedance_ohm, which nothing in relay_physics.py
    consumes -- confirmed by reading the real method directly)."""
    env = VectorizedDynamicDOCREnv(net, n_envs=1, seed=seed + 900, fault_type=unseen_fault_type,
                                    curriculum_stage=3)
    pipeline = _get_eval_pipeline()
    act_fn, _mean_fn, _comm_graph, _actor = pipeline._get_actor_for_eval(env, result, net=net)
    rng = np.random.default_rng(seed + 900)

    env.reset(seed=seed + 900, fault_type=unseen_fault_type)
    env.set_state(result.best_tms.copy()[None, :], result.best_ps.copy()[None, :])
    obs = env._get_obs()[0]
    done, steps = False, 0
    while not done and steps < env.max_steps:
        actions = np.zeros((env.n_agents, env.ACT_DIM))
        for i in range(env.n_agents):
            a, _ = act_fn(obs, i, rng)
            actions[i] = a
        next_obs, _, done, _info = env.step(actions[None, :, :])
        obs = next_obs[0]
        steps += 1

    TMS, PS = env.get_state()
    of_unseen, _, _ = objective_function(net, TMS[0], PS[0], fault_type=unseen_fault_type)
    constraints = check_constraints(net, TMS[0], PS[0], fault_type=unseen_fault_type)
    return {"unseen_of": float(of_unseen), "unseen_fault_cti_ok": constraints["cti_satisfied"]}


def _eval_zero_shot_transfer_vectorized(result: TrainingResult, eval_system: str = "14bus",
                                         pv_level: float = 0.6, fault_type: Optional[str] = None,
                                         seed: int = 0) -> Dict:
    """Vectorized counterpart of training_pipeline.py's
    _eval_zero_shot_transfer (Fig 25). Rebuilds the actor/comm_graph
    against the TARGET system's adjacency (build_relay_adjacency(target_net)
    inside _get_actor_for_eval), then lets the trained policy act from the
    target's own random initial state -- a true zero-shot test, no
    set_state() call."""
    fault_type = fault_type if fault_type is not None else FAULT_TYPE
    target_net = load_network(eval_system)
    if pv_level > 0 and hasattr(target_net, "apply_pv_penetration"):
        target_net = target_net.apply_pv_penetration(pv_level)

    env = VectorizedDynamicDOCREnv(target_net, n_envs=1, seed=seed + 1100, fault_type=fault_type,
                                    curriculum_stage=3)
    pipeline = _get_eval_pipeline()
    act_fn, _mean_fn, _comm_graph, _actor = pipeline._get_actor_for_eval(env, result, net=target_net)
    rng = np.random.default_rng(seed + 1100)

    obs = env.reset(seed=seed + 1100)[0]
    done, steps = False, 0
    while not done and steps < env.max_steps:
        actions = np.zeros((env.n_agents, env.ACT_DIM))
        for i in range(env.n_agents):
            a, _ = act_fn(obs, i, rng)
            actions[i] = a
        next_obs, _, done, _info = env.step(actions[None, :, :])
        obs = next_obs[0]
        steps += 1

    TMS, PS = env.get_state()
    of_transfer, _, _ = objective_function(target_net, TMS[0], PS[0], fault_type=fault_type)
    constraints = check_constraints(target_net, TMS[0], PS[0], fault_type=fault_type)
    return {"transfer_of": float(of_transfer), "transfer_cti_ok": constraints["cti_satisfied"],
            "transfer_system": eval_system}


def run_core_H_vectorized(seed: Optional[int] = None, n_episodes: Optional[int] = None,
                           cache: Optional[CacheManager] = None,
                           fast_mode: Optional[bool] = None) -> Dict:
    """Vectorized-path counterpart of run_core_training.py's run_core_H()
    (transfer learning). Writes to a DIFFERENT cache name
    ("core_H_transfer_vectorized"). The cold-start baseline reuses
    _run_baseline_single_vectorized("MAPPO-Safe-EWC", ..., system="14bus")
    directly -- already built for Core C, zero new work needed (confirmed
    by reading run_core_training.py's own Fig 25 cold-start call site:
    it dispatches to the exact same _run_baseline_single function Core C
    uses). `fast_mode` (see run_core_G_vectorized's docstring) only
    affects the base-policy training budget, not n_episodes (the
    cold-start baseline's own, separate, already-existing override)."""
    from network.relay_physics import FAULT_TYPES
    seed = seed if seed is not None else SEED
    n_episodes = n_episodes if n_episodes is not None else (EPISODES_FAST if FAST_MODE else EPISODES_DEFAULT)
    cache = cache if cache is not None else CacheManager(CACHE_DIR)
    fast_mode = fast_mode if fast_mode is not None else FAST_MODE

    config = {
        "name": "core_H_transfer_vectorized",
        "evaluations": ["unseen_faults", "zero_shot_8to14"],
        "seed": seed, "n_episodes": n_episodes, "fault_type": FAULT_TYPE, "fast_mode": fast_mode,
        "vectorized_path_v14_7": True,
        "topology_fp": topology_fp("8bus", "14bus"),
    }
    cached = cache.get(config)
    if cached is not None:
        return cached

    result = _train_base_policy_vectorized(system="8bus", seed=seed, fast_mode=fast_mode)
    net = load_network("8bus")
    base_of = result.best_of

    unseen_types = [ft for ft in FAULT_TYPES if ft != FAULT_TYPE]
    unseen_faults = {}
    for ft in unseen_types:
        d = _eval_unseen_faults_vectorized(net, result, unseen_fault_type=ft, seed=seed)
        unseen_faults[ft] = float(base_of / max(d["unseen_of"], 1e-9))

    zero_shot = _eval_zero_shot_transfer_vectorized(result, eval_system="14bus", seed=seed)
    zero_shot = {**zero_shot, "base_of": base_of}

    cold_returns, cold_metrics = _run_baseline_single_vectorized(
        "MAPPO-Safe-EWC", seed + 40000, n_episodes, system="14bus")
    if cold_metrics and cold_metrics.get("final_of") is not None:
        zero_shot["cold_start"] = {
            "performance": float(base_of / max(cold_metrics["final_of"], 1e-9)),
            "convergence_ep": cold_metrics["convergence_ep"],
        }

    results = {"unseen_faults": unseen_faults, "zero_shot": zero_shot}
    cache.put(config, results)
    return results


# ── Core I: Deployment benchmark ─────────────────────────────────────────

def _measure_latency_vectorized(net, result: TrainingResult, seed: int = 0) -> Dict:
    """Vectorized counterpart of run_core_training.py's
    _core_I_measure_latency(): real per-decision inference latency, timing
    the reconstructed actor's deterministic mean_fn forward pass (NOT
    env.reset()/RNG overhead). Identical whether the actor came from
    MAPPOMARLTrainer or VectorizedMAPPOTrainer -- same SharedGaussianActor.
    get_mean call -- so no env stepping is involved at all."""
    env = VectorizedDynamicDOCREnv(net, n_envs=1, seed=seed, fault_type=FAULT_TYPE, curriculum_stage=3)
    pipeline = _get_eval_pipeline()
    _act_fn, mean_fn, _comm_graph, _actor = pipeline._get_actor_for_eval(env, result, net=net)

    latencies = []
    obs = env.reset(seed=seed)[0]
    for _ in range(100):
        t0 = time.time()
        for i in range(env.n_agents):
            _ = mean_fn(obs[i])
        latencies.append((time.time() - t0) * 1000)
    latencies_arr = np.array(latencies)
    return {
        "latencies_ms": latencies_arr,
        "mean_ms": float(np.mean(latencies_arr)),
        "p50_ms": float(np.percentile(latencies_arr, 50)),
        "p99_ms": float(np.percentile(latencies_arr, 99)),
    }


def _eval_compression_vectorized(net, result: TrainingResult, quantize_bits: Optional[int] = None,
                                  prune_ratio: float = 0.0, fault_type: Optional[str] = None,
                                  seed: int = 0) -> Dict:
    """Vectorized counterpart of training_pipeline.py's _eval_compression
    (Fig 53). Genuinely quantizes/prunes the reconstructed ACTOR's flat
    parameter vector (not the comm graph's -- Fig 22/_eval_bandwidth_vectorized
    already isolates comm-channel compression), then lets the degraded
    policy act for real."""
    fault_type = fault_type if fault_type is not None else FAULT_TYPE
    base_params = len(result.best_tms) + len(result.best_ps)
    bytes_per_param = (quantize_bits / 8) if quantize_bits else 4.0
    pruned_params = int(base_params * (1 - prune_ratio))
    size_kb = pruned_params * bytes_per_param / 1024.0

    env = VectorizedDynamicDOCREnv(net, n_envs=1, seed=seed + 1600, fault_type=fault_type, curriculum_stage=3)
    pipeline = _get_eval_pipeline()
    act_fn, _mean_fn, _comm_graph, actor = pipeline._get_actor_for_eval(env, result, net=net)

    compressed_params = actor.get_all_params().copy()
    if quantize_bits and quantize_bits < 32:
        levels = 2 ** quantize_bits
        w_min, w_max = compressed_params.min(), compressed_params.max()
        w_range = max(w_max - w_min, 1e-9)
        step = w_range / levels
        compressed_params = np.round((compressed_params - w_min) / step) * step + w_min
    if prune_ratio > 0:
        n_prune = int(len(compressed_params) * prune_ratio)
        if n_prune > 0:
            prune_idx = np.argsort(np.abs(compressed_params))[:n_prune]
            compressed_params[prune_idx] = 0.0
    actor.set_all_params(compressed_params)

    rng = np.random.default_rng(seed + 1600)
    obs = env.reset(seed=seed + 1600)[0]
    done, steps = False, 0
    while not done and steps < 300:
        actions = np.zeros((env.n_agents, env.ACT_DIM))
        for i in range(env.n_agents):
            a, _ = act_fn(obs, i, rng)
            actions[i] = a
        next_obs, _, done, _info = env.step(actions[None, :, :])
        obs = next_obs[0]
        steps += 1

    TMS, PS = env.get_state()
    of_comp, _, _ = objective_function(net, TMS[0], PS[0], fault_type=fault_type)
    return {"of": float(of_comp), "size_kb": float(size_kb), "drop": float(of_comp - result.best_of)}


def run_core_I_vectorized(seed: Optional[int] = None, cache: Optional[CacheManager] = None,
                           fast_mode: Optional[bool] = None) -> Dict:
    """Vectorized-path counterpart of run_core_training.py's run_core_I()
    (deployment benchmark). Writes to a DIFFERENT cache name
    ("core_I_deployment_vectorized"). Trains 2 base policies (8bus, 39bus,
    same systems the real Core I measures), each with its own real
    latency measurement, plus a 3-point compression sweep on the 8bus
    policy. `fast_mode` -- see run_core_G_vectorized's docstring."""
    seed = seed if seed is not None else SEED
    cache = cache if cache is not None else CacheManager(CACHE_DIR)
    fast_mode = fast_mode if fast_mode is not None else FAST_MODE

    config = {
        "name": "core_I_deployment_vectorized",
        "benchmarks": ["inference_latency", "model_compression"],
        "seed": seed, "fault_type": FAULT_TYPE, "fast_mode": fast_mode,
        "vectorized_path_v14_7": True,
        "topology_fp": topology_fp("8bus", "39bus"),
    }
    cached = cache.get(config)
    if cached is not None:
        return cached

    results: Dict = {}
    net_8 = load_network("8bus")
    result_8 = _train_base_policy_vectorized(system="8bus", seed=seed, fast_mode=fast_mode)
    results["latency"] = _measure_latency_vectorized(net_8, result_8, seed=seed)

    net_39 = load_network("39bus")
    result_39 = _train_base_policy_vectorized(system="39bus", seed=seed, fast_mode=fast_mode)
    results["latency_39bus"] = _measure_latency_vectorized(net_39, result_39, seed=seed)

    compression_configs = {
        "original": {"quantize_bits": None, "prune_ratio": 0.0},
        "int8": {"quantize_bits": 8, "prune_ratio": 0.0},
        "pruning": {"quantize_bits": None, "prune_ratio": 0.3},
    }
    base_of = result_8.best_of
    compression = {}
    for method_name, kw in compression_configs.items():
        d = _eval_compression_vectorized(net_8, result_8, seed=seed, **kw)
        compression[method_name] = {
            "size_mb": d["size_kb"] / 1024.0,
            "performance": float(base_of / d["of"]) if d["of"] else None,
        }
    results["compression"] = compression

    cache.put(config, results)
    return results


# ── Core J: Protection engineering ───────────────────────────────────────

def _eval_daily_pv_vectorized(net, result: TrainingResult, fault_type: Optional[str] = None,
                               pv_level: float = 0.6, seed: int = 0) -> Dict:
    """Vectorized counterpart of training_pipeline.py's _eval_daily_pv
    (Fig 45). reset(pv_level=...) is confirmed to exist on
    VectorizedDynamicDOCREnv (marl/vectorized_marl_env.py:198-215) with
    the same kwarg the real method relies on to genuinely re-thread PV
    level into the fault-current/power-flow recompute each hour."""
    fault_type = fault_type if fault_type is not None else FAULT_TYPE
    hours = np.arange(24)
    irradiance = np.maximum(0, np.sin(np.pi * hours / 12) * 1000)

    env = VectorizedDynamicDOCREnv(net, n_envs=1, seed=seed + 1200, fault_type=fault_type, curriculum_stage=3)
    pipeline = _get_eval_pipeline()
    act_fn, _mean_fn, _comm_graph, _actor = pipeline._get_actor_for_eval(env, result, net=net)
    rng = np.random.default_rng(seed + 1200)

    tms_traj, ps_traj, of_traj = [], [], []
    for h in range(24):
        pv_factor = irradiance[h] / 1000.0
        hour_pv_level = float(np.clip(pv_level * pv_factor, 0.0, 1.0))
        obs = env.reset(seed=seed + 1200 + h, pv_level=hour_pv_level)[0]
        done, steps = False, 0
        while not done and steps < 50:
            actions = np.zeros((env.n_agents, env.ACT_DIM))
            for i in range(env.n_agents):
                a, _ = act_fn(obs, i, rng)
                actions[i] = a
            next_obs, _, done, _info = env.step(actions[None, :, :])
            obs = next_obs[0]
            steps += 1

        TMS, PS = env.get_state()
        tms_traj.append(TMS[0].copy())
        ps_traj.append(PS[0].copy())
        try:
            of_h, _, _ = objective_function(net, TMS[0], PS[0], fault_type=fault_type)
            of_traj.append(float(of_h))
        except Exception:
            of_traj.append(float("nan"))

    return {
        "tms": np.array(tms_traj), "ps": np.array(ps_traj),
        "of": np.array(of_traj), "irradiance": irradiance,
    }


def _eval_sympathetic_tripping_vectorized(net, result: TrainingResult, seed: int = 0) -> Dict:
    """Vectorized counterpart of training_pipeline.py's
    _eval_sympathetic_tripping (Fig 47). Uses the real, physically
    meaningful signal (check_constraints()['cti_violations'], a backup
    relay's margin collapsing below the CTI threshold) rather than any
    nonexistent 'backup_trip' info key. The traditional baseline is zero
    action every step (env actions are deltas, not absolute settings) --
    same zero-action pattern already used for Core C's "Traditional" arm."""
    def _count_trips(fault_type: str):
        env_marl = VectorizedDynamicDOCREnv(net, n_envs=1, seed=seed + 1300, fault_type=fault_type,
                                             curriculum_stage=3)
        pipeline = _get_eval_pipeline()
        act_fn, _mean_fn, _comm_graph, _actor = pipeline._get_actor_for_eval(env_marl, result, net=net)
        rng = np.random.default_rng(seed + 1300)

        n_marl = 0
        for ep in range(10):
            obs = env_marl.reset(seed=seed + 1300 + ep)[0]
            done, steps = False, 0
            while not done and steps < 50:
                actions = np.zeros((env_marl.n_agents, env_marl.ACT_DIM))
                for i in range(env_marl.n_agents):
                    a, _ = act_fn(obs, i, rng)
                    actions[i] = a
                next_obs, _, done, _info = env_marl.step(actions[None, :, :])
                obs = next_obs[0]
                steps += 1
            TMS, PS = env_marl.get_state()
            c = check_constraints(net, TMS[0], PS[0], fault_type=fault_type)
            n_marl += c["cti_violations"]

        env_trad = VectorizedDynamicDOCREnv(net, n_envs=1, seed=seed + 1300, fault_type=fault_type,
                                             curriculum_stage=3)
        n_trad = 0
        for ep in range(10):
            env_trad.reset(seed=seed + 1300 + ep)
            env_trad.set_state(np.ones((1, env_trad.n_agents)), np.full((1, env_trad.n_agents), 2.0))
            done, steps = False, 0
            while not done and steps < 50:
                _obs, _, done, _info = env_trad.step(np.zeros((1, env_trad.n_agents, env_trad.ACT_DIM)))
                steps += 1
            TMS, PS = env_trad.get_state()
            c = check_constraints(net, TMS[0], PS[0], fault_type=fault_type)
            n_trad += c["cti_violations"]

        return n_marl, n_trad

    fault_types = ["1ph", "LL", "3ph"]
    by_fault_type = {}
    for ft in fault_types:
        n_marl, n_trad = _count_trips(ft)
        by_fault_type[ft] = {"marl": n_marl, "traditional": n_trad}

    headline = by_fault_type.get(FAULT_TYPE)
    return {
        "sympathetic": headline["marl"] if headline else None,
        "traditional": headline["traditional"] if headline else None,
        "by_fault_type": by_fault_type,
    }


def run_core_J_vectorized(seed: Optional[int] = None, cache: Optional[CacheManager] = None,
                           fast_mode: Optional[bool] = None) -> Dict:
    """Vectorized-path counterpart of run_core_training.py's run_core_J()
    (protection engineering). Writes to a DIFFERENT cache name
    ("core_J_protection_vectorized"). Figs 44 (_eval_fault_contributions)
    and 46 (_eval_relay_curves) are REUSED UNMODIFIED from the real
    TrainingPipeline -- confirmed by reading both directly: neither
    touches any env class at all (44 is a pure pandapower short-circuit
    study on `net`+pv_level; 46 is pure closed-form IDMT math off
    result.best_tms/best_ps), so there is nothing for a vectorized-path
    counterpart to do differently. `fast_mode` -- see
    run_core_G_vectorized's docstring."""
    seed = seed if seed is not None else SEED
    cache = cache if cache is not None else CacheManager(CACHE_DIR)
    fast_mode = fast_mode if fast_mode is not None else FAST_MODE

    config = {
        "name": "core_J_protection_vectorized",
        "analyses": ["coordination", "fault_contribution", "daily_profile", "relay_curves", "sympathetic_tripping"],
        "seed": seed, "fault_type": FAULT_TYPE, "fast_mode": fast_mode,
        "vectorized_path_v14_7": True,
        "topology_fp": topology_fp("8bus"),
    }
    cached = cache.get(config)
    if cached is not None:
        return cached

    result = _train_base_policy_vectorized(system="8bus", seed=seed, fast_mode=fast_mode)
    net = load_network("8bus")
    pipeline = _get_eval_pipeline()

    n_pairs = 20
    distances = np.linspace(0, 10, n_pairs)
    rng_j = np.random.default_rng(seed + 43)
    op_times_primary = 0.1 + 0.05 * distances + rng_j.normal(0, 0.01, n_pairs)
    op_times_backup = 0.3 + 0.08 * distances + rng_j.normal(0, 0.01, n_pairs)

    fig44_config = TrainingConfig(method="MAPPO-Safe-EWC", system="8bus", pv_level=0.6,
                                   fault_type=FAULT_TYPE, run_shortcircuit_study=True)
    result_44 = pipeline._eval_fault_contributions(fig44_config, net, result)

    fig46_config = TrainingConfig(method="MAPPO-Safe-EWC", system="8bus", pv_level=0.6,
                                   fault_type=FAULT_TYPE, relay_curve_eval=True)
    result_46 = pipeline._eval_relay_curves(fig46_config, net, result)

    results = {
        "coordination": {"distances": distances, "primary": op_times_primary, "backup": op_times_backup},
        "fault_contribution": result_44.fault_contributions or {},
        "daily_profile": _eval_daily_pv_vectorized(net, result, seed=seed),
        "relay_curves": result_46.relay_curves or {},
        "sympathetic": _eval_sympathetic_tripping_vectorized(net, result, seed=seed),
    }
    cache.put(config, results)
    return results


# ── Core K: Seasonal generalization ──────────────────────────────────────

def _eval_seasonal_vectorized(net, result: TrainingResult, season: str, pv_level: float = 0.6,
                               fault_type: Optional[str] = None, seed: int = 0) -> Dict:
    """Vectorized counterpart of training_pipeline.py's _eval_seasonal
    (Fig 58). TrainingPipeline._apply_seasonal_temperature is reused
    UNMODIFIED (it operates only on net.pairs' If_pri/If_bkp -- no env
    dependency at all)."""
    fault_type = fault_type if fault_type is not None else FAULT_TYPE
    seasonal_profiles = {
        "spring": {"irradiance_factor": 0.7, "temp": 15.0},
        "summer": {"irradiance_factor": 1.0, "temp": 35.0},
        "autumn": {"irradiance_factor": 0.5, "temp": 10.0},
        "winter": {"irradiance_factor": 0.3, "temp": 0.0},
    }
    profile = seasonal_profiles.get(season, {"irradiance_factor": 1.0, "temp": 20.0})
    pipeline = _get_eval_pipeline()
    net_seasonal = pipeline._apply_seasonal_temperature(net, profile["temp"])
    seasonal_pv_level = float(np.clip(pv_level * profile["irradiance_factor"], 0.0, 1.0))
    if seasonal_pv_level > 0 and hasattr(net_seasonal, "apply_pv_penetration"):
        net_seasonal = net_seasonal.apply_pv_penetration(seasonal_pv_level)

    env = VectorizedDynamicDOCREnv(net_seasonal, n_envs=1, seed=seed + 1800, fault_type=fault_type,
                                    curriculum_stage=3)
    act_fn, _mean_fn, _comm_graph, _actor = pipeline._get_actor_for_eval(env, result, net=net_seasonal)
    rng = np.random.default_rng(seed + 1800)

    obs = env.reset(seed=seed + 1800, pv_level=seasonal_pv_level)[0]
    done, steps = False, 0
    while not done and steps < env.max_steps:
        actions = np.zeros((env.n_agents, env.ACT_DIM))
        for i in range(env.n_agents):
            a, _ = act_fn(obs, i, rng)
            actions[i] = a
        next_obs, _, done, _info = env.step(actions[None, :, :])
        obs = next_obs[0]
        steps += 1

    TMS, PS = env.get_state()
    of_seasonal, _, _ = objective_function(net_seasonal, TMS[0], PS[0], fault_type=fault_type)
    constraints = check_constraints(net_seasonal, TMS[0], PS[0], fault_type=fault_type)

    hours = np.arange(24)
    seasonal_irradiance = np.maximum(0, np.sin(np.pi * hours / 12) * 1000) * profile["irradiance_factor"]
    hour_pv_levels = np.clip(pv_level * seasonal_irradiance / 1000.0, 0.0, 1.0)

    return {
        "of": float(of_seasonal), "cti_ok": constraints["cti_satisfied"],
        "pv_variability": float(np.std(hour_pv_levels)),
    }


def run_core_K_vectorized(seed: Optional[int] = None, cache: Optional[CacheManager] = None,
                           fast_mode: Optional[bool] = None) -> Dict:
    """Vectorized-path counterpart of run_core_training.py's run_core_K()
    (seasonal generalization). Writes to a DIFFERENT cache name
    ("core_K_seasonal_vectorized"). `fast_mode` -- see
    run_core_G_vectorized's docstring."""
    seed = seed if seed is not None else SEED
    cache = cache if cache is not None else CacheManager(CACHE_DIR)
    fast_mode = fast_mode if fast_mode is not None else FAST_MODE

    config = {
        "name": "core_K_seasonal_vectorized",
        "seasons": ["spring", "summer", "autumn", "winter"],
        "seed": seed, "fault_type": FAULT_TYPE, "fast_mode": fast_mode,
        "vectorized_path_v14_7": True,
        "topology_fp": topology_fp("8bus"),
    }
    cached = cache.get(config)
    if cached is not None:
        return cached

    result = _train_base_policy_vectorized(system="8bus", seed=seed, fast_mode=fast_mode)
    net = load_network("8bus")
    base_of = result.best_of

    results: Dict = {}
    for season in config["seasons"]:
        d = _eval_seasonal_vectorized(net, result, season=season, seed=seed)
        seasonal_of = d.get("of")
        performance = float(base_of / seasonal_of) if seasonal_of else None
        results[season] = {**d, "performance": performance}

    cache.put(config, results)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core", choices=["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K"], default="A")
    parser.add_argument("--n-runs", type=int, default=None)
    parser.add_argument("--n-episodes", type=int, default=None, help="Core A/C only")
    parser.add_argument("--episodes-per-task", type=int, default=None, help="Core B only")
    parser.add_argument("--system", default="8bus", help="Core B/C only")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.core == "A":
        result = run_core_A_vectorized(n_runs=args.n_runs, n_episodes=args.n_episodes, seed=args.seed)
        print(f"Core A (vectorized): n_runs={result['n_runs']}  "
              f"final global_return_mean={result['global_returns_mean'][-1]:.4f}  "
              f"final cti_violation_rate_mean={result['cti_violation_rates_mean'][-1]:.2f}%")
    elif args.core == "B":
        result = run_core_B_vectorized(n_runs=args.n_runs, episodes_per_task=args.episodes_per_task,
                                        seed=args.seed, system=args.system)
        last_ewc = result["with_ewc"][-1]
        print(f"Core B (vectorized, {args.system}): n_runs={len(result['with_ewc'])}  "
              f"pv_evaluation={last_ewc['pv_evaluation']}")
    elif args.core == "C":
        result = run_core_C_vectorized(n_runs=args.n_runs, n_episodes=args.n_episodes,
                                        seed=args.seed, system=args.system)
        for method, returns in result.items():
            print(f"Core C (vectorized, {args.system}) {method}: "
                  f"mean_final_return={np.mean(returns[:, -1]) if returns.size else float('nan'):.4f}")
    elif args.core == "D":
        result = run_core_D_vectorized(n_runs=args.n_runs, n_episodes=args.n_episodes, seed=args.seed)
        for cfg_name, runs in result.items():
            of_mean = np.mean([r["OF"] for r in runs])
            cti_mean = np.mean([r["CTI"] for r in runs])
            print(f"Core D (vectorized) {cfg_name}: OF_mean={of_mean:.4f}  CTI_mean={cti_mean:.4f}")
    elif args.core == "E":
        result = run_core_E_vectorized(n_seeds=args.n_runs, n_steps=args.n_episodes, seed=args.seed)
        print(f"Core E (vectorized): {len(result)} grid points  "
              f"mean_OF={np.mean([r['OF'] for r in result]):.4f}  "
              f"mean_CTI={np.mean([r['CTI'] for r in result]):.4f}")
    elif args.core == "F":
        result = run_core_F_vectorized(seed=args.seed)
        for sys_name, r in result.items():
            print(f"Core F (vectorized) {sys_name}: n_relays={r['n_relays']}  "
                  f"train_time={r['train_time']:.1f}s  final_performance={r['final_performance']:.4f}")
    elif args.core == "G":
        result = run_core_G_vectorized(seed=args.seed)
        print(f"Core G (vectorized): {list(result.keys())}")
    elif args.core == "H":
        result = run_core_H_vectorized(seed=args.seed)
        print(f"Core H (vectorized): base_of={result['zero_shot']['base_of']:.4f}  "
              f"transfer_of={result['zero_shot']['transfer_of']:.4f}")
    elif args.core == "I":
        result = run_core_I_vectorized(seed=args.seed)
        print(f"Core I (vectorized): latency_mean_ms={result['latency']['mean_ms']:.3f}  "
              f"compression={list(result['compression'].keys())}")
    elif args.core == "J":
        result = run_core_J_vectorized(seed=args.seed)
        print(f"Core J (vectorized): {list(result.keys())}")
    else:
        result = run_core_K_vectorized(seed=args.seed)
        for season, r in result.items():
            print(f"Core K (vectorized) {season}: of={r['of']:.4f}  performance={r['performance']:.4f}")
