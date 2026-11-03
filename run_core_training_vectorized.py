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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core", choices=["A", "B", "C", "D", "E"], default="A")
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
    else:
        result = run_core_E_vectorized(n_seeds=args.n_runs, n_steps=args.n_episodes, seed=args.seed)
        print(f"Core E (vectorized): {len(result)} grid points  "
              f"mean_OF={np.mean([r['OF'] for r in result]):.4f}  "
              f"mean_CTI={np.mean([r['CTI'] for r in result]):.4f}")
