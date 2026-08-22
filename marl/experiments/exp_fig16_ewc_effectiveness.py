
"""
exp_fig16_ewc_effectiveness.py  —  Figure 16
─────────────────────────────────────────────
EWC Effectiveness — Performance Drop After Task Transition.

Performance on Task k after training Task k+1, with vs. without EWC.
Backward/forward transfer metrics.

Run two ablation experiments:
  (a) MAPPO-Safe-EWC (full)
  (b) MAPPO-Safe-NoEWC (λ_EWC = 0)

Both with curriculum 0%→100% PV.
After each task transition, evaluate on ALL previous tasks.

Outputs
-------
  outputs/fig16_ewc_effectiveness.npz
  outputs/fig16_ewc_effectiveness.png
"""

import numpy as np
import os
import sys
import logging
import argparse
import multiprocessing as mp
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
os.makedirs("outputs", exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import threadpoolctl

from network.network_data import get_8bus
from marl.marl_env import DynamicDOCREnv
from marl.marl_agents import MAPPOMARLTrainer
from marl.safe_rl import create_safe_rl_env, SafeRLConfig
from marl.continual_trainer import create_ewc_trainer, EWCConfig, TaskSimilarityEstimator

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

TASKS = [0.0, 0.25, 0.50, 0.75, 1.0]
EPISODES_PER_TASK = 500
N_EVAL = 30


def run_curriculum(use_ewc: bool, seed: int = 42, system: str = "8bus",
                    episodes_per_task: int = EPISODES_PER_TASK, n_eval: int = N_EVAL):
    """Run curriculum training with or without EWC."""
    if system == "8bus":
        net = get_8bus()
    elif system == "14bus":
        from network.network_data import get_14bus
        net = get_14bus()
    elif system == "30bus":
        from network.network_data import get_30bus
        net = get_30bus()
    elif system == "39bus":
        from network.network_data import get_39bus
        net = get_39bus()
    else:
        net = get_8bus()
    raw_env = DynamicDOCREnv(net, seed=seed, use_pandapower=False,
                              fault_type="3ph", curriculum_stage=0)
    # Both conditions here are "MAPPO-Safe" (per module docstring: (a)
    # MAPPO-Safe-EWC, (b) MAPPO-Safe-NoEWC) — only EWC on/off differs, so
    # both get the real Safe RL wrapper. `trainer.use_safety = True` was a
    # dead attribute set (MAPPOMARLTrainer never defines `use_safety`).
    safe_env = create_safe_rl_env(raw_env, SafeRLConfig(
        project_actions=True, enforce_cti=True,
        lambda_init=1.0, lambda_lr=0.01, rho_penalty=10.0,
        use_pid=True, adaptive_rho=True,
    ))
    trainer = MAPPOMARLTrainer(safe_env, use_comm=True, seed=seed)

    # `trainer.use_ewc` was a dead attribute set -- MAPPOMARLTrainer never
    # defines it, so "with EWC" vs "without EWC" trained identically and
    # this figure (literally named "EWC effectiveness") never applied any
    # EWC penalty at all. Real EWC needs an actual create_ewc_trainer
    # wrapper plus per-task Fisher registration -- the curriculum pattern
    # (lagged consolidation via register_task/update_fisher/finalize) is
    # the same one already fixed this session in run_core_training.py's
    # Core B/D and marl/experiments/exp_fig39_ablation.py.
    ewc_trainer = create_ewc_trainer(trainer, EWCConfig()) if use_ewc else None
    similarity_estimator = TaskSimilarityEstimator() if use_ewc else None

    rng = np.random.default_rng(seed)

    def _act(all_obs, agent_idx, rng):
        """Route raw obs through the comm graph before sampling. The
        actor was built expecting the enhanced (obs+message) dimension
        whenever use_comm is on (see MAPPOMARLTrainer.train()'s own loop)
        -- feeding it raw obs directly mismatches dimensions and crashes.
        """
        if trainer.comm_graph is not None:
            enhanced, _, _ = trainer.comm_graph.forward(all_obs, rng)
            return trainer.actor.sample(enhanced[agent_idx], rng)
        return trainer.actor.sample(all_obs[agent_idx], rng)

    # Track performance on each task after each training phase
    task_performances = {task_idx: [] for task_idx in range(len(TASKS))}

    # Convert episode budget to a step budget for trainer.train(). The
    # previous version of this loop never called trainer.train() at all --
    # only trainer.actor.sample() against a frozen, randomly-initialized
    # network -- so every "task performance" it produced was noise from an
    # untrained policy, not genuine curriculum learning (the same bug
    # already fixed in experiments/exp_fig12_convergence_curves.py).
    steps_per_task = episodes_per_task * 300

    prev_task_name = None
    prev_fisher = None
    prev_conditions = None
    prev_params = None

    for train_task_idx, pv_level in enumerate(TASKS):
        logger.info("  Training on Task %d (%.0f%% PV) EWC=%s...", train_task_idx, pv_level*100, use_ewc)
        raw_env.curriculum_stage = int(pv_level * 5)

        if use_ewc and ewc_trainer is not None and prev_task_name is not None:
            # Apply the EWC gradient correction before training the new
            # task starts (same one-shot correction used in Core B/D).
            cur = trainer.actor.get_all_params()
            ewc_grad = ewc_trainer.apply_ewc_to_gradient(cur, np.zeros_like(cur))
            trainer.actor.set_all_params(cur - 0.01 * ewc_grad)

        # Train
        trainer.train(total_steps=steps_per_task, verbose=False)
        task_name = f"PV{int(pv_level*100):03d}"

        if use_ewc and ewc_trainer is not None:
            cur = trainer.actor.get_all_params()
            obs_f = safe_env.reset(seed=seed + 9000 + train_task_idx)
            s_obs, s_act = [], []
            done_f, sf = False, 0
            while not done_f and sf < 30:
                acts = np.zeros((raw_env.n_agents, raw_env.ACT_DIM))
                enhanced_f, _, _ = trainer.comm_graph.forward(obs_f, rng) if trainer.comm_graph is not None else (obs_f, None, None)
                for i in range(raw_env.n_agents):
                    a, _ = trainer.actor.sample(enhanced_f[i], rng)
                    acts[i] = a
                    s_obs.append(enhanced_f[i].copy())
                    s_act.append(a.copy())
                obs_f, _, done_f, _ = safe_env.step(acts); sf += 1
            n_p = len(cur)
            fisher = np.zeros(n_p)
            if s_obs:
                idx = rng.choice(len(s_obs), size=min(30, len(s_obs)), replace=False)
                h = 1e-3
                for p_idx in rng.choice(n_p, size=min(n_p, 50), replace=False):
                    bl = np.mean([trainer.actor.log_prob(s_obs[j], s_act[j]) for j in idx])
                    b2 = cur.copy(); b2[p_idx] += h
                    trainer.actor.set_all_params(b2)
                    ul = np.mean([trainer.actor.log_prob(s_obs[j], s_act[j]) for j in idx])
                    fisher[p_idx] = ((ul - bl) / h) ** 2
                trainer.actor.set_all_params(cur)

            cur_conditions = {"pv_level": pv_level}
            # register_task() must run FIRST -- it's what actually
            # consolidates the previous task into task_memory. adapt_lambda()
            # only updates entries already in task_memory, so calling it
            # before register_task would be a silent no-op.
            ewc_trainer.ewc_regularizer.register_task(task_name, cur)
            if prev_conditions is not None:
                similarity = similarity_estimator.estimate_similarity(
                    prev_conditions, prev_params, cur_conditions, cur
                )
                ewc_trainer.ewc_regularizer.adapt_lambda(similarity)
            if prev_task_name is not None:
                ewc_trainer.ewc_regularizer.update_fisher(prev_task_name, prev_fisher)
            prev_task_name = task_name
            prev_fisher = fisher
            prev_conditions = cur_conditions
            prev_params = cur.copy()

        # Evaluate on ALL tasks (including previous)
        for eval_task_idx in range(len(TASKS)):
            eval_pv = TASKS[eval_task_idx]
            raw_env.curriculum_stage = int(eval_pv * 5)

            rewards = []
            for ep in range(n_eval):
                obs = safe_env.reset(seed=seed + 50000 + eval_task_idx * 1000 + ep)
                done = False
                ep_reward = 0
                while not done:
                    actions = np.zeros((raw_env.n_agents, raw_env.ACT_DIM))
                    for i in range(raw_env.n_agents):
                        a, _ = _act(obs, i, rng)
                        actions[i] = a
                    obs, r, done, _ = safe_env.step(actions)
                    ep_reward += float(np.mean(r)) if isinstance(r, np.ndarray) else r
                rewards.append(ep_reward)

            task_performances[eval_task_idx].append(np.mean(rewards))
            logger.info("    After T%d: T%d perf = %.4f", train_task_idx, eval_task_idx, np.mean(rewards))

    if use_ewc and ewc_trainer is not None:
        # TASKS[-1] is never consolidated by the loop above (that only
        # happens lazily when the *next* task's register_task() fires) --
        # finalize() closes it out explicitly, using the real just-computed
        # Fisher (stashed via this update_fisher call) instead of the
        # zero-information uniform placeholder finalize() would otherwise
        # fall back to.
        ewc_trainer.ewc_regularizer.update_fisher(prev_task_name, prev_fisher)
        ewc_trainer.ewc_regularizer.finalize()

    return task_performances


def compute_transfer_metrics(performances: dict):
    """Compute backward and forward transfer metrics."""
    n_tasks = len(TASKS)
    backward_transfer = np.zeros((n_tasks, n_tasks))
    forward_transfer = np.zeros((n_tasks, n_tasks))
    
    for i in range(n_tasks):
        final_perf_i = performances[i][i]  # Performance on task i after training on task i
        for j in range(n_tasks):
            if j > i:
                # Forward: perf on task j after training up to task i
                forward_transfer[i, j] = performances[j][i]
            elif j < i:
                # Backward: perf on task j after training task i
                perf_after_i = performances[j][i]
                bt = (perf_after_i - final_perf_i) / max(abs(final_perf_i), 1e-9) * 100
                backward_transfer[i, j] = bt
    
    return backward_transfer, forward_transfer


def _pool_worker_init():
    threadpoolctl.threadpool_limits(1)


def _run_job(args):
    kind, seed_idx, actual_seed, system, episodes_per_task, n_eval = args
    perf = run_curriculum(use_ewc=(kind == "ewc"), seed=actual_seed, system=system,
                           episodes_per_task=episodes_per_task, n_eval=n_eval)
    return kind, seed_idx, perf


def main(n_seeds: int = 10, output_dir: str = "outputs", system: str = "8bus", n_workers: int = 1,
         episodes_per_task: int = EPISODES_PER_TASK, n_eval: int = N_EVAL):
    # A full run is 5 TASKS x episodes_per_task*300 steps x 2 conditions
    # (EWC/no-EWC) x n_seeds -- at the defaults (500, 10) that's ~15M
    # ES-based training steps, likely impractical to actually finish for
    # a quick check. --episodes_per_task/--n_eval (CLI below) let a
    # caller trade fidelity for a run that actually completes; the
    # defaults are unchanged so existing full-run invocations still get
    # the original episode budget.
    # Applied unconditionally — see baselines/experiments/unified_exp_fig05.py
    # for why: unpinned BLAS threads oversubscribe the machine under
    # multiprocessing and can make n_workers>1 slower than sequential, and
    # this also makes single-process results reproducible regardless of the
    # machine's ambient BLAS thread configuration.
    threadpoolctl.threadpool_limits(1)

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    logger.info("Figure 16 — EWC Effectiveness Analysis")
    logger.info("  Seeds: %d", n_seeds)

    # Same seed scheme as before: EWC run uses `seed`, no-EWC run uses
    # `seed + 1000` — kept as two jobs per seed index rather than paired
    # same-seed twins, matching the original sequential behavior.
    jobs = []
    for seed_idx in range(n_seeds):
        jobs.append(("ewc", seed_idx, seed_idx, system, episodes_per_task, n_eval))
        jobs.append(("noewc", seed_idx, seed_idx + 1000, system, episodes_per_task, n_eval))

    ewc_perfs_all = [None] * n_seeds
    noewc_perfs_all = [None] * n_seeds

    if n_workers > 1:
        with mp.Pool(n_workers, initializer=_pool_worker_init) as pool:
            for kind, seed_idx, perf in pool.imap_unordered(_run_job, jobs):
                if kind == "ewc":
                    ewc_perfs_all[seed_idx] = perf
                else:
                    noewc_perfs_all[seed_idx] = perf
                logger.info("  [done] %s seed_idx=%d", kind, seed_idx)
    else:
        for job in jobs:
            kind, seed_idx, perf = _run_job(job)
            if kind == "ewc":
                ewc_perfs_all[seed_idx] = perf
            else:
                noewc_perfs_all[seed_idx] = perf
            logger.info("  [done] %s seed_idx=%d", kind, seed_idx)
    
    # Average across seeds
    n_tasks = len(TASKS)
    ewc_matrix = np.zeros((n_tasks, n_tasks))
    noewc_matrix = np.zeros((n_tasks, n_tasks))
    
    for i in range(n_tasks):
        for j in range(n_tasks):
            ewc_matrix[i, j] = np.mean([p[i][j] for p in ewc_perfs_all])
            noewc_matrix[i, j] = np.mean([p[i][j] for p in noewc_perfs_all])
    
    # Save
    np.savez(
        out_path / "fig16_ewc_effectiveness.npz",
        ewc_matrix=ewc_matrix,
        noewc_matrix=noewc_matrix,
        tasks=np.array(TASKS),
        n_seeds=np.array([n_seeds]),
    )
    logger.info("Saved: %s", out_path / "fig16_ewc_effectiveness.npz")
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.suptitle("Figure 16 — EWC Effectiveness: Performance Drop After Task Transition", fontsize=12)
    
    # Panel 1: With EWC heatmap
    ax = axes[0]
    im = ax.imshow(ewc_matrix, cmap="RdYlGn", vmin=ewc_matrix.min(), vmax=ewc_matrix.max())
    ax.set_xticks(range(n_tasks))
    ax.set_yticks(range(n_tasks))
    ax.set_xticklabels([f"T{i}" for i in range(n_tasks)], fontsize=9)
    ax.set_yticklabels([f"After T{i}" for i in range(n_tasks)], fontsize=9)
    ax.set_xlabel("Evaluated on Task", fontsize=10)
    ax.set_ylabel("Trained up to Task", fontsize=10)
    ax.set_title("With EWC", fontsize=11)
    plt.colorbar(im, ax=ax, shrink=0.6)
    
    # Panel 2: Without EWC heatmap
    ax = axes[1]
    im = ax.imshow(noewc_matrix, cmap="RdYlGn", vmin=ewc_matrix.min(), vmax=ewc_matrix.max())
    ax.set_xticks(range(n_tasks))
    ax.set_yticks(range(n_tasks))
    ax.set_xticklabels([f"T{i}" for i in range(n_tasks)], fontsize=9)
    ax.set_yticklabels([f"After T{i}" for i in range(n_tasks)], fontsize=9)
    ax.set_xlabel("Evaluated on Task", fontsize=10)
    ax.set_ylabel("Trained up to Task", fontsize=10)
    ax.set_title("Without EWC", fontsize=11)
    plt.colorbar(im, ax=ax, shrink=0.6)
    
    # Panel 3: Diagonal comparison (forgetting)
    ax = axes[2]
    x = np.arange(n_tasks)
    ax.plot(x, np.diag(ewc_matrix), "o-", color="#2563eb", lw=2, markersize=8, label="With EWC")
    ax.plot(x, np.diag(noewc_matrix), "s-", color="#ff6b35", lw=2, markersize=8, label="Without EWC")
    ax.set_xticks(x)
    ax.set_xticklabels([f"T{i}" for i in range(n_tasks)], fontsize=9)
    ax.set_xlabel("Task", fontsize=10)
    ax.set_ylabel("Performance on Current Task", fontsize=10)
    ax.set_title("Current Task Performance (Forgetting)", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(out_path / "fig16_ewc_effectiveness.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", out_path / "fig16_ewc_effectiveness.png")
    
    # Summary
    print("\\n" + "=" * 60)
    print("EWC EFFECTIVENESS SUMMARY")
    print("=" * 60)
    print("Performance matrix (rows=trained up to, cols=evaluated on):")
    print("\\nWith EWC:")
    print(ewc_matrix)
    print("\\nWithout EWC:")
    print(noewc_matrix)
    print("=" * 60)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Figure 16 — EWC Effectiveness")
    p.add_argument("--n_seeds", type=int, default=10)
    p.add_argument("--output_dir", type=str, default="outputs")
    p.add_argument("--system", type=str, default="8bus", help="Test system (8bus, 14bus, 30bus, 39bus)")
    p.add_argument("--n_workers", type=int, default=1,
                   help="Parallel worker processes (default: 1 = sequential)")
    p.add_argument("--episodes_per_task", type=int, default=EPISODES_PER_TASK,
                   help=f"Training episodes per curriculum task (default: {EPISODES_PER_TASK}, "
                        f"a full run at that default is 5 tasks x {EPISODES_PER_TASK}*300 steps x "
                        "2 conditions x n_seeds -- lower this for a run that actually completes)")
    p.add_argument("--n_eval", type=int, default=N_EVAL,
                   help=f"Evaluation episodes per task after each training phase (default: {N_EVAL})")
    args = p.parse_args()
    main(args.n_seeds, args.output_dir, args.system, args.n_workers,
         args.episodes_per_task, args.n_eval)