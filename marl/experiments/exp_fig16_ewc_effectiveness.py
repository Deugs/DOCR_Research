
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
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
os.makedirs("outputs", exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from network.network_data import get_8bus
from marl.marl_env import DynamicDOCREnv
from marl.marl_agents import MAPPOMARLTrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

TASKS = [0.0, 0.25, 0.50, 0.75, 1.0]
EPISODES_PER_TASK = 500
N_EVAL = 30


def run_curriculum(use_ewc: bool, seed: int = 42, system: str = "8bus"):
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
    env = DynamicDOCREnv(net, seed=seed, use_pandapower=False,
                          fault_type="3ph", curriculum_stage=0)
    trainer = MAPPOMARLTrainer(env, use_comm=True, seed=seed)
    
    if hasattr(trainer, 'use_safety'):
        trainer.use_safety = True
    if hasattr(trainer, 'use_ewc'):
        trainer.use_ewc = use_ewc
    
    rng = np.random.default_rng(seed)
    
    # Track performance on each task after each training phase
    task_performances = {task_idx: [] for task_idx in range(len(TASKS))}
    
    for train_task_idx, pv_level in enumerate(TASKS):
        logger.info("  Training on Task %d (%.0f%% PV) EWC=%s...", train_task_idx, pv_level*100, use_ewc)
        env.curriculum_stage = int(pv_level * 5)
        
        # Train
        for ep in range(EPISODES_PER_TASK):
            obs = env.reset(seed=seed + train_task_idx * 10000 + ep)
            done = False
            while not done:
                actions = np.zeros((env.n_agents, env.ACT_DIM))
                for i in range(env.n_agents):
                    a, _ = trainer.actor.sample(obs[i], rng)
                    actions[i] = a
                obs, _, done, _ = env.step(actions)
        
        # Evaluate on ALL tasks (including previous)
        for eval_task_idx in range(len(TASKS)):
            eval_pv = TASKS[eval_task_idx]
            env.curriculum_stage = int(eval_pv * 5)
            
            rewards = []
            for ep in range(N_EVAL):
                obs = env.reset(seed=seed + 50000 + eval_task_idx * 1000 + ep)
                done = False
                ep_reward = 0
                while not done:
                    actions = np.zeros((env.n_agents, env.ACT_DIM))
                    for i in range(env.n_agents):
                        a, _ = trainer.actor.sample(obs[i], rng)
                        actions[i] = a
                    obs, r, done, _ = env.step(actions)
                    ep_reward += r
                rewards.append(ep_reward)
            
            task_performances[eval_task_idx].append(np.mean(rewards))
            logger.info("    After T%d: T%d perf = %.4f", train_task_idx, eval_task_idx, np.mean(rewards))
    
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


def main(n_seeds: int = 10, output_dir: str = "outputs"):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    logger.info("Figure 16 — EWC Effectiveness Analysis")
    logger.info("  Seeds: %d", n_seeds)
    
    ewc_perfs_all = []
    noewc_perfs_all = []
    
    for seed in range(n_seeds):
        logger.info("\\nSeed %d/%d", seed + 1, n_seeds)
        logger.info("  Running with EWC...")
        ewc_perf = run_curriculum(use_ewc=True, seed=seed)
        ewc_perfs_all.append(ewc_perf)
        
        logger.info("  Running without EWC...")
        noewc_perf = run_curriculum(use_ewc=False, seed=seed + 1000)
        noewc_perfs_all.append(noewc_perf)
    
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
    args = p.parse_args()
    main(args.n_seeds, args.output_dir, args.system)