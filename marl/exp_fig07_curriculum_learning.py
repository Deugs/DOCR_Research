"""
exp_fig07_curriculum_learning.py  —  Figure 7 (IMPROVED)
─────────────────────────────────────────────────────────────
Curriculum Learning Progression (MAPPO-Safe-EWC).
Spans ALL IEEE test systems: 3, 8, 14, 30, 39, 57, and 118 buses.
"""

import numpy as np
import os
import sys
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from network.network_data import load_network, get_network_metadata, NETWORKS
from marl.marl_env import DynamicDOCREnv
from marl.marl_agents import MAPPOMARLTrainer
from marl.safe_rl import create_safe_rl_env, SafeRLConfig
from network.relay_physics import check_constraints

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

TASKS = [0.0, 0.25, 0.50, 0.75, 1.0]
EPISODES_PER_TASK = 1000


def _get_current_obs(trainer, obs, rng):
    """Helper: enhance observations via comm_graph if enabled."""
    if trainer.use_comm and trainer.comm_graph is not None:
        enhanced_obs, _, _ = trainer.comm_graph.forward(obs, rng)
        return enhanced_obs
    return obs


def evaluate_on_task(trainer, env, task_idx: int, n_eval: int = 20, seed: int = 0) -> float:
    """Evaluate current policy on a specific task."""
    rng = np.random.default_rng(seed)
    env.curriculum_stage = int(TASKS[task_idx] * 5)
    
    rewards = []
    for ep in range(n_eval):
        obs = env.reset(seed=seed + ep)
        done = False
        ep_reward = 0
        while not done:
            current_obs = _get_current_obs(trainer, obs, rng)
            actions = np.zeros((env.n_agents, env.ACT_DIM))
            for i in range(env.n_agents):
                a, _ = trainer.actor.sample(current_obs[i], rng)
                actions[i] = a
            obs, r, done, _ = env.step(actions)
            ep_reward += r
        rewards.append(ep_reward)
    
    return float(np.mean(rewards))


def run_curriculum_training(
    network_name: str = "8bus",
    seed: int = 42,
    episodes_per_task: int = 1000,
) -> Dict:
    """Run full curriculum training with comprehensive logging."""
    net = load_network(network_name)
    raw_env = DynamicDOCREnv(net, seed=seed, use_pandapower=False,
                              fault_type="3ph", curriculum_stage=0)
    # This run is "MAPPO-Safe-EWC" (see title everywhere below) — wrap with
    # the real Safe RL layer BEFORE building the trainer.
    # `trainer.use_safety = True` was a dead attribute set (MAPPOMARLTrainer
    # never defines `use_safety`), so this run was silently unconstrained.
    env = create_safe_rl_env(raw_env, SafeRLConfig(
        project_actions=True, enforce_cti=True,
        lambda_init=1.0, lambda_lr=0.01, rho_penalty=10.0,
        use_pid=True, adaptive_rho=True,
    ))

    # ── FIX: use_comm=False to avoid obs_dim mismatch, or keep True and use _get_current_obs ──
    # Option A (recommended for now): disable communication to match the manual loop
    trainer = MAPPOMARLTrainer(env, use_comm=False, hidden_actor=64,
                               hidden_critic=64, seed=seed)

    # Enable EWC
    if hasattr(trainer, 'use_ewc'):
        trainer.use_ewc = True
    
    # Storage
    log = {
        "episode": [], "task_id": [], "mean_reward": [],
        "cti_violated": [], "ewc_loss": [], "fisher_norm": [],
        "param_drift": [], "task_boundary": [],
    }
    
    # Store initial parameters for drift computation
    initial_params = None
    if hasattr(trainer.actor, 'get_all_params'):
        initial_params = trainer.actor.get_all_params().copy()
    
    # Track per-task final performance for transfer metrics
    task_final_performance = {}  # task_idx -> performance at end of task training
    backward_transfer = {}       # (eval_task, after_task) -> performance
    forward_transfer = {}        # (eval_task, after_task) -> performance
    
    total_episodes = 0
    rng = np.random.default_rng(seed)
    
    for task_idx, pv_level in enumerate(TASKS):
        logger.info("Task %d: %.0f%% PV (%d episodes)", task_idx, pv_level*100, episodes_per_task)
        
        # Update environment for new task
        env.curriculum_stage = int(pv_level * 5)
        
        # Store parameters at start of task (for drift after task)
        task_start_params = None
        if hasattr(trainer.actor, 'get_all_params'):
            task_start_params = trainer.actor.get_all_params().copy()
        
        # Train on this task
        for ep in range(episodes_per_task):
            obs = env.reset(seed=seed + total_episodes)
            done = False
            ep_reward = 0
            ep_cti = 0
            
            while not done:
                # ── FIX: use enhanced observations if communication is enabled ──
                current_obs = _get_current_obs(trainer, obs, rng)
                
                actions = np.zeros((env.n_agents, env.ACT_DIM))
                for i in range(env.n_agents):
                    a, _ = trainer.actor.sample(current_obs[i], rng)
                    actions[i] = a
                obs, reward, done, info = env.step(actions)
                ep_reward += reward
            
            TMS, PS = env.get_state()
            c = check_constraints(net, TMS, PS)
            ep_cti = c["cti_violations"]
            
            # EWC loss
            ewc_val = 0.0
            if hasattr(trainer, 'ewc_loss') and trainer.ewc_loss is not None:
                ewc_val = float(trainer.ewc_loss)
            elif hasattr(trainer, 'ewc_regularizer'):
                ewc_val = float(trainer.ewc_regularizer.loss()) if hasattr(trainer.ewc_regularizer, 'loss') else 0.0
            
            # Fisher norm
            fisher_norm = 0.0
            if hasattr(trainer, 'fisher_diagonal'):
                fisher_norm = float(np.linalg.norm(trainer.fisher_diagonal))
            
            # Parameter drift from initial
            param_drift = 0.0
            if initial_params is not None and hasattr(trainer.actor, 'get_all_params'):
                current_params = trainer.actor.get_all_params()
                param_drift = float(np.linalg.norm(current_params - initial_params) ** 2)
            
            log["episode"].append(total_episodes)
            log["task_id"].append(task_idx)
            log["mean_reward"].append(ep_reward)
            log["cti_violated"].append(ep_cti)
            log["ewc_loss"].append(ewc_val)
            log["fisher_norm"].append(fisher_norm)
            log["param_drift"].append(param_drift)
            
            total_episodes += 1
        
        # Record final performance on current task
        task_final_performance[task_idx] = evaluate_on_task(trainer, env, task_idx, n_eval=30, seed=seed + 10000)
        logger.info("  Task %d final performance: %.4f", task_idx, task_final_performance[task_idx])
        
        # Backward transfer: evaluate on ALL previous tasks
        if task_idx > 0:
            for prev_task in range(task_idx):
                perf_after = evaluate_on_task(trainer, env, prev_task, n_eval=20, 
                                               seed=seed + 20000 + prev_task * 1000)
                # Forgetting = performance drop
                forgetting = task_final_performance[prev_task] - perf_after
                backward_transfer[(prev_task, task_idx)] = {
                    "before": task_final_performance[prev_task],
                    "after": perf_after,
                    "forgetting": forgetting,
                    "forgetting_pct": forgetting / max(abs(task_final_performance[prev_task]), 1e-9) * 100,
                }
                logger.info("  Backward T%d→T%d: before=%.4f after=%.4f forgetting=%.2f%%",
                           prev_task, task_idx, task_final_performance[prev_task], perf_after,
                           backward_transfer[(prev_task, task_idx)]["forgetting_pct"])
        
        # Forward transfer: evaluate on next task (if exists) with current policy
        if task_idx < len(TASKS) - 1:
            next_perf = evaluate_on_task(trainer, env, task_idx + 1, n_eval=20,
                                          seed=seed + 30000 + task_idx * 1000)
            forward_transfer[(task_idx, task_idx + 1)] = next_perf
            logger.info("  Forward T%d→T%d: %.4f", task_idx, task_idx + 1, next_perf)
        
        # Mark task boundary
        log["task_boundary"].append(total_episodes)
    
    return log, task_final_performance, backward_transfer, forward_transfer


def main(
    network: str = "8bus",
    seed: int = 42,
    episodes_per_task: int = 1000,
    n_seeds: int = 3,
    output_dir: str = "outputs",
):
    # Create network-specific output directory
    net_dir = Path(output_dir) / network
    net_dir.mkdir(parents=True, exist_ok=True)
    
    # Log network metadata
    meta = get_network_metadata()[network]
    logger.info("=" * 70)
    logger.info("Figure 7 — Curriculum Learning Progression")
    logger.info("  Network: %s (%d buses, %d relays, %d pairs)",
                network, meta["n_buses"], meta["n_relays"], meta["n_pairs"])
    logger.info("  %s", meta["description"])
    logger.info("  Tasks: %s", TASKS)
    logger.info("  Episodes per task: %d, Seeds: %d", episodes_per_task, n_seeds)
    logger.info("=" * 70)
    
    all_logs = []
    all_bt = []
    all_ft = []
    
    for s in range(n_seeds):
        logger.info("\n=== Seed %d/%d ===", s + 1, n_seeds)
        log, task_perf, bt, ft = run_curriculum_training(
            network_name=network,
            seed=seed + s * 10000,
            episodes_per_task=episodes_per_task,
        )
        all_logs.append(log)
        all_bt.append(bt)
        all_ft.append(ft)
    
    # Aggregate across seeds
    n_total_episodes = len(all_logs[0]["episode"])
    
    rewards = np.array([log["mean_reward"] for log in all_logs])
    cti_viols = np.array([log["cti_violated"] for log in all_logs])
    ewc_losses = np.array([log["ewc_loss"] for log in all_logs])
    fisher_norms = np.array([log["fisher_norm"] for log in all_logs])
    param_drifts = np.array([log["param_drift"] for log in all_logs])
    
    mean_rewards = rewards.mean(axis=0)
    std_rewards = rewards.std(axis=0)
    mean_cti = cti_viols.mean(axis=0)
    std_cti = cti_viols.std(axis=0)
    mean_ewc = ewc_losses.mean(axis=0)
    std_ewc = ewc_losses.std(axis=0)
    mean_fisher = fisher_norms.mean(axis=0)
    mean_drift = param_drifts.mean(axis=0)
    
    episodes = np.array(all_logs[0]["episode"])
    task_ids = np.array(all_logs[0]["task_id"])
    task_boundaries = all_logs[0]["task_boundary"]
    
    # Aggregate backward transfer
    avg_forgetting = {}
    for key in all_bt[0].keys():
        forget_vals = [bt[key]["forgetting_pct"] for bt in all_bt]
        avg_forgetting[key] = {
            "mean": np.mean(forget_vals),
            "std": np.std(forget_vals),
            "forgetting": np.mean([bt[key]["forgetting"] for bt in all_bt]),
        }
    
    # Save
    np.savez(
        net_dir / "fig07_curriculum.npz",
        episodes=episodes, task_ids=task_ids,
        mean_rewards=mean_rewards, std_rewards=std_rewards,
        mean_cti=mean_cti, std_cti=std_cti,
        mean_ewc=mean_ewc, std_ewc=std_ewc,
        mean_fisher=mean_fisher, mean_drift=mean_drift,
        task_boundaries=np.array(task_boundaries),
        n_seeds=np.array([n_seeds]),
        network=network,
        n_buses=meta["n_buses"],
        n_relays=meta["n_relays"],
        n_pairs=meta["n_pairs"],
    )
    logger.info("Saved: %s", net_dir / "fig07_curriculum.npz")
    
    # Plot
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle(
        f"Figure 7 — Curriculum Learning ({network}, {meta['n_buses']}-Bus, "
        f"{n_seeds} seeds, MAPPO-Safe-EWC)",
        fontsize=12
    )
    
    window = 50
    def smooth(y):
        if len(y) < window:
            return y
        return np.convolve(y, np.ones(window)/window, mode='valid')
    
    # Panel 1: Reward with confidence bands
    ax = axes[0, 0]
    ax.fill_between(episodes, mean_rewards - std_rewards, mean_rewards + std_rewards,
                    alpha=0.2, color="#2563eb")
    ax.plot(episodes, mean_rewards, color="#2563eb", lw=2, label="Mean ± std")
    for tb in task_boundaries:
        ax.axvline(tb, color="#ff6b35", ls="--", lw=1, alpha=0.7)
    ax.set_ylabel("Episode Reward", fontsize=10)
    ax.set_title("Learning Curve with Task Transitions", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Panel 2: CTI Violations
    ax = axes[0, 1]
    ax.fill_between(episodes, mean_cti - std_cti, mean_cti + std_cti,
                    alpha=0.2, color="#ff6b35")
    ax.plot(episodes, mean_cti, color="#ff6b35", lw=2)
    for tb in task_boundaries:
        ax.axvline(tb, color="#ff6b35", ls="--", lw=1, alpha=0.7)
    ax.set_ylabel("CTI Violations", fontsize=10)
    ax.set_title("Constraint Violations", fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Panel 3: EWC Loss
    ax = axes[1, 0]
    ax.fill_between(episodes, mean_ewc - std_ewc, mean_ewc + std_ewc,
                    alpha=0.2, color="#39d353")
    ax.plot(episodes, mean_ewc, color="#39d353", lw=2)
    for tb in task_boundaries:
        ax.axvline(tb, color="#ff6b35", ls="--", lw=1, alpha=0.7)
    ax.set_ylabel("EWC Loss", fontsize=10)
    ax.set_title("Elastic Weight Consolidation", fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Panel 4: Fisher Norm
    ax = axes[1, 1]
    ax.plot(episodes, mean_fisher, color="#a371f7", lw=2)
    for tb in task_boundaries:
        ax.axvline(tb, color="#ff6b35", ls="--", lw=1, alpha=0.7)
    ax.set_ylabel("||F|| (Fisher norm)", fontsize=10)
    ax.set_title("Fisher Information Magnitude", fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Panel 5: Parameter Drift
    ax = axes[2, 0]
    ax.plot(episodes, mean_drift, color="#ff6b6b", lw=2)
    for tb in task_boundaries:
        ax.axvline(tb, color="#ff6b35", ls="--", lw=1, alpha=0.7)
    ax.set_ylabel("||θ - θ₀||²", fontsize=10)
    ax.set_xlabel("Episode", fontsize=10)
    ax.set_title("Parameter Drift from Initialization", fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Panel 6: Backward Transfer (Forgetting) Matrix
    ax = axes[2, 1]
    n_tasks = len(TASKS)
    forgetting_matrix = np.zeros((n_tasks, n_tasks))
    forgetting_matrix[:] = np.nan
    
    for (eval_task, after_task), vals in avg_forgetting.items():
        forgetting_matrix[eval_task, after_task] = vals["mean"]
    
    im = ax.imshow(forgetting_matrix, cmap="RdYlGn_r", vmin=-50, vmax=50, aspect="auto")
    ax.set_xticks(range(n_tasks))
    ax.set_yticks(range(n_tasks))
    ax.set_xticklabels([f"After T{i}" for i in range(n_tasks)], fontsize=8)
    ax.set_yticklabels([f"T{i}" for i in range(n_tasks)], fontsize=8)
    ax.set_xlabel("Training Phase", fontsize=9)
    ax.set_ylabel("Evaluated Task", fontsize=9)
    ax.set_title("Backward Transfer (% forgetting)", fontsize=11)
    plt.colorbar(im, ax=ax, shrink=0.6)
    
    # Add text annotations
    for i in range(n_tasks):
        for j in range(n_tasks):
            if not np.isnan(forgetting_matrix[i, j]):
                ax.text(j, i, f"{forgetting_matrix[i,j]:.1f}%", ha="center", va="center",
                       fontsize=7, color="white" if abs(forgetting_matrix[i,j]) > 25 else "black")
    
    # Add task labels
    for i, tb in enumerate(task_boundaries):
        if i < len(TASKS):
            for ax in axes.flat[:5]:
                ax.text(tb - 30, ax.get_ylim()[1] * 0.9, f"T{i}\n{TASKS[i]*100:.0f}%",
                       fontsize=7, ha="right", color="#ff6b35", fontweight="bold")
    
    fig.tight_layout()
    fig.savefig(net_dir / "fig07_curriculum.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", net_dir / "fig07_curriculum.png")
    
    # Summary
    print("\n" + "=" * 70)
    print(f"CURRICULUM LEARNING SUMMARY — {network} ({meta['n_buses']}-Bus)")
    print("=" * 70)
    print(f"{'Task':>8} {'PV%':>8} {'Final Reward':>14} {'CTI Viol':>10} {'EWC Loss':>12}")
    print("-" * 70)
    for i in range(len(TASKS)):
        mask = task_ids == i
        print(f"{i:>8} {TASKS[i]*100:>7.0f}% {mean_rewards[mask][-1]:>14.4f} {mean_cti[mask][-1]:>10.2f} {mean_ewc[mask][-1]:>12.4f}")
    print("=" * 70)
    print("\nBACKWARD TRANSFER (Forgetting):")
    print(f"{'Eval Task':>10} {'After Task':>12} {'Before':>10} {'After':>10} {'Forget%':>10} {'Std':>8}")
    print("-" * 70)
    for (eval_task, after_task), vals in avg_forgetting.items():
        print(f"T{eval_task:>8} T{after_task:>10} {vals['forgetting']:>10.4f} {vals['mean']:>9.1f}% ±{vals['std']:>5.1f}%")
    print("=" * 70)


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Figure 7 — Curriculum Learning (All IEEE Test Systems)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available networks:
  3bus   — IEEE 3-Bus   (  6 relays,   6 pairs)  — toy validation
  8bus   — IEEE 8-Bus   ( 14 relays,  20 pairs)  — small-scale MARL
  14bus  — IEEE 14-Bus  ( 20 relays,  38 pairs)  — standard benchmark
  30bus  — IEEE 30-Bus  ( 42 relays,  70 pairs)  — medium-scale MARL
  39bus  — IEEE 39-Bus  ( 46 relays,  56 pairs)  — stability studies
  57bus  — IEEE 57-Bus  ( 80 relays, 128 pairs)  — large-scale MARL
  118bus — IEEE 118-Bus (186 relays, 304 pairs)  — very large-scale
        """
    )
    p.add_argument("--network", type=str, default="8bus",
                   choices=list(NETWORKS.keys()),
                   help="IEEE test system to use (default: 8bus)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--episodes_per_task", type=int, default=1000)
    p.add_argument("--n_seeds", type=int, default=3)
    p.add_argument("--output_dir", type=str, default="outputs")
    args = p.parse_args()
    main(args.network, args.seed, args.episodes_per_task, args.n_seeds, args.output_dir)