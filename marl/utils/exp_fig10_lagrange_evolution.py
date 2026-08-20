"""
exp_fig10_lagrange_evolution.py  —  Figure 10
─────────────────────────────────────────────
Lagrange Multiplier Evolution per P/B Pair (MAPPO-Safe, 8-Bus).

Individual λ_ij trajectories for all primary/backup pairs.
Steady-state values, convergence times.

For 8-bus with 7 lines and 2 directions, ~14 pairs.
Plot all trajectories with transparency.
Identify: fastest/slowest converging, highest steady-state λ.

Outputs
-------
  outputs/fig10_lagrange.npz
  outputs/fig10_lagrange.png
"""

import numpy as np
import os
import sys
import logging
import argparse
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
os.makedirs("outputs", exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from network.network_data import get_8bus
from marl.marl_env import DynamicDOCREnv
from marl.marl_agents import MAPPOMARLTrainer
from marl.safe_rl import create_safe_rl_env, SafeRLConfig
from network.relay_physics import check_constraints

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def run_with_lagrange_logging(env, trainer, n_episodes: int = 2000, seed: int = 42):
    """Run training and log per-pair Lagrange multipliers."""
    rng = np.random.default_rng(seed)
    n_pairs = env.net.n_pairs
    
    log = {
        "episode": [],
        "lambda_matrix": [],  # (n_episodes, n_pairs)
        "violation_flags": [],  # (n_episodes, n_pairs)
    }
    
    for ep in range(n_episodes):
        obs = env.reset(seed=seed + ep)
        done = False
        
        while not done:
            actions = np.zeros((env.n_agents, env.ACT_DIM))
            for i in range(env.n_agents):
                a, _ = trainer.actor.sample(obs[i], rng)
                actions[i] = a
            obs, _, done, info = env.step(actions)
        
        # Extract Lagrange multipliers from the real Safe RL wrapper — these
        # evolve via genuine dual-ascent updates (SafeRLWrapper.step()
        # auto-triggers update_lagrangian() on episode end).
        lambdas = env.lagrangian.lambdas.copy()

        # Check violations per pair
        TMS, PS = env.get_state()
        c = check_constraints(env.net, TMS, PS)
        viol_flags = np.zeros(n_pairs)
        if "pair_ctis" in c:
            pair_ctis = c["pair_ctis"]
            viol_flags = (np.array(pair_ctis) < 0.3).astype(float)
        
        log["episode"].append(ep)
        log["lambda_matrix"].append(lambdas)
        log["violation_flags"].append(viol_flags)
        
        if ep % 200 == 0:
            logger.info("Ep %d: mean λ=%.3f max λ=%.3f active=%d/%d",
                       ep, np.mean(lambdas), np.max(lambdas), int(np.sum(lambdas > 0.01)), n_pairs)
    
    return log


def main(n_episodes: int = 2000, seed: int = 42, output_dir: str = "outputs"):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    logger.info("Figure 10 — Lagrange Multiplier Evolution")
    logger.info("  Episodes: %d", n_episodes)
    
    net = get_8bus()
    raw_env = DynamicDOCREnv(net, seed=seed, use_pandapower=False,
                              fault_type="3ph", curriculum_stage=5)
    # Wrap with the real Safe RL layer BEFORE building the trainer — this is
    # "MAPPO-Safe" (Figure 10 title), so its Lagrange multipliers must come
    # from genuine dual-ascent updates, not a dead hasattr check.
    env = create_safe_rl_env(raw_env, SafeRLConfig(
        project_actions=True, enforce_cti=True,
        lambda_init=1.0, lambda_lr=0.01, rho_penalty=10.0,
        use_pid=True, adaptive_rho=True,
    ))
    trainer = MAPPOMARLTrainer(env, use_comm=False, seed=seed)

    log = run_with_lagrange_logging(env, trainer, n_episodes, seed)
    
    lambda_mat = np.array(log["lambda_matrix"])  # (n_episodes, n_pairs)
    viol_mat = np.array(log["violation_flags"])
    episodes = np.array(log["episode"])
    n_pairs = lambda_mat.shape[1]
    
    # Compute statistics
    steady_state_start = int(0.8 * n_episodes)
    steady_lambdas = lambda_mat[steady_state_start:].mean(axis=0)
    
    # Convergence time: first episode where λ stays within 5% of final
    convergence_eps = []
    for pair_idx in range(n_pairs):
        final_val = steady_lambdas[pair_idx]
        threshold = 0.05 * max(final_val, 0.01)
        conv_ep = n_episodes
        for ep in range(n_episodes):
            if ep > 100 and np.all(np.abs(lambda_mat[ep-100:ep+1, pair_idx] - final_val) < threshold):
                conv_ep = ep
                break
        convergence_eps.append(conv_ep)
    
    # Save
    np.savez(
        out_path / "fig10_lagrange.npz",
        episodes=episodes,
        lambda_matrix=lambda_mat,
        violation_flags=viol_mat,
        steady_state_lambdas=steady_lambdas,
        convergence_episodes=np.array(convergence_eps),
        pair_names=np.array([f"P{i}-B{j}" for i in range(n_pairs) for j in range(n_pairs)][:n_pairs]),
    )
    logger.info("Saved: %s", out_path / "fig10_lagrange.npz")
    
    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Figure 10 — Lagrange Multiplier Evolution per P/B Pair (MAPPO-Safe, 8-Bus)", fontsize=12)
    
    # Panel 1: All trajectories
    ax = axes[0, 0]
    colors = plt.cm.viridis(np.linspace(0, 1, n_pairs))
    for i in range(n_pairs):
        ax.plot(episodes, lambda_mat[:, i], color=colors[i], lw=1, alpha=0.6,
               label=f"Pair {i}" if i < 5 else "")
    ax.set_ylabel("λ_ij", fontsize=10)
    ax.set_xlabel("Episode", fontsize=10)
    ax.set_title("All Pair Trajectories", fontsize=11)
    ax.grid(True, alpha=0.3)
    if n_pairs <= 10:
        ax.legend(fontsize=7, ncol=2)
    
    # Panel 2: Steady-state ranking
    ax = axes[0, 1]
    sorted_idx = np.argsort(steady_lambdas)[::-1]
    bars = ax.barh(range(n_pairs), steady_lambdas[sorted_idx], color="#2563eb", edgecolor="white")
    ax.set_yticks(range(n_pairs))
    ax.set_yticklabels([f"Pair {i}" for i in sorted_idx], fontsize=8)
    ax.set_xlabel("Steady-State λ", fontsize=10)
    ax.set_title("Steady-State Ranking (Most → Least Constrained)", fontsize=11)
    ax.grid(True, alpha=0.3, axis="x")
    ax.invert_yaxis()
    
    # Panel 3: Convergence time
    ax = axes[1, 0]
    bars = ax.bar(range(n_pairs), convergence_eps, color="#39d353", edgecolor="white")
    ax.set_xlabel("Pair Index", fontsize=10)
    ax.set_ylabel("Convergence Episode", fontsize=10)
    ax.set_title("Convergence Time (episodes to steady-state)", fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")
    
    # Panel 4: Correlation with violations
    ax = axes[1, 1]
    total_viols = viol_mat.sum(axis=0)
    ax.scatter(steady_lambdas, total_viols, c=convergence_eps, cmap="plasma", s=80, edgecolors="white")
    ax.set_xlabel("Steady-State λ", fontsize=10)
    ax.set_ylabel("Total Violations", fontsize=10)
    ax.set_title("λ vs. Violation Frequency", fontsize=11)
    cbar = plt.colorbar(ax.collections[0], ax=ax, shrink=0.6)
    cbar.set_label("Conv. Episode", fontsize=8)
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(out_path / "fig10_lagrange.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", out_path / "fig10_lagrange.png")
    
    # Summary
    print("\\n" + "=" * 60)
    print("LAGRANGE MULTIPLIER SUMMARY")
    print("=" * 60)
    print(f"{'Pair':>8} {'Steady λ':>12} {'Conv@Ep':>10} {'Total Viols':>12}")
    print("-" * 60)
    for i in range(n_pairs):
        print(f"{i:>8} {steady_lambdas[i]:>12.4f} {convergence_eps[i]:>10} {int(total_viols[i]):>12}")
    print("=" * 60)
    print(f"Highest λ: Pair {np.argmax(steady_lambdas)} (λ={np.max(steady_lambdas):.4f})")
    print(f"Slowest convergence: Pair {np.argmax(convergence_eps)} (@ep {np.max(convergence_eps)})")
    print(f"Most violations: Pair {np.argmax(total_viols)} ({int(np.max(total_viols))} total)")
    print("=" * 60)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Figure 10 — Lagrange Evolution")
    p.add_argument("--n_episodes", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", type=str, default="outputs")
    args = p.parse_args()
    main(args.n_episodes, args.seed, args.output_dir)
