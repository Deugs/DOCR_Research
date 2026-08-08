
"""
exp_fig41_sample_efficiency.py — Figure 41
─────────────────────────────────────────────
Sample Efficiency Comparison.

Performance vs. total environment steps (not episodes) for all methods.
Computes area under curve (AUC) as sample efficiency metric.

Compares: MAPPO-Safe-EWC reaches 95% final performance in X steps vs. baselines.

Outputs
-------
  outputs/fig41_sample_efficiency.npz
  outputs/fig41_sample_efficiency.png
"""

import numpy as np
import os
import sys
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
os.makedirs("outputs", exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from network.network_data import get_8bus
from marl.marl_env import DynamicDOCREnv
from marl.marl_agents import MAPPOMARLTrainer
from network.relay_physics import check_constraints

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

METHODS = {
    "MAPPO-Safe-EWC": {"use_comm": False, "use_safe": True, "use_ewc": True, "color": "#2563eb"},
    "MAPPO-Safe": {"use_comm": False, "use_safe": True, "use_ewc": False, "color": "#39d353"},
    "MAPPO": {"use_comm": False, "use_safe": False, "use_ewc": False, "color": "#f7c948"},
    "IPPO": {"use_comm": False, "use_safe": False, "use_ewc": False, "decentralized": True, "color": "#ff6b35"},
    "GA-SQP": {"classical": True, "color": "#ff6b6b"},
}


def run_method_with_step_logging(method_name: str, config: dict, net,
                                 max_steps: int = 30000, seed: int = 42,
                                 log_interval: int = 500) -> Tuple[np.ndarray, np.ndarray]:
    """
    Train a method and log performance at regular step intervals.
    Returns (steps_array, of_array).
    """
    env = DynamicDOCREnv(net, use_pandapower=False, fault_type="3ph", curriculum_stage=5)
    rng = np.random.default_rng(seed)

    if config.get("classical"):
        # Classical baseline: run once, constant performance
        from baselines.ga_sqp import run_ga_sqp
        result = run_ga_sqp(net, ga_pop=50, ga_gen=100, seed=seed, verbose=False)
        of_final = result["OF"]
        steps = np.arange(0, max_steps + 1, log_interval)
        ofs = np.full_like(steps, of_final, dtype=float)
        return steps, ofs

    # MARL methods
    trainer = MAPPOMARLTrainer(env, use_comm=config.get("use_comm", False),
                                hidden_actor=64, hidden_critic=128, seed=seed)

    # Configure safety and EWC
    if config.get("use_safe") and hasattr(trainer, 'use_safety'):
        trainer.use_safety = True
    if config.get("use_ewc") and hasattr(trainer, 'use_ewc'):
        trainer.use_ewc = True

    steps_log = []
    of_log = []
    cumulative_steps = 0

    # Train in chunks to log intermediate performance
    chunk_size = log_interval
    n_chunks = max_steps // chunk_size

    for chunk in range(n_chunks):
        trainer.train(total_steps=chunk_size, verbose=False)
        cumulative_steps += chunk_size

        # Evaluate current policy
        of_eval = _evaluate_policy(trainer, env, n_episodes=10, seed=seed + chunk)
        steps_log.append(cumulative_steps)
        of_log.append(of_eval)

        if chunk % 10 == 0:
            logger.info("  %s @ step %d: OF=%.4f", method_name, cumulative_steps, of_eval)

    return np.array(steps_log), np.array(of_log)


def _evaluate_policy(trainer, env, n_episodes: int = 10, seed: int = 0) -> float:
    """Quick evaluation of current policy."""
    rng = np.random.default_rng(seed)
    ofs = []
    for ep in range(n_episodes):
        obs = env.reset(seed=seed + ep)
        done = False
        ep_of = []
        while not done:
            actions = np.zeros((env.n_agents, env.ACT_DIM))
            for i in range(env.n_agents):
                a, _ = trainer.actor.sample(obs[i], rng)
                actions[i] = a
            obs, _, done, info = env.step(actions)
            ep_of.append(info.get("OF", float("inf")))
        ofs.append(float(np.min(ep_of)))
    return float(np.mean(ofs))


def compute_auc(steps: np.ndarray, ofs: np.ndarray, normalize: bool = True) -> float:
    """Compute area under the learning curve (lower OF = better)."""
    if normalize:
        # Normalize to [0, 1] where 1 = best (lowest OF)
        of_min = np.min(ofs)
        of_max = np.max(ofs)
        if of_max > of_min:
            ofs_norm = 1.0 - (ofs - of_min) / (of_max - of_min)
        else:
            ofs_norm = np.ones_like(ofs)
    else:
        ofs_norm = -ofs  # Negative because lower OF is better

    # Trapezoidal integration
    auc = np.trapz(ofs_norm, steps)
    return auc


def compute_steps_to_threshold(steps: np.ndarray, ofs: np.ndarray,
                                threshold_fraction: float = 0.95) -> int:
    """Compute steps to reach threshold_fraction of final performance."""
    final_of = ofs[-1]
    initial_of = ofs[0]
    target_of = initial_of - threshold_fraction * (initial_of - final_of)

    # Find first step where OF <= target
    above_target = ofs <= target_of
    if np.any(above_target):
        idx = np.where(above_target)[0][0]
        return int(steps[idx])
    return int(steps[-1])  # Never reached


def main(max_steps: int = 20000, n_seeds: int = 3, seed: int = 42,
         output_dir: str = "outputs"):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    net = get_8bus()

    logger.info("Figure 41 — Sample Efficiency Comparison")
    logger.info("  Max steps: %d, Seeds: %d", max_steps, n_seeds)

    all_results = {}

    for method_name, config in METHODS.items():
        logger.info("\\nMethod: %s", method_name)
        seed_results = []
        for s in range(n_seeds):
            steps, ofs = run_method_with_step_logging(
                method_name, config, net,
                max_steps=max_steps, seed=seed + s * 1000
            )
            seed_results.append((steps, ofs))
            logger.info("  Seed %d: final OF=%.4f, steps to 95%%=%d",
                        s, ofs[-1], compute_steps_to_threshold(steps, ofs))

        # Aggregate across seeds
        # Interpolate to common step grid
        common_steps = seed_results[0][0]
        ofs_matrix = np.array([np.interp(common_steps, steps, ofs) for steps, ofs in seed_results])
        mean_of = ofs_matrix.mean(axis=0)
        std_of = ofs_matrix.std(axis=0)

        auc = compute_auc(common_steps, mean_of)
        steps_to_95 = compute_steps_to_threshold(common_steps, mean_of)

        all_results[method_name] = {
            "steps": common_steps,
            "mean_of": mean_of,
            "std_of": std_of,
            "auc": auc,
            "steps_to_95": steps_to_95,
            "color": config["color"],
        }

    # ── Save ─────────────────────────────────────────────────────────────────
    save_dict = {}
    for method, res in all_results.items():
        save_dict[f"{method}_steps"] = res["steps"]
        save_dict[f"{method}_mean"] = res["mean_of"]
        save_dict[f"{method}_std"] = res["std_of"]
        save_dict[f"{method}_auc"] = np.array([res["auc"]])
        save_dict[f"{method}_steps95"] = np.array([res["steps_to_95"]])

    np.savez(out_path / "fig41_sample_efficiency.npz", **save_dict)
    logger.info("Saved: %s", out_path / "fig41_sample_efficiency.npz")

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Figure 41 — Sample Efficiency Comparison", fontsize=12)

    # Left: Learning curves
    ax = axes[0]
    for method, res in all_results.items():
        ax.plot(res["steps"], res["mean_of"], color=res["color"], lw=2,
                label=f"{method}")
        ax.fill_between(res["steps"],
                        res["mean_of"] - res["std_of"],
                        res["mean_of"] + res["std_of"],
                        color=res["color"], alpha=0.15)

    ax.set_xlabel("Environment Steps", fontsize=10)
    ax.set_ylabel("Objective Function (s)", fontsize=10)
    ax.set_title("Performance vs. Steps", fontsize=11)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)

    # Right: AUC and steps-to-95% comparison
    ax = axes[1]
    methods = list(all_results.keys())
    aucs = [all_results[m]["auc"] for m in methods]
    steps95 = [all_results[m]["steps_to_95"] for m in methods]
    colors = [all_results[m]["color"] for m in methods]

    x = np.arange(len(methods))
    width = 0.35

    bars1 = ax.bar(x - width / 2, aucs, width, label="AUC (normalized)",
                   color=colors, edgecolor="white", alpha=0.85)
    ax2 = ax.twinx()
    bars2 = ax2.bar(x + width / 2, steps95, width, label="Steps to 95%",
                    color=colors, edgecolor="white", alpha=0.5, hatch="//")

    ax.set_ylabel("AUC (higher = better)", fontsize=10)
    ax2.set_ylabel("Steps to 95% (lower = better)", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha="right", fontsize=8)
    ax.set_title("Efficiency Metrics", fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")

    # Add value labels
    for bar, val in zip(bars1, aucs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(aucs) * 0.01,
                f"{val:.0f}", ha="center", va="bottom", fontsize=7)
    for bar, val in zip(bars2, steps95):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(steps95) * 0.01,
                 f"{val:.0f}", ha="center", va="bottom", fontsize=7)

    fig.tight_layout()
    fig.savefig(out_path / "fig41_sample_efficiency.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", out_path / "fig41_sample_efficiency.png")

    # Summary table
    print("\\n" + "=" * 70)
    print("SAMPLE EFFICIENCY SUMMARY")
    print("=" * 70)
    print(f"{'Method':<20} {'AUC':>12} {'Steps to 95%':>15} {'Final OF':>12}")
    print("-" * 70)
    for method in methods:
        res = all_results[method]
        print(f"{method:<20} {res['auc']:>12.0f} {res['steps_to_95']:>15d} {res['mean_of'][-1]:>12.4f}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Figure 41 — Sample Efficiency")
    parser.add_argument("--max_steps", type=int, default=20000)
    parser.add_argument("--n_seeds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="outputs")
    args = parser.parse_args()
    main(args.max_steps, args.n_seeds, args.seed, args.output_dir)
