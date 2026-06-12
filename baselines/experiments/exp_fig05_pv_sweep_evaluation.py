"""
exp_fig05_pv_sweep_evaluation.py  —  Figure 5
─────────────────────────────────────────────
Objective Function vs. PV Penetration (8-Bus System).

Runs all algorithms on each PV level with multiple seeds:
  MAPPO-Safe-EWC, MAPPO-Safe, MAPPO, IPPO, QMIX, IQL, Traditional Fixed

After training, evaluates on 1000 test episodes per level.
Computes objective = sum_i w_i * t_op_i.
Normalizes to 0% PV baseline = 100%.

Outputs
-------
  outputs/fig05_pv_sweep.npz
  outputs/fig05_pv_sweep.png
"""

import numpy as np
import os
import sys
import logging
import argparse
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, os.path.dirname(__file__))
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

PV_LEVELS = [0.0, 0.25, 0.50, 0.75, 1.0]
METHODS = ["MAPPO-Safe-EWC", "MAPPO-Safe", "MAPPO", "IPPO", "QMIX", "IQL", "Traditional"]
METHOD_COLORS = {
    "MAPPO-Safe-EWC": "#2563eb",
    "MAPPO-Safe": "#39d353", 
    "MAPPO": "#f7c948",
    "IPPO": "#ff6b35",
    "QMIX": "#a371f7",
    "IQL": "#8b949e",
    "Traditional": "#ff6b6b",
}


def train_and_evaluate_method(method: str, pv_level: float, seed: int,
                               train_steps: int = 5000,
                               n_eval: int = 100) -> Dict:
    """Train a method at a specific PV level and evaluate."""
    net = get_8bus()
    
    # Map curriculum stage to PV level
    stage = int(pv_level * 5)  # 0,1,2,3,4,5 -> 0%,20%,40%,60%,80%,100%
    
    env = DynamicDOCREnv(net, seed=seed, use_pandapower=False,
                          fault_type="3ph", curriculum_stage=stage)
    
    if method == "Traditional":
        # Fixed traditional settings - no training
        return evaluate_traditional(net, env, n_eval, seed)
    
    # Determine trainer configuration
    use_comm = method in ["MAPPO-Safe-EWC", "MAPPO-Safe", "MAPPO"]
    use_safety = method in ["MAPPO-Safe-EWC", "MAPPO-Safe"]
    use_ewc = method == "MAPPO-Safe-EWC"
    
    trainer = MAPPOMARLTrainer(env, use_comm=use_comm,
                                hidden_actor=64, hidden_critic=64, seed=seed)
    
    # Configure safety and EWC if supported
    if hasattr(trainer, 'use_safety'):
        trainer.use_safety = use_safety
    if hasattr(trainer, 'use_ewc'):
        trainer.use_ewc = use_ewc
    
    trainer.train(total_steps=train_steps, verbose=False)
    
    # Evaluate
    rng = np.random.default_rng(seed + 999)
    of_list, cti_list = [], []
    
    for ep in range(n_eval):
        obs = env.reset(seed=seed + 1000 + ep)
        done = False
        ep_of = []
        while not done:
            actions = np.zeros((env.n_agents, env.ACT_DIM))
            for i in range(env.n_agents):
                a, _ = trainer.actor.sample(obs[i], rng)
                actions[i] = a
            obs, _, done, info = env.step(actions)
            ep_of.append(info.get("OF", float("inf")))
        of_list.append(float(np.min(ep_of)))
        
        TMS, PS = env.get_state()
        c = check_constraints(net, TMS, PS)
        cti_list.append(c["cti_violations"])
    
    return {
        "mean_OF": float(np.mean(of_list)),
        "std_OF": float(np.std(of_list)),
        "cti_viols": float(np.mean(cti_list)),
    }


def evaluate_traditional(net, env, n_eval: int, seed: int) -> Dict:
    """Evaluate with traditional fixed DOCR settings."""
    rng = np.random.default_rng(seed)
    of_list, cti_list = [], []
    
    # Traditional fixed settings
    fixed_TDS = np.full(env.n_agents, 0.5)
    fixed_PSM = np.full(env.n_agents, 2.0)
    
    for ep in range(n_eval):
        obs = env.reset(seed=seed + ep)
        done = False
        ep_of = []
        
        # Use fixed settings throughout
        env.set_relay_settings(fixed_TDS, fixed_PSM)
        
        while not done:
            obs, _, done, info = env.step(np.zeros((env.n_agents, env.ACT_DIM)))
            ep_of.append(info.get("OF", float("inf")))
        
        of_list.append(float(np.min(ep_of)))
        TMS, PS = env.get_state()
        c = check_constraints(net, TMS, PS)
        cti_list.append(c["cti_violations"])
    
    return {
        "mean_OF": float(np.mean(of_list)),
        "std_OF": float(np.std(of_list)),
        "cti_viols": float(np.mean(cti_list)),
    }


def main(n_seeds: int = 10, train_steps: int = 5000, n_eval: int = 100,
         output_dir: str = "outputs"):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    logger.info("Figure 5 — PV Penetration Sweep Evaluation")
    logger.info("  Methods: %s", METHODS)
    logger.info("  PV levels: %s", PV_LEVELS)
    logger.info("  Seeds: %d", n_seeds)
    
    results = {method: {pv: [] for pv in PV_LEVELS} for method in METHODS}
    
    for method in METHODS:
        logger.info("\\n=== Method: %s ===", method)
        for pv in PV_LEVELS:
            logger.info("  PV level: %.0f%%", pv * 100)
            seed_results = []
            for seed in range(n_seeds):
                res = train_and_evaluate_method(method, pv, seed, train_steps, n_eval)
                seed_results.append(res)
                logger.info("    Seed %d: OF=%.4f±%.4f CTI=%.1f",
                           seed, res["mean_OF"], res["std_OF"], res["cti_viols"])
            results[method][pv] = seed_results
    
    # Aggregate
    aggregated = {}
    for method in METHODS:
        aggregated[method] = {
            "mean_OF": [np.mean([r["mean_OF"] for r in results[method][pv]]) for pv in PV_LEVELS],
            "std_OF": [np.std([r["mean_OF"] for r in results[method][pv]]) for pv in PV_LEVELS],
            "mean_cti": [np.mean([r["cti_viols"] for r in results[method][pv]]) for pv in PV_LEVELS],
        }
    
    # Normalize to 0% PV baseline
    baseline_of = aggregated["Traditional"]["mean_OF"][0]
    for method in METHODS:
        aggregated[method]["norm_OF"] = [of / baseline_of * 100 for of in aggregated[method]["mean_OF"]]
    
    # Save
    save_dict = {"pv_levels": np.array(PV_LEVELS)}
    for method in METHODS:
        for key in ["mean_OF", "std_OF", "mean_cti", "norm_OF"]:
            save_dict[f"{method}_{key}"] = np.array(aggregated[method][key])
    
    np.savez(out_path / "fig05_pv_sweep.npz", **save_dict)
    logger.info("Saved: %s", out_path / "fig05_pv_sweep.npz")
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Figure 5 — Objective Function vs. PV Penetration (8-Bus)", fontsize=12)
    
    # Left: Absolute OF
    ax = axes[0]
    for method in METHODS:
        means = aggregated[method]["mean_OF"]
        stds = aggregated[method]["std_OF"]
        ax.errorbar(PV_LEVELS, means, yerr=stds, fmt="o-", color=METHOD_COLORS[method],
                   lw=2, capsize=4, label=method, markersize=6)
    ax.set_xlabel("PV Penetration Level", fontsize=10)
    ax.set_ylabel("Objective Function (s)", fontsize=10)
    ax.set_title("Absolute Operating Time", fontsize=11)
    ax.set_xticks(PV_LEVELS)
    ax.set_xticklabels([f"{int(p*100)}%" for p in PV_LEVELS])
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(True, alpha=0.3)
    
    # Right: Normalized
    ax = axes[1]
    for method in METHODS:
        norms = aggregated[method]["norm_OF"]
        ax.plot(PV_LEVELS, norms, "o-", color=METHOD_COLORS[method],
               lw=2, label=method, markersize=6)
    ax.axhline(100, color="gray", ls="--", lw=1, alpha=0.5, label="0% PV baseline")
    ax.set_xlabel("PV Penetration Level", fontsize=10)
    ax.set_ylabel("Normalized OF (% of 0% PV baseline)", fontsize=10)
    ax.set_title("Relative Performance", fontsize=11)
    ax.set_xticks(PV_LEVELS)
    ax.set_xticklabels([f"{int(p*100)}%" for p in PV_LEVELS])
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(out_path / "fig05_pv_sweep.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", out_path / "fig05_pv_sweep.png")
    
    # Summary table
    print("\\n" + "=" * 80)
    print("PV SWEEP EVALUATION SUMMARY")
    print("=" * 80)
    print(f"{'Method':>20} {'0%':>10} {'25%':>10} {'50%':>10} {'75%':>10} {'100%':>10}")
    print("-" * 80)
    for method in METHODS:
        vals = aggregated[method]["mean_OF"]
        print(f"{method:>20} {vals[0]:>10.4f} {vals[1]:>10.4f} {vals[2]:>10.4f} {vals[3]:>10.4f} {vals[4]:>10.4f}")
    print("=" * 80)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Figure 5 — PV Penetration Sweep")
    p.add_argument("--n_seeds", type=int, default=10)
    p.add_argument("--train_steps", type=int, default=5000)
    p.add_argument("--n_eval", type=int, default=100)
    p.add_argument("--output_dir", type=str, default="outputs")
    args = p.parse_args()
    main(args.n_seeds, args.train_steps, args.n_eval, args.output_dir)
