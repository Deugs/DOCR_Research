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
import multiprocessing as mp
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
os.makedirs("outputs", exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import threadpoolctl

from network.network_data import NETWORKS
from marl.marl_env import DynamicDOCREnv
from marl.marl_agents import MAPPOMARLTrainer
from network.relay_physics import check_constraints

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

PV_LEVELS = [0.0, 0.20, 0.40, 0.60]
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
                               n_eval: int = 100,
                               system: str = "8bus") -> Dict:
    """Train a method at a specific PV level and evaluate."""
    if system not in NETWORKS:
        raise ValueError(f"Unknown system '{system}'. Available: {sorted(NETWORKS.keys())}")
    net = NETWORKS[system]()
    
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

    # Evaluate using the correct observation dimensionality.
    # When use_comm=True the actor was trained on comm_graph.forward()
    # output (obs_dim + msg_dim = 18-dim), NOT raw env observations
    # (10-dim). Calling trainer.actor.sample(obs[i], rng) directly
    # therefore raised ValueError on every comm-enabled method row.
    rng = np.random.default_rng(seed + 999)
    of_list, cti_list = [], []

    def _sample_action(all_obs, agent_idx):
        """Sample action, applying comm-graph enhancement when needed."""
        if use_comm and trainer.comm_graph is not None:
            enhanced, _, _ = trainer.comm_graph.forward(all_obs, rng)
            return trainer.actor.sample(enhanced[agent_idx], rng)
        return trainer.actor.sample(all_obs[agent_idx], rng)

    for ep in range(n_eval):
        obs = env.reset(seed=seed + 1000 + ep)
        done = False
        ep_of = []
        while not done:
            actions = np.zeros((env.n_agents, env.ACT_DIM))
            for i in range(env.n_agents):
                a, _ = _sample_action(obs, i)
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
        env.set_state(fixed_TDS, fixed_PSM)
        
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


def _pool_worker_init():
    threadpoolctl.threadpool_limits(1)


def _run_job(args):
    method, pv, seed, train_steps, n_eval, system = args
    res = train_and_evaluate_method(method, pv, seed, train_steps, n_eval, system=system)
    return method, pv, seed, res


def main(n_seeds: int = 10, train_steps: int = 5000, n_eval: int = 100,
         output_dir: str = "outputs", system: str = "8bus", n_workers: int = 1):
    # Applied unconditionally — see unified_exp_fig05.py for why: unpinned
    # BLAS threads oversubscribe the machine under multiprocessing and can
    # make n_workers>1 slower than sequential, and this also makes
    # single-process results reproducible regardless of the machine's
    # ambient BLAS thread configuration.
    threadpoolctl.threadpool_limits(1)

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    logger.info("Figure 5 — PV Penetration Sweep Evaluation")
    logger.info("  System: %s", system)
    logger.info("  Methods: %s", METHODS)
    logger.info("  PV levels: %s", PV_LEVELS)
    logger.info("  Seeds: %d", n_seeds)

    jobs = [
        (method, pv, seed, train_steps, n_eval, system)
        for method in METHODS for pv in PV_LEVELS for seed in range(n_seeds)
    ]
    logger.info("Dispatching %d jobs across %d worker(s)", len(jobs), n_workers)

    # Pre-allocated by seed index rather than appended in completion order —
    # this file's aggregation (mean/std over seeds) is order-independent,
    # but indexing costs nothing and avoids relying on that staying true.
    results = {method: {pv: [None] * n_seeds for pv in PV_LEVELS} for method in METHODS}

    if n_workers > 1:
        with mp.Pool(n_workers, initializer=_pool_worker_init) as pool:
            for method, pv, seed, res in pool.imap_unordered(_run_job, jobs):
                results[method][pv][seed] = res
                logger.info("  [done] %s PV=%.0f%% seed=%d: OF=%.4f±%.4f CTI=%.1f",
                            method, pv * 100, seed, res["mean_OF"], res["std_OF"], res["cti_viols"])
    else:
        for job in jobs:
            method, pv, seed, res = _run_job(job)
            results[method][pv][seed] = res
            logger.info("  [done] %s PV=%.0f%% seed=%d: OF=%.4f±%.4f CTI=%.1f",
                        method, pv * 100, seed, res["mean_OF"], res["std_OF"], res["cti_viols"])
    
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
    print("\n" + "=" * 80)
    print("PV SWEEP EVALUATION SUMMARY")
    print("=" * 80)
    header = f"{'Method':>20}" + "".join(f"{f'{int(p*100)}%':>10}" for p in PV_LEVELS)
    print(header)
    print("-" * 80)
    for method in METHODS:
        vals = aggregated[method]["mean_OF"]
        print(f"{method:>20}" + "".join(f"{v:>10.4f}" for v in vals))
    print("=" * 80)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Figure 5 — PV Penetration Sweep")
    p.add_argument("--n_seeds", type=int, default=10)
    p.add_argument("--train_steps", type=int, default=5000)
    p.add_argument("--n_eval", type=int, default=100)
    p.add_argument("--output_dir", type=str, default="outputs")
    p.add_argument("--system", type=str, default="8bus",
                   help="Test system: 8bus, 14bus, 39bus, etc.")
    p.add_argument("--n_workers", type=int, default=1,
                   help="Parallel worker processes (default: 1 = sequential)")
    args = p.parse_args()
    main(args.n_seeds, args.train_steps, args.n_eval, args.output_dir, args.system, args.n_workers)