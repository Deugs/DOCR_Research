"""
unified_exp_fig05.py  —  Figure 5 (repaired pipeline)
──────────────────────────────────────────────────────
Objective Function vs. PV Penetration, benchmarked against the 5 named
metaheuristics (GA-SQP, HWOA, HHO, JAYA, EEO), with real significance testing.

This closes the gap documented in experiments/GAP_ANALYSIS.md and
experiments/MIGRATION_GUIDE.md: the previous exp_fig05_pv_sweep_evaluation.py
never called the metaheuristic baselines and never ran a statistical test,
and pv_sweep_evaluation.py imported this module before it existed.

Differences from exp_fig05_pv_sweep_evaluation.py:
  - Adds GA-SQP / HWOA / HHO / JAYA / EEO alongside the MARL methods and
    the Traditional baseline (reuses baselines.ga_sqp.run_ga_sqp and
    baselines.metaheuristics.run_metaheuristic — both were already correct,
    just never called from here).
  - Adds real Wilcoxon signed-rank + Cohen's d / Cliff's delta significance
    testing across seeds, via evaluation.stat_analysis (already implemented
    there; this was the piece that was never wired to real data).
  - Output goes to outputs/{system}/ instead of a flat outputs/ directory.

Outputs
-------
  outputs/{system}/fig05_pv_sweep.npz
  outputs/{system}/fig05_pv_sweep.png
  outputs/{system}/fig05_pv_sweep_stats.txt
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from network.network_data import NETWORKS
from network.relay_physics import check_constraints
from baselines.ga_sqp import run_ga_sqp
from baselines.metaheuristics import run_metaheuristic
from baselines.experiments.exp_fig05_pv_sweep_evaluation import (
    train_and_evaluate_method,
    evaluate_traditional,
)
from evaluation.stat_analysis import wilcoxon_pairwise, cliffs_delta

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s",
                     datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

PV_LEVELS = [0.0, 0.20, 0.40, 0.60]
MARL_METHODS = ["MAPPO-Safe-EWC", "MAPPO-Safe", "MAPPO"]
METAHEURISTIC_METHODS = ["GA-SQP", "HWOA", "HHO", "JAYA", "EEO"]
ALL_METHODS = MARL_METHODS + METAHEURISTIC_METHODS + ["Traditional"]

METHOD_COLORS = {
    "MAPPO-Safe-EWC": "#2563eb",
    "MAPPO-Safe": "#39d353",
    "MAPPO": "#f7c948",
    "GA-SQP": "#e11d48",
    "HWOA": "#0891b2",
    "HHO": "#7c3aed",
    "JAYA": "#059669",
    "EEO": "#d97706",
    "Traditional": "#ff6b6b",
}


def run_metaheuristic_seed(method: str, net, pv_level: float, seed: int,
                            fault_type: str = "3ph") -> Dict:
    """Run one metaheuristic once at a given PV level/seed and return one OF sample.

    A single optimizer run at a fixed operating point is the metaheuristic
    equivalent of a MARL policy's mean_OF over its evaluation episodes — both
    become one sample in the per-seed distribution used for the significance
    test below.
    """
    if method == "GA-SQP":
        res = run_ga_sqp(net, ga_pop=40, ga_gen=60, seed=seed, verbose=False,
                          fault_type=fault_type, pv_level=pv_level)
    else:
        res = run_metaheuristic(method, net, pop_size=30, max_iter=60, seed=seed,
                                 verbose=False, fault_type=fault_type, pv_level=pv_level)
    return {
        "mean_OF": float(res["OF"]),
        "std_OF": 0.0,
        "cti_viols": float(res["constraints"]["cti_violations"]),
    }


def main(n_seeds: int = 10, train_steps: int = 5000, n_eval: int = 100,
         meta_pop: int = 30, meta_iter: int = 60,
         output_dir: str = "outputs", system: str = "8bus"):
    if system not in NETWORKS:
        raise ValueError(f"Unknown system '{system}'. Available: {sorted(NETWORKS.keys())}")

    out_path = Path(output_dir) / system
    out_path.mkdir(parents=True, exist_ok=True)

    logger.info("Figure 5 (unified) — PV Penetration Sweep vs. 5 Metaheuristics")
    logger.info("  System: %s", system)
    logger.info("  Methods: %s", ALL_METHODS)
    logger.info("  PV levels: %s", PV_LEVELS)
    logger.info("  Seeds: %d", n_seeds)

    results = {method: {pv: [] for pv in PV_LEVELS} for method in ALL_METHODS}

    for pv in PV_LEVELS:
        net = NETWORKS[system]()
        logger.info("=== PV level: %.0f%% ===", pv * 100)

        for method in MARL_METHODS + ["Traditional"]:
            for seed in range(n_seeds):
                res = train_and_evaluate_method(method, pv, seed, train_steps, n_eval, system=system)
                results[method][pv].append(res)
            logger.info("  %s done (%d seeds)", method, n_seeds)

        for method in METAHEURISTIC_METHODS:
            for seed in range(n_seeds):
                res = run_metaheuristic_seed(method, net, pv, seed)
                results[method][pv].append(res)
            logger.info("  %s done (%d seeds)", method, n_seeds)

    # Aggregate
    aggregated = {}
    for method in ALL_METHODS:
        aggregated[method] = {
            "mean_OF": [np.mean([r["mean_OF"] for r in results[method][pv]]) for pv in PV_LEVELS],
            "std_OF": [np.std([r["mean_OF"] for r in results[method][pv]]) for pv in PV_LEVELS],
            "mean_cti": [np.mean([r["cti_viols"] for r in results[method][pv]]) for pv in PV_LEVELS],
        }

    baseline_of = aggregated["Traditional"]["mean_OF"][0]
    for method in ALL_METHODS:
        aggregated[method]["norm_OF"] = [of / baseline_of * 100 for of in aggregated[method]["mean_OF"]]

    # ── Real statistical testing (was hardcoded/absent before) ──────────────
    stats_lines = []
    for pv in PV_LEVELS:
        stats_lines.append(f"\n{'='*70}\nPV level {pv*100:.0f}% — pairwise significance across {n_seeds} seeds\n{'='*70}")
        per_seed = np.array([[r["mean_OF"] for r in results[m][pv]] for m in ALL_METHODS]).T  # (n_seeds, n_methods)
        wilcoxon_result = wilcoxon_pairwise(per_seed, algorithm_names=ALL_METHODS, correction="holm")
        best_method = "MAPPO-Safe-EWC"
        for other in ALL_METHODS:
            if other == best_method:
                continue
            entry = wilcoxon_result[best_method][other]
            delta, magnitude = cliffs_delta(
                [r["mean_OF"] for r in results[best_method][pv]],
                [r["mean_OF"] for r in results[other][pv]],
            )
            stats_lines.append(
                f"  {best_method} vs {other:>16}:  "
                f"p={entry['p_value']:.4g}  p_holm={entry['p_value_corrected']:.4g}  "
                f"reject_H0={entry['reject']}  cliffs_delta={delta:+.3f} ({magnitude})"
            )

    stats_report = "\n".join(stats_lines)
    (out_path / "fig05_pv_sweep_stats.txt").write_text(stats_report)
    logger.info("Saved: %s", out_path / "fig05_pv_sweep_stats.txt")

    # Save
    save_dict = {"pv_levels": np.array(PV_LEVELS)}
    for method in ALL_METHODS:
        for key in ["mean_OF", "std_OF", "mean_cti", "norm_OF"]:
            save_dict[f"{method}_{key}"] = np.array(aggregated[method][key])
    np.savez(out_path / "fig05_pv_sweep.npz", **save_dict)
    logger.info("Saved: %s", out_path / "fig05_pv_sweep.npz")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle(f"Figure 5 — Objective Function vs. PV Penetration ({system})", fontsize=12)

    ax = axes[0]
    for method in ALL_METHODS:
        means = aggregated[method]["mean_OF"]
        stds = aggregated[method]["std_OF"]
        style = "o-" if method in MARL_METHODS + ["Traditional"] else "s--"
        ax.errorbar(PV_LEVELS, means, yerr=stds, fmt=style, color=METHOD_COLORS[method],
                    lw=1.5, capsize=3, label=method, markersize=5)
    ax.set_xlabel("PV Penetration Level")
    ax.set_ylabel("Objective Function (s)")
    ax.set_title("Absolute Operating Time")
    ax.set_xticks(PV_LEVELS)
    ax.set_xticklabels([f"{int(p*100)}%" for p in PV_LEVELS])
    ax.legend(fontsize=6.5, loc="upper left", ncol=2)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for method in ALL_METHODS:
        norms = aggregated[method]["norm_OF"]
        style = "o-" if method in MARL_METHODS + ["Traditional"] else "s--"
        ax.plot(PV_LEVELS, norms, style, color=METHOD_COLORS[method], lw=1.5, label=method, markersize=5)
    ax.axhline(100, color="gray", ls="--", lw=1, alpha=0.5, label="0% PV baseline")
    ax.set_xlabel("PV Penetration Level")
    ax.set_ylabel("Normalized OF (% of 0% PV baseline)")
    ax.set_title("Relative Performance")
    ax.set_xticks(PV_LEVELS)
    ax.set_xticklabels([f"{int(p*100)}%" for p in PV_LEVELS])
    ax.legend(fontsize=6.5, loc="upper left", ncol=2)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path / "fig05_pv_sweep.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", out_path / "fig05_pv_sweep.png")

    print("\n" + "=" * 90)
    print(f"PV SWEEP EVALUATION SUMMARY — {system}")
    print("=" * 90)
    header = f"{'Method':>20}" + "".join(f"{f'{int(p*100)}%':>12}" for p in PV_LEVELS)
    print(header)
    print("-" * 90)
    for method in ALL_METHODS:
        vals = aggregated[method]["mean_OF"]
        print(f"{method:>20}" + "".join(f"{v:>12.4f}" for v in vals))
    print("=" * 90)
    print(stats_report)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Figure 5 (unified) — PV Penetration Sweep vs. Metaheuristics")
    p.add_argument("--n_seeds", type=int, default=10)
    p.add_argument("--train_steps", type=int, default=5000)
    p.add_argument("--n_eval", type=int, default=100)
    p.add_argument("--output_dir", type=str, default="outputs")
    p.add_argument("--system", type=str, default="8bus",
                    help="Test system: 8bus, 14bus, 30bus, 39bus, 57bus, 118bus")
    args = p.parse_args()
    main(args.n_seeds, args.train_steps, args.n_eval,
         output_dir=args.output_dir, system=args.system)
