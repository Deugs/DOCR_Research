#!/usr/bin/env python3
"""
run_all_systems.py
──────────────────
Orchestrates Figure 5 experiments across ALL test systems (8/14/39-bus).

Runs unified_exp_fig05.py sequentially for each system, then produces
a combined comparison figure showing scalability trends.

Usage:
    python run_all_systems.py --n_seeds 10 --output_dir outputs

Outputs:
    outputs/8bus/fig05_pv_sweep.npz
    outputs/14bus/fig05_pv_sweep.npz
    outputs/39bus/fig05_pv_sweep.npz
    outputs/combined_scalability.png   ← Figure 15 replacement
"""

import argparse
import sys
import os
from pathlib import Path
import subprocess
import logging

sys.path.insert(0, str(Path(__file__).parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

SYSTEMS = ["8bus", "14bus", "39bus"]

METHOD_COLORS = {
    "MAPPO-Safe-EWC": "#2563eb",
    "MAPPO-Safe": "#39d353",
    "MAPPO": "#f7c948",
    "IPPO": "#ff6b35",
    "QMIX": "#a371f7",
    "IQL": "#8b949e",
    "GA-SQP": "#e11d48",
    "HWOA": "#0891b2",
    "HHO": "#7c3aed",
    "JAYA": "#059669",
    "EEO": "#d97706",
    "Traditional": "#ff6b6b",
}

METHOD_LINESTYLES = {
    "MAPPO-Safe-EWC": "-",
    "MAPPO-Safe": "-",
    "MAPPO": "-",
    "IPPO": "-",
    "QMIX": "-",
    "IQL": "-",
    "GA-SQP": "--",
    "HWOA": "--",
    "HHO": "--",
    "JAYA": "--",
    "EEO": "--",
    "Traditional": ":",
}


def run_system(system: str, n_seeds: int, output_dir: str):
    """Run unified_exp_fig05 for a single system."""
    logger.info("=" * 70)
    logger.info("Running %s", system.upper())
    logger.info("=" * 70)

    # Import and call directly (avoids subprocess overhead)
    from baselines.experiments.unified_exp_fig05 import main as fig05_main

    fig05_main(
        system=system,
        n_seeds=n_seeds,
        train_steps=5000,
        n_eval=100,
        output_dir=output_dir,
    )

    logger.info("%s complete.", system.upper())


def plot_combined(output_dir: str):
    """Generate combined scalability comparison across all systems."""
    out_path = Path(output_dir)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        "Scalability Analysis: Objective Function vs. PV Penetration",
        fontsize=13, fontweight="bold"
    )

    for idx, system in enumerate(SYSTEMS):
        ax = axes[idx]
        data_path = out_path / system / "fig05_pv_sweep.npz"

        if not data_path.exists():
            logger.warning("Data not found for %s: %s", system, data_path)
            ax.text(0.5, 0.5, f"{system}\n(data missing)",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=12, color="red")
            ax.set_title(system.upper())
            continue

        data = np.load(data_path)
        pv_levels = data["pv_levels"]

        # Plot key methods
        key_methods = [
            "MAPPO-Safe-EWC", "MAPPO-Safe", "MAPPO",
            "GA-SQP", "HWOA", "Traditional"
        ]

        for method in key_methods:
            key = f"{method}_mean_OF"
            if key in data:
                means = data[key]
                std_key = f"{method}_std_OF"
                stds = data[std_key] if std_key in data else np.zeros_like(means)
                ax.errorbar(
                    pv_levels, means, yerr=stds,
                    fmt="o-", color=METHOD_COLORS.get(method, "#333"),
                    linestyle=METHOD_LINESTYLES.get(method, "-"),
                    lw=2, capsize=3, label=method, markersize=5,
                )

        ax.set_xlabel("PV Penetration Level", fontsize=10)
        ax.set_ylabel("Objective Function (s)", fontsize=10)
        ax.set_title(f"{system.upper()}", fontsize=11, fontweight="bold")
        ax.set_xticks(pv_levels)
        ax.set_xticklabels([f"{int(p*100)}%" for p in pv_levels])
        ax.legend(fontsize=7, loc="upper left")
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    combined_path = out_path / "combined_scalability.png"
    fig.savefig(combined_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved combined figure: %s", combined_path)


def main():
    p = argparse.ArgumentParser(
        description="Orchestrate Figure 5 across all test systems"
    )
    p.add_argument(
        "--n_seeds", type=int, default=10,
        help="Number of seeds per method-PV combo"
    )
    p.add_argument(
        "--output_dir", type=str, default="outputs",
        help="Root output directory"
    )
    p.add_argument(
        "--systems", type=str, nargs="+", default=SYSTEMS,
        choices=SYSTEMS,
        help="Systems to run (default: all)"
    )
    p.add_argument(
        "--skip_run", action="store_true",
        help="Skip experiment runs, only generate combined plot"
    )
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if not args.skip_run:
        for system in args.systems:
            run_system(system, args.n_seeds, args.output_dir)

    plot_combined(args.output_dir)
    logger.info("All done. Results in: %s", args.output_dir)


if __name__ == "__main__":
    main()