#!/usr/bin/env python3
"""
unified_exp_fig06_cti_violations.py  —  Figure 6 (Fixed)
───────────────────────────────────────────────────────────
CTI Violations vs. PV Penetration — 8/30/39-Bus Systems.

Extends Fig 5 data to show CTI violation rates across all methods.
Can be run standalone or consume Fig 5 NPZ outputs.

Usage:
    python unified_exp_fig06_cti_violations.py --system 8bus --output_dir outputs
    python unified_exp_fig06_cti_violations.py --system 30bus --output_dir outputs
    python unified_exp_fig06_cti_violations.py --system 39bus --output_dir outputs
"""

import numpy as np
import os
import sys
import argparse
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PV_LEVELS = [0.0, 0.25, 0.50, 0.75, 1.0]

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

ALL_METHODS = list(METHOD_COLORS.keys())


def plot_cti_violations(system: str, output_dir: str):
    """Generate CTI violation plot from Fig 5 NPZ data."""
    data_path = Path(output_dir) / system / "fig05_pv_sweep.npz"
    if not data_path.exists():
        raise FileNotFoundError(
            f"Fig 5 data not found: {data_path}. Run unified_exp_fig05 first."
        )

    data = np.load(data_path)
    pv_levels = data["pv_levels"]

    fig, ax = plt.subplots(figsize=(10, 6))

    for method in ALL_METHODS:
        key = f"{method}_mean_cti"
        if key in data:
            cti_vals = data[key]
            ax.plot(
                pv_levels, cti_vals, "o-",
                color=METHOD_COLORS[method],
                linestyle=METHOD_LINESTYLES[method],
                lw=2, label=method, markersize=6,
            )

    ax.axhline(
        y=0, color="green", ls="--", lw=1.5, alpha=0.7,
        label="Zero violations (ideal)"
    )
    ax.set_xlabel("PV Penetration Level", fontsize=11)
    ax.set_ylabel("Mean CTI Violations per Episode", fontsize=11)
    ax.set_title(
        f"Figure 6 — CTI Violations vs. PV Penetration ({system.upper()})",
        fontsize=12, fontweight="bold"
    )
    ax.set_xticks(pv_levels)
    ax.set_xticklabels([f"{int(p*100)}%" for p in pv_levels])
    ax.legend(fontsize=7, loc="upper left", ncol=2)
    ax.grid(True, alpha=0.3)

    out_path = Path(output_dir) / system / "fig06_cti_violations.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")

    # Also save a table
    table_path = Path(output_dir) / system / "fig06_cti_violations.txt"
    with open(table_path, "w") as f:
        f.write(f"CTI Violations Summary — {system.upper()}\n")
        f.write("=" * 80 + "\n")
        hdr = f"{'Method':>18}"
        for pv in pv_levels:
            hdr += f" {int(pv*100):>9}"
        f.write(hdr + "\n")
        f.write("-" * 80 + "\n")
        for method in ALL_METHODS:
            key = f"{method}_mean_cti"
            if key in data:
                vals = data[key]
                row = f"{method:>18}"
                for v in vals:
                    row += f" {v:>9.2f}"
                f.write(row + "\n")
        f.write("=" * 80 + "\n")
    print(f"Saved: {table_path}")


def main():
    p = argparse.ArgumentParser(description="Figure 6 — CTI Violations (Fixed)")
    p.add_argument(
        "--system", type=str, required=True,
        choices=["8bus", "30bus", "39bus"],
        help="Test system"
    )
    p.add_argument(
        "--output_dir", type=str, default="outputs",
        help="Root output directory (contains system subdirs)"
    )
    args = p.parse_args()
    plot_cti_violations(args.system, args.output_dir)


if __name__ == "__main__":
    main()
