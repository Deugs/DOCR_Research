#!/usr/bin/env python3
"""CLI Adapter: coordination_diagram.py — Fig 43

WARNING: SYNTHETIC DATA. This script plots hand-typed literal number arrays,
not output from a trained model or a real experiment run. See
experiments/GAP_ANALYSIS.md. Do not cite this figure as experimental evidence
until it is replaced with a real data pipeline."""
import argparse, sys, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from evaluation.experiments._synthetic_label import mark_synthetic

def main():
    p = argparse.ArgumentParser(description="Coordination Diagram Adapter (Fig 43)")
    p.add_argument("--system", type=str, default="8bus")
    p.add_argument("--pv", type=float, default=1.0)
    p.add_argument("--output", type=str, default="outputs/fig43_coordination.pdf")
    args = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 6))

    distance = np.linspace(0, 10, 100)
    t_primary = 0.3 + 0.05 * distance
    t_backup = 0.3 + 0.3 + 0.08 * distance

    ax.fill_between(distance, t_primary, t_backup, alpha=0.2, color="green", label="CTI margin")
    ax.plot(distance, t_primary, "b-", lw=2, label="Primary relay")
    ax.plot(distance, t_backup, "r--", lw=2, label="Backup relay")

    ax.set_xlabel("Electrical Distance from Fault (Ω)", fontsize=11)
    ax.set_ylabel("Operating Time (s)", fontsize=11)
    ax.set_title("Figure 43 — Relay Coordination Time-Distance Diagram", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 2)

    fig.tight_layout()
    mark_synthetic(fig)
    fig.savefig(args.output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[Adapter] Saved: {args.output}")

if __name__ == "__main__":
    main()
