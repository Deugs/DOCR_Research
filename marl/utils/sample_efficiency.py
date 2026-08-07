#!/usr/bin/env python3
"""CLI Adapter: sample_efficiency.py — Fig 41

WARNING: SYNTHETIC DATA. This script plots hand-typed literal number arrays,
not output from a trained model or a real experiment run. See
experiments/GAP_ANALYSIS.md. Do not cite this figure as experimental evidence
until it is replaced with a real data pipeline."""
import argparse, sys, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from evaluation.experiments._synthetic_label import mark_synthetic

def main():
    p = argparse.ArgumentParser(description="Sample Efficiency Adapter (Fig 41)")
    p.add_argument("--data", type=str, default="")
    p.add_argument("--output", type=str, default="outputs/fig41_sample_efficiency.pdf")
    args = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))

    steps = np.logspace(3, 7, 50)

    methods = {
        "MAPPO-Safe-EWC": (0.92, 2e5),
        "MAPPO-Safe": (0.85, 3e5),
        "MAPPO": (0.68, 5e5),
        "Traditional": (0.60, 1e3),
    }
    colors = {"MAPPO-Safe-EWC": "#2563eb", "MAPPO-Safe": "#39d353", 
              "MAPPO": "#a371f7", "Traditional": "#64748b"}

    for method, (final_perf, half_step) in methods.items():
        perf = final_perf * (1 - np.exp(-steps / half_step))
        ax.semilogx(steps, perf, lw=2, color=colors[method], label=method)

    ax.set_xlabel("Environment Steps", fontsize=11)
    ax.set_ylabel("Performance", fontsize=11)
    ax.set_title("Figure 41 — Sample Efficiency Comparison", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    fig.tight_layout()
    mark_synthetic(fig)
    fig.savefig(args.output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[Adapter] Saved: {args.output}")

if __name__ == "__main__":
    main()
