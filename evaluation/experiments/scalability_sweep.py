#!/usr/bin/env python3
"""CLI Adapter: scalability_sweep.py — Fig 15

WARNING: SYNTHETIC DATA. This script plots hand-typed literal number arrays,
not output from a trained model or a real experiment run. See
experiments/GAP_ANALYSIS.md. Do not cite this figure as experimental evidence
until it is replaced with a real data pipeline."""
import argparse, sys, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from evaluation.experiments._synthetic_label import mark_synthetic

def main():
    p = argparse.ArgumentParser(description="Scalability Sweep Adapter (Fig 15)")
    p.add_argument("--systems", nargs="+", type=str, default=["8bus", "14bus", "30bus"])
    p.add_argument("--method", type=str, default="MAPPO-Safe-EWC")
    p.add_argument("--output", type=str, default="outputs/fig15_scalability.pdf")
    args = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    systems = args.systems or ["8bus", "14bus", "30bus"]
    n_relays = [7, 14, 28]
    train_time = [2.5, 8.2, 35.0]

    ax = axes[0]
    x = np.arange(len(systems))
    bars = ax.bar(x, train_time, color="#2563eb", edgecolor="white", width=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(systems)
    ax.set_ylabel("Training Time (hours)", fontsize=11)
    ax.set_title("Training Time vs. System Size", fontsize=12)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, axis="y")

    for bar, t in zip(bars, train_time):
        ax.text(bar.get_x() + bar.get_width()/2, t * 1.1, f"{t}h", 
               ha="center", fontsize=9, fontweight="bold")

    ax = axes[1]
    performance = [0.92, 0.88, 0.82]
    ax.plot(n_relays, performance, "o-", color="#39d353", lw=2, markersize=10)
    for i, sys_name in enumerate(systems):
        ax.annotate(sys_name, (n_relays[i], performance[i]), 
                   textcoords="offset points", xytext=(0, 10), ha="center", fontsize=9)
    ax.set_xlabel("Number of Relays", fontsize=11)
    ax.set_ylabel("Final Performance", fontsize=11)
    ax.set_title("Performance vs. System Size", fontsize=12)
    ax.set_ylim(0.7, 1.0)
    ax.grid(True, alpha=0.3)

    fig.suptitle("Figure 15 — Scalability Analysis Across Test Systems", fontsize=12)
    fig.tight_layout()
    mark_synthetic(fig)
    fig.savefig(args.output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[Adapter] Saved: {args.output}")

if __name__ == "__main__":
    main()
