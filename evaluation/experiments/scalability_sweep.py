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

    # Per-system placeholder values, looked up by name so an arbitrary
    # --systems list (any length, any order) never runs off the end of a
    # hardcoded fixed-length array (previously n_relays/train_time/
    # performance were literal length-3 lists that only matched the
    # 3-system default).
    N_RELAYS = {"8bus": 7, "14bus": 14, "30bus": 28, "39bus": 39, "57bus": 57, "118bus": 118}
    TRAIN_TIME_HOURS = {"8bus": 2.5, "14bus": 8.2, "30bus": 35.0, "39bus": 55.0,
                         "57bus": 90.0, "118bus": 180.0}
    FINAL_PERFORMANCE = {"8bus": 0.92, "14bus": 0.88, "30bus": 0.82, "39bus": 0.80,
                          "57bus": 0.75, "118bus": 0.70}

    n_relays = [N_RELAYS.get(s, 14) for s in systems]
    train_time = [TRAIN_TIME_HOURS.get(s, 10.0) for s in systems]
    performance = [FINAL_PERFORMANCE.get(s, 0.85) for s in systems]

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
