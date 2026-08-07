#!/usr/bin/env python3
"""CLI Adapter: unseen_faults.py — Fig 23

WARNING: SYNTHETIC DATA. This script plots hand-typed literal number arrays,
not output from a trained model or a real experiment run. See
experiments/GAP_ANALYSIS.md. Do not cite this figure as experimental evidence
until it is replaced with a real data pipeline."""
import argparse, sys, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from evaluation.experiments._synthetic_label import mark_synthetic

def main():
    p = argparse.ArgumentParser(description="Unseen Faults Adapter (Fig 23)")
    p.add_argument("--system", type=str, default="8bus")
    p.add_argument("--pv", type=float, default=1.0)
    p.add_argument("--fault-types", nargs="+", type=str, default=["bolted", "high-impedance", "evolving", "simultaneous"])
    p.add_argument("--n_eval", type=int, default=100)
    p.add_argument("--output", type=str, default="outputs/fig23_unseen_faults.pdf")
    args = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))

    fault_types = args.fault_types or ["bolted", "high-impedance", "evolving", "simultaneous"]
    seen_perf = [0.92, 0.75, 0.68, 0.55]
    colors = ["#2563eb", "#f59e0b", "#ef4444", "#a371f7"]

    bars = ax.bar(fault_types, seen_perf, color=colors, edgecolor="white", width=0.6)
    ax.axhline(y=0.92, color="green", linestyle="--", alpha=0.7, label="Training distribution")
    ax.set_ylabel("Performance", fontsize=11)
    ax.set_xlabel("Fault Type", fontsize=11)
    ax.set_title("Figure 23 — Generalization to Unseen Fault Types", fontsize=12)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    for bar, p in zip(bars, seen_perf):
        ax.text(bar.get_x() + bar.get_width()/2, p + 0.02, f"{p:.2f}", 
               ha="center", fontsize=10, fontweight="bold")

    fig.tight_layout()
    mark_synthetic(fig)
    fig.savefig(args.output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[Adapter] Saved: {args.output}")

if __name__ == "__main__":
    main()
