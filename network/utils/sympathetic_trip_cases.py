#!/usr/bin/env python3
"""CLI Adapter: sympathetic_trip_cases.py — Fig 47

WARNING: SYNTHETIC DATA. This script plots hand-typed literal number arrays,
not output from a trained model or a real experiment run. See
experiments/GAP_ANALYSIS.md. Do not cite this figure as experimental evidence
until it is replaced with a real data pipeline."""
import argparse, sys, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from evaluation.experiments._synthetic_label import mark_synthetic

def main():
    p = argparse.ArgumentParser(description="Sympathetic Tripping Adapter (Fig 47)")
    p.add_argument("--system", type=str, default="8bus")
    p.add_argument("--pv", type=float, default=1.0)
    p.add_argument("--output", type=str, default="outputs/fig47_sympathetic.pdf")
    args = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    cases = ["Normal", "PV fault\ninjection", "Reverse\nflow", "Islanding\ndetect"]
    trips_without = [2, 8, 6, 5]
    colors = ["#ef4444" if t > 3 else "#f59e0b" for t in trips_without]
    bars = ax.bar(cases, trips_without, color=colors, edgecolor="white", width=0.6)
    ax.set_ylabel("Unnecessary Trips", fontsize=11)
    ax.set_title("Without Sympathetic Trip Prevention", fontsize=12)
    ax.grid(True, alpha=0.3, axis="y")

    ax = axes[1]
    trips_with = [2, 2, 2, 2]
    bars = ax.bar(cases, trips_with, color="#2563eb", edgecolor="white", width=0.6)
    ax.set_ylabel("Unnecessary Trips", fontsize=11)
    ax.set_title("With Sympathetic Trip Prevention (MAPPO-Safe-EWC)", fontsize=12)
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Figure 47 — Sympathetic Tripping Prevention Analysis", fontsize=12)
    fig.tight_layout()
    mark_synthetic(fig)
    fig.savefig(args.output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[Adapter] Saved: {args.output}")

if __name__ == "__main__":
    main()
