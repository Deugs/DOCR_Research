#!/usr/bin/env python3
"""CLI Adapter: seasonal_generalization.py — Fig 58

WARNING: SYNTHETIC DATA. This script plots hand-typed literal number arrays,
not output from a trained model or a real experiment run. See
experiments/GAP_ANALYSIS.md. Do not cite this figure as experimental evidence
until it is replaced with a real data pipeline."""
import argparse, sys, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from evaluation.experiments._synthetic_label import mark_synthetic

def main():
    p = argparse.ArgumentParser(description="Seasonal Generalization Adapter (Fig 58)")
    p.add_argument("--system", type=str, default="8bus")
    p.add_argument("--seasons", nargs="+", type=str, default=["spring", "summer", "autumn", "winter"])
    p.add_argument("--n_eval", type=int, default=100)
    p.add_argument("--output", type=str, default="outputs/fig58_seasonal.pdf")
    args = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))

    seasons = args.seasons or ["spring", "summer", "autumn", "winter"]
    performance = [0.90, 0.88, 0.91, 0.85]
    colors = ["#39d353", "#f59e0b", "#a371f7", "#2563eb"]

    bars = ax.bar(seasons, performance, color=colors, edgecolor="white", width=0.6)
    ax.axhline(y=0.92, color="green", linestyle="--", alpha=0.7, label="Training season (summer)")
    ax.set_ylabel("Performance", fontsize=11)
    ax.set_xlabel("Season", fontsize=11)
    ax.set_title("Figure 58 — Seasonal/Weather Generalization (Extended PV Profiles)", fontsize=12)
    ax.set_ylim(0.7, 1.0)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    for bar, p in zip(bars, performance):
        ax.text(bar.get_x() + bar.get_width()/2, p + 0.01, f"{p:.2f}", 
               ha="center", fontsize=10, fontweight="bold")

    fig.tight_layout()
    mark_synthetic(fig)
    fig.savefig(args.output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[Adapter] Saved: {args.output}")

if __name__ == "__main__":
    main()
