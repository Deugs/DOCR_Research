#!/usr/bin/env python3
"""CLI Adapter: n1_contingency.py — Fig 14

WARNING: SYNTHETIC DATA. This script plots hand-typed literal number arrays,
not output from a trained model or a real experiment run. See
experiments/GAP_ANALYSIS.md. Do not cite this figure as experimental evidence
until it is replaced with a real data pipeline."""
import argparse, sys, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from evaluation.experiments._synthetic_label import mark_synthetic

def main():
    p = argparse.ArgumentParser(description="N-1 Contingency Adapter (Fig 14)")
    p.add_argument("--system", type=str, default="30bus")
    p.add_argument("--method", type=str, default="MAPPO-Safe-EWC")
    p.add_argument("--finetune-episodes", type=int, default=50)
    p.add_argument("--output", type=str, default="outputs/fig14_n1_contingency.pdf")
    args = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))

    episodes = np.arange(args.finetune_episodes + 1)
    perf = 0.92 - 0.35 * np.exp(-episodes / 10) + np.random.normal(0, 0.02, len(episodes))
    perf = np.clip(perf, 0, 1)

    ax.plot(episodes, perf, "o-", color="#2563eb", lw=2, markersize=6)
    ax.axhline(y=0.92, color="green", linestyle="--", alpha=0.7, label="Pre-contingency")
    ax.axvline(x=0, color="red", linestyle="-", alpha=0.7, label="Contingency event")
    ax.fill_between(episodes, 0.92*0.95, 0.92*1.05, alpha=0.1, color="green")

    ax.set_xlabel("Episodes After Contingency", fontsize=11)
    ax.set_ylabel("Normalized Performance", fontsize=11)
    ax.set_title("Figure 14 — N-1 Contingency Impact and Recovery (30-Bus)", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    mark_synthetic(fig)
    fig.savefig(args.output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[Adapter] Saved: {args.output}")

if __name__ == "__main__":
    main()
