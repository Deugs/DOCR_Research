#!/usr/bin/env python3
"""CLI Adapter: fault_contribution.py — Fig 44

WARNING: SYNTHETIC DATA. This script plots hand-typed literal number arrays,
not output from a trained model or a real experiment run. See
experiments/GAP_ANALYSIS.md. Do not cite this figure as experimental evidence
until it is replaced with a real data pipeline."""
import argparse, sys, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from evaluation.experiments._synthetic_label import mark_synthetic

def main():
    p = argparse.ArgumentParser(description="Fault Contribution Adapter (Fig 44)")
    p.add_argument("--system", type=str, default="8bus")
    p.add_argument("--pv", type=float, default=1.0)
    p.add_argument("--output", type=str, default="outputs/fig44_fault_contribution.pdf")
    args = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 8))

    sources = ["Utility Grid", "PV Inverters", "Synchronous Generators", "Motors/Loads"]
    contributions = [35, 40, 20, 5]
    colors = ["#2563eb", "#f59e0b", "#39d353", "#a371f7"]
    explode = (0, 0.1, 0, 0)

    wedges, texts, autotexts = ax.pie(contributions, explode=explode, labels=sources, colors=colors,
                                       autopct="%1.1f%%", startangle=90, textprops={"fontsize": 11})
    ax.set_title("Figure 44 — Fault Current Contribution Pie Chart (100% PV)", fontsize=12)

    fig.tight_layout()
    mark_synthetic(fig)
    fig.savefig(args.output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[Adapter] Saved: {args.output}")

if __name__ == "__main__":
    main()
