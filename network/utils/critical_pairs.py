#!/usr/bin/env python3
"""CLI Adapter: critical_pairs.py — Fig 50

WARNING: SYNTHETIC DATA. This script plots hand-typed literal number arrays,
not output from a trained model or a real experiment run. See
experiments/GAP_ANALYSIS.md. Do not cite this figure as experimental evidence
until it is replaced with a real data pipeline."""
import argparse, sys, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from evaluation.experiments._synthetic_label import mark_synthetic

def main():
    p = argparse.ArgumentParser(description="Critical Pairs Adapter (Fig 50)")
    p.add_argument("--system", type=str, default="8bus")
    p.add_argument("--pv", type=float, default=1.0)
    p.add_argument("--lambda-data", type=str, default="")
    p.add_argument("--violation-data", type=str, default="")
    p.add_argument("--output", type=str, default="outputs/fig50_critical_pairs.pdf")
    args = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    pairs = [f"P{i}-B{j}" for i in range(7) for j in range(7) if i != j][:20]
    criticality = np.random.exponential(2.0, len(pairs))
    criticality = np.sort(criticality)[::-1]

    colors = ["#ef4444" if c > 4 else "#f59e0b" if c > 2 else "#2563eb" for c in criticality]
    bars = ax.barh(range(len(pairs)), criticality, color=colors, edgecolor="white")
    ax.set_yticks(range(len(pairs)))
    ax.set_yticklabels(pairs, fontsize=7)
    ax.set_xlabel("Criticality Score (λ × violations)", fontsize=11)
    ax.set_title("P/B Pair Criticality Ranking", fontsize=12)
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis="x")

    ax = axes[1]
    lambdas = np.random.exponential(1.5, 50)
    violations = np.random.poisson(lambdas * 2)
    ax.scatter(lambdas, violations, c=violations, cmap="YlOrRd", s=80, edgecolors="white", alpha=0.7)
    ax.set_xlabel("Lagrange Multiplier λ", fontsize=11)
    ax.set_ylabel("Violation Count", fontsize=11)
    ax.set_title("λ vs. Violation Frequency", fontsize=12)
    ax.grid(True, alpha=0.3)

    fig.suptitle("Figure 50 — Critical P/B Pair Identification & Priority Ranking", fontsize=12)
    fig.tight_layout()
    mark_synthetic(fig)
    fig.savefig(args.output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[Adapter] Saved: {args.output}")

if __name__ == "__main__":
    main()
