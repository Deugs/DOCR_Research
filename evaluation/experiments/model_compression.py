#!/usr/bin/env python3
"""CLI Adapter: model_compression.py — Fig 53

WARNING: SYNTHETIC DATA. This script plots hand-typed literal number arrays,
not output from a trained model or a real experiment run. See
experiments/GAP_ANALYSIS.md. Do not cite this figure as experimental evidence
until it is replaced with a real data pipeline."""
import argparse, sys, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from evaluation.experiments._synthetic_label import mark_synthetic

def main():
    p = argparse.ArgumentParser(description="Model Compression Adapter (Fig 53)")
    p.add_argument("--model", type=str, default="")
    p.add_argument("--methods", nargs="+", type=str, default=["int8", "pruning", "distillation"])
    p.add_argument("--n_eval", type=int, default=100)
    p.add_argument("--output", type=str, default="outputs/fig53_compression.pdf")
    args = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    methods = args.methods or ["int8", "pruning", "distillation"]

    ax = axes[0]
    original_size = 100
    sizes = {"int8": 25, "pruning": 40, "distillation": 35}
    size_vals = [sizes.get(m, 50) for m in methods]
    colors = ["#2563eb", "#39d353", "#f59e0b"]
    bars = ax.bar(["Original"] + methods, [original_size] + size_vals, 
                  color=["#64748b"] + colors, edgecolor="white", width=0.6)
    ax.set_ylabel("Model Size (% of original)", fontsize=11)
    ax.set_title("Size Reduction", fontsize=12)
    ax.grid(True, alpha=0.3, axis="y")

    ax = axes[1]
    perf = {"int8": 0.91, "pruning": 0.88, "distillation": 0.90}
    perf_vals = [perf.get(m, 0.85) for m in methods]
    ax.scatter(size_vals, perf_vals, s=200, c=colors, edgecolors="black", linewidths=2, zorder=5)
    for i, m in enumerate(methods):
        ax.annotate(m, (size_vals[i], perf_vals[i]), textcoords="offset points", 
                   xytext=(10, 5), fontsize=10)
    ax.set_xlabel("Model Size (% of original)", fontsize=11)
    ax.set_ylabel("Performance", fontsize=11)
    ax.set_title("Performance vs. Compression", fontsize=12)
    ax.set_xlim(0, 100)
    ax.set_ylim(0.8, 0.95)
    ax.grid(True, alpha=0.3)

    fig.suptitle("Figure 53 — Model Compression & Quantization Impact", fontsize=12)
    fig.tight_layout()
    mark_synthetic(fig)
    fig.savefig(args.output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[Adapter] Saved: {args.output}")

if __name__ == "__main__":
    main()
