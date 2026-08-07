#!/usr/bin/env python3
"""CLI Adapter: bandwidth_tradeoff.py — Fig 22

WARNING: SYNTHETIC DATA. This script plots hand-typed literal number arrays,
not output from a trained model or a real experiment run. See
experiments/GAP_ANALYSIS.md. Do not cite this figure as experimental evidence
until it is replaced with a real data pipeline."""
import argparse, sys, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from evaluation.experiments._synthetic_label import mark_synthetic

def main():
    p = argparse.ArgumentParser(description="Bandwidth Tradeoff Adapter (Fig 22)")
    p.add_argument("--system", type=str, default="8bus")
    p.add_argument("--pv", type=float, default=1.0)
    p.add_argument("--gnn-depths", nargs="+", type=int, default=[1, 2, 3, 4])
    p.add_argument("--compressions", nargs="+", type=str, default=["none", "int8", "topk50"])
    p.add_argument("--output", type=str, default="outputs/fig22_bandwidth.pdf")
    args = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    depths = args.gnn_depths or [1, 2, 3, 4]
    bandwidth = [d * 2.5 for d in depths]
    performance = [0.78, 0.88, 0.92, 0.91]
    ax = axes[0]
    ax2 = ax.twinx()
    bars = ax.bar([str(d) for d in depths], bandwidth, color="#a371f7", alpha=0.6, label="Bandwidth")
    line = ax2.plot([str(d) for d in depths], performance, "o-", color="#2563eb", lw=2, markersize=8, label="Performance")
    ax.set_xlabel("GNN Depth (hops)", fontsize=11)
    ax.set_ylabel("Bandwidth (KB/episode)", fontsize=11, color="#a371f7")
    ax2.set_ylabel("Performance", fontsize=11, color="#2563eb")
    ax.set_title("GNN Depth Trade-off", fontsize=12)
    ax.grid(True, alpha=0.3, axis="y")

    ax = axes[1]
    methods = args.compressions or ["none", "int8", "topk50"]
    sizes = [100, 25, 15]
    perf = [0.92, 0.91, 0.89]
    x = np.arange(len(methods))
    ax.bar(x - 0.2, sizes, 0.4, color="#f59e0b", label="Size (%)")
    ax2 = ax.twinx()
    ax2.bar(x + 0.2, perf, 0.4, color="#2563eb", label="Performance")
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylabel("Model Size (%)", fontsize=11, color="#f59e0b")
    ax2.set_ylabel("Performance", fontsize=11, color="#2563eb")
    ax.set_title("Compression Methods", fontsize=12)
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Figure 22 — Bandwidth vs. Performance Trade-off", fontsize=12)
    fig.tight_layout()
    mark_synthetic(fig)
    fig.savefig(args.output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[Adapter] Saved: {args.output}")

if __name__ == "__main__":
    main()
