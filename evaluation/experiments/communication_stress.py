#!/usr/bin/env python3
"""CLI Adapter: communication_stress.py — Fig 13

WARNING: SYNTHETIC DATA. This script plots hand-typed literal number arrays,
not output from a trained model or a real experiment run. See
experiments/GAP_ANALYSIS.md. Do not cite this figure as experimental evidence
until it is replaced with a real data pipeline."""
import argparse, sys, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from evaluation.experiments._synthetic_label import mark_synthetic

def main():
    p = argparse.ArgumentParser(description="Communication Stress Adapter (Fig 13)")
    p.add_argument("--system", type=str, default="8bus")
    p.add_argument("--pv", type=float, default=1.0)
    p.add_argument("--method", type=str, default="MAPPO-Safe-EWC")
    p.add_argument("--packet-loss", nargs="+", type=float, default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    p.add_argument("--latency", nargs="+", type=int, default=[0, 100, 200, 300, 400, 500])
    p.add_argument("--output", type=str, default="outputs/fig13_comm_stress.pdf")
    args = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    packet_loss = args.packet_loss or [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    perf = [0.92, 0.88, 0.82, 0.72, 0.58, 0.40]
    ax = axes[0]
    ax.plot(packet_loss, perf, "o-", color="#2563eb", lw=2, markersize=8)
    ax.fill_between(packet_loss, [p*0.95 for p in perf], [p*1.05 for p in perf], alpha=0.2, color="#2563eb")
    ax.set_xlabel("Packet Loss Rate", fontsize=11)
    ax.set_ylabel("Normalized Performance", fontsize=11)
    ax.set_title("Performance vs. Packet Loss", fontsize=12)
    ax.grid(True, alpha=0.3)

    latency = args.latency or [0, 100, 200, 300, 400, 500]
    perf2 = [0.92, 0.90, 0.87, 0.82, 0.75, 0.65]
    ax = axes[1]
    ax.plot(latency, perf2, "s-", color="#ff6b35", lw=2, markersize=8)
    ax.fill_between(latency, [p*0.95 for p in perf2], [p*1.05 for p in perf2], alpha=0.2, color="#ff6b35")
    ax.set_xlabel("Latency (ms)", fontsize=11)
    ax.set_ylabel("Normalized Performance", fontsize=11)
    ax.set_title("Performance vs. Communication Latency", fontsize=12)
    ax.grid(True, alpha=0.3)

    fig.suptitle("Figure 13 — Performance Degradation under Communication Stress", fontsize=12)
    fig.tight_layout()
    mark_synthetic(fig)
    fig.savefig(args.output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[Adapter] Saved: {args.output}")

if __name__ == "__main__":
    main()
