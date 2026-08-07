#!/usr/bin/env python3
"""CLI Adapter: relay_curves.py — Fig 46

WARNING: SYNTHETIC DATA. This script plots hand-typed literal number arrays,
not output from a trained model or a real experiment run. See
experiments/GAP_ANALYSIS.md. Do not cite this figure as experimental evidence
until it is replaced with a real data pipeline."""
import argparse, sys, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from evaluation.experiments._synthetic_label import mark_synthetic

def main():
    p = argparse.ArgumentParser(description="Relay Curves Adapter (Fig 46)")
    p.add_argument("--system", type=str, default="8bus")
    p.add_argument("--pv", type=float, default=1.0)
    p.add_argument("--method", type=str, default="MAPPO-Safe-EWC")
    p.add_argument("--output", type=str, default="outputs/fig46_relay_curves.pdf")
    args = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 8))

    M = np.linspace(1.0, 20, 200)

    tms_values = [0.5, 0.8, 1.0, 1.2, 1.5]
    colors = plt.cm.viridis(np.linspace(0, 1, len(tms_values)))

    for tms, color in zip(tms_values, colors):
        t = tms * (7.03 / (M**0.02 - 1) + 0.025)
        t = np.clip(t, 0, 10)
        ax.loglog(M, t, lw=2, color=color, label=f"TMS={tms}")

    ax.set_xlabel("Multiple of Pickup Current (M)", fontsize=11)
    ax.set_ylabel("Operating Time (s)", fontsize=11)
    ax.set_title("Figure 46 — Relay Operating Characteristic Curves with Learned Settings", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, which="both")
    ax.set_ylim(0.01, 10)

    fig.tight_layout()
    mark_synthetic(fig)
    fig.savefig(args.output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[Adapter] Saved: {args.output}")

if __name__ == "__main__":
    main()
