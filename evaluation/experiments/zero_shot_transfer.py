#!/usr/bin/env python3
"""CLI Adapter: zero_shot_transfer.py — Fig 25

WARNING: SYNTHETIC DATA. This script plots hand-typed literal number arrays,
not output from a trained model or a real experiment run. See
experiments/GAP_ANALYSIS.md. Do not cite this figure as experimental evidence
until it is replaced with a real data pipeline."""
import argparse, sys, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from evaluation.experiments._synthetic_label import mark_synthetic

def main():
    p = argparse.ArgumentParser(description="Zero-Shot Transfer Adapter (Fig 25)")
    p.add_argument("--source-system", type=str, default="8bus")
    p.add_argument("--target-system", type=str, default="30bus")
    p.add_argument("--pv", type=float, default=1.0)
    p.add_argument("--finetune-steps", type=int, default=500)
    p.add_argument("--scratch-steps", type=int, default=5000)
    p.add_argument("--output", type=str, default="outputs/fig25_transfer.pdf")
    args = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))

    steps = np.arange(0, max(args.finetune_steps, args.scratch_steps) + 1, 50)

    transfer_perf = 0.65 + 0.25 * (1 - np.exp(-steps / 100))
    transfer_perf = np.clip(transfer_perf, 0, 1)

    scratch_perf = 0.20 + 0.70 * (1 - np.exp(-steps / 800))
    scratch_perf = np.clip(scratch_perf, 0, 1)

    ax.plot(steps, transfer_perf, "o-", color="#2563eb", lw=2, markersize=3, 
            markevery=200, label=f"Transfer (8→30 bus) + {args.finetune_steps} steps fine-tune")
    ax.plot(steps, scratch_perf, "s-", color="#ff6b35", lw=2, markersize=3,
            markevery=200, label=f"From scratch ({args.scratch_steps} steps)")

    ax.axhline(y=0.92, color="green", linestyle="--", alpha=0.5, label="Source performance")
    ax.set_xlabel("Training Steps", fontsize=11)
    ax.set_ylabel("Performance on Target System", fontsize=11)
    ax.set_title("Figure 25 — Zero-Shot Transfer: 8-Bus → 30-Bus (100% PV)", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    ax.annotate("10× faster convergence", xy=(200, 0.85), fontsize=11,
               color="#2563eb", fontweight="bold")

    fig.tight_layout()
    mark_synthetic(fig)
    fig.savefig(args.output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[Adapter] Saved: {args.output}")

if __name__ == "__main__":
    main()
