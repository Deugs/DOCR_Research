#!/usr/bin/env python3
"""CLI Adapter: daily_adaptive_response.py — Fig 45

WARNING: SYNTHETIC DATA. This script plots hand-typed literal number arrays,
not output from a trained model or a real experiment run. See
experiments/GAP_ANALYSIS.md. Do not cite this figure as experimental evidence
until it is replaced with a real data pipeline."""
import argparse, sys, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from evaluation.experiments._synthetic_label import mark_synthetic

def main():
    p = argparse.ArgumentParser(description="Daily Adaptive Response Adapter (Fig 45)")
    p.add_argument("--system", type=str, default="8bus")
    p.add_argument("--tmy3", type=str, default="")
    p.add_argument("--output", type=str, default="outputs/fig45_daily_profile.pdf")
    args = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    hours = np.arange(0, 24, 0.5)

    ax = axes[0]
    pv_profile = np.maximum(0, np.sin((hours - 6) * np.pi / 12)) * np.random.uniform(0.8, 1.0, len(hours))
    ax.fill_between(hours, 0, pv_profile, alpha=0.3, color="#f59e0b")
    ax.plot(hours, pv_profile, "o-", color="#f59e0b", lw=2, markersize=4, markevery=4)
    ax.set_ylabel("PV Generation (p.u.)", fontsize=11)
    ax.set_title("Daily PV Generation Profile", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.2)

    ax = axes[1]
    tms_adaptive = 1.0 - 0.3 * pv_profile + np.random.normal(0, 0.02, len(hours))
    tms_traditional = np.ones_like(hours) * 1.0

    ax.plot(hours, tms_adaptive, "o-", color="#2563eb", lw=2, markersize=4, markevery=4, label="Adaptive (MAPPO-Safe-EWC)")
    ax.plot(hours, tms_traditional, "--", color="#64748b", lw=2, label="Traditional (fixed)")
    ax.fill_between(hours, tms_adaptive, tms_traditional, alpha=0.2, color="#2563eb")

    ax.set_xlabel("Hour of Day", fontsize=11)
    ax.set_ylabel("TMS Setting", fontsize=11)
    ax.set_title("Adaptive Relay Response", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.5, 1.2)

    fig.suptitle("Figure 45 — Daily PV Generation Profile & Adaptive Relay Response", fontsize=12)
    fig.tight_layout()
    mark_synthetic(fig)
    fig.savefig(args.output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[Adapter] Saved: {args.output}")

if __name__ == "__main__":
    main()
