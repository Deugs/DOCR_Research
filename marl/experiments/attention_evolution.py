#!/usr/bin/env python3
"""
Standalone CLI script for Figure 38 (Attention Weight Evolution) — NOT
currently invoked by the live pipeline.

This file's own docstring previously claimed `run_thesis_experiments.py
--phase3` calls this script via subprocess. That is no longer true (if it
ever was): `run_thesis_experiments.py`'s figure generation
(`generate_figures_batch()` / `load_post_processing_module()`) calls
`run_post_processing.py`'s in-process `FIGURE_FUNCTIONS` dict directly
(subprocessing to `run_post_processing.py --fig 38` only as an explicit
`--use-subprocess` fallback) — never this file. The real, live Fig 38 path
is `run_post_processing.py::fig_38_attention_evolution()`, which loads
real attention snapshots from the `core_A_mappo_safe_ewc_8bus_60pv` cache
when present, falling back to hardcoded/synthetic data (clearly logged as
such) only if that cache is missing.

This script is still an honestly-labelled STUB in its own right — it has
no real attention-tracking mechanism and always generates a synthetic
visualization — but it is orphaned from the pipeline that actually
produces the thesis figures. Left in place undeleted (a working, self-
contained script may still be useful for manual/ad-hoc use), matching how
other orphaned-but-functional scripts in this codebase are handled.
"""
import argparse
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


def main():
    p = argparse.ArgumentParser(description="Attention Evolution Adapter (Fig 38) — STUB")
    p.add_argument("--system", type=str, default="8bus", help="Test system (8bus, 14bus, 30bus, 39bus)")
    p.add_argument("--pv", type=float, default=1.0)
    p.add_argument("--snapshots", nargs="+", type=int, default=[0, 100, 500, 1000, 5000])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=str, default="outputs/fig38_attention_evolution.pdf")
    args = p.parse_args()

    print(f"[Adapter] Fig 38 Attention Evolution -> STUB (no actual attention tracking in codebase)")
    print(f"  Generating synthetic attention evolution visualization...")

    out_dir = os.path.dirname(args.output) or "outputs"
    os.makedirs(out_dir, exist_ok=True)

    # Generate synthetic attention weight evolution
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    np.random.seed(args.seed)
    n_agents = 7 if args.system == "8bus" else 14
    snapshots = args.snapshots

    fig, axes = plt.subplots(1, len(snapshots), figsize=(4 * len(snapshots), 4))
    if len(snapshots) == 1:
        axes = [axes]

    for idx, snap in enumerate(snapshots):
        ax = axes[idx]
        # Synthetic attention matrix that becomes more structured over time
        base = np.random.rand(n_agents, n_agents) * 0.3
        # Add structure: diagonal and near-diagonal stronger
        for i in range(n_agents):
            base[i, i] = 0.5 + 0.3 * min(snap / 1000, 1.0)
            if i > 0:
                base[i, i-1] = 0.3 + 0.2 * min(snap / 1000, 1.0)
            if i < n_agents - 1:
                base[i, i+1] = 0.3 + 0.2 * min(snap / 1000, 1.0)
        # Normalize rows
        base = base / base.sum(axis=1, keepdims=True)

        im = ax.imshow(base, cmap="YlOrRd", vmin=0, vmax=1)
        ax.set_title(f"Episode {snap}", fontsize=10)
        ax.set_xlabel("Agent j", fontsize=8)
        ax.set_ylabel("Agent i", fontsize=8)
        plt.colorbar(im, ax=ax, shrink=0.6)

    fig.suptitle("Figure 38 — Attention Weight Evolution During Training (STUB)", fontsize=12)
    fig.tight_layout()
    fig.savefig(args.output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[Adapter] Saved stub visualization: {args.output}")


if __name__ == "__main__":
    main()