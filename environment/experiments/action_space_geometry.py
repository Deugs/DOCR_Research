#!/usr/bin/env python3
"""CLI Adapter: action_space_geometry.py — Fig 36

WARNING: SYNTHETIC DATA. This script plots hand-typed literal number arrays,
not output from a trained model or a real experiment run. See
experiments/GAP_ANALYSIS.md. Do not cite this figure as experimental evidence
until it is replaced with a real data pipeline."""
import argparse, sys, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from evaluation.experiments._synthetic_label import mark_synthetic

def main():
    p = argparse.ArgumentParser(description="Action Space Geometry Adapter (Fig 36)")
    p.add_argument("--system", type=str, default="8bus")
    p.add_argument("--grid", type=int, default=100)
    p.add_argument("--output", type=str, default="outputs/fig36_action_space.pdf")
    args = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    grid = args.grid or 100
    tms_range = np.linspace(0.5, 2.0, grid)
    ps_range = np.linspace(1.0, 5.0, grid)
    TMS, PS = np.meshgrid(tms_range, ps_range)

    Z = -((TMS - 1.2)**2 + (PS - 3.0)**2) + np.sin(TMS * 3) * 0.3
    im = ax.contourf(TMS, PS, Z, levels=20, cmap="viridis")
    plt.colorbar(im, ax=ax, shrink=0.6, label="Objective")

    safe_tms = np.linspace(0.8, 1.6, 50)
    safe_ps_min = 2.0 + 0.5 * (safe_tms - 1.2)**2
    safe_ps_max = 4.0 - 0.3 * (safe_tms - 1.2)**2
    ax.fill_between(safe_tms, safe_ps_min, safe_ps_max, alpha=0.2, color="green", label="Safe region")

    raw = [(1.0, 4.5), (1.8, 2.2), (0.6, 3.0)]
    safe = [(1.1, 3.8), (1.5, 2.8), (0.9, 2.5)]
    for (x1, y1), (x2, y2) in zip(raw, safe):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                   arrowprops=dict(arrowstyle="->", color="red", lw=2))
        ax.scatter(x1, y1, c="red", s=50, zorder=5)
        ax.scatter(x2, y2, c="green", s=50, zorder=5)

    ax.set_xlabel("TMS (Time Multiplier Setting)", fontsize=10)
    ax.set_ylabel("PS (Plug Setting)", fontsize=10)
    ax.set_title("Action Space with Safety Projection", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    theta = np.linspace(0, 2*np.pi, 100)
    ax.plot(np.cos(theta), np.sin(theta), "k--", lw=1, alpha=0.5, label="Unit ball")

    ax.axhline(y=0.5, color="red", linestyle="-", alpha=0.7, label="CTI constraint")
    ax.axvline(x=0.3, color="blue", linestyle="-", alpha=0.7, label="TMS bound")

    ax.scatter([0.8], [0.2], c="red", s=100, marker="x", linewidths=3, label="Raw action a")
    ax.scatter([0.3], [0.5], c="green", s=100, marker="o", label="Projected a\'")
    ax.annotate("", xy=(0.3, 0.5), xytext=(0.8, 0.2),
               arrowprops=dict(arrowstyle="->", color="purple", lw=2))
    ax.text(0.55, 0.35, "Projection\nΔa", fontsize=10, color="purple")

    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_aspect("equal")
    ax.set_xlabel("Dimension 1", fontsize=10)
    ax.set_ylabel("Dimension 2", fontsize=10)
    ax.set_title("Safety Projection Geometry", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.suptitle("Figure 36 — Action Space Bounds & Safety Projection Geometry", fontsize=12)
    fig.tight_layout()
    mark_synthetic(fig)
    fig.savefig(args.output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[Adapter] Saved: {args.output}")

if __name__ == "__main__":
    main()
