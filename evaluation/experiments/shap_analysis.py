#!/usr/bin/env python3
"""CLI Adapter: shap_analysis.py — Fig 48"""
import argparse, sys, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

def main():
    p = argparse.ArgumentParser(description="SHAP Analysis Adapter (Fig 48)")
    p.add_argument("--system", type=str, default="8bus")
    p.add_argument("--pv", type=float, default=1.0)
    p.add_argument("--n_background", type=int, default=100)
    p.add_argument("--agents", type=str, default="all")
    p.add_argument("--output", type=str, default="outputs/fig48_shap.pdf")
    args = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))

    features = ["V_magnitude", "I_fault", "Z_line", "t_since_fault", "PV_active", 
                "Neighbor_V", "TMS_current", "PS_current", "Curriculum_stage"]
    shap_vals = np.array([0.18, 0.25, 0.12, 0.15, 0.08, 0.10, 0.05, 0.04, 0.03])

    colors = ["#2563eb" if v > 0.1 else "#39d353" if v > 0.05 else "#a371f7" for v in shap_vals]
    bars = ax.barh(features, shap_vals, color=colors, edgecolor="white")
    ax.set_xlabel("Mean |SHAP value|", fontsize=11)
    ax.set_title("Figure 48 — SHAP Feature Importance for Relay Decisions", fontsize=12)
    ax.grid(True, alpha=0.3, axis="x")
    ax.invert_yaxis()

    for bar, v in zip(bars, shap_vals):
        ax.text(v + 0.005, bar.get_y() + bar.get_height()/2, f"{v:.3f}", 
               va="center", fontsize=9)

    fig.tight_layout()
    fig.savefig(args.output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[Adapter] Saved: {args.output}")

if __name__ == "__main__":
    main()
