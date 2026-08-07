#!/usr/bin/env python3
"""CLI Adapter: sensor_degradation.py — Fig 24

WARNING: SYNTHETIC DATA. This script plots hand-typed literal number arrays,
not output from a trained model or a real experiment run. See
experiments/GAP_ANALYSIS.md. Do not cite this figure as experimental evidence
until it is replaced with a real data pipeline."""
import argparse, sys, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from evaluation.experiments._synthetic_label import mark_synthetic

def main():
    p = argparse.ArgumentParser(description="Sensor Degradation Adapter (Fig 24)")
    p.add_argument("--system", type=str, default="8bus")
    p.add_argument("--pv", type=float, default=1.0)
    p.add_argument("--noise-levels", nargs="+", type=float, default=[0.0, 0.05, 0.1, 0.15, 0.2])
    p.add_argument("--bias-levels", nargs="+", type=float, default=[0.0, 0.05, 0.1])
    p.add_argument("--missing-rates", nargs="+", type=float, default=[0.0, 0.1, 0.2])
    p.add_argument("--output", type=str, default="outputs/fig24_sensor.pdf")
    args = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    noise = args.noise_levels or [0.0, 0.05, 0.1, 0.15, 0.2]
    perf_noise = [0.92, 0.88, 0.82, 0.73, 0.62]
    axes[0].plot(noise, perf_noise, "o-", color="#2563eb", lw=2, markersize=8)
    axes[0].set_xlabel("Noise Level (σ)", fontsize=11)
    axes[0].set_ylabel("Performance", fontsize=11)
    axes[0].set_title("Gaussian Noise", fontsize=12)
    axes[0].grid(True, alpha=0.3)

    bias = args.bias_levels or [0.0, 0.05, 0.1]
    perf_bias = [0.92, 0.85, 0.75]
    axes[1].plot(bias, perf_bias, "s-", color="#ff6b35", lw=2, markersize=8)
    axes[1].set_xlabel("Bias Level", fontsize=11)
    axes[1].set_ylabel("Performance", fontsize=11)
    axes[1].set_title("Systematic Bias", fontsize=12)
    axes[1].grid(True, alpha=0.3)

    missing = args.missing_rates or [0.0, 0.1, 0.2]
    perf_missing = [0.92, 0.80, 0.65]
    axes[2].plot(missing, perf_missing, "^-", color="#39d353", lw=2, markersize=8)
    axes[2].set_xlabel("Missing Data Rate", fontsize=11)
    axes[2].set_ylabel("Performance", fontsize=11)
    axes[2].set_title("Missing Measurements", fontsize=12)
    axes[2].grid(True, alpha=0.3)

    fig.suptitle("Figure 24 — Sensor Robustness: Performance Under Measurement Degradation", fontsize=12)
    fig.tight_layout()
    mark_synthetic(fig)
    fig.savefig(args.output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[Adapter] Saved: {args.output}")

if __name__ == "__main__":
    main()
