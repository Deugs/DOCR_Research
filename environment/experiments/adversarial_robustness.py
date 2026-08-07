#!/usr/bin/env python3
"""CLI Adapter: adversarial_robustness.py — Fig 54

WARNING: SYNTHETIC DATA. This script plots hand-typed literal number arrays,
not output from a trained model or a real experiment run. See
experiments/GAP_ANALYSIS.md. Do not cite this figure as experimental evidence
until it is replaced with a real data pipeline."""
import argparse, sys, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from evaluation.experiments._synthetic_label import mark_synthetic

def main():
    p = argparse.ArgumentParser(description="Adversarial Robustness Adapter (Fig 54)")
    p.add_argument("--system", type=str, default="8bus")
    p.add_argument("--pv", type=float, default=1.0)
    p.add_argument("--attack-types", nargs="+", type=str, default=["fgsm", "pgd"])
    p.add_argument("--epsilons", nargs="+", type=float, default=[0.01, 0.05, 0.1, 0.2])
    p.add_argument("--n_eval", type=int, default=100)
    p.add_argument("--output", type=str, default="outputs/fig54_adversarial.pdf")
    args = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))

    epsilons = args.epsilons or [0.01, 0.05, 0.1, 0.2]
    attacks = args.attack_types or ["fgsm", "pgd"]

    for attack, color in zip(attacks, ["#ef4444", "#f59e0b"]):
        if attack == "fgsm":
            perf = [0.90, 0.82, 0.68, 0.45]
        else:
            perf = [0.88, 0.75, 0.55, 0.30]
        ax.plot(epsilons, perf, "o-", color=color, lw=2, markersize=8, label=attack.upper())

    ax.axhline(y=0.92, color="green", linestyle="--", alpha=0.7, label="Clean (no attack)")
    ax.set_xlabel("Perturbation Magnitude (ε)", fontsize=11)
    ax.set_ylabel("Accuracy / Performance", fontsize=11)
    ax.set_title("Figure 54 — Adversarial Robustness: Perturbed Observation Response", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    mark_synthetic(fig)
    fig.savefig(args.output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[Adapter] Saved: {args.output}")

if __name__ == "__main__":
    main()
