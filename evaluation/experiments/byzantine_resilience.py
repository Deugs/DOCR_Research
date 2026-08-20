#!/usr/bin/env python3
"""CLI Adapter: byzantine_resilience.py — Fig 20

WARNING: SYNTHETIC DATA. This script plots hand-typed literal number arrays,
not output from a trained model or a real experiment run. See
experiments/GAP_ANALYSIS.md. Do not cite this figure as experimental evidence
until it is replaced with a real data pipeline."""
import argparse, sys, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from evaluation.experiments._synthetic_label import mark_synthetic

def main():
    p = argparse.ArgumentParser(description="Byzantine Resilience Adapter (Fig 20)")
    p.add_argument("--system", type=str, default="8bus")
    p.add_argument("--pv", type=float, default=1.0)
    p.add_argument("--n_byzantine", type=int, default=2)
    p.add_argument("--attack-types", nargs="+", type=str, default=["random", "collusion", "fdi"])
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=str, default="outputs/fig20_byzantine.pdf")
    args = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Explicit seeded RNG rather than the unseeded/global np.random.seed +
    # np.random.normal -- keeps this adapter's output reproducible without
    # mutating global RNG state that other code might depend on.
    rng = np.random.default_rng(args.seed)
    episodes = np.arange(args.episodes)

    ax = axes[0]
    for i, agent in enumerate(["Honest A", "Honest B", "Honest C", "Byzantine 1", "Byzantine 2"]):
        if "Byzantine" in agent:
            trust = 1.0 - 0.8 * (1 - np.exp(-episodes / 15)) + rng.normal(0, 0.03, len(episodes))
        else:
            trust = 0.9 + 0.1 * (1 - np.exp(-episodes / 20)) + rng.normal(0, 0.02, len(episodes))
        trust = np.clip(trust, 0, 1)
        color = "#ef4444" if "Byzantine" in agent else "#2563eb"
        ax.plot(episodes, trust, lw=1.5, alpha=0.8, color=color, label=agent)

    ax.axhline(y=0.5, color="orange", linestyle="--", alpha=0.7, label="Detection threshold")
    ax.set_xlabel("Episode", fontsize=11)
    ax.set_ylabel("Trust Score", fontsize=11)
    ax.set_title("Trust Score Evolution", fontsize=12)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    attacks = args.attack_types or ["random", "collusion", "fdi"]
    # Keyed by attack-type name so an arbitrary --attack-types list (any
    # length/order) never runs off the end of a fixed-length literal.
    PERF_BY_ATTACK = {"random": 0.85, "collusion": 0.78, "fdi": 0.72}
    COLOR_BY_ATTACK = {"random": "#2563eb", "collusion": "#f59e0b", "fdi": "#ef4444"}
    perf = [PERF_BY_ATTACK.get(a, 0.75) for a in attacks]
    colors = [COLOR_BY_ATTACK.get(a, "#8b949e") for a in attacks]
    bars = ax.bar(attacks, perf, color=colors, edgecolor="white", width=0.6)
    ax.set_ylabel("Normalized Performance", fontsize=11)
    ax.set_title("Performance Under Attack Types", fontsize=12)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis="y")

    for bar, p in zip(bars, perf):
        ax.text(bar.get_x() + bar.get_width()/2, p + 0.02, f"{p:.2f}", 
               ha="center", fontsize=10, fontweight="bold")

    fig.suptitle("Figure 20 — Byzantine Agent Detection via Trust Score Evolution", fontsize=12)
    fig.tight_layout()
    mark_synthetic(fig)
    fig.savefig(args.output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[Adapter] Saved: {args.output}")

if __name__ == "__main__":
    main()
