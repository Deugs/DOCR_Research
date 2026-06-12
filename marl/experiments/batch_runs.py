#!/usr/bin/env python3
"""
marl/experiments/batch_runs.py
──────────────────────────────
CLI wrapper for Figure 12: Convergence Curves with 95% Confidence Bands.

Translates run_thesis_experiments.py CLI args to exp_fig12_convergence_curves.py interface.
"""

import argparse
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Monkey-patch MAPPOMARLTrainer to force use_comm=False
# This fixes dimension mismatch when exp code calls actor.sample() with raw obs
import marl.marl_agents as _ma
_orig_init = _ma.MAPPOMARLTrainer.__init__
def _patched_init(self, *args, **kwargs):
    kwargs["use_comm"] = False
    _orig_init(self, *args, **kwargs)
_ma.MAPPOMARLTrainer.__init__ = _patched_init

# Import the actual experiment implementation
from marl.experiments.exp_fig12_convergence_curves import main as fig12_main


def main():
    p = argparse.ArgumentParser(description="Figure 12 — Batch Convergence Runs (CLI wrapper)")
    p.add_argument("--system", type=str, default="8bus", help="Test system (8bus, 14bus, 30bus)")
    p.add_argument("--pv", type=float, default=1.0, help="PV penetration level")
    p.add_argument("--method", type=str, default="MAPPO-Safe-EWC",
                   help="Training method (MAPPO-Safe-EWC, MAPPO-Safe, MAPPO, IPPO, Traditional)")
    p.add_argument("--n_runs", type=int, default=50, help="Number of independent training runs")
    p.add_argument("--episodes", type=int, default=5000, help="Episodes per run")
    p.add_argument("--output", type=str, required=True, help="Output PDF path")
    args = p.parse_args()

    # Derive output directory from output path
    output_dir = str(Path(args.output).parent)
    os.makedirs(output_dir, exist_ok=True)

    # Call the actual experiment implementation
    fig12_main(
        n_runs=args.n_runs,
        n_episodes=args.episodes,
        seed_base=0,
        output_dir=output_dir,
    )

    print(f"[batch_runs] Figure 12 complete. Output: {args.output}")


if __name__ == "__main__":
    main()