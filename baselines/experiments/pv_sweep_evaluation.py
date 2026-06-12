#!/usr/bin/env python3
"""
baselines/experiments/pv_sweep_evaluation.py
───────────────────────────────────────────
CLI wrapper for Figure 5: Objective Function vs. PV Penetration.

Translates run_thesis_experiments.py CLI args to exp_fig05_pv_sweep_evaluation.py interface.
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
from baselines.experiments.exp_fig05_pv_sweep_evaluation import main as fig05_main


def main():
    p = argparse.ArgumentParser(description="Figure 5 — PV Sweep Evaluation (CLI wrapper)")
    p.add_argument("--system", type=str, default="8bus", help="Test system (8bus, 14bus, 30bus)")
    p.add_argument("--methods", type=str, nargs="+",
                   default=["MAPPO-Safe-EWC", "MAPPO-Safe", "MAPPO", "IPPO", "QMIX", "IQL", "Traditional"],
                   help="List of methods to evaluate")
    p.add_argument("--pv-levels", type=float, nargs="+",
                   default=[0.0, 0.25, 0.5, 0.75, 1.0],
                   help="PV penetration levels to evaluate")
    p.add_argument("--n_runs", type=int, default=50, help="Number of seeds per method-PV combo")
    p.add_argument("--output", type=str, required=True, help="Output PDF path")
    args = p.parse_args()

    # Derive output directory from output path
    output_dir = str(Path(args.output).parent)
    os.makedirs(output_dir, exist_ok=True)

    # Call the actual experiment implementation
    fig05_main(
        n_seeds=args.n_runs,
        train_steps=5000,
        n_eval=100,
        output_dir=output_dir,
    )

    print(f"[pv_sweep_evaluation] Figure 5 complete. Output: {args.output}")


if __name__ == "__main__":
    main()