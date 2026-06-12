#!/usr/bin/env python3
"""
marl/experiments/hyperparameter_grid.py
───────────────────────────────────────
CLI wrapper for Figure 40: Hyperparameter Sensitivity Heatmap.

Translates run_thesis_experiments.py CLI args to exp_fig40_hparam_grid.py interface.
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
from marl.experiments.exp_fig40_hparam_grid import main as fig40_main


def main():
    p = argparse.ArgumentParser(description="Figure 40 — Hyperparameter Grid (CLI wrapper)")
    p.add_argument("--system", type=str, default="8bus", help="Test system")
    p.add_argument("--pv", type=float, default=1.0, help="PV penetration level")
    p.add_argument("--lambda-ewc", type=float, nargs="+",
                   default=[1e2, 1e3, 1e4, 1e5, 1e6],
                   help="EWC regularization strength values")
    p.add_argument("--alpha-lambda", type=float, nargs="+",
                   default=[1e-5, 1e-4, 1e-3, 1e-2],
                   help="Lagrange multiplier learning rates")
    p.add_argument("--hidden-dims", type=int, nargs="+",
                   default=[16, 32, 64, 128],
                   help="Hidden layer dimensions")
    p.add_argument("--n_runs", type=int, default=3, help="Seeds per config")
    p.add_argument("--episodes", type=int, default=2000, help="Training episodes per config")
    p.add_argument("--output", type=str, required=True, help="Output PDF path")
    args = p.parse_args()

    # Derive output directory from output path
    output_dir = str(Path(args.output).parent)
    os.makedirs(output_dir, exist_ok=True)

    # Call the actual experiment implementation
    fig40_main(
        n_seeds=args.n_runs,
        output_dir=output_dir,
    )

    print(f"[hyperparameter_grid] Figure 40 complete. Output: {args.output}")


if __name__ == "__main__":
    main()