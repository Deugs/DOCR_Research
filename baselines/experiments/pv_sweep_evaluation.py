#!/usr/bin/env python3
"""
baselines/experiments/pv_sweep_evaluation.py  —  FIXED CLI Wrapper for Figure 5
───────────────────────────────────────────────────────────────────────────────
Translates run_thesis_experiments.py CLI args to unified_exp_fig05.py interface.

FIXES APPLIED (vs original buggy version):
  1. --system is now ACTUALLY PASSED through to the experiment (was dropped)
  2. --methods is now ACTUALLY PASSED through (was accepted by argparse and
     only echoed in the final print -- never forwarded to fig05_main, so
     the hardcoded default method set ran regardless of what was requested)
  3. --pv-levels is now ACTUALLY PASSED through (same bug as --methods)
  4. Imports unified_exp_fig05 which includes metaheuristic baselines
"""

import argparse
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import the unified experiment implementation
from baselines.experiments.unified_exp_fig05 import main as fig05_main


def main():
    p = argparse.ArgumentParser(
        description="Figure 5 — PV Sweep Evaluation (CLI wrapper, FIXED)"
    )
    p.add_argument(
        "--system", type=str, default="8bus",
        choices=["8bus", "30bus", "39bus"],
        help="Test system (8bus, 30bus, 39bus)"
    )
    p.add_argument(
        "--methods", type=str, nargs="+",
        default=["MAPPO-Safe-EWC", "MAPPO-Safe", "MAPPO", "IPPO",
                 "QMIX", "IQL", "GA-SQP", "HWOA", "HHO", "JAYA", "EEO",
                 "Traditional"],
        help="List of methods to evaluate"
    )
    p.add_argument(
        "--pv-levels", type=float, nargs="+",
        default=[0.0, 0.25, 0.5, 0.75, 1.0],
        help="PV penetration levels to evaluate"
    )
    p.add_argument(
        "--n_runs", type=int, default=10,
        help="Number of seeds per method-PV combo"
    )
    p.add_argument(
        "--output", type=str, required=True,
        help="Output directory path (parent of system subdir)"
    )
    args = p.parse_args()

    # Derive output directory from output path
    output_dir = str(Path(args.output).parent) if "." in args.output else args.output
    os.makedirs(output_dir, exist_ok=True)

    # ── FIX: Actually pass --system / --methods / --pv-levels through ───────
    fig05_main(
        system=args.system,
        n_seeds=args.n_runs,
        train_steps=5000,
        n_eval=100,
        output_dir=output_dir,
        methods=args.methods,
        pv_levels=args.pv_levels,
    )

    print(f"[pv_sweep_evaluation] Figure 5 complete for {args.system}.")
    print(f"  Output dir: {output_dir}/{args.system}/")
    print(f"  Methods requested: {args.methods}")
    print(f"  PV levels: {args.pv_levels}")


if __name__ == "__main__":
    main()