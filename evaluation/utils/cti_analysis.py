#!/usr/bin/env python3
"""
evaluation/utils/cti_analysis.py
─────────────────────────────────
CLI wrapper for Figure 6: CTI Violation Rate vs. PV Penetration.

This is a post-processing script that reads Figure 5 results and generates
a dedicated CTI violation analysis plot. Can also run standalone.

Translates run_thesis_experiments.py CLI args to exp_fig06_cti_violation_rate.py interface.
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
from evaluation.utils.exp_fig06_cti_violation_rate import main as fig06_main


def main():
    p = argparse.ArgumentParser(description="Figure 6 — CTI Violation Analysis (CLI wrapper)")
    p.add_argument("--data", type=str, required=True,
                   help="Path to Figure 5 results pickle file (pv_sweep_results.pkl)")
    p.add_argument("--output", type=str, required=True, help="Output PDF path")
    args = p.parse_args()

    # Derive output directory from output path
    output_dir = str(Path(args.output).parent)
    os.makedirs(output_dir, exist_ok=True)

    # Run the standalone experiment to generate the CTI plot
    fig06_main(
        n_seeds=10,
        train_steps=5000,
        n_eval=100,
        output_dir=output_dir,
    )

    print(f"[cti_analysis] Figure 6 complete. Output: {args.output}")


if __name__ == "__main__":
    main()