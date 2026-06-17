#!/usr/bin/env python3
"""
marl/experiments/ablation_study.py
──────────────────────────────────
CLI wrapper for Figure 39: Systematic Ablation Study.

Translates run_thesis_experiments.py CLI args to exp_fig39_ablation.py interface.
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
from marl.experiments.exp_fig39_ablation import main as fig39_main


def main():
    p = argparse.ArgumentParser(description="Figure 39 — Ablation Study (CLI wrapper)")
    p.add_argument("--system", type=str, default="8bus", help="Test system")
    p.add_argument("--pv", type=float, default=1.0, help="PV penetration level")
    p.add_argument("--configs", type=str, nargs="+",
                   default=["MAPPO", "MAPPO-GNN", "MAPPO-GNN-Safety",
                            "MAPPO-GNN-Safety-EWC", "MAPPO-Safe-EWC-Curriculum"],
                   help="Ablation configurations to test")
    p.add_argument("--n_runs", type=int, default=10, help="Number of seeds per config")
    p.add_argument("--output", type=str, required=True, help="Output PDF path")
    args = p.parse_args()

    # Derive output directory from output path
    output_dir = str(Path(args.output).parent)
    os.makedirs(output_dir, exist_ok=True)

    # Call the actual experiment implementation
    fig39_main(
        n_seeds=args.n_runs,
        output_dir=output_dir,
        system=args.system,
    )

    print(f"[ablation_study] Figure 39 complete. Output: {args.output}")


if __name__ == "__main__":
    main()