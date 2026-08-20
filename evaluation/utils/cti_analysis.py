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

# NOTE: this module previously monkey-patched MAPPOMARLTrainer.__init__ to
# force use_comm=False on every instantiation process-wide, to work around
# a "matmul dimension mismatch" crash: comm-enabled policies are trained on
# comm-enhanced (obs_dim + msg_dim) input, and sampling directly from raw
# obs[i] raised ValueError for every comm-enabled method. That silently
# undid exp_fig06_cti_violation_rate.py's own correct per-method use_comm
# selection (use_comm = method in ["MAPPO-Safe-EWC", "MAPPO-Safe", "MAPPO"]),
# collapsing Figure 6's comm-aware methods to non-comm behavior everywhere.
# The root cause is already fixed at the source: evaluate_cti_severity() in
# exp_fig06_cti_violation_rate.py routes obs through trainer.comm_graph.
# forward() before calling actor.sample() whenever use_comm and a comm_graph
# are both present (the same pattern as run_core_training.py's
# _make_act_fn, fixed 2026-08-15) — so the monkeypatch is no longer needed
# and has been removed; each method's real use_comm setting now flows
# through unmodified.

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