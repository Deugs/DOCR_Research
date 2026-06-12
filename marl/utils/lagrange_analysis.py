#!/usr/bin/env python3
"""
CLI Adapter: lagrange_analysis.py
Maps: run_thesis_experiments.py --phase3 Fig 10 call
  to: exp_fig10_lagrange_evolution.py

run_thesis_experiments.py calls:
  python marl/utils/lagrange_analysis.py --log "...single_task_8bus_100pv.json" \
      --output "...fig_10_lagrange_multipliers.pdf"

We ignore the --log argument (which doesn't exist) and run the actual experiment.
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import importlib.util
spec = importlib.util.spec_from_file_location(
    "exp_fig10", os.path.join(os.path.dirname(__file__), "..", "exp_fig10_lagrange_evolution.py")
)
exp_fig10 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exp_fig10)


def main():
    p = argparse.ArgumentParser(description="Lagrange Analysis Adapter (Fig 10)")
    p.add_argument("--log", type=str, default="", help="Ignored — runs experiment directly")
    p.add_argument("--output", type=str, default="outputs/fig10_lagrange.pdf")
    p.add_argument("--n_episodes", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    print(f"[Adapter] Fig 10 Lagrange -> exp_fig10_lagrange_evolution")
    print(f"  log arg ignored, running experiment with n_episodes={args.n_episodes}")

    out_dir = os.path.dirname(args.output) or "outputs"
    os.makedirs(out_dir, exist_ok=True)

    exp_fig10.main(n_episodes=args.n_episodes, seed=args.seed, output_dir=out_dir)


if __name__ == "__main__":
    main()
