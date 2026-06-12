#!/usr/bin/env python3
"""
CLI Adapter: ewc_ablation.py
Maps: run_thesis_experiments.py --phase3 Fig 16 call
  to: exp_fig16_ewc_effectiveness.py

run_thesis_experiments.py calls:
  python marl/experiments/ewc_ablation.py --system 8bus --tasks 0.0 0.25 0.5 0.75 1.0 \
      --n_runs 10 --output "...fig_16_ewc_effectiveness.pdf"
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import importlib.util
spec = importlib.util.spec_from_file_location(
    "exp_fig16", os.path.join(os.path.dirname(__file__), "..", "exp_fig16_ewc_effectiveness.py")
)
exp_fig16 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exp_fig16)


def main():
    p = argparse.ArgumentParser(description="EWC Ablation Adapter (Fig 16)")
    p.add_argument("--system", type=str, default="8bus")
    p.add_argument("--tasks", nargs="+", type=float, default=[0.0, 0.25, 0.5, 0.75, 1.0])
    p.add_argument("--n_runs", type=int, default=10)
    p.add_argument("--output", type=str, default="outputs/fig16_ewc_effectiveness.pdf")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    print(f"[Adapter] Fig 16 EWC Ablation -> exp_fig16_ewc_effectiveness")
    print(f"  system={args.system}, tasks={args.tasks}, n_runs={args.n_runs}")

    out_dir = os.path.dirname(args.output) or "outputs"
    os.makedirs(out_dir, exist_ok=True)

    # The actual experiment already has built-in EWC vs No-EWC comparison
    exp_fig16.main(n_seeds=args.n_runs, output_dir=out_dir)


if __name__ == "__main__":
    main()
