#!/usr/bin/env python3
"""
CLI Adapter: single_task_training.py
Maps: run_thesis_experiments.py --phase3 Fig 9 call
  to: exp_fig09_cti_violation_dynamics.py

run_thesis_experiments.py calls:
  python marl/experiments/single_task_training.py --system 8bus --pv 1.0 \
      --method MAPPO-Safe --episodes 5000 --seed 42 --output "...fig_09_cti_violations.pdf"
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import importlib.util
spec = importlib.util.spec_from_file_location(
    "exp_fig09", os.path.join(os.path.dirname(__file__), "..", "exp_fig09_cti_violation_dynamics.py")
)
exp_fig09 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exp_fig09)


def main():
    p = argparse.ArgumentParser(description="Single Task Training Adapter (Fig 9)")
    p.add_argument("--system", type=str, default="8bus")
    p.add_argument("--pv", type=float, default=1.0)
    p.add_argument("--method", type=str, default="MAPPO-Safe")
    p.add_argument("--episodes", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=str, default="outputs/fig09_cti_violations.pdf")
    args = p.parse_args()

    print(f"[Adapter] Fig 9 CTI Violations -> exp_fig09_cti_violation_dynamics")
    print(f"  system={args.system}, pv={args.pv}, method={args.method}, episodes={args.episodes}")

    out_dir = os.path.dirname(args.output) or "outputs"
    os.makedirs(out_dir, exist_ok=True)

    # Map pv level to curriculum stage: 1.0 -> stage 5
    # The actual script uses curriculum_stage parameter
    exp_fig09.main(n_episodes=args.episodes, seed=args.seed, output_dir=out_dir)


if __name__ == "__main__":
    main()
