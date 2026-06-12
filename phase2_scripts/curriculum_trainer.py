#!/usr/bin/env python3
"""
CLI Adapter: curriculum_trainer.py
Explicit import from exp_fig09_cti_violation_dynamics
Place this file alongside exp_fig09_cti_violation_dynamics.py (project root)
"""
import argparse
import sys
import os

# Add project root to path
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from exp_fig09_cti_violation_dynamics import main as fig09_main


def main():
    p = argparse.ArgumentParser(description="Adapter for curriculum_trainer.py")
    p.add_argument("--system", type=str, default="8bus")
    p.add_argument("--tasks", nargs="+", type=float, default=[0.0, 0.25, 0.5, 0.75, 1.0])
    p.add_argument("--episodes", type=int, default=5000)
    p.add_argument("--n_runs", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=str, default="outputs/fig07_curriculum.pdf")
    args = p.parse_args()

    out_dir = os.path.dirname(args.output) if args.output else "outputs"
    os.makedirs(out_dir, exist_ok=True)

    print(f"[Adapter] curriculum_trainer.py -> calling fig09_main()")
    print(f"  Output dir: {out_dir}")

    fig09_main(n_episodes=args.episodes, seed=args.seed, output_dir=out_dir)


if __name__ == "__main__":
    main()
