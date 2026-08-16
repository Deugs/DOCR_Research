#!/usr/bin/env python3
"""
CLI Adapter: curriculum_trainer.py
Explicit import from exp_fig07_curriculum_learning
Place this file alongside exp_fig07_curriculum_learning.py (project root)
"""
import argparse
import sys
import os

# The real exp_fig*.py this adapter wraps lives in marl/ (top level), not alongside this file (see README_PHASE2_ADAPTERS.md —
# this package was never actually merged into that layout).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TARGET_DIR = os.path.join(_REPO_ROOT, "marl")
if _TARGET_DIR not in sys.path:
    sys.path.insert(0, _TARGET_DIR)

from exp_fig07_curriculum_learning import main as fig07_main


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

    print(f"[Adapter] curriculum_trainer.py -> calling fig07_main()")
    print(f"  Output dir: {out_dir}")

    fig07_main(network=args.system, episodes_per_task=args.episodes,
               n_seeds=args.n_runs, seed=args.seed, output_dir=out_dir)


if __name__ == "__main__":
    main()
