#!/usr/bin/env python3
"""
CLI Adapter: ewc_ablation.py
Explicit import from exp_fig16_ewc_effectiveness
Place this file alongside exp_fig16_ewc_effectiveness.py (project root)
"""
import argparse
import sys
import os

# The real exp_fig*.py this adapter wraps lives in marl/experiments/, not alongside this file (see README_PHASE2_ADAPTERS.md —
# this package was never actually merged into that layout).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TARGET_DIR = os.path.join(_REPO_ROOT, "marl", "experiments")
if _TARGET_DIR not in sys.path:
    sys.path.insert(0, _TARGET_DIR)

from exp_fig16_ewc_effectiveness import main as fig16_main


def main():
    p = argparse.ArgumentParser(description="Adapter for ewc_ablation.py")
    p.add_argument("--system", type=str, default="8bus")
    p.add_argument("--tasks", nargs="+", type=float, default=[0.0, 0.25, 0.5, 0.75, 1.0])
    p.add_argument("--n_runs", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=str, default="outputs/fig16_ewc_effectiveness.pdf")
    args = p.parse_args()

    out_dir = os.path.dirname(args.output) if args.output else "outputs"
    os.makedirs(out_dir, exist_ok=True)

    print(f"[Adapter] ewc_ablation.py -> calling fig16_main()")
    print(f"  Output dir: {out_dir}")

    fig16_main(n_seeds=args.n_runs, output_dir=out_dir)


if __name__ == "__main__":
    main()
