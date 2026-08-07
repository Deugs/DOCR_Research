#!/usr/bin/env python3
"""
CLI Adapter: lagrange_analysis.py
Explicit import from exp_fig10_lagrange_evolution
Place this file alongside exp_fig10_lagrange_evolution.py (project root)
"""
import argparse
import sys
import os

# The real exp_fig*.py this adapter wraps lives in marl/utils/, not alongside this file (see README_PHASE2_ADAPTERS.md —
# this package was never actually merged into that layout).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TARGET_DIR = os.path.join(_REPO_ROOT, "marl", "utils")
if _TARGET_DIR not in sys.path:
    sys.path.insert(0, _TARGET_DIR)

from exp_fig10_lagrange_evolution import main as fig10_main


def main():
    p = argparse.ArgumentParser(description="Adapter for lagrange_analysis.py")
    p.add_argument("--log", type=str, default="", help="Ignored — runs experiment directly")
    p.add_argument("--output", type=str, default="outputs/fig10_lagrange.pdf")
    p.add_argument("--n_episodes", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    out_dir = os.path.dirname(args.output) if args.output else "outputs"
    os.makedirs(out_dir, exist_ok=True)

    print(f"[Adapter] lagrange_analysis.py -> calling fig10_main()")
    print(f"  Output dir: {out_dir}")

    fig10_main(n_episodes=args.n_episodes, seed=args.seed, output_dir=out_dir)


if __name__ == "__main__":
    main()
