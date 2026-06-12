#!/usr/bin/env python3
"""
CLI Adapter: safety_projection_analysis.py
Explicit import from exp_fig21_action_projection
Place this file alongside exp_fig21_action_projection.py (project root)
"""
import argparse
import sys
import os

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from exp_fig21_action_projection import main as fig21_main


def main():
    p = argparse.ArgumentParser(description="Adapter for safety_projection_analysis.py")
    p.add_argument("--log", type=str, default="", help="Ignored — runs experiment directly")
    p.add_argument("--output", type=str, default="outputs/fig21_projection.pdf")
    p.add_argument("--n_episodes", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    out_dir = os.path.dirname(args.output) if args.output else "outputs"
    os.makedirs(out_dir, exist_ok=True)

    print(f"[Adapter] safety_projection_analysis.py -> calling fig21_main()")
    print(f"  Output dir: {out_dir}")

    fig21_main(n_episodes=args.n_episodes, seed=args.seed, output_dir=out_dir)


if __name__ == "__main__":
    main()
