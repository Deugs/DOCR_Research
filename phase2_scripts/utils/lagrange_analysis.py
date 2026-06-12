#!/usr/bin/env python3
"""
CLI Adapter: lagrange_analysis.py
Maps run_thesis_experiments.py calls to actual experiment files.
Uses robust file search to find exp_fig*.py regardless of project structure.
"""
import argparse
import sys
import os
from pathlib import Path
import importlib.util

def _find_exp_file(filename):
    """Search for experiment file in common locations."""
    script_dir = Path(__file__).resolve().parent
    search_roots = [script_dir]
    for _ in range(3):
        parent = script_dir.parent
        if parent != script_dir:
            search_roots.append(parent)
            script_dir = parent
        else:
            break

    search_paths = []
    for root in search_roots:
        search_paths.extend([
            root / filename,
            root / "scripts" / filename,
            root / "experiments" / filename,
            root / "src" / filename,
            root / "marl" / filename,
            root / "phase2_scripts" / filename,
        ])

    for p in search_paths:
        if p.exists():
            return str(p.resolve())

    cwd_paths = [Path(os.getcwd()) / filename, Path(os.getcwd()).parent / filename]
    for p in cwd_paths:
        if p.exists():
            return str(p.resolve())

    raise FileNotFoundError(f"Could not find {filename}. Searched: {[str(p) for p in search_paths + cwd_paths]}")


def main():
    # Parse args that run_thesis_experiments.py passes
    p = argparse.ArgumentParser(description="Adapter for lagrange_analysis.py")
    p.add_argument("--system", type=str, default="8bus")
    p.add_argument("--pv", type=float, default=1.0)
    p.add_argument("--method", type=str, default="MAPPO-Safe")
    p.add_argument("--episodes", type=int, default=5000)
    p.add_argument("--n_episodes", type=int, default=2000)
    p.add_argument("--n_runs", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=str, default="outputs/fig_stub.pdf")
    p.add_argument("--tasks", nargs="+", type=float, default=None)
    p.add_argument("--log", type=str, default="")
    args = p.parse_args()

    # Find and load the actual experiment module
    target_file = "exp_fig10_lagrange_evolution.py"
    try:
        exp_path = _find_exp_file(target_file)
    except FileNotFoundError as e:
        print(f"[Adapter] ERROR: {e}")
        print(f"[Adapter] Falling back to stub mode (synthetic data)...")
        _run_stub(args)
        return

    print(f"[Adapter] Found actual experiment: {exp_path}")

    # Add the experiment's directory to sys.path so its imports work
    exp_dir = str(Path(exp_path).parent)
    if exp_dir not in sys.path:
        sys.path.insert(0, exp_dir)

    spec = importlib.util.spec_from_file_location("exp_fig10", exp_path)
    exp_fig10 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exp_fig10)

    out_dir = os.path.dirname(args.output) or "outputs"
    os.makedirs(out_dir, exist_ok=True)

    exp_fig10.main(n_episodes=args.n_episodes, seed=args.seed, output_dir=out_dir)


def _run_stub(args):
    """Generate a synthetic placeholder when actual experiment file is not found."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    out_dir = os.path.dirname(args.output) or "outputs"
    os.makedirs(out_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.text(0.5, 0.5, "STUB: Actual experiment file not found\n\n" + 
            "exp_fig10_lagrange_evolution.py\nwas not found in project.", 
            ha="center", va="center", fontsize=12, transform=ax.transAxes)
    ax.set_title("Placeholder - lagrange_analysis", fontsize=14)
    ax.axis("off")
    fig.savefig(args.output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[Adapter] Saved stub: {args.output}")


if __name__ == "__main__":
    main()
