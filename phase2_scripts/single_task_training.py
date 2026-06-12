#!/usr/bin/env python3
"""
single_task_training.py — CLI adapter for Figure 9

Maps: run_thesis_experiments.py --figure 9
  To: exp_fig09_cti_violation_dynamics.py

Handles:
  --system {8bus,30bus}      → network selection
  --pv {0.0,0.25,0.5,0.75,1.0} → curriculum_stage = int(pv * 5)
  --method {MAPPO-Safe,MAPPO}  → use_safety flag
  --episodes N               → --n_episodes N
  --seed S                   → --seed S
  --output PATH              → --output_dir dirname(PATH)
"""

import argparse
import sys
import os
from pathlib import Path

# Add parent to path so we can import exp_fig09
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import the actual experiment module
import importlib.util
spec = importlib.util.spec_from_file_location(
    "exp_fig09", 
    str(Path(__file__).parent.parent / "exp_fig09_cti_violation_dynamics.py")
)
exp_fig09 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exp_fig09)


def main():
    p = argparse.ArgumentParser(description="Figure 9 adapter — CTI Violation Dynamics")
    p.add_argument("--system", type=str, default="8bus", choices=["8bus", "30bus"])
    p.add_argument("--pv", type=float, default=1.0)
    p.add_argument("--method", type=str, default="MAPPO-Safe", 
                   choices=["MAPPO-Safe", "MAPPO-Safe-EWC", "MAPPO", "IPPO", "QMIX", "IQL", "Traditional"])
    p.add_argument("--episodes", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=str, default="outputs/fig09_cti_violations.pdf")
    
    args = p.parse_args()
    
    # Derive output directory from --output path
    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Map PV level to curriculum stage (0.0→0, 0.25→1, 0.5→2, 0.75→3, 1.0→5)
    curriculum_stage = int(args.pv * 5)
    
    print(f"[Adapter] Fig 9: system={args.system}, pv={args.pv}→stage={curriculum_stage}, "
          f"method={args.method}, episodes={args.episodes}, seed={args.seed}")
    print(f"[Adapter] Output dir: {output_dir}")
    
    # The exp_fig09 script uses --n_episodes and --output_dir
    # We need to monkey-patch the network