#!/usr/bin/env python3
"""
Safe RL Dual Approach — Action Projection + Augmented Lagrangian
Generated from figs.py — standalone executable script for Phase 1.
Usage: python exp_fig28.py --output <path>
"""

import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle, Polygon, Wedge
import numpy as np


# Global style
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.edgecolor': 'black',
    'axes.labelcolor': 'black',
    'axes.titlecolor': 'black',
    'xtick.color': 'black',
    'ytick.color': 'black',
    'grid.color': '#cccccc',
    'grid.linewidth': 0.5,
    'text.color': 'black',
    'legend.facecolor': 'white',
    'legend.edgecolor': 'black',
    'font.family': 'serif',
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'figure.dpi': 150,
})



def generate_figure(output_dir):
    """Generate Safe RL Dual Approach — Action Projection + Augmented Lagrangian"""
    os.makedirs(output_dir, exist_ok=True)


    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('Figure 28: Dual Safe RL Approach — Immediate + Learned Safety', fontsize=14, fontweight='bold', pad=20)

    # Input
    draw_box = lambda x, y, w, h, text, color: (
        ax.add_patch(FancyBboxPatch((x-w/2, y-h/2), w, h, boxstyle="round,pad=0.05",
                                     facecolor=color, edgecolor='#333', linewidth=1.5)),
        ax.text(x, y, text, fontsize=9, ha='center', va='center', fontweight='bold', wrap=True)
    )

    # Proposed action
    ax.add_patch(FancyBboxPatch((4.5, 8.5), 3, 0.6, boxstyle="round,pad=0.05",
                                 facecolor='#e3f2fd', edgecolor='#1976d2', linewidth=2))
    ax.text(6, 8.8, 'Proposed Action: a = (ΔTMS, ΔPS)', fontsize=10, ha='center', fontweight='bold')

    # Split arrow
    ax.annotate('', xy=(3, 7.8), xytext=(5, 8.5),
                arrowprops=dict(arrowstyle='->', color='#333', lw=2))
    ax.annotate('', xy=(9, 7.8), xytext=(7, 8.5),
                arrowprops=dict(arrowstyle='->', color='#333', lw=2))

    # LEFT: Action Projection (Immediate Safety)
    ax.add_patch(FancyBboxPatch((0.5, 5.5), 5, 2.5, boxstyle="round,pad=0.1",
                                 facecolor='#ffebee', edgecolor='#c62828', linewidth=2))
    ax.text(3, 7.6, 'ACTION PROJECTION', fontsize=11, ha='center', fontweight='bold', color='#c62828')
    ax.text(3, 7.2, '(Immediate Safety)', fontsize=9, ha='center', color='#c62828', style='italic')

    # Sub-steps
    steps_left = [
        (3, 6.6, '1. TMS bounds: clip(TMS + ΔTMS, 0.05, 1.1)'),
        (3, 6.1, '2. PS bounds: clip(PS + ΔPS, PS_min, PS_max)'),
        (3, 5.6, '3. CTI lookahead: binary search for feasible β*'),
    ]
    for x, y, text in steps_left:
        ax.text(x, y, text, fontsize=8, ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='#c62828', alpha=0.8))

    # RIGHT: Augmented Lagrangian (Learned Safety)
    ax.add_patch(FancyBboxPatch((6.5, 5.5), 5, 2.5, boxstyle="round,pad=0.1",
                                 facecolor='#e8f5e9', edgecolor='#388e3c', linewidth=2))
    ax.text(9, 7.6, 'AUGMENTED LAGRANGIAN', fontsize=11, ha='center', fontweight='bold', color='#2e7d32')
    ax.text(9, 7.2, '(Learned Safety)', fontsize=9, ha='center', color='#2e7d32', style='italic')

    steps_right = [
        (9, 6.6, '1. Compute constraint: g_j = 0.3 - (t_bkp - t_pri)'),
        (9, 6.1, '2. AL cost: L_AL = Σ λ_j·g_j + (ρ/2)·Σ max(0,g_j)²'),
        (9, 5.6, '3. Dual ascent: λ_{k+1} = max(0, λ_k + α·ḡ_j) [PID]'),
    ]
    for x, y, text in steps_right:
        ax.text(x, y, text, fontsize=8, ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='#388e3c', alpha=0.8))

    # Merge arrow
    ax.annotate('', xy=(6, 4.2), xytext=(3, 5.5),
                arrowprops=dict(arrowstyle='->', color='#333', lw=2))
    ax.annotate('', xy=(6, 4.2), xytext=(9, 5.5),
                arrowprops=dict(arrowstyle='->', color='#333', lw=2))

    # Output: Safe action
    ax.add_patch(FancyBboxPatch((4, 3.5), 4, 0.8, boxstyle="round,pad=0.1",
                                 facecolor='#c8e6c9', edgecolor='#2e7d32', linewidth=2))
    ax.text(6, 4.0, 'Safe Action: a_safe = P(a)\nPolicy Loss: L = L_PPO + L_AL + L_EWC', 
            fontsize=9, ha='center', va='center', fontweight='bold')

    # Execute arrow
    ax.annotate('', xy=(6, 2.5), xytext=(6, 3.5),
                arrowprops=dict(arrowstyle='->', color='#333', lw=2))

    # Environment
    ax.add_patch(FancyBboxPatch((3.5, 1.5), 5, 1, boxstyle="round,pad=0.1",
                                 facecolor='#d1c4e9', edgecolor='#512da8', linewidth=2))
    ax.text(6, 2.1, 'Environment Step → (r, s\', done)', fontsize=10, ha='center', fontweight='bold')

    # Key insight boxes
    ax.text(0.3, 4.5, 'Prevents ALL\nviolations during\ntraining & deploy', fontsize=8, ha='center', color='#c62828',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#ffebee', edgecolor='#c62828'))
    ax.text(11.7, 4.5, 'Shapes policy\ntoward inherently\nsafe behavior', fontsize=8, ha='center', color='#2e7d32',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#e8f5e9', edgecolor='#388e3c'))

    # Bottom insight
    ax.text(6, 0.5, 'Rationale: Projection provides physical safety (hard guarantee); AL provides convergence guarantee (soft shaping).\nNeither alone is sufficient — dual approach achieves both immediate and learned safety.',
            fontsize=9, ha='center', style='italic',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#f5f5f5', edgecolor='#999'))

    plt.tight_layout()
    plt.savefig(output_dir + '/fig28_safe_rl_dual.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Safe RL Dual Approach — Action Projection + Augmented Lagrangian')
    parser.add_argument('--output', type=str, default='results/figs_01_10',
                        help='Output directory for the figure')
    args = parser.parse_args()
    generate_figure(args.output)
    print(f"Figure saved to {args.output}")
