#!/usr/bin/env python3
"""
Dec-POMDP Formulation Diagram
Generated from figs.py — standalone executable script for Phase 1.
Usage: python exp_fig27.py --output <path>
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
    """Generate Dec-POMDP Formulation Diagram"""
    os.makedirs(output_dir, exist_ok=True)


    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('Figure 27: Dec-POMDP Formulation for DOCR Coordination', fontsize=14, fontweight='bold', pad=20)

    # Central state box
    state_rect = FancyBboxPatch((4, 7.5), 4, 1.5, boxstyle="round,pad=0.1",
                                 facecolor='#e3f2fd', edgecolor='#1976d2', linewidth=2)
    ax.add_patch(state_rect)
    ax.text(6, 8.6, 'Global State s ∈ S', fontsize=11, ha='center', fontweight='bold', color='#1565c0')
    ax.text(6, 8.1, '[TMS_all ‖ PS_all ‖ PV_level ‖ topology ‖ fault_type]', fontsize=8, ha='center', color='#333')
    ax.text(6, 7.75, 'ξ: cloud transitions, load fluctuations', fontsize=8, ha='center', color='#666', style='italic')

    # Agents
    agent_colors = ['#ff9800', '#4caf50', '#2196f3', '#9c27b0']
    agent_names = ['Agent 1\n(Relay R1)', 'Agent 2\n(Relay R2)', 'Agent i\n(Relay Ri)', 'Agent N\n(Relay RN)']
    for i, (x, color, name) in enumerate(zip([1, 3.5, 6, 8.5], agent_colors, agent_names)):
        # Agent box
        rect = FancyBboxPatch((x-0.7, 4.5), 1.4, 2, boxstyle="round,pad=0.05",
                               facecolor=color, edgecolor='black', linewidth=1.5, alpha=0.3)
        ax.add_patch(rect)
        ax.text(x, 6.0, name, fontsize=8, ha='center', va='center', fontweight='bold')
    
        # Observation
        ax.text(x, 5.4, f'o_{i+1} ∈ O', fontsize=8, ha='center', va='center', style='italic')
        ax.text(x, 5.0, 'ℝ¹⁰ local', fontsize=7, ha='center', va='center', color='#666')
    
        # Action
        ax.text(x, 4.8, f'a_{i+1} ∈ A', fontsize=8, ha='center', va='center', style='italic')
        ax.text(x, 4.4, '(ΔTMS, ΔPS)', fontsize=7, ha='center', va='center', color='#666')
    
        # Arrows from state to observation
        ax.annotate('', xy=(x, 6.5), xytext=(6, 7.5),
                    arrowprops=dict(arrowstyle='->', color='#999', lw=1, 
                                   connectionstyle=f"arc3,rad={0.2-i*0.13}"))
    
        # Arrows from action to environment
        ax.annotate('', xy=(x, 3.5), xytext=(x, 4.5),
                    arrowprops=dict(arrowstyle='->', color=color, lw=1.5))

    # Environment
    env_rect = FancyBboxPatch((3, 2.5), 6, 1, boxstyle="round,pad=0.1",
                               facecolor='#e8f5e9', edgecolor='#388e3c', linewidth=2)
    ax.add_patch(env_rect)
    ax.text(6, 3.15, 'Environment: T(s\'|s, a, ξ)', fontsize=11, ha='center', fontweight='bold', color='#2e7d32')
    ax.text(6, 2.75, 'pandapower AC power flow + short-circuit + PV inverter dynamics', fontsize=8, ha='center', color='#333')

    # Reward
    reward_rect = FancyBboxPatch((4.5, 1.0), 3, 1, boxstyle="round,pad=0.1",
                                  facecolor='#fff3e0', edgecolor='#f57c00', linewidth=2)
    ax.add_patch(reward_rect)
    ax.text(6, 1.7, 'Team Reward R', fontsize=11, ha='center', fontweight='bold', color='#e65100')
    ax.text(6, 1.3, 'R = w₁·I_CTI - w₂·CTI_viol - w₃·T_total - w₄·C_comm + w₅·v_adapt', fontsize=7, ha='center', color='#333')

    # Arrows from env to reward
    ax.annotate('', xy=(6, 2.0), xytext=(6, 2.5),
                arrowprops=dict(arrowstyle='->', color='#333', lw=1.5))

    # Arrows from reward back to state (loop)
    ax.annotate('', xy=(6, 7.5), xytext=(9.5, 1.5),
                arrowprops=dict(arrowstyle='->', color='#333', lw=1.5, 
                               connectionstyle="arc3,rad=0.5"))
    ax.text(10.5, 4.5, 'γ = 0.99', fontsize=9, color='#333', fontweight='bold', rotation=90)

    # Communication arrows between agents
    for i in range(3):
        x1 = [1, 3.5, 6][i]
        x2 = [3.5, 6, 8.5][i]
        ax.annotate('', xy=(x2-0.7, 5.5), xytext=(x1+0.7, 5.5),
                    arrowprops=dict(arrowstyle='<->', color='#c62828', lw=1, linestyle='--'))
    ax.text(4.75, 5.7, 'GNN Messages (8-float, 0.256 kbps)', fontsize=8, ha='center', color='#c62828', style='italic')

    plt.tight_layout()
    plt.savefig(output_dir + '/fig27_decpomdp.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.savefig(output_dir + '/fig27_decpomdp.pdf', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dec-POMDP Formulation Diagram')
    parser.add_argument('--output', type=str, default='results/figs_01_10',
                        help='Output directory for the figure')
    args = parser.parse_args()
    generate_figure(args.output)
    print(f"Figure saved to {args.output}")
