#!/usr/bin/env python3
"""
Algorithm Flowchart — MAPPO-Safe-EWC Training Loop
Generated from figs.py — standalone executable script for Phase 1.
Usage: python exp_fig26.py --output <path>
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
    """Generate Algorithm Flowchart — MAPPO-Safe-EWC Training Loop"""
    os.makedirs(output_dir, exist_ok=True)


    fig, ax = plt.subplots(1, 1, figsize=(12, 14))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 14)
    ax.axis('off')
    ax.set_title('Figure 26: MAPPO-Safe-EWC Training Algorithm Flowchart', fontsize=14, fontweight='bold', pad=20)

    def draw_box(ax, x, y, w, h, text, color='#e3f2fd', text_color='black', fontsize=9, bold=True):
        rect = FancyBboxPatch((x-w/2, y-h/2), w, h, boxstyle="round,pad=0.05",
                               facecolor=color, edgecolor='#333', linewidth=1.5)
        ax.add_patch(rect)
        weight = 'bold' if bold else 'normal'
        ax.text(x, y, text, fontsize=fontsize, ha='center', va='center', 
                color=text_color, fontweight=weight, wrap=True)

    def draw_arrow(ax, x1, y1, x2, y2, color='#333'):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color=color, lw=1.5))

    def draw_diamond(ax, x, y, size, text, color='#fff3e0'):
        diamond = plt.Polygon([[x, y+size], [x+size*1.5, y], [x, y-size], [x-size*1.5, y]], 
                              facecolor=color, edgecolor='#333', linewidth=1.5)
        ax.add_patch(diamond)
        ax.text(x, y, text, fontsize=8, ha='center', va='center', fontweight='bold')

    # Start
    draw_box(ax, 6, 13.2, 2.5, 0.5, 'Initialize: θ, φ, λ, ρ, task_memory', '#c8e6c9')
    draw_arrow(ax, 6, 12.95, 6, 12.5)

    # Curriculum stage
    draw_box(ax, 6, 12.2, 3.5, 0.5, 'Set Curriculum Stage s (PV level, topology)', '#e3f2fd')
    draw_arrow(ax, 6, 11.95, 6, 11.5)

    # Episode loop
    draw_box(ax, 6, 11.2, 2, 0.4, 'For each episode e = 1, 2, ...', '#fce4ec')
    draw_arrow(ax, 6, 11.0, 6, 10.6)

    # Step loop
    draw_box(ax, 6, 10.3, 2, 0.4, 'For each step t = 1, 2, ..., T', '#fce4ec')
    draw_arrow(ax, 6, 10.1, 6, 9.7)

    # Observation
    draw_box(ax, 6, 9.4, 3, 0.5, 'Each agent i observes o_i = [V, I, P_dir, I_fault, ...]', '#e8f5e9')
    draw_arrow(ax, 6, 9.15, 6, 8.75)

    # GNN
    draw_box(ax, 6, 8.45, 3.5, 0.5, 'GNN: m_i = Encoder(o_i); m̄_i = Attention(m_j, j∈N_i)', '#fff9c4')
    draw_arrow(ax, 6, 8.2, 6, 7.8)

    # Policy
    draw_box(ax, 6, 7.5, 3, 0.5, 'Sample action: a_i ~ π_θ(·|õ_i = [o_i ‖ m̄_i])', '#ffccbc')
    draw_arrow(ax, 6, 7.25, 6, 6.85)

    # Action Projection
    draw_box(ax, 6, 6.55, 3.5, 0.5, 'Action Projection: a_i,safe = P(a_i) [bounds + CTI]', '#f8bbd9')
    draw_arrow(ax, 6, 6.3, 6, 5.9)

    # Environment step
    draw_box(ax, 6, 5.6, 3, 0.5, 'Execute: s\' = T(s, a_safe); compute r, CTI', '#d1c4e9')
    draw_arrow(ax, 6, 5.35, 6, 4.95)

    # Store
    draw_box(ax, 6, 4.65, 4, 0.5, 'Store: (o_i, õ_i, a_safe, r, o\'_i, V_φ, log π_θ)', '#b2dfdb')
    draw_arrow(ax, 6, 4.4, 6, 4.0)

    # Decision: buffer full?
    draw_diamond(ax, 6, 3.6, 0.35, 'Buffer\nFull?')
    draw_arrow(ax, 6, 3.25, 6, 2.85)
    ax.text(6.8, 3.6, 'No', fontsize=8, color='#333')
    ax.annotate('', xy=(6, 10.3), xytext=(6, 3.95),
                arrowprops=dict(arrowstyle='->', color='#999', lw=1, connectionstyle="arc3,rad=0.3"))
    ax.text(9.5, 7, 'Continue', fontsize=8, color='#999', rotation=90)

    # Update
    draw_box(ax, 6, 2.55, 4, 0.5, 'PPO Update: ∇L_PPO + EWC + AL constraint', '#ffccbc')
    draw_arrow(ax, 6, 2.3, 6, 1.9)

    # Decision: task change?
    draw_diamond(ax, 6, 1.55, 0.35, 'Task\nChange?')
    draw_arrow(ax, 6, 1.2, 6, 0.8)
    ax.text(6.8, 1.55, 'No', fontsize=8, color='#333')
    ax.annotate('', xy=(6, 11.2), xytext=(6, 1.9),
                arrowprops=dict(arrowstyle='->', color='#999', lw=1, connectionstyle="arc3,rad=-0.4"))
    ax.text(2.5, 6, 'Next episode', fontsize=8, color='#999', rotation=90)

    # EWC consolidation
    draw_box(ax, 6, 0.55, 4, 0.5, 'EWC: Consolidate; Save θ*, F; Warm-start', '#c8e6c9')
    ax.annotate('', xy=(6, 11.2), xytext=(6, 0.8),
                arrowprops=dict(arrowstyle='->', color='#999', lw=1, connectionstyle="arc3,rad=0.5"))
    ax.text(10.5, 6, 'Next stage', fontsize=8, color='#999', rotation=90)

    # Side annotations
    ax.text(0.5, 8.45, 'DECENTRALIZED\nEXECUTION', fontsize=8, ha='center', color='#1565c0', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#e3f2fd', edgecolor='#1565c0'))
    ax.text(0.5, 4.65, 'CENTRALIZED\nTRAINING', fontsize=8, ha='center', color='#c62828', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#ffebee', edgecolor='#c62828'))

    plt.tight_layout()
    plt.savefig(output_dir + '/fig26_algorithm_flowchart_a.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Algorithm Flowchart — MAPPO-Safe-EWC Training Loop')
    parser.add_argument('--output', type=str, default='results/figs_01_10',
                        help='Output directory for the figure')
    args = parser.parse_args()
    generate_figure(args.output)
    print(f"Figure saved to {args.output}")
