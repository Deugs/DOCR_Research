#!/usr/bin/env python3
"""
EWC Continual Learning Mechanism
Generated from figs.py — standalone executable script for Phase 1.
Usage: python exp_fig29.py --output <path>
"""

import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle, Polygon, Wedge
import numpy as np
from matplotlib.patches import Ellipse


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
    """Generate EWC Continual Learning Mechanism"""
    os.makedirs(output_dir, exist_ok=True)


    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('Figure 29: Elastic Weight Consolidation (EWC) for Continual Adaptation', fontsize=14, fontweight='bold', pad=20)

    # Task timeline
    timeline_y = 9.2
    ax.plot([1, 11], [timeline_y, timeline_y], 'k-', linewidth=2)
    for i, (x, label) in enumerate([(2, 'Task 1\n(0% PV)'), (4.5, 'Task 2\n(20% PV)'), 
                                      (7, 'Task 3\n(60% PV)'), (9.5, 'Task 4\n(100% PV)')]):
        ax.plot(x, timeline_y, 'o', color='#1976d2', markersize=12, zorder=5)
        ax.text(x, timeline_y + 0.3, label, fontsize=8, ha='center', fontweight='bold')
        if i < 3:
            ax.annotate('', xy=(x+2.2, timeline_y), xytext=(x+0.3, timeline_y),
                        arrowprops=dict(arrowstyle='->', color='#999', lw=1.5))

    ax.text(6, 9.8, 'Task Sequence (Curriculum Stages)', fontsize=11, ha='center', fontweight='bold')

    # Central diagram: Parameter space
    ax.add_patch(Circle((6, 5.5), 2.5, facecolor='#f5f5f5', edgecolor='#333', linewidth=2, alpha=0.5))
    ax.text(6, 7.8, 'Parameter Space θ', fontsize=10, ha='center', fontweight='bold')

    # Task optimal points
    task_points = [(4.5, 6.5), (5.5, 5.0), (6.5, 4.5), (7.5, 5.5)]
    task_colors = ['#4caf50', '#ff9800', '#2196f3', '#9c27b0']
    for i, ((x, y), color) in enumerate(zip(task_points, task_colors)):
        ax.plot(x, y, 'o', color=color, markersize=15, zorder=5)
        ax.text(x, y, f'θ*_{i+1}', fontsize=9, ha='center', va='center', color='white', fontweight='bold')

    # Fisher Information ellipses (importance contours)
    from matplotlib.patches import Ellipse
    for i, ((x, y), color) in enumerate(zip(task_points, task_colors)):
        ellipse = Ellipse((x, y), 1.2, 0.8, angle=i*30, facecolor='none', edgecolor=color, 
                          linewidth=2, linestyle='--', alpha=0.7)
        ax.add_patch(ellipse)

    # Current parameter path
    path_x = [4.5, 5.2, 5.8, 6.3, 6.8, 7.2, 7.5]
    path_y = [6.5, 6.0, 5.5, 5.2, 5.0, 5.3, 5.5]
    ax.plot(path_x, path_y, 'k-', linewidth=2, alpha=0.6, zorder=3)
    for i in range(len(path_x)-1):
        ax.annotate('', xy=(path_x[i+1], path_y[i+1]), xytext=(path_x[i], path_y[i]),
                    arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    ax.text(5.5, 5.8, 'θ_current', fontsize=9, ha='center', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='black'))

    # EWC penalty visualization
    ax.annotate('', xy=(5.5, 5.0), xytext=(4.5, 6.5),
                arrowprops=dict(arrowstyle='->', color='#4caf50', lw=2, linestyle='--'))
    ax.text(4.5, 5.5, 'F₁·(θ-θ*₁)²', fontsize=8, color='#4caf50', fontweight='bold')

    ax.annotate('', xy=(6.5, 4.5), xytext=(5.5, 5.0),
                arrowprops=dict(arrowstyle='->', color='#ff9800', lw=2, linestyle='--'))
    ax.text(5.5, 4.5, 'F₂·(θ-θ*₂)²', fontsize=8, color='#ff9800', fontweight='bold')

    # Formula box
    ax.text(6, 1.5, 'L_EWC(θ) = L_new(θ) + (λ/2) Σ_task Σ_i F_task,i · (θ_i - θ*_task,i)²', 
            fontsize=10, ha='center', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#fff3e0', edgecolor='#f57c00'))
    ax.text(6, 0.8, 'F_i = E[(∂log π(a|o)/∂θ_i)²]  (Fisher Information diagonal)', 
            fontsize=9, ha='center', style='italic', color='#666')

    # Side labels
    ax.text(0.5, 5.5, 'High Fisher\n→ Protected', fontsize=8, ha='center', color='#c62828',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#ffebee', edgecolor='#c62828'))
    ax.text(11.5, 5.5, 'Low Fisher\n→ Plastic', fontsize=8, ha='center', color='#2e7d32',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#e8f5e9', edgecolor='#388e3c'))

    plt.tight_layout()
    plt.savefig(output_dir + '/fig29_ewc_mechanism.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


    import matplotlib
    matplotlib.use('Agg')


    # Set global style
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='EWC Continual Learning Mechanism')
    parser.add_argument('--output', type=str, default='results/figs_01_10',
                        help='Output directory for the figure')
    args = parser.parse_args()
    generate_figure(args.output)
    print(f"Figure saved to {args.output}")
