#!/usr/bin/env python3
"""
environment/experiments/data_flow_logging.py

Generate Figure 2: Data Flow Architecture (Single Training Step).
Logs and visualizes the single-step data flow through the MAPPO-Safe-EWC system.

Usage:
    python environment/experiments/data_flow_logging.py --system 8bus --episodes 1 --debug
"""

import argparse
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np


def plot_data_flow_architecture(output_path):
    """Plot Figure 2: Data Flow Architecture (Single Training Step)."""
    fig, ax = plt.subplots(1, 1, figsize=(16, 8))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 8)
    ax.axis('off')
    ax.set_title('Figure 2: Data Flow Architecture (Single Training Step)', 
                 fontsize=14, fontweight='bold', pad=20)

    # Define flow stages
    stages = [
        {'name': 'Sensor', 'x': 1, 'color': '#bbdefb'},
        {'name': 'Encode', 'x': 3, 'color': '#c8e6c9'},
        {'name': 'GNN\\nGraph', 'x': 5, 'color': '#fff9c4'},
        {'name': 'Decode', 'x': 7, 'color': '#c8e6c9'},
        {'name': 'Policy\\nπθ', 'x': 9, 'color': '#ffccbc'},
        {'name': 'Project', 'x': 11, 'color': '#f8bbd9'},
        {'name': 'Env\\nStep', 'x': 13, 'color': '#d1c4e9'},
        {'name': 'Critic\\nVφ', 'x': 15, 'color': '#b2dfdb'},
    ]

    # Draw stages as boxes
    for stage in stages:
        rect = FancyBboxPatch((stage['x']-0.7, 3.5), 1.4, 1.2,
                               boxstyle="round,pad=0.05",
                               facecolor=stage['color'], edgecolor='#333', linewidth=1.5)
        ax.add_patch(rect)
        ax.text(stage['x'], 4.1, stage['name'], fontsize=9, ha='center', va='center', fontweight='bold')

    # Draw arrows between stages
    for i in range(len(stages)-1):
        ax.annotate('', xy=(stages[i+1]['x']-0.7, 4.1), xytext=(stages[i]['x']+0.7, 4.1),
                    arrowprops=dict(arrowstyle='->', color='#333', lw=1.5))

    # Data labels above arrows
    labels = [
        ('o_i ∈ ℝ¹⁰', 2, 5.2),
        ('m_i ∈ ℝ⁸', 4, 5.2),
        ('m̄_i = Σα_ij·m_j', 6, 5.2),
        ('õ_i = [o_i‖m̄_i]', 8, 5.2),
        ('a_i = (ΔTMS, ΔPS)', 10, 5.2),
        ('a_i,safe = P(a_i)', 12, 5.2),
        ('(r, o_i\\)', 14, 5.2),
        ]

    for label, x, y in labels:
        ax.text(x, y, label, fontsize=8, ha='center', style='italic', color='#1565c0')

    # Detail boxes below
    # Observation detail
    obs_rect = FancyBboxPatch((0.3, 1.0), 2.8, 1.8, boxstyle="round,pad=0.05",
                               facecolor='#e3f2fd', edgecolor='#1976d2', linewidth=1)
    ax.add_patch(obs_rect)
    ax.text(1.7, 2.5, 'Local Observation (10-dim)', fontsize=8, ha='center', fontweight='bold', color='#1565c0')
    obs_items = ['V, I, P_dir, I_fault', 'PV_level, topology_flag', 'neighbor_TMS, neighbor_PS', 'CTI_slack, time']
    for j, item in enumerate(obs_items):
        ax.text(1.7, 2.15 - j*0.25, f'• {item}', fontsize=7, ha='center', color='#333')

    # GNN detail
    gnn_rect = FancyBboxPatch((4.3, 1.0), 2.8, 1.8, boxstyle="round,pad=0.05",
                               facecolor='#fffde7', edgecolor='#f57f17', linewidth=1)
    ax.add_patch(gnn_rect)
    ax.text(5.7, 2.5, 'GNN Attention', fontsize=8, ha='center', fontweight='bold', color='#f57f17')
    gnn_items = ['α_ij = softmax(LeakyReLU(aᵀ[Wh_i‖Wh_j]))', 'Message: 8 floats (32 bytes)', 'Bandwidth: 0.256 kbps', 'Graph: P/B pairs + same-bus']
    for j, item in enumerate(gnn_items):
        ax.text(5.7, 2.15 - j*0.25, f'• {item}', fontsize=7, ha='center', color='#333')

    # Safe RL detail
    safe_rect = FancyBboxPatch((9.3, 1.0), 2.8, 1.8, boxstyle="round,pad=0.05",
                                facecolor='#fce4ec', edgecolor='#c2185b', linewidth=1)
    ax.add_patch(safe_rect)
    ax.text(10.7, 2.5, 'Safe RL Projection', fontsize=8, ha='center', fontweight='bold', color='#c2185b')
    safe_items = ['TMS ∈ [0.05, 1.1]', 'PS ∈ [PS_min, PS_max]', 'CTI lookahead binary search', 'β* = max feasible scale']
    for j, item in enumerate(safe_items):
        ax.text(10.7, 2.15 - j*0.25, f'• {item}', fontsize=7, ha='center', color='#333')

    # Critic detail
    critic_rect = FancyBboxPatch((13.3, 1.0), 2.4, 1.8, boxstyle="round,pad=0.05",
                                  facecolor='#e0f2f1', edgecolor='#00796b', linewidth=1)
    ax.add_patch(critic_rect)
    ax.text(14.5, 2.5, 'Centralized Critic', fontsize=8, ha='center', fontweight='bold', color='#00796b')
    critic_items = ['s_global = [TMS_all‖PS_all‖', 'PV_level‖topology‖fault_type]', 'GAE(λ=0.95, γ=0.99)', 'Value: Vφ(s_global)']
    for j, item in enumerate(critic_items):
        ax.text(14.5, 2.15 - j*0.25, f'• {item}', fontsize=7, ha='center', color='#333')

    # Bottom: Storage
    store_rect = FancyBboxPatch((5.5, 0.1), 5, 0.7, boxstyle="round,pad=0.03",
                                 facecolor='#f5f5f5', edgecolor='#616161', linewidth=1, linestyle='--')
    ax.add_patch(store_rect)
    ax.text(8, 0.45, 'Rollout Buffer: (o_i, õ_i, a_i,safe, r, o_i\\, Vφ(s), log πθ(ai|õi))', fontsize=8, ha='center', style='italic', color='#616161')

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Data Flow Architecture visualization')
    parser.add_argument('--system', type=str, default='8bus', help='Test system')
    parser.add_argument('--episodes', type=int, default=1, help='Number of episodes')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    parser.add_argument('--output', type=str, default=None, help='Output file path')
    args = parser.parse_args()

    if args.output is None:
        args.output = f'results/figs_01_10/fig_02_data_flow_{args.system}.png'
    
    plot_data_flow_architecture(args.output)


if __name__ == '__main__':
    main()
