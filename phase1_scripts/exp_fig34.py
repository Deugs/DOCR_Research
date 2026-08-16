#!/usr/bin/env python3
"""
MAPPO-Safe-EWC Neural Network Architecture Detail
Generated from figs.py — standalone executable script for Phase 1.
Usage: python exp_fig34.py --output <path>
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

colors = {
    'primary': '#1565c0',
    'secondary': '#388e3c',
    'accent': '#f57c00',
    'danger': '#c2185b',
    'neutral': '#616161',
    'light_blue': '#e3f2fd',
    'light_green': '#e8f5e9',
    'light_orange': '#fff3e0',
    'light_pink': '#fce4ec',
    'white': '#ffffff',
    'dark': '#212121'
}



def generate_figure(output_dir):
    """Generate MAPPO-Safe-EWC Neural Network Architecture Detail"""
    os.makedirs(output_dir, exist_ok=True)


    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 12)
    ax.axis('off')
    ax.set_title('Figure 34: MAPPO-Safe-EWC Neural Network Architecture Detail', 
                 fontsize=15, fontweight='bold', pad=20, color=colors['dark']) #  

    # Helper for drawing blocks
    def draw_block(ax, x, y, w, h, label, sublabel='', color='#e3f2fd', edge='#1565c0', fontsize=9):
        rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                               facecolor=color, edgecolor=edge, linewidth=2)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2 + 0.15, label, fontsize=fontsize, ha='center', va='center', fontweight='bold')
        if sublabel:
            ax.text(x + w/2, y + h/2 - 0.2, sublabel, fontsize=7, ha='center', va='center', color='#666')

    # Input Layer
    draw_block(ax, 0.5, 10.2, 2.2, 0.9, 'Local Observation', 'o_i ∈ ℝ¹⁰', colors['light_blue'], colors['primary'])
    draw_block(ax, 3.2, 10.2, 2.2, 0.9, 'Neighbor Messages', 'm_j ∈ ℝ⁸', colors['light_green'], colors['secondary'])

    # GNN Encoder
    draw_block(ax, 0.5, 8.5, 5.0, 1.2, 'GNN Message Encoder', 'MLP: 10 → 32 → 8  |  ReLU', colors['light_blue'], colors['primary'])

    # Attention
    draw_block(ax, 0.5, 6.8, 5.0, 1.2, 'GAT Attention Layer', 'α_ij = softmax(LeakyReLU(aᵀ[Wh_i‖Wh_j]))', colors['light_green'], colors['secondary'])

    # Aggregated Message
    draw_block(ax, 0.5, 5.1, 5.0, 1.2, 'Message Aggregation', 'm̄_i = Σ_j α_ij · m_j  |  ℝ⁸', colors['light_orange'], colors['accent'])

    # Decoder / Enhanced Observation
    draw_block(ax, 0.5, 3.4, 5.0, 1.2, 'Observation Decoder', 'õ_i = [o_i ‖ m̄_i] ∈ ℝ¹⁸', colors['light_blue'], colors['primary'])

    # Shared Trunk
    draw_block(ax, 6.5, 8.5, 3.5, 1.2, 'Shared MLP Trunk', '18 → 64 → 64  |  ReLU, LayerNorm', colors['light_pink'], colors['danger'])

    # Policy Head
    draw_block(ax, 6.5, 6.5, 3.5, 1.2, 'Policy Head (Actor)', 'μ_θ(õ_i), σ_θ  |  Gaussian π(a|õ)', colors['light_blue'], colors['primary'], 9)

    # Value Head  
    draw_block(ax, 6.5, 4.5, 3.5, 1.2, 'Value Head (Critic)', 'V_φ(s_global)  |  Global State', colors['light_green'], colors['secondary'])

    # Safety Projection Layer
    draw_block(ax, 10.5, 6.5, 4.5, 1.2, 'Safety Projection Layer', 'Bounds Clip + CTI Lookahead (Binary Search)', colors['light_orange'], colors['accent'])

    # EWC Hook
    draw_block(ax, 10.5, 4.5, 4.5, 1.2, 'EWC Regularization Hook', 'Fisher Diagonal F_i  |  λ_EWC Σ F_i(θ_i−θ*_i)²', colors['light_pink'], colors['danger'])

    # Output
    draw_block(ax, 10.5, 2.5, 4.5, 1.2, 'Safe Action Output', 'a_safe = (ΔTMS_i, ΔPS_i)  |  CTI ≥ 0.3s', colors['light_blue'], colors['primary'])

    # Global State Input (for critic)
    draw_block(ax, 6.5, 10.5, 3.5, 0.9, 'Global State', 's_global ∈ ℝ^(2N+7)', colors['light_green'], colors['secondary'])

    # Arrows
    arrow_style = dict(arrowstyle='->', color=colors['dark'], lw=1.5, mutation_scale=15)

    # Input to GNN
    ax.annotate('', xy=(2.5, 9.7), xytext=(1.6, 10.2), arrowprops=arrow_style)
    ax.annotate('', xy=(3.0, 9.7), xytext=(4.3, 10.2), arrowprops=arrow_style)

    # GNN flow
    for y1, y2 in [(9.7, 8.5), (8.0, 7.4), (7.0, 6.3), (5.5, 5.1), (4.7, 4.3)]:
        ax.annotate('', xy=(3.0, y2), xytext=(3.0, y1), arrowprops=arrow_style)

    # To shared trunk
    ax.annotate('', xy=(6.5, 9.1), xytext=(5.5, 4.9), arrowprops=dict(arrowstyle='->', color=colors['dark'], lw=1.5, connectionstyle="arc3,rad=0.2"))

    # Global state to critic
    ax.annotate('', xy=(8.25, 8.5), xytext=(8.25, 10.5), arrowprops=arrow_style)

    # Trunk to heads
    ax.annotate('', xy=(8.25, 7.7), xytext=(8.25, 8.5), arrowprops=arrow_style)
    ax.annotate('', xy=(8.25, 5.7), xytext=(8.25, 6.5), arrowprops=arrow_style)

    # Policy to Safety
    ax.annotate('', xy=(10.5, 7.1), xytext=(10.0, 7.1), arrowprops=arrow_style)

    # Safety to Output
    ax.annotate('', xy=(12.75, 3.7), xytext=(12.75, 5.3), arrowprops=arrow_style)

    # EWC connection (dashed)
    ax.annotate('', xy=(12.75, 5.7), xytext=(12.75, 6.3), 
                arrowprops=dict(arrowstyle='<->', color=colors['danger'], lw=1.5, ls='--'))

    # Parameter counts annotation
    # Numbers below are verified against the real classes in marl/marl_agents.py
    # and marl/comm_graph.py (obs_dim=10, msg_dim=8, GNN hidden=32, attention
    # hidden=16, actor hidden=64, act_dim=2, critic hidden=128, n_relays=14):
    #   GNNMessageEncoder(10).n_params            == 616
    #   GNNAttention(8).n_params                   == 160
    #   SharedGaussianActor(18,2,hidden=64)
    #       .net.n_params() (=Trunk 5,376 + Policy Head 130) == 5,506; + log_std(2) -> get_all_params() == 5,508
    #   MAPPOCentralCritic(14, hidden=128).n_params() == 21,889
    ax.text(0.5, 2.0, 'Parameter Counts:', fontsize=10, fontweight='bold', color=colors['dark'])
    ax.text(0.5, 1.5, '• GNN Encoder:  10×32 + 32×8 + biases(32+8)  =  616 params', fontsize=8, color=colors['neutral'])
    ax.text(0.5, 1.1, '• Attention (a, W):  8×16 + 2×16  =  160 params', fontsize=8, color=colors['neutral'])
    ax.text(0.5, 0.7, '• Shared Trunk:  18×64 + 64×64 + biases(64+64)  =  5,376 params', fontsize=8, color=colors['neutral'])
    ax.text(0.5, 0.3, '• Policy Head:  64×2 + bias(2) + log_std(2)  =  132 params  |  Value Head (critic, centralized):  40×128 + 128×128 + 128×1 + biases  =  21,889 params', fontsize=8, color=colors['neutral'])
    ax.text(8.5, 2.0, 'Total per agent (shared, GNN+Actor):  6,284 params  |  Critic (centralized):  21,889 params', fontsize=9, fontweight='bold', color=colors['primary'])

    # Legend box
    legend_elements = [
        mpatches.Patch(facecolor=colors['light_blue'], edgecolor=colors['primary'], label='Observation / Policy'),
        mpatches.Patch(facecolor=colors['light_green'], edgecolor=colors['secondary'], label='Communication / Value'),
        mpatches.Patch(facecolor=colors['light_orange'], edgecolor=colors['accent'], label='Safety / Projection'),
        mpatches.Patch(facecolor=colors['light_pink'], edgecolor=colors['danger'], label='Regularization / Critic')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=8, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(output_dir + '/fig34_neural_architecture.png', dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MAPPO-Safe-EWC Neural Network Architecture Detail')
    parser.add_argument('--output', type=str, default='results/figs_01_10',
                        help='Output directory for the figure')
    args = parser.parse_args()
    generate_figure(args.output)
    print(f"Figure saved to {args.output}")
