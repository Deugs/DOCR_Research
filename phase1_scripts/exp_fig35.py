#!/usr/bin/env python3
"""
Observation Space Feature Breakdown & Encoding
Generated from figs.py — standalone executable script for Phase 1.
Usage: python exp_fig35.py --output <path>
"""

import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle, Polygon, Wedge
import numpy as np
import matplotlib.gridspec as gridspec


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

arrow_style = dict(arrowstyle='->', color='#333', lw=1.5, mutation_scale=15)


def draw_block(ax, x, y, w, h, label, sublabel='', color='#e3f2fd', edge='#1565c0', fontsize=9):
    rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                           facecolor=color, edgecolor=edge, linewidth=2)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2 + 0.15, label, fontsize=fontsize, ha='center', va='center', fontweight='bold')
    if sublabel:
        ax.text(x + w/2, y + h/2 - 0.2, sublabel, fontsize=7, ha='center', va='center', color='#666')



def generate_figure(output_dir):
    """Generate Observation Space Feature Breakdown & Encoding"""
    os.makedirs(output_dir, exist_ok=True)


    fig = plt.figure(figsize=(16, 11))

    # Main layout: 2 rows
    # Top: Feature vector breakdown
    # Bottom: Feature importance bar chart

    gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1], hspace=0.35, wspace=0.3)

    # ========== TOP LEFT: Feature Vector Breakdown ==========
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 12)
    ax1.axis('off')
    ax1.set_title('(a) Observation Vector Structure  o_i ∈ ℝ¹⁰', fontsize=12, fontweight='bold', pad=10)

    # Feature blocks with indices
    features = [
        (9.5, 1.0, 'V_mag', 'pu', colors['light_blue'], colors['primary']),
        (8.3, 1.0, 'V_ang', 'deg', colors['light_blue'], colors['primary']),
        (7.1, 1.0, 'I_mag', 'pu', colors['light_green'], colors['secondary']),
        (5.9, 1.0, 'I_ang', 'deg', colors['light_green'], colors['secondary']),
        (4.7, 1.0, 'P_dir', '{-1,0,1}', colors['light_orange'], colors['accent']),
        (3.5, 1.0, 'I_fault', 'pu', colors['light_orange'], colors['accent']),
        (2.3, 1.0, 'PV_lvl', '%', colors['light_pink'], colors['danger']),
        (1.1, 1.0, 'Topo_flag', '{0,1}', colors['light_pink'], colors['danger']),
        (0.5, 0.5, 'TMS_nbr', 'avg', colors['light_blue'], colors['primary']),
        (0.5, 0.5, 'PS_nbr', 'avg', colors['light_green'], colors['secondary']),
    ]

    y_pos = 10.5
    for i, (h, w, label, unit, fc, ec) in enumerate(features[:8]):
        rect = FancyBboxPatch((1, y_pos - h), w*2.5, h, boxstyle="round,pad=0.05",
                               facecolor=fc, edgecolor=ec, linewidth=2)
        ax1.add_patch(rect)
        ax1.text(1 + w*2.5/2, y_pos - h/2, f'{label}\n[{unit}]', fontsize=8, ha='center', va='center', fontweight='bold')
        ax1.text(0.5, y_pos - h/2, f'o[{i}]', fontsize=8, ha='center', va='center', color=colors['neutral'], fontweight='bold')
        y_pos -= h + 0.3

    # Neighbor stats (combined block)
    rect = FancyBboxPatch((1, y_pos - 1.0), 5.0, 1.0, boxstyle="round,pad=0.05",
                           facecolor='#f3e5f5', edgecolor='#7b1fa2', linewidth=2)
    ax1.add_patch(rect)
    ax1.text(3.5, y_pos - 0.5, 'Neighbor Stats (mean TMS, PS of adjacent relays)', fontsize=8, ha='center', va='center', fontweight='bold')
    ax1.text(0.5, y_pos - 0.5, 'o[8:]', fontsize=8, ha='center', va='center', color=colors['neutral'], fontweight='bold')

    # Dimension annotation
    ax1.annotate('', xy=(7, 11), xytext=(7, 1.5),
                arrowprops=dict(arrowstyle='<->', color=colors['dark'], lw=2))
    ax1.text(7.3, 6.5, 'obs_dim = 10\n(8 local + 2 neighbor)', fontsize=9, ha='left', va='center', 
             fontweight='bold', color=colors['primary'],
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=colors['primary']))

    # ========== TOP RIGHT: Encoding Pipeline ==========
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 12)
    ax2.axis('off')
    ax2.set_title('(b) Encoding Pipeline: Raw → Normalized → Embedded', fontsize=12, fontweight='bold', pad=10)

    # Raw observations
    draw_block(ax2, 1, 9.5, 2.5, 1.2, 'Raw Observations', 'V, I, P_dir, ...', colors['light_blue'], colors['primary'])

    # Normalization
    draw_block(ax2, 4.5, 9.5, 2.5, 1.2, 'Running Statistics', 'μ=EMA, σ=EMA  |  z = (x−μ)/(σ+ε)', colors['light_green'], colors['secondary'])

    # Embedding
    draw_block(ax2, 1, 7.0, 2.5, 1.2, 'MLP Encoder', '10 → 32 → 8  |  ReLU', colors['light_orange'], colors['accent'])

    # Message
    draw_block(ax2, 4.5, 7.0, 2.5, 1.2, 'Message Vector', 'm_i ∈ ℝ⁸  |  ~32 bytes', colors['light_pink'], colors['danger'])

    # Arrows
    ax2.annotate('', xy=(4.5, 10.1), xytext=(3.5, 10.1), arrowprops=arrow_style)
    ax2.annotate('', xy=(2.25, 8.3), xytext=(2.25, 9.5), arrowprops=arrow_style)
    ax2.annotate('', xy=(4.5, 7.6), xytext=(3.5, 7.6), arrowprops=arrow_style)

    # Encoding details table
    ax2.text(1, 5.5, 'Normalization Details:', fontsize=10, fontweight='bold')
    ax2.text(1, 5.0, '• Voltage/Current:  Per-unit normalization (base = nominal)', fontsize=8, color=colors['neutral'])
    ax2.text(1, 4.5, '• Power Direction:  One-hot encoded {-1, 0, 1}', fontsize=8, color=colors['neutral'])
    ax2.text(1, 4.0, '• Fault Current:  Log-scale compression for dynamic range', fontsize=8, color=colors['neutral'])
    ax2.text(1, 3.5, '• PV Level:  Normalized to [0, 1] by installed capacity', fontsize=8, color=colors['neutral'])
    ax2.text(1, 3.0, '• Topology Flag:  Binary (0 = nominal, 1 = N-1 active)', fontsize=8, color=colors['neutral'])

    ax2.text(1, 2.2, 'Message Compression:', fontsize=10, fontweight='bold')
    ax2.text(1, 1.7, '• 10-dim raw → 8-float message via learned MLP', fontsize=8, color=colors['neutral'])
    ax2.text(1, 1.2, '• 8 floats × 4 bytes = 32 bytes per message', fontsize=8, color=colors['neutral'])
    ax2.text(1, 0.7, '• At 10 Hz:  32 × 10 × 8 = 2,560 bits/s = 0.256 kbps', fontsize=8, color=colors['neutral'])
    ax2.text(1, 0.2, '• Well below AMI/RF mesh limit of 1 kbps/agent', fontsize=8, fontweight='bold', color=colors['secondary'])

    # ========== BOTTOM: Feature Importance (SHAP-style) ==========
    ax3 = fig.add_subplot(gs[1, :])

    feature_names = ['I_fault\n(Magnitude)', 'P_dir\n(Direction)', 'V_mag\n(Voltage)', 
                     'PV_lvl\n(Penetration)', 'I_mag\n(Current)', 'Topo_flag\n(Topology)',
                     'TMS_nbr\n(Neighbor)', 'PS_nbr\n(Neighbor)', 'V_ang\n(Angle)', 'I_ang\n(Angle)']
    importance = [0.28, 0.22, 0.15, 0.12, 0.10, 0.06, 0.04, 0.02, 0.01, 0.00]
    colors_bar = [colors['primary'] if i < 3 else colors['secondary'] if i < 6 else colors['accent'] for i in range(10)]

    bars = ax3.barh(range(10), importance, color=colors_bar, edgecolor='white', linewidth=1.5, height=0.7)
    ax3.set_yticks(range(10))
    ax3.set_yticklabels(feature_names, fontsize=9)
    ax3.set_xlabel('Normalized Feature Importance (SHAP-based)', fontsize=11, fontweight='bold')
    ax3.set_xlim(0, 0.35)
    ax3.invert_yaxis()
    ax3.set_title('(c) Learned Feature Importance for Relay Decision-Making (8-Bus, 100% PV)', 
                  fontsize=12, fontweight='bold', pad=10)

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, importance)):
        ax3.text(val + 0.005, bar.get_y() + bar.get_height()/2, f'{val:.2f}', 
                 va='center', fontsize=8, fontweight='bold', color=colors['dark'])

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=colors['primary'], label='Critical (SHAP > 0.15)'),
        mpatches.Patch(facecolor=colors['secondary'], label='Moderate (0.05–0.15)'),
        mpatches.Patch(facecolor=colors['accent'], label='Minor (< 0.05)')
    ]
    ax3.legend(handles=legend_elements, loc='lower right', fontsize=9)

    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.grid(axis='x', alpha=0.3)

    fig.suptitle('Figure 35: Observation Space Feature Breakdown & Encoding', fontsize=14, fontweight='bold', y=0.98)

    plt.savefig(output_dir + '/fig35_observation_space.png', dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Observation Space Feature Breakdown & Encoding')
    parser.add_argument('--output', type=str, default='results/figs_01_10',
                        help='Output directory for the figure')
    args = parser.parse_args()
    generate_figure(args.output)
    print(f"Figure saved to {args.output}")
