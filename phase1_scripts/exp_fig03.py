#!/usr/bin/env python3
"""
IEEE 8-Bus Test System Single-Line Diagram
Generated from figs.py — standalone executable script for Phase 1.
Usage: python exp_fig03.py --output <path>
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
    """Generate IEEE 8-Bus Test System Single-Line Diagram"""
    os.makedirs(output_dir, exist_ok=True)


    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(-1, 15)
    ax.set_ylim(-1, 11)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Figure 3: IEEE 8-Bus Test System with DOCR Placement (0% PV)', fontsize=14, fontweight='bold', pad=20)

    # Bus positions
    buses = {
        1: (2, 9),    # Source bus
        2: (6, 9),
        3: (10, 9),
        4: (2, 5),
        5: (6, 5),
        6: (10, 5),
        7: (2, 1),
        8: (10, 1),
    }

    # Lines (bus connections)
    lines = [
        (1, 2), (2, 3), (1, 4), (2, 5), (3, 6),
        (4, 5), (5, 6), (4, 7), (5, 7), (6, 8), (7, 8)
    ]

    # Relay placements (on lines, with direction)
    # Format: (line, position_ratio, direction, relay_id)
    relays = [
        ((1,2), 0.3, '→', 'R1'), ((1,2), 0.7, '←', 'R2'),
        ((2,3), 0.3, '→', 'R3'), ((2,3), 0.7, '←', 'R4'),
        ((1,4), 0.3, '↓', 'R5'), ((1,4), 0.7, '↑', 'R6'),
        ((2,5), 0.3, '↓', 'R7'), ((2,5), 0.7, '↑', 'R8'),
        ((3,6), 0.3, '↓', 'R9'), ((3,6), 0.7, '↑', 'R10'),
        ((4,5), 0.3, '→', 'R11'), ((4,5), 0.7, '←', 'R12'),
        ((5,6), 0.3, '→', 'R13'), ((5,6), 0.7, '←', 'R14'),
    ]

    # Draw lines
    for b1, b2 in lines:
        x1, y1 = buses[b1]
        x2, y2 = buses[b2]
        ax.plot([x1, x2], [y1, y2], 'k-', linewidth=2, zorder=1)
        # Line label
        mid_x, mid_y = (x1+x2)/2, (y1+y2)/2
        ax.text(mid_x, mid_y, f'L{b1}-{b2}', fontsize=7, ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='none', alpha=0.8))

    # Draw buses
    for bus_id, (x, y) in buses.items():
        circle = Circle((x, y), 0.35, facecolor='#1976d2', edgecolor='black', linewidth=2, zorder=3)
        ax.add_patch(circle)
        ax.text(x, y, str(bus_id), fontsize=12, ha='center', va='center', color='white', fontweight='bold', zorder=4)

    # Source bus special marker
    ax.text(2, 9.7, 'SOURCE', fontsize=8, ha='center', fontweight='bold', color='#d32f2f')
    ax.annotate('', xy=(2, 9.4), xytext=(2, 9.6),
                arrowprops=dict(arrowstyle='->', color='#d32f2f', lw=2))

    # Draw relays
    for (b1, b2), ratio, direction, rid in relays:
        x1, y1 = buses[b1]
        x2, y2 = buses[b2]
        rx = x1 + ratio * (x2 - x1)
        ry = y1 + ratio * (y2 - y1)
    
        # Relay box
        rect = Rectangle((rx-0.25, ry-0.15), 0.5, 0.3, 
                           facecolor='#ff9800', edgecolor='black', linewidth=1.5, zorder=5)
        ax.add_patch(rect)
        ax.text(rx, ry, rid, fontsize=7, ha='center', va='center', fontweight='bold', zorder=6)
    
        # Direction arrow
        if direction == '→':
            ax.annotate('', xy=(rx+0.15, ry), xytext=(rx-0.15, ry),
                        arrowprops=dict(arrowstyle='->', color='green', lw=1.5), zorder=7)
        elif direction == '←':
            ax.annotate('', xy=(rx-0.15, ry), xytext=(rx+0.15, ry),
                        arrowprops=dict(arrowstyle='->', color='green', lw=1.5), zorder=7)
        elif direction == '↓':
            ax.annotate('', xy=(rx, ry-0.15), xytext=(rx, ry+0.15),
                        arrowprops=dict(arrowstyle='->', color='green', lw=1.5), zorder=7)
        elif direction == '↑':
            ax.annotate('', xy=(rx, ry+0.15), xytext=(rx, ry-0.15),
                        arrowprops=dict(arrowstyle='->', color='green', lw=1.5), zorder=7)

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor='#1976d2', edgecolor='black', label='Bus'),
        mpatches.Patch(facecolor='#ff9800', edgecolor='black', label='DOCR'),
        plt.Line2D([0], [0], color='green', lw=1.5, label='Protection Direction'),
        plt.Line2D([0], [0], color='black', lw=2, label='Line'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9, framealpha=0.9)

    # System info box
    info_text = """System Parameters:
    • Buses: 8 | Relays: 14 | P/B Pairs: 20
    • CTI ≥ 0.30 s | TMS ∈ [0.05, 1.1]
    • Standard Inverse: t = TMS × 0.02 / ((I/I_pickup)^0.02 - 1)
    • Source: Bus 1 (slack) | Load buses: 4, 5, 6, 7, 8"""
    ax.text(0.5, -0.3, info_text, fontsize=8, va='top', ha='left',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#f5f5f5', edgecolor='#999'))

    plt.tight_layout()
    plt.savefig(output_dir + '/fig3_ieee8bus_diagram.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.savefig(output_dir + '/fig3_ieee8bus_diagram.pdf', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='IEEE 8-Bus Test System Single-Line Diagram')
    parser.add_argument('--output', type=str, default='results/figs_01_10',
                        help='Output directory for the figure')
    args = parser.parse_args()
    generate_figure(args.output)
    print(f"Figure saved to {args.output}")
