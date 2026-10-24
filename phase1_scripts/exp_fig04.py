#!/usr/bin/env python3
"""
IEEE 30-Bus Test System Single-Line Diagram
Generated from figs.py — standalone executable script for Phase 1.
Usage: python exp_fig04.py --output <path>
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
    """Generate IEEE 30-Bus Test System Single-Line Diagram"""
    os.makedirs(output_dir, exist_ok=True)


    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    ax.set_xlim(-1, 17)
    ax.set_ylim(-1, 13)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Figure 4: IEEE 30-Bus Test System with DOCR Placement', fontsize=14, fontweight='bold', pad=20)

    # Bus positions (simplified radial layout)
    buses = {
        1: (1, 11),    # Source
        2: (4, 11), 3: (7, 11), 4: (10, 11), 5: (13, 11),
        6: (1, 8), 7: (4, 8), 8: (7, 8), 9: (10, 8), 10: (13, 8),
        11: (1, 5), 12: (4, 5), 13: (7, 5), 14: (10, 5), 15: (13, 5),
        16: (1, 2), 17: (4, 2), 18: (7, 2), 19: (10, 2), 20: (13, 2),
        21: (2.5, 9.5), 22: (5.5, 9.5), 23: (8.5, 9.5), 24: (11.5, 9.5),
        25: (2.5, 6.5), 26: (5.5, 6.5), 27: (8.5, 6.5), 28: (11.5, 6.5),
        29: (2.5, 3.5), 30: (5.5, 3.5),
    }

    # Lines (simplified topology)
    lines = [
        (1,2), (2,3), (3,4), (4,5),
        (1,6), (2,7), (3,8), (4,9), (5,10),
        (6,7), (7,8), (8,9), (9,10),
        (6,11), (7,12), (8,13), (9,14), (10,15),
        (11,12), (12,13), (13,14), (14,15),
        (11,16), (12,17), (13,18), (14,19), (15,20),
        (16,17), (17,18), (18,19), (19,20),
        (21,22), (22,23), (23,24),
        (25,26), (26,27), (27,28),
        (2,21), (3,22), (4,23), (5,24),
        (7,25), (8,26), (9,27), (10,28),
        (11,29), (12,30),
    ]

    # Draw lines
    for b1, b2 in lines:
        if b1 in buses and b2 in buses:
            x1, y1 = buses[b1]
            x2, y2 = buses[b2]
            ax.plot([x1, x2], [y1, y2], 'k-', linewidth=1, zorder=1, alpha=0.6)

    # Draw buses
    for bus_id, (x, y) in buses.items():
        if bus_id == 1:
            color = '#d32f2f'  # Source
        elif bus_id in [21, 22, 23, 24, 25, 26, 27, 28, 29, 30]:
            color = '#388e3c'  # DG buses (PV)
        else:
            color = '#1976d2'  # Regular bus
    
        circle = Circle((x, y), 0.25, facecolor=color, edgecolor='black', linewidth=1.5, zorder=3)
        ax.add_patch(circle)
        ax.text(x, y, str(bus_id), fontsize=8, ha='center', va='center', 
                color='white', fontweight='bold', zorder=4)

    # Source label
    ax.text(1, 11.7, 'SOURCE', fontsize=8, ha='center', fontweight='bold', color='#d32f2f')

    # PV labels
    ax.text(7, 10.3, 'PV DGs', fontsize=8, ha='center', fontweight='bold', color='#388e3c')

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor='#1976d2', edgecolor='black', label='Load Bus'),
        mpatches.Patch(facecolor='#d32f2f', edgecolor='black', label='Source Bus'),
        mpatches.Patch(facecolor='#388e3c', edgecolor='black', label='PV Bus (DG)'),
        plt.Line2D([0], [0], color='black', lw=1, label='Line'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9, framealpha=0.9)

    # System info
    info_text = """IEEE 30-Bus System
    • Buses: 30 | Relays: 42 | P/B Pairs: 68
    • PV Buses: 10 (DG penetration variable 0-100%)
    • Topology: Meshed with radial feeders
    • Standard: IEC C37.112-2018 inverse-time DOCRs"""
    ax.text(0.5, -0.3, info_text, fontsize=8, va='top', ha='left',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#f5f5f5', edgecolor='#999'))

    plt.tight_layout()
    plt.savefig(output_dir + '/fig4_ieee30bus_diagram.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.savefig(output_dir + '/fig4_ieee30bus_diagram.pdf', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='IEEE 30-Bus Test System Single-Line Diagram')
    parser.add_argument('--output', type=str, default='results/figs_01_10',
                        help='Output directory for the figure')
    args = parser.parse_args()
    generate_figure(args.output)
    print(f"Figure saved to {args.output}")
