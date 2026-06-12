
#!/usr/bin/env python3
"""
network/utils/plot_topology.py

Generate IEEE test system single-line diagrams with DOCR placement.
Supports IEEE 8-bus and 30-bus systems with configurable PV penetration.

Usage:
    python network/utils/plot_topology.py --system 8bus --pv 0.0 --output fig.png
    python network/utils/plot_topology.py --system 30bus --pv 0.0 --output fig.png
"""

import argparse
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle
import numpy as np


def get_8bus_data():
    """Return IEEE 8-bus system topology and relay data."""
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
    lines = [
        (1, 2), (2, 3), (1, 4), (2, 5), (3, 6),
        (4, 5), (5, 6), (4, 7), (5, 7), (6, 8), (7, 8)
    ]
    # Relay placements: (line, position_ratio, direction, relay_id)
    relays = [
        ((1,2), 0.3, '→', 'R1'), ((1,2), 0.7, '←', 'R2'),
        ((2,3), 0.3, '→', 'R3'), ((2,3), 0.7, '←', 'R4'),
        ((1,4), 0.3, '↓', 'R5'), ((1,4), 0.7, '↑', 'R6'),
        ((2,5), 0.3, '↓', 'R7'), ((2,5), 0.7, '↑', 'R8'),
        ((3,6), 0.3, '↓', 'R9'), ((3,6), 0.7, '↑', 'R10'),
        ((4,5), 0.3, '→', 'R11'), ((4,5), 0.7, '←', 'R12'),
        ((5,6), 0.3, '→', 'R13'), ((5,6), 0.7, '←', 'R14'),
    ]
    return buses, lines, relays


def get_30bus_data():
    """Return IEEE 30-bus system topology."""
    buses = {
        1: (1, 11), 2: (4, 11), 3: (7, 11), 4: (10, 11), 5: (13, 11),
        6: (1, 8), 7: (4, 8), 8: (7, 8), 9: (10, 8), 10: (13, 8),
        11: (1, 5), 12: (4, 5), 13: (7, 5), 14: (10, 5), 15: (13, 5),
        16: (1, 2), 17: (4, 2), 18: (7, 2), 19: (10, 2), 20: (13, 2),
        21: (2.5, 9.5), 22: (5.5, 9.5), 23: (8.5, 9.5), 24: (11.5, 9.5),
        25: (2.5, 6.5), 26: (5.5, 6.5), 27: (8.5, 6.5), 28: (11.5, 6.5),
        29: (2.5, 3.5), 30: (5.5, 3.5),
    }
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
    return buses, lines


def draw_8bus_diagram(buses, lines, relays, pv_level, output_path):
    """Draw IEEE 8-bus single-line diagram with DOCRs."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(-1, 15)
    ax.set_ylim(-1, 11)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(f'Figure 3: IEEE 8-Bus Test System with DOCR Placement ({pv_level*100:.0f}% PV)', 
                 fontsize=14, fontweight='bold', pad=20)

    # Draw lines
    for b1, b2 in lines:
        x1, y1 = buses[b1]
        x2, y2 = buses[b2]
        ax.plot([x1, x2], [y1, y2], 'k-', linewidth=2, zorder=1)
        mid_x, mid_y = (x1+x2)/2, (y1+y2)/2
        ax.text(mid_x, mid_y, f'L{b1}-{b2}', fontsize=7, ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='none', alpha=0.8))

    # Draw buses
    for bus_id, (x, y) in buses.items():
        circle = Circle((x, y), 0.35, facecolor='#1976d2', edgecolor='black', linewidth=2, zorder=3)
        ax.add_patch(circle)
        ax.text(x, y, str(bus_id), fontsize=12, ha='center', va='center', 
                color='white', fontweight='bold', zorder=4)

    # Source bus marker
    ax.text(2, 9.7, 'SOURCE', fontsize=8, ha='center', fontweight='bold', color='#d32f2f')
    ax.annotate('', xy=(2, 9.4), xytext=(2, 9.6),
                arrowprops=dict(arrowstyle='->', color='#d32f2f', lw=2))

    # Draw relays
    for (b1, b2), ratio, direction, rid in relays:
        x1, y1 = buses[b1]
        x2, y2 = buses[b2]
        rx = x1 + ratio * (x2 - x1)
        ry = y1 + ratio * (y2 - y1)
        
        rect = Rectangle((rx-0.25, ry-0.15), 0.5, 0.3, 
                           facecolor='#ff9800', edgecolor='black', linewidth=1.5, zorder=5)
        ax.add_patch(rect)
        ax.text(rx, ry, rid, fontsize=7, ha='center', va='center', fontweight='bold', zorder=6)
        
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

    # System info
    info_text = f"""System Parameters:
• Buses: 8 | Relays: 14 | P/B Pairs: 20
• CTI >= 0.30 s | TMS ∈ [0.05, 1.1]
• Standard Inverse: t = TMS × 0.14 / ((I/I_pickup)^0.02 - 1)
• Source: Bus 1 (slack) | Load buses: 4, 5, 6, 7, 8
• PV Penetration: {pv_level*100:.0f}%"""
    ax.text(0.5, -0.3, info_text, fontsize=8, va='top', ha='left',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#f5f5f5', edgecolor='#999'))

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def draw_30bus_diagram(buses, lines, pv_level, output_path):
    """Draw IEEE 30-bus single-line diagram."""
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    ax.set_xlim(-1, 17)
    ax.set_ylim(-1, 13)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(f'Figure 4: IEEE 30-Bus Test System with DOCR Placement ({pv_level*100:.0f}% PV)', 
                 fontsize=14, fontweight='bold', pad=20)

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
    ax.text(7, 10.3, 'PV DGs', fontsize=8, ha='center', fontweight='bold', color='#388e3c')

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor='#1976d2', edgecolor='black', label='Load Bus'),
        mpatches.Patch(facecolor='#d32f2f', edgecolor='black', label='Source Bus'),
        mpatches.Patch(facecolor='#388e3c', edgecolor='black', label='PV Bus (DG)'),
        plt.Line2D([0], [0], color='black', lw=1, label='Line'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9, framealpha=0.9)

    info_text = f"""IEEE 30-Bus System
• Buses: 30 | Relays: 42 | P/B Pairs: 68
• PV Buses: 10 (DG penetration variable 0-100%)
• Topology: Meshed with radial feeders
• Standard: IEC C37.112-2018 inverse-time DOCRs
• PV Penetration: {pv_level*100:.0f}%"""
    ax.text(0.5, -0.3, info_text, fontsize=8, va='top', ha='left',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#f5f5f5', edgecolor='#999'))

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Plot IEEE test system topology')
    parser.add_argument('--system', type=str, required=True, choices=['8bus', '30bus'],
                        help='Test system: 8bus or 30bus')
    parser.add_argument('--pv', type=float, default=0.0, help='PV penetration level (0.0-1.0)')
    parser.add_argument('--output', type=str, required=True, help='Output file path')
    args = parser.parse_args()

    if args.system == '8bus':
        buses, lines, relays = get_8bus_data()
        draw_8bus_diagram(buses, lines, relays, args.pv, args.output)
    elif args.system == '30bus':
        buses, lines = get_30bus_data()
        draw_30bus_diagram(buses, lines, args.pv, args.output)


if __name__ == '__main__':
    main()
