#!/usr/bin/env python3
"""
PV Inverter Fault Ride-Through Curve
Generated from figs.py — standalone executable script for Phase 1.
Usage: python exp_fig19.py --output <path>
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
    """Generate PV Inverter Fault Ride-Through Curve"""
    os.makedirs(output_dir, exist_ok=True)


    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Voltage (pu)
    voltage = np.linspace(0.0, 1.2, 500)

    # Simplified ride-through curve (loosely informed by IEEE 1547-2018 / IEC 61727)
    # Must remain connected in region, may disconnect outside
    def ride_through_time(v):
        if v >= 1.1:
            return 0.16  # Instantaneous trip
        elif v >= 1.05:
            return 1.0  # 1 second
        elif v >= 0.88:
            return 999  # Continuous operation
        elif v >= 0.5:
            return 2.0  # 2 seconds
        elif v >= 0.0:
            return 0.16  # Instantaneous trip
        return 0

    times = np.array([ride_through_time(v) for v in voltage])

    # Plot regions
    ax.fill_between(voltage, 0.001, times, where=(voltage >= 0.88) & (voltage <= 1.05), 
                    alpha=0.3, color='green', label='Continuous Operation Zone')
    ax.fill_between(voltage, 0.001, times, where=(voltage >= 0.5) & (voltage < 0.88), 
                    alpha=0.3, color='yellow', label='Mandatory Ride-Through (2s)')
    ax.fill_between(voltage, 0.001, times, where=(voltage < 0.5) | (voltage > 1.1), 
                    alpha=0.3, color='red', label='May Disconnect')
    ax.fill_between(voltage, 0.001, times, where=(voltage >= 1.05) & (voltage <= 1.1), 
                    alpha=0.3, color='orange', label='Reduced Time (1s)')

    # Plot curve
    ax.semilogy(voltage, np.clip(times, 0.001, 1000), 'k-', linewidth=2.5, label='Ride-Through Boundary')

    # Fault current contribution annotation
    ax.annotate('Fault Current Injection\n1.2-1.5× I_rated', xy=(0.3, 0.5), xytext=(0.5, 5),
                arrowprops=dict(arrowstyle='->', color='#1565c0', lw=1.5),
                fontsize=9, color='#1565c0', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#e3f2fd', edgecolor='#1565c0'))

    # Normal operating point
    ax.plot(1.0, 999, 'go', markersize=12, zorder=5)
    ax.text(1.0, 500, 'Normal\nOperation', fontsize=8, ha='center', color='green', fontweight='bold')

    # Fault point
    ax.plot(0.3, 2, 'ro', markersize=12, zorder=5)
    ax.text(0.3, 0.5, 'Fault\nCondition', fontsize=8, ha='center', color='red', fontweight='bold')

    ax.set_xlabel('Voltage (pu)', fontsize=12)
    ax.set_ylabel('Maximum Trip Time (s)', fontsize=12)
    ax.set_title('Figure 19: PV Inverter Fault Ride-Through Curve (Simplified Approximation)', fontsize=13, fontweight='bold')
    ax.set_xlim(0, 1.2)
    ax.set_ylim(0.001, 1000)
    ax.set_yticks([0.001, 0.01, 0.1, 1, 10, 100, 1000])
    ax.set_yticklabels(['1 ms', '10 ms', '100 ms', '1 s', '10 s', '100 s', '∞'])
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(fontsize=9, loc='upper right', framealpha=0.9)

    plt.tight_layout()
    plt.savefig(output_dir + '/fig19_ride_through_curve.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PV Inverter Fault Ride-Through Curve')
    parser.add_argument('--output', type=str, default='results/figs_01_10',
                        help='Output directory for the figure')
    args = parser.parse_args()
    generate_figure(args.output)
    print(f"Figure saved to {args.output}")
