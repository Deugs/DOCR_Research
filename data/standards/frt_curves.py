
#!/usr/bin/env python3
"""
data/standards/frt_curves.py

Generate Figure 19: PV Inverter Fault Ride-Through Curve.
Visualizes a simplified voltage ride-through curve for PV inverters under
fault conditions, loosely informed by the general shape of grid-code
requirements (see IEEE 1547-2018 / IEC 61727 for the standards that actually
govern grid-connected PV fault behavior).

Usage:
    python data/standards/frt_curves.py --standards generic ieee1547 --output fig.pdf
"""

import argparse
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def ride_through_time(v):
    """Simplified ride-through time as a function of voltage (pu).

    A generic step-function approximation, loosely informed by the general
    shape of grid-code low-voltage ride-through requirements — not a literal
    reproduction of any single standard's trip-time table.
    """
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


def plot_frt_curves(output_path, standards=None):
    """Plot Figure 19: PV Inverter Fault Ride-Through Curve."""
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Voltage (pu)
    voltage = np.linspace(0.0, 1.2, 500)
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
    ax.annotate('Fault Current Injection\n1.2-1.5x I_rated', xy=(0.3, 0.5), xytext=(0.5, 5),
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
    ax.set_title('Figure 19: PV Inverter Fault Ride-Through Curve (Simplified Approximation)',
                 fontsize=13, fontweight='bold')
    ax.set_xlim(0, 1.2)
    ax.set_ylim(0.001, 1000)
    ax.set_yticks([0.001, 0.01, 0.1, 1, 10, 100, 1000])
    ax.set_yticklabels(['1 ms', '10 ms', '100 ms', '1 s', '10 s', '100 s', 'inf'])
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(fontsize=9, loc='upper right', framealpha=0.9)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='PV Inverter Fault Ride-Through Curves')
    parser.add_argument('--standards', nargs='+', default=['generic', 'ieee1547'],
                        help='Standards to plot')
    parser.add_argument('--output', type=str, required=True, help='Output file path')
    args = parser.parse_args()
    
    plot_frt_curves(args.output, standards=args.standards)


if __name__ == '__main__':
    main()
