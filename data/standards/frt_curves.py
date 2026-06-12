
#!/usr/bin/env python3
"""
data/standards/frt_curves.py

Generate PV Inverter Fault Ride-Through (FRT) curves per IEC 61427-1 / IEEE 1547-2018.

Usage:
    python data/standards/frt_curves.py --standards iec61427 ieee1547 --output fig.pdf
"""

import argparse
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def iec61427_ride_through(v):
    """IEC 61427-1 ride-through time (seconds) vs voltage (pu)."""
    if v >= 1.1:
        return 0.16
    elif v >= 1.05:
        return 1.0
    elif v >= 0.88:
        return 999  # Continuous
    elif v >= 0.5:
        return 2.0
    else:
        return 0.16


def ieee1547_ride_through(v):
    """IEEE 1547-2018 ride-through time (seconds) vs voltage (pu)."""
    if v >= 1.1:
        return 0.16
    elif v >= 1.05:
        return 1.0
    elif v >= 0.88:
        return 999
    elif v >= 0.5:
        return 2.0
    else:
        return 0.16


def plot_frt_curves(standards, output_path):
    """Plot FRT curves for specified standards."""
    voltage = np.linspace(0.0, 1.2, 500)
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    colors = {'iec61427': '#1565c0', 'ieee1547': '#388e3c'}
    labels = {'iec61427': 'IEC 61427-1', 'ieee1547': 'IEEE 1547-2018'}
    
    for std in standards:
        if std == 'iec61427':
            times = np.array([iec61427_ride_through(v) for v in voltage])
        elif std == 'ieee1547':
            times = np.array([ieee1547_ride_through(v) for v in voltage])
        else:
            continue
        
        # Plot regions
        ax.fill_between(voltage, 0.001, times, where=(voltage >= 0.88) & (voltage <= 1.05), 
                        alpha=0.2, color='green', label='Continuous Operation' if std == standards[0] else '')
        ax.fill_between(voltage, 0.001, times, where=(voltage >= 0.5) & (voltage < 0.88), 
                        alpha=0.2, color='yellow', label='Mandatory Ride-Through (2s)' if std == standards[0] else '')
        ax.fill_between(voltage, 0.001, times, where=(voltage < 0.5) | (voltage > 1.1), 
                        alpha=0.2, color='red', label='May Disconnect' if std == standards[0] else '')
        ax.fill_between(voltage, 0.001, times, where=(voltage >= 1.05) & (voltage <= 1.1), 
                        alpha=0.2, color='orange', label='Reduced Time (1s)' if std == standards[0] else '')
        
        ax.semilogy(voltage, np.clip(times, 0.001, 1000), '-', color=colors.get(std, 'black'), 
                    linewidth=2.5, label=labels.get(std, std))
    
    # Annotations
    ax.annotate('Fault Current Injection\\n1.2-1.5x I_rated', xy=(0.3, 0.5), xytext=(0.5, 5),
                arrowprops=dict(arrowstyle='->', color='#1565c0', lw=1.5),
                fontsize=9, color='#1565c0', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#e3f2fd', edgecolor='#1565c0'))
    
    ax.plot(1.0, 999, 'go', markersize=12, zorder=5)
    ax.text(1.0, 500, 'Normal\\nOperation', fontsize=8, ha='center', color='green', fontweight='bold')
    ax.plot(0.3, 2, 'ro', markersize=12, zorder=5)
    ax.text(0.3, 0.5, 'Fault\\nCondition', fontsize=8, ha='center', color='red', fontweight='bold')
    
    ax.set_xlabel('Voltage (pu)', fontsize=12)
    ax.set_ylabel('Maximum Trip Time (s)', fontsize=12)
    ax.set_title('Figure 19: PV Inverter Fault Ride-Through Curve (IEC 61427-1 / IEEE 1547-2018)', 
                 fontsize=13, fontweight='bold')
    ax.set_xlim(0, 1.2)
    ax.set_ylim(0.001, 1000)
    ax.set_yticks([0.001, 0.01, 0.1, 1, 10, 100, 1000])
    ax.set_yticklabels(['1 ms', '10 ms', '100 ms', '1 s', '10 s', '100 s', '∞'])
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(fontsize=9, loc='upper right', framealpha=0.9)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Generate PV FRT curves')
    parser.add_argument('--standards', nargs='+', required=True, 
                        choices=['iec61427', 'ieee1547'],
                        help='Standards to plot')
    parser.add_argument('--output', type=str, required=True, help='Output file path')
    args = parser.parse_args()
    
    plot_frt_curves(args.standards, args.output)


if __name__ == '__main__':
    main()
