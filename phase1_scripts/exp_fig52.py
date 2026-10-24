#!/usr/bin/env python3
"""
Digital Twin Training-to-Deployment Pipeline
Generated from figs.py — standalone executable script for Phase 1.
Usage: python exp_fig52.py --output <path>
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
    """Generate Digital Twin Training-to-Deployment Pipeline"""
    os.makedirs(output_dir, exist_ok=True)


    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('Figure 52: Digital Twin Training-to-Deployment Pipeline', 
                 fontsize=14, fontweight='bold', pad=20, color=colors['dark'])

    stages = [
        {'name': 'Physical\nGrid', 'x': 1.5, 'color': '#e3f2fd', 'edge': '#1565c0',
         'details': 'SCADA/PMU\nReal-time data\n~10 GB/day'},
        {'name': 'Data\nIngestion', 'x': 3.5, 'color': '#e8f5e9', 'edge': '#388e3c',
         'details': 'Kafka/Kinesis\nStream processing\nETL pipeline'},
        {'name': 'Digital Twin\nCalibration', 'x': 5.5, 'color': '#fff3e0', 'edge': '#f57c00',
         'details': 'OpenDSS model\nParameter tuning\nValidation < 2% error'},
        {'name': 'Offline\nMARL Training', 'x': 7.5, 'color': '#fce4ec', 'edge': '#c2185b',
         'details': 'GPU cluster\nRay RLlib\n~40 GPU-hours'},
        {'name': 'Validation &\nTesting', 'x': 9.5, 'color': '#f3e5f5', 'edge': '#7b1fa2',
         'details': 'Held-out scenarios\nN-1 contingencies\nStatistical tests'},
        {'name': 'Model\nExport', 'x': 11.5, 'color': '#e0f7fa', 'edge': '#00838f',
         'details': 'ONNX/TorchScript\nINT8 quantization\n~2.1 MB/agent'},
        {'name': 'Edge\nDeployment', 'x': 13.5, 'color': '#efebe9', 'edge': '#5d4037',
         'details': 'Intel NUC i5\nIEC 61850 bridge\n< 100 ms latency'},
    ]

    for stage in stages:
        rect = FancyBboxPatch((stage['x']-0.7, 6.5), 1.4, 2.2, 
                               boxstyle="round,pad=0.08",
                               facecolor=stage['color'], edgecolor=stage['edge'], linewidth=2.5)
        ax.add_patch(rect)
        ax.text(stage['x'], 8.2, stage['name'], fontsize=9, ha='center', va='center', 
                fontweight='bold', color=stage['edge'])
        ax.text(stage['x'], 7.2, stage['details'], fontsize=6.5, ha='center', va='center', 
                color='#555', linespacing=1.3)

    for i in range(len(stages)-1):
        ax.annotate('', xy=(stages[i+1]['x']-0.7, 7.6), xytext=(stages[i]['x']+0.7, 7.6),
                    arrowprops=dict(arrowstyle='->', color='#333', lw=2))

    ax.annotate('', xy=(5.5, 5.8), xytext=(13.5, 5.8),
                arrowprops=dict(arrowstyle='->', color='#c62828', lw=2, 
                               connectionstyle="arc3,rad=-0.3", linestyle='--'))
    ax.text(9.5, 5.3, 'Drift Detection -> Retrigger Pipeline', fontsize=9, ha='center', 
            color='#c62828', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#ffebee', edgecolor='#c62828'))

    monitor_rect = FancyBboxPatch((1.0, 3.5), 14, 1.5, boxstyle="round,pad=0.1",
                                   facecolor='#fafafa', edgecolor='#616161', linewidth=2, linestyle='--')
    ax.add_patch(monitor_rect)
    ax.text(8, 4.7, 'CONTINUOUS MONITORING LAYER', fontsize=11, ha='center', 
            fontweight='bold', color='#616161')

    monitor_items = [
        (2.5, 4.0, 'Performance\nMetrics', '#e3f2fd', '#1565c0'),
        (5.0, 4.0, 'CTI\nCompliance', '#e8f5e9', '#388e3c'),
        (7.5, 4.0, 'Communication\nHealth', '#fff3e0', '#f57c00'),
        (10.0, 4.0, 'Adversarial\nDetection', '#fce4ec', '#c2185b'),
        (12.5, 4.0, 'Model Version\nControl', '#f3e5f5', '#7b1fa2'),
    ]
    for x, y, label, fc, ec in monitor_items:
        rect = FancyBboxPatch((x-0.6, y-0.35), 1.2, 0.7, boxstyle="round,pad=0.05",
                               facecolor=fc, edgecolor=ec, linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x, y, label, fontsize=7, ha='center', va='center', fontweight='bold', color=ec)

    ax.text(8, 2.5, 'A/B Testing Framework:  Canary deployment (5% -> 25% -> 100% rollout)', 
            fontsize=9, ha='center', color='#1565c0', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#e3f2fd', edgecolor='#1565c0'))
    ax.text(8, 1.8, 'Rollback Trigger:  CTI violation rate > 1%  OR  inference latency > 100 ms  OR  Byzantine detection > 2 agents', 
            fontsize=8, ha='center', color='#c62828',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#ffebee', edgecolor='#c62828'))
    ax.text(8, 1.0, 'Data Flow Rates:  SCADA (1 Hz) -> PMU (60 Hz) -> Twin (event-driven) -> Edge (10 Hz inference) -> IED (GOOSE < 3 ms)', 
            fontsize=8, ha='center', style='italic', color='#555')

    plt.tight_layout()
    plt.savefig(output_dir + '/fig52_digital_twin_pipeline.png', dpi=200, bbox_inches='tight', facecolor='white')
    plt.savefig(output_dir + '/fig52_digital_twin_pipeline.pdf', dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Digital Twin Training-to-Deployment Pipeline')
    parser.add_argument('--output', type=str, default='results/figs_01_10',
                        help='Output directory for the figure')
    args = parser.parse_args()
    generate_figure(args.output)
    print(f"Figure saved to {args.output}")
