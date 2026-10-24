#!/usr/bin/env python3
"""
IEC 61850 Deployment Architecture
Generated from figs.py — standalone executable script for Phase 1.
Usage: python exp_fig17.py --output <path>
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
    """Generate IEC 61850 Deployment Architecture"""
    os.makedirs(output_dir, exist_ok=True)



    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 12)
    ax.axis('off')
    ax.set_title('Figure 17: IEC 61850 Deployment Architecture for Real-Time Relay Coordination', 
                 fontsize=14, fontweight='bold', pad=20, color='#212121')

    # Color scheme
    c_phys = '#e3f2fd'
    ec_phys = '#1565c0'
    c_comm = '#e8f5e9'
    ec_comm = '#2e7d32'
    c_edge = '#fff3e0'
    ec_edge = '#e65100'
    c_cloud = '#fce4ec'
    ec_cloud = '#ad1457'

    # Physical Layer
    phys_rect = FancyBboxPatch((0.5, 0.3), 15, 2.4, boxstyle="round,pad=0.12",
                                facecolor=c_phys, edgecolor=ec_phys, linewidth=2.5)
    ax.add_patch(phys_rect)
    ax.text(8, 2.55, 'PHYSICAL LAYER — Field Equipment', fontsize=12, ha='center', fontweight='bold', color=ec_phys)

    # Relay IEDs
    relay_positions = [(2, 1.4), (4.8, 1.4), (7.6, 1.4), (10.4, 1.4), (13.2, 1.4)]
    for i, (x, y) in enumerate(relay_positions):
        rect = FancyBboxPatch((x-0.7, y-0.45), 1.4, 0.9, boxstyle="round,pad=0.06",
                               facecolor='white', edgecolor=ec_phys, linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x, y+0.05, f'Relay {i+1}', fontsize=9, ha='center', va='center', fontweight='bold')
        ax.text(x, y-0.22, 'IED (PTOC)', fontsize=7, ha='center', va='center', color='#666')

    # CT/PT
    ct_rect = FancyBboxPatch((0.8, 0.55), 1.0, 0.55, boxstyle="round,pad=0.04",
                              facecolor='#fff3e0', edgecolor='#f57c00', linewidth=1.2)
    ax.add_patch(ct_rect)
    ax.text(1.3, 0.82, 'CT/PT', fontsize=8, ha='center', va='center', fontweight='bold')
    ax.text(1.3, 0.62, 'TCTR/TVTR', fontsize=6, ha='center', va='center', color='#666')

    # Circuit Breaker
    cb_rect = FancyBboxPatch((14.0, 0.55), 1.2, 0.55, boxstyle="round,pad=0.04",
                              facecolor='#ffcdd2', edgecolor='#c62828', linewidth=1.2)
    ax.add_patch(cb_rect)
    ax.text(14.6, 0.82, 'Breaker', fontsize=8, ha='center', va='center', fontweight='bold')
    ax.text(14.6, 0.62, 'XCBR', fontsize=6, ha='center', va='center', color='#666')

    # Communication Layer
    comm_rect = FancyBboxPatch((0.5, 3.0), 15, 2.4, boxstyle="round,pad=0.12",
                                facecolor=c_comm, edgecolor=ec_comm, linewidth=2.5)
    ax.add_patch(comm_rect)
    ax.text(8, 5.25, 'COMMUNICATION LAYER — IEC 61850 Protocol Stack', fontsize=12, ha='center', fontweight='bold', color=ec_comm)

    # GOOSE
    rect = FancyBboxPatch((1.5, 3.3), 3.0, 0.9, boxstyle="round,pad=0.06",
                           facecolor='white', edgecolor=ec_comm, linewidth=1.5)
    ax.add_patch(rect)
    ax.text(3.0, 3.9, 'GOOSE', fontsize=10, ha='center', va='center', fontweight='bold')
    ax.text(3.0, 3.55, 'Setting Updates\n(< 3 ms, T0–T3)', fontsize=7, ha='center', va='center', color='#666')

    # SV
    rect = FancyBboxPatch((5.5, 3.3), 3.0, 0.9, boxstyle="round,pad=0.06",
                           facecolor='white', edgecolor=ec_comm, linewidth=1.5)
    ax.add_patch(rect)
    ax.text(7.0, 3.9, 'Sampled Values (SV)', fontsize=10, ha='center', va='center', fontweight='bold')
    ax.text(7.0, 3.55, 'I/V Measurements\n(4 kHz, 80 samples/cycle)', fontsize=7, ha='center', va='center', color='#666')

    # MMS
    rect = FancyBboxPatch((9.5, 3.3), 3.0, 0.9, boxstyle="round,pad=0.06",
                           facecolor='white', edgecolor=ec_comm, linewidth=1.5)
    ax.add_patch(rect)
    ax.text(11.0, 3.9, 'MMS', fontsize=10, ha='center', va='center', fontweight='bold')
    ax.text(11.0, 3.55, 'Configuration\nClient/Server (TCP)', fontsize=7, ha='center', va='center', color='#666')

    # Process Bus label
    ax.text(14.2, 3.6, 'Process\nBus', fontsize=8, ha='center', va='center', fontweight='bold', 
            color=ec_comm, rotation=90)

    # Edge Compute Layer
    edge_rect = FancyBboxPatch((0.5, 5.7), 15, 2.4, boxstyle="round,pad=0.12",
                                facecolor=c_edge, edgecolor=ec_edge, linewidth=2.5)
    ax.add_patch(edge_rect)
    ax.text(8, 7.95, 'EDGE COMPUTE LAYER — Real-Time Inference', fontsize=12, ha='center', fontweight='bold', color=ec_edge)

    # ONNX Inference
    rect = FancyBboxPatch((1.5, 6.0), 3.0, 0.9, boxstyle="round,pad=0.06",
                           facecolor='white', edgecolor=ec_edge, linewidth=1.5)
    ax.add_patch(rect)
    ax.text(3.0, 6.6, 'ONNX Runtime', fontsize=10, ha='center', va='center', fontweight='bold')
    ax.text(3.0, 6.25, 'Inference: 45 ms\n(95th: 82 ms)', fontsize=7, ha='center', va='center', color='#666')

    # Byzantine Filter
    rect = FancyBboxPatch((5.5, 6.0), 3.0, 0.9, boxstyle="round,pad=0.06",
                           facecolor='white', edgecolor=ec_edge, linewidth=1.5)
    ax.add_patch(rect)
    ax.text(7.0, 6.6, 'Byzantine Filter', fontsize=10, ha='center', va='center', fontweight='bold')
    ax.text(7.0, 6.25, 'Median Consensus\n(50% fault tolerance)', fontsize=7, ha='center', va='center', color='#666')

    # Policy Model
    rect = FancyBboxPatch((9.5, 6.0), 3.0, 0.9, boxstyle="round,pad=0.06",
                           facecolor='white', edgecolor=ec_edge, linewidth=1.5)
    ax.add_patch(rect)
    ax.text(11.0, 6.6, 'Trained Policy', fontsize=10, ha='center', va='center', fontweight='bold')
    ax.text(11.0, 6.25, 'MAPPO-Safe-EWC\n(2.1 MB/agent, FP32)', fontsize=7, ha='center', va='center', color='#666')

    # Safety Monitor
    rect = FancyBboxPatch((13.0, 6.0), 2.0, 0.9, boxstyle="round,pad=0.06",
                           facecolor='#ffebee', edgecolor='#c62828', linewidth=1.5)
    ax.add_patch(rect)
    ax.text(14.0, 6.6, 'Safety Monitor', fontsize=9, ha='center', va='center', fontweight='bold', color='#c62828')
    ax.text(14.0, 6.25, 'CTI Guard\nHard Override', fontsize=7, ha='center', va='center', color='#666')

    # Cloud Training Layer
    cloud_rect = FancyBboxPatch((0.5, 8.4), 15, 2.2, boxstyle="round,pad=0.12",
                                 facecolor=c_cloud, edgecolor=ec_cloud, linewidth=2.5)
    ax.add_patch(cloud_rect)
    ax.text(8, 10.35, 'CLOUD TRAINING LAYER — Offline Model Development', fontsize=12, ha='center', fontweight='bold', color=ec_cloud)

    rect = FancyBboxPatch((1.5, 8.7), 3.0, 0.8, boxstyle="round,pad=0.05",
                           facecolor='white', edgecolor=ec_cloud, linewidth=1.5)
    ax.add_patch(rect)
    ax.text(3.0, 9.25, 'Training Cluster', fontsize=10, ha='center', va='center', fontweight='bold')
    ax.text(3.0, 8.95, 'PyTorch + PyG\n(RTX 3090 / A100)', fontsize=7, ha='center', va='center', color='#666')

    rect = FancyBboxPatch((5.5, 8.7), 3.0, 0.8, boxstyle="round,pad=0.05",
                           facecolor='white', edgecolor=ec_cloud, linewidth=1.5)
    ax.add_patch(rect)
    ax.text(7.0, 9.25, 'Model Export', fontsize=10, ha='center', va='center', fontweight='bold')
    ax.text(7.0, 8.95, 'ONNX + INT8 Quant\n+ Pruning', fontsize=7, ha='center', va='center', color='#666')

    rect = FancyBboxPatch((9.5, 8.7), 3.0, 0.8, boxstyle="round,pad=0.05",
                           facecolor='white', edgecolor=ec_cloud, linewidth=1.5)
    ax.add_patch(rect)
    ax.text(11.0, 9.25, 'OTA Update', fontsize=10, ha='center', va='center', fontweight='bold')
    ax.text(11.0, 8.95, 'Secure Firmware\n(TLS 1.3 + Sig)', fontsize=7, ha='center', va='center', color='#666')

    rect = FancyBboxPatch((13.0, 8.7), 2.0, 0.8, boxstyle="round,pad=0.05",
                           facecolor='white', edgecolor=ec_cloud, linewidth=1.5)
    ax.add_patch(rect)
    ax.text(14.0, 9.25, 'MLflow', fontsize=10, ha='center', va='center', fontweight='bold')
    ax.text(14.0, 8.95, 'Version Control\n+ Metrics', fontsize=7, ha='center', va='center', color='#666')

    # Arrows between layers
    for x in [2, 4.8, 7.6, 10.4, 13.2]:
        ax.annotate('', xy=(x, 3.0), xytext=(x, 2.7),
                    arrowprops=dict(arrowstyle='<->', color='#333', lw=1.5))
        ax.annotate('', xy=(x, 5.7), xytext=(x, 5.4),
                    arrowprops=dict(arrowstyle='<->', color='#333', lw=1.5))
        ax.annotate('', xy=(x, 8.4), xytext=(x, 8.1),
                    arrowprops=dict(arrowstyle='<->', color='#333', lw=1.5))

    # Latency annotations
    ax.text(0.2, 2.85, '<3 ms', fontsize=8, ha='center', color='#666', rotation=90)
    ax.text(0.2, 5.55, '<10 ms', fontsize=8, ha='center', color='#666', rotation=90)
    ax.text(0.2, 8.25, 'Offline', fontsize=8, ha='center', color='#666', rotation=90)

    # End-to-end latency banner
    banner = FancyBboxPatch((1.0, 0.05), 14, 0.55, boxstyle="round,pad=0.08",
                             facecolor=c_phys, edgecolor=ec_phys, linewidth=2)
    ax.add_patch(banner)
    ax.text(8, 0.32, 'End-to-End Latency: Measurement → Inference → Setting Update = 67 ms (< 100 ms protection target)', 
            fontsize=10, ha='center', fontweight='bold', color=ec_phys)

    # Right-side annotations
    ax.text(15.7, 1.4, 'IEC 61850-7-4\nLogical Nodes', fontsize=8, ha='center', va='center', 
            color=ec_phys, fontweight='bold', rotation=90)
    ax.text(15.7, 4.2, 'IEC 61850-8-1\nGOOSE/SV/MMS', fontsize=8, ha='center', va='center', 
            color=ec_comm, fontweight='bold', rotation=90)
    ax.text(15.7, 6.9, 'Intel NUC i5\n16 GB RAM', fontsize=8, ha='center', va='center', 
            color=ec_edge, fontweight='bold', rotation=90)
    ax.text(15.7, 9.5, 'AWS/GCP/Azure\nGPU Cluster', fontsize=8, ha='center', va='center', 
            color=ec_cloud, fontweight='bold', rotation=90)

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=c_phys, edgecolor=ec_phys, label='Physical Layer (Field IEDs)'),
        mpatches.Patch(facecolor=c_comm, edgecolor=ec_comm, label='Communication (IEC 61850)'),
        mpatches.Patch(facecolor=c_edge, edgecolor=ec_edge, label='Edge Compute (Inference)'),
        mpatches.Patch(facecolor=c_cloud, edgecolor=ec_cloud, label='Cloud Training (Offline)'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9, framealpha=0.95)

    plt.tight_layout()
    plt.savefig(output_dir + '/fig17_iec61850_deployment.png', dpi=200, bbox_inches='tight', facecolor='white')
    plt.savefig(output_dir + '/fig17_iec61850_deployment.pdf', dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='IEC 61850 Deployment Architecture')
    parser.add_argument('--output', type=str, default='results/figs_01_10',
                        help='Output directory for the figure')
    args = parser.parse_args()
    generate_figure(args.output)
    print(f"Figure saved to {args.output}")
