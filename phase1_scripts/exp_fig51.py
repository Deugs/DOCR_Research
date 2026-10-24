#!/usr/bin/env python3
"""
Communication Protocol Stack (OSI Layer Mapping)
Generated from figs.py — standalone executable script for Phase 1.
Usage: python exp_fig51.py --output <path>
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
    """Generate Communication Protocol Stack (OSI Layer Mapping)"""
    os.makedirs(output_dir, exist_ok=True)


    fig, ax = plt.subplots(1, 1, figsize=(14, 12))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 14)
    ax.axis('off')
    ax.set_title('Figure 51: Communication Protocol Stack — OSI 7-Layer Mapping for GNN Message Passing onto IEC 61850', 
                 fontsize=13, fontweight='bold', pad=20, color=colors['dark'])

    # OSI Layer blocks (left side)
    layers = [
        (7, 'Application Layer', 'MARL Agent Messages\nMMS (Manufacturing Message Specification)\nACSI (Abstract Communication Service Interface)', 
         '#e8f5e9', '#2e7d32', 2.2),
        (6, 'Presentation Layer', 'ASN.1 BER Encoding\nX.690 Basic Encoding Rules\nMessage Serialization: 8 floats → 32 bytes', 
         '#fff3e0', '#f57c00', 1.8),
        (5, 'Session Layer', 'TLS 1.3 Secure Session\nMutual Authentication\nSession Key Exchange (ECDHE)', 
         '#e3f2fd', '#1565c0', 1.6),
        (4, 'Transport Layer', 'TCP (Model Updates, >1 KB)\nUDP + Custom Reliability (Agent Messages, <100 B)\nPort: 61850 (IEC reserved)', 
         '#fce4ec', '#c2185b', 1.8),
        (3, 'Network Layer', 'IPv4 / IPv6 Routing\nVLAN Segmentation (Protection VLAN 100)\nMulticast: 239.0.0.1 (GOOSE group)', 
         '#f3e5f5', '#7b1fa2', 1.6),
        (2, 'Data Link Layer', 'IEC 61850-8-1 GOOSE (Type 1A, <3ms)\nIEC 61850-9-2 SV (Sampled Values, 4 kHz)\nEthernet II, Priority Tagging (IEEE 802.1Q)', 
         '#e0f7fa', '#00838f', 2.0),
        (1, 'Physical Layer', '100 Mbps Ethernet (Copper/Fiber)\nProcess Bus: Fiber Optic (62.5/125 μm)\nStation Bus: Copper Cat 6A, <100m', 
         '#efebe9', '#5d4037', 1.6),
    ]

    y_start = 12.5
    for layer_num, name, content, fc, ec, h in layers:
        y = y_start - (7 - layer_num) * 1.7
    
        # Layer number circle
        circle = Circle((0.8, y + h/2), 0.4, facecolor=ec, edgecolor='white', linewidth=2)
        ax.add_patch(circle)
        ax.text(0.8, y + h/2, str(layer_num), fontsize=10, ha='center', va='center', 
                color='white', fontweight='bold')
    
        # Layer block
        rect = FancyBboxPatch((1.5, y), 5.5, h, boxstyle="round,pad=0.08",
                               facecolor=fc, edgecolor=ec, linewidth=2.5)
        ax.add_patch(rect)
        ax.text(1.7, y + h - 0.25, name, fontsize=10, ha='left', va='top', fontweight='bold', color=ec)
        ax.text(1.7, y + 0.2, content, fontsize=7, ha='left', va='bottom', color='#444')

    # Right side: Protocol mapping details
    mapping_x = 8.0

    # Title
    ax.text(mapping_x, 13.2, 'GNN Message → IEC 61850 Mapping', fontsize=12, fontweight='bold', 
            color=colors['dark'],
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#fafafa', edgecolor=colors['dark']))

    # Message structure
    ax.text(mapping_x, 12.3, 'GNN Message Structure (32 bytes):', fontsize=10, fontweight='bold', color=colors['primary'])
    msg_fields = [
        ('Bytes 0–3', 'float32', 'Feature 1: Local V_mag (normalized)'),
        ('Bytes 4–7', 'float32', 'Feature 2: Local I_mag (normalized)'),
        ('Bytes 8–11', 'float32', 'Feature 3: P_dir (−1, 0, 1 encoded)'),
        ('Bytes 12–15', 'float32', 'Feature 4: I_fault (log-compressed)'),
        ('Bytes 16–19', 'float32', 'Feature 5: PV generation level (%)'),
        ('Bytes 20–23', 'float32', 'Feature 6: Topology flag (0/1)'),
        ('Bytes 24–27', 'float32', 'Feature 7: Neighbor TMS (avg)'),
        ('Bytes 28–31', 'float32', 'Feature 8: Neighbor PS (avg)'),
    ]

    y_msg = 11.8
    for bytes_range, dtype, desc in msg_fields:
        ax.text(mapping_x, y_msg, f'{bytes_range}', fontsize=7, ha='left', va='top', 
                fontweight='bold', color=colors['dark'])
        ax.text(mapping_x + 1.3, y_msg, f'[{dtype}]', fontsize=7, ha='left', va='top', 
                color=colors['secondary'])
        ax.text(mapping_x + 2.3, y_msg, desc, fontsize=7, ha='left', va='top', color='#555')
        y_msg -= 0.35

    # GOOSE mapping
    ax.text(mapping_x, 8.8, 'GOOSE Message Mapping (Setting Updates):', fontsize=10, fontweight='bold', color=colors['secondary'])
    goose_items = [
        '• stVal (BOOLEAN): Setting update trigger',
        '• f (FLOAT32): New TMS value (×1000 for precision)',
        '• f (FLOAT32): New PS value (×100 for precision)',
        '• q (Quality): Validity flag + timestamp',
        '• T0–T3: Retransmission sequence (2, 4, 8, 16 ms)',
    ]
    for i, item in enumerate(goose_items):
        ax.text(mapping_x, 8.4 - i*0.35, item, fontsize=7, ha='left', va='top', color='#555')

    # SV mapping
    ax.text(mapping_x, 6.5, 'SV (Sampled Values) Mapping (Measurements):', fontsize=10, fontweight='bold', color=colors['accent'])
    sv_items = [
        '• i (INT32): Instantaneous current (1 mA resolution)',
        '• v (INT32): Instantaneous voltage (10 mV resolution)',
        '• SmpCnt: Sample counter (0–3999 for 4 kHz)',
        '• SmpSynch: Synchronization status (IEEE 1588 PTP)',
    ]
    for i, item in enumerate(sv_items):
        ax.text(mapping_x, 6.1 - i*0.35, item, fontsize=7, ha='left', va='top', color='#555')

    # MMS mapping
    ax.text(mapping_x, 4.5, 'MMS Mapping (Configuration & Monitoring):', fontsize=10, fontweight='bold', color=colors['danger'])
    mms_items = [
        '• GetDataValues: Read current relay settings (TMS, PS)',
        '• SetDataValues: Write new settings (authenticated)',
        '• Report (URCB): Unbuffered report on data change',
        '• FileTransfer: Model update (ONNX binary, ~2 MB)',
    ]
    for i, item in enumerate(mms_items):
        ax.text(mapping_x, 4.1 - i*0.35, item, fontsize=7, ha='left', va='top', color='#555')

    # Performance metrics box
    perf_rect = FancyBboxPatch((mapping_x - 0.2, 2.0), 5.5, 1.5, boxstyle="round,pad=0.1",
                                facecolor='#fafafa', edgecolor=colors['primary'], linewidth=2)
    ax.add_patch(perf_rect)
    ax.text(mapping_x, 3.2, 'End-to-End Performance Metrics:', fontsize=10, fontweight='bold', color=colors['primary'])
    ax.text(mapping_x, 2.8, '• GNN message latency:  < 3 ms (GOOSE T0 retransmission)', fontsize=8, color='#444')
    ax.text(mapping_x, 2.45, '• SV measurement delay:  < 250 μs (4 kHz sampling)', fontsize=8, color='#444')
    ax.text(mapping_x, 2.1, '• Total inference-to-action:  45 ms (mean)  |  82 ms (95th percentile)', fontsize=8, color='#444')

    # Bandwidth annotation
    ax.text(mapping_x, 1.5, 'Bandwidth Budget:', fontsize=10, fontweight='bold', color=colors['secondary'])
    ax.text(mapping_x, 1.15, '• Per-agent GNN:  0.256 kbps  (8 floats × 4 bytes × 10 Hz)', fontsize=8, color='#444')
    ax.text(mapping_x, 0.8, '• GOOSE overhead:  ~0.05 kbps  (setting updates, ~1 per minute)', fontsize=8, color='#444')
    ax.text(mapping_x, 0.45, '• SV reception:  ~64 kbps  (4 kHz × 8 bytes × 2 channels)', fontsize=8, color='#444')
    ax.text(mapping_x, 0.1, '• Total per-agent:  ~64.3 kbps  <<  AMI/RF mesh limit (1 Mbps)', fontsize=8, fontweight='bold', color=colors['secondary'])

    # Legend for layer colors
    legend_elements = [
        mpatches.Patch(facecolor='#e8f5e9', edgecolor='#2e7d32', label='Application'),
        mpatches.Patch(facecolor='#fff3e0', edgecolor='#f57c00', label='Presentation'),
        mpatches.Patch(facecolor='#e3f2fd', edgecolor='#1565c0', label='Session'),
        mpatches.Patch(facecolor='#fce4ec', edgecolor='#c2185b', label='Transport'),
        mpatches.Patch(facecolor='#f3e5f5', edgecolor='#7b1fa2', label='Network'),
        mpatches.Patch(facecolor='#e0f7fa', edgecolor='#00838f', label='Data Link'),
        mpatches.Patch(facecolor='#efebe9', edgecolor='#5d4037', label='Physical'),
    ]
    ax.legend(handles=legend_elements, loc='lower left', fontsize=8, ncol=4, 
              title='OSI Layers', title_fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir + '/fig51_protocol_stack.png', dpi=200, bbox_inches='tight', facecolor='white')
    plt.savefig(output_dir + '/fig51_protocol_stack.pdf', dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Communication Protocol Stack (OSI Layer Mapping)')
    parser.add_argument('--output', type=str, default='results/figs_01_10',
                        help='Output directory for the figure')
    args = parser.parse_args()
    generate_figure(args.output)
    print(f"Figure saved to {args.output}")
