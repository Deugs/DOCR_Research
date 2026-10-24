#!/usr/bin/env python3
"""
High-Level System Architecture Diagram
Generated from figs.py — standalone executable script for Phase 1.
Usage: python exp_fig01.py --output <path>
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
    """Generate High-Level System Architecture Diagram"""
    os.makedirs(output_dir, exist_ok=True)


    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('Figure 1: High-Level System Architecture', fontsize=14, fontweight='bold', pad=20)

    # Layer definitions with colors
    layers = [
        {'name': 'Layer 5: Evaluation & Deployment', 'y': 8.5, 'color': '#e8f4f8', 'items': ['BenchmarkSuite', 'StatAnalysis', 'ResultsAggregator', 'InferenceEngine']},
        {'name': 'Layer 4: Safety & Adaptation', 'y': 6.8, 'color': '#fff3e0', 'items': ['ActionProjector', 'AugmentedLagrangian', 'FisherInfoEstimator', 'EWCRegularizer']},
        {'name': 'Layer 3: Multi-Agent RL', 'y': 5.1, 'color': '#e8f5e9', 'items': ['DynamicDOCREnv', 'SharedGaussianActor', 'MAPPOCentralCritic', 'CommGraph']},
        {'name': 'Layer 2: Scenario Generation', 'y': 3.4, 'color': '#fce4ec', 'items': ['ScenarioGenerator', 'MarkovCloudModel', 'DailyProfileGenerator']},
        {'name': 'Layer 1: Physical Environment', 'y': 1.7, 'color': '#f3e5f5', 'items': ['DistributionNetwork', 'PVInverter', 'TopologyManager', 'RelaySensor']},
    ]

    for layer in layers:
        # Layer background
        rect = FancyBboxPatch((0.3, layer['y']-0.5), 13.4, 1.3, 
                               boxstyle="round,pad=0.05", 
                               facecolor=layer['color'], edgecolor='#333', linewidth=1.5)
        ax.add_patch(rect)
    
        # Layer name
        ax.text(0.5, layer['y']+0.5, layer['name'], fontsize=11, fontweight='bold', va='center')
    
        # Component boxes
        n_items = len(layer['items'])
        width = 2.8
        start_x = (14 - n_items * width) / 2 + 0.3
        for i, item in enumerate(layer['items']):
            x = start_x + i * width
            comp_rect = FancyBboxPatch((x, layer['y']-0.3), width-0.2, 0.6,
                                        boxstyle="round,pad=0.03",
                                        facecolor='white', edgecolor='#555', linewidth=1)
            ax.add_patch(comp_rect)
            ax.text(x + (width-0.2)/2, layer['y'], item, fontsize=8, ha='center', va='center')

    # Arrows between layers
    for i in range(len(layers)-1):
        y_top = layers[i]['y'] - 0.5
        y_bottom = layers[i+1]['y'] + 0.8
        ax.annotate('', xy=(7, y_bottom), xytext=(7, y_top),
                    arrowprops=dict(arrowstyle='<->', color='#333', lw=2))

    plt.tight_layout()
    plt.savefig(output_dir + '/fig1_system_architecture.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.savefig(output_dir + '/fig1_system_architecture.pdf', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='High-Level System Architecture Diagram')
    parser.add_argument('--output', type=str, default='results/figs_01_10',
                        help='Output directory for the figure')
    args = parser.parse_args()
    generate_figure(args.output)
    print(f"Figure saved to {args.output}")
