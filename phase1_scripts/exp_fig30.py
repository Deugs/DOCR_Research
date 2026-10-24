#!/usr/bin/env python3
"""
Complete Algorithm Pseudocode Block
Generated from figs.py — standalone executable script for Phase 1.
Usage: python exp_fig30.py --output <path>
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
    """Generate Complete Algorithm Pseudocode Block"""
    os.makedirs(output_dir, exist_ok=True)


    fig, ax = plt.subplots(1, 1, figsize=(12, 14))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 14)
    ax.axis('off')
    ax.set_title('Algorithm 1: MAPPO-Safe-EWC Training Procedure', fontsize=14, fontweight='bold', pad=20)

    # Algorithm box
    algo_box = FancyBboxPatch((0.5, 0.5), 11, 13, boxstyle="round,pad=0.1",
                               facecolor='#fafafa', edgecolor='#333', linewidth=2)
    ax.add_patch(algo_box)

    # Algorithm text
    algo_text = """
    Require: Network N, PV penetration levels P = {0%, 20%, ..., 100%}
    Require: Curriculum stages S = {0, 1, ..., 5}, advancement threshold τ = 0.95
    Require: Hyperparameters: γ, λ_GAE, ε_clip, lr_actor, lr_critic, λ_EWC

    1:  Initialize shared actor parameters θ, centralized critic φ
    2:  Initialize Lagrange multipliers λ_j ← 1.0 ∀j ∈ P/B pairs
    3:  Initialize penalty coefficient ρ ← 10.0
    4:  Initialize EWC task memory M ← ∅
    5:  
    6:  for stage s = 0 to 5 do
    7:      Set PV level p_s ∈ P and topology condition
    8:      
    9:      while CTI_satisfaction_rate < τ over 100 episodes do
    10:         // Collect rollouts
    11:         for t = 1 to T_rollout do
    12:             for each agent i = 1 to N do
    13:                 o_i ← local_observation(V, I, P_dir, I_fault, ...)
    14:                 m_i ← Encoder_θ(o_i)                    // 10-dim → 8-float
    15:                 m̄_i ← Σ_j∈N_i α_ij · m_j                // GNN attention
    16:                 õ_i ← [o_i ‖ m̄_i]                      // 18-dim enhanced obs
    17:                 a_i ← μ_θ(õ_i) + σ_θ · ε,  ε ~ N(0,I)  // Gaussian sample
    18:                 a_i^safe ← P(a_i)                        // Action projection
    19:             end for
    20:             
    21:             s' ← T(s, a^safe, ξ)                        // Environment step
    22:             r ← R(s, a^safe)                            // Team reward
    23:             Store (o, õ, a^safe, r, o', V_φ(s), log π_θ) in buffer
    24:         end for
    25:         
    26:         // PPO policy update
    27:         Compute advantages Â_t using GAE(λ_GAE, γ)
    28:         L_CLIP(θ) = E_t[min(r_t(θ)·Â_t, clip(r_t(θ), 1±ε)·Â_t)]
    29:         
    30:         // Safe RL constraint cost
    31:         g_j ← 0.3 - (t_bkp,j - t_pri,j)  ∀j ∈ P/B pairs
    32:         L_AL(θ) = -Σ_j λ_j·E[g_j] - (ρ/2)·Σ_j E[max(0, g_j)²]
    33:         
    34:         // EWC continual learning penalty
    35:         if |M| > 0 then
    36:             L_EWC(θ) = (λ_EWC/2)·Σ_task∈M Σ_i F_task,i·(θ_i - θ*_task,i)²
    37:         end if
    38:         
    39:         // Combined gradient update
    40:         θ ← θ - α·∇_θ[L_CLIP + L_AL + L_EWC]
    41:         φ ← φ - α_critic·∇_φ[(V_φ(s) - R_GAE)²]
    42:         
    43:         // Dual ascent for Lagrange multipliers
    44:         λ_j ← max(0, λ_j + α_λ·ḡ_j)  with PID stabilization
    45:         
    46:         // Adaptive penalty coefficient
    47:         if violation_rate > 0.1 then ρ ← ρ × 1.5
    48:         elif violation_rate < 0.05 then ρ ← ρ / 1.5
    49:         end if
    50:     end while
    51:     
    52:     // EWC task consolidation at stage transition
    53:     F ← estimate_fisher(θ, D_current)      // Monte Carlo sampling
    54:     M ← M ∪ {(task_s, θ, F, λ_EWC)}       // Store task memory
    55:     
    56:     if task_similarity(new_task, M) > 0.8 then
    57:         θ ← warm_start_from_most_similar(M)
    58:     end if
    59: end for
    60: 
    61: return θ*, φ*, λ*, M
    """

    ax.text(6, 7, algo_text, fontsize=7.5, ha='center', va='center', 
            family='monospace', transform=ax.transData,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='none'))

    plt.tight_layout()
    plt.savefig(output_dir + '/fig30_algorithm_pseudocode.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.savefig(output_dir + '/fig30_algorithm_pseudocode.pdf', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Complete Algorithm Pseudocode Block')
    parser.add_argument('--output', type=str, default='results/figs_01_10',
                        help='Output directory for the figure')
    args = parser.parse_args()
    generate_figure(args.output)
    print(f"Figure saved to {args.output}")
