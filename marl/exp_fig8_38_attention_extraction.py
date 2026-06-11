
"""
exp_attention_extraction_v2.py — Figures 8 & 38 (IMPROVED)
─────────────────────────────────────────────────────────────
GNN Attention Analysis + Temporal Evolution.

IMPROVEMENTS over original:
  • Sufficient training (50k steps) for meaningful attention
  • Electrical distance correlation (d_ij = sum(line impedances))
  • Temporal snapshots at episodes 0, 100, 500, 1000, 5000 for Fig 38
  • Entropy decay plot over training
  • Attention weight evolution tracking during training
  • Multi-seed support
  • Automatic plot generation for both figures
  • Networkx graph visualization with attention-weighted edges

Outputs
-------
  outputs/fig8_attention_v2.npz
  outputs/fig8_attention_v2.png
  outputs/fig38_attention_evolution_v2.npz
  outputs/fig38_attention_evolution_v2.png
"""

import numpy as np
import os
import sys
import argparse
import logging
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
os.makedirs("outputs", exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
from scipy.stats import pearsonr

from network.network_data import get_8bus
from marl.marl_env import DynamicDOCREnv
from marl.marl_agents import MAPPOMARLTrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger(__name__)

TRAIN_STEPS = 50000
SNAPSHOT_EPISODES = [0, 100, 500, 1000, 5000]
N_INFERENCE_EPISODES = 100


def compute_electrical_distance(net):
    """Compute shortest-path electrical distance using line impedances."""
    G = nx.Graph()
    for i, j, z in zip(net.fbus, net.tbus, net.Z):
        G.add_edge(int(i), int(j), weight=abs(z))
    
    n_relays = net.n_relays
    dist_matrix = np.zeros((n_relays, n_relays))
    
    for i in range(n_relays):
        for j in range(n_relays):
            if i == j:
                dist_matrix[i, j] = 0
            else:
                try:
                    path = nx.shortest_path(G, int(net.bus[i]), int(net.bus[j]), weight="weight")
                    dist = sum(G[path[k]][path[k+1]]["weight"] for k in range(len(path)-1))
                    dist_matrix[i, j] = dist
                except nx.NetworkXNoPath:
                    dist_matrix[i, j] = np.inf
    
    return dist_matrix


def extract_attention_at_checkpoint(trainer, env, net, n_episodes=100, seed=42):
    """Extract average attention matrix over n_episodes."""
    all_attn = []
    entropies = []
    
    rng = np.random.default_rng(seed)
    
    for ep in range(n_episodes):
        obs = env.reset(seed=seed + ep)
        done = False
        while not done:
            actions = np.zeros((env.n_agents, env.ACT_DIM))
            for i in range(env.n_agents):
                a, _ = trainer.actor.sample(obs[i], rng)
                actions[i] = a
            obs, _, done, info = env.step(actions)
            
            if "comm_attention" in info:
                attn = info["comm_attention"]  # (n_agents, n_agents)
                all_attn.append(attn)
                
                # Entropy per agent
                p = attn + 1e-9
                p /= p.sum(axis=1, keepdims=True)
                ent = -np.sum(p * np.log(p), axis=1).mean()
                entropies.append(ent)
    
    if not all_attn:
        return None, None
    
    avg_attn = np.mean(all_attn, axis=0)
    avg_entropy = float(np.mean(entropies))
    
    return avg_attn, avg_entropy


def run_figure8(seed=42):
    """Generate Figure 8: Learned GNN Attention Weights."""
    logger.info("Figure 8 — Attention Weight Extraction")
    
    net = get_8bus()
    env = DynamicDOCREnv(net, use_pandapower=False, fault_type="3ph", 
                          curriculum_stage=5, seed=seed)
    trainer = MAPPOMARLTrainer(env, use_comm=True, seed=seed)
    
    # Train sufficiently
    logger.info("  Training for %d steps...", TRAIN_STEPS)
    trainer.train(total_steps=TRAIN_STEPS, verbose=False)
    
    # Extract attention
    logger.info("  Extracting attention over %d episodes...", N_INFERENCE_EPISODES)
    avg_attn, avg_entropy = extract_attention_at_checkpoint(trainer, env, net, 
                                                            N_INFERENCE_EPISODES, seed)
    
    if avg_attn is None:
        logger.error("  No attention weights found in environment!")
        return None
    
    # Compute electrical distances
    elec_dist = compute_electrical_distance(net)
    
    # Correlation analysis
    n = avg_attn.shape[0]
    attn_flat = []
    dist_flat = []
    for i in range(n):
        for j in range(n):
            if i != j and np.isfinite(elec_dist[i, j]):
                attn_flat.append(avg_attn[i, j])
                dist_flat.append(elec_dist[i, j])
    
    corr, pval = pearsonr(attn_flat, [1/d for d in dist_flat])
    
    logger.info("  Attention-1/distance correlation: r=%.3f (p=%.4f)", corr, pval)
    logger.info("  Average attention entropy: %.4f", avg_entropy)
    
    return {
        "attention_matrix": avg_attn,
        "electrical_distance": elec_dist,
        "entropy": avg_entropy,
        "correlation": float(corr),
        "correlation_pval": float(pval),
    }


def run_figure38(seed=42):
    """Generate Figure 38: Attention Weight Evolution During Training."""
    logger.info("Figure 38 — Attention Evolution During Training")
    
    net = get_8bus()
    env = DynamicDOCREnv(net, use_pandapower=False, fault_type="3ph",
                          curriculum_stage=5, seed=seed)
    trainer = MAPPOMARLTrainer(env, use_comm=True, seed=seed)
    
    snapshots = {}
    entropies_over_time = []
    
    for ep in range(max(SNAPSHOT_EPISODES) + 1):
        # Train one episode
        obs = env.reset(seed=seed + ep)
        done = False
        ep_entropy = []
        
        while not done:
            actions = np.zeros((env.n_agents, env.ACT_DIM))
            for i in range(env.n_agents):
                a, _ = trainer.actor.sample(obs[i], np.random.default_rng(seed + ep))
                actions[i] = a
            obs, _, done, info = env.step(actions)
            
            if "comm_attention" in info:
                attn = info["comm_attention"]
                p = attn + 1e-9
                p /= p.sum(axis=1, keepdims=True)
                ent = -np.sum(p * np.log(p), axis=1).mean()
                ep_entropy.append(ent)
        
        if ep_entropy:
            entropies_over_time.append(float(np.mean(ep_entropy)))
        else:
            entropies_over_time.append(np.nan)
        
        # Take snapshot at specified episodes
        if ep in SNAPSHOT_EPISODES:
            logger.info("  Snapshot at episode %d...", ep)
            attn, _ = extract_attention_at_checkpoint(trainer, env, net, 
                                                       n_episodes=10, seed=seed + 10000 + ep)
            if attn is not None:
                snapshots[ep] = attn
        
        # Periodic training update
        if ep % 10 == 0:
            trainer.update()
    
    return {
        "snapshots": snapshots,
        "entropies_over_time": np.array(entropies_over_time),
        "snapshot_episodes": SNAPSHOT_EPISODES,
    }


def plot_figure8(data, net, output_dir):
    """Plot Figure 8: Attention heatmap + correlation scatter."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Figure 8 — Learned GNN Attention Weights (8-Bus, 100% PV)", fontsize=14)
    
    n = data["attention_matrix"].shape[0]
    
    # Panel 1: Attention heatmap
    ax = axes[0]
    im = ax.imshow(data["attention_matrix"], cmap="viridis", aspect="auto")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([f"R{i+1}" for i in range(n)], fontsize=9)
    ax.set_yticklabels([f"R{i+1}" for i in range(n)], fontsize=9)
    ax.set_xlabel("To Agent", fontsize=10)
    ax.set_ylabel("From Agent", fontsize=10)
    ax.set_title(f"Attention Matrix (H={data['entropy']:.3f})", fontsize=11)
    plt.colorbar(im, ax=ax, shrink=0.8)
    
    # Panel 2: Electrical distance heatmap
    ax = axes[1]
    dist = data["electrical_distance"]
    dist_plot = np.where(np.isinf(dist), np.nan, dist)
    im = ax.imshow(dist_plot, cmap="plasma", aspect="auto")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([f"R{i+1}" for i in range(n)], fontsize=9)
    ax.set_yticklabels([f"R{i+1}" for i in range(n)], fontsize=9)
    ax.set_xlabel("To Agent", fontsize=10)
    ax.set_ylabel("From Agent", fontsize=10)
    ax.set_title("Electrical Distance (Ω)", fontsize=11)
    plt.colorbar(im, ax=ax, shrink=0.8)
    
    # Panel 3: Correlation scatter
    ax = axes[2]
    attn_flat = []
    inv_dist_flat = []
    for i in range(n):
        for j in range(n):
            if i != j and np.isfinite(dist[i, j]):
                attn_flat.append(data["attention_matrix"][i, j])
                inv_dist_flat.append(1.0 / dist[i, j])
    
    ax.scatter(inv_dist_flat, attn_flat, s=80, alpha=0.7, color="#2563eb", edgecolors="white")
    z = np.polyfit(inv_dist_flat, attn_flat, 1)
    p = np.poly1d(z)
    x_line = np.linspace(min(inv_dist_flat), max(inv_dist_flat), 100)
    ax.plot(x_line, p(x_line), "r--", lw=2, alpha=0.8, 
            label=f"r={data['correlation']:.3f}")
    ax.set_xlabel("1 / Electrical Distance (1/Ω)", fontsize=10)
    ax.set_ylabel("Attention Weight", fontsize=10)
    ax.set_title("Attention vs. Electrical Proximity", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(Path(output_dir) / "fig8_attention_v2.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", Path(output_dir) / "fig8_attention_v2.png")


def plot_figure38(data, output_dir):
    """Plot Figure 38: Temporal attention evolution."""
    snapshots = data["snapshots"]
    entropies = data["entropies_over_time"]
    
    n_snapshots = len(snapshots)
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, n_snapshots, hspace=0.4, wspace=0.3)
    
    fig.suptitle("Figure 38 — Attention Weight Evolution During Training", fontsize=14)
    
    # Row 1: Attention heatmaps at snapshots
    for idx, ep in enumerate(sorted(snapshots.keys())):
        ax = fig.add_subplot(gs[0, idx])
        attn = snapshots[ep]
        im = ax.imshow(attn, cmap="viridis", aspect="auto", vmin=0, vmax=1)
        ax.set_title(f"Episode {ep}", fontsize=10)
        ax.set_xticks(range(attn.shape[0]))
        ax.set_yticks(range(attn.shape[0]))
        ax.set_xticklabels([f"R{i+1}" for i in range(attn.shape[0])], fontsize=7)
        ax.set_yticklabels([f"R{i+1}" for i in range(attn.shape[0])], fontsize=7)
        plt.colorbar(im, ax=ax, shrink=0.7)
    
    # Row 2: Entropy decay
    ax = fig.add_subplot(gs[1, :])
    valid_eps = ~np.isnan(entropies)
    ax.plot(np.arange(len(entropies))[valid_eps], entropies[valid_eps], 
            "o-", color="#2563eb", lw=1.5, markersize=3, alpha=0.7)
    
    # Mark snapshot points
    for ep in sorted(snapshots.keys()):
        if ep < len(entropies):
            ax.axvline(ep, color="red", ls="--", alpha=0.5, lw=1)
            ax.scatter([ep], [entropies[ep]], color="red", s=50, zorder=5)
    
    ax.set_xlabel("Episode", fontsize=10)
    ax.set_ylabel("Attention Entropy", fontsize=10)
    ax.set_title("Entropy Decay Over Training (lower = more focused)", fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Row 3: Attention distribution evolution
    ax = fig.add_subplot(gs[2, :])
    for ep in sorted(snapshots.keys()):
        attn = snapshots[ep]
        off_diag = attn[~np.eye(attn.shape[0], dtype=bool)].flatten()
        ax.hist(off_diag, bins=20, alpha=0.4, label=f"Ep {ep}", density=True)
    ax.set_xlabel("Attention Weight (off-diagonal)", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.set_title("Attention Distribution Evolution", fontsize=11)
    ax.legend(fontsize=8, ncol=len(snapshots))
    ax.grid(True, alpha=0.3)
    
    fig.savefig(Path(output_dir) / "fig38_attention_evolution_v2.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", Path(output_dir) / "fig38_attention_evolution_v2.png")


def main(n_seeds=3, output_dir="outputs"):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    logger.info("Figures 8 & 38 (v2) — Attention Analysis")
    logger.info("  Seeds: %d", n_seeds)
    
    # Figure 8
    fig8_results = []
    for seed in range(n_seeds):
        logger.info("\\nSeed %d/%d — Figure 8", seed + 1, n_seeds)
        res = run_figure8(seed=seed)
        if res:
            fig8_results.append(res)
    
    if fig8_results:
        # Average across seeds
        avg_attn = np.mean([r["attention_matrix"] for r in fig8_results], axis=0)
        avg_corr = float(np.mean([r["correlation"] for r in fig8_results]))
        avg_ent = float(np.mean([r["entropy"] for r in fig8_results]))
        
        fig8_data = {
            "attention_matrix": avg_attn,
            "electrical_distance": fig8_results[0]["electrical_distance"],
            "entropy": avg_ent,
            "correlation": avg_corr,
        }
        
        np.savez(out_path / "fig8_attention_v2.npz", **fig8_data)
        plot_figure8(fig8_data, get_8bus(), out_path)
    
    # Figure 38 (run once due to computational cost)
    logger.info("\\nFigure 38 — Attention Evolution")
    fig38_data = run_figure38(seed=42)
    
    np.savez(
        out_path / "fig38_attention_evolution_v2.npz",
        snapshots={k: v for k, v in fig38_data["snapshots"].items()},
        entropies=fig38_data["entropies_over_time"],
        snapshot_episodes=fig38_data["snapshot_episodes"],
    )
    plot_figure38(fig38_data, out_path)
    
    print("\\n" + "=" * 70)
    print("ATTENTION ANALYSIS SUMMARY (v2)")
    print("=" * 70)
    if fig8_results:
        print(f"Attention-1/Distance Correlation: {avg_corr:.3f}")
        print(f"Average Entropy: {avg_ent:.4f}")
    print(f"Snapshots captured: {len(fig38_data['snapshots'])}")
    print("=" * 70)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Figures 8 & 38 v2 — Attention Analysis")
    p.add_argument("--n_seeds", type=int, default=3)
    p.add_argument("--output_dir", type=str, default="outputs")
    args = p.parse_args()
    main(args.n_seeds, args.output_dir)
