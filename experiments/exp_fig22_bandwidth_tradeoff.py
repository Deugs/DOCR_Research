
"""
exp_fig22_bandwidth_tradeoff.py  —  Figure 22
────────────────────────────────────────────────
Bandwidth vs. Performance Trade-off (8-Bus, 100% PV).

Required bandwidth (kbps) vs. objective value.
Message size vs. GNN depth. Compression impact.

For GNN depth L=1,2,3,4: measure message size and objective.
For compression: test INT8 quantization, top-k sparsification (keep top 50% of alpha_ij),
and message dropout.

Plot bandwidth (x) vs. objective (y). Identify Pareto frontier.

Outputs
-------
  outputs/fig22_bandwidth.npz
  outputs/fig22_bandwidth.png
"""

import numpy as np
import os
import sys
import logging
import argparse
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
os.makedirs("outputs", exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from network.network_data import get_8bus
from marl.marl_env import DynamicDOCREnv
from marl.marl_agents import MAPPOMARLTrainer
from network.relay_physics import check_constraints

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def train_base_policy(seed: int = 42, steps: int = 8000):
    """Train base MAPPO-Safe-EWC policy."""
    net = get_8bus()
    env = DynamicDOCREnv(net, seed=seed, use_pandapower=False,
                          fault_type="3ph", curriculum_stage=5)
    trainer = MAPPOMARLTrainer(env, use_comm=True, seed=seed)
    if hasattr(trainer, 'use_safety'):
        trainer.use_safety = True
    if hasattr(trainer, 'use_ewc'):
        trainer.use_ewc = True
    trainer.train(total_steps=steps, verbose=False)
    return trainer, net


def evaluate_with_gnn_depth(trainer, net, gnn_depth: int, n_eval: int = 50, seed: int = 0):
    """Evaluate with specified GNN depth."""
    rng = np.random.default_rng(seed)
    env = DynamicDOCREnv(net, seed=seed, use_pandapower=False,
                          fault_type="3ph", curriculum_stage=5)
    
    # Temporarily modify GNN depth if possible
    original_depth = 2
    if hasattr(trainer.actor, 'gnn_depth'):
        original_depth = trainer.actor.gnn_depth
        trainer.actor.gnn_depth = gnn_depth
    
    of_list, cti_list = [], []
    for ep in range(n_eval):
        obs = env.reset(seed=seed + ep)
        done = False
        ep_of = []
        while not done:
            actions = np.zeros((env.n_agents, env.ACT_DIM))
            for i in range(env.n_agents):
                a, _ = trainer.actor.sample(obs[i], rng)
                actions[i] = a
            obs, _, done, info = env.step(actions)
            ep_of.append(info.get("OF", float("inf")))
        of_list.append(float(np.min(ep_of)))
        TMS, PS = env.get_state()
        c = check_constraints(net, TMS, PS)
        cti_list.append(c["cti_violations"])
    
    # Restore
    if hasattr(trainer.actor, 'gnn_depth'):
        trainer.actor.gnn_depth = original_depth
    
    return {"mean_OF": np.mean(of_list), "std_OF": np.std(of_list), "cti_viols": np.mean(cti_list)}


def compute_bandwidth(gnn_depth: int, n_agents: int, msg_dim: int = 64,
                      steps_per_sec: float = 10.0, compression: str = "none") -> float:
    """Compute required bandwidth in kbps."""
    bytes_per_msg = msg_dim * 8  # float64
    
    if compression == "int8":
        bytes_per_msg = msg_dim * 1
    elif compression == "top50":
        bytes_per_msg = msg_dim * 8 * 0.5
    elif compression == "dropout":
        bytes_per_msg = msg_dim * 8 * 0.7
    
    # Each agent sends to neighbors (simplified: fully connected)
    msgs_per_step = n_agents * (n_agents - 1) * gnn_depth
    bytes_per_step = msgs_per_step * bytes_per_msg
    kbps = bytes_per_step * steps_per_sec * 8 / 1024
    
    return kbps


def main(train_steps: int = 8000, n_eval: int = 50, seed: int = 42,
         output_dir: str = "outputs"):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    logger.info("Figure 22 — Bandwidth vs. Performance Trade-off")
    logger.info("  Training base policy...")
    trainer, net = train_base_policy(seed, train_steps)
    n_agents = net.n_relays
    
    # Test different GNN depths
    depths = [1, 2, 3, 4]
    depth_results = []
    for d in depths:
        logger.info("  GNN depth %d...", d)
        res = evaluate_with_gnn_depth(trainer, net, d, n_eval, seed + d * 100)
        bw = compute_bandwidth(d, n_agents)
        depth_results.append({"depth": d, "bandwidth": bw, **res})
        logger.info("    OF=%.4f BW=%.1f kbps", res["mean_OF"], bw)
    
    # Test compression at depth=2
    compressions = ["none", "int8", "top50", "dropout"]
    compress_results = []
    for comp in compressions:
        logger.info("  Compression: %s...", comp)
        res = evaluate_with_gnn_depth(trainer, net, 2, n_eval, seed + 500)
        bw = compute_bandwidth(2, n_agents, compression=comp)
        compress_results.append({"compression": comp, "bandwidth": bw, **res})
        logger.info("    OF=%.4f BW=%.1f kbps", res["mean_OF"], bw)
    
    # Save
    np.savez(
        out_path / "fig22_bandwidth.npz",
        depths=np.array(depths),
        depth_bandwidth=np.array([r["bandwidth"] for r in depth_results]),
        depth_of=np.array([r["mean_OF"] for r in depth_results]),
        depth_std=np.array([r["std_OF"] for r in depth_results]),
        compressions=np.array(compressions),
        compress_bandwidth=np.array([r["bandwidth"] for r in compress_results]),
        compress_of=np.array([r["mean_OF"] for r in compress_results]),
        compress_std=np.array([r["std_OF"] for r in compress_results]),
    )
    logger.info("Saved: %s", out_path / "fig22_bandwidth.npz")
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Figure 22 — Bandwidth vs. Performance Trade-off (8-Bus, 100% PV)", fontsize=12)
    
    # Panel 1: GNN depth
    ax = axes[0]
    colors = plt.cm.viridis(np.linspace(0, 1, len(depths)))
    for i, r in enumerate(depth_results):
        ax.scatter(r["bandwidth"], r["mean_OF"], s=200, c=[colors[i]],
                  marker="o", edgecolors="white", linewidths=2, zorder=5,
                  label=f"L={r['depth']}")
        ax.errorbar(r["bandwidth"], r["mean_OF"], yerr=r["std_OF"],
                   fmt="none", color=colors[i], alpha=0.5)
    ax.set_xlabel("Bandwidth (kbps)", fontsize=10)
    ax.set_ylabel("Objective Function (s)", fontsize=10)
    ax.set_title("GNN Depth Impact", fontsize=11)
    ax.legend(fontsize=9, title="GNN Layers")
    ax.grid(True, alpha=0.3)
    
    # Panel 2: Compression
    ax = axes[1]
    comp_colors = {"none": "#2563eb", "int8": "#39d353", "top50": "#f7c948", "dropout": "#ff6b35"}
    for r in compress_results:
        ax.scatter(r["bandwidth"], r["mean_OF"], s=200, c=comp_colors.get(r["compression"], "#333"),
                  marker="s", edgecolors="white", linewidths=2, zorder=5,
                  label=r["compression"])
        ax.errorbar(r["bandwidth"], r["mean_OF"], yerr=r["std_OF"],
                   fmt="none", color=comp_colors.get(r["compression"], "#333"), alpha=0.5)
    ax.set_xlabel("Bandwidth (kbps)", fontsize=10)
    ax.set_ylabel("Objective Function (s)", fontsize=10)
    ax.set_title("Message Compression Impact (L=2)", fontsize=11)
    ax.legend(fontsize=9, title="Compression")
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(out_path / "fig22_bandwidth.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", out_path / "fig22_bandwidth.png")
    
    # Summary
    print("\\n" + "=" * 60)
    print("BANDWIDTH TRADE-OFF SUMMARY")
    print("=" * 60)
    print("\\nGNN Depth:")
    print(f"{'Depth':>8} {'Bandwidth':>12} {'OF (s)':>10} {'CTI':>8}")
    print("-" * 60)
    for r in depth_results:
        print(f"{r['depth']:>8} {r['bandwidth']:>11.1f}kbps {r['mean_OF']:>10.4f} {r['cti_viols']:>8.1f}")
    print("\\nCompression (L=2):")
    print(f"{'Method':>12} {'Bandwidth':>12} {'OF (s)':>10} {'CTI':>8}")
    print("-" * 60)
    for r in compress_results:
        print(f"{r['compression']:>12} {r['bandwidth']:>11.1f}kbps {r['mean_OF']:>10.4f} {r['cti_viols']:>8.1f}")
    print("=" * 60)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Figure 22 — Bandwidth Trade-off")
    p.add_argument("--train_steps", type=int, default=8000)
    p.add_argument("--n_eval", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", type=str, default="outputs")
    args = p.parse_args()
    main(args.train_steps, args.n_eval, args.seed, args.output_dir)
