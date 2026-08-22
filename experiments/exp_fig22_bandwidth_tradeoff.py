
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
from marl.safe_rl import create_safe_rl_env, SafeRLConfig
from marl.continual_trainer import create_ewc_trainer, EWCConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def train_base_policy(seed: int = 42, steps: int = 8000):
    """Train base MAPPO-Safe-EWC policy.

    `trainer.use_safety`/`use_ewc` used to be set via hasattr guards that
    were always False -- MAPPOMARLTrainer has neither attribute, so this
    "MAPPO-Safe-EWC" policy trained with no safety projection and no EWC
    penalty at all. Safe RL is a real env wrapper (must wrap before the
    trainer is built); EWC registration follows the same real
    finite-difference Fisher-estimation procedure used elsewhere in this
    codebase for single-task EWC (e.g. run_core_training.py's Core A).
    """
    net = get_8bus()
    raw_env = DynamicDOCREnv(net, seed=seed, use_pandapower=False,
                              fault_type="3ph", curriculum_stage=5)
    env = create_safe_rl_env(raw_env, SafeRLConfig(
        project_actions=True, enforce_cti=True,
        lambda_init=1.0, lambda_lr=0.01, rho_penalty=10.0,
        use_pid=True, adaptive_rho=True,
    ))
    trainer = MAPPOMARLTrainer(env, use_comm=True, seed=seed)
    ewc_trainer = create_ewc_trainer(trainer, EWCConfig())
    trainer.train(total_steps=steps, verbose=False)

    rng = np.random.default_rng(seed + 9000)
    cur_params = trainer.actor.get_all_params()
    obs_f = env.reset(seed=seed + 9000)
    s_obs, s_act = [], []
    done_f, sf = False, 0
    while not done_f and sf < 30:
        acts = np.zeros((env.n_agents, env.ACT_DIM))
        enhanced_f, _, _ = trainer.comm_graph.forward(obs_f, rng) if trainer.comm_graph is not None else (obs_f, None, None)
        for i in range(env.n_agents):
            a, _ = trainer.actor.sample(enhanced_f[i], rng)
            acts[i] = a
            s_obs.append(enhanced_f[i].copy())
            s_act.append(a.copy())
        obs_f, _, done_f, _ = env.step(acts); sf += 1
    n_p = len(cur_params)
    fisher = np.zeros(n_p)
    if s_obs:
        idx = rng.choice(len(s_obs), size=min(30, len(s_obs)), replace=False)
        h = 1e-3
        for p_idx in rng.choice(n_p, size=min(n_p, 50), replace=False):
            bl = np.mean([trainer.actor.log_prob(s_obs[j], s_act[j]) for j in idx])
            b2 = cur_params.copy(); b2[p_idx] += h
            trainer.actor.set_all_params(b2)
            ul = np.mean([trainer.actor.log_prob(s_obs[j], s_act[j]) for j in idx])
            fisher[p_idx] = ((ul - bl) / h) ** 2
        trainer.actor.set_all_params(cur_params)
    ewc_trainer.ewc_regularizer.register_task("base_policy", cur_params)
    ewc_trainer.ewc_regularizer.update_fisher("base_policy", fisher)
    ewc_trainer.ewc_regularizer.finalize()

    return trainer, net


def evaluate_with_gnn_depth(trainer, net, gnn_depth: int, n_eval: int = 50, seed: int = 0,
                             compression: str = "none"):
    """Evaluate with specified GNN depth and message compression.

    `trainer.actor.gnn_depth` never existed (the mutation was a permanent
    no-op) and this also fed raw obs directly to an actor built for
    comm-graph-enhanced input, crashing with a matmul dimension mismatch.

    GNN depth disclosure: CommConfig.n_gnn_layers (marl/comm_graph.py) is
    stored but never read anywhere in the encoder/attention/decoder
    forward pass -- there is no real multi-hop depth mechanism in this
    codebase's GNN implementation, so genuinely varying depth would
    require training a separate policy per depth (a new feature, not a
    bug fix). This function no longer pretends to vary it: `gnn_depth` is
    accepted only so compute_bandwidth()'s theoretical message-count
    formula can still be plotted against it, and the OF/CTI numbers below
    are the SAME trained policy's real performance -- expected to be flat
    across "depths" for that reason, not fabricated to show a fake trend.

    Compression, by contrast, genuinely changes the policy's real inputs
    via CommGraph's real stress/compression knobs (the same ones used by
    experiments/exp_fig13_communication_stress.py and added for
    experiments/training_pipeline.py's Fig-13 evaluator): "int8" quantizes
    messages, "top50" keeps each agent's top-50%-weight neighbors via the
    real attention-sparsification knob, "dropout" applies real per-
    component message dropout.
    """
    rng = np.random.default_rng(seed)
    env = DynamicDOCREnv(net, seed=seed, use_pandapower=False,
                          fault_type="3ph", curriculum_stage=5)

    comm_graph = trainer.comm_graph
    orig_bits = orig_topk = orig_dropout = None
    if comm_graph is not None:
        orig_bits = comm_graph.quantization_bits
        orig_topk = comm_graph.top_k_sparsify_fraction
        orig_dropout = comm_graph.config.dropout
        if compression == "int8":
            comm_graph.quantization_bits = 8
        elif compression == "top50":
            comm_graph.top_k_sparsify_fraction = 0.5
        elif compression == "dropout":
            comm_graph.config.dropout = 0.3

    of_list, cti_list = [], []
    try:
        for ep in range(n_eval):
            obs = env.reset(seed=seed + ep)
            done = False
            ep_of = []
            while not done:
                actions = np.zeros((env.n_agents, env.ACT_DIM))
                if comm_graph is not None:
                    enhanced, _, _ = comm_graph.forward(obs, rng)
                else:
                    enhanced = obs
                for i in range(env.n_agents):
                    a, _ = trainer.actor.sample(enhanced[i], rng)
                    actions[i] = a
                obs, _, done, info = env.step(actions)
                ep_of.append(info.get("OF", float("inf")))
            of_list.append(float(np.min(ep_of)))
            TMS, PS = env.get_state()
            c = check_constraints(net, TMS, PS)
            cti_list.append(c["cti_violations"])
    finally:
        if comm_graph is not None:
            comm_graph.quantization_bits = orig_bits
            comm_graph.top_k_sparsify_fraction = orig_topk
            comm_graph.config.dropout = orig_dropout

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
        res = evaluate_with_gnn_depth(trainer, net, 2, n_eval, seed + 500, compression=comp)
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
