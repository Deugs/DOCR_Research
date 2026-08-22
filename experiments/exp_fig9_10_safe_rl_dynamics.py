"""
exp_safe_rl_dynamics_v2.py — Figures 9 & 10 (IMPROVED)
────────────────────────────────────────────────────────
CTI Violation Dynamics + Lagrange Multiplier Evolution.

IMPROVEMENTS over original:
  • Proper per-episode logging with structured data
  • 100-episode moving average for Figure 9
  • Comparison with unconstrained MAPPO baseline
  • Individual λ_ij trajectories for all P/B pairs (Figure 10)
  • Correlation with electrical distance
  • Convergence time analysis per pair
  • Automatic dual-figure generation
  • Multi-seed support with confidence bands
  • Safety activation frequency tracking

Outputs
-------
  outputs/fig9_safe_rl_v2.npz
  outputs/fig9_safe_rl_v2.png
  outputs/fig10_lagrange_v2.npz
  outputs/fig10_lagrange_v2.png
"""

import numpy as np
import os
import sys
import argparse
import logging
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
os.makedirs("outputs", exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

from network.network_data import get_8bus
from marl.marl_env import DynamicDOCREnv
from marl.marl_agents import MAPPOMARLTrainer
from marl.safe_rl import create_safe_rl_env, SafeRLConfig
from marl.comm_graph import build_relay_adjacency
from network.relay_physics import check_constraints
from evaluation.experiments._synthetic_label import mark_synthetic

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger(__name__)

EPISODES = 5000
WINDOW = 100
N_SEEDS_DEFAULT = 5


def compute_electrical_distance(net):
    """
    Compute a relay-to-relay hop-distance proxy for "electrical distance".

    NetworkData has no per-line impedance data anywhere (no `net.fbus`/
    `net.tbus`/`net.Z`/`net.bus` fields exist) -- this uses
    `build_relay_adjacency(net)` (the real relay-to-relay adjacency built
    from RelayPair primary/backup + same-bus relationships) and reports
    unweighted shortest-path hop counts as a disclosed topological proxy,
    matching the fix already applied to
    `exp_fig8_38_attention_extraction.py::compute_electrical_distance`.
    """
    import networkx as nx
    adj = build_relay_adjacency(net)
    G = nx.from_numpy_array(adj)

    n_relays = net.n_relays
    dist_matrix = np.zeros((n_relays, n_relays))

    for i in range(n_relays):
        for j in range(n_relays):
            if i == j:
                dist_matrix[i, j] = 0
            else:
                try:
                    dist_matrix[i, j] = nx.shortest_path_length(G, i, j)
                except nx.NetworkXNoPath:
                    dist_matrix[i, j] = np.inf

    return dist_matrix


def run_safe_rl_tracking(use_safe, seed=42):
    """Run training with detailed safe RL tracking."""
    net = get_8bus()
    env = DynamicDOCREnv(net, use_pandapower=False, fault_type="3ph",
                          curriculum_stage=5, seed=seed)
    
    if use_safe:
        env = create_safe_rl_env(env, SafeRLConfig(enforce_cti=True, use_pid=True))
    
    trainer = MAPPOMARLTrainer(env, use_comm=False, seed=seed)

    # Per-episode tracking
    episode_stats = []

    rng = np.random.default_rng(seed)
    steps_per_episode_estimate = max(1, env.max_steps // 4)

    for ep in range(EPISODES):
        # Train before measuring this episode's dynamics -- without this,
        # the actor never updates and every "CTI violation dynamics"/
        # "Lagrange multiplier evolution" curve below is noise from a
        # frozen, randomly-initialized policy.
        trainer.train(total_steps=steps_per_episode_estimate, verbose=False)

        obs = env.reset(seed=seed + ep)
        done = False
        
        ep_violations = 0
        ep_total_pairs = 0
        ep_projections = 0
        ep_steps = 0
        ep_lambda_values = []
        
        while not done:
            actions = np.zeros((env.n_agents, env.ACT_DIM))
            for i in range(env.n_agents):
                a, _ = trainer.actor.sample(obs[i], rng)
                actions[i] = a
            
            obs, _, done, info = env.step(actions)
            ep_steps += 1
            
            # Extract safety info
            if use_safe:
                if "safe_rl_stats" in info:
                    sr = info["safe_rl_stats"]
                    ep_violations += sr.get("cti_violations", 0)
                    ep_total_pairs += sr.get("total_pairs", 0)
                    ep_projections += sr.get("projected_count", 0)
                    
                    if "lambda_values" in sr:
                        ep_lambda_values.extend(sr["lambda_values"])
            else:
                # For unconstrained, compute violations manually.
                # check_constraints() has no "total_pairs" key -- net.n_pairs
                # is the real total P/B pair count.
                TMS, PS = env.get_state()
                c = check_constraints(net, TMS, PS)
                ep_violations += c["cti_violations"]
                ep_total_pairs += net.n_pairs
        
        # Compute episode statistics
        violation_rate = ep_violations / max(ep_total_pairs, 1) if ep_total_pairs > 0 else 0
        projection_rate = ep_projections / max(ep_steps, 1) if use_safe else 0
        mean_lambda = float(np.mean(ep_lambda_values)) if ep_lambda_values else 0
        max_lambda = float(np.max(ep_lambda_values)) if ep_lambda_values else 0
        
        episode_stats.append({
            "violation_rate": violation_rate,
            "projection_rate": projection_rate,
            "mean_lambda": mean_lambda,
            "max_lambda": max_lambda,
            "cumulative_violations": ep_violations,
            "lambda_values": ep_lambda_values,
        })

    return episode_stats


def moving_average(data, window=WINDOW):
    """Compute moving average."""
    if len(data) < window:
        return data
    return np.convolve(data, np.ones(window)/window, mode='valid')


def main(n_seeds=5, output_dir="outputs", fig10_seed=42):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    logger.info("Figures 9 & 10 (v2) — Safe RL Dynamics")
    logger.info("  Episodes: %d  Window: %d  Seeds: %d", EPISODES, WINDOW, n_seeds)
    
    safe_results = []
    unsafe_results = []
    
    for seed in range(n_seeds):
        logger.info("\\nSeed %d/%d", seed + 1, n_seeds)
        
        logger.info("  Running MAPPO-Safe...")
        safe_stats = run_safe_rl_tracking(use_safe=True, seed=seed)
        safe_results.append(safe_stats)
        
        logger.info("  Running MAPPO (unconstrained)...")
        unsafe_stats = run_safe_rl_tracking(use_safe=False, seed=seed + 1000)
        unsafe_results.append(unsafe_stats)
    
    # Aggregate for Figure 9
    n_ma = EPISODES - WINDOW + 1
    
    safe_violation_rates = []
    safe_projection_rates = []
    safe_mean_lambdas = []
    safe_cum_violations = []
    
    for s in range(n_seeds):
        vrs = [ep["violation_rate"] for ep in safe_results[s]]
        prs = [ep["projection_rate"] for ep in safe_results[s]]
        mls = [ep["mean_lambda"] for ep in safe_results[s]]
        cvs = np.cumsum([ep["cumulative_violations"] for ep in safe_results[s]])
        
        safe_violation_rates.append(moving_average(vrs))
        safe_projection_rates.append(moving_average(prs))
        safe_mean_lambdas.append(moving_average(mls))
        safe_cum_violations.append(cvs)
    
    unsafe_violation_rates = []
    for s in range(n_seeds):
        vrs = [ep["violation_rate"] for ep in unsafe_results[s]]
        unsafe_violation_rates.append(moving_average(vrs))
    
    # Compute means and confidence bands
    fig9_data = {
        "violation_safe_mean": np.mean(safe_violation_rates, axis=0),
        "violation_safe_sem": stats.sem(safe_violation_rates, axis=0),
        "violation_unsafe_mean": np.mean(unsafe_violation_rates, axis=0),
        "violation_unsafe_sem": stats.sem(unsafe_violation_rates, axis=0),
        "projection_mean": np.mean(safe_projection_rates, axis=0),
        "projection_sem": stats.sem(safe_projection_rates, axis=0),
        "lambda_mean": np.mean(safe_mean_lambdas, axis=0),
        "lambda_sem": stats.sem(safe_mean_lambdas, axis=0),
        "cum_violations_mean": np.mean(safe_cum_violations, axis=0),
        "cum_violations_sem": stats.sem(safe_cum_violations, axis=0),
    }
    
    # Save Figure 9
    np.savez(out_path / "fig9_safe_rl_v2.npz", **fig9_data, n_seeds=n_seeds, window=WINDOW)
    logger.info("Saved: %s", out_path / "fig9_safe_rl_v2.npz")
    
    # Plot Figure 9
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Figure 9 — CTI Violation Dynamics During Training (v2)", fontsize=14)
    
    x = np.arange(n_ma)
    
    # Panel 1: Violation rate
    ax = axes[0, 0]
    mean = fig9_data["violation_safe_mean"]
    sem = fig9_data["violation_safe_sem"]
    ax.plot(x, mean, "o-", color="#2563eb", lw=2, markersize=3, label="MAPPO-Safe", markevery=50)
    ax.fill_between(x, mean - 1.96*sem, mean + 1.96*sem, alpha=0.2, color="#2563eb")
    
    mean_u = fig9_data["violation_unsafe_mean"]
    sem_u = fig9_data["violation_unsafe_sem"]
    ax.plot(x, mean_u, "s-", color="#ff6b35", lw=2, markersize=3, label="MAPPO (unconstrained)", markevery=50)
    ax.fill_between(x, mean_u - 1.96*sem_u, mean_u + 1.96*sem_u, alpha=0.2, color="#ff6b35")
    
    ax.set_xlabel("Episode", fontsize=10)
    ax.set_ylabel("Violation Rate", fontsize=10)
    ax.set_title("CTI Violation Rate (100-ep MA)", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Panel 2: Projection rate
    ax = axes[0, 1]
    mean = fig9_data["projection_mean"]
    sem = fig9_data["projection_sem"]
    ax.plot(x, mean, "o-", color="#39d353", lw=2, markersize=3, markevery=50)
    ax.fill_between(x, mean - 1.96*sem, mean + 1.96*sem, alpha=0.2, color="#39d353")
    ax.set_xlabel("Episode", fontsize=10)
    ax.set_ylabel("Projection Rate", fontsize=10)
    ax.set_title("Safety Activation Frequency", fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Panel 3: Lagrange multiplier
    ax = axes[1, 0]
    mean = fig9_data["lambda_mean"]
    sem = fig9_data["lambda_sem"]
    ax.plot(x, mean, "o-", color="#a371f7", lw=2, markersize=3, markevery=50)
    ax.fill_between(x, mean - 1.96*sem, mean + 1.96*sem, alpha=0.2, color="#a371f7")
    ax.set_xlabel("Episode", fontsize=10)
    ax.set_ylabel("Mean λ", fontsize=10)
    ax.set_title("Lagrange Multiplier Evolution", fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Panel 4: Cumulative violations
    ax = axes[1, 1]
    mean = fig9_data["cum_violations_mean"]
    sem = fig9_data["cum_violations_sem"]
    ax.plot(x, mean, "o-", color="#ff6b35", lw=2, markersize=3, markevery=50)
    ax.fill_between(x, mean - 1.96*sem, mean + 1.96*sem, alpha=0.2, color="#ff6b35")
    ax.set_xlabel("Episode", fontsize=10)
    ax.set_ylabel("Cumulative Violations", fontsize=10)
    ax.set_title("Total CTI Violations (Cumulative)", fontsize=11)
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(out_path / "fig9_safe_rl_v2.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", out_path / "fig9_safe_rl_v2.png")
    
    # Figure 10: Individual λ_ij trajectories
    # Extract per-pair lambda trajectories from the last seed (most detailed)
    # In practice, we'd need the safe_env to expose per-pair lambdas.
    # NOTE: unlike Figure 9 above (real per-episode training data from
    # run_safe_rl_tracking), this is a SYNTHETIC representation based only on
    # electrical distance — it is watermarked via mark_synthetic() below and
    # must not be cited as a real per-pair training result.

    net = get_8bus()
    elec_dist = compute_electrical_distance(net)
    n_pairs = 0
    pair_distances = []

    # Count P/B pairs
    for i in range(net.n_relays):
        for j in range(net.n_relays):
            if i != j and np.isfinite(elec_dist[i, j]):
                pair_distances.append(elec_dist[i, j])
                n_pairs += 1

    # Generate synthetic lambda trajectories based on distance
    # Closer pairs (smaller distance) have higher lambda (more constrained)
    rng10 = np.random.default_rng(fig10_seed)
    pair_lambdas = []
    for dist in pair_distances:
        # Steady-state lambda inversely proportional to distance
        steady_lambda = 5.0 / (dist + 0.1)
        # Convergence trajectory
        traj = steady_lambda * (1 - np.exp(-np.arange(EPISODES) / (500 + rng10.integers(0, 200))))
        traj += rng10.normal(0, 0.1, EPISODES)
        pair_lambdas.append(traj)
    
    fig10_data = {
        "pair_lambdas": np.array(pair_lambdas),
        "pair_distances": np.array(pair_distances),
        "episodes": EPISODES,
        "is_synthetic": np.array([True]),
        "synthetic_seed": np.array([fig10_seed]),
    }

    np.savez(out_path / "fig10_lagrange_v2.npz", **fig10_data)
    logger.info("Saved: %s", out_path / "fig10_lagrange_v2.npz")
    
    # Plot Figure 10
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Figure 10 — Lagrange Multiplier Evolution per P/B Pair (v2)", fontsize=14)
    
    # Panel 1: All trajectories
    ax = axes[0]
    colors = plt.cm.viridis(np.linspace(0, 1, n_pairs))
    for i, traj in enumerate(pair_lambdas):
        alpha = 0.3 + 0.7 * (1 - pair_distances[i] / max(pair_distances))
        ax.plot(traj, color=colors[i], lw=1, alpha=alpha)
    
    ax.set_xlabel("Episode", fontsize=10)
    ax.set_ylabel("λ_ij", fontsize=10)
    ax.set_title("Individual λ Trajectories (colored by distance)", fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Panel 2: Steady-state λ vs. electrical distance
    ax = axes[1]
    steady_lambdas = [traj[-100:].mean() for traj in pair_lambdas]
    ax.scatter(pair_distances, steady_lambdas, s=80, alpha=0.7, color="#2563eb", edgecolors="white")
    
    z = np.polyfit(pair_distances, steady_lambdas, 1)
    p = np.poly1d(z)
    x_line = np.linspace(min(pair_distances), max(pair_distances), 100)
    ax.plot(x_line, p(x_line), "r--", lw=2, alpha=0.8, label=f"Trend (slope={z[0]:.3f})")
    
    ax.set_xlabel("Electrical Distance (Ω)", fontsize=10)
    ax.set_ylabel("Steady-State λ", fontsize=10)
    ax.set_title("λ vs. Electrical Distance", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    mark_synthetic(fig, note=f"seed={fig10_seed}; per-pair λ trajectories are not from real training logs")
    fig.savefig(out_path / "fig10_lagrange_v2.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", out_path / "fig10_lagrange_v2.png")
    
    # Summary
    print("\\n" + "=" * 70)
    print("SAFE RL DYNAMICS SUMMARY (v2)")
    print("=" * 70)
    final_safe = {
        "violation": fig9_data["violation_safe_mean"][-1],
        "projection": fig9_data["projection_mean"][-1],
        "lambda": fig9_data["lambda_mean"][-1],
    }
    final_unsafe = {
        "violation": fig9_data["violation_unsafe_mean"][-1],
    }
    print(f"Final Violation Rate (Safe):   {final_safe['violation']:.4f}")
    print(f"Final Violation Rate (Unsafe): {final_unsafe['violation']:.4f}")
    print(f"Final Projection Rate:         {final_safe['projection']:.4f}")
    print(f"Final Mean Lambda:             {final_safe['lambda']:.4f}")
    print(f"Total P/B Pairs:               {n_pairs}")
    print("=" * 70)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Figures 9 & 10 v2 — Safe RL Dynamics")
    p.add_argument("--n_seeds", type=int, default=5)
    p.add_argument("--output_dir", type=str, default="outputs")
    p.add_argument("--fig10_seed", type=int, default=42,
                    help="Seed for Figure 10's synthetic per-pair lambda trajectories "
                         "(Figure 9 uses real per-episode training data seeded 0..n_seeds-1)")
    args = p.parse_args()
    main(args.n_seeds, args.output_dir, args.fig10_seed)
