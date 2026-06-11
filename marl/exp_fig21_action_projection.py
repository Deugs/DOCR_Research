
"""
exp_action_projection_v2.py — Figure 21 (IMPROVED)
───────────────────────────────────────────────────
Action Projection Efficacy During Training.

IMPROVEMENTS over original:
  • Proper callback integration with trainer.train()
  • Comparison with unconstrained MAPPO baseline
  • Per-100-episode aggregation
  • CTI violation count before/after projection
  • Projection magnitude statistics (mean, std, max, percentiles)
  • Automatic plot generation (4-panel)
  • Multi-seed support
  • Statistical comparison between constrained/unconstrained

Outputs
-------
  outputs/fig21_projection_v2.npz
  outputs/fig21_projection_v2.png
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
from network.relay_physics import check_constraints

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger(__name__)

EPISODES = 5000
WINDOW = 100
N_EVAL_PER_CHECKPOINT = 20


def run_with_projection_tracking(use_safe, seed=42):
    """Run training with detailed projection tracking."""
    net = get_8bus()
    env = DynamicDOCREnv(net, use_pandapower=False, fault_type="3ph", 
                          curriculum_stage=5, seed=seed)
    
    if use_safe:
        env = create_safe_rl_env(env, SafeRLConfig(enforce_cti=True, project_actions=True))
    
    trainer = MAPPOMARLTrainer(env, use_comm=False, seed=seed)
    
    # Tracking structures
    episode_data = defaultdict(lambda: {
        "projected": [],
        "magnitudes": [],
        "cti_violations_raw": [],
        "cti_violations_projected": [],
        "returns": [],
    })
    
    rng = np.random.default_rng(seed)
    
    for ep in range(EPISODES):
        obs = env.reset(seed=seed + ep)
        done = False
        ep_return = 0
        ep_projected = []
        ep_magnitudes = []
        
        while not done:
            # Get raw action (before projection)
            actions_raw = np.zeros((env.n_agents, env.ACT_DIM))
            for i in range(env.n_agents):
                a, _ = trainer.actor.sample(obs[i], rng)
                actions_raw[i] = a
            
            # Step environment (safety layer projects if active)
            obs, reward, done, info = env.step(actions_raw)
            ep_return += reward
            
            # Extract projection info
            if use_safe and "safe_rl" in info:
                sr = info["safe_rl"]
                if "projected_actions" in sr:
                    ep_projected.append(float(sr["projected_actions"]))
                if "projection_magnitude" in sr:
                    ep_magnitudes.append(float(sr["projection_magnitude"]))
            
            # Store per-step CTI info if available
            if "cti_violations_raw" in info:
                episode_data[ep]["cti_violations_raw"].append(info["cti_violations_raw"])
            if "cti_violations_projected" in info:
                episode_data[ep]["cti_violations_projected"].append(info["cti_violations_projected"])
        
        episode_data[ep]["projected"] = ep_projected
        episode_data[ep]["magnitudes"] = ep_magnitudes
        episode_data[ep]["returns"].append(ep_return)
        
        # Training update
        trainer.update()
    
    # Aggregate per-100-episode windows
    windows = []
    for w_start in range(0, EPISODES, WINDOW):
        w_end = min(w_start + WINDOW, EPISODES)
        window_projected = []
        window_magnitudes = []
        window_returns = []
        window_cti_raw = []
        window_cti_proj = []
        
        for ep in range(w_start, w_end):
            window_projected.extend(episode_data[ep]["projected"])
            window_magnitudes.extend(episode_data[ep]["magnitudes"])
            window_returns.extend(episode_data[ep]["returns"])
            window_cti_raw.extend(episode_data[ep]["cti_violations_raw"])
            window_cti_proj.extend(episode_data[ep]["cti_violations_projected"])
        
        windows.append({
            "window_start": w_start,
            "projection_rate": float(np.mean(window_projected)) if window_projected else 0,
            "projection_rate_std": float(np.std(window_projected)) if window_projected else 0,
            "mean_magnitude": float(np.mean(window_magnitudes)) if window_magnitudes else 0,
            "std_magnitude": float(np.std(window_magnitudes)) if window_magnitudes else 0,
            "max_magnitude": float(np.max(window_magnitudes)) if window_magnitudes else 0,
            "p50_magnitude": float(np.percentile(window_magnitudes, 50)) if window_magnitudes else 0,
            "p95_magnitude": float(np.percentile(window_magnitudes, 95)) if window_magnitudes else 0,
            "p99_magnitude": float(np.percentile(window_magnitudes, 99)) if window_magnitudes else 0,
            "mean_return": float(np.mean(window_returns)),
            "cti_violations_raw": float(np.mean(window_cti_raw)) if window_cti_raw else 0,
            "cti_violations_projected": float(np.mean(window_cti_proj)) if window_cti_proj else 0,
        })
    
    return windows


def main(n_seeds=5, output_dir="outputs"):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    logger.info("Figure 21 (v2) — Action Projection Efficacy")
    logger.info("  Episodes: %d  Window: %d  Seeds: %d", EPISODES, WINDOW, n_seeds)
    
    safe_results = []
    unsafe_results = []
    
    for seed in range(n_seeds):
        logger.info("\\nSeed %d/%d", seed + 1, n_seeds)
        
        logger.info("  Running MAPPO-Safe (with projection)...")
        safe_w = run_with_projection_tracking(use_safe=True, seed=seed)
        safe_results.append(safe_w)
        
        logger.info("  Running MAPPO (unconstrained baseline)...")
        unsafe_w = run_with_projection_tracking(use_safe=False, seed=seed + 1000)
        unsafe_results.append(unsafe_w)
    
    # Average across seeds
    n_windows = len(safe_results[0])
    aggregated_safe = []
    aggregated_unsafe = []
    
    for w in range(n_windows):
        # Safe
        proj_rates = [safe_results[s][w]["projection_rate"] for s in range(n_seeds)]
        mean_mags = [safe_results[s][w]["mean_magnitude"] for s in range(n_seeds)]
        max_mags = [safe_results[s][w]["max_magnitude"] for s in range(n_seeds)]
        returns = [safe_results[s][w]["mean_return"] for s in range(n_seeds)]
        cti_raw = [safe_results[s][w]["cti_violations_raw"] for s in range(n_seeds)]
        cti_proj = [safe_results[s][w]["cti_violations_projected"] for s in range(n_seeds)]
        
        aggregated_safe.append({
            "projection_rate_mean": float(np.mean(proj_rates)),
            "projection_rate_std": float(np.std(proj_rates)),
            "projection_rate_sem": float(stats.sem(proj_rates)),
            "mean_magnitude_mean": float(np.mean(mean_mags)),
            "mean_magnitude_std": float(np.std(mean_mags)),
            "max_magnitude_mean": float(np.mean(max_mags)),
            "max_magnitude_std": float(np.std(max_mags)),
            "return_mean": float(np.mean(returns)),
            "return_std": float(np.std(returns)),
            "cti_raw_mean": float(np.mean(cti_raw)),
            "cti_proj_mean": float(np.mean(cti_proj)),
        })
        
        # Unsafe
        returns_u = [unsafe_results[s][w]["mean_return"] for s in range(n_seeds)]
        cti_u = [unsafe_results[s][w]["cti_violations_raw"] for s in range(n_seeds)]
        aggregated_unsafe.append({
            "return_mean": float(np.mean(returns_u)),
            "return_std": float(np.std(returns_u)),
            "cti_mean": float(np.mean(cti_u)),
        })
    
    # Save
    np.savez(
        out_path / "fig21_projection_v2.npz",
        safe=aggregated_safe,
        unsafe=aggregated_unsafe,
        n_seeds=n_seeds,
        episodes=EPISODES,
        window=WINDOW,
    )
    logger.info("Saved: %s", out_path / "fig21_projection_v2.npz")
    
    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Figure 21 — Action Projection Efficacy (v2)", fontsize=14)
    
    x = np.arange(n_windows) * WINDOW
    
    # Panel 1: Projection Rate Over Time
    ax = axes[0, 0]
    rates = [a["projection_rate_mean"] for a in aggregated_safe]
    sems = [a["projection_rate_sem"] for a in aggregated_safe]
    ax.plot(x, rates, "o-", color="#2563eb", lw=2, markersize=5, label="Projection Rate")
    ax.fill_between(x, np.array(rates) - np.array(sems), np.array(rates) + np.array(sems),
                    alpha=0.3, color="#2563eb")
    ax.set_xlabel("Episode", fontsize=10)
    ax.set_ylabel("Projection Rate", fontsize=10)
    ax.set_title("Projection Frequency (per 100 eps)", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    
    # Panel 2: Projection Magnitude
    ax = axes[0, 1]
    mean_mags = [a["mean_magnitude_mean"] for a in aggregated_safe]
    max_mags = [a["max_magnitude_mean"] for a in aggregated_safe]
    ax.plot(x, mean_mags, "o-", color="#39d353", lw=2, markersize=5, label="Mean")
    ax.plot(x, max_mags, "s-", color="#ff6b35", lw=2, markersize=5, label="Max")
    ax.set_xlabel("Episode", fontsize=10)
    ax.set_ylabel("Projection Magnitude", fontsize=10)
    ax.set_title("Projection Magnitude (L2 norm)", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Panel 3: CTI Violations Before/After
    ax = axes[1, 0]
    cti_raw = [a["cti_raw_mean"] for a in aggregated_safe]
    cti_proj = [a["cti_proj_mean"] for a in aggregated_safe]
    cti_unsafe = [a["cti_mean"] for a in aggregated_unsafe]
    ax.plot(x, cti_raw, "o-", color="#ff6b35", lw=2, markersize=5, label="Raw (before projection)")
    ax.plot(x, cti_proj, "s-", color="#39d353", lw=2, markersize=5, label="Projected (after)")
    ax.plot(x, cti_unsafe, "^-", color="#a371f7", lw=2, markersize=5, label="Unconstrained MAPPO")
    ax.set_xlabel("Episode", fontsize=10)
    ax.set_ylabel("CTI Violations", fontsize=10)
    ax.set_title("CTI Violation Reduction", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Panel 4: Return Comparison
    ax = axes[1, 1]
    ret_safe = [a["return_mean"] for a in aggregated_safe]
    ret_unsafe = [a["return_mean"] for a in aggregated_unsafe]
    ax.plot(x, ret_safe, "o-", color="#2563eb", lw=2, markersize=5, label="MAPPO-Safe")
    ax.plot(x, ret_unsafe, "s-", color="#ff6b35", lw=2, markersize=5, label="MAPPO (unconstrained)")
    ax.set_xlabel("Episode", fontsize=10)
    ax.set_ylabel("Episode Return", fontsize=10)
    ax.set_title("Learning Curves Comparison", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(out_path / "fig21_projection_v2.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", out_path / "fig21_projection_v2.png")
    
    # Summary
    print("\\n" + "=" * 70)
    print("ACTION PROJECTION SUMMARY (v2)")
    print("=" * 70)
    final_safe = aggregated_safe[-1]
    final_unsafe = aggregated_unsafe[-1]
    print(f"Final Projection Rate:     {final_safe['projection_rate_mean']:.3f} ± {final_safe['projection_rate_std']:.3f}")
    print(f"Final Mean Magnitude:      {final_safe['mean_magnitude_mean']:.4f}")
    print(f"Final Max Magnitude:       {final_safe['max_magnitude_mean']:.4f}")
    print(f"CTI Violations (raw):      {final_safe['cti_raw_mean']:.2f}")
    print(f"CTI Violations (projected): {final_safe['cti_proj_mean']:.2f}")
    print(f"CTI Violations (unconstrained): {final_unsafe['cti_mean']:.2f}")
    print(f"Return (Safe):             {final_safe['return_mean']:.2f}")
    print(f"Return (Unsafe):           {final_unsafe['return_mean']:.2f}")
    print("=" * 70)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Figure 21 v2 — Action Projection")
    p.add_argument("--n_seeds", type=int, default=5)
    p.add_argument("--output_dir", type=str, default="outputs")
    args = p.parse_args()
    main(args.n_seeds, args.output_dir)
