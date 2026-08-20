"""
exp_fig09_cti_violation_dynamics.py  —  Figure 9
───────────────────────────────────────────────
CTI Violation Dynamics During Training (8-Bus, 100% PV).

Per-episode CTI violation rate, Lagrange multiplier norm,
safety activation frequency, cumulative violations.

Compares MAPPO-Safe (constrained) vs. unconstrained MAPPO baseline.

Outputs
-------
  outputs/fig09_cti_dynamics.npz
  outputs/fig09_cti_dynamics.png
"""

import numpy as np
import os
import sys
import logging
import argparse
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
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


def run_training_with_logging(env, trainer, n_episodes: int = 2000, seed: int = 42):
    """Run training and log CTI dynamics."""
    rng = np.random.default_rng(seed)
    log = {
        "episode": [], "cti_violation_rate": [], "lagrange_norm": [],
        "safety_activations": [], "cumulative_violations": [],
        "mean_reward": [],
    }
    
    cum_viol = 0
    total_pairs = env.net.n_pairs
    
    for ep in range(n_episodes):
        obs = env.reset(seed=seed + ep)
        done = False
        ep_reward = 0.0
        ep_violations = 0
        ep_safety_acts = 0
        
        while not done:
            actions_raw = np.zeros((env.n_agents, env.ACT_DIM))
            actions_safe = np.zeros((env.n_agents, env.ACT_DIM))
            
            for i in range(env.n_agents):
                a_raw, _ = trainer.actor.sample(obs[i], rng)
                actions_raw[i] = a_raw
                
                # Safety projection if available
                if hasattr(trainer, 'safety_layer') and trainer.safety_layer is not None:
                    a_safe = trainer.safety_layer.project(a_raw, obs[i])
                    if not np.allclose(a_raw, a_safe):
                        ep_safety_acts += 1
                    actions_safe[i] = a_safe
                else:
                    actions_safe[i] = a_raw
            
            obs, reward, done, info = env.step(actions_safe)
            # FIX: reward may be a per-agent array; collapse to scalar
            ep_reward += float(np.mean(reward))
        
        # Check CTI violations
        TMS, PS = env.get_state()
        c = check_constraints(env.net, TMS, PS)
        viol_count = c["cti_violations"]
        viol_rate = viol_count / max(total_pairs, 1)
        cum_viol += viol_count
        
        # Lagrange multiplier norm
        lag_norm = 0.0
        if hasattr(trainer, 'lagrange_multipliers'):
            lag_norm = float(np.linalg.norm(trainer.lagrange_multipliers))
        elif hasattr(trainer, 'safety_layer') and hasattr(trainer.safety_layer, 'lambdas'):
            lag_norm = float(np.linalg.norm(trainer.safety_layer.lambdas))
        
        log["episode"].append(ep)
        log["cti_violation_rate"].append(viol_rate * 100)
        log["lagrange_norm"].append(lag_norm)
        log["safety_activations"].append(ep_safety_acts)
        log["cumulative_violations"].append(cum_viol)
        log["mean_reward"].append(ep_reward)
        
        if ep % 200 == 0:
            logger.info("Ep %d: viol_rate=%.2f%% lag_norm=%.3f cum_viol=%d reward=%.3f",
                       ep, viol_rate*100, lag_norm, cum_viol, ep_reward)
    
    return log


def main(n_episodes: int = 2000, seed: int = 42, output_dir: str = "outputs", system: str = "8bus"):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    logger.info("Figure 9 — CTI Violation Dynamics During Training")
    logger.info("  Episodes: %d", n_episodes)
    logger.info("  System: %s", system)

    if system == "8bus":
        net = get_8bus()
    elif system == "14bus":
        from network.network_data import get_14bus
        net = get_14bus()
    elif system == "30bus":
        from network.network_data import get_30bus
        net = get_30bus()
    elif system == "39bus":
        from network.network_data import get_39bus
        net = get_39bus()
    else:
        net = get_8bus()
    
    # Run MAPPO-Safe
    logger.info("\n[1/2] Training MAPPO-Safe...")
    env_safe = DynamicDOCREnv(net, seed=seed, use_pandapower=False,
                                 fault_type="3ph", curriculum_stage=5)
    trainer_safe = MAPPOMARLTrainer(env_safe, use_comm=False, seed=seed)
    if hasattr(trainer_safe, 'use_safety'):
        trainer_safe.use_safety = True
    log_safe = run_training_with_logging(env_safe, trainer_safe, n_episodes, seed)
    
    # Run unconstrained MAPPO
    logger.info("\n[2/2] Training unconstrained MAPPO...")
    env_unsafe = DynamicDOCREnv(net, seed=seed+1000, use_pandapower=False,
                                 fault_type="3ph", curriculum_stage=5)
    trainer_unsafe = MAPPOMARLTrainer(env_unsafe, use_comm=False, seed=seed+1000)
    if hasattr(trainer_unsafe, 'use_safety'):
        trainer_unsafe.use_safety = False
    log_unsafe = run_training_with_logging(env_unsafe, trainer_unsafe, n_episodes, seed+1000)
    
    # Save
    np.savez(
        out_path / "fig09_cti_dynamics.npz",
        episodes=np.array(log_safe["episode"]),
        safe_cti_rate=np.array(log_safe["cti_violation_rate"]),
        safe_lagrange=np.array(log_safe["lagrange_norm"]),
        safe_safety_acts=np.array(log_safe["safety_activations"]),
        safe_cum_viol=np.array(log_safe["cumulative_violations"]),
        safe_reward=np.array(log_safe["mean_reward"]),
        unsafe_cti_rate=np.array(log_unsafe["cti_violation_rate"]),
        unsafe_cum_viol=np.array(log_unsafe["cumulative_violations"]),
        unsafe_reward=np.array(log_unsafe["mean_reward"]),
    )
    logger.info("Saved: %s", out_path / "fig09_cti_dynamics.npz")
    
    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("Figure 9 — CTI Violation Dynamics During Training (8-Bus, 100% PV)", fontsize=12)
    
    eps = np.array(log_safe["episode"])
    window = 50
    
    def smooth(y):
        if len(y) < window:
            return y
        return np.convolve(y, np.ones(window)/window, mode='valid')
    
    # Panel 1: CTI Violation Rate
    ax = axes[0, 0]
    ax.plot(eps, log_safe["cti_violation_rate"], alpha=0.2, color="#2563eb", lw=0.5)
    ax.plot(eps[window-1:], smooth(log_safe["cti_violation_rate"]), color="#2563eb", lw=2, label="MAPPO-Safe")
    ax.plot(eps, log_unsafe["cti_violation_rate"], alpha=0.2, color="#ff6b35", lw=0.5)
    ax.plot(eps[window-1:], smooth(log_unsafe["cti_violation_rate"]), color="#ff6b35", lw=2, label="Unconstrained MAPPO")
    ax.set_ylabel("CTI Violation Rate (%)", fontsize=9)
    ax.set_title("Per-Episode Violation Rate", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Panel 2: Lagrange Multiplier Norm
    ax = axes[0, 1]
    ax.plot(eps, log_safe["lagrange_norm"], color="#39d353", lw=1.5)
    ax.set_ylabel("||λ|| (L2 norm)", fontsize=9)
    ax.set_title("Lagrange Multiplier Evolution", fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Panel 3: Cumulative Violations
    ax = axes[1, 0]
    ax.plot(eps, log_safe["cumulative_violations"], color="#2563eb", lw=2, label="MAPPO-Safe")
    ax.plot(eps, log_unsafe["cumulative_violations"], color="#ff6b35", lw=2, label="Unconstrained MAPPO")
    ax.set_ylabel("Cumulative CTI Violations", fontsize=9)
    ax.set_xlabel("Episode", fontsize=9)
    ax.set_title("Cumulative Violations", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Panel 4: Safety Activation Frequency
    ax = axes[1, 1]
    safety_rate = np.array(log_safe["safety_activations"]) / env_safe.n_agents * 100
    ax.plot(eps, safety_rate, alpha=0.2, color="#a371f7", lw=0.5)
    ax.plot(eps[window-1:], smooth(safety_rate), color="#a371f7", lw=2)
    ax.set_ylabel("Safety Activation (% of actions)", fontsize=9)
    ax.set_xlabel("Episode", fontsize=9)
    ax.set_title("Safety Projection Frequency", fontsize=10)
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(out_path / "fig09_cti_dynamics.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", out_path / "fig09_cti_dynamics.png")
    
    # Summary
    print("\n" + "=" * 60)
    print("CTI DYNAMICS SUMMARY")
    print("=" * 60)
    print(f"{'Metric':>30} {'MAPPO-Safe':>14} {'Unconstrained':>14}")
    print("-" * 60)
    print(f"{'Final violation rate (%)':>30} {log_safe['cti_violation_rate'][-1]:>14.2f} {log_unsafe['cti_violation_rate'][-1]:>14.2f}")
    print(f"{'Cumulative violations':>30} {log_safe['cumulative_violations'][-1]:>14} {log_unsafe['cumulative_violations'][-1]:>14}")
    print(f"{'Mean reward (last 100)':>30} {np.mean(log_safe['mean_reward'][-100:]):>14.3f} {np.mean(log_unsafe['mean_reward'][-100:]):>14.3f}")
    print(f"{'Final Lagrange norm':>30} {log_safe['lagrange_norm'][-1]:>14.3f} {'N/A':>14}")
    print("=" * 60)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Figure 9 — CTI Violation Dynamics")
    p.add_argument("--n_episodes", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", type=str, default="outputs")
    p.add_argument("--system", type=str, default="8bus", help="Test system (8bus, 14bus, 30bus, 39bus)")
    args = p.parse_args()
    main(args.n_episodes, args.seed, args.output_dir, args.system)