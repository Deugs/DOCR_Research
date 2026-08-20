"""
exp_fig21_action_projection.py  —  Figure 21
─────────────────────────────────────────────
Action Projection Efficacy During Training (MAPPO-Safe, 8-Bus, 100% PV).

Projection frequency, projection magnitude ||a_raw - a_safe||,
CTI violation rate before/after projection.

In safety_layer.py, log:
  (a) binary flag: was projection applied?
  (b) L2 norm of projection delta
  (c) CTI violation count using a_raw vs. a_safe

Aggregate per 100 episodes.
Compare with unconstrained MAPPO baseline.

Outputs
-------
  outputs/fig21_projection_efficacy.npz
  outputs/fig21_projection_efficacy.png
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
from marl.safe_rl import create_safe_rl_env, SafeRLConfig
from network.relay_physics import check_constraints

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def run_with_projection_logging(env, trainer, n_episodes: int = 2000, seed: int = 42,
                                 is_safe: bool = False):
    """Run training and log projection statistics.

    `env` must already be the fully-formed environment for this condition —
    a `SafeRLWrapper` (from `create_safe_rl_env`) when `is_safe=True`, or the
    raw `DynamicDOCREnv` (no projection at all) when `is_safe=False`. The
    wrapper performs real action projection inside its own `step()`, so we
    just hand it the raw policy actions and read the real projection stats
    back out of `info["safe_rl"]` — there is no per-agent `safety_layer` on
    the trainer to reach into.
    """
    rng = np.random.default_rng(seed)
    log = {
        "episode": [], "projection_freq": [], "projection_magnitude": [],
        "cti_before": [], "cti_after": [], "reward": [],
    }

    for ep in range(n_episodes):
        obs = env.reset(seed=seed + ep)
        done = False
        ep_projections = 0
        ep_steps = 0
        ep_proj_magnitude = []
        ep_reward = 0

        while not done:
            actions_raw = np.zeros((env.n_agents, env.ACT_DIM))
            for i in range(env.n_agents):
                a_raw, _ = trainer.actor.sample(obs[i], rng)
                actions_raw[i] = a_raw

            # `env.step` applies real action projection internally when this
            # is the Safe RL wrapper; the raw env applies none.
            obs, reward, done, info = env.step(actions_raw)
            ep_reward += reward
            ep_steps += 1

            if is_safe:
                safe_info = info["safe_rl"]
                if safe_info.get("projected_actions"):
                    ep_projections += 1
                ep_proj_magnitude.append(safe_info.get("projection_magnitude", 0.0))

        # Get final CTI
        TMS, PS = env.get_state()
        c = check_constraints(env.net, TMS, PS)

        log["episode"].append(ep)
        log["projection_freq"].append(ep_projections / max(ep_steps, 1) * 100)
        log["projection_magnitude"].append(np.mean(ep_proj_magnitude) if ep_proj_magnitude else 0)
        log["cti_after"].append(c["cti_violations"])
        log["reward"].append(ep_reward)

        if ep % 200 == 0:
            logger.info("Ep %d: proj_freq=%.1f%% avg_mag=%.4f cti=%.1f",
                       ep, log["projection_freq"][-1], log["projection_magnitude"][-1], c["cti_violations"])

    return log


def main(n_episodes: int = 2000, seed: int = 42, output_dir: str = "outputs"):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    logger.info("Figure 21 — Action Projection Efficacy")
    logger.info("  Episodes: %d", n_episodes)
    
    net = get_8bus()

    # MAPPO-Safe: wrap the env with the real Safe RL layer BEFORE building
    # the trainer, so action projection genuinely happens.
    logger.info("[1/2] Training MAPPO-Safe...")
    raw_env_safe = DynamicDOCREnv(net, seed=seed, use_pandapower=False,
                                   fault_type="3ph", curriculum_stage=5)
    env_safe = create_safe_rl_env(raw_env_safe, SafeRLConfig(
        project_actions=True, enforce_cti=True,
        lambda_init=1.0, lambda_lr=0.01, rho_penalty=10.0,
        use_pid=True, adaptive_rho=True,
    ))
    trainer_safe = MAPPOMARLTrainer(env_safe, use_comm=False, seed=seed)
    log_safe = run_with_projection_logging(env_safe, trainer_safe, n_episodes, seed, is_safe=True)

    # Unconstrained MAPPO: genuinely raw env, no projection, no Lagrangian.
    logger.info("[2/2] Training unconstrained MAPPO...")
    env_unsafe = DynamicDOCREnv(net, seed=seed+1000, use_pandapower=False,
                                 fault_type="3ph", curriculum_stage=5)
    trainer_unsafe = MAPPOMARLTrainer(env_unsafe, use_comm=False, seed=seed+1000)
    log_unsafe = run_with_projection_logging(env_unsafe, trainer_unsafe, n_episodes, seed+1000, is_safe=False)
    
    # Save
    np.savez(
        out_path / "fig21_projection_efficacy.npz",
        episodes=np.array(log_safe["episode"]),
        safe_proj_freq=np.array(log_safe["projection_freq"]),
        safe_proj_mag=np.array(log_safe["projection_magnitude"]),
        safe_cti=np.array(log_safe["cti_after"]),
        safe_reward=np.array(log_safe["reward"]),
        unsafe_cti=np.array(log_unsafe["cti_after"]),
        unsafe_reward=np.array(log_unsafe["reward"]),
    )
    logger.info("Saved: %s", out_path / "fig21_projection_efficacy.npz")
    
    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("Figure 21 — Action Projection Efficacy During Training", fontsize=12)
    
    eps = np.array(log_safe["episode"])
    window = 50
    
    def smooth(y):
        if len(y) < window:
            return y
        return np.convolve(y, np.ones(window)/window, mode='valid')
    
    # Panel 1: Projection frequency
    ax = axes[0, 0]
    ax.plot(eps, log_safe["projection_freq"], alpha=0.2, color="#2563eb", lw=0.5)
    ax.plot(eps[window-1:], smooth(log_safe["projection_freq"]), color="#2563eb", lw=2)
    ax.set_ylabel("Projection Frequency (%)", fontsize=10)
    ax.set_title("Safety Projection Activation Rate", fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Panel 2: Projection magnitude
    ax = axes[0, 1]
    mag = np.array(log_safe["projection_magnitude"])
    ax.plot(eps, mag, alpha=0.2, color="#39d353", lw=0.5)
    ax.plot(eps[window-1:], smooth(mag), color="#39d353", lw=2)
    ax.set_ylabel("Mean Projection Magnitude", fontsize=10)
    ax.set_title("Action Correction Magnitude", fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Panel 3: CTI comparison
    ax = axes[1, 0]
    ax.plot(eps, log_safe["cti_after"], alpha=0.2, color="#2563eb", lw=0.5, label="MAPPO-Safe")
    ax.plot(eps[window-1:], smooth(log_safe["cti_after"]), color="#2563eb", lw=2)
    ax.plot(eps, log_unsafe["cti_after"], alpha=0.2, color="#ff6b35", lw=0.5, label="Unconstrained")
    ax.plot(eps[window-1:], smooth(log_unsafe["cti_after"]), color="#ff6b35", lw=2)
    ax.set_ylabel("CTI Violations", fontsize=10)
    ax.set_xlabel("Episode", fontsize=10)
    ax.set_title("CTI Violations: Constrained vs. Unconstrained", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Panel 4: Reward comparison
    ax = axes[1, 1]
    ax.plot(eps, log_safe["reward"], alpha=0.2, color="#2563eb", lw=0.5, label="MAPPO-Safe")
    ax.plot(eps[window-1:], smooth(log_safe["reward"]), color="#2563eb", lw=2)
    ax.plot(eps, log_unsafe["reward"], alpha=0.2, color="#ff6b35", lw=0.5, label="Unconstrained")
    ax.plot(eps[window-1:], smooth(log_unsafe["reward"]), color="#ff6b35", lw=2)
    ax.set_ylabel("Episode Reward", fontsize=10)
    ax.set_xlabel("Episode", fontsize=10)
    ax.set_title("Learning Speed Comparison", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(out_path / "fig21_projection_efficacy.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", out_path / "fig21_projection_efficacy.png")
    
    # Summary
    print("\\n" + "=" * 60)
    print("PROJECTION EFFICACY SUMMARY")
    print("=" * 60)
    print(f"{'Metric':>30} {'MAPPO-Safe':>14} {'Unconstrained':>14}")
    print("-" * 60)
    print(f"{'Mean projection freq (%)':>30} {np.mean(log_safe['projection_freq']):>14.2f} {'N/A':>14}")
    print(f"{'Mean projection magnitude':>30} {np.mean(log_safe['projection_magnitude']):>14.4f} {'N/A':>14}")
    print(f"{'Final CTI violations':>30} {np.mean(log_safe['cti_after'][-100:]):>14.2f} {np.mean(log_unsafe['cti_after'][-100:]):>14.2f}")
    print(f"{'Final reward':>30} {np.mean(log_safe['reward'][-100:]):>14.3f} {np.mean(log_unsafe['reward'][-100:]):>14.3f}")
    print("=" * 60)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Figure 21 — Action Projection Efficacy")
    p.add_argument("--n_episodes", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", type=str, default="outputs")
    args = p.parse_args()
    main(args.n_episodes, args.seed, args.output_dir)
