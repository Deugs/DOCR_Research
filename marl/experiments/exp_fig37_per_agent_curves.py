"""
exp_fig37_per_agent_curves.py  —  Figure 37
─────────────────────────────────────────────
Per-Agent Learning Curves (Individual Convergence).

Separate return curves for each of the 7 agents (8-bus) or 14 agents (30-bus).
Heterogeneous learning speeds.

Modify logger in train.py to record per-agent reward.
For CTDE with global reward, approximate individual contribution via
marginal reward: r_i = r_global * (agent_i_action_quality / sum_all_actions).
Or use individual value function losses as proxy.

Outputs
-------
  outputs/fig37_per_agent.npz
  outputs/fig37_per_agent.png
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def run_per_agent_logging(env, trainer, n_episodes: int = 1000, seed: int = 42):
    """Run training and log per-agent rewards."""
    rng = np.random.default_rng(seed)
    n_agents = env.n_agents
    
    log = {
        "episode": [],
        "per_agent_reward": [],  # (n_episodes, n_agents)
        "per_agent_value_loss": [],  # if available
    }
    
    for ep in range(n_episodes):
        obs = env.reset(seed=seed + ep)
        done = False
        ep_rewards = np.zeros(n_agents)
        
        while not done:
            actions = np.zeros((n_agents, env.ACT_DIM))
            action_qualities = np.zeros(n_agents)
            
            for i in range(n_agents):
                a, log_prob = trainer.actor.sample(obs[i], rng)
                actions[i] = a
                # Proxy for action quality: negative log probability (lower = more confident)
                action_qualities[i] = -log_prob if log_prob is not None else 0.5
            
            obs, global_reward, done, info = env.step(actions)
            
            # Distribute global reward proportionally to action quality
            total_quality = np.sum(action_qualities) + 1e-9
            for i in range(n_agents):
                ep_rewards[i] += global_reward * (action_qualities[i] / total_quality)
        
        log["episode"].append(ep)
        log["per_agent_reward"].append(ep_rewards)
        
        if ep % 200 == 0:
            logger.info("Ep %d: agent rewards = %s", ep, np.round(ep_rewards, 3))
    
    return log


def main(n_episodes: int = 1000, seed: int = 42, output_dir: str = "outputs", system: str = "8bus"):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    logger.info("Figure 37 — Per-Agent Learning Curves")
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
    raw_env = DynamicDOCREnv(net, seed=seed, use_pandapower=False,
                              fault_type="3ph", curriculum_stage=5)
    # This run is "MAPPO-Safe-EWC" — wrap with the real Safe RL layer
    # BEFORE building the trainer. `trainer.use_safety = True` was a dead
    # attribute set (MAPPOMARLTrainer never defines `use_safety`).
    env = create_safe_rl_env(raw_env, SafeRLConfig(
        project_actions=True, enforce_cti=True,
        lambda_init=1.0, lambda_lr=0.01, rho_penalty=10.0,
        use_pid=True, adaptive_rho=True,
    ))
    trainer = MAPPOMARLTrainer(env, use_comm=True, seed=seed)
    if hasattr(trainer, 'use_ewc'):
        trainer.use_ewc = True
    
    log = run_per_agent_logging(env, trainer, n_episodes, seed)
    
    per_agent_rewards = np.array(log["per_agent_reward"])  # (n_episodes, n_agents)
    episodes = np.array(log["episode"])
    n_agents = per_agent_rewards.shape[1]
    
    # Save
    np.savez(
        out_path / "fig37_per_agent.npz",
        episodes=episodes,
        per_agent_rewards=per_agent_rewards,
        n_agents=np.array([n_agents]),
    )
    logger.info("Saved: %s", out_path / "fig37_per_agent.npz")
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Figure 37 — Per-Agent Learning Curves (Individual Convergence)", fontsize=12)
    
    # Panel 1: All agents overlaid
    ax = axes[0]
    colors = plt.cm.tab10(np.linspace(0, 1, n_agents))
    window = 50
    
    for i in range(n_agents):
        rewards = per_agent_rewards[:, i]
        smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
        ax.plot(episodes[window-1:], smoothed, color=colors[i], lw=1.5,
               label=f"Agent {i}", alpha=0.8)
    
    ax.set_xlabel("Episode", fontsize=10)
    ax.set_ylabel("Smoothed Reward", fontsize=10)
    ax.set_title("All Agents (Overlaid)", fontsize=11)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    
    # Panel 2: Convergence speed comparison
    ax = axes[1]
    # Find convergence episode for each agent (first episode where reward > 90% of final)
    final_rewards = per_agent_rewards[-50:].mean(axis=0)
    convergence_eps = []
    
    for i in range(n_agents):
        target = 0.9 * final_rewards[i]
        conv_ep = n_episodes
        for ep in range(50, n_episodes):
            if np.mean(per_agent_rewards[ep-50:ep, i]) >= target:
                conv_ep = ep
                break
        convergence_eps.append(conv_ep)
    
    sorted_idx = np.argsort(convergence_eps)
    bars = ax.barh(range(n_agents), [convergence_eps[i] for i in sorted_idx],
                   color=[colors[i] for i in sorted_idx], edgecolor="white")
    ax.set_yticks(range(n_agents))
    ax.set_yticklabels([f"Agent {i}" for i in sorted_idx], fontsize=8)
    ax.set_xlabel("Convergence Episode", fontsize=10)
    ax.set_title("Convergence Speed Ranking (Fastest → Slowest)", fontsize=11)
    ax.grid(True, alpha=0.3, axis="x")
    ax.invert_yaxis()
    
    fig.tight_layout()
    fig.savefig(out_path / "fig37_per_agent.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", out_path / "fig37_per_agent.png")
    
    # Summary
    print("\\n" + "=" * 60)
    print("PER-AGENT CONVERGENCE SUMMARY")
    print("=" * 60)
    print(f"{'Agent':>8} {'Final Reward':>14} {'Conv@Ep':>10} {'Status':>12}")
    print("-" * 60)
    for i in range(n_agents):
        status = "Bottleneck" if convergence_eps[i] == max(convergence_eps) else "Fast" if convergence_eps[i] == min(convergence_eps) else "Normal"
        print(f"{i:>8} {final_rewards[i]:>14.4f} {convergence_eps[i]:>10} {status:>12}")
    print("=" * 60)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Figure 37 — Per-Agent Curves")
    p.add_argument("--n_episodes", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", type=str, default="outputs")
    p.add_argument("--system", type=str, default="8bus", help="Test system (8bus, 14bus, 30bus, 39bus)")
    args = p.parse_args()
    main(args.n_episodes, args.seed, args.output_dir, args.system)