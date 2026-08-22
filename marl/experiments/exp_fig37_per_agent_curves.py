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
from marl.continual_trainer import create_ewc_trainer, EWCConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def run_per_agent_logging(env, trainer, n_episodes: int = 1000, seed: int = 42):
    """Run training and log per-agent rewards.

    Previously never called trainer.train() -- only trainer.actor.sample()
    against a frozen, randomly-initialized network across all n_episodes,
    so every "per-agent learning curve" this produced was noise, not
    genuine per-agent learning (the same bug class already fixed in
    experiments/exp_fig12_convergence_curves.py). Also fed raw obs
    directly to actor.sample() despite the trainer being built with
    use_comm=True (see main()), crashing with a matmul dimension
    mismatch -- now routes through trainer.comm_graph.forward() first.
    """
    rng = np.random.default_rng(seed)
    n_agents = env.n_agents

    log = {
        "episode": [],
        "per_agent_reward": [],  # (n_episodes, n_agents)
        "per_agent_value_loss": [],  # if available
    }

    steps_per_episode_estimate = max(1, env.max_steps // 4)

    for ep in range(n_episodes):
        # One real training step per "episode" tick on the x-axis -- this
        # is what makes the per-agent trajectories genuine learning curves.
        trainer.train(total_steps=steps_per_episode_estimate, verbose=False)

        obs = env.reset(seed=seed + ep)
        done = False
        ep_rewards = np.zeros(n_agents)

        while not done:
            actions = np.zeros((n_agents, env.ACT_DIM))
            action_qualities = np.zeros(n_agents)

            if trainer.comm_graph is not None:
                enhanced, _, _ = trainer.comm_graph.forward(obs, rng)
            else:
                enhanced = obs

            for i in range(n_agents):
                a, log_prob = trainer.actor.sample(enhanced[i], rng)
                actions[i] = a
                # Proxy for action quality: negative log probability (lower = more confident)
                action_qualities[i] = -log_prob if log_prob is not None else 0.5

            obs, reward, done, info = env.step(actions)
            # env.step() returns a per-agent reward array, not a single
            # global scalar -- sum it into a "global" pool before
            # redistributing by action quality, matching this function's
            # own stated intent ("Distribute global reward proportionally
            # to action quality").
            global_reward = float(np.sum(reward))

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
    # `trainer.use_ewc` was a dead attribute set -- MAPPOMARLTrainer never
    # defines it, so this "MAPPO-Safe-EWC" run never applied any EWC
    # penalty at all. Real EWC needs an actual create_ewc_trainer wrapper
    # plus real Fisher registration (the same single-task pattern already
    # used in run_core_training.py's Core A and the sibling exp_fig12/13/
    # 14/22/23 figure scripts).
    ewc_trainer = create_ewc_trainer(trainer, EWCConfig())

    log = run_per_agent_logging(env, trainer, n_episodes, seed)

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
    ewc_trainer.ewc_regularizer.register_task("fig37_base_policy", cur_params)
    ewc_trainer.ewc_regularizer.update_fisher("fig37_base_policy", fisher)
    ewc_trainer.ewc_regularizer.finalize()
    
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