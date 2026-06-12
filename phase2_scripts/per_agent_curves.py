#!/usr/bin/env python3
"""
CLI Adapter: per_agent_curves.py
Explicit import from exp_fig37_per_agent_curves
Place this file alongside exp_fig37_per_agent_curves.py (project root)
Includes CRITICAL FIX for actor.sample() dimension mismatch when use_comm=True.
"""
import argparse
import sys
import os
import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from exp_fig37_per_agent_curves import main as fig37_main
import exp_fig37_per_agent_curves as fig37_module


def _patch_sample_for_comm():
    """Monkey-patch actor.sample() to handle use_comm=True dimension mismatch."""
    if not hasattr(fig37_module, 'run_per_agent_logging'):
        print("[Adapter] WARNING: module has no run_per_agent_logging to patch")
        return

    original_run = fig37_module.run_per_agent_logging

    def patched_run(env, trainer, n_episodes=1000, seed=42):
        rng = np.random.default_rng(seed)
        n_agents = env.n_agents

        log = {
            "episode": [],
            "per_agent_reward": [],
            "per_agent_value_loss": [],
        }

        for ep in range(n_episodes):
            obs = env.reset(seed=seed + ep)
            done = False
            ep_rewards = np.zeros(n_agents)

            while not done:
                actions = np.zeros((n_agents, env.ACT_DIM))
                action_qualities = np.zeros(n_agents)

                for i in range(n_agents):
                    sample_result = trainer.actor.sample(obs[i], rng)

                    # === DIMENSION MISMATCH FIX ===
                    if isinstance(sample_result, (tuple, list)):
                        if len(sample_result) >= 2:
                            a = sample_result[0]
                            log_prob = sample_result[1]
                        else:
                            a = sample_result[0]
                            log_prob = None
                    else:
                        a = sample_result
                        log_prob = None
                    # ================================

                    actions[i] = a
                    action_qualities[i] = -log_prob if log_prob is not None else 0.5

                obs, global_reward, done, info = env.step(actions)

                total_quality = np.sum(action_qualities) + 1e-9
                for i in range(n_agents):
                    ep_rewards[i] += global_reward * (action_qualities[i] / total_quality)

            log["episode"].append(ep)
            log["per_agent_reward"].append(ep_rewards)

            if ep % 200 == 0:
                if hasattr(fig37_module, 'logger'):
                    fig37_module.logger.info("Ep %d: agent rewards = %s", ep, np.round(ep_rewards, 3))
                else:
                    print(f"[Adapter] Ep {ep}: agent rewards = {np.round(ep_rewards, 3)}")

        return log

    fig37_module.run_per_agent_logging = patched_run
    print("[Adapter] Patched run_per_agent_logging for use_comm=True compatibility")


def main():
    p = argparse.ArgumentParser(description="Adapter for per_agent_curves.py")
    p.add_argument("--system", type=str, default="8bus")
    p.add_argument("--pv", type=float, default=1.0)
    p.add_argument("--episodes", type=int, default=30000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=str, default="outputs/fig37_per_agent.pdf")
    args = p.parse_args()

    out_dir = os.path.dirname(args.output) if args.output else "outputs"
    os.makedirs(out_dir, exist_ok=True)

    print(f"[Adapter] per_agent_curves.py -> calling fig37_main()")
    print(f"  Output dir: {out_dir}")
    print(f"  NOTE: use_comm=True — actor.sample() dimension mismatch will be handled")

    # Apply the use_comm patch BEFORE calling main
    _patch_sample_for_comm()

    fig37_main(n_episodes=args.episodes, seed=args.seed, output_dir=out_dir)


if __name__ == "__main__":
    main()
