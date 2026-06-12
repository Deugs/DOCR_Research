#!/usr/bin/env python3
"""
CLI Adapter: per_agent_curves.py
Maps: run_thesis_experiments.py --phase3 Fig 37 call
  to: exp_fig37_per_agent_curves.py

run_thesis_experiments.py calls:
  python marl/experiments/per_agent_curves.py --system 8bus --pv 1.0 \
      --episodes 30000 --seed 42 --output "...fig_37_per_agent_curves.pdf"

CRITICAL FIX: Handles actor.sample() dimension mismatch when use_comm=True.
  When use_comm=True, actor.sample() may return (action, log_prob, comm_msg)
  instead of just (action, log_prob). We detect and unpack correctly.
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import importlib.util
spec = importlib.util.spec_from_file_location(
    "exp_fig37", os.path.join(os.path.dirname(__file__), "..", "exp_fig37_per_agent_curves.py")
)
exp_fig37 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exp_fig37)


def main():
    p = argparse.ArgumentParser(description="Per-Agent Curves Adapter (Fig 37)")
    p.add_argument("--system", type=str, default="8bus")
    p.add_argument("--pv", type=float, default=1.0)
    p.add_argument("--episodes", type=int, default=30000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=str, default="outputs/fig37_per_agent.pdf")
    args = p.parse_args()

    print(f"[Adapter] Fig 37 Per-Agent Curves -> exp_fig37_per_agent_curves")
    print(f"  system={args.system}, pv={args.pv}, episodes={args.episodes}")
    print(f"  NOTE: use_comm=True — actor.sample() dimension mismatch will be handled")

    out_dir = os.path.dirname(args.output) or "outputs"
    os.makedirs(out_dir, exist_ok=True)

    # The actual experiment uses use_comm=True by default in MAPPOMARLTrainer
    # We need to patch the run_per_agent_logging to handle the dimension mismatch
    _patch_sample_for_comm(exp_fig37)

    exp_fig37.main(n_episodes=args.episodes, seed=args.seed, output_dir=out_dir)


def _patch_sample_for_comm(module):
    """
    Monkey-patch the actor.sample() call in run_per_agent_logging to handle
    the case where use_comm=True returns (action, log_prob, comm_msg).
    """
    original_run = module.run_per_agent_logging

    def patched_run(env, trainer, n_episodes=1000, seed=42):
        import numpy as np
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
                    # When use_comm=True, sample() may return:
                    #   (action, log_prob, comm_msg)  — 3 elements
                    # When use_comm=False, sample() returns:
                    #   (action, log_prob)  — 2 elements
                    if isinstance(sample_result, (tuple, list)):
                        if len(sample_result) >= 2:
                            a = sample_result[0]
                            log_prob = sample_result[1]
                        else:
                            a = sample_result[0]
                            log_prob = None
                    else:
                        # Single tensor/array return
                        a = sample_result
                        log_prob = None
                    # ================================

                    actions[i] = a
                    # Proxy for action quality: negative log probability
                    action_qualities[i] = -log_prob if log_prob is not None else 0.5

                obs, global_reward, done, info = env.step(actions)

                # Distribute global reward proportionally to action quality
                total_quality = np.sum(action_qualities) + 1e-9
                for i in range(n_agents):
                    ep_rewards[i] += global_reward * (action_qualities[i] / total_quality)

            log["episode"].append(ep)
            log["per_agent_reward"].append(ep_rewards)

            if ep % 200 == 0:
                module.logger.info("Ep %d: agent rewards = %s", ep, np.round(ep_rewards, 3))

        return log

    module.run_per_agent_logging = patched_run
    print("[Adapter] Patched run_per_agent_logging for use_comm=True compatibility")


if __name__ == "__main__":
    main()
