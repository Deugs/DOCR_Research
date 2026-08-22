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

# The real exp_fig*.py this adapter wraps lives in marl/experiments/, not alongside this file (see README_PHASE2_ADAPTERS.md —
# this package was never actually merged into that layout).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TARGET_DIR = os.path.join(_REPO_ROOT, "marl", "experiments")
if _TARGET_DIR not in sys.path:
    sys.path.insert(0, _TARGET_DIR)

from exp_fig37_per_agent_curves import main as fig37_main
import exp_fig37_per_agent_curves as fig37_module


def _patch_sample_for_comm():
    """Monkey-patch run_per_agent_logging to fix the use_comm=True
    dimension mismatch AND make it genuinely train.

    This adapter's patch previously only changed how actor.sample()'s
    return value was unpacked -- it still fed raw obs directly to an
    actor built for comm-graph-enhanced input (crashing with a matmul
    dimension mismatch whenever use_comm=True, which fig37_main() always
    sets), and it never called trainer.train() at all, so "per-agent
    learning curves" would have been pure noise from a frozen policy even
    if the crash were fixed. Now routes obs through
    trainer.comm_graph.forward() and calls trainer.train() once per
    episode tick, mirroring the real fix applied directly to
    marl/experiments/exp_fig37_per_agent_curves.py's own
    run_per_agent_logging.
    """
    if not hasattr(fig37_module, 'run_per_agent_logging'):
        print("[Adapter] WARNING: module has no run_per_agent_logging to patch")
        return

    def patched_run(env, trainer, n_episodes=1000, seed=42):
        rng = np.random.default_rng(seed)
        n_agents = env.n_agents

        log = {
            "episode": [],
            "per_agent_reward": [],
            "per_agent_value_loss": [],
        }

        steps_per_episode_estimate = max(1, env.max_steps // 4)

        for ep in range(n_episodes):
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
                    action_qualities[i] = -log_prob if log_prob is not None else 0.5

                obs, reward, done, info = env.step(actions)
                # env.step() returns a per-agent reward array, not a
                # single global scalar -- sum it into a "global" pool
                # before redistributing by action quality.
                global_reward = float(np.sum(reward))

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
    print("[Adapter] Patched run_per_agent_logging: real comm-graph routing + real training")


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

    fig37_main(n_episodes=args.episodes, seed=args.seed, output_dir=out_dir, system=args.system)


if __name__ == "__main__":
    main()
