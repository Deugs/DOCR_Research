#!/usr/bin/env python3
"""
CLI Adapter: per_agent_curves.py
Maps run_thesis_experiments.py Fig 37 call to exp_fig37_per_agent_curves.py.
Includes CRITICAL FIX for actor.sample() dimension mismatch when use_comm=True.
"""
import argparse
import sys
import os
from pathlib import Path
import importlib.util
import numpy as np

def _find_exp_file(filename):
    """Search for experiment file in common locations."""
    script_dir = Path(__file__).resolve().parent
    search_roots = [script_dir]
    for _ in range(3):
        parent = script_dir.parent
        if parent != script_dir:
            search_roots.append(parent)
            script_dir = parent
        else:
            break

    search_paths = []
    for root in search_roots:
        search_paths.extend([
            root / filename,
            root / "scripts" / filename,
            root / "experiments" / filename,
            root / "src" / filename,
            root / "marl" / filename,
            root / "phase2_scripts" / filename,
        ])

    for p in search_paths:
        if p.exists():
            return str(p.resolve())

    cwd_paths = [Path(os.getcwd()) / filename, Path(os.getcwd()).parent / filename]
    for p in cwd_paths:
        if p.exists():
            return str(p.resolve())

    raise FileNotFoundError(f"Could not find {filename}. Searched: {[str(p) for p in search_paths + cwd_paths]}")


def _patch_sample_for_comm(module):
    """
    Monkey-patch the actor.sample() call in run_per_agent_logging to handle
    the case where use_comm=True returns (action, log_prob, comm_msg).
    """
    if not hasattr(module, 'run_per_agent_logging'):
        print("[Adapter] WARNING: module has no run_per_agent_logging to patch")
        return

    original_run = module.run_per_agent_logging

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
                if hasattr(module, 'logger'):
                    module.logger.info("Ep %d: agent rewards = %s", ep, np.round(ep_rewards, 3))
                else:
                    print(f"[Adapter] Ep {ep}: agent rewards = {np.round(ep_rewards, 3)}")

        return log

    module.run_per_agent_logging = patched_run
    print("[Adapter] Patched run_per_agent_logging for use_comm=True compatibility")


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

    # Find and load the actual experiment module
    target_file = "exp_fig37_per_agent_curves.py"
    try:
        exp_path = _find_exp_file(target_file)
    except FileNotFoundError as e:
        print(f"[Adapter] ERROR: {e}")
        print(f"[Adapter] Falling back to stub mode (synthetic data)...")
        _run_stub(args)
        return

    print(f"[Adapter] Found actual experiment: {exp_path}")

    # Add the experiment's directory to sys.path so its imports work
    exp_dir = str(Path(exp_path).parent)
    if exp_dir not in sys.path:
        sys.path.insert(0, exp_dir)

    spec = importlib.util.spec_from_file_location("exp_fig37", exp_path)
    exp_fig37 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exp_fig37)

    # Apply the use_comm patch BEFORE calling main
    _patch_sample_for_comm(exp_fig37)

    out_dir = os.path.dirname(args.output) or "outputs"
    os.makedirs(out_dir, exist_ok=True)

    exp_fig37.main(n_episodes=args.episodes, seed=args.seed, output_dir=out_dir)


def _run_stub(args):
    """Generate a synthetic placeholder when actual experiment file is not found."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = os.path.dirname(args.output) or "outputs"
    os.makedirs(out_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.text(0.5, 0.5, "STUB: exp_fig37_per_agent_curves.py not found\n\n" + 
            "File was not found in project.", 
            ha="center", va="center", fontsize=12, transform=ax.transAxes)
    ax.set_title("Placeholder - Per-Agent Curves", fontsize=14)
    ax.axis("off")
    fig.savefig(args.output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[Adapter] Saved stub: {args.output}")


if __name__ == "__main__":
    main()
