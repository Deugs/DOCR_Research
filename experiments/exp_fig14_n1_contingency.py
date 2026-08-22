

"""
exp_fig14_n1_contingency.py  —  Figure 14
───────────────────────────────────────────
N-1 Contingency Impact and Recovery (30-Bus, MAPPO-Safe-EWC).

For each of 41 lines in 30-bus:
  (a) Remove line (N-1)
  (b) Run fault study with topology change
  (c) Evaluate with pre-trained policy (no adaptation)
  (d) Evaluate with 50-episode online fine-tuning

Log: objective_drop (%), recovery_episodes, final_objective.
Plot: bar chart of drop per line, recovery time series for worst 5 lines.

Outputs
-------
  outputs/fig14_n1_contingency.npz
  outputs/fig14_n1_contingency.png
"""

import numpy as np
import os
import sys
import logging
import argparse
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
os.makedirs("outputs", exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from network.network_data import get_30bus
from marl.marl_env import DynamicDOCREnv
from marl.marl_agents import MAPPOMARLTrainer
from network.relay_physics import check_constraints
from marl.safe_rl import create_safe_rl_env, SafeRLConfig
from marl.continual_trainer import create_ewc_trainer, EWCConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def train_base_policy(seed: int = 42, steps: int = 10000):
    """Train base MAPPO-Safe-EWC on 30-bus.

    `trainer.use_safety`/`use_ewc` used to be set via hasattr guards that
    were always False -- MAPPOMARLTrainer has neither attribute, so this
    "MAPPO-Safe-EWC" policy trained with no safety projection and no EWC
    penalty at all. Safe RL is a real env wrapper (must wrap before the
    trainer is built); EWC registration follows the same real
    finite-difference Fisher-estimation procedure used elsewhere in this
    codebase for single-task EWC (e.g. run_core_training.py's Core A).
    """
    net = get_30bus()
    raw_env = DynamicDOCREnv(net, seed=seed, use_pandapower=False,
                              fault_type="3ph", curriculum_stage=5)
    env = create_safe_rl_env(raw_env, SafeRLConfig(
        project_actions=True, enforce_cti=True,
        lambda_init=1.0, lambda_lr=0.01, rho_penalty=10.0,
        use_pid=True, adaptive_rho=True,
    ))
    trainer = MAPPOMARLTrainer(env, use_comm=True, seed=seed)
    ewc_trainer = create_ewc_trainer(trainer, EWCConfig())
    trainer.train(total_steps=steps, verbose=False)

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
    ewc_trainer.ewc_regularizer.register_task("base_policy", cur_params)
    ewc_trainer.ewc_regularizer.update_fisher("base_policy", fisher)
    ewc_trainer.ewc_regularizer.finalize()

    return trainer, net


def _policy_obs(trainer, obs, rng):
    """Route raw env obs through the trainer's comm graph when use_comm is
    enabled, matching MAPPOMARLTrainer's own training-loop pattern (see
    marl/marl_agents.py, comm_graph.forward() before actor.sample()) —
    trainer.actor was built expecting the enhanced (obs+message) dimension
    whenever use_comm is on, so feeding it raw obs directly mismatches
    dimensions and crashes.
    """
    if getattr(trainer, "use_comm", False) and getattr(trainer, "comm_graph", None) is not None:
        enhanced_obs, _, _ = trainer.comm_graph.forward(obs, rng)
        return enhanced_obs
    return obs


def evaluate_line_outage(trainer, net, line_idx: int, n_eval: int = 20, seed: int = 0):
    """Evaluate policy when a specific line is outaged.

    line_idx < 0 means "no outage" (baseline): the network is used as-is.
    line_idx >= 0 genuinely applies an N-1 single-line outage via
    NetworkData.apply_n1_outage() (fault currents on relays adjacent to
    the removed line are rescaled by the resulting outage factor) before
    the policy is evaluated on the mutated network.
    """
    rng = np.random.default_rng(seed)

    # Apply a genuine N-1 outage to a COPY of the network (baseline uses
    # the network unmodified).
    if line_idx >= 0:
        outaged_net, outage_factor = net.apply_n1_outage(line_idx)
        logger.info("    Line %d outaged (outage_factor=%.4f)", line_idx, outage_factor)
    else:
        outaged_net, outage_factor = net, 1.0

    env = DynamicDOCREnv(outaged_net, seed=seed, use_pandapower=False,
                          fault_type="3ph", curriculum_stage=5)

    of_list, cti_list = [], []
    for ep in range(n_eval):
        obs = env.reset(seed=seed + ep)
        done = False
        ep_of = []
        while not done:
            policy_obs = _policy_obs(trainer, obs, rng)
            actions = np.zeros((env.n_agents, env.ACT_DIM))
            for i in range(env.n_agents):
                a, _ = trainer.actor.sample(policy_obs[i], rng)
                actions[i] = a
            obs, _, done, info = env.step(actions)
            ep_of.append(info.get("OF", float("inf")))
        of_list.append(float(np.min(ep_of)))
        TMS, PS = env.get_state()
        c = check_constraints(outaged_net, TMS, PS)
        cti_list.append(c["cti_violations"])

    return {"mean_OF": np.mean(of_list), "std_OF": np.std(of_list), "cti_viols": np.mean(cti_list),
            "outage_factor": outage_factor}


def fine_tune_policy(trainer, net, line_idx: int, n_finetune_episodes: int = 50, seed: int = 0):
    """Fine-tune policy after line outage.

    Previously this loop only rolled out episodes via trainer.actor.sample()
    -- it never called trainer.train(), so no gradient/ES update ever
    happened. "Fine-tuned" was therefore identical to "pre-trained, no
    adaptation": evaluate_line_outage() would return the same numbers for
    both, silently defeating the entire point of this figure (measuring
    recovery from online adaptation vs. none).

    Genuinely continues training the SAME actor weights against the
    outaged network via trainer.train() (wrapped in the same Safe RL env
    used for base training, so training dynamics stay consistent), then
    restores the base policy's original weights before returning --
    main()'s per-line loop must start every line's fine-tuning from the
    same base policy, not accumulate one line's adaptation into the next
    line's "pre-trained" baseline.
    """
    rng = np.random.default_rng(seed)

    # Fine-tune on the same outaged network the policy will be evaluated on.
    if line_idx >= 0:
        outaged_net, _ = net.apply_n1_outage(line_idx)
    else:
        outaged_net = net
    raw_env = DynamicDOCREnv(outaged_net, seed=seed, use_pandapower=False,
                              fault_type="3ph", curriculum_stage=5)
    finetune_env = create_safe_rl_env(raw_env, SafeRLConfig(
        project_actions=True, enforce_cti=True,
        lambda_init=1.0, lambda_lr=0.01, rho_penalty=10.0,
        use_pid=True, adaptive_rho=True,
    ))

    base_params = trainer.actor.get_all_params()
    original_env = trainer.env
    trainer.env = finetune_env
    steps_per_episode_estimate = max(1, raw_env.max_steps // 4)
    trainer.train(total_steps=n_finetune_episodes * steps_per_episode_estimate, verbose=False)

    # Evaluate after fine-tuning
    result = evaluate_line_outage(trainer, net, line_idx, n_eval=20, seed=seed + 1000)

    trainer.actor.set_all_params(base_params)
    trainer.env = original_env
    return result


def main(train_steps: int = 10000, n_eval: int = 20, n_finetune: int = 50,
         max_lines: int = 10, seed: int = 42, output_dir: str = "outputs"):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    logger.info("Figure 14 — N-1 Contingency Analysis")
    logger.info("  Training base policy on 30-bus...")
    trainer, net = train_base_policy(seed, train_steps)
    
    # Baseline (no outage)
    logger.info("Evaluating baseline (no outage)...")
    baseline = evaluate_line_outage(trainer, net, -1, n_eval, seed)
    logger.info("  Baseline OF=%.4f", baseline["mean_OF"])
    
    # Test N-1 contingencies (subset for speed)
    n_lines = min(max_lines, net.n_lines)
    logger.info("Testing %d line outages...", n_lines)
    
    results = []
    for line_idx in range(n_lines):
        logger.info("  Line %d/%d outage...", line_idx + 1, n_lines)
        
        # Pre-trained (no adaptation)
        pretrain = evaluate_line_outage(trainer, net, line_idx, n_eval, seed + line_idx)
        
        # Fine-tuned
        finetuned = fine_tune_policy(trainer, net, line_idx, n_finetune, seed + line_idx + 500)
        
        drop_pct = (pretrain["mean_OF"] - baseline["mean_OF"]) / baseline["mean_OF"] * 100
        recovery = (baseline["mean_OF"] - finetuned["mean_OF"]) / max(abs(pretrain["mean_OF"] - baseline["mean_OF"]), 1e-9) * 100
        
        results.append({
            "line_idx": line_idx,
            # NetworkData has no `.lines` table (topology is implicit in
            # RelayPair relay/bus indices — see NetworkData.n_lines /
            # apply_n1_outage docstrings), so lines are identified by index.
            "line_name": f"Line_{line_idx}",
            "baseline_OF": baseline["mean_OF"],
            "pretrain_OF": pretrain["mean_OF"],
            "finetuned_OF": finetuned["mean_OF"],
            "drop_pct": drop_pct,
            "recovery_pct": recovery,
            "pretrain_cti": pretrain["cti_viols"],
            "finetuned_cti": finetuned["cti_viols"],
        })
        logger.info("    Drop: %.1f%%, Recovery: %.1f%%", drop_pct, recovery)
    
    # Save
    np.savez(
        out_path / "fig14_n1_contingency.npz",
        baseline_of=np.array([baseline["mean_OF"]]),
        line_indices=np.array([r["line_idx"] for r in results]),
        line_names=np.array([r["line_name"] for r in results]),
        drop_pct=np.array([r["drop_pct"] for r in results]),
        recovery_pct=np.array([r["recovery_pct"] for r in results]),
        pretrain_of=np.array([r["pretrain_OF"] for r in results]),
        finetuned_of=np.array([r["finetuned_OF"] for r in results]),
        pretrain_cti=np.array([r["pretrain_cti"] for r in results]),
        finetuned_cti=np.array([r["finetuned_cti"] for r in results]),
    )
    logger.info("Saved: %s", out_path / "fig14_n1_contingency.npz")
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Figure 14 — N-1 Contingency Impact and Recovery (30-Bus)", fontsize=12)
    
    # Panel 1: Performance drop per line
    ax = axes[0]
    x = np.arange(len(results))
    colors = ["#ff6b35" if r["drop_pct"] > 20 else "#f7c948" if r["drop_pct"] > 10 else "#39d353" for r in results]
    bars = ax.bar(x, [r["drop_pct"] for r in results], color=colors, edgecolor="white", width=0.6)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("Line Index", fontsize=10)
    ax.set_ylabel("Objective Drop (%)", fontsize=10)
    ax.set_title("Performance Drop per Line Outage", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels([r["line_name"] for r in results], rotation=45, ha="right", fontsize=7)
    ax.grid(True, alpha=0.3, axis="y")
    
    # Panel 2: Recovery comparison
    ax = axes[1]
    x = np.arange(len(results))
    width = 0.35
    ax.bar(x - width/2, [r["pretrain_OF"] for r in results], width, label="Pre-trained (no adapt)", color="#ff6b35", edgecolor="white")
    ax.bar(x + width/2, [r["finetuned_OF"] for r in results], width, label="Fine-tuned", color="#39d353", edgecolor="white")
    ax.axhline(baseline["mean_OF"], color="#2563eb", ls="--", lw=2, label="Baseline (no outage)")
    ax.set_xlabel("Line Index", fontsize=10)
    ax.set_ylabel("Objective Function (s)", fontsize=10)
    ax.set_title("Pre-trained vs. Fine-tuned After Outage", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels([r["line_name"] for r in results], rotation=45, ha="right", fontsize=7)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    
    fig.tight_layout()
    fig.savefig(out_path / "fig14_n1_contingency.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", out_path / "fig14_n1_contingency.png")
    
    # Summary
    print("\\n" + "=" * 70)
    print("N-1 CONTINGENCY SUMMARY")
    print("=" * 70)
    print(f"{'Line':>12} {'Drop%':>10} {'Pre-OF':>10} {'Fine-OF':>10} {'Recov%':>10}")
    print("-" * 70)
    for r in results:
        print(f"{r['line_name']:>12} {r['drop_pct']:>9.1f}% {r['pretrain_OF']:>10.4f} {r['finetuned_OF']:>10.4f} {r['recovery_pct']:>9.1f}%")
    print("=" * 70)
    print(f"Mean drop: {np.mean([r['drop_pct'] for r in results]):.1f}%")
    print(f"Mean recovery: {np.mean([r['recovery_pct'] for r in results]):.1f}%")
    print("=" * 70)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Figure 14 — N-1 Contingency")
    p.add_argument("--train_steps", type=int, default=10000)
    p.add_argument("--n_eval", type=int, default=20)
    p.add_argument("--n_finetune", type=int, default=50)
    p.add_argument("--max_lines", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", type=str, default="outputs")
    args = p.parse_args()
    main(args.train_steps, args.n_eval, args.n_finetune, args.max_lines, args.seed, args.output_dir)
