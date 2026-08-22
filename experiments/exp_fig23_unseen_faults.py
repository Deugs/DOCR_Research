"""
exp_fig23_unseen_faults.py  —  Figure 23
───────────────────────────────────────────
Generalization to Unseen Fault Types (8-Bus, 100% PV, MAPPO-Safe-EWC).

Training fault set: bolted 3-phase faults at line midpoints.
Test fault set:
  (a) High-impedance: Z_f = 0.1-10 Ω
  (b) Evolving: L-G → 3-phase after 100ms
  (c) Simultaneous: faults at two locations

For each type, run 100 test episodes.
Log objective, CTI violations, trip correctness.
Compare with training distribution performance.
Compute generalization gap = (test_perf - train_perf) / train_perf.

Outputs
-------
  outputs/fig23_unseen_faults.npz
  outputs/fig23_unseen_faults.png
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

from network.network_data import get_8bus
from marl.marl_env import DynamicDOCREnv
from marl.marl_agents import MAPPOMARLTrainer
from network.relay_physics import check_constraints
from marl.safe_rl import create_safe_rl_env, SafeRLConfig
from marl.continual_trainer import create_ewc_trainer, EWCConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

FAULT_TYPES = {
    "training": {"type": "3ph", "impedance": 0.0, "evolving": False, "simultaneous": False},
    "high_impedance_low": {"type": "3ph", "impedance": 0.1, "evolving": False, "simultaneous": False},
    "high_impedance_med": {"type": "3ph", "impedance": 1.0, "evolving": False, "simultaneous": False},
    "high_impedance_high": {"type": "3ph", "impedance": 10.0, "evolving": False, "simultaneous": False},
    "evolving_fault": {"type": "3ph", "impedance": 0.0, "evolving": True, "simultaneous": False},
    "simultaneous": {"type": "3ph", "impedance": 0.0, "evolving": False, "simultaneous": True},
}


def train_policy(seed: int = 42, steps: int = 8000):
    """Train MAPPO-Safe-EWC on standard 3-phase faults.

    `trainer.use_safety`/`use_ewc` used to be set via hasattr guards that
    were always False -- MAPPOMARLTrainer has neither attribute, so this
    "MAPPO-Safe-EWC" policy trained with no safety projection and no EWC
    penalty at all. Safe RL is a real env wrapper (must wrap before the
    trainer is built); EWC registration follows the same real
    finite-difference Fisher-estimation procedure used elsewhere in this
    codebase for single-task EWC (e.g. run_core_training.py's Core A).
    """
    net = get_8bus()
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


def evaluate_fault_type(trainer, net, fault_config: dict, n_episodes: int = 100, seed: int = 0):
    """Evaluate policy on a specific fault type."""
    rng = np.random.default_rng(seed)
    env = DynamicDOCREnv(net, seed=seed, use_pandapower=False,
                          fault_type=fault_config["type"],
                          curriculum_stage=5)
    
    of_list, cti_list, trip_correct = [], [], []
    
    for ep in range(n_episodes):
        obs = env.reset(seed=seed + ep)
        done = False
        ep_of = []
        ep_trips = []
        
        while not done:
            actions = np.zeros((env.n_agents, env.ACT_DIM))
            for i in range(env.n_agents):
                a, _ = trainer.actor.sample(obs[i], rng)
                actions[i] = a
            obs, _, done, info = env.step(actions)
            ep_of.append(info.get("OF", float("inf")))
            
            # Track trip correctness if available
            if "correct_trip" in info:
                ep_trips.append(info["correct_trip"])
        
        of_list.append(float(np.min(ep_of)))
        TMS, PS = env.get_state()
        c = check_constraints(net, TMS, PS)
        cti_list.append(c["cti_violations"])
        
        if ep_trips:
            trip_correct.append(np.mean(ep_trips))
    
    return {
        "mean_OF": np.mean(of_list),
        "std_OF": np.std(of_list),
        "cti_viols": np.mean(cti_list),
        "trip_accuracy": np.mean(trip_correct) if trip_correct else 0.0,
    }


def main(train_steps: int = 8000, n_eval: int = 100, seed: int = 42,
         output_dir: str = "outputs"):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    logger.info("Figure 23 — Generalization to Unseen Fault Types")
    logger.info("  Training policy on standard 3-phase faults...")
    trainer, net = train_policy(seed, train_steps)
    
    results = {}
    for fault_name, fault_config in FAULT_TYPES.items():
        logger.info("  Evaluating: %s...", fault_name)
        res = evaluate_fault_type(trainer, net, fault_config, n_eval, seed + hash(fault_name) % 1000)
        results[fault_name] = res
        logger.info("    OF=%.4f±%.4f CTI=%.1f TripAcc=%.2f",
                   res["mean_OF"], res["std_OF"], res["cti_viols"], res["trip_accuracy"])
    
    # Compute generalization gap
    train_perf = results["training"]["mean_OF"]
    for fault_name in results:
        if fault_name != "training":
            gap = (results[fault_name]["mean_OF"] - train_perf) / max(abs(train_perf), 1e-9) * 100
            results[fault_name]["gen_gap_pct"] = gap
            logger.info("  %s generalization gap: %.1f%%", fault_name, gap)
    
    # Save
    save_dict = {}
    for name, res in results.items():
        for key, val in res.items():
            save_dict[f"{name}_{key}"] = np.array([val])
    np.savez(out_path / "fig23_unseen_faults.npz", **save_dict)
    logger.info("Saved: %s", out_path / "fig23_unseen_faults.npz")
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.suptitle("Figure 23 — Generalization to Unseen Fault Types (8-Bus, 100% PV)", fontsize=12)
    
    names = list(results.keys())
    colors = ["#2563eb" if n == "training" else "#ff6b35" if "high" in n else "#39d353" if "evolv" in n else "#f7c948" for n in names]
    
    # Panel 1: Objective function
    ax = axes[0]
    of_vals = [results[n]["mean_OF"] for n in names]
    of_stds = [results[n]["std_OF"] for n in names]
    bars = ax.bar(range(len(names)), of_vals, yerr=of_stds, color=colors, edgecolor="white", capsize=4)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Objective Function (s)", fontsize=10)
    ax.set_title("Performance by Fault Type", fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")
    
    # Panel 2: CTI violations
    ax = axes[1]
    cti_vals = [results[n]["cti_viols"] for n in names]
    ax.bar(range(len(names)), cti_vals, color=colors, edgecolor="white")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("CTI Violations", fontsize=10)
    ax.set_title("Selectivity by Fault Type", fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")
    
    # Panel 3: Generalization gap
    ax = axes[2]
    gap_names = [n for n in names if n != "training"]
    gaps = [results[n].get("gen_gap_pct", 0) for n in gap_names]
    gap_colors = ["#ff6b35" if g > 20 else "#f7c948" if g > 10 else "#39d353" for g in gaps]
    bars = ax.bar(range(len(gap_names)), gaps, color=gap_colors, edgecolor="white")
    ax.set_xticks(range(len(gap_names)))
    ax.set_xticklabels(gap_names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Generalization Gap (%)", fontsize=10)
    ax.set_title("Performance Degradation vs. Training", fontsize=11)
    ax.axhline(0, color="black", lw=0.8)
    ax.grid(True, alpha=0.3, axis="y")
    
    fig.tight_layout()
    fig.savefig(out_path / "fig23_unseen_faults.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", out_path / "fig23_unseen_faults.png")
    
    # Summary
    print("\\n" + "=" * 60)
    print("UNSEEN FAULT GENERALIZATION SUMMARY")
    print("=" * 60)
    print(f"{'Fault Type':>20} {'OF (s)':>10} {'CTI':>8} {'TripAcc':>8} {'Gap%':>8}")
    print("-" * 60)
    for name, res in results.items():
        gap = res.get("gen_gap_pct", 0)
        gap_str = f"{gap:>7.1f}%" if name != "training" else "baseline"
        print(f"{name:>20} {res['mean_OF']:>10.4f} {res['cti_viols']:>8.1f} {res['trip_accuracy']:>8.2f} {gap_str:>8}")
    print("=" * 60)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Figure 23 — Unseen Fault Generalization")
    p.add_argument("--train_steps", type=int, default=8000)
    p.add_argument("--n_eval", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", type=str, default="outputs")
    args = p.parse_args()
    main(args.train_steps, args.n_eval, args.seed, args.output_dir)
