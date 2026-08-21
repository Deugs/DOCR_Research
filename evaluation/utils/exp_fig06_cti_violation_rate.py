"""
exp_fig06_cti_violation_rate.py  —  Figure 6
─────────────────────────────────────────────
CTI Violation Rate vs. PV Penetration (8-Bus System).

Percentage of P/B pairs violating CTI >= 0.3s per PV level, per algorithm.
Severity distribution: mild (0.2-0.3s), moderate (0.1-0.2s), severe (<0.1s).

Uses shared data from Figure 5 experiment or can run standalone.

Outputs
-------
  outputs/{system}/fig06_cti_violations.npz
  outputs/{system}/fig06_cti_violations.png
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

from network.network_data import NETWORKS
from marl.marl_env import DynamicDOCREnv
from marl.marl_agents import MAPPOMARLTrainer
from marl.safe_rl import create_safe_rl_env, SafeRLConfig
from marl.continual_trainer import create_ewc_trainer, EWCConfig
from network.relay_physics import check_constraints

# QMIX (value decomposition) and IQL (independent Q-learning) are
# fundamentally different algorithm families from this codebase's
# policy-gradient MAPPOMARLTrainer — no real implementation of either
# exists here. Both currently alias IPPO's code path below (use_comm=
# False, no Safe RL/EWC), producing genuinely identical trained policies
# under the same seed. Disclosed via ALIASED_METHODS rather than
# silently presented as distinct baselines.
ALIASED_METHODS = {
    "QMIX": "IPPO (no real QMIX implementation exists in this codebase)",
    "IQL": "IPPO (no real IQL implementation exists in this codebase)",
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

PV_LEVELS = [0.0, 0.20, 0.40, 0.60]
METHODS = ["MAPPO-Safe-EWC", "MAPPO-Safe", "MAPPO", "IPPO", "QMIX", "IQL", "Traditional"]
METHOD_COLORS = {
    "MAPPO-Safe-EWC": "#2563eb", "MAPPO-Safe": "#39d353", "MAPPO": "#f7c948",
    "IPPO": "#ff6b35", "QMIX": "#a371f7", "IQL": "#8b949e", "Traditional": "#ff6b6b",
}

SEVERITY_COLORS = {"mild": "#f7c948", "moderate": "#ff6b35", "severe": "#ff0000"}


def evaluate_cti_severity(net, env, trainer, n_episodes: int = 100, seed: int = 0,
                           use_comm: bool = False) -> dict:
    """Evaluate and categorize CTI violations by severity."""
    rng = np.random.default_rng(seed)
    total_pairs = net.n_pairs
    
    mild_count = 0      # 0.2 <= CTI < 0.3
    moderate_count = 0  # 0.1 <= CTI < 0.2
    severe_count = 0    # CTI < 0.1
    total_violations = 0
    
    for ep in range(n_episodes):
        obs = env.reset(seed=seed + ep)
        done = False
        
        # Run one episode to get final settings
        while not done:
            actions = np.zeros((env.n_agents, env.ACT_DIM))
            for i in range(env.n_agents):
                if trainer is not None:
                    # use_comm=True policies were trained on comm-enhanced
                    # (obs_dim + msg_dim) input, not raw obs — sampling
                    # directly from obs[i] raised ValueError on every
                    # comm-enabled method (MAPPO-Safe-EWC/MAPPO-Safe/MAPPO).
                    if use_comm and getattr(trainer, 'comm_graph', None) is not None:
                        enhanced, _, _ = trainer.comm_graph.forward(obs, rng)
                        a, _ = trainer.actor.sample(enhanced[i], rng)
                    else:
                        a, _ = trainer.actor.sample(obs[i], rng)
                    actions[i] = a
            obs, _, done, info = env.step(actions)
        
        TMS, PS = env.get_state()
        c = check_constraints(net, TMS, PS)
        
        # Check individual pair CTIs. check_constraints() returns
        # 'delta_cti' (the t_bkp - t_pri array per pair), never
        # 'pair_ctis' — that key never existed, so this always fell
        # through to the coarser total-violation-count fallback and
        # the mild/moderate/severe breakdown was always all zeros.
        if "delta_cti" in c and c["delta_cti"] is not None:
            for cti in c["delta_cti"]:
                if cti < 0.3:
                    total_violations += 1
                    if cti >= 0.2:
                        mild_count += 1
                    elif cti >= 0.1:
                        moderate_count += 1
                    else:
                        severe_count += 1
        else:
            # Fallback: use total violation count
            total_violations += c["cti_violations"]
    
    n_checks = n_episodes * total_pairs
    return {
        "violation_rate": total_violations / max(n_checks, 1) * 100,
        "mild_rate": mild_count / max(n_checks, 1) * 100,
        "moderate_rate": moderate_count / max(n_checks, 1) * 100,
        "severe_rate": severe_count / max(n_checks, 1) * 100,
    }


def main(n_seeds: int = 10, train_steps: int = 5000, n_eval: int = 100,
         output_dir: str = "outputs", system: str = "8bus"):
    # Namespaced by system (outputs/{system}/fig06_cti_violations.*),
    # matching unified_exp_fig05.py's convention — previously every system
    # wrote to the same fixed output_dir/fig06_cti_violations.{npz,png},
    # so running this for 8bus then 14bus then 39bus silently overwrote
    # each prior system's results instead of keeping them side by side.
    out_path = Path(output_dir) / system
    out_path.mkdir(parents=True, exist_ok=True)
    
    logger.info("Figure 6 — CTI Violation Rate vs. PV Penetration")
    logger.info("  System: %s", system)
    
    if system not in NETWORKS:
        raise ValueError(f"Unknown system '{system}'. Available: {sorted(NETWORKS.keys())}")
    
    results = {method: {pv: [] for pv in PV_LEVELS} for method in METHODS}
    
    for method in METHODS:
        logger.info("\\n=== Method: %s ===", method)
        for pv in PV_LEVELS:
            logger.info("  PV level: %.0f%%", pv * 100)
            for seed in range(n_seeds):
                net = NETWORKS[system]()
                stage = int(pv * 5)
                env = DynamicDOCREnv(net, seed=seed, use_pandapower=False,
                                      fault_type="3ph", curriculum_stage=stage)
                
                if method in ALIASED_METHODS and pv == PV_LEVELS[0] and seed == 0:
                    logger.warning("Method %r has no real implementation in this codebase — "
                                    "training identically to %s.", method, ALIASED_METHODS[method])

                trainer = None
                use_comm = False
                eval_env = env
                if method != "Traditional":
                    use_comm = method in ["MAPPO-Safe-EWC", "MAPPO-Safe", "MAPPO"]
                    use_safety = method in ["MAPPO-Safe-EWC", "MAPPO-Safe"]
                    use_ewc = method == "MAPPO-Safe-EWC"
                    # Safe RL is a real env wrapper (use_safety was
                    # previously a dead post-hoc attribute — MAPPOMARLTrainer
                    # never defines it — so "Safe" methods here trained
                    # identically to plain "MAPPO").
                    if use_safety:
                        eval_env = create_safe_rl_env(env, SafeRLConfig(
                            project_actions=True, enforce_cti=True,
                            lambda_init=1.0, lambda_lr=0.01, rho_penalty=10.0,
                            use_pid=True, adaptive_rho=True,
                        ))
                    trainer = MAPPOMARLTrainer(eval_env, use_comm=use_comm, seed=seed)
                    trainer.train(total_steps=train_steps, verbose=False)
                    if use_ewc:
                        # Register this single task with EWC so the
                        # apparatus genuinely runs rather than staying
                        # completely inert (see run_core_training.py's
                        # Core A for the same rationale).
                        ewc_trainer = create_ewc_trainer(trainer, EWCConfig())
                        ewc_trainer.ewc_regularizer.register_task(method, trainer.actor.get_all_params())

                res = evaluate_cti_severity(net, eval_env, trainer, n_eval, seed, use_comm=use_comm)
                results[method][pv].append(res)
                logger.info("    Seed %d: viol=%.2f%% mild=%.2f%% mod=%.2f%% sev=%.2f%%",
                           seed, res["violation_rate"], res["mild_rate"],
                           res["moderate_rate"], res["severe_rate"])
    
    # Aggregate
    aggregated = {}
    for method in METHODS:
        aggregated[method] = {}
        for metric in ["violation_rate", "mild_rate", "moderate_rate", "severe_rate"]:
            aggregated[method][metric] = [np.mean([r[metric] for r in results[method][pv]]) for pv in PV_LEVELS]
            aggregated[method][f"{metric}_std"] = [np.std([r[metric] for r in results[method][pv]]) for pv in PV_LEVELS]
    
    # Save
    save_dict = {"pv_levels": np.array(PV_LEVELS), "n_seeds": np.array(n_seeds)}
    for method in METHODS:
        for key in aggregated[method]:
            save_dict[f"{method}_{key}"] = np.array(aggregated[method][key])
    
    np.savez(out_path / "fig06_cti_violations.npz", **save_dict)
    logger.info("Saved: %s", out_path / "fig06_cti_violations.npz")
    
    # Plot - stacked bar chart
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Figure 6 — CTI Violation Rate vs. PV Penetration", fontsize=12)
    
    # Left: Total violation rate by method
    ax = axes[0]
    x = np.arange(len(PV_LEVELS))
    width = 0.11
    for i, method in enumerate(METHODS):
        offset = (i - len(METHODS)/2) * width
        means = aggregated[method]["violation_rate"]
        stds = aggregated[method]["violation_rate_std"]
        ax.bar(x + offset, means, width, yerr=stds, label=method,
               color=METHOD_COLORS[method], edgecolor="white", capsize=2)
    ax.set_xlabel("PV Penetration", fontsize=10)
    ax.set_ylabel("CTI Violation Rate (%)", fontsize=10)
    ax.set_title("Total Violation Rate by Method", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(p*100)}%" for p in PV_LEVELS])
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3, axis="y")
    
    # Right: Severity breakdown for MAPPO-Safe-EWC
    ax = axes[1]
    method = "MAPPO-Safe-EWC"
    mild = aggregated[method]["mild_rate"]
    moderate = aggregated[method]["moderate_rate"]
    severe = aggregated[method]["severe_rate"]
    
    ax.bar(x, severe, width=0.5, label="Severe (<0.1s)", color=SEVERITY_COLORS["severe"], edgecolor="white")
    ax.bar(x, moderate, width=0.5, bottom=severe, label="Moderate (0.1-0.2s)", color=SEVERITY_COLORS["moderate"], edgecolor="white")
    ax.bar(x, mild, width=0.5, bottom=np.array(severe)+np.array(moderate), label="Mild (0.2-0.3s)", color=SEVERITY_COLORS["mild"], edgecolor="white")
    
    ax.set_xlabel("PV Penetration", fontsize=10)
    ax.set_ylabel("Violation Rate (%)", fontsize=10)
    ax.set_title("Severity Breakdown (MAPPO-Safe-EWC)", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(p*100)}%" for p in PV_LEVELS])
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    
    if ALIASED_METHODS:
        fig.text(0.5, -0.02,
                  "* QMIX/IQL: no real implementation in this codebase — both alias IPPO's training",
                  ha="center", fontsize=7, style="italic", color="#888888")

    fig.tight_layout()
    fig.savefig(out_path / "fig06_cti_violations.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", out_path / "fig06_cti_violations.png")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Figure 6 — CTI Violation Rate")
    p.add_argument("--n_seeds", type=int, default=10)
    p.add_argument("--train_steps", type=int, default=5000)
    p.add_argument("--n_eval", type=int, default=100)
    p.add_argument("--output_dir", type=str, default="outputs")
    p.add_argument("--system", type=str, default="8bus")
    args = p.parse_args()
    main(args.n_seeds, args.train_steps, args.n_eval, args.output_dir, args.system)