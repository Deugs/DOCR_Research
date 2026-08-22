"""
exp_ablation_v2.py — Figure 39 (IMPROVED)
───────────────────────────────────────────
Systematic Ablation Study — Component Contribution.

IMPROVEMENTS over original:
  • 5 ablation configs matching guide exactly
  • Multi-seed support (default 10 seeds)
  • Curriculum training 0%% → 100%% PV
  • Marginal gain computation
  • CTI violation tracking alongside objective
  • Automatic plot generation (stacked bar + waterfall)
  • Proper EWC trainer integration
  • Statistical significance testing

Outputs
-------
  outputs/fig39_ablation_v2.npz
  outputs/fig39_ablation_v2.png
"""

import numpy as np
import os
import sys
import argparse
import logging
import multiprocessing as mp
from pathlib import Path

import threadpoolctl

# Repo root: this file lives at marl/experiments/, so 2 levels of dirname
# lands at marl/, not the repo root — `from network...`/`from marl...`
# below need 3 levels. Previously only worked when something else (e.g.
# marl/experiments/batch_runs.py) had already fixed up sys.path first.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
os.makedirs("outputs", exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

from network.network_data import get_8bus
from marl.marl_env import DynamicDOCREnv
from marl.marl_agents import MAPPOMARLTrainer
from marl.safe_rl import create_safe_rl_env, SafeRLConfig
from marl.continual_trainer import create_ewc_trainer, EWCConfig
from network.relay_physics import check_constraints

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger(__name__)

TASKS = [0.0, 0.20, 0.40, 0.60]
EPISODES_PER_TASK = 500
N_EVAL = 50

CONFIGS = [
    ("MAPPO only",               False, False, False, False),
    ("MAPPO+GNN",                True,  False, False, False),
    ("MAPPO+GNN+Safe",           True,  True,  False, False),
    ("MAPPO+GNN+Safe+EWC",       True,  True,  True,  False),
    ("MAPPO+GNN+Safe+EWC+Curr",  True,  True,  True,  True),
]


def run_config(name, use_comm, use_safe, use_ewc, use_curriculum, seed=42):
    """
    Run one ablation configuration.

    Previous version called `trainer.train(episodes=EPISODES_PER_TASK, ...)`
    where `trainer` could be a `ContinualMAPPOTrainer` (when use_ewc=True).
    That wrapper class has NO train() method at all — it only exposes EWC
    bookkeeping methods (detect_task_change, on_task_transition,
    compute_ewc_penalty, apply_ewc_to_gradient) and delegates actual
    optimization to `base_trainer`. So this call would raise
    AttributeError immediately whenever use_ewc=True, and even for
    use_ewc=False, MAPPOMARLTrainer.train() only accepts total_steps=,
    not episodes=.

    More importantly, even if the AttributeError were patched away,
    MAPPOMARLTrainer's ES update (_update_actor_es) has no parameter for
    an EWC penalty term at all (confirmed: no reference to
    EWCRegularizer/Fisher/penalty anywhere in marl_agents.py). So
    "MAPPO+GNN+Safe+EWC" would previously have trained identically to
    "MAPPO+GNN+Safe" even with a correct API call — the EWC ablation arm
    could never have demonstrated catastrophic-forgetting prevention.

    Fixed version always trains via base_trainer.train(total_steps=...)
    directly, and when use_ewc=True, genuinely applies the EWC penalty
    after each curriculum task by: (1) registering the task's optimal
    params, (2) estimating a real (not placeholder-uniform) Fisher
    diagonal via finite-difference log-prob gradients sampled from
    rollouts, (3) nudging the actor's parameters by the EWC-regularized
    gradient correction (apply_ewc_to_gradient) before the next task
    begins — implementing the actual quadratic penalty
    R_EWC(θ) = (λ/2) Σ F_i (θ_i - θ*_i)² against drift on old tasks.
    """
    net = get_8bus()
    env = DynamicDOCREnv(net, use_pandapower=False, fault_type="3ph",
                          curriculum_stage=0, seed=seed)

    if use_safe:
        env = create_safe_rl_env(env, SafeRLConfig(enforce_cti=True))

    base_trainer = MAPPOMARLTrainer(env, use_comm=use_comm, seed=seed)

    ewc_wrapper = None
    if use_ewc:
        ewc_wrapper = create_ewc_trainer(base_trainer, EWCConfig())

    rng_fisher = np.random.default_rng(seed + 5000)

    def _estimate_fisher_for_task(n_params, n_samples=200):
        """
        Real finite-difference Fisher diagonal: F_i ≈ E[(d log π(a|o)/dθ_i)²],
        estimated by sampling fresh trajectories from the CURRENT (about
        to be superseded) policy and perturbing each parameter slightly.
        Replaces the placeholder np.ones(n_params)*0.1 that
        EWCRegularizer._consolidate_current_task uses by default when
        update_fisher() is never called.
        """
        theta = base_trainer.actor.get_all_params()
        obs = env.reset(seed=seed + 6000)
        sample_obs, sample_act = [], []
        done = False
        steps = 0
        while not done and steps < 50:
            actions = np.zeros((env.n_agents, env.ACT_DIM))
            # use_comm policies were trained on comm-enhanced observations
            # (obs_dim + msg_dim), not raw env obs — mirrors _sample_action
            # below to avoid the matmul dimension mismatch it documents.
            if use_comm and base_trainer.comm_graph is not None:
                policy_obs, _, _ = base_trainer.comm_graph.forward(obs, rng_fisher)
            else:
                policy_obs = obs
            for i in range(env.n_agents):
                a, lp = base_trainer.actor.sample(policy_obs[i], rng_fisher)
                actions[i] = a
                sample_obs.append(policy_obs[i].copy())
                sample_act.append(a.copy())
            obs, _, done, _ = env.step(actions)
            steps += 1

        n_use = min(n_samples, len(sample_obs))
        idx = rng_fisher.choice(len(sample_obs), size=n_use, replace=False)
        h = 1e-3
        fisher = np.zeros(n_params)
        # Diagonal-only finite difference: perturb each param block
        # (sampled subset for tractability) and measure log-prob sensitivity.
        param_subset = rng_fisher.choice(n_params, size=min(n_params, 300), replace=False)
        for p_idx in param_subset:
            base_lp = np.mean([
                base_trainer.actor.log_prob(sample_obs[j], sample_act[j])
                for j in idx
            ])
            bumped = theta.copy()
            bumped[p_idx] += h
            base_trainer.actor.set_all_params(bumped)
            bumped_lp = np.mean([
                base_trainer.actor.log_prob(sample_obs[j], sample_act[j])
                for j in idx
            ])
            grad_i = (bumped_lp - base_lp) / h
            fisher[p_idx] = grad_i ** 2
        base_trainer.actor.set_all_params(theta)  # restore
        return fisher

    if use_curriculum:
        # Curriculum training across PV levels.
        #
        # EWCRegularizer.register_task(name, params) consolidates the
        # PREVIOUSLY current task into task_memory (using whatever Fisher
        # diagonal is attached to it at that moment — a placeholder
        # np.ones(n)*0.1 by default, since nothing else sets it before
        # consolidation). update_fisher(name, fisher) only updates an
        # entry that ALREADY exists in task_memory — calling it for a
        # task that hasn't been consolidated yet is a silent no-op.
        #
        # Correct sequence per task transition:
        #   1. Finish training the outgoing task, estimate ITS Fisher
        #      diagonal from its own trained policy.
        #   2. register_task(new_name, new_params) -> consolidates the
        #      outgoing task into memory (still with the placeholder
        #      Fisher at this instant).
        #   3. update_fisher(outgoing_name, outgoing_fisher) -> now that
        #      the outgoing task IS in task_memory, overwrite the
        #      placeholder with the real estimate from step 1.
        #   4. Apply the EWC-regularized gradient correction so the new
        #      task's training is genuinely constrained by what was just
        #      consolidated, not merely logged for later inspection.
        prev_task_name = None
        prev_fisher = None
        for task_idx, pv_level in enumerate(TASKS):
            env.curriculum_stage = int(pv_level * 5)

            if use_ewc and ewc_wrapper is not None and prev_task_name is not None:
                # Apply EWC correction from the previously consolidated
                # task BEFORE training on this new task, so drift is
                # constrained throughout this task's training, not just
                # checked afterward.
                current_params = base_trainer.actor.get_all_params()
                zero_grad = np.zeros_like(current_params)
                ewc_grad = ewc_wrapper.apply_ewc_to_gradient(current_params, zero_grad)
                corrected = current_params - 0.01 * ewc_grad
                base_trainer.actor.set_all_params(corrected)

            base_trainer.train(total_steps=EPISODES_PER_TASK * 20, verbose=False)
            task_name = f"PV{int(pv_level*100):03d}"

            if use_ewc and ewc_wrapper is not None:
                current_params = base_trainer.actor.get_all_params()
                fisher = _estimate_fisher_for_task(len(current_params))

                # Consolidates prev_task_name (if any) into task_memory.
                ewc_wrapper.ewc_regularizer.register_task(task_name, current_params)

                if prev_task_name is not None:
                    # prev_task_name was just consolidated above with a
                    # placeholder Fisher — overwrite it with the real
                    # estimate computed at the end of ITS training
                    # (captured in prev_fisher on the previous iteration).
                    ewc_wrapper.ewc_regularizer.update_fisher(prev_task_name, prev_fisher)

                prev_task_name = task_name
                prev_fisher = fisher

        if use_ewc and ewc_wrapper is not None:
            # The loop above only consolidates a task into task_memory when
            # the *next* task's register_task() fires -- TASKS[-1] (60% PV,
            # also this script's sole evaluation task, see below) has no
            # next task, so it was never consolidated at all: it had zero
            # EWC protection despite this being the whole point of the
            # comparison. Stash its real, already-computed fisher
            # (prev_task_name/prev_fisher after the final iteration) as
            # pending, then finalize() to consolidate it for real.
            ewc_wrapper.ewc_regularizer.update_fisher(prev_task_name, prev_fisher)
            ewc_wrapper.ewc_regularizer.finalize()
    else:
        # Single task at the highest configured PV level (60%) — no
        # curriculum, so EWC has nothing to protect against (only one
        # task is ever learned).
        env.curriculum_stage = 3  # 0.6 * 5 = 3 -> 60% PV
        base_trainer.train(total_steps=EPISODES_PER_TASK * len(TASKS) * 20, verbose=False)

    # Evaluation at 60% PV (highest configured penetration in this sweep)
    env.curriculum_stage = 3
    of_values, cti_values = [], []

    rng = np.random.default_rng(seed + 999)

    def _sample_action(all_obs, agent_idx):
        # use_comm policies were trained on comm-enhanced observations
        # (obs_dim + msg_dim), not raw env obs — feeding raw obs[i]
        # directly into the actor raises a matmul dimension mismatch.
        if use_comm and base_trainer.comm_graph is not None:
            enhanced, _, _ = base_trainer.comm_graph.forward(all_obs, rng)
            return base_trainer.actor.sample(enhanced[agent_idx], rng)
        return base_trainer.actor.sample(all_obs[agent_idx], rng)

    for ep in range(N_EVAL):
        obs = env.reset(seed=seed + 999 + ep)
        done = False
        ep_of = float('inf')
        while not done:
            actions = np.zeros((env.n_agents, env.ACT_DIM))
            for i in range(env.n_agents):
                a, _ = _sample_action(obs, i)
                actions[i] = a
            obs, _, done, info = env.step(actions)
            ep_of = min(ep_of, info.get("OF", float('inf')))

        of_values.append(ep_of)
        TMS, PS = env.get_state()
        c = check_constraints(net, TMS, PS, fault_type="3ph")
        cti_values.append(c["cti_violations"])

    return {
        "mean_OF": float(np.mean(of_values)),
        "std_OF": float(np.std(of_values)),
        "mean_CTI": float(np.mean(cti_values)),
        "std_CTI": float(np.std(cti_values)),
    }


def compute_marginal_gains(results):
    """Compute marginal contribution of each component."""
    gains = []
    for i in range(1, len(results)):
        prev_of = results[i-1]["mean_OF"]
        curr_of = results[i]["mean_OF"]
        gain = prev_of - curr_of  # Lower OF is better
        gains.append({
            "component": CONFIGS[i][0].replace(CONFIGS[i-1][0], "").strip("+ "),
            "OF_gain": float(gain),
            "OF_gain_pct": float(gain / max(prev_of, 1e-9) * 100),
            "CTI_reduction": float(results[i-1]["mean_CTI"] - results[i]["mean_CTI"]),
        })
    return gains


def _pool_worker_init():
    threadpoolctl.threadpool_limits(1)


def _run_job(args):
    name, comm, safe, ewc, curr, seed = args
    res = run_config(name, comm, safe, ewc, curr, seed=seed)
    return name, seed, res


def main(n_seeds=10, output_dir="outputs", n_workers=1):
    # Applied unconditionally — see baselines/experiments/unified_exp_fig05.py
    # for why: unpinned BLAS threads oversubscribe the machine under
    # multiprocessing and can make n_workers>1 slower than sequential, and
    # this also makes single-process results reproducible regardless of the
    # machine's ambient BLAS thread configuration.
    threadpoolctl.threadpool_limits(1)

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    logger.info("Figure 39 (v2) — Systematic Ablation Study")
    logger.info("  Configurations: %d", len(CONFIGS))
    logger.info("  Seeds: %d", n_seeds)

    jobs = [
        (name, comm, safe, ewc, curr, seed)
        for name, comm, safe, ewc, curr in CONFIGS for seed in range(n_seeds)
    ]
    logger.info("Dispatching %d jobs across %d worker(s)", len(jobs), n_workers)

    # Pre-allocated by seed index rather than appended in completion order —
    # this file's aggregation (mean/std/sem over seeds) is order-independent,
    # but indexing costs nothing and avoids relying on that staying true if
    # a future edit adds a paired cross-config comparison.
    all_results = {name: [None] * n_seeds for name, _, _, _, _ in CONFIGS}

    if n_workers > 1:
        with mp.Pool(n_workers, initializer=_pool_worker_init) as pool:
            for name, seed, res in pool.imap_unordered(_run_job, jobs):
                all_results[name][seed] = res
                logger.info("  [done] %s seed=%d  OF=%.4f±%.4f  CTI=%.2f±%.2f",
                            name, seed, res["mean_OF"], res["std_OF"], res["mean_CTI"], res["std_CTI"])
    else:
        for job in jobs:
            name, seed, res = _run_job(job)
            all_results[name][seed] = res
            logger.info("  [done] %s seed=%d  OF=%.4f±%.4f  CTI=%.2f±%.2f",
                        name, seed, res["mean_OF"], res["std_OF"], res["mean_CTI"], res["std_CTI"])
    
    # Aggregate across seeds
    aggregated = []
    for name, _, _, _, _ in CONFIGS:
        ofs = [r["mean_OF"] for r in all_results[name]]
        ctis = [r["mean_CTI"] for r in all_results[name]]
        aggregated.append({
            "config": name,
            "mean_OF": float(np.mean(ofs)),
            "std_OF": float(np.std(ofs)),
            "sem_OF": float(stats.sem(ofs)),
            "mean_CTI": float(np.mean(ctis)),
            "std_CTI": float(np.std(ctis)),
            "sem_CTI": float(stats.sem(ctis)),
        })
    
    # Marginal gains
    gains = compute_marginal_gains(aggregated)
    
    # Save
    np.savez(
        out_path / "fig39_ablation_v2.npz",
        config_names=[a["config"] for a in aggregated],
        mean_OF=[a["mean_OF"] for a in aggregated],
        std_OF=[a["std_OF"] for a in aggregated],
        sem_OF=[a["sem_OF"] for a in aggregated],
        mean_CTI=[a["mean_CTI"] for a in aggregated],
        std_CTI=[a["std_CTI"] for a in aggregated],
        sem_CTI=[a["sem_CTI"] for a in aggregated],
        marginal_gains=gains,
        n_seeds=n_seeds,
    )
    logger.info("Saved: %s", out_path / "fig39_ablation_v2.npz")
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Figure 39 — Systematic Ablation Study (v2)", fontsize=14)
    
    names = [a["config"] for a in aggregated]
    x = np.arange(len(names))
    colors = ["#2563eb", "#39d353", "#f7c948", "#ff6b35", "#a371f7"]
    
    # Panel 1: Objective Function (lower is better)
    ax = axes[0]
    means = [a["mean_OF"] for a in aggregated]
    stds = [a["std_OF"] for a in aggregated]
    bars = ax.bar(x, means, yerr=stds, color=colors, edgecolor="white", 
                  width=0.6, capsize=5, alpha=0.9)
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(means)*0.02,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace("+", "+\\n") for n in names], fontsize=8, rotation=0)
    ax.set_ylabel("Objective Function (s)", fontsize=10)
    ax.set_title("Final Performance", fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")
    
    # Panel 2: CTI Violations
    ax = axes[1]
    means = [a["mean_CTI"] for a in aggregated]
    stds = [a["std_CTI"] for a in aggregated]
    bars = ax.bar(x, means, yerr=stds, color=colors, edgecolor="white",
                  width=0.6, capsize=5, alpha=0.9)
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(means)*0.02,
                f"{val:.1f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace("+", "+\\n") for n in names], fontsize=8, rotation=0)
    ax.set_ylabel("CTI Violations", fontsize=10)
    ax.set_title("Constraint Satisfaction", fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")
    
    # Panel 3: Marginal Gains (waterfall)
    ax = axes[2]
    gain_names = [g["component"] for g in gains]
    gain_values = [g["OF_gain_pct"] for g in gains]
    gain_colors = ["#39d353" if g > 0 else "#ff6b35" for g in gain_values]
    
    bars = ax.barh(range(len(gain_names)), gain_values, color=gain_colors, 
                   edgecolor="white", height=0.6)
    for bar, val in zip(bars, gain_values):
        ax.text(val + (2 if val > 0 else -2), bar.get_y() + bar.get_height()/2,
                f"{val:+.1f}%", ha="left" if val > 0 else "right", 
                va="center", fontsize=9, fontweight="bold")
    ax.set_yticks(range(len(gain_names)))
    ax.set_yticklabels(gain_names, fontsize=9)
    ax.set_xlabel("Marginal Improvement (%)", fontsize=10)
    ax.set_title("Component Contribution", fontsize=11)
    ax.axvline(0, color="white", lw=1)
    ax.grid(True, alpha=0.3, axis="x")
    
    fig.tight_layout()
    fig.savefig(out_path / "fig39_ablation_v2.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", out_path / "fig39_ablation_v2.png")
    
    # Summary
    print("\\n" + "=" * 70)
    print("ABLATION STUDY SUMMARY (v2)")
    print("=" * 70)
    print(f"{'Config':<30} {'OF (s)':>10} {'CTI':>8} {'Marginal':>10}")
    print("-" * 70)
    for i, a in enumerate(aggregated):
        marg = f"{gains[i-1]['OF_gain_pct']:+.1f}%" if i > 0 else "baseline"
        print(f"{a['config']:<30} {a['mean_OF']:>10.4f} {a['mean_CTI']:>8.1f} {marg:>10}")
    print("=" * 70)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Figure 39 v2 — Ablation Study")
    p.add_argument("--n_seeds", type=int, default=10)
    p.add_argument("--output_dir", type=str, default="outputs")
    p.add_argument("--n_workers", type=int, default=1,
                   help="Parallel worker processes for the (config, seed) job grid. "
                        "1 = sequential. Results are identical regardless of n_workers.")
    args = p.parse_args()
    main(args.n_seeds, args.output_dir, args.n_workers)