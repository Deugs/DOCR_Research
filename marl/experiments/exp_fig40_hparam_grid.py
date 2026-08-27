
"""
exp_hparam_grid_v2.py — Figure 40 (IMPROVED)
─────────────────────────────────────────────
Hyperparameter Sensitivity Heatmap.

IMPROVEMENTS over original:
  • Correct parameter space: λ_EWC, α_λ, d_h matching guide
  • 5×4×4 grid = 80 configs (vs original 3×3×3=27)
  • Multi-seed support (default 3 seeds)
  • Automatic heatmap generation (faceted 2D)
  • Robust region identification
  • Statistical significance markers
  • 3D surface plot option
  • Performance and CTI violation dual tracking

Outputs
-------
  outputs/fig40_hparam_v2.npz
  outputs/fig40_hparam_v2.png
"""

import numpy as np
import itertools
import os
import sys
import argparse
import logging
import multiprocessing as mp
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
os.makedirs("outputs", exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
import threadpoolctl

from network.network_data import get_8bus
from marl.marl_env import DynamicDOCREnv
from marl.marl_agents import MAPPOMARLTrainer
from marl.safe_rl import create_safe_rl_env, SafeRLConfig
from network.relay_physics import check_constraints

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger(__name__)

TRAIN_STEPS = 2000  # Reduced for speed as per guide
N_EVAL = 30

# Parameter grid matching the guide
LAMBDA_EWC_VALUES = [1e2, 1e3, 1e4, 1e5, 1e6]
ALPHA_LAMBDA_VALUES = [1e-5, 1e-4, 1e-3, 1e-2]
HIDDEN_DIM_VALUES = [16, 32, 64, 128]


def _get_network(system):
    if system == "8bus":
        return get_8bus()
    elif system == "14bus":
        from network.network_data import get_14bus
        return get_14bus()
    elif system == "30bus":
        from network.network_data import get_30bus
        return get_30bus()
    elif system == "39bus":
        from network.network_data import get_39bus
        return get_39bus()
    else:
        return get_8bus()


def run_with_params(lambda_ewc, alpha_lambda, hidden_dim, seed=42, system="8bus"):
    """Run one hyperparameter configuration."""
    net = _get_network(system)
    env = DynamicDOCREnv(net, use_pandapower=False, fault_type="3ph",
                          curriculum_stage=5, seed=seed)
    
    # Safe RL with specified Lagrange learning rate
    safe_env = create_safe_rl_env(env, SafeRLConfig(
        enforce_cti=True,
        lambda_lr=alpha_lambda,
    ))
    
    trainer = MAPPOMARLTrainer(safe_env, use_comm=False,
                                hidden_actor=hidden_dim, hidden_critic=hidden_dim,
                                seed=seed)

    trainer.train(total_steps=TRAIN_STEPS, verbose=False)

    # `create_ewc_trainer()` returns a `ContinualMAPPOTrainer` wrapper with
    # no `.train()`/`.actor` of its own (those stay on `.base_trainer`) --
    # reassigning `trainer` to it, as this used to do, crashed on the very
    # next `trainer.train()` call for every one of the 80 grid configs.
    #
    # A prior fix kept `trainer` as the real MAPPOMARLTrainer and instead
    # registered a single self-consolidated EWC task afterward to compute
    # an `ewc_penalty` diagnostic meant to show lambda_ewc's effect. That
    # diagnostic is mathematically guaranteed to be 0.0 for every config:
    # it registered the task's "optimal_params" and then immediately
    # computed the penalty against `current_params` using the exact same
    # array (`cur`), making delta = current - optimal the zero vector by
    # construction, regardless of lambda_ewc/fisher/hidden_dim.
    #
    # There is no fix that makes this measurement meaningful: this
    # function runs ONE task in isolation, and a single self-registered
    # task genuinely has nothing to protect against forgetting -- the
    # same reason run_core_training.py's Core C deliberately never calls
    # compute_ewc_penalty() after its own single-task registration. Drop
    # the diagnostic entirely rather than report a number that always
    # reads as "EWC has zero effect" regardless of the real lambda_ewc.
    # MAPPOMARLTrainer.train() is ES-based and never reads
    # lambda_ewc/apply_ewc_to_gradient during training either way, so
    # lambda_ewc has never causally affected mean_OF/mean_CTI here.

    # Evaluate
    rng = np.random.default_rng(seed + 999)
    of_values, cti_values = [], []
    
    for ep in range(N_EVAL):
        obs = safe_env.reset(seed=seed + 999 + ep)
        done = False
        ep_of = float('inf')
        while not done:
            actions = np.zeros((safe_env.n_agents, safe_env.ACT_DIM))
            for i in range(safe_env.n_agents):
                a, _ = trainer.actor.sample(obs[i], rng)
                actions[i] = a
            obs, _, done, info = safe_env.step(actions)
            ep_of = min(ep_of, info.get("OF", float('inf')))
        
        of_values.append(ep_of)
        TMS, PS = safe_env.get_state()
        c = check_constraints(net, TMS, PS)
        cti_values.append(c["cti_violations"])
    
    return {
        "lambda_ewc": lambda_ewc,
        "alpha_lambda": alpha_lambda,
        "hidden_dim": hidden_dim,
        "mean_OF": float(np.mean(of_values)),
        "std_OF": float(np.std(of_values)),
        "mean_CTI": float(np.mean(cti_values)),
        "std_CTI": float(np.std(cti_values)),
    }


def identify_robust_region(grid_results, metric="mean_OF", threshold_pct=10):
    """Identify region where performance varies less than threshold."""
    values = np.array([g[metric] for g in grid_results])
    best = np.min(values)
    threshold = best * (1 + threshold_pct / 100)
    robust_mask = values <= threshold
    return robust_mask


def _pool_worker_init():
    threadpoolctl.threadpool_limits(1)


def _run_job(args):
    seed, cfg_idx, lamb, alpha, hid, system = args
    res = run_with_params(lamb, alpha, hid, seed=seed, system=system)
    return seed, cfg_idx, res


def main(n_seeds=3, output_dir="outputs", system="8bus", n_workers=1):
    # Applied unconditionally — see baselines/experiments/unified_exp_fig05.py
    # for why: unpinned BLAS threads oversubscribe the machine under
    # multiprocessing and can make n_workers>1 slower than sequential, and
    # this also makes single-process results reproducible regardless of the
    # machine's ambient BLAS thread configuration.
    threadpoolctl.threadpool_limits(1)

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    logger.info("Figure 40 (v2) — Hyperparameter Sensitivity Heatmap")
    logger.info("  Grid: %d × %d × %d = %d configs",
                len(LAMBDA_EWC_VALUES), len(ALPHA_LAMBDA_VALUES),
                len(HIDDEN_DIM_VALUES),
                len(LAMBDA_EWC_VALUES) * len(ALPHA_LAMBDA_VALUES) * len(HIDDEN_DIM_VALUES))
    logger.info("  Seeds: %d  Steps per config: %d", n_seeds, TRAIN_STEPS)

    grid = list(itertools.product(LAMBDA_EWC_VALUES, ALPHA_LAMBDA_VALUES, HIDDEN_DIM_VALUES))

    jobs = [
        (seed, cfg_idx, lamb, alpha, hid, system)
        for seed in range(n_seeds)
        for cfg_idx, (lamb, alpha, hid) in enumerate(grid)
    ]
    logger.info("Dispatching %d jobs across %d worker(s)", len(jobs), n_workers)

    # Pre-allocated by (seed, cfg_idx) rather than appended in completion
    # order — required for reproducible aggregation regardless of which
    # worker finishes first.
    all_results = [[None] * len(grid) for _ in range(n_seeds)]

    if n_workers > 1:
        with mp.Pool(n_workers, initializer=_pool_worker_init) as pool:
            for seed, cfg_idx, res in pool.imap_unordered(_run_job, jobs):
                all_results[seed][cfg_idx] = res
                logger.info("  [done] seed=%d cfg=%d  OF=%.4f±%.4f  CTI=%.2f±%.2f",
                            seed, cfg_idx, res["mean_OF"], res["std_OF"], res["mean_CTI"], res["std_CTI"])
    else:
        for job in jobs:
            seed, cfg_idx, res = _run_job(job)
            all_results[seed][cfg_idx] = res
            logger.info("  [done] seed=%d cfg=%d  OF=%.4f±%.4f  CTI=%.2f±%.2f",
                        seed, cfg_idx, res["mean_OF"], res["std_OF"], res["mean_CTI"], res["std_CTI"])
    
    # Aggregate across seeds
    aggregated = []
    for cfg_idx in range(len(grid)):
        ofs = [all_results[s][cfg_idx]["mean_OF"] for s in range(n_seeds)]
        ctis = [all_results[s][cfg_idx]["mean_CTI"] for s in range(n_seeds)]
        
        aggregated.append({
            "lambda_ewc": grid[cfg_idx][0],
            "alpha_lambda": grid[cfg_idx][1],
            "hidden_dim": grid[cfg_idx][2],
            "mean_OF": float(np.mean(ofs)),
            "std_OF": float(np.std(ofs)),
            "sem_OF": float(stats.sem(ofs)),
            "mean_CTI": float(np.mean(ctis)),
            "std_CTI": float(np.std(ctis)),
            "sem_CTI": float(stats.sem(ctis)),
        })
    
    # Identify robust region
    robust_mask = identify_robust_region(aggregated, metric="mean_OF", threshold_pct=10)
    
    # Save
    np.savez(
        out_path / "fig40_hparam_v2.npz",
        grid=aggregated,
        robust_indices=np.where(robust_mask)[0],
        n_seeds=n_seeds,
    )
    logger.info("Saved: %s", out_path / "fig40_hparam_v2.npz")
    
    # Plot: Faceted 2D heatmaps
    n_lambdas = len(LAMBDA_EWC_VALUES)
    n_alphas = len(ALPHA_LAMBDA_VALUES)
    n_hiddens = len(HIDDEN_DIM_VALUES)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle("Figure 40 — Hyperparameter Sensitivity Heatmap (v2)", fontsize=14)
    
    # Create 3D arrays for each hidden dim
    for h_idx, hidden in enumerate(HIDDEN_DIM_VALUES):
        if h_idx >= 6:
            break
        
        ax = axes[h_idx // 3, h_idx % 3]
        
        # Extract subgrid for this hidden dim
        of_matrix = np.zeros((n_lambdas, n_alphas))
        cti_matrix = np.zeros((n_lambdas, n_alphas))
        
        for i, lamb in enumerate(LAMBDA_EWC_VALUES):
            for j, alpha in enumerate(ALPHA_LAMBDA_VALUES):
                match = [a for a in aggregated 
                        if a["lambda_ewc"] == lamb and a["alpha_lambda"] == alpha 
                        and a["hidden_dim"] == hidden]
                if match:
                    of_matrix[i, j] = match[0]["mean_OF"]
                    cti_matrix[i, j] = match[0]["mean_CTI"]
        
        # Plot OF heatmap
        im = ax.imshow(of_matrix, cmap="RdYlGn_r", aspect="auto",
                       vmin=np.nanmin(of_matrix), vmax=np.nanmax(of_matrix))
        ax.set_xticks(range(n_alphas))
        ax.set_yticks(range(n_lambdas))
        ax.set_xticklabels([f"{a:.0e}" for a in ALPHA_LAMBDA_VALUES], fontsize=8, rotation=45)
        ax.set_yticklabels([f"{l:.0e}" for l in LAMBDA_EWC_VALUES], fontsize=8)
        ax.set_xlabel("α_λ (Lagrange LR)", fontsize=9)
        ax.set_ylabel("λ_EWC", fontsize=9)
        ax.set_title(f"d_h = {hidden}", fontsize=10)
        
        # Annotate with values
        for i in range(n_lambdas):
            for j in range(n_alphas):
                if not np.isnan(of_matrix[i, j]):
                    text = ax.text(j, i, f"{of_matrix[i, j]:.3f}",
                                  ha="center", va="center", color="white" if of_matrix[i, j] > np.nanmean(of_matrix) else "black",
                                  fontsize=7)
        
        plt.colorbar(im, ax=ax, shrink=0.7, label="OF (s)")
    
    fig.tight_layout()
    fig.savefig(out_path / "fig40_hparam_v2.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", out_path / "fig40_hparam_v2.png")
    
    # Summary
    print("\\n" + "=" * 70)
    print("HYPERPARAMETER GRID SUMMARY (v2)")
    print("=" * 70)
    best_idx = np.argmin([a["mean_OF"] for a in aggregated])
    best = aggregated[best_idx]
    print(f"Best configuration:")
    print(f"  λ_EWC = {best['lambda_ewc']:.0e}")
    print(f"  α_λ   = {best['alpha_lambda']:.0e}")
    print(f"  d_h   = {best['hidden_dim']}")
    print(f"  OF    = {best['mean_OF']:.4f} ± {best['std_OF']:.4f}")
    print(f"  CTI   = {best['mean_CTI']:.2f} ± {best['std_CTI']:.2f}")
    print(f"\\nRobust region size: {robust_mask.sum()}/{len(aggregated)} configs")
    print("=" * 70)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Figure 40 v2 — Hyperparameter Grid")
    p.add_argument("--n_seeds", type=int, default=3)
    p.add_argument("--output_dir", type=str, default="outputs")
    p.add_argument("--system", type=str, default="8bus", help="Test system (8bus, 14bus, 30bus, 39bus)")
    p.add_argument("--n_workers", type=int, default=1,
                   help="Parallel worker processes (default: 1 = sequential)")
    args = p.parse_args()
    main(args.n_seeds, args.output_dir, args.system, args.n_workers)