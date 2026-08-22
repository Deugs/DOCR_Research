
"""
exp_fig42_warm_start.py — Figure 42
─────────────────────────────────────
Warm-Start vs. Cold-Start Initialization.

Compares training from random initialization (cold-start) vs.
warm-start with classical DOCR optimization solution.

Classical solution computed via linear programming or existing solver.

Outputs
-------
  outputs/fig42_warm_start.npz
  outputs/fig42_warm_start.png
"""

import numpy as np
import os
import sys
import logging
import argparse
from pathlib import Path
from typing import Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
os.makedirs("outputs", exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from network.network_data import get_8bus
from marl.marl_env import DynamicDOCREnv
from marl.marl_agents import MAPPOMARLTrainer
from network.relay_physics import check_constraints

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def compute_classical_solution(net, env) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute classical DOCR coordination via simplified optimization.
    Uses a greedy heuristic: minimize TDS while maintaining CTI ≥ 0.3s.
    """
    n_relays = net.n_relays
    TDS = np.full(n_relays, 0.5)
    PSM = np.full(n_relays, 2.0)

    # Greedy refinement: try to reduce TDS while maintaining CTI
    best_of = float('inf')
    best_TDS = TDS.copy()
    best_PSM = PSM.copy()

    # Grid search over reasonable ranges
    tds_candidates = np.linspace(0.1, 1.0, 20)
    psm_candidates = np.linspace(1.0, 5.0, 15)

    for tds in tds_candidates:
        for psm in psm_candidates:
            TDS_test = np.full(n_relays, tds)
            PSM_test = np.full(n_relays, psm)
            env.set_state(TDS_test, PSM_test)

            # Quick evaluation
            c = check_constraints(net, TDS_test, PSM_test)
            if c.get("cti_satisfied", False):
                # Compute objective: sum of operating times
                of = tds * n_relays  # Simplified
                if of < best_of:
                    best_of = of
                    best_TDS = TDS_test.copy()
                    best_PSM = PSM_test.copy()

    logger.info("Classical solution: OF=%.4f, TDS=[%.3f, ...], PSM=[%.2f, ...]",
                best_of, best_TDS[0], best_PSM[0])
    return best_TDS, best_PSM


def make_warm_start_env(net, classical_TDS: np.ndarray, classical_PSM: np.ndarray, **env_kwargs):
    """
    Wrap a fresh DynamicDOCREnv so every reset() starts from the classical
    DOCR solution instead of a random point.

    `DynamicDOCREnv.ACT_DIM` is `[DeltaTMS, DeltaPS]` -- the policy outputs
    per-step *increments* to TMS/PS (see marl/marl_env.py), not absolute
    settings. `SharedGaussianActor` also has no `set_output_bias`/
    `mean_network` attribute anywhere -- both hasattr checks previously
    used here were always False, so warm-start silently fell through to a
    no-op "not supported" warning and trained identically to cold-start
    regardless of the classical solution computed above.

    For a delta/increment-controlled policy, the correct way to warm-start
    is to change *where the environment starts*, not to bias the policy's
    output layer toward an absolute target it was never designed to
    produce directly. `env.reset()` always randomizes TMS/PS via
    `self.rng.uniform(...)` with no initial-state override, so `reset` is
    monkey-patched here to call the env's own `set_state()` (the same
    method `compute_classical_solution` above already uses) right after
    each reset.
    """
    env = DynamicDOCREnv(net, **env_kwargs)
    original_reset = env.reset

    def warm_reset(*args, **kwargs):
        original_reset(*args, **kwargs)
        env.set_state(classical_TDS.copy(), classical_PSM.copy())
        return env._get_obs()

    env.reset = warm_reset
    return env


def train_with_logging(trainer, env, total_steps: int, eval_interval: int = 500,
                       seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Train and log OF at regular intervals."""
    rng = np.random.default_rng(seed)
    steps_log = []
    of_log = []
    cti_viol_log = []

    n_chunks = total_steps // eval_interval
    for chunk in range(n_chunks):
        trainer.train(total_steps=eval_interval, verbose=False)
        cumulative = (chunk + 1) * eval_interval

        # Evaluate
        ofs = []
        cti_viols = []
        for ep in range(5):
            obs = env.reset(seed=seed + ep + chunk * 100)
            done = False
            ep_of = []
            while not done:
                actions = np.zeros((env.n_agents, env.ACT_DIM))
                for i in range(env.n_agents):
                    a, _ = trainer.actor.sample(obs[i], rng)
                    actions[i] = a
                obs, _, done, info = env.step(actions)
                ep_of.append(info.get("OF", float("inf")))
            ofs.append(float(np.min(ep_of)))
            TMS, PS = env.get_state()
            c = check_constraints(env.net, TMS, PS)
            cti_viols.append(c.get("cti_violations", 0))

        steps_log.append(cumulative)
        of_log.append(np.mean(ofs))
        cti_viol_log.append(np.mean(cti_viols))

    return np.array(steps_log), np.array(of_log), np.array(cti_viol_log)


def main(train_steps: int = 10000, n_seeds: int = 3, seed: int = 42,
         output_dir: str = "outputs"):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    net = get_8bus()
    env = DynamicDOCREnv(net, use_pandapower=False, fault_type="3ph", curriculum_stage=5)

    logger.info("Figure 42 — Warm-Start vs. Cold-Start Initialization")
    logger.info("  Training steps: %d, Seeds: %d", train_steps, n_seeds)

    # Compute classical solution
    logger.info("Computing classical DOCR solution...")
    classical_TDS, classical_PSM = compute_classical_solution(net, env)
    env.reset()  # Reset after classical eval

    cold_results = []
    warm_results = []

    for s in range(n_seeds):
        logger.info("\\nSeed %d:", s)

        # ── Cold-start ──────────────────────────────────────────────────────
        logger.info("  Cold-start training...")
        env_cold = DynamicDOCREnv(net, use_pandapower=False, fault_type="3ph", curriculum_stage=5)
        trainer_cold = MAPPOMARLTrainer(env_cold, use_comm=False, seed=seed + s * 100)
        steps_c, of_c, cti_c = train_with_logging(trainer_cold, env_cold, train_steps,
                                                   eval_interval=500, seed=seed + s * 100)
        cold_results.append((steps_c, of_c, cti_c))

        # ── Warm-start ──────────────────────────────────────────────────────
        logger.info("  Warm-start training...")
        env_warm = make_warm_start_env(net, classical_TDS, classical_PSM,
                                        use_pandapower=False, fault_type="3ph", curriculum_stage=5)
        trainer_warm = MAPPOMARLTrainer(env_warm, use_comm=False, seed=seed + s * 100 + 500)
        steps_w, of_w, cti_w = train_with_logging(trainer_warm, env_warm, train_steps,
                                                   eval_interval=500, seed=seed + s * 100 + 500)
        warm_results.append((steps_w, of_w, cti_w))

    # Aggregate
    steps_common = cold_results[0][0]
    of_cold_mat = np.array([np.interp(steps_common, steps, of) for steps, of, _ in cold_results])
    of_warm_mat = np.array([np.interp(steps_common, steps, of) for steps, of, _ in warm_results])
    cti_cold_mat = np.array([np.interp(steps_common, steps, cti) for steps, _, cti in cold_results])
    cti_warm_mat = np.array([np.interp(steps_common, steps, cti) for steps, _, cti in warm_results])

    mean_cold = of_cold_mat.mean(axis=0)
    std_cold = of_cold_mat.std(axis=0)
    mean_warm = of_warm_mat.mean(axis=0)
    std_warm = of_warm_mat.std(axis=0)
    mean_cti_cold = cti_cold_mat.mean(axis=0)
    mean_cti_warm = cti_warm_mat.mean(axis=0)

    # Compute speedup
    final_of_cold = mean_cold[-1]
    final_of_warm = mean_warm[-1]
    threshold_of = final_of_cold + 0.05 * (mean_cold[0] - final_of_cold)

    cold_reach = np.where(mean_cold <= threshold_of)[0]
    warm_reach = np.where(mean_warm <= threshold_of)[0]
    cold_steps = steps_common[cold_reach[0]] if len(cold_reach) > 0 else train_steps
    warm_steps = steps_common[warm_reach[0]] if len(warm_reach) > 0 else train_steps
    speedup = cold_steps / max(warm_steps, 1)

    # ── Save ─────────────────────────────────────────────────────────────────
    np.savez(
        out_path / "fig42_warm_start.npz",
        steps=steps_common,
        cold_mean=mean_cold, cold_std=std_cold,
        warm_mean=mean_warm, warm_std=std_warm,
        cti_cold_mean=mean_cti_cold, cti_warm_mean=mean_cti_warm,
        classical_TDS=classical_TDS, classical_PSM=classical_PSM,
        speedup=np.array([speedup]),
        cold_steps_to_threshold=np.array([cold_steps]),
        warm_steps_to_threshold=np.array([warm_steps]),
    )
    logger.info("Saved: %s", out_path / "fig42_warm_start.npz")

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Figure 42 — Warm-Start vs. Cold-Start Initialization", fontsize=12)

    # Left: OF convergence
    ax = axes[0]
    ax.plot(steps_common, mean_cold, color="#ff6b35", lw=2, label="Cold-start (random init)")
    ax.fill_between(steps_common, mean_cold - std_cold, mean_cold + std_cold,
                    color="#ff6b35", alpha=0.15)
    ax.plot(steps_common, mean_warm, color="#39d353", lw=2, label="Warm-start (classical init)")
    ax.fill_between(steps_common, mean_warm - std_warm, mean_warm + std_warm,
                    color="#39d353", alpha=0.15)

    ax.axhline(threshold_of, color="gray", ls="--", lw=1, alpha=0.5,
               label=f"95% convergence threshold")
    ax.axvline(cold_steps, color="#ff6b35", ls=":", lw=1.5, alpha=0.7)
    ax.axvline(warm_steps, color="#39d353", ls=":", lw=1.5, alpha=0.7)

    ax.set_xlabel("Training Steps", fontsize=10)
    ax.set_ylabel("Objective Function (s)", fontsize=10)
    ax.set_title(f"Convergence Comparison (Speedup: {speedup:.1f}×)", fontsize=11)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)

    # Right: CTI violations comparison
    ax = axes[1]
    ax.plot(steps_common, mean_cti_cold, color="#ff6b35", lw=2, label="Cold-start")
    ax.plot(steps_common, mean_cti_warm, color="#39d353", lw=2, label="Warm-start")
    ax.axhline(0, color="gray", ls="--", lw=1, alpha=0.5)
    ax.set_xlabel("Training Steps", fontsize=10)
    ax.set_ylabel("Mean CTI Violations / Episode", fontsize=10)
    ax.set_title("Early Constraint Violations", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path / "fig42_warm_start.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", out_path / "fig42_warm_start.png")

    # Summary
    print("\\n" + "=" * 60)
    print("WARM-START vs. COLD-START SUMMARY")
    print("=" * 60)
    print(f"Classical solution: TDS={classical_TDS.mean():.3f}, PSM={classical_PSM.mean():.2f}")
    print(f"Cold-start final OF:  {mean_cold[-1]:.4f} ± {std_cold[-1]:.4f}")
    print(f"Warm-start final OF:  {mean_warm[-1]:.4f} ± {std_warm[-1]:.4f}")
    print(f"Cold-start steps to 95%: {cold_steps}")
    print(f"Warm-start steps to 95%: {warm_steps}")
    print(f"Speedup:              {speedup:.1f}×")
    print(f"Early CTI violations (cold): {mean_cti_cold[:5].mean():.2f}")
    print(f"Early CTI violations (warm): {mean_cti_warm[:5].mean():.2f}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Figure 42 — Warm-Start vs. Cold-Start")
    parser.add_argument("--train_steps", type=int, default=10000)
    parser.add_argument("--n_seeds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="outputs")
    args = parser.parse_args()
    main(args.train_steps, args.n_seeds, args.seed, args.output_dir)
