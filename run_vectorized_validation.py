#!/usr/bin/env python3
"""
run_vectorized_validation.py
──────────────────────────────
Validation run for the GPU-native vectorized-environment redesign (Phases
V1-V10 -- see elegant-rolling-map.md). NOT a replacement for the real Core
A-K training pipeline (`run_core_training.py`) or its figures
(`run_post_processing.py`) -- those continue to use the original,
untouched single-environment path and are unaffected by anything here.

What this script is and is not
-------------------------------
This trains `VectorizedMAPPOTrainer` (Phase V9) on each requested system
and compares it against the existing single-env `MAPPOMARLTrainer` as a
baseline, producing plots and a JSON results file. It exists to answer one
question: does the vectorized path actually train (OF converging, CTI
constraints getting satisfied) and land in the same ballpark as the
single-env path on a short run -- not to reproduce the thesis's full
results.

It deliberately does NOT attempt full parity with the real Core A-K suite:
- No GNN communication (`use_comm`) -- VectorizedMAPPOTrainer doesn't
  support it yet (see that module's docstring).
- No Safe RL (Augmented Lagrangian / action projection) -- no
  VectorizedSafeRLWrapper exists yet (Phase V8 built the batched
  primitives, not the wrapper).
- No curriculum learning or EWC/continual learning.
- Only 8/14/39-bus -- VectorizedDynamicDOCREnv requires real per-branch
  impedance data for its power-flow solver (see network/vectorized_power_flow.py).
- Short training budgets by default (thousands of steps, not the Core
  suite's tens of millions) -- this is a correctness/sanity check, not a
  publication-quality training run.
- CPU only on this dev machine (no CUDA-enabled torch build here --
  `--device cuda` will raise if run where one isn't available). The
  vectorized path's actual GPU throughput case is Phase V10's job
  (`scripts/benchmark_vectorized_env.py`), not this script's.

Output
------
- `results/vectorized_validation/<system>_results.json` -- per-system
  training history (OF, CTI satisfaction rate, actor/critic loss per
  rollout) for both trainers, plus final evaluated (OF, CTI-satisfied,
  mean CTI margin, mean primary operating time) via the exact
  `check_constraints`/`objective_function` calls `run_core_training.py`
  itself uses to evaluate a trained policy, so the numbers are
  apples-to-apples comparable to a real Core run's own final-policy metric.
- `results/figures/vectorized_validation/<system>_validation.png` (+ .pdf)
  -- a 4-panel comparison plot (OF convergence, CTI satisfaction rate,
  critic loss, mean reward) for the vectorized vs. single-env trainer,
  using this project's existing matplotlib conventions
  (run_post_processing.py's color palette / figsize / dpi / save-both-
  pdf-and-png pattern).

Usage
-----
    python run_vectorized_validation.py                       # all 3 systems, CPU
    python run_vectorized_validation.py --systems 8bus 39bus
    python run_vectorized_validation.py --n-envs 256 --total-steps 20000
    python run_vectorized_validation.py --device cuda          # on a CUDA-enabled machine
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from network.network_data import get_8bus, get_14bus, get_39bus
from network.relay_physics import check_constraints, objective_function
from marl.marl_env import DynamicDOCREnv
from marl.marl_agents import MAPPOMARLTrainer
from marl.vectorized_marl_env import VectorizedDynamicDOCREnv
from marl.vectorized_marl_agents import VectorizedMAPPOTrainer

SYSTEMS = {"8bus": get_8bus, "14bus": get_14bus, "39bus": get_39bus}

# Matches run_post_processing.py's existing palette (see save_fig()/figure
# code there) -- kept consistent with the thesis's real figures rather than
# introducing a new one for this validation-only script.
COLOR_VECTORIZED = "#2563eb"  # primary blue
COLOR_SINGLE_ENV = "#ff6b35"  # orange
FIGSIZE = (14, 10)
DPI = 200

RESULTS_DIR = Path("results/vectorized_validation")
FIGURES_DIR = Path("results/figures/vectorized_validation")


def save_fig(fig, path: Path, dpi: int = DPI):
    """Matches run_post_processing.py's save_fig(): saves both .pdf and
    .png, bbox_inches='tight'."""
    path.parent.mkdir(parents=True, exist_ok=True)
    stem = path.with_suffix("")
    for ext in (".pdf", ".png"):
        fig.savefig(stem.with_suffix(ext), dpi=dpi, bbox_inches="tight")


def evaluate_final_policy(net, TMS: np.ndarray, PS: np.ndarray, fault_type: str = "3ph") -> dict:
    """Exactly `run_core_training.py`'s own final-policy evaluation pattern
    (see e.g. its line ~1976-1982), reused here so the vectorized and
    single-env trainers' final results are evaluated identically."""
    final_constraints = check_constraints(net, TMS, PS, fault_type=fault_type)
    of_val, _, _ = objective_function(net, TMS, PS, fault_type=fault_type)
    return {
        "OF": float(of_val),
        "cti_satisfied": bool(final_constraints["cti_satisfied"]),
        "cti_min": float(final_constraints["cti_min"]),
        "delta_cti_mean": float(np.mean(final_constraints["delta_cti"])),
        "top_pri_mean": float(final_constraints["top_pri_mean"]),
    }


def run_single_env_baseline(system: str, seed: int, total_steps: int,
                             rollout_len: int, hidden_actor: int, hidden_critic: int) -> dict:
    net = SYSTEMS[system]()
    env = DynamicDOCREnv(net, use_pandapower=False, seed=seed)
    trainer = MAPPOMARLTrainer(env, hidden_actor=hidden_actor, hidden_critic=hidden_critic,
                                rollout_len=rollout_len, n_epochs=4, seed=seed)
    result = trainer.train(total_steps=total_steps, verbose=False)
    final_eval = evaluate_final_policy(net, result["TMS"], result["PS"], env.fault_type)
    return {
        "method": "MAPPO (single-env)",
        "history": result["history"],
        "cti_satisfaction_history": result["cti_satisfaction_history"],
        "elapsed": result["elapsed"],
        "final_eval": final_eval,
    }


def run_vectorized(system: str, seed: int, total_steps: int, rollout_len: int,
                    n_envs: int, hidden_actor: int, hidden_critic: int, device) -> dict:
    net = SYSTEMS[system]()
    env = VectorizedDynamicDOCREnv(net, n_envs=n_envs, seed=seed, device=device)
    trainer = VectorizedMAPPOTrainer(env, hidden_actor=hidden_actor, hidden_critic=hidden_critic,
                                      rollout_len=rollout_len, n_epochs=4, seed=seed, device=device)
    result = trainer.train(total_steps=total_steps, verbose=False)
    final_eval = evaluate_final_policy(net, result["TMS"], result["PS"], env.fault_type)
    return {
        "method": "VectorizedMAPPO",
        "n_envs": n_envs,
        "history": result["history"],
        "cti_satisfaction_history": result["cti_satisfaction_history"],
        "critic_loss_history": result["critic_loss_history"],
        "mean_reward_history": result["mean_reward_history"],
        "elapsed": result["elapsed"],
        "final_eval": final_eval,
    }


def make_comparison_plot(system: str, single_env: dict, vectorized: dict,
                          n_envs: int, rollout_len: int, same_total_steps: bool) -> plt.Figure:
    """The fair x-axis depends on which quantity main() actually matched
    between the two trainers:
    - Default mode matches the NUMBER OF ROLLOUT UPDATES (same number of
      Adam steps each) -- real env steps consumed deliberately differ by
      ~n_envs, so plotting against real steps would misleadingly stretch
      the vectorized curve across a much wider x-range for no comparability
      benefit; rollout-update index is the actually-matched, comparable axis.
    - --same-total-steps mode matches real env steps instead (and gives the
      vectorized trainer far fewer updates on purpose) -- there, real env
      steps consumed is the fair/matched axis and update index would not be.
    """
    if same_total_steps:
        x_label = "Real environment steps consumed"
        x_single = rollout_len * np.arange(1, len(single_env["history"]) + 1)
        x_vec = rollout_len * n_envs * np.arange(1, len(vectorized["history"]) + 1)
    else:
        x_label = "Rollout update"
        x_single = np.arange(1, len(single_env["history"]) + 1)
        x_vec = np.arange(1, len(vectorized["history"]) + 1)
    steps_single, steps_vec = x_single, x_vec

    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE)

    ax = axes[0, 0]
    ax.plot(steps_single, single_env["history"], "o-", color=COLOR_SINGLE_ENV, lw=2, markersize=4,
            label="MAPPO (single-env)")
    ax.plot(steps_vec, vectorized["history"], "o-", color=COLOR_VECTORIZED, lw=2, markersize=4,
            label=f"VectorizedMAPPO (n_envs={n_envs})")
    ax.set_xlabel(x_label)
    ax.set_ylabel("Objective Function (OF)")
    ax.set_title(f"{system}: OF convergence")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.plot(steps_single, single_env["cti_satisfaction_history"], "o-", color=COLOR_SINGLE_ENV, lw=2,
            markersize=4, label="MAPPO (single-env)")
    ax.plot(steps_vec, vectorized["cti_satisfaction_history"], "o-", color=COLOR_VECTORIZED, lw=2,
            markersize=4, label=f"VectorizedMAPPO (n_envs={n_envs})")
    ax.set_xlabel(x_label)
    ax.set_ylabel("CTI satisfaction rate")
    ax.set_title(f"{system}: CTI constraint satisfaction")
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ax.plot(steps_vec, vectorized["critic_loss_history"], "o-", color=COLOR_VECTORIZED, lw=2, markersize=4)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Critic loss (MSE)")
    ax.set_title(f"{system}: VectorizedMAPPO critic loss")
    ax.set_yscale("log")
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.plot(steps_vec, vectorized["mean_reward_history"], "o-", color=COLOR_VECTORIZED, lw=2, markersize=4)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Mean reward")
    ax.set_title(f"{system}: VectorizedMAPPO mean reward")
    ax.grid(alpha=0.3)

    fig.suptitle(
        f"Vectorized-env validation -- {system}  "
        f"(final OF: single-env={single_env['final_eval']['OF']:.3f} "
        f"[{len(single_env['history'])} updates] vs "
        f"vectorized={vectorized['final_eval']['OF']:.3f} "
        f"[{len(vectorized['history'])} updates, n_envs={n_envs}])",
        fontsize=11,
    )
    fig.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--systems", nargs="+", choices=list(SYSTEMS), default=list(SYSTEMS),
                         help="Systems to validate (default: all three -- 8bus/14bus/39bus, "
                              "the only ones VectorizedDynamicDOCREnv supports)")
    parser.add_argument("--n-envs", type=int, default=64,
                         help="Number of parallel environments for the vectorized trainer (default: 64)")
    parser.add_argument("--total-steps", type=int, default=20000,
                         help="Single-env trainer's env-step budget (default: 20000 -- a short "
                              "sanity-check budget, NOT the Core suite's full scale). The "
                              "vectorized trainer's budget is DERIVED from this (see "
                              "--same-total-steps) rather than reusing it directly: one "
                              "vectorized rollout consumes n_envs times as many real env.step() "
                              "calls as one single-env rollout of the same --rollout-len, so "
                              "reusing this value as-is would give the vectorized trainer ~n_envs "
                              "times FEWER gradient updates on the same nominal step budget -- a "
                              "gradient-starved, not-actually-comparable run, not a fair test of "
                              "whether the vectorized math trains correctly.")
    parser.add_argument("--same-total-steps", action="store_true",
                         help="Give the vectorized trainer the SAME total env-step budget as the "
                              "single-env trainer (the literal reading of --total-steps) instead "
                              "of the default (matching the NUMBER OF ROLLOUT UPDATES instead -- "
                              "see --total-steps's help). Useful for a wall-clock/throughput-style "
                              "comparison, but expect the vectorized trainer to look under-trained "
                              "at any n_envs > 1 under this mode, by design, not because its "
                              "training math is wrong.")
    parser.add_argument("--rollout-len", type=int, default=32)
    parser.add_argument("--hidden-actor", type=int, default=64)
    parser.add_argument("--hidden-critic", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu", help="torch device for the vectorized trainer "
                                                          "(default: cpu -- this dev machine has no "
                                                          "CUDA-enabled torch build)")
    args = parser.parse_args()

    print("=" * 78)
    print("VECTORIZED-ENV VALIDATION RUN (not a Core-suite replacement -- see module docstring)")
    print("=" * 78)

    all_results = {}
    for system in args.systems:
        print(f"\n--- {system} ---")
        print(f"  Running single-env MAPPO baseline ({args.total_steps} steps)...")
        single_env = run_single_env_baseline(
            system, args.seed, args.total_steps, args.rollout_len,
            args.hidden_actor, args.hidden_critic,
        )
        print(f"    final OF={single_env['final_eval']['OF']:.4f}  "
              f"cti_satisfied={single_env['final_eval']['cti_satisfied']}  "
              f"n_rollout_updates={len(single_env['history'])}  "
              f"elapsed={single_env['elapsed']:.1f}s")

        n_updates_target = max(1, args.total_steps // args.rollout_len)
        if args.same_total_steps:
            vec_total_steps = args.total_steps
        else:
            # Same number of rollout updates (and therefore Adam steps) as
            # the single-env run, not the same raw step count -- see
            # --total-steps's help for why matching raw steps alone would
            # starve the vectorized trainer of gradient updates.
            vec_total_steps = n_updates_target * args.rollout_len * args.n_envs

        print(f"  Running VectorizedMAPPO (n_envs={args.n_envs}, device={args.device}, "
              f"{vec_total_steps} steps -> ~{n_updates_target} rollout updates, matching "
              f"the single-env run's update count)..."
              if not args.same_total_steps else
              f"  Running VectorizedMAPPO (n_envs={args.n_envs}, device={args.device}, "
              f"{vec_total_steps} steps -- same raw step budget as single-env, "
              f"~{max(1, vec_total_steps // (args.rollout_len * args.n_envs))} rollout updates)...")
        vectorized = run_vectorized(
            system, args.seed, vec_total_steps, args.rollout_len, args.n_envs,
            args.hidden_actor, args.hidden_critic, args.device,
        )
        print(f"    final OF={vectorized['final_eval']['OF']:.4f}  "
              f"cti_satisfied={vectorized['final_eval']['cti_satisfied']}  "
              f"n_rollout_updates={len(vectorized['history'])}  "
              f"elapsed={vectorized['elapsed']:.1f}s")

        all_results[system] = {"single_env": single_env, "vectorized": vectorized}

        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        with open(RESULTS_DIR / f"{system}_results.json", "w") as f:
            json.dump(all_results[system], f, indent=2)

        fig = make_comparison_plot(system, single_env, vectorized, args.n_envs,
                                    args.rollout_len, args.same_total_steps)
        save_fig(fig, FIGURES_DIR / f"{system}_validation.png")
        plt.close(fig)
        print(f"  Saved results/figures to {RESULTS_DIR}/{system}_results.json and "
              f"{FIGURES_DIR}/{system}_validation.(png|pdf)")

    print("\n" + "=" * 78)
    print("Summary (final OF, lower is better; CTI satisfied = all relay pairs meet the")
    print("0.3s coordination time interval; n_upd = number of rollout updates each trainer")
    print(f"actually ran -- {'matched by construction' if not args.same_total_steps else 'NOT matched (--same-total-steps was passed)'})")
    print("=" * 78)
    print(f"{'System':<10} {'Single-env OF':>15} {'Vectorized OF':>15} {'Single CTI':>12} "
          f"{'Vec CTI':>10} {'n_upd(S/V)':>12}")
    for system, r in all_results.items():
        se, ve = r["single_env"]["final_eval"], r["vectorized"]["final_eval"]
        n_upd = f"{len(r['single_env']['history'])}/{len(r['vectorized']['history'])}"
        print(f"{system:<10} {se['OF']:>15.4f} {ve['OF']:>15.4f} "
              f"{str(se['cti_satisfied']):>12} {str(ve['cti_satisfied']):>10} {n_upd:>12}")
    print("=" * 78)
    print("This is a short-budget sanity check, not a converged result -- real Core-scale")
    print("comparisons (tens of millions of steps) and real GPU throughput numbers are out")
    print("of scope for this script (see scripts/benchmark_vectorized_env.py for the latter).")
    print("A similar final OF/CTI outcome at a matched update count is the useful signal here --")
    print("it means the vectorized math trains comparably to the single-env path on real data,")
    print("not that either has converged to its best achievable policy at this short budget.")


if __name__ == "__main__":
    main()
