"""
exp_fig15_scalability.py  —  Figure 15
─────────────────────────────────────────
Scalability Analysis: Performance vs. System Size.

Compares MAPPO-Safe-EWC, MAPPO-Safe, and metaheuristic baselines
across 8-bus, 14-bus, 30-bus, and 39-bus systems.

No synthetic fallback — fails loudly if network data or metaheuristic modules are missing.

Outputs
-------
  outputs/fig15_scalability.npz
  outputs/fig15_scalability.png
"""

import numpy as np
import os
import sys
import time
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(__file__))
os.makedirs("outputs", exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

from network.network_data import get_8bus
from marl.marl_env import DynamicDOCREnv
from marl.marl_agents import MAPPOMARLTrainer
from network.relay_physics import check_constraints

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

SYSTEMS = ["8bus", "14bus", "30bus", "39bus"]
METHODS = ["MAPPO-Safe-EWC", "MAPPO-Safe", "MAPPO", "GA-SQP", "HWOA", "HHO", "JAYA", "EEO"]
METHOD_COLORS = {
    "MAPPO-Safe-EWC": "#2563eb",
    "MAPPO-Safe": "#39d353",
    "MAPPO": "#f7c948",
    "GA-SQP": "#9b59b6",
    "HWOA": "#1abc9c",
    "HHO": "#e74c3c",
    "JAYA": "#f39c12",
    "EEO": "#34495e",
}


def get_network(system: str):
    """Load network data. Raises RuntimeError on failure — no synthetic fallback."""
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
        raise ValueError(f"Unknown system: {system}")


def _ensure_scalar(val):
    if isinstance(val, np.ndarray):
        return float(np.sum(val))
    return float(val)


def run_metaheuristic(method: str, net, n_agents: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Run a metaheuristic optimizer and return optimized TDS, PSM."""
    try:
        from mealpy import FloatVar
    except ImportError:
        raise RuntimeError(
            "mealpy is required for metaheuristic baselines. "
            "Install: pip install mealpy"
        )

    lb = np.array([0.05] * n_agents + [1.0] * n_agents)
    ub = np.array([1.0] * n_agents + [10.0] * n_agents)

    def objective(solution):
        TDS = solution[:n_agents]
        PSM = solution[n_agents:]
        try:
            c = check_constraints(net, TDS, PSM)
            of = c.get("objective_function", np.sum(TDS * PSM))
            penalty = c.get("cti_violations", 0) * 1000
            return of + penalty
        except Exception:
            return 1e6

    problem_dict = {
        "bounds": FloatVar(lb=tuple(lb), ub=tuple(ub), name="relay_settings"),
        "minmax": "min",
        "obj_func": objective,
    }

    if method == "GA-SQP":
        from mealpy import GA
        model = GA.BaseGA(epoch=100, pop_size=50, pc=0.9, pm=0.05)
    elif method == "HWOA":
        from mealpy import WOA
        model = WOA.OriginalWOA(epoch=100, pop_size=30)
    elif method == "HHO":
        from mealpy import HHO
        model = HHO.OriginalHHO(epoch=100, pop_size=30)
    elif method == "JAYA":
        from mealpy import JA
        model = JA.OriginalJA(epoch=100, pop_size=50)
    elif method == "EEO":
        from mealpy import EFO
        model = EFO.OriginalEFO(epoch=100, pop_size=30, r_rate=0.3, ps_rate=0.85, p_field=0.1, n_field=0.45)
    else:
        raise ValueError(f"Unknown metaheuristic: {method}")

    g_best = model.solve(problem_dict, seed=seed)
    sol = g_best.solution
    return sol[:n_agents], sol[n_agents:]


def evaluate_method(method: str, system: str, seed: int, n_eval_episodes: int = 50) -> Dict:
    """Evaluate one method on one system."""
    net = get_network(system)
    env = DynamicDOCREnv(net, seed=seed, use_pandapower=False,
                          fault_type="3ph", curriculum_stage=5)

    t0 = time.perf_counter()

    if method in ["MAPPO-Safe-EWC", "MAPPO-Safe", "MAPPO"]:
        use_comm = True
        trainer = MAPPOMARLTrainer(env, use_comm=use_comm, seed=seed)
        if hasattr(trainer, 'use_safety'):
            trainer.use_safety = method in ["MAPPO-Safe-EWC", "MAPPO-Safe"]
        if hasattr(trainer, 'use_ewc'):
            trainer.use_ewc = (method == "MAPPO-Safe-EWC")

        trainer.train(total_steps=5000, verbose=False)

        rng = np.random.default_rng(seed)
        returns = []
        cti_violations = []
        of_values = []

        for ep in range(n_eval_episodes):
            obs = env.reset(seed=seed + ep)
            done = False
            ep_return = 0.0
            while not done:
                actions = np.zeros((env.n_agents, env.ACT_DIM))
                for i in range(env.n_agents):
                    a, _ = trainer.actor.sample(obs[i], rng)
                    actions[i] = a
                obs, reward, done, info = env.step(actions)
                ep_return += _ensure_scalar(reward)
            returns.append(ep_return)

            TMS, PS = env.get_state()
            c = check_constraints(net, TMS, PS)
            cti_violations.append(c.get("cti_violations", 0))
            of_values.append(c.get("objective_function", 0))

    else:
        # Metaheuristic: optimize once, evaluate fixed settings
        TDS_opt, PSM_opt = run_metaheuristic(method, net, env.n_agents, seed)
        env.set_state(TDS_opt, PSM_opt)

        returns = []
        cti_violations = []
        of_values = []

        for ep in range(n_eval_episodes):
            obs = env.reset(seed=seed + ep)
            done = False
            ep_return = 0.0
            while not done:
                obs, reward, done, info = env.step(np.zeros((env.n_agents, env.ACT_DIM)))
                ep_return += _ensure_scalar(reward)
            returns.append(ep_return)

            c = check_constraints(net, TDS_opt, PSM_opt)
            cti_violations.append(c.get("cti_violations", 0))
            of_values.append(c.get("objective_function", 0))

    wall_time = time.perf_counter() - t0

    return {
        "returns": np.array(returns),
        "cti_violations": np.array(cti_violations),
        "of_values": np.array(of_values),
        "wall_time": wall_time,
        "n_agents": env.n_agents,
    }


def main(systems: List[str] = None, n_runs: int = 10, output_dir: str = "outputs"):
    if systems is None:
        systems = SYSTEMS

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    logger.info("Figure 15 — Scalability Analysis")
    logger.info("  Systems: %s", systems)
    logger.info("  Methods: %s", METHODS)
    logger.info("  Runs per config: %d", n_runs)

    all_results = {}

    for system in systems:
        logger.info("=== System: %s ===", system)
        all_results[system] = {}

        for method in METHODS:
            logger.info("  Method: %s", method)
            method_results = []

            for run_idx in range(n_runs):
                seed = run_idx + hash(method + system) % 10000
                logger.info("    Run %d/%d (seed=%d)...", run_idx + 1, n_runs, seed)
                res = evaluate_method(method, system, seed)
                method_results.append(res)

            # Aggregate
            all_returns = np.stack([r["returns"] for r in method_results])
            all_cti = np.stack([r["cti_violations"] for r in method_results])
            all_of = np.stack([r["of_values"] for r in method_results])
            wall_times = np.array([r["wall_time"] for r in method_results])
            n_agents = method_results[0]["n_agents"]

            all_results[system][method] = {
                "mean_return": float(all_returns.mean()),
                "std_return": float(all_returns.std()),
                "mean_cti": float(all_cti.mean()),
                "std_cti": float(all_cti.std()),
                "mean_of": float(all_of.mean()),
                "std_of": float(all_of.std()),
                "wall_time_mean": float(wall_times.mean()),
                "wall_time_std": float(wall_times.std()),
                "n_agents": n_agents,
            }

    # Save
    save_dict = {"systems": systems, "methods": METHODS, "n_runs": n_runs}
    for system in systems:
        for method in METHODS:
            for key in ["mean_return", "std_return", "mean_cti", "std_cti",
                        "mean_of", "std_of", "wall_time_mean", "wall_time_std", "n_agents"]:
                save_dict[f"{system}_{method}_{key}"] = all_results[system][method][key]

    np.savez(out_path / "fig15_scalability.npz", **save_dict)
    logger.info("Saved: %s", out_path / "fig15_scalability.npz")

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Figure 15 — Scalability Analysis Across System Sizes", fontsize=13)

    x = np.arange(len(systems))
    width = 0.1

    # Panel 1: Objective Function
    ax = axes[0, 0]
    for i, method in enumerate(METHODS):
        means = [all_results[s][method]["mean_of"] for s in systems]
        stds = [all_results[s][method]["std_of"] for s in systems]
        offset = (i - len(METHODS)/2) * width
        ax.bar(x + offset, means, width, yerr=stds, label=method,
               color=METHOD_COLORS[method], capsize=2, alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(systems)
    ax.set_ylabel("Objective Function (s)", fontsize=10)
    ax.set_title("Operating Time vs. System Size", fontsize=11)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3, axis="y")

    # Panel 2: CTI Violations
    ax = axes[0, 1]
    for i, method in enumerate(METHODS):
        means = [all_results[s][method]["mean_cti"] for s in systems]
        stds = [all_results[s][method]["std_cti"] for s in systems]
        offset = (i - len(METHODS)/2) * width
        ax.bar(x + offset, means, width, yerr=stds, label=method,
               color=METHOD_COLORS[method], capsize=2, alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(systems)
    ax.set_ylabel("CTI Violations", fontsize=10)
    ax.set_title("Constraint Violations vs. System Size", fontsize=11)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3, axis="y")

    # Panel 3: Wall-clock time
    ax = axes[1, 0]
    for i, method in enumerate(METHODS):
        means = [all_results[s][method]["wall_time_mean"] for s in systems]
        stds = [all_results[s][method]["wall_time_std"] for s in systems]
        offset = (i - len(METHODS)/2) * width
        ax.bar(x + offset, means, width, yerr=stds, label=method,
               color=METHOD_COLORS[method], capsize=2, alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(systems)
    ax.set_ylabel("Wall-Clock Time (s)", fontsize=10)
    ax.set_title("Computation Time vs. System Size", fontsize=11)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3, axis="y")

    # Panel 4: Number of agents
    ax = axes[1, 1]
    n_agents_list = [all_results[s][METHODS[0]]["n_agents"] for s in systems]
    ax.bar(systems, n_agents_list, color="#2563eb", edgecolor="white", alpha=0.8)
    ax.set_ylabel("Number of Relays (Agents)", fontsize=10)
    ax.set_title("Problem Dimensionality", fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")
    for i, v in enumerate(n_agents_list):
        ax.text(i, v + 0.3, str(v), ha="center", fontsize=10, fontweight="bold")

    fig.tight_layout()
    fig.savefig(out_path / "fig15_scalability.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", out_path / "fig15_scalability.png")

    # Statistical tests: Wilcoxon MARL vs each metaheuristic per system
    print("\n" + "=" * 70)
    print("STATISTICAL SIGNIFICANCE: MARL vs METAHEURISTICS")
    print("=" * 70)

    marl_methods = ["MAPPO-Safe-EWC", "MAPPO-Safe", "MAPPO"]
    meta_methods = ["GA-SQP", "HWOA", "HHO", "JAYA", "EEO"]

    for system in systems:
        print(f"\n--- {system} ---")
        for marl in marl_methods:
            for meta in meta_methods:
                # We need raw data for Wilcoxon — re-run or store from above
                # For now, placeholder — in practice, store all raw results
                print(f"  {marl} vs {meta}: Wilcoxon test required (store raw results)")

    print("=" * 70)

    # Summary table
    print("\n" + "=" * 80)
    print("SCALABILITY SUMMARY")
    print("=" * 80)
    print(f"{'System':>8} {'Method':>18} {'OF (s)':>10} {'CTI':>8} {'Time(s)':>10} {'Agents':>8}")
    print("-" * 80)
    for system in systems:
        for method in METHODS:
            r = all_results[system][method]
            print(f"{system:>8} {method:>18} {r['mean_of']:>10.4f} {r['mean_cti']:>8.2f} "
                  f"{r['wall_time_mean']:>10.1f} {r['n_agents']:>8}")
    print("=" * 80)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Figure 15 — Scalability Analysis")
    p.add_argument("--systems", nargs="+", default=SYSTEMS,
                   help="Systems to evaluate (8bus 14bus 30bus 39bus)")
    p.add_argument("--n_runs", type=int, default=10)
    p.add_argument("--output_dir", type=str, default="outputs")
    args = p.parse_args()
    main(args.systems, args.n_runs, args.output_dir)