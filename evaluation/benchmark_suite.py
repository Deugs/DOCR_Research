"""
benchmark_suite.py
──────────────────
Phase 4: Automated benchmark comparison suite for DOCR-MARL.

Orchestrates:
1. Run all methods (GA-SQP, metaheuristics, MARL) across test systems
2. Multiple independent runs per method for statistical validity
3. Collect metrics: OF, CTI violations, elapsed time, constraint satisfaction
4. Generate comparison tables, convergence plots, CD diagrams
5. Run statistical tests (Friedman, Nemenyi, Cliff's delta, Wilcoxon)

Usage:
    from evaluation.benchmark_suite import BenchmarkSuite
    
    suite = BenchmarkSuite(systems=["8bus", "14bus", "39bus"], n_runs=10)
    results = suite.run_all()
    suite.print_comparison_table()
    suite.plot_convergence()
    suite.plot_cd_diagram()
    suite.run_statistical_tests()
"""

import numpy as np
import time
import zlib
import multiprocessing as mp
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field

from network.network_data import NetworkData, NETWORKS
from network.relay_physics import check_constraints, compute_fault_currents_dynamic

# Import metaheuristics
from baselines.metaheuristics import (
    run_hwoa, run_hho, run_jaya, run_eeo,
    METAHEURISTICS, run_all_metaheuristics
)

# Import statistical analysis
from evaluation.stat_analysis import (
    run_benchmark_analysis,
    aggregate_results,
    friedman_test,
    nemenyi_test,
    cliffs_delta,
    wilcoxon_pairwise,
)


def _method_seed_offset(name: str, modulus: int = 10000) -> int:
    # zlib.crc32 instead of hash(): Python's hash() is randomized per
    # process by default (PEP 456), so the same method name would derive
    # a different offset across separate invocations/worker processes,
    # silently breaking reproducibility. Same convention as
    # run_core_training.py's _method_seed_offset.
    return zlib.crc32(name.encode()) % modulus


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark experiments."""
    systems: List[str] = field(default_factory=lambda: ["8bus"])
    methods: List[str] = field(default_factory=lambda: [
        "GA-SQP", "HWOA", "HHO", "JAYA", "EEO"
    ])
    n_runs: int = 10              # Independent runs per method
    pop_size: int = 30            # Population size for metaheuristics
    max_iter: int = 500           # Iterations for metaheuristics
    marl_steps: int = 10000       # Training steps for MARL
    ga_pop: int = 100             # GA population
    ga_gen: int = 200             # GA generations
    fault_type: str = "3ph"
    pv_level: float = 0.0
    use_sqp: bool = True          # SQP refinement for metaheuristics
    use_pandapower: bool = False
    seed_base: int = 42
    verbose: bool = True
    n_workers: int = 1            # >1 parallelizes across (system, run, method)
                                   # jobs via a process Pool; matches the
                                   # --n-workers convention used everywhere
                                   # else in this codebase (see
                                   # EXPERIMENT_GUIDE.md §2.2.2/§2.2.4)


class BenchmarkSuite:
    """
    Automated benchmark suite for DOCR algorithm comparison.
    
    Runs all configured methods across all test systems with
    multiple independent runs, then performs statistical analysis.
    """
    
    def __init__(self, config: Optional[BenchmarkConfig] = None):
        self.config = config or BenchmarkConfig()
        self.results: Dict[str, List[Dict]] = {}  # system -> [run_results]
        self.aggregated: Dict[str, Dict[str, float]] = {}  # system -> {method: mean_OF}
        self.stat_results: Optional[Dict] = None
    
    def run_all(self) -> Dict[str, List[Dict]]:
        """
        Run complete benchmark suite.

        Returns:
            results: {system_name: [run1, run2, ...]}
                     where each run = {method_name: result_dict}
        """
        cfg = self.config

        valid_systems = [s for s in cfg.systems if s in NETWORKS]
        for s in cfg.systems:
            if s not in NETWORKS:
                print(f"Warning: Unknown system {s}, skipping")

        for sys_key in valid_systems:
            net = NETWORKS[sys_key]()
            print(f"\\n{'█'*60}")
            print(f"  SYSTEM: {net.name}")
            print(f"  Relays: {net.n_relays}  |  Pairs: {net.n_pairs}")
            print(f"  Runs: {cfg.n_runs}  |  Methods: {', '.join(cfg.methods)}")
            print(f"{'█'*60}")
            # Pre-allocated by run_idx (not appended in completion order)
            # so results stay in the same slot regardless of which
            # worker/job finishes first when n_workers > 1.
            self.results[sys_key] = [{} for _ in range(cfg.n_runs)]

        # Every (system, run, method) job is independent, so they can all
        # be dispatched to one shared pool across the whole sweep rather
        # than per-system/per-run batches -- maximizes parallelism when
        # n_workers is large relative to any single system's method count.
        jobs = [
            (sys_key, run_idx, method,
             # Per-method offset (not just per-run): without this, every
             # method within a run shared the exact same seed, so any two
             # methods whose code paths happened to be otherwise identical
             # (see _run_marl_method_impl's Safe/EWC fix) produced
             # byte-identical results -- not distinct observations of a
             # real difference.
             cfg.seed_base + run_idx * 100 + _method_seed_offset(method),
             cfg)
            for sys_key in valid_systems
            for run_idx in range(cfg.n_runs)
            for method in cfg.methods
        ]

        total = len(jobs)
        done = 0
        for (sys_key, run_idx, method), result in _run_pool(jobs, _benchmark_job, cfg.n_workers):
            done += 1
            if "error" in result and len(result) == 1:
                print(f"    ERROR running {method} on {sys_key} (run {run_idx + 1}): {result['error']}")
            else:
                self.results[sys_key][run_idx][method] = result
            if cfg.verbose and (done % max(total // 20, 1) == 0 or done == total):
                print(f"  [{done}/{total}] {sys_key} run {run_idx + 1}/{cfg.n_runs}: {method}")

        # Aggregate results
        self.aggregated = aggregate_results(
            self.results, metric="OF", aggregation="mean"
        )

        return self.results
    
    def _run_method(self, method: str, net: NetworkData, seed: int,
                    cfg: BenchmarkConfig, verbose: bool = False) -> Dict:
        """Instance-method entry point, kept for any external caller --
        delegates to the top-level _run_method_impl (a plain function, so
        it's picklable for the multiprocessing Pool in run_all() without
        needing to pickle a whole BenchmarkSuite instance)."""
        return _run_method_impl(method, net, seed, cfg, verbose)


    def print_comparison_table(self):
        """Print formatted comparison table."""
        cfg = self.config
        
        LINE = "─" * 120
        print(f"\\n{LINE}")
        print(f"  {'System':<10} {'Method':<15} {'OF (s)':>10} {'OF_std':>10} "
              f"{'CTI viol':>10} {'CTI min':>10} {'Time (s)':>10} {'Success':>8}")
        print(LINE)
        
        for sys_key in sorted(self.results.keys()):
            runs = self.results[sys_key]
            methods = sorted(runs[0].keys()) if runs else []
            
            for method in methods:
                ofs = [r[method]["OF"] for r in runs if method in r]
                cti_viols = [r[method].get("constraints", {}).get("cti_violations", 0) 
                            for r in runs if method in r]
                cti_mins = [r[method].get("constraints", {}).get("cti_min", 0.0) 
                           for r in runs if method in r]
                times = [r[method]["elapsed"] for r in runs if method in r]
                
                if ofs:
                    of_mean = np.mean(ofs)
                    of_std = np.std(ofs)
                    cti_v = int(np.mean(cti_viols))
                    cti_m = np.mean(cti_mins)
                    t_mean = np.mean(times)
                    success = "✓" if cti_v == 0 else "✗"
                    
                    print(f"  {sys_key:<10} {method:<15} {of_mean:>10.4f} {of_std:>10.4f} "
                          f"{cti_v:>10d} {cti_m:>10.4f} {t_mean:>10.2f} {success:>8}")
            print()
        
        print(LINE)
    
    def run_statistical_tests(self, alpha: float = 0.05) -> Dict:
        """
        Run Friedman, Nemenyi, Cliff's delta, and Wilcoxon tests.
        
        Returns:
            stat_results: Dict with all test results
        """
        if not self.aggregated:
            print("No results to analyze. Run run_all() first.")
            return {}
        
        print(f"\\n{'='*70}")
        print(f"  STATISTICAL ANALYSIS")
        print(f"{'='*70}")
        
        self.stat_results = run_benchmark_analysis(
            self.aggregated, metric="OF", alpha=alpha, verbose=True
        )
        
        return self.stat_results
    
    def plot_convergence(self, out_path: str = "outputs/benchmark_convergence.png"):
        """Plot convergence curves for all methods."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not available, skipping convergence plot")
            return
        
        cfg = self.config
        n_systems = len(cfg.systems)
        fig, axes = plt.subplots(1, n_systems, figsize=(6 * n_systems, 5))
        if n_systems == 1:
            axes = [axes]
        
        COLORS = {
            "GA-SQP": "#2563eb", "HWOA": "#f778ba", "HHO": "#a371f7",
            "JAYA": "#39d353", "EEO": "#ff6b35",
            "IPPO": "#00d4ff", "MAPPO": "#f7c948",
            "MAPPO-Comm": "#ff9f43", "MAPPO-Safe": "#ee5a24",
            "MAPPO-Safe-EWC": "#c44569", "Single-PPO": "#778beb",
        }
        
        for idx, sys_key in enumerate(cfg.systems):
            ax = axes[idx]
            runs = self.results.get(sys_key, [])
            if not runs:
                continue
            
            methods = sorted(runs[0].keys())
            
            for method in methods:
                # Plot mean convergence with std band
                histories = []
                for run in runs:
                    if method in run and "history" in run[method]:
                        h = run[method]["history"]
                        if isinstance(h, list) and len(h) > 0:
                            histories.append(np.array(h))
                
                if not histories:
                    continue
                
                max_len = max(len(h) for h in histories)
                padded = []
                for h in histories:
                    if len(h) < max_len:
                        h = np.pad(h, (0, max_len - len(h)), mode="edge")
                    padded.append(h)
                
                padded = np.array(padded)
                mean_curve = np.mean(padded, axis=0)
                std_curve = np.std(padded, axis=0)
                
                iters = np.arange(len(mean_curve))
                color = COLORS.get(method, "#888")
                
                ax.plot(iters, mean_curve, label=method, color=color, linewidth=1.5)
                ax.fill_between(iters, mean_curve - std_curve, mean_curve + std_curve,
                                color=color, alpha=0.15)
            
            ax.set_title(f"{sys_key.upper()} — Convergence", fontsize=11)
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Objective Function (s)")
            ax.legend(fontsize=7, loc="upper right")
            ax.grid(True, alpha=0.3)
            ax.set_yscale("log")
        
        fig.tight_layout()
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  ✓ Convergence plot: {out_path}")
    
    def plot_cd_diagram(self, out_path: str = "outputs/cd_diagram.png"):
        """Plot Critical Difference diagram."""
        if self.stat_results is None:
            print("Run statistical tests first.")
            return
        
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not available, skipping CD diagram")
            return
        
        nemenyi = self.stat_results["nemenyi"]
        ranking = self.stat_results["ranking"]
        methods = ranking["methods"]
        mean_ranks = ranking["mean_ranks"]
        CD = nemenyi["critical_difference"]
        
        # Sort by rank
        sorted_idx = np.argsort(mean_ranks)
        sorted_methods = [methods[i] for i in sorted_idx]
        sorted_ranks = [mean_ranks[i] for i in sorted_idx]
        
        fig, ax = plt.subplots(figsize=(10, len(sorted_methods) * 0.6 + 2))
        
        # Draw axis
        ax.axhline(y=0, color="black", linewidth=1)
        
        # Plot methods as points
        y_positions = np.arange(len(sorted_methods))
        ax.scatter(sorted_ranks, y_positions, s=100, zorder=5, color="#2563eb")
        
        # Draw CD bar at top
        ax.plot([sorted_ranks[0], sorted_ranks[0] + CD], 
                [len(sorted_methods) + 0.5, len(sorted_methods) + 0.5],
                "k-", linewidth=2)
        ax.text(sorted_ranks[0] + CD / 2, len(sorted_methods) + 0.8, 
                f"CD = {CD:.3f}", ha="center", fontsize=10)
        
        # Connect non-significantly different methods
        for i in range(len(sorted_methods)):
            for j in range(i + 1, len(sorted_methods)):
                if abs(sorted_ranks[i] - sorted_ranks[j]) < CD:
                    ax.plot([sorted_ranks[i], sorted_ranks[j]], 
                            [y_positions[i], y_positions[j]], 
                            "r-", linewidth=3, alpha=0.3, zorder=1)
        
        ax.set_yticks(y_positions)
        ax.set_yticklabels(sorted_methods)
        ax.set_xlabel("Average Rank (lower is better)")
        ax.set_title("Critical Difference Diagram (Nemenyi Test)", fontsize=12)
        ax.grid(True, axis="x", alpha=0.3)
        ax.set_ylim(-0.5, len(sorted_methods) + 1.5)
        
        fig.tight_layout()
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  ✓ CD diagram: {out_path}")
    
    def generate_latex_table(self, out_path: str = "outputs/benchmark_table.tex"):
        """Generate LaTeX table of results."""
        cfg = self.config
        
        lines = [
            "\\\\begin{table}[t]",
            "\\\\centering",
            "\\\\caption{Benchmark Comparison: Objective Function (s) across Test Systems}",
            "\\\\label{tab:benchmark}",
            "\\\\begin{tabular}{l" + "c" * len(cfg.systems) + "}",
            "\\\\toprule",
            "Method & " + " & ".join(cfg.systems) + " \\\\\\\\" ,
            "\\\\midrule",
        ]
        
        methods = cfg.methods
        for method in methods:
            row = [method]
            for sys_key in cfg.systems:
                runs = self.results.get(sys_key, [])
                ofs = [r[method]["OF"] for r in runs if method in r]
                if ofs:
                    mean = np.mean(ofs)
                    std = np.std(ofs)
                    row.append(f"${mean:.4f} \\pm {std:.4f}$")
                else:
                    row.append("—")
            lines.append(" & ".join(row) + " \\\\\\\\" )
        
        lines.extend([
            "\\\\bottomrule",
            "\\\\end{tabular}",
            "\\\\end{table}",
        ])
        
        with open(out_path, "w") as f:
            f.write("\\n".join(lines))
        
        print(f"  ✓ LaTeX table: {out_path}")
    
    def save_results(self, out_path: str = "outputs/benchmark_results.npz"):
        """Save raw results to NPZ file for later analysis."""
        # Convert nested dict to serializable format
        save_dict = {}
        for sys_key, runs in self.results.items():
            for run_idx, run in enumerate(runs):
                for method, result in run.items():
                    prefix = f"{sys_key}_run{run_idx}_{method}_"
                    for key, val in result.items():
                        if isinstance(val, np.ndarray):
                            save_dict[prefix + key] = val
                        elif isinstance(val, (int, float, bool, str)):
                            save_dict[prefix + key] = val
                        # Skip dicts (constraints) and lists (history) for simplicity
        
        np.savez_compressed(out_path, **save_dict)
        print(f"  ✓ Results saved: {out_path}")


def _run_method_impl(method: str, net: NetworkData, seed: int,
                     cfg: BenchmarkConfig, verbose: bool = False) -> Dict:
    """Run a single method and return standardized result dict."""

    # PV-scale the network before it reaches any objective/constraint
    # evaluation. `topology_state["fault_currents"]` carries the
    # PV-penetration-dependent fault currents (see
    # network.relay_physics.compute_fault_currents_dynamic) that
    # objective_function/check_constraints use when present, falling
    # back to the network's static (0% PV) fault currents otherwise.
    # Without this, cfg.pv_level was accepted by every run_* call only
    # for logging/result-dict bookkeeping and never actually changed
    # what was optimized or measured.
    topology_state = {
        "fault_currents": compute_fault_currents_dynamic(
            net, pv_penetration=cfg.pv_level, fault_type=cfg.fault_type
        )
    }

    if method == "GA-SQP":
        from baselines.ga_sqp import run_ga_sqp
        result = run_ga_sqp(
            net, ga_pop=cfg.ga_pop, ga_gen=cfg.ga_gen,
            seed=seed, verbose=verbose,
            fault_type=cfg.fault_type, pv_level=cfg.pv_level,
            topology_state=topology_state,
        )
        result["method"] = "GA-SQP"
        return result

    elif method == "HWOA":
        return run_hwoa(
            net, n_whales=cfg.pop_size, max_iter=cfg.max_iter,
            seed=seed, verbose=verbose,
            fault_type=cfg.fault_type, pv_level=cfg.pv_level,
            use_sqp=cfg.use_sqp, topology_state=topology_state,
        )

    elif method == "HHO":
        return run_hho(
            net, n_hawks=cfg.pop_size, max_iter=cfg.max_iter,
            seed=seed, verbose=verbose,
            fault_type=cfg.fault_type, pv_level=cfg.pv_level,
            use_sqp=cfg.use_sqp, topology_state=topology_state,
        )

    elif method == "JAYA":
        return run_jaya(
            net, n_pop=cfg.pop_size, max_iter=cfg.max_iter,
            seed=seed, verbose=verbose,
            fault_type=cfg.fault_type, pv_level=cfg.pv_level,
            use_sqp=cfg.use_sqp, topology_state=topology_state,
        )

    elif method == "EEO":
        return run_eeo(
            net, n_particles=cfg.pop_size, max_iter=cfg.max_iter,
            seed=seed, verbose=verbose,
            fault_type=cfg.fault_type, pv_level=cfg.pv_level,
            use_sqp=cfg.use_sqp, topology_state=topology_state,
        )

    elif method in ("IPPO", "MAPPO", "MAPPO-Comm", "MAPPO-Safe", "MAPPO-Safe-EWC"):
        # MARL methods - import dynamically to avoid dependency issues
        return _run_marl_method_impl(method, net, seed, cfg, verbose)

    elif method == "Single-PPO":
        # Centralized (not multi-agent) baseline -- needs its own env
        # wrapper (CentralizedPPOEnvironment), not DynamicDOCREnv, so it
        # can't go through _run_marl_method_impl. Previously routed
        # there anyway by the tuple above, which silently built a plain
        # MAPPOMARLTrainer and returned MAPPO's numbers mislabeled
        # "Single-PPO" instead of erroring.
        return _run_single_ppo_impl(net, seed, cfg, verbose)

    else:
        raise ValueError(f"Unknown method: {method}")


def _run_single_ppo_impl(net: NetworkData, seed: int, cfg: BenchmarkConfig,
                         verbose: bool = False) -> Dict:
    """Run the centralized (single-agent) PPO baseline."""
    from marl.single_agent_ppo import CentralizedPPOEnvironment, SinglePPOTrainer

    env = CentralizedPPOEnvironment(net, use_pandapower=cfg.use_pandapower, seed=seed)
    env.reset(pv_level=cfg.pv_level, fault_type=cfg.fault_type, seed=seed)
    trainer = SinglePPOTrainer(env, seed=seed)

    t0 = time.perf_counter()
    result = trainer.train(total_steps=cfg.marl_steps, verbose=verbose,
                           log_every=max(cfg.marl_steps // 10, 1))
    elapsed = time.perf_counter() - t0

    # Normalize to the same shape _run_marl_method_impl's results have:
    # "best_of" (not just "OF") and a real "constraints" dict, both read
    # downstream by experiments/core_to_phase5_bridge.py's
    # _from_benchmark_pv().
    result["best_of"] = result["OF"]
    topology_state = {
        "fault_currents": compute_fault_currents_dynamic(
            net, pv_penetration=cfg.pv_level, fault_type=cfg.fault_type
        )
    }
    result["constraints"] = check_constraints(
        net, result["TMS"], result["PS"], fault_type=cfg.fault_type,
        topology_state=topology_state,
    )
    result["elapsed"] = elapsed
    result["method"] = "Single-PPO"
    result["fault_type"] = cfg.fault_type
    result["pv_level"] = cfg.pv_level

    return result


def _run_marl_method_impl(method: str, net: NetworkData, seed: int,
                          cfg: BenchmarkConfig, verbose: bool = False) -> Dict:
    """Run MARL methods (requires marl modules)."""
    from marl.marl_env import DynamicDOCREnv
    from marl.marl_agents import MAPPOMARLTrainer

    # Map the requested PV level onto DynamicDOCREnv's curriculum_stage,
    # matching the convention run_core_training.py's curriculum runs use
    # (env.curriculum_stage = int(pv * 5): 0.0->0, 0.2->1, 0.4->2, 0.6->3,
    # 0.8->4, 1.0->5) — env.pv_levels[curriculum_stage] then sets
    # env.current_pv_level for the whole episode. Previously this
    # constructor call never set curriculum_stage, so every MARL method
    # trained/evaluated at the env default (0% PV) regardless of
    # cfg.pv_level.
    curriculum_stage = int(round(cfg.pv_level * 5))
    env = DynamicDOCREnv(
        net, action_scale=0.05, max_steps=300, seed=seed,
        use_pandapower=cfg.use_pandapower, fault_type=cfg.fault_type,
        curriculum_stage=curriculum_stage,
    )

    use_comm = "Comm" in method
    # "Safe"/"EWC" in the method name previously did nothing here --
    # "MAPPO-Safe"/"MAPPO-Safe-EWC" trained through the exact same
    # unconstrained MAPPOMARLTrainer as plain "MAPPO", with no Safe RL
    # wrapper or EWC regularizer ever applied, and (combined with
    # run_all()'s previously-shared per-run seed) produced literally
    # byte-identical results across all three. Same bug class as
    # run_core_training.py's Core C had (see its _run_baseline_single),
    # fixed here the same way: Safe RL is a real env wrapper (action
    # projection + Augmented Lagrangian) and must wrap env BEFORE the
    # trainer is constructed.
    use_safe = method in ("MAPPO-Safe", "MAPPO-Safe-EWC")
    use_ewc = method == "MAPPO-Safe-EWC"

    train_env = env
    if use_safe:
        from marl.safe_rl import create_safe_rl_env, SafeRLConfig
        train_env = create_safe_rl_env(env, SafeRLConfig(
            project_actions=True, enforce_cti=True,
            lambda_init=1.0, lambda_lr=0.01, rho_penalty=10.0,
            use_pid=True, adaptive_rho=True,
        ))

    trainer = MAPPOMARLTrainer(
        train_env, use_comm=use_comm,
        hidden_actor=64, hidden_critic=128,
        lr_actor=0.02, lr_critic=0.01,
        clip_eps=0.2, gamma=0.99, lam=0.95,
        rollout_len=64, n_epochs=4, seed=seed, curriculum=False,
    )

    t0 = time.perf_counter()
    result = trainer.train(
        total_steps=cfg.marl_steps, verbose=verbose,
        log_every=max(cfg.marl_steps // 10, 1)
    )
    elapsed = time.perf_counter() - t0

    if use_ewc:
        # Register this (single) task with EWC so the apparatus
        # genuinely runs (n_tasks_learned reflects reality) rather
        # than staying entirely inert -- same pattern and rationale
        # as run_core_training.py's Core A/C: a lone task has nothing
        # to protect against forgetting (expected, not a bug), but
        # the real finite-difference Fisher estimate is computed
        # rather than a placeholder.
        from marl.continual_trainer import create_ewc_trainer, EWCConfig
        ewc_trainer = create_ewc_trainer(trainer, EWCConfig())
        fisher_rng = np.random.default_rng(seed + 8000)
        cur_params = trainer.actor.get_all_params()
        obs_f = train_env.reset(seed=seed + 9000)
        s_obs, s_act = [], []
        done_f, sf = False, 0
        while not done_f and sf < 30:
            acts = np.zeros((env.n_agents, env.ACT_DIM))
            enhanced_f, _, _ = trainer.comm_graph.forward(obs_f, fisher_rng) if trainer.comm_graph is not None else (obs_f, None, None)
            for i in range(env.n_agents):
                a, _ = trainer.actor.sample(enhanced_f[i], fisher_rng)
                acts[i] = a
                s_obs.append(enhanced_f[i].copy())
                s_act.append(a.copy())
            obs_f, _, done_f, _ = train_env.step(acts); sf += 1
        n_p = len(cur_params)
        fisher = np.zeros(n_p)
        if s_obs:
            idx = fisher_rng.choice(len(s_obs), size=min(30, len(s_obs)), replace=False)
            h = 1e-3
            for p_idx in fisher_rng.choice(n_p, size=min(n_p, 50), replace=False):
                bl = np.mean([trainer.actor.log_prob(s_obs[j], s_act[j]) for j in idx])
                b2 = cur_params.copy(); b2[p_idx] += h
                trainer.actor.set_all_params(b2)
                ul = np.mean([trainer.actor.log_prob(s_obs[j], s_act[j]) for j in idx])
                fisher[p_idx] = ((ul - bl) / h) ** 2
            trainer.actor.set_all_params(cur_params)
        ewc_trainer.ewc_regularizer.register_task(method, cur_params)
        ewc_trainer.ewc_regularizer.update_fisher(method, fisher)
        ewc_trainer.ewc_regularizer.finalize()
        result["ewc_stats"] = ewc_trainer.get_adaptation_stats()

    if use_safe:
        result["safe_rl_stats"] = train_env.get_safe_rl_stats()

    topology_state = {
        "fault_currents": compute_fault_currents_dynamic(
            net, pv_penetration=cfg.pv_level, fault_type=cfg.fault_type
        )
    }
    result["constraints"] = check_constraints(
        net, result["TMS"], result["PS"], fault_type=cfg.fault_type,
        topology_state=topology_state,
    )
    result["elapsed"] = elapsed
    result["method"] = method
    result["fault_type"] = cfg.fault_type
    result["pv_level"] = cfg.pv_level

    return result


def _run_pool(jobs, worker_fn, n_workers):
    """Run `worker_fn` over `jobs` either sequentially or via a Pool,
    yielding (job, result) pairs as they complete. Sequential path calls
    worker_fn directly (no process overhead) when n_workers <= 1. Same
    convention as run_core_training.py's _run_pool."""
    if n_workers > 1:
        with mp.Pool(n_workers, initializer=_benchmark_pool_worker_init) as pool:
            for job, result in pool.imap_unordered(worker_fn, jobs):
                yield job, result
    else:
        for job in jobs:
            yield worker_fn(job)


def _benchmark_pool_worker_init():
    import threadpoolctl
    threadpoolctl.threadpool_limits(1)


def _benchmark_job(args):
    """Top-level, picklable worker for BenchmarkSuite.run_all()'s Pool.
    Rebuilds `net` from the system name inside the worker rather than
    pickling a NetworkData across the process boundary -- same pattern
    run_core_training.py's job functions use (e.g. _core_C_job passing
    `system` and calling load_network(system) inside the worker)."""
    sys_key, run_idx, method, seed, cfg = args
    net = NETWORKS[sys_key]()
    try:
        result = _run_method_impl(method, net, seed, cfg, verbose=False)
    except Exception as e:
        import traceback
        traceback.print_exc()
        result = {"error": str(e)}
    return (sys_key, run_idx, method), result



def quick_benchmark(
    systems: List[str] = ["8bus"],
    methods: List[str] = ["GA-SQP", "HWOA", "HHO", "JAYA", "EEO"],
    n_runs: int = 5,
    fast: bool = False,
) -> BenchmarkSuite:
    """
    Quick benchmark run with sensible defaults.
    
    Usage:
        suite = quick_benchmark()
        suite.print_comparison_table()
        suite.run_statistical_tests()
    """
    cfg = BenchmarkConfig(
        systems=systems,
        methods=methods,
        n_runs=n_runs,
        pop_size=20 if fast else 30,
        max_iter=200 if fast else 500,
        ga_pop=50 if fast else 100,
        ga_gen=100 if fast else 200,
        marl_steps=5000 if fast else 10000,
        verbose=True,
    )
    
    suite = BenchmarkSuite(cfg)
    suite.run_all()
    return suite