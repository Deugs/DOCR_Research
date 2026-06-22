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
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field

from network.network_data import NetworkData, NETWORKS
from network.relay_physics import check_constraints

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
        
        for sys_key in cfg.systems:
            if sys_key not in NETWORKS:
                print(f"Warning: Unknown system {sys_key}, skipping")
                continue
            
            net = NETWORKS[sys_key]()
            print(f"\\n{'█'*60}")
            print(f"  SYSTEM: {net.name}")
            print(f"  Relays: {net.n_relays}  |  Pairs: {net.n_pairs}")
            print(f"  Runs: {cfg.n_runs}  |  Methods: {', '.join(cfg.methods)}")
            print(f"{'█'*60}")
            
            self.results[sys_key] = []
            
            for run_idx in range(cfg.n_runs):
                print(f"\\n  --- Run {run_idx + 1}/{cfg.n_runs} ---")
                run_results = {}
                seed = cfg.seed_base + run_idx * 100
                
                for method in cfg.methods:
                    try:
                        result = self._run_method(
                            method, net, seed, cfg,
                            verbose=(cfg.verbose and run_idx == 0)
                        )
                        run_results[method] = result
                    except Exception as e:
                        print(f"    ERROR running {method}: {e}")
                        import traceback
                        traceback.print_exc()
                
                self.results[sys_key].append(run_results)
        
        # Aggregate results
        self.aggregated = aggregate_results(
            self.results, metric="OF", aggregation="mean"
        )
        
        return self.results
    
    def _run_method(self, method: str, net: NetworkData, seed: int, 
                    cfg: BenchmarkConfig, verbose: bool = False) -> Dict:
        """Run a single method and return standardized result dict."""
        
        if method == "GA-SQP":
            from baselines.ga_sqp import run_ga_sqp
            result = run_ga_sqp(
                net, ga_pop=cfg.ga_pop, ga_gen=cfg.ga_gen,
                seed=seed, verbose=verbose,
                fault_type=cfg.fault_type, pv_level=cfg.pv_level
            )
            result["method"] = "GA-SQP"
            return result
        
        elif method == "HWOA":
            return run_hwoa(
                net, n_whales=cfg.pop_size, max_iter=cfg.max_iter,
                seed=seed, verbose=verbose,
                fault_type=cfg.fault_type, pv_level=cfg.pv_level,
                use_sqp=cfg.use_sqp
            )
        
        elif method == "HHO":
            return run_hho(
                net, n_hawks=cfg.pop_size, max_iter=cfg.max_iter,
                seed=seed, verbose=verbose,
                fault_type=cfg.fault_type, pv_level=cfg.pv_level,
                use_sqp=cfg.use_sqp
            )
        
        elif method == "JAYA":
            return run_jaya(
                net, n_pop=cfg.pop_size, max_iter=cfg.max_iter,
                seed=seed, verbose=verbose,
                fault_type=cfg.fault_type, pv_level=cfg.pv_level,
                use_sqp=cfg.use_sqp
            )
        
        elif method == "EEO":
            return run_eeo(
                net, n_particles=cfg.pop_size, max_iter=cfg.max_iter,
                seed=seed, verbose=verbose,
                fault_type=cfg.fault_type, pv_level=cfg.pv_level,
                use_sqp=cfg.use_sqp
            )
        
        elif method in ("IPPO", "MAPPO", "MAPPO-Comm", "MAPPO-Safe", "MAPPO-Safe-EWC", "Single-PPO"):
            # MARL methods - import dynamically to avoid dependency issues
            return self._run_marl_method(method, net, seed, cfg, verbose)
        
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _run_marl_method(self, method: str, net: NetworkData, seed: int,
                         cfg: BenchmarkConfig, verbose: bool = False) -> Dict:
        """Run MARL methods (requires marl modules)."""
        from marl.marl_env import DynamicDOCREnv
        from marl.marl_agents import MAPPOMARLTrainer
        
        env = DynamicDOCREnv(
            net, action_scale=0.05, max_steps=300, seed=seed,
            use_pandapower=cfg.use_pandapower, fault_type=cfg.fault_type
        )
        
        use_comm = "Comm" in method
        
        trainer = MAPPOMARLTrainer(
            env, use_comm=use_comm,
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
        
        result["constraints"] = check_constraints(
            net, result["TMS"], result["PS"], fault_type=cfg.fault_type
        )
        result["elapsed"] = elapsed
        result["method"] = method
        result["fault_type"] = cfg.fault_type
        result["pv_level"] = cfg.pv_level
        
        return result
    
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