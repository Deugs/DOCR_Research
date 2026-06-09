"""
experiments/results_aggregator.py
───────────────────────────────────
Results Aggregation and Analysis for Phase 5 Large-Scale Experiments.

Aggregates results across:
- Multiple test systems
- Multiple PV penetration levels
- Multiple fault types
- Multiple algorithms
- Multiple independent runs

Produces:
- Summary statistics (mean, std, min, max per method/system/PV level)
- PV penetration sweep analysis
- Curriculum learning progress analysis
- Topology robustness analysis
- LaTeX tables for thesis
- Publication-ready plots
"""

import numpy as np
import json
import os
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class AggregatedResult:
    """Aggregated results for a specific configuration."""
    method: str
    system: str
    pv_level: float
    fault_type: str

    n_runs: int = 0
    of_mean: float = 0.0
    of_std: float = 0.0
    of_min: float = 0.0
    of_max: float = 0.0

    cti_ok_rate: float = 0.0
    cti_violations_mean: float = 0.0
    cti_min_mean: float = 0.0
    cti_min_std: float = 0.0

    time_mean: float = 0.0
    time_std: float = 0.0

    # Convergence
    convergence_history: List[List[float]] = None

    def __post_init__(self):
        if self.convergence_history is None:
            self.convergence_history = []


class ResultsAggregator:
    """
    Aggregates and analyzes large-scale experiment results.

    Usage:
        aggregator = ResultsAggregator(results)
        aggregator.compute_all_statistics()
        aggregator.plot_pv_sweep("outputs/pv_sweep.png")
        aggregator.generate_latex_table("outputs/results_table.tex")
    """

    def __init__(self, results: List[Dict]):
        """
        Args:
            results: List of result dictionaries from TrainingPipeline
        """
        self.raw_results = results
        self.aggregated: Dict[str, AggregatedResult] = {}
        self.by_method: Dict[str, List[Dict]] = defaultdict(list)
        self.by_system: Dict[str, List[Dict]] = defaultdict(list)
        self.by_pv: Dict[float, List[Dict]] = defaultdict(list)
        self.by_fault: Dict[str, List[Dict]] = defaultdict(list)

        self._index_results()

    def _index_results(self):
        """Index results by various dimensions."""
        for r in self.raw_results:
            config = r.get("config", r)
            self.by_method[config.get("method", "unknown")].append(r)
            self.by_system[config.get("system", "unknown")].append(r)
            self.by_pv[config.get("pv_level", 0.0)].append(r)
            self.by_fault[config.get("fault_type", "3ph")].append(r)

    def aggregate(self, group_by: List[str] = None) -> Dict[str, AggregatedResult]:
        """
        Aggregate results by specified dimensions.

        Args:
            group_by: List of dimension names to group by
                     (e.g., ["method", "system"], ["method", "pv_level"])

        Returns:
            Dictionary mapping group key -> AggregatedResult
        """
        if group_by is None:
            group_by = ["method", "system", "pv_level", "fault_type"]

        groups = defaultdict(list)

        for r in self.raw_results:
            config = r.get("config", r)
            key_parts = []
            for dim in group_by:
                val = config.get(dim, "unknown")
                key_parts.append(f"{dim}={val}")
            key = "|".join(key_parts)
            groups[key].append(r)

        aggregated = {}
        for key, group_results in groups.items():
            agg = self._compute_aggregated(group_results, group_by)
            aggregated[key] = agg

        self.aggregated = aggregated
        return aggregated

    def _compute_aggregated(self, results: List[Dict], group_by: List[str]) -> AggregatedResult:
        """Compute aggregated statistics for a group of results."""
        if not results:
            return AggregatedResult(method="", system="", pv_level=0.0, fault_type="")

        # Extract first result for metadata
        first = results[0]
        config = first.get("config", first)

        # Collect metrics
        ofs = [r["best_of"] for r in results]
        cti_ok = [r.get("cti_satisfied", False) for r in results]
        cti_violations = [r.get("cti_violations", 0) for r in results]
        cti_mins = [r.get("cti_min", 0.0) for r in results]
        times = [r.get("elapsed_seconds", 0.0) for r in results]
        histories = [r.get("history", []) for r in results]

        return AggregatedResult(
            method=config.get("method", ""),
            system=config.get("system", ""),
            pv_level=config.get("pv_level", 0.0),
            fault_type=config.get("fault_type", ""),
            n_runs=len(results),
            of_mean=float(np.mean(ofs)),
            of_std=float(np.std(ofs)),
            of_min=float(np.min(ofs)),
            of_max=float(np.max(ofs)),
            cti_ok_rate=float(np.mean(cti_ok)),
            cti_violations_mean=float(np.mean(cti_violations)),
            cti_min_mean=float(np.mean(cti_mins)),
            cti_min_std=float(np.std(cti_mins)),
            time_mean=float(np.mean(times)),
            time_std=float(np.std(times)),
            convergence_history=histories,
        )

    def get_pv_sweep_data(self, method: str = None, system: str = None) -> Dict:
        """
        Get data for PV penetration sweep analysis.

        Returns:
            Dictionary with PV levels as keys and statistics as values
        """
        data = defaultdict(lambda: {"ofs": [], "cti_ok": [], "times": []})

        for r in self.raw_results:
            config = r.get("config", r)
            if method and config.get("method") != method:
                continue
            if system and config.get("system") != system:
                continue

            pv = config.get("pv_level", 0.0)
            data[pv]["ofs"].append(r["best_of"])
            data[pv]["cti_ok"].append(r.get("cti_satisfied", False))
            data[pv]["times"].append(r.get("elapsed_seconds", 0.0))

        # Compute statistics per PV level
        sweep = {}
        for pv, metrics in sorted(data.items()):
            sweep[pv] = {
                "of_mean": float(np.mean(metrics["ofs"])),
                "of_std": float(np.std(metrics["ofs"])),
                "of_min": float(np.min(metrics["ofs"])),
                "of_max": float(np.max(metrics["ofs"])),
                "cti_ok_rate": float(np.mean(metrics["cti_ok"])),
                "time_mean": float(np.mean(metrics["times"])),
                "n_runs": len(metrics["ofs"]),
            }

        return sweep

    def get_curriculum_progress(self, method: str = "MAPPO-Safe-EWC") -> Dict:
        """
        Get curriculum learning progress data.

        Returns:
            Dictionary with curriculum stages as keys
        """
        data = defaultdict(lambda: {"ofs": [], "histories": []})

        for r in self.raw_results:
            config = r.get("config", r)
            if config.get("method") != method:
                continue

            stage = config.get("curriculum_stage", 0)
            data[stage]["ofs"].append(r["best_of"])
            data[stage]["histories"].append(r.get("history", []))

        progress = {}
        for stage, metrics in sorted(data.items()):
            progress[stage] = {
                "of_mean": float(np.mean(metrics["ofs"])),
                "of_std": float(np.std(metrics["ofs"])),
                "of_min": float(np.min(metrics["ofs"])),
                "n_runs": len(metrics["ofs"]),
            }

        return progress

    def get_topology_robustness(self, method: str = None) -> Dict:
        """
        Get topology robustness data (N-1 outage impact).

        Returns:
            Dictionary comparing with/without outages
        """
        with_outage = defaultdict(lambda: {"ofs": [], "cti_ok": []})
        without_outage = defaultdict(lambda: {"ofs": [], "cti_ok": []})

        for r in self.raw_results:
            config = r.get("config", r)
            if method and config.get("method") != method:
                continue

            sys_name = config.get("system", "unknown")
            topo = config.get("topology_changes", {})
            has_outage = topo is not None and len(topo) > 0

            if has_outage:
                with_outage[sys_name]["ofs"].append(r["best_of"])
                with_outage[sys_name]["cti_ok"].append(r.get("cti_satisfied", False))
            else:
                without_outage[sys_name]["ofs"].append(r["best_of"])
                without_outage[sys_name]["cti_ok"].append(r.get("cti_satisfied", False))

        robustness = {}
        for sys_name in set(list(with_outage.keys()) + list(without_outage.keys())):
            wo = with_outage.get(sys_name, {"ofs": [], "cti_ok": []})
            wout = without_outage.get(sys_name, {"ofs": [], "cti_ok": []})

            if wo["ofs"] and wout["ofs"]:
                mean_with = np.mean(wo["ofs"])
                mean_without = np.mean(wout["ofs"])
                degradation = ((mean_with - mean_without) / mean_without * 100) if mean_without > 0 else 0.0

                robustness[sys_name] = {
                    "with_outage_of_mean": float(mean_with),
                    "without_outage_of_mean": float(mean_without),
                    "degradation_percent": float(degradation),
                    "cti_ok_with": float(np.mean(wo["cti_ok"])) if wo["cti_ok"] else 0.0,
                    "cti_ok_without": float(np.mean(wout["cti_ok"])) if wout["cti_ok"] else 0.0,
                }

        return robustness

    def plot_pv_sweep(self, output_path: str, methods: List[str] = None):
        """Plot PV penetration sweep analysis."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not available, skipping plot")
            return

        if methods is None:
            methods = sorted(self.by_method.keys())

        STYLE = {
            "figure.facecolor": "#0d1117",
            "axes.facecolor":   "#161b22",
            "axes.edgecolor":   "#30363d",
            "axes.labelcolor":  "#8b949e",
            "axes.titlecolor":  "#e6edf3",
            "xtick.color":      "#8b949e",
            "ytick.color":      "#8b949e",
            "grid.color":       "#21262d",
            "grid.linewidth":   0.8,
            "text.color":       "#e6edf3",
            "legend.facecolor": "#161b22",
            "legend.edgecolor": "#30363d",
            "font.family":      "monospace",
        }
        plt.rcParams.update(STYLE)

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Phase 5: PV Penetration Sweep Analysis", fontsize=12)

        COLORS = {
            "GA-SQP": "#2563eb",
            "Single-PPO": "#f778ba",
            "IPPO": "#a371f7",
            "MAPPO": "#39d353",
            "MAPPO-Comm": "#00d4ff",
            "MAPPO-Safe": "#ff6b35",
            "MAPPO-Safe-EWC": "#f7c948",
            "HWOA": "#ff6b6b",
            "HHO": "#4ecdc4",
            "JAYA": "#45b7d1",
            "EEO": "#96ceb4",
        }

        # OF vs PV Level
        ax = axes[0, 0]
        for method in methods:
            sweep = self.get_pv_sweep_data(method=method)
            if sweep:
                pvs = sorted(sweep.keys())
                means = [sweep[pv]["of_mean"] for pv in pvs]
                stds = [sweep[pv]["of_std"] for pv in pvs]
                color = COLORS.get(method, "#fff")
                ax.plot(pvs, means, "o-", label=method, color=color, linewidth=2, markersize=6)
                ax.fill_between(pvs, 
                               [m - s for m, s in zip(means, stds)],
                               [m + s for m, s in zip(means, stds)],
                               alpha=0.2, color=color)
        ax.set_xlabel("PV Penetration Level")
        ax.set_ylabel("Objective Function (s)")
        ax.set_title("OF vs PV Penetration")
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)
        ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_xticklabels(["0%", "20%", "40%", "60%", "80%", "100%"])

        # CTI Satisfaction Rate vs PV Level
        ax = axes[0, 1]
        for method in methods:
            sweep = self.get_pv_sweep_data(method=method)
            if sweep:
                pvs = sorted(sweep.keys())
                rates = [sweep[pv]["cti_ok_rate"] * 100 for pv in pvs]
                color = COLORS.get(method, "#fff")
                ax.plot(pvs, rates, "s-", label=method, color=color, linewidth=2, markersize=6)
        ax.axhline(y=95, color="red", linestyle="--", alpha=0.5, label="95% threshold")
        ax.set_xlabel("PV Penetration Level")
        ax.set_ylabel("CTI Satisfaction Rate (%)")
        ax.set_title("CTI Satisfaction vs PV Penetration")
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)
        ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_xticklabels(["0%", "20%", "40%", "60%", "80%", "100%"])
        ax.set_ylim([0, 105])

        # Computation Time vs PV Level
        ax = axes[1, 0]
        for method in methods:
            sweep = self.get_pv_sweep_data(method=method)
            if sweep:
                pvs = sorted(sweep.keys())
                times = [sweep[pv]["time_mean"] for pv in pvs]
                color = COLORS.get(method, "#fff")
                ax.plot(pvs, times, "^-", label=method, color=color, linewidth=2, markersize=6)
        ax.set_xlabel("PV Penetration Level")
        ax.set_ylabel("Computation Time (s)")
        ax.set_title("Time vs PV Penetration")
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)
        ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_xticklabels(["0%", "20%", "40%", "60%", "80%", "100%"])

        # OF Distribution (Box plot) at 100% PV
        ax = axes[1, 1]
        pv_100_data = []
        pv_100_labels = []
        for method in methods:
            method_data = [r["best_of"] for r in self.raw_results
                          if r.get("config", r).get("method") == method
                          and r.get("config", r).get("pv_level") == 1.0]
            if method_data:
                pv_100_data.append(method_data)
                pv_100_labels.append(method)

        if pv_100_data:
            bp = ax.boxplot(pv_100_data, labels=pv_100_labels, patch_artist=True)
            for patch, label in zip(bp["boxes"], pv_100_labels):
                patch.set_facecolor(COLORS.get(label, "#fff"))
                patch.set_alpha(0.7)
            ax.set_ylabel("Objective Function (s)")
            ax.set_title("OF Distribution at 100% PV")
            ax.tick_params(axis="x", rotation=45, labelsize=7)
            ax.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"  ✓ {output_path}")

    def plot_curriculum_progress(self, output_path: str, method: str = "MAPPO-Safe-EWC"):
        """Plot curriculum learning progress."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not available, skipping plot")
            return

        progress = self.get_curriculum_progress(method=method)
        if not progress:
            print(f"No curriculum data for {method}")
            return

        STYLE = {
            "figure.facecolor": "#0d1117",
            "axes.facecolor":   "#161b22",
            "axes.edgecolor":   "#30363d",
            "axes.labelcolor":  "#8b949e",
            "axes.titlecolor":  "#e6edf3",
            "xtick.color":      "#8b949e",
            "ytick.color":      "#8b949e",
            "grid.color":       "#21262d",
            "grid.linewidth":   0.8,
            "text.color":       "#e6edf3",
            "legend.facecolor": "#161b22",
            "legend.edgecolor": "#30363d",
            "font.family":      "monospace",
        }
        plt.rcParams.update(STYLE)

        fig, ax = plt.subplots(figsize=(10, 6))

        stages = sorted(progress.keys())
        means = [progress[s]["of_mean"] for s in stages]
        stds = [progress[s]["of_std"] for s in stages]

        ax.plot(stages, means, "o-", color="#39d353", linewidth=2, markersize=8, label=method)
        ax.fill_between(stages,
                       [m - s for m, s in zip(means, stds)],
                       [m + s for m, s in zip(means, stds)],
                       alpha=0.2, color="#39d353")

        # Add PV level annotations
        pv_labels = ["0%", "20%", "40%", "60%", "80%", "100%"]
        for i, (stage, mean) in enumerate(zip(stages, means)):
            if i < len(pv_labels):
                ax.annotate(pv_labels[i], (stage, mean), 
                           textcoords="offset points", xytext=(0, 10),
                           ha="center", fontsize=8, color="#8b949e")

        ax.set_xlabel("Curriculum Stage")
        ax.set_ylabel("Objective Function (s)")
        ax.set_title(f"Curriculum Learning Progress: {method}")
        ax.set_xticks(stages)
        ax.set_xticklabels([f"Stage {s}" for s in stages])
        ax.legend()
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"  ✓ {output_path}")

    def generate_latex_table(self, output_path: str, group_by: List[str] = None):
        """Generate LaTeX table of aggregated results."""
        if group_by is None:
            group_by = ["method", "system"]

        aggregated = self.aggregate(group_by=group_by)

        # Build LaTeX lines using explicit backslash character
        b = chr(92)
        lines = []

        # Table header
        lines.append(b + "begin{table}[htbp]")
        lines.append("  " + b + "centering")
        lines.append("  " + b + "caption{Phase 5: Large-Scale Experiment Results}")
        lines.append("  " + b + "label{tab:phase5_results}")
        lines.append("  " + b + "begin{tabular}{lcccccc}")
        lines.append("    " + b + "toprule")

        # Column header - use explicit backslash for percent
        header = "    Method & System & OF (mean$" + b + "pm$std) & CTI OK (" + b + "%) & CTI min (s) & Time (s) & Runs " + b + b + ""
        lines.append(header)
        lines.append("    " + b + "midrule")

        # Rows
        for key, agg in sorted(aggregated.items()):
            parts = key.split("|")
            method = parts[0].split("=")[1] if "=" in parts[0] else parts[0]
            system = parts[1].split("=")[1] if len(parts) > 1 and "=" in parts[1] else ""

            # Build OF string with escaped plus-minus
            of_str = "{:.4f}$" + b + "pm${:.4f}".format(agg.of_mean, agg.of_std)
            cti_str = "{:.1f}".format(agg.cti_ok_rate * 100)
            cti_min_str = "{:.4f}".format(agg.cti_min_mean)
            time_str = "{:.1f}$" + b + "pm${:.1f}".format(agg.time_mean, agg.time_std)

            # Build row with escaped backslash
            row_parts = [method, system, of_str, cti_str, cti_min_str, time_str, str(agg.n_runs)]
            row = "    " + " & ".join(row_parts) + " " + b + b + ""
            lines.append(row)

        lines.append("    " + b + "bottomrule")
        lines.append("  " + b + "end{tabular}")
        lines.append(b + "end{table}")

        with open(output_path, 'w') as f:
            f.write(b + "n".join(lines))

        print(f"  ✓ {output_path}")

    def save_summary_json(self, output_path: str):
        """Save comprehensive summary to JSON."""
        summary = {
            "total_runs": len(self.raw_results),
            "by_method": {},
            "by_system": {},
            "by_pv_level": {},
            "pv_sweep": {},
            "curriculum_progress": {},
            "topology_robustness": {},
        }

        # Aggregate by method
        for method, results in self.by_method.items():
            ofs = [r["best_of"] for r in results]
            cti_ok = [r.get("cti_satisfied", False) for r in results]
            summary["by_method"][method] = {
                "n_runs": len(results),
                "of_mean": float(np.mean(ofs)),
                "of_std": float(np.std(ofs)),
                "cti_ok_rate": float(np.mean(cti_ok)),
            }

        # PV sweep (all methods combined)
        summary["pv_sweep"] = self.get_pv_sweep_data()

        # Curriculum
        summary["curriculum_progress"] = self.get_curriculum_progress()

        # Topology robustness
        summary["topology_robustness"] = self.get_topology_robustness()

        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"  ✓ {output_path}")

    def print_summary_table(self):
        """Print formatted summary table to console."""
        print(f"\n{'='*100}")
        print(f"  PHASE 5: LARGE-SCALE EXPERIMENT SUMMARY")
        print(f"{'='*100}")
        print(f"  Total runs: {len(self.raw_results)}")
        print(f"{'='*100}")

        # By method
        print(f"\n  By Method:")
        print(f"  {'Method':<20} {'Runs':>6} {'OF mean':>12} {'OF std':>10} {'CTI OK %':>10} {'Time (s)':>12}")
        print(f"  {'-'*80}")
        for method, results in sorted(self.by_method.items()):
            ofs = [r["best_of"] for r in results]
            cti_ok = [r.get("cti_satisfied", False) for r in results]
            times = [r.get("elapsed_seconds", 0.0) for r in results]
            print(f"  {method:<20} {len(results):>6} {np.mean(ofs):>12.4f} {np.std(ofs):>10.4f} "
                  f"{np.mean(cti_ok)*100:>9.1f}% {np.mean(times):>12.1f}")

        # By PV level
        print(f"\n  By PV Penetration Level:")
        print(f"  {'PV Level':<12} {'Runs':>6} {'OF mean':>12} {'OF std':>10} {'CTI OK %':>10}")
        print(f"  {'-'*60}")
        for pv, results in sorted(self.by_pv.items()):
            ofs = [r["best_of"] for r in results]
            cti_ok = [r.get("cti_satisfied", False) for r in results]
            print(f"  {pv*100:>5.0f}%       {len(results):>6} {np.mean(ofs):>12.4f} {np.std(ofs):>10.4f} "
                  f"{np.mean(cti_ok)*100:>9.1f}%")

        print(f"\n{'='*100}")


if __name__ == "__main__":
    # Example usage with dummy data
    dummy_results = [
        {"config": {"method": "GA-SQP", "system": "8bus", "pv_level": 0.0, "fault_type": "3ph"},
         "best_of": 5.2, "cti_satisfied": True, "cti_violations": 0, "cti_min": 0.35, "elapsed_seconds": 10.0},
        {"config": {"method": "GA-SQP", "system": "8bus", "pv_level": 1.0, "fault_type": "3ph"},
         "best_of": 6.5, "cti_satisfied": False, "cti_violations": 2, "cti_min": 0.15, "elapsed_seconds": 12.0},
        {"config": {"method": "MAPPO", "system": "8bus", "pv_level": 0.0, "fault_type": "3ph"},
         "best_of": 5.1, "cti_satisfied": True, "cti_violations": 0, "cti_min": 0.40, "elapsed_seconds": 120.0},
        {"config": {"method": "MAPPO", "system": "8bus", "pv_level": 1.0, "fault_type": "3ph"},
         "best_of": 5.8, "cti_satisfied": True, "cti_violations": 0, "cti_min": 0.32, "elapsed_seconds": 130.0},
    ]

    agg = ResultsAggregator(dummy_results)
    agg.print_summary_table()
