"""
evaluation/large_scale_eval.py
──────────────────────────────
Large-Scale Evaluation Metrics for DOCR-MARL Phase 5.
Supports all IEEE test systems: 3, 8, 14, 30, 39, 57, 118 buses.

Computes comprehensive evaluation metrics:
1. Scalability metrics (performance vs system size)
2. Generalization metrics (unseen fault types, topologies)
3. Robustness metrics (N-1 outages, sensor noise)
4. Efficiency metrics (computation time, sample efficiency)
5. Communication metrics (bandwidth, message compression)
6. Safety metrics (constraint violation rates, recovery time)
7. Adaptation metrics (continual learning performance)

All metrics designed for thesis publication and statistical validation.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict

try:
    from network.network_data import NETWORKS
except ImportError:
    NETWORKS = {}


@dataclass
class ScalabilityMetrics:
    """Metrics for algorithm scalability across system sizes."""
    system_name: str
    n_relays: int
    n_pairs: int

    # Performance
    of_value: float
    cti_satisfied: bool

    # Computational
    training_time: float
    inference_time: float
    memory_usage_mb: float

    # Scaling behavior
    time_per_relay: float = 0.0
    memory_per_relay: float = 0.0

    def compute_derived(self):
        """Compute derived scaling metrics."""
        if self.n_relays > 0:
            self.time_per_relay = self.training_time / self.n_relays
            self.memory_per_relay = self.memory_usage_mb / self.n_relays


@dataclass
class GeneralizationMetrics:
    """Metrics for generalization to unseen conditions."""
    method: str

    # Seen vs unseen performance
    seen_of_mean: float = 0.0
    unseen_of_mean: float = 0.0

    # Generalization gap
    generalization_gap: float = 0.0  # (unseen - seen) / seen * 100

    # Per-fault-type performance
    fault_type_performance: Dict[str, float] = None

    def __post_init__(self):
        if self.fault_type_performance is None:
            self.fault_type_performance = {}

    def compute_gap(self):
        """Compute generalization gap percentage."""
        if self.seen_of_mean > 0:
            self.generalization_gap = ((self.unseen_of_mean - self.seen_of_mean) 
                                     / self.seen_of_mean * 100)


@dataclass
class RobustnessMetrics:
    """Metrics for robustness to perturbations."""
    method: str
    system: str

    # Baseline (no perturbation)
    baseline_of: float = 0.0
    baseline_cti_ok: bool = False

    # N-1 outage impact
    n1_degradation_percent: float = 0.0
    n1_cti_ok_rate: float = 0.0

    # Sensor noise impact
    noise_degradation_percent: float = 0.0
    noise_cti_ok_rate: float = 0.0

    # Combined stress
    combined_degradation_percent: float = 0.0

    # Recovery
    recovery_time_steps: int = 0


@dataclass
class EfficiencyMetrics:
    """Metrics for computational and sample efficiency."""
    method: str

    # Sample efficiency
    samples_to_convergence: int = 0
    final_of: float = 0.0

    # Computational efficiency
    total_training_time: float = 0.0
    time_to_target: float = 0.0  # Time to reach target OF

    # Throughput
    episodes_per_second: float = 0.0
    evaluations_per_second: float = 0.0

    # Scalability
    speedup_vs_baseline: float = 1.0  # vs GA-SQP


@dataclass
class CommunicationMetrics:
    """Metrics for communication efficiency."""
    method: str

    # Bandwidth
    avg_bandwidth_kbps: float = 0.0
    peak_bandwidth_kbps: float = 0.0
    bandwidth_std: float = 0.0

    # Message efficiency
    message_dim: int = 0
    messages_per_step: int = 0
    compression_ratio: float = 1.0

    # Performance impact
    performance_with_comm: float = 0.0
    performance_without_comm: float = 0.0
    comm_benefit_percent: float = 0.0


@dataclass
class SafetyMetrics:
    """Metrics for safety (constraint satisfaction)."""
    method: str

    # Violation rates
    cti_violation_rate: float = 0.0
    tms_bound_violation_rate: float = 0.0
    ps_bound_violation_rate: float = 0.0

    # Severity
    max_cti_violation: float = 0.0
    mean_cti_violation: float = 0.0

    # Recovery
    recovery_rate: float = 0.0  # % of violations recovered within N steps
    mean_recovery_steps: float = 0.0

    # Safe RL specific
    lagrange_multiplier_mean: float = 0.0
    lagrange_multiplier_max: float = 0.0
    penalty_coefficient_final: float = 0.0


@dataclass
class AdaptationMetrics:
    """Metrics for continual adaptation (EWC)."""
    method: str

    # Task performance
    n_tasks: int = 0
    task_performance: List[float] = None

    # Forgetting
    forgetting_measure: float = 0.0  # Performance drop on old tasks
    backward_transfer: float = 0.0  # Negative = forgetting

    # Forward transfer
    forward_transfer: float = 0.0  # Performance gain on new tasks from old experience

    # Adaptation speed
    adaptation_time_steps: int = 0
    adaptation_time_seconds: float = 0.0

    # EWC specific
    ewc_penalty_mean: float = 0.0
    fisher_information_sparsity: float = 0.0


class LargeScaleEvaluator:
    """
    Comprehensive evaluator for Phase 5 large-scale experiments.

    Computes all metrics from raw results and produces structured output.
    """

    def __init__(self, results: List[Dict]):
        self.results = results
        self.by_method = defaultdict(list)
        self.by_system = defaultdict(list)
        self.by_pv = defaultdict(list)

        for r in results:
            config = r.get("config", r)
            self.by_method[config.get("method", "unknown")].append(r)
            self.by_system[config.get("system", "unknown")].append(r)
            self.by_pv[config.get("pv_level", 0.0)].append(r)

    def compute_scalability_metrics(self) -> Dict[str, List[ScalabilityMetrics]]:
        """Compute scalability metrics per method across systems."""
        metrics = defaultdict(list)

        # System sizes (relay counts)
        system_sizes = {
            "3bus": 6,
            "8bus": 14,
            "14bus": 20,
            "30bus": 42,
            "39bus": 46,
            "57bus": 80,
            "118bus": 186,
        }

        for method, results in self.by_method.items():
            for r in results:
                config = r.get("config", r)
                system = config.get("system", "unknown")
                n_relays = system_sizes.get(system, 0)

                # Use the system's real n_pairs (from its NetworkData
                # object) rather than a fixed flat guess — n_pairs differs
                # across the 3/8/14/30/39/57/118-bus systems this evaluator
                # spans (e.g. 8bus=20, 14bus=26/38 depending on definition,
                # 30bus=70, ...). Mirrors the correct approach used in
                # compute_safety_metrics below.
                n_pairs = NETWORKS[system]().n_pairs if system in NETWORKS else r.get("n_pairs", n_relays * 2)

                m = ScalabilityMetrics(
                    system_name=system,
                    n_relays=n_relays,
                    n_pairs=n_pairs,
                    of_value=r["best_of"],
                    cti_satisfied=r.get("cti_satisfied", False),
                    training_time=r.get("elapsed_seconds", 0.0),
                    inference_time=0.0,  # Would need separate measurement
                    memory_usage_mb=0.0,  # Would need separate measurement
                )
                m.compute_derived()
                metrics[method].append(m)

        return dict(metrics)

    def compute_generalization_metrics(self) -> Dict[str, GeneralizationMetrics]:
        """Compute generalization metrics per method."""
        metrics = {}

        # Define seen vs unseen fault types
        seen_faults = {"3ph"}

        for method, results in self.by_method.items():
            seen_ofs = []
            unseen_ofs = []
            fault_perf = {}

            for r in results:
                config = r.get("config", r)
                fault = config.get("fault_type", "3ph")
                of = r["best_of"]

                fault_perf[fault] = fault_perf.get(fault, []) + [of]

                if fault in seen_faults:
                    seen_ofs.append(of)
                else:
                    unseen_ofs.append(of)

            # Average per fault type
            for fault, ofs in fault_perf.items():
                fault_perf[fault] = float(np.mean(ofs))

            m = GeneralizationMetrics(
                method=method,
                seen_of_mean=float(np.mean(seen_ofs)) if seen_ofs else 0.0,
                unseen_of_mean=float(np.mean(unseen_ofs)) if unseen_ofs else 0.0,
                fault_type_performance=fault_perf,
            )
            m.compute_gap()
            metrics[method] = m

        return metrics

    def compute_robustness_metrics(self) -> Dict[str, Dict[str, RobustnessMetrics]]:
        """Compute robustness metrics per method and system."""
        metrics = defaultdict(dict)

        for method, results in self.by_method.items():
            # Group by system
            by_system = defaultdict(lambda: {"baseline": [], "n1": [], "noise": []})

            for r in results:
                config = r.get("config", r)
                system = config.get("system", "unknown")
                topo = config.get("topology_changes", {})
                has_outage = topo is not None and len(topo) > 0
                has_noise = config.get("sensor_noise_level", 0.0) > 0

                if has_outage:
                    by_system[system]["n1"].append(r)
                elif has_noise:
                    by_system[system]["noise"].append(r)
                else:
                    by_system[system]["baseline"].append(r)

            for system, groups in by_system.items():
                baseline_ofs = [r["best_of"] for r in groups["baseline"]]
                n1_ofs = [r["best_of"] for r in groups["n1"]]
                noise_ofs = [r["best_of"] for r in groups["noise"]]

                baseline_mean = np.mean(baseline_ofs) if baseline_ofs else 0.0
                n1_mean = np.mean(n1_ofs) if n1_ofs else 0.0
                noise_mean = np.mean(noise_ofs) if noise_ofs else 0.0

                n1_deg = ((n1_mean - baseline_mean) / baseline_mean * 100) if baseline_mean > 0 else 0.0
                noise_deg = ((noise_mean - baseline_mean) / baseline_mean * 100) if baseline_mean > 0 else 0.0

                m = RobustnessMetrics(
                    method=method,
                    system=system,
                    baseline_of=baseline_mean,
                    n1_degradation_percent=n1_deg,
                    noise_degradation_percent=noise_deg,
                )
                metrics[method][system] = m

        return dict(metrics)

    def compute_efficiency_metrics(self, baseline_method: str = "GA-SQP") -> Dict[str, EfficiencyMetrics]:
        """Compute efficiency metrics per method."""
        metrics = {}

        # Get baseline time
        baseline_times = [r.get("elapsed_seconds", 0.0) for r in self.by_method.get(baseline_method, [])]
        baseline_time = np.mean(baseline_times) if baseline_times else 1.0

        for method, results in self.by_method.items():
            times = [r.get("elapsed_seconds", 0.0) for r in results]
            ofs = [r["best_of"] for r in results]

            m = EfficiencyMetrics(
                method=method,
                total_training_time=float(np.mean(times)),
                final_of=float(np.mean(ofs)),
                speedup_vs_baseline=baseline_time / np.mean(times) if np.mean(times) > 0 else 0.0,
            )
            metrics[method] = m

        return metrics

    def compute_communication_metrics(self) -> Dict[str, CommunicationMetrics]:
        """Compute communication metrics for methods with communication."""
        metrics = {}

        for method, results in self.by_method.items():
            if "Comm" not in method:
                continue

            bw_values = [r.get("comm_bandwidth", 0.0) for r in results]
            ofs = [r["best_of"] for r in results]

            # Find corresponding non-comm method for comparison
            base_method = method.replace("-Comm", "")
            base_results = self.by_method.get(base_method, [])
            base_ofs = [r["best_of"] for r in base_results] if base_results else ofs

            m = CommunicationMetrics(
                method=method,
                avg_bandwidth_kbps=float(np.mean(bw_values)) if bw_values else 0.0,
                peak_bandwidth_kbps=float(np.max(bw_values)) if bw_values else 0.0,
                bandwidth_std=float(np.std(bw_values)) if bw_values else 0.0,
                message_dim=8,
                messages_per_step=1,
                performance_with_comm=float(np.mean(ofs)),
                performance_without_comm=float(np.mean(base_ofs)) if base_ofs else 0.0,
            )

            if m.performance_without_comm > 0:
                # OF is lower-is-better throughout this codebase (see
                # friedman_test's docstring, and n1_degradation_percent /
                # noise_degradation_percent's "(worse - baseline)/baseline"
                # convention just above). "with - without" had the sign
                # backwards for a *benefit* metric: comm cutting OF in half
                # (a strong real improvement) reported as -50%, reading as
                # comm making things worse.
                m.comm_benefit_percent = ((m.performance_without_comm - m.performance_with_comm)
                                         / m.performance_without_comm * 100)

            metrics[method] = m

        return metrics

    def compute_safety_metrics(self) -> Dict[str, SafetyMetrics]:
        """Compute safety metrics per method."""
        metrics = {}

        for method, results in self.by_method.items():
            cti_mins = []

            # Use each result's own n_pairs (looked up from its recorded
            # system) rather than a fixed flat guess — n_pairs differs
            # across the 3/8/14/30/39/57/118-bus systems this evaluator
            # spans (e.g. 8bus=20, 14bus=26/38 depending on definition,
            # 30bus=70, ...). Mirrors the correct approach already used in
            # evaluation/thesis_validation.py's cti_violation_rate metric.
            total_pairs = 0
            total_viols = 0

            for r in results:
                cti_mins.append(r.get("cti_min", 0.0))

                config = r.get("config", r)
                system = config.get("system")
                n_pairs = NETWORKS[system]().n_pairs if system in NETWORKS else r.get("n_pairs")
                if n_pairs:
                    total_pairs += n_pairs
                    total_viols += r.get("cti_violations", 0)

            m = SafetyMetrics(
                method=method,
                cti_violation_rate=total_viols / total_pairs if total_pairs > 0 else 0.0,
                max_cti_violation=float(np.max(cti_mins)) if cti_mins else 0.0,
                mean_cti_violation=float(np.mean(cti_mins)) if cti_mins else 0.0,
            )

            # Safe RL specific
            safe_stats = [r.get("safe_rl_stats", {}) for r in results if "safe_rl_stats" in r]
            if safe_stats:
                lambdas = [s.get("lambda_mean", 0.0) for s in safe_stats]
                m.lagrange_multiplier_mean = float(np.mean(lambdas))
                m.lagrange_multiplier_max = float(np.max(lambdas))

            metrics[method] = m

        return metrics

    def compute_adaptation_metrics(self) -> Dict[str, AdaptationMetrics]:
        """Compute adaptation metrics for EWC methods."""
        metrics = {}

        for method, results in self.by_method.items():
            if "EWC" not in method:
                continue

            ewc_stats_list = [r.get("ewc_stats", {}) for r in results if "ewc_stats" in r]

            if not ewc_stats_list:
                continue

            n_tasks_list = [s.get("n_tasks_learned", 0) for s in ewc_stats_list]

            m = AdaptationMetrics(
                method=method,
                n_tasks=int(np.max(n_tasks_list)) if n_tasks_list else 0,
            )

            metrics[method] = m

        return metrics

    def compute_all_metrics(self) -> Dict:
        """Compute all metrics and return comprehensive report."""
        return {
            "scalability": self.compute_scalability_metrics(),
            "generalization": self.compute_generalization_metrics(),
            "robustness": self.compute_robustness_metrics(),
            "efficiency": self.compute_efficiency_metrics(),
            "communication": self.compute_communication_metrics(),
            "safety": self.compute_safety_metrics(),
            "adaptation": self.compute_adaptation_metrics(),
        }

    def generate_report(self) -> str:
        """Generate a text report of all metrics."""
        all_metrics = self.compute_all_metrics()

        lines = [
            "=" * 80,
            "PHASE 5: LARGE-SCALE EVALUATION REPORT",
            "=" * 80,
            "",
        ]

        # Scalability
        lines.append("SCALABILITY METRICS")
        lines.append("-" * 40)
        for method, metrics in all_metrics["scalability"].items():
            lines.append(f"\n  {method}:")
            for m in metrics:
                lines.append(f"    {m.system_name}: OF={m.of_value:.4f}, "
                           f"time={m.training_time:.1f}s, time/relay={m.time_per_relay:.3f}s")

        # Generalization
        lines.append("\n\nGENERALIZATION METRICS")
        lines.append("-" * 40)
        for method, m in all_metrics["generalization"].items():
            lines.append(f"\n  {method}:")
            lines.append(f"    Seen OF: {m.seen_of_mean:.4f}")
            lines.append(f"    Unseen OF: {m.unseen_of_mean:.4f}")
            lines.append(f"    Generalization gap: {m.generalization_gap:.2f}%")
            for fault, perf in m.fault_type_performance.items():
                lines.append(f"    {fault}: {perf:.4f}")

        # Robustness
        lines.append("\n\nROBUSTNESS METRICS")
        lines.append("-" * 40)
        for method, systems in all_metrics["robustness"].items():
            lines.append(f"\n  {method}:")
            for system, m in systems.items():
                lines.append(f"    {system}: N-1 degradation={m.n1_degradation_percent:.2f}%, "
                           f"noise degradation={m.noise_degradation_percent:.2f}%")

        # Efficiency
        lines.append("\n\nEFFICIENCY METRICS")
        lines.append("-" * 40)
        for method, m in all_metrics["efficiency"].items():
            lines.append(f"\n  {method}:")
            lines.append(f"    Total time: {m.total_training_time:.1f}s")
            lines.append(f"    Speedup vs GA-SQP: {m.speedup_vs_baseline:.2f}x")

        # Communication
        lines.append("\n\nCOMMUNICATION METRICS")
        lines.append("-" * 40)
        for method, m in all_metrics["communication"].items():
            lines.append(f"\n  {method}:")
            lines.append(f"    Avg bandwidth: {m.avg_bandwidth_kbps:.3f} kbps")
            lines.append(f"    Performance benefit: {m.comm_benefit_percent:.2f}%")

        # Safety
        lines.append("\n\nSAFETY METRICS")
        lines.append("-" * 40)
        for method, m in all_metrics["safety"].items():
            lines.append(f"\n  {method}:")
            lines.append(f"    CTI violation rate: {m.cti_violation_rate:.4f}")
            lines.append(f"    Mean CTI min: {m.mean_cti_violation:.4f}s")

        lines.append("\n" + "=" * 80)

        return "\n".join(lines)


if __name__ == "__main__":
    # Example with dummy data
    dummy_results = [
        {"config": {"method": "GA-SQP", "system": "3bus", "pv_level": 0.0, "fault_type": "3ph"},
         "best_of": 1.2, "cti_satisfied": True, "cti_violations": 0, "cti_min": 0.35, "elapsed_seconds": 2.0},
        {"config": {"method": "GA-SQP", "system": "8bus", "pv_level": 0.0, "fault_type": "3ph"},
         "best_of": 5.2, "cti_satisfied": True, "cti_violations": 0, "cti_min": 0.35, "elapsed_seconds": 10.0},
        {"config": {"method": "GA-SQP", "system": "14bus", "pv_level": 0.0, "fault_type": "3ph"},
         "best_of": 8.5, "cti_satisfied": True, "cti_violations": 0, "cti_min": 0.38, "elapsed_seconds": 15.0},
        {"config": {"method": "GA-SQP", "system": "30bus", "pv_level": 0.0, "fault_type": "3ph"},
         "best_of": 12.3, "cti_satisfied": True, "cti_violations": 0, "cti_min": 0.36, "elapsed_seconds": 25.0},
        {"config": {"method": "GA-SQP", "system": "39bus", "pv_level": 0.0, "fault_type": "3ph"},
         "best_of": 15.1, "cti_satisfied": True, "cti_violations": 0, "cti_min": 0.37, "elapsed_seconds": 35.0},
        {"config": {"method": "GA-SQP", "system": "57bus", "pv_level": 0.0, "fault_type": "3ph"},
         "best_of": 22.8, "cti_satisfied": True, "cti_violations": 0, "cti_min": 0.34, "elapsed_seconds": 55.0},
        {"config": {"method": "GA-SQP", "system": "118bus", "pv_level": 0.0, "fault_type": "3ph"},
         "best_of": 45.6, "cti_satisfied": True, "cti_violations": 0, "cti_min": 0.33, "elapsed_seconds": 120.0},
        {"config": {"method": "MAPPO", "system": "8bus", "pv_level": 0.0, "fault_type": "3ph"},
         "best_of": 5.1, "cti_satisfied": True, "cti_violations": 0, "cti_min": 0.40, "elapsed_seconds": 120.0},
        {"config": {"method": "MAPPO", "system": "30bus", "pv_level": 0.0, "fault_type": "3ph"},
         "best_of": 11.8, "cti_satisfied": True, "cti_violations": 0, "cti_min": 0.39, "elapsed_seconds": 350.0},
        {"config": {"method": "MAPPO", "system": "39bus", "pv_level": 0.0, "fault_type": "3ph"},
         "best_of": 14.5, "cti_satisfied": True, "cti_violations": 0, "cti_min": 0.41, "elapsed_seconds": 480.0},
        {"config": {"method": "MAPPO-Comm", "system": "8bus", "pv_level": 0.0, "fault_type": "3ph"},
         "best_of": 4.9, "cti_satisfied": True, "cti_violations": 0, "cti_min": 0.42, "elapsed_seconds": 130.0, "comm_bandwidth": 0.256},
        {"config": {"method": "MAPPO-Comm", "system": "39bus", "pv_level": 0.0, "fault_type": "3ph"},
         "best_of": 14.0, "cti_satisfied": True, "cti_violations": 0, "cti_min": 0.43, "elapsed_seconds": 520.0, "comm_bandwidth": 0.512},
    ]

    evaluator = LargeScaleEvaluator(dummy_results)
    print(evaluator.generate_report())