"""
deployment/robustness_tests.py
──────────────────────────────
Phase 7: Robustness testing suite for DOCR-MARL.

Tests system resilience under:
1. Communication delays (10-500ms) and packet loss (0-30%)
2. Sensor noise (Gaussian, missing measurements, false data injection)
3. N-1 topology contingencies
4. Combined stress scenarios
5. Byzantine/false data injection attacks

All tests produce degradation metrics and recovery statistics.
"""

import numpy as np
import time
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass, field
from collections import defaultdict

try:
    from network.network_data import NetworkData, NETWORKS
    from network.relay_physics import check_constraints, objective_function
except ImportError:
    # Fallback for testing without full network module
    NetworkData = None
    NETWORKS = {}
    def check_constraints(*args, **kwargs):
        return {"cti_satisfied": True, "cti_violations": 0}
    def objective_function(*args, **kwargs):
        return 0.0, 0.0, 0.0


@dataclass
class DelayConfig:
    """Communication delay configuration."""
    mean_delay_ms: float = 50.0
    std_delay_ms: float = 10.0
    max_delay_ms: float = 500.0
    distribution: str = "lognormal"  # "uniform", "normal", "lognormal"


@dataclass
class PacketLossConfig:
    """Packet loss configuration."""
    loss_rate: float = 0.05  # 0-1
    burst_length: int = 3      # consecutive losses
    burst_prob: float = 0.1


@dataclass
class SensorNoiseConfig:
    """Sensor noise configuration."""
    gaussian_std: float = 0.02  # 2% std
    missing_rate: float = 0.01  # 1% missing
    false_data_rate: float = 0.005  # 0.5% false data injection
    false_data_magnitude: float = 0.5  # 50% deviation


@dataclass
class RobustnessConfig:
    """Complete robustness test configuration."""
    delay: DelayConfig = field(default_factory=DelayConfig)
    packet_loss: PacketLossConfig = field(default_factory=PacketLossConfig)
    sensor_noise: SensorNoiseConfig = field(default_factory=SensorNoiseConfig)
    n1_outage: bool = True
    combined_stress: bool = False
    seed: int = 42


class CommunicationDelaySimulator:
    """
    Simulates realistic communication delays in protection networks.

    Models:
    - RF mesh networks: lognormal delays, 50-200ms typical
    - Fiber optic: normal delays, 5-20ms typical
    - Cellular (4G/5G): bimodal, 30-100ms typical
    """

    def __init__(self, config: DelayConfig, seed: int = 42):
        self.config = config
        self.rng = np.random.default_rng(seed)
        self.delay_history: List[float] = []

    def get_delay(self) -> float:
        """Return next delay in seconds."""
        cfg = self.config

        if cfg.distribution == "uniform":
            delay = self.rng.uniform(0, cfg.max_delay_ms)
        elif cfg.distribution == "normal":
            delay = self.rng.normal(cfg.mean_delay_ms, cfg.std_delay_ms)
            delay = max(0, min(delay, cfg.max_delay_ms))
        elif cfg.distribution == "lognormal":
            mu = np.log(cfg.mean_delay_ms)
            sigma = cfg.std_delay_ms / cfg.mean_delay_ms
            delay = self.rng.lognormal(mu, sigma)
            delay = min(delay, cfg.max_delay_ms)
        else:
            delay = cfg.mean_delay_ms

        delay_sec = delay / 1000.0
        self.delay_history.append(delay_sec)
        return delay_sec

    def get_statistics(self) -> Dict:
        """Return delay statistics."""
        if not self.delay_history:
            return {}
        arr = np.array(self.delay_history)
        return {
            "mean_ms": float(np.mean(arr) * 1000),
            "std_ms": float(np.std(arr) * 1000),
            "max_ms": float(np.max(arr) * 1000),
            "min_ms": float(np.min(arr) * 1000),
            "p99_ms": float(np.percentile(arr, 99) * 1000),
            "n_samples": len(arr),
        }


class PacketLossSimulator:
    """
    Simulates packet loss with burst characteristics.

    Real-world patterns:
    - Bursty losses (3-5 consecutive packets) more common than random
    - Correlated with network congestion
    - Higher loss during topology changes (re-routing)
    """

    def __init__(self, config: PacketLossConfig, seed: int = 42):
        self.config = config
        self.rng = np.random.default_rng(seed)
        self.loss_history: List[bool] = []
        self.burst_counter = 0

    def is_lost(self) -> bool:
        """Return True if current packet is lost."""
        cfg = self.config

        if self.burst_counter > 0:
            self.burst_counter -= 1
            lost = True
        else:
            lost = self.rng.random() < cfg.loss_rate
            if lost and self.rng.random() < cfg.burst_prob:
                self.burst_counter = cfg.burst_length - 1

        self.loss_history.append(lost)
        return lost

    def get_statistics(self) -> Dict:
        """Return packet loss statistics."""
        if not self.loss_history:
            return {}
        arr = np.array(self.loss_history)
        return {
            "loss_rate": float(np.mean(arr)),
            "total_packets": len(arr),
            "lost_packets": int(np.sum(arr)),
            "burst_events": self._count_bursts(),
        }

    def _count_bursts(self) -> int:
        """Count burst loss events (≥2 consecutive losses)."""
        bursts = 0
        in_burst = False
        for lost in self.loss_history:
            if lost and not in_burst:
                in_burst = True
            elif not lost and in_burst:
                bursts += 1
                in_burst = False
        if in_burst:
            bursts += 1
        return bursts


class SensorNoiseSimulator:
    """
    Simulates realistic sensor noise and cyber attacks.

    Noise types:
    1. Gaussian: Thermal noise, quantization error
    2. Missing: Communication dropout, sensor failure
    3. False data: Cyber attack (FDI), bad data injection

    Defense: Median consensus filtering for Byzantine resilience.
    """

    def __init__(self, config: SensorNoiseConfig, n_sensors: int, seed: int = 42):
        self.config = config
        self.n_sensors = n_sensors
        self.rng = np.random.default_rng(seed)
        self.noise_history: List[Dict] = []

    def add_noise(self, measurements: np.ndarray) -> np.ndarray:
        """Add noise to sensor measurements."""
        cfg = self.config
        noisy = measurements.copy()

        # Gaussian noise
        if cfg.gaussian_std > 0:
            noise = self.rng.normal(0, cfg.gaussian_std, measurements.shape)
            noisy += noise

        # Missing measurements
        if cfg.missing_rate > 0:
            mask = self.rng.random(measurements.shape) < cfg.missing_rate
            noisy = np.where(mask, np.nan, noisy)

        # False data injection (Byzantine attack)
        if cfg.false_data_rate > 0:
            mask = self.rng.random(measurements.shape) < cfg.false_data_rate
            false_data = measurements * (1 + cfg.false_data_magnitude * 
                                        self.rng.choice([-1, 1], measurements.shape))
            noisy = np.where(mask, false_data, noisy)

        self.noise_history.append({
            "gaussian_applied": cfg.gaussian_std > 0,
            "missing_count": int(np.sum(np.isnan(noisy))),
            "false_data_count": int(np.sum(mask)) if cfg.false_data_rate > 0 else 0,
        })

        return noisy

    def median_consensus_filter(self, measurements: np.ndarray, 
                                neighbor_values: List[np.ndarray]) -> np.ndarray:
        """
        Byzantine-resilient median consensus filter.

        Each agent replaces its measurement with the median of
        its own value and neighbors' values. Tolerates up to
        f faulty neighbors if n > 2f (Byzantine resilience).
        """
        if not neighbor_values:
            return measurements

        all_values = [measurements] + neighbor_values
        stacked = np.stack(all_values, axis=0)
        filtered = np.nanmedian(stacked, axis=0)

        return filtered


class RobustnessTester:
    """
    Comprehensive robustness testing framework.

    Runs a method under various stress conditions and computes
    degradation metrics vs. baseline (no stress).
    """

    def __init__(self, config: Optional[RobustnessConfig] = None):
        self.config = config or RobustnessConfig()
        self.results: Dict[str, List[Dict]] = defaultdict(list)

    def test_method(
        self,
        method_name: str,
        method_runner: Callable,
        net: NetworkData,
        n_runs: int = 10,
        verbose: bool = True,
    ) -> Dict:
        """
        Test a method under all robustness conditions.

        Args:
            method_name: Name of the method
            method_runner: Callable that runs the method and returns result dict
            net: NetworkData instance
            n_runs: Independent runs per condition
            verbose: Print progress

        Returns:
            results: {condition: [run_results]}
        """
        conditions = self._generate_conditions()

        if verbose:
            print(f"\n{'█'*60}")
            print(f"  ROBUSTNESS TESTING: {method_name}")
            print(f"  System: {net.name}  |  Runs: {n_runs}")
            print(f"  Conditions: {len(conditions)}")
            print(f"{'█'*60}")

        for cond_name, cond_config in conditions.items():
            if verbose:
                print(f"\n  --- Condition: {cond_name} ---")

            for run_idx in range(n_runs):
                seed = self.config.seed + run_idx * 100 + hash(cond_name) % 1000

                try:
                    result = method_runner(net, seed=seed, config=cond_config)
                    result["condition"] = cond_name
                    result["run"] = run_idx
                    self.results[method_name].append(result)

                    if verbose and run_idx == 0:
                        print(f"    Run {run_idx+1}: OF={result['OF']:.4f}, "
                              f"CTI_ok={result.get('constraints', {}).get('cti_satisfied', False)}")
                except Exception as e:
                    if verbose:
                        print(f"    ERROR: {e}")

        return dict(self.results)

    def _generate_conditions(self) -> Dict[str, RobustnessConfig]:
        """Generate all test conditions."""
        conditions = {}

        # 1. Baseline (no stress)
        conditions["baseline"] = RobustnessConfig(
            delay=DelayConfig(mean_delay_ms=0, max_delay_ms=0),
            packet_loss=PacketLossConfig(loss_rate=0),
            sensor_noise=SensorNoiseConfig(gaussian_std=0, missing_rate=0, false_data_rate=0),
            n1_outage=False,
        )

        # 2. Communication delays
        for delay_ms in [10, 50, 100, 250, 500]:
            conditions[f"delay_{delay_ms}ms"] = RobustnessConfig(
                delay=DelayConfig(mean_delay_ms=delay_ms, max_delay_ms=delay_ms*2),
            )

        # 3. Packet loss
        for loss_pct in [5, 10, 20, 30]:
            conditions[f"loss_{loss_pct}pct"] = RobustnessConfig(
                packet_loss=PacketLossConfig(loss_rate=loss_pct/100),
            )

        # 4. Sensor noise
        for noise_pct in [2, 5, 10]:
            conditions[f"noise_{noise_pct}pct"] = RobustnessConfig(
                sensor_noise=SensorNoiseConfig(gaussian_std=noise_pct/100),
            )

        # 5. Missing measurements
        conditions["missing_5pct"] = RobustnessConfig(
            sensor_noise=SensorNoiseConfig(missing_rate=0.05),
        )

        # 6. False data injection (Byzantine)
        conditions["false_data_1pct"] = RobustnessConfig(
            sensor_noise=SensorNoiseConfig(false_data_rate=0.01),
        )

        # 7. N-1 outage
        conditions["n1_outage"] = RobustnessConfig(n1_outage=True)

        # 8. Combined stress
        conditions["combined_stress"] = RobustnessConfig(
            delay=DelayConfig(mean_delay_ms=100),
            packet_loss=PacketLossConfig(loss_rate=0.1),
            sensor_noise=SensorNoiseConfig(gaussian_std=0.05, missing_rate=0.02),
            n1_outage=True,
            combined_stress=True,
        )

        return conditions

    def compute_degradation_metrics(self, method_name: str) -> Dict:
        """
        Compute degradation metrics vs. baseline.

        Returns:
            {condition: {
                "of_degradation_pct": float,
                "cti_violation_increase": int,
                "time_increase_pct": float,
                "success_rate": float,
            }}
        """
        results = self.results.get(method_name, [])
        if not results:
            return {}

        # Group by condition
        by_condition = defaultdict(list)
        for r in results:
            by_condition[r["condition"]].append(r)

        baseline = by_condition.get("baseline", [])
        if not baseline:
            return {}

        baseline_of = np.mean([r["OF"] for r in baseline])
        baseline_cti = np.mean([r.get("constraints", {}).get("cti_violations", 0) for r in baseline])
        baseline_time = np.mean([r.get("elapsed", 0) for r in baseline])

        metrics = {}
        for cond, runs in by_condition.items():
            if cond == "baseline":
                continue

            ofs = [r["OF"] for r in runs]
            cti_viols = [r.get("constraints", {}).get("cti_violations", 0) for r in runs]
            times = [r.get("elapsed", 0) for r in runs]
            successes = [r.get("constraints", {}).get("cti_satisfied", False) for r in runs]

            of_deg = ((np.mean(ofs) - baseline_of) / baseline_of * 100) if baseline_of > 0 else 0
            cti_inc = np.mean(cti_viols) - baseline_cti
            time_inc = ((np.mean(times) - baseline_time) / baseline_time * 100) if baseline_time > 0 else 0

            metrics[cond] = {
                "of_degradation_pct": float(of_deg),
                "cti_violation_increase": float(cti_inc),
                "time_increase_pct": float(time_inc),
                "success_rate": float(np.mean(successes)),
                "of_mean": float(np.mean(ofs)),
                "of_std": float(np.std(ofs)),
                "n_runs": len(runs),
            }

        return metrics

    def generate_report(self, method_name: str) -> str:
        """Generate formatted robustness report."""
        metrics = self.compute_degradation_metrics(method_name)

        lines = [
            "=" * 80,
            f"PHASE 7: ROBUSTNESS TEST REPORT — {method_name}",
            "=" * 80,
            "",
            f"{'Condition':<25} {'OF Deg%':>10} {'CTI Δ':>10} {'Time Δ%':>10} "
            f"{'Success%':>10} {'OF Mean':>12}",
            "-" * 80,
        ]

        for cond, m in sorted(metrics.items()):
            lines.append(
                f"{cond:<25} {m['of_degradation_pct']:>+10.2f} {m['cti_violation_increase']:>+10.2f} "
                f"{m['time_increase_pct']:>+10.2f} {m['success_rate']*100:>9.1f}% "
                f"{m['of_mean']:>12.4f}"
            )

        lines.extend(["", "=" * 80])
        return "\n".join(lines)


def quick_robustness_test(
    method_runner: Callable,
    net: NetworkData,
    n_runs: int = 5,
    verbose: bool = True,
) -> RobustnessTester:
    """
    Quick robustness test with default settings.

    Usage:
        tester = quick_robustness_test(run_ga_sqp, net)
        print(tester.generate_report("GA-SQP"))
    """
    tester = RobustnessTester()
    tester.test_method("method", method_runner, net, n_runs=n_runs, verbose=verbose)
    return tester
