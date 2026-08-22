"""
deployment/robustness_tests.py
──────────────────────────────
Phase 7: Robustness testing suite for DOCR-MARL.

Tests system resilience under:
1. Communication delays (10–500 ms) and packet loss (0–30%)
2. Sensor noise (Gaussian, missing measurements, false data injection)
3. N-1 topology contingencies
4. Combined stress scenarios
5. Byzantine / false data injection attacks

All tests produce degradation metrics and recovery statistics.
"""

from __future__ import annotations

import zlib
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from typing import Dict, List, Tuple, Optional, Callable

__all__ = [
    "DelayConfig",
    "PacketLossConfig",
    "SensorNoiseConfig",
    "RobustnessConfig",
    "CommunicationDelaySimulator",
    "PacketLossSimulator",
    "SensorNoiseSimulator",
    "RobustnessTester",
    "quick_robustness_test",
]

# ── Optional network module imports ───────────────────────────────────────

try:
    from network.network_data import NetworkData, NETWORKS
    from network.relay_physics import check_constraints, objective_function
except ImportError:
    NetworkData = None  # type: ignore[misc,assignment]
    NETWORKS = {}  # type: ignore[misc]

    def check_constraints(*args, **kwargs):  # type: ignore[misc]
        return {"cti_satisfied": True, "cti_violations": 0}

    def objective_function(*args, **kwargs):  # type: ignore[misc]
        return 0.0, 0.0, 0.0


# ── Configuration dataclasses ───────────────────────────────────────────────

@dataclass
class DelayConfig:
    """Communication delay configuration."""

    mean_delay_ms: float = 50.0
    std_delay_ms: float = 10.0
    max_delay_ms: float = 500.0
    distribution: str = "lognormal"  # "uniform" | "normal" | "lognormal"


@dataclass
class PacketLossConfig:
    """Packet loss configuration."""

    loss_rate: float = 0.05  # 0–1
    burst_length: int = 3    # consecutive losses
    burst_prob: float = 0.1


@dataclass
class SensorNoiseConfig:
    """Sensor noise configuration."""

    gaussian_std: float = 0.02       # 2% std
    missing_rate: float = 0.01       # 1% missing
    false_data_rate: float = 0.005   # 0.5% false data injection
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


# ── Simulators ──────────────────────────────────────────────────────────────

class CommunicationDelaySimulator:
    """Simulates realistic communication delays in protection networks.

    Models:
    - RF mesh networks: lognormal delays, 50–200 ms typical
    - Fiber optic: normal delays, 5–20 ms typical
    - Cellular (4G/5G): bimodal, 30–100 ms typical
    """

    def __init__(self, config: DelayConfig, seed: int = 42) -> None:
        self.config = config
        self._rng = np.random.default_rng(seed)
        self._delay_history: list[float] = []

    def get_delay(self) -> float:
        """Return next delay in seconds."""
        cfg = self.config

        if cfg.distribution == "uniform":
            delay = self._rng.uniform(0.0, cfg.max_delay_ms)
        elif cfg.distribution == "normal":
            delay = self._rng.normal(cfg.mean_delay_ms, cfg.std_delay_ms)
            delay = max(0.0, min(delay, cfg.max_delay_ms))
        elif cfg.distribution == "lognormal":
            # Exact moment matching for a lognormal: if X ~
            # Lognormal(mu, sigma), E[X] = exp(mu + sigma^2/2) and
            # Var[X] = (exp(sigma^2) - 1) * exp(2*mu + sigma^2). Setting
            # mu = log(mean_delay_ms) directly (the previous code) makes
            # E[X] = mean_delay_ms * exp(sigma^2/2) — a systematic
            # upward bias, worse the larger std_delay_ms is relative to
            # the mean. Solving both equations for the desired
            # (mean_delay_ms, std_delay_ms) gives the below.
            cv_sq = (cfg.std_delay_ms / cfg.mean_delay_ms) ** 2
            sigma = np.sqrt(np.log(1.0 + cv_sq))
            mu = np.log(cfg.mean_delay_ms) - sigma ** 2 / 2.0
            delay = self._rng.lognormal(mu, sigma)
            delay = min(delay, cfg.max_delay_ms)
        else:
            delay = cfg.mean_delay_ms

        delay_sec = delay / 1000.0
        self._delay_history.append(delay_sec)
        return delay_sec

    def get_statistics(self) -> dict:
        """Return delay statistics."""
        if not self._delay_history:
            return {}

        arr = np.array(self._delay_history)
        return {
            "mean_ms": float(np.mean(arr) * 1000.0),
            "std_ms": float(np.std(arr) * 1000.0),
            "max_ms": float(np.max(arr) * 1000.0),
            "min_ms": float(np.min(arr) * 1000.0),
            "p99_ms": float(np.percentile(arr, 99) * 1000.0),
            "n_samples": len(arr),
        }


class PacketLossSimulator:
    """Simulates packet loss with burst characteristics.

    Real-world patterns:
    - Bursty losses (3–5 consecutive packets) more common than random
    - Correlated with network congestion
    - Higher loss during topology changes (re-routing)
    """

    def __init__(self, config: PacketLossConfig, seed: int = 42) -> None:
        self.config = config
        self._rng = np.random.default_rng(seed)
        self._loss_history: list[bool] = []
        self._burst_counter = 0

    def is_lost(self) -> bool:
        """Return True if current packet is lost."""
        cfg = self.config

        if self._burst_counter > 0:
            self._burst_counter -= 1
            lost = True
        else:
            lost = self._rng.random() < cfg.loss_rate
            if lost and self._rng.random() < cfg.burst_prob:
                self._burst_counter = cfg.burst_length - 1

        self._loss_history.append(lost)
        return lost

    def get_statistics(self) -> dict:
        """Return packet loss statistics."""
        if not self._loss_history:
            return {}

        arr = np.array(self._loss_history)
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
        for lost in self._loss_history:
            if lost and not in_burst:
                in_burst = True
            elif not lost and in_burst:
                bursts += 1
                in_burst = False
        if in_burst:
            bursts += 1
        return bursts


class SensorNoiseSimulator:
    """Simulates realistic sensor noise and cyber attacks.

    Noise types:
    1. Gaussian: Thermal noise, quantization error
    2. Missing: Communication dropout, sensor failure
    3. False data: Cyber attack (FDI), bad data injection

    Defense: Median consensus filtering for Byzantine resilience.
    """

    def __init__(
        self,
        config: SensorNoiseConfig,
        n_sensors: int,
        seed: int = 42,
    ) -> None:
        self.config = config
        self.n_sensors = n_sensors
        self._rng = np.random.default_rng(seed)
        self._noise_history: list[dict] = []

    def add_noise(self, measurements: np.ndarray) -> np.ndarray:
        """Add noise to sensor measurements.

        Parameters
        ----------
        measurements : np.ndarray
            Clean measurement array.

        Returns
        -------
        np.ndarray
            Noisy measurements (may contain NaN for missing values).
        """
        cfg = self.config
        noisy = measurements.copy()

        # Gaussian noise
        if cfg.gaussian_std > 0.0:
            noise = self._rng.normal(0.0, cfg.gaussian_std, measurements.shape)
            noisy += noise

        # Missing measurements
        if cfg.missing_rate > 0.0:
            mask = self._rng.random(measurements.shape) < cfg.missing_rate
            noisy = np.where(mask, np.nan, noisy)

        # False data injection (Byzantine attack)
        false_data_count = 0
        if cfg.false_data_rate > 0.0:
            mask = self._rng.random(measurements.shape) < cfg.false_data_rate
            false_data = measurements * (
                1.0
                + cfg.false_data_magnitude
                * self._rng.choice([-1, 1], size=measurements.shape)
            )
            noisy = np.where(mask, false_data, noisy)
            false_data_count = int(np.sum(mask))

        self._noise_history.append({
            "gaussian_applied": cfg.gaussian_std > 0.0,
            "missing_count": int(np.sum(np.isnan(noisy))),
            "false_data_count": false_data_count,
        })

        return noisy

    def median_consensus_filter(
        self,
        measurements: np.ndarray,
        neighbor_values: list[np.ndarray],
    ) -> np.ndarray:
        """Byzantine-resilient median consensus filter.

        Each agent replaces its measurement with the median of
        its own value and neighbors' values. Tolerates up to
        f faulty neighbors if n > 2f (Byzantine resilience).

        Parameters
        ----------
        measurements : np.ndarray
            Own measurements.
        neighbor_values : list[np.ndarray]
            Neighbor measurement arrays.

        Returns
        -------
        np.ndarray
            Element-wise median across all agents.
        """
        if not neighbor_values:
            return measurements

        all_values = [measurements] + neighbor_values
        stacked = np.stack(all_values, axis=0)
        return np.nanmedian(stacked, axis=0)


# ── Robustness Tester ───────────────────────────────────────────────────────

class RobustnessTester:
    """Comprehensive robustness testing framework.

    Runs a method under various stress conditions and computes
    degradation metrics vs. baseline (no stress).
    """

    def __init__(self, config: RobustnessConfig | None = None) -> None:
        self.config = config or RobustnessConfig()
        self.results: dict[str, list[dict]] = defaultdict(list)

    def test_method(
        self,
        method_name: str,
        method_runner: Callable,
        net: NetworkData,
        n_runs: int = 10,
        verbose: bool = True,
    ) -> dict:
        """Test a method under all robustness conditions.

        Delegates to test_method_with_stress(). This used to call
        ``method_runner(net, seed=seed, config=cond_config)`` directly on
        the UNPERTURBED net -- but no real baseline runner in this
        codebase (run_ga_sqp, run_jaya, etc.) accepts a `config` kwarg,
        so every call silently raised inside the try/except below and
        self.results stayed empty regardless of the requested condition.
        Even a caller that worked around the crash by accepting-and-
        discarding `config` (as mains/main6.py's run_phase7_robustness
        used to) would still measure only seed-to-seed noise on the
        identical, unperturbed network -- never a real stress condition.
        test_method_with_stress() applies each condition for real via
        apply_stress_to_net() before calling
        ``method_runner(stressed_net, seed=seed)``, which matches every
        real baseline runner's actual signature.

        Parameters
        ----------
        method_name : str
            Name of the method under test.
        method_runner : Callable
            Callable accepting ``(stressed_net, seed)`` and returning a
            result dict (same contract as test_method_with_stress).
        net : NetworkData
            Network data instance.
        n_runs : int
            Independent runs per condition.
        verbose : bool
            Print progress to stdout.

        Returns
        -------
        dict
            {condition: [run_results, ...]}
        """
        return self.test_method_with_stress(
            method_name, method_runner, net, n_runs=n_runs, verbose=verbose
        )

    def _generate_conditions(self) -> dict[str, RobustnessConfig]:
        """Generate all test conditions."""
        conditions: dict[str, RobustnessConfig] = {}

        # 1. Baseline (no stress)
        conditions["baseline"] = RobustnessConfig(
            delay=DelayConfig(mean_delay_ms=0.0, max_delay_ms=0.0),
            packet_loss=PacketLossConfig(loss_rate=0.0),
            sensor_noise=SensorNoiseConfig(
                gaussian_std=0.0, missing_rate=0.0, false_data_rate=0.0
            ),
            n1_outage=False,
        )

        # 2. Communication delays
        for delay_ms in (10, 50, 100, 250, 500):
            conditions[f"delay_{delay_ms}ms"] = RobustnessConfig(
                delay=DelayConfig(mean_delay_ms=delay_ms, max_delay_ms=delay_ms * 2),
            )

        # 3. Packet loss
        for loss_pct in (5, 10, 20, 30):
            conditions[f"loss_{loss_pct}pct"] = RobustnessConfig(
                packet_loss=PacketLossConfig(loss_rate=loss_pct / 100.0),
            )

        # 4. Sensor noise
        for noise_pct in (2, 5, 10):
            conditions[f"noise_{noise_pct}pct"] = RobustnessConfig(
                sensor_noise=SensorNoiseConfig(gaussian_std=noise_pct / 100.0),
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
            delay=DelayConfig(mean_delay_ms=100.0),
            packet_loss=PacketLossConfig(loss_rate=0.1),
            sensor_noise=SensorNoiseConfig(
                gaussian_std=0.05, missing_rate=0.02
            ),
            n1_outage=True,
            combined_stress=True,
        )

        return conditions

    def compute_degradation_metrics(self, method_name: str) -> dict:
        """Compute degradation metrics vs. baseline.

        Returns
        -------
        dict
            {condition: {
                "of_degradation_pct": float,
                "cti_violation_increase": float,
                "time_increase_pct": float,
                "success_rate": float,
                "of_mean": float,
                "of_std": float,
                "n_runs": int,
            }}
        """
        results = self.results.get(method_name, [])
        if not results:
            return {}

        # Group by condition
        by_condition: dict[str, list[dict]] = defaultdict(list)
        for r in results:
            by_condition[r["condition"]].append(r)

        baseline = by_condition.get("baseline", [])
        if not baseline:
            return {}

        baseline_of = float(np.mean([r["OF"] for r in baseline]))
        baseline_cti = float(
            np.mean([r.get("constraints", {}).get("cti_violations", 0) for r in baseline])
        )
        baseline_time = float(
            np.mean([r.get("elapsed", 0.0) for r in baseline])
        )

        metrics: dict[str, dict] = {}
        for cond, runs in by_condition.items():
            if cond == "baseline":
                continue

            ofs = [r["OF"] for r in runs]
            cti_viols = [
                r.get("constraints", {}).get("cti_violations", 0) for r in runs
            ]
            times = [r.get("elapsed", 0.0) for r in runs]
            successes = [
                r.get("constraints", {}).get("cti_satisfied", False) for r in runs
            ]

            of_deg = (
                ((np.mean(ofs) - baseline_of) / baseline_of * 100.0)
                if baseline_of > 0.0
                else 0.0
            )
            cti_inc = float(np.mean(cti_viols) - baseline_cti)
            time_inc = (
                ((np.mean(times) - baseline_time) / baseline_time * 100.0)
                if baseline_time > 0.0
                else 0.0
            )

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
                f"{cond:<25} {m['of_degradation_pct']:>+10.2f} "
                f"{m['cti_violation_increase']:>+10.2f} "
                f"{m['time_increase_pct']:>+10.2f} "
                f"{m['success_rate']*100:>9.1f}% "
                f"{m['of_mean']:>12.4f}"
            )

        lines.extend(["", "=" * 80])
        return "\n".join(lines)

    def apply_stress_to_net(
        self,
        net: NetworkData,
        cond_config: RobustnessConfig,
        seed: int = 0,
    ) -> NetworkData:
        """Return a *new* NetworkData with fault currents perturbed.

        Perturbation rules
        ------------------
        Sensor noise (gaussian_std > 0):
            If_pri, If_bkp scaled by N(1, σ) for each pair.

        Missing measurements (missing_rate > 0):
            A fraction of pairs have If_bkp replaced by If_pri * 0.10
            (relay sees almost no current — simulates CT failure on backup).

        False data injection (false_data_rate > 0):
            A fraction of pairs have If_bkp multiplied by (1 ± magnitude).

        Communication delay (mean_delay_ms > 0):
            Dead-time approximated as a small upward bias on CTI requirement,
            modelled by reducing If_bkp by delay_fraction = mean_delay / 5000
            (up to 20%).

        N-1 outage:
            All fault currents scaled by the topology outage factor (0.8).
        """
        import copy

        rng = np.random.default_rng(seed)
        cfg = cond_config
        pairs = copy.deepcopy(net.pairs)

        delay_factor = 1.0
        if cfg.delay.mean_delay_ms > 0.0:
            delay_factor = max(0.80, 1.0 - cfg.delay.mean_delay_ms / 5000.0)

        outage_factor = 0.80 if cfg.n1_outage else 1.0

        for i, p in enumerate(pairs):
            scale_pri = outage_factor
            scale_bkp = outage_factor * delay_factor

            # Gaussian noise on fault current readings
            if cfg.sensor_noise.gaussian_std > 0.0:
                noise_pri = rng.normal(1.0, cfg.sensor_noise.gaussian_std)
                noise_bkp = rng.normal(1.0, cfg.sensor_noise.gaussian_std)
                scale_pri *= max(0.5, noise_pri)
                scale_bkp *= max(0.5, noise_bkp)

            # Missing measurement on backup side
            if cfg.sensor_noise.missing_rate > 0.0:
                if rng.random() < cfg.sensor_noise.missing_rate:
                    scale_bkp *= 0.10  # backup sees almost nothing

            # False data injection
            if cfg.sensor_noise.false_data_rate > 0.0:
                if rng.random() < cfg.sensor_noise.false_data_rate:
                    direction = rng.choice([-1, 1])
                    fdi_factor = 1.0 + direction * cfg.sensor_noise.false_data_magnitude
                    scale_bkp *= max(0.1, fdi_factor)

            # Packet loss: lost measurements → relay uses last known (stale) value
            if cfg.packet_loss.loss_rate > 0.0:
                if rng.random() < cfg.packet_loss.loss_rate:
                    scale_bkp *= rng.uniform(0.90, 1.10)  # stale ±10%

            pairs[i] = type(p)(
                primary=p.primary,
                backup=p.backup,
                If_pri=max(1.0, p.If_pri * scale_pri),
                If_bkp=max(1.0, p.If_bkp * scale_bkp),
                bus_pri=p.bus_pri,
                bus_bkp=p.bus_bkp,
            )

        # Build a new NetworkData with the perturbed pairs
        stressed_net = type(net)(
            name=f"{net.name}_stressed",
            n_relays=net.n_relays,
            pairs=pairs,
            Iload_max=net.Iload_max.copy(),
            If_min=net.If_min.copy(),
            OLF=net.OLF,
            n_buses=net.n_buses,
            base_voltage=net.base_voltage,
            base_mva=net.base_mva,
            dg_buses=net.dg_buses,
            line_data=net.line_data,
            trafo_data=net.trafo_data,
        )
        return stressed_net

    def test_method_with_stress(
        self,
        method_name: str,
        method_runner: Callable,
        net: NetworkData,
        n_runs: int = 5,
        verbose: bool = True,
    ) -> dict:
        """Test with fault-current perturbations applied directly to NetworkData.

        This is the recommended entry point for Objective 5 robustness testing.
        Every condition produces genuinely different OF and CTI outcomes
        regardless of whether ``method_runner`` is stress-aware.

        Parameters
        ----------
        method_name : str
            Name of the method under test.
        method_runner : Callable
            Callable accepting ``(stressed_net, seed)`` and returning a result dict.
        net : NetworkData
            Base network data (unmodified).
        n_runs : int
            Independent runs per condition.
        verbose : bool
            Print progress.

        Returns
        -------
        dict
            {condition: [run_result, ...]}
        """
        conditions = self._generate_conditions()

        if verbose:
            print(f"\n{'█'*60}")
            print(f"  ROBUSTNESS TESTING (with fault-current stress): {method_name}")
            print(f"  System: {net.name}  |  Runs: {n_runs}")
            print(f"  Conditions: {len(conditions)}")
            print(f"{'█'*60}")

        for cond_name, cond_config in conditions.items():
            if verbose:
                print(f"\n  --- Condition: {cond_name} ---")

            for run_idx in range(n_runs):
                seed = (
                    self.config.seed
                    + run_idx * 100
                    + (zlib.crc32(cond_name.encode()) & 0xFFF)
                )
                stressed_net = self.apply_stress_to_net(net, cond_config, seed=seed)

                try:
                    result = method_runner(stressed_net, seed=seed)
                    result["condition"] = cond_name
                    result["run"] = run_idx
                    result["stress_applied"] = True
                    self.results[method_name].append(result)

                    if verbose and run_idx == 0:
                        cti_ok = result.get("constraints", {}).get(
                            "cti_satisfied", False
                        )
                        cti_viols = result.get("constraints", {}).get(
                            "cti_violations", "?"
                        )
                        print(
                            f"    Run 1: OF={result.get('OF', float('nan')):.4f}  "
                            f"CTI_ok={cti_ok}  violations={cti_viols}"
                        )
                except Exception as exc:
                    if verbose:
                        print(f"    ERROR in run {run_idx}: {exc}")

        return dict(self.results)


# ── Convenience function ──────────────────────────────────────────────────

def quick_robustness_test(
    method_runner: Callable,
    net: NetworkData,
    n_runs: int = 5,
    verbose: bool = True,
) -> RobustnessTester:
    """Quick robustness test with default settings.

    Usage::

        tester = quick_robustness_test(run_ga_sqp, net)
        print(tester.generate_report("method"))

    method_runner must accept ``(stressed_net, seed=...)`` -- e.g.
    ``lambda net, seed=42: run_ga_sqp(net, seed=seed)`` -- matching
    test_method_with_stress()'s contract (the net it receives is already
    perturbed for the current condition; no `config` kwarg is passed).
    """
    tester = RobustnessTester()
    tester.test_method_with_stress("method", method_runner, net, n_runs=n_runs, verbose=verbose)
    return tester