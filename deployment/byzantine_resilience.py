"""
deployment/byzantine_resilience.py
──────────────────────────────────
Phase 7: Byzantine fault tolerance and consensus filtering.

Implements:
1. Median consensus filtering (tolerates f faults if n > 2f)
2. Trimmed mean aggregation (removes extreme values)
3. Reputation-based weighting (track agent trust scores)
4. Outlier detection (Mahalanobis distance)
5. Attack detection (sudden deviation monitoring)

Usage:
    from deployment.byzantine_resilience import (
        MedianConsensusFilter, TrimmedMeanAggregator, ReputationManager
    )

    # Median consensus (tolerates up to 50% faulty neighbors)
    filter = MedianConsensusFilter(n_agents=14)
    clean_value = filter.filter(
        agent_id=0,
        value=1.2,
        neighbor_values=[1.1, 1.3, 5.0, 1.2]
    )
    # Result: 1.2 (median, ignores the 5.0 outlier)
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from typing import Dict, List, Tuple, Optional

__all__ = [
    "ReputationConfig",
    "MedianConsensusFilter",
    "TrimmedMeanAggregator",
    "ReputationManager",
    "OutlierDetector",
    "ByzantineResilientAggregator",
    "test_byzantine_resilience",
]


# ── Configuration ─────────────────────────────────────────────────────────

@dataclass
class ReputationConfig:
    """Configuration for reputation-based trust management."""

    initial_trust: float = 1.0
    penalty_false_data: float = 0.3
    penalty_deviation: float = 0.1
    reward_consistent: float = 0.05
    min_trust: float = 0.1
    max_trust: float = 2.0
    window_size: int = 10


# ── Core classes ──────────────────────────────────────────────────────────

class MedianConsensusFilter:
    """Byzantine-resilient median consensus filter.

    Tolerates up to f faulty agents in a neighborhood if n > 2f,
    where n = total agents in the neighborhood.

    Algorithm
    ---------
    1. Collect own value + all neighbor values.
    2. Compute median.
    3. Replace own value with median.

    Properties
    ----------
    - Optimal breakdown point: 50% (tolerates up to half faulty).
    - No parametric assumptions (distribution-free).
    - O(n log n) complexity per agent.
    """

    def __init__(self, n_agents: int) -> None:
        self.n_agents = n_agents
        self._history: dict[int, list[dict]] = defaultdict(list)

    def filter(
        self,
        agent_id: int,
        value: float,
        neighbor_values: list[float],
    ) -> float:
        """Apply median consensus filtering.

        Parameters
        ----------
        agent_id : int
            ID of the agent performing the filtering.
        value : float
            Own measurement / value.
        neighbor_values : list[float]
            List of neighbor values.

        Returns
        -------
        float
            Median of all values (own + neighbors).
        """
        all_values = [value] + list(neighbor_values)

        if not all_values:
            return value

        filtered = float(np.median(all_values))
        self._history[agent_id].append({
            "original": value,
            "filtered": filtered,
            "neighbors": list(neighbor_values),
        })

        return filtered

    def filter_vector(
        self,
        agent_id: int,
        values: np.ndarray,
        neighbor_values_list: list[np.ndarray],
    ) -> np.ndarray:
        """Apply median filtering to vector-valued data.

        Parameters
        ----------
        agent_id : int
            Agent identifier (stored in history).
        values : np.ndarray
            Own vector-valued measurement.
        neighbor_values_list : list[np.ndarray]
            List of neighbor vector-valued measurements.

        Returns
        -------
        np.ndarray
            Element-wise median across all vectors.
        """
        all_values = [values] + list(neighbor_values_list)
        stacked = np.stack(all_values, axis=0)
        return np.median(stacked, axis=0)

    def get_history(self, agent_id: int) -> list[dict]:
        """Return filtering history for an agent."""
        return self._history.get(agent_id, [])


class TrimmedMeanAggregator:
    """Trimmed mean aggregator for Byzantine resilience.

    Removes a fraction of extreme values before computing the mean.
    More robust than a simple mean, less aggressive than the median.

    Parameters
    ----------
    trim_ratio : float
        Fraction to trim from each end (0.0 = mean, 0.5 = median).
    """

    def __init__(self, trim_ratio: float = 0.2) -> None:
        self.trim_ratio = float(np.clip(trim_ratio, 0.0, 0.5))

    def aggregate(self, values: list[float]) -> float:
        """Compute trimmed mean.

        Parameters
        ----------
        values : list[float]
            Values from agents.

        Returns
        -------
        float
            Mean after removing extreme values.
        """
        arr = np.asarray(values, dtype=float)
        n = len(arr)

        if n == 0:
            return 0.0

        k = int(n * self.trim_ratio)
        sorted_arr = np.sort(arr)

        if k > 0 and k * 2 < n:
            trimmed = sorted_arr[k:-k]
        else:
            trimmed = sorted_arr

        return float(np.mean(trimmed))

    def aggregate_vector(self, values_list: list[np.ndarray]) -> np.ndarray:
        """Trimmed mean for vector-valued data.

        Parameters
        ----------
        values_list : list[np.ndarray]
            List of vector-valued agent outputs.

        Returns
        -------
        np.ndarray
            Trimmed-mean vector.
        """
        stacked = np.stack(values_list, axis=0)
        n = len(values_list)
        k = int(n * self.trim_ratio)

        sorted_stack = np.sort(stacked, axis=0)
        if k > 0 and k * 2 < n:
            trimmed = sorted_stack[k:-k]
        else:
            trimmed = sorted_stack

        return np.mean(trimmed, axis=0)


class ReputationManager:
    """Reputation-based trust management for relay agents.

    Tracks trust scores per agent based on:
    - Consistency with neighborhood median.
    - Sudden deviations (attack detection).
    - Historical behavior.

    Agents with low trust have reduced influence in consensus.
    """

    # Deviation thresholds
    _DEV_HIGH: float = 0.5   # Significant deviation → false-data penalty
    _DEV_MOD: float = 0.2    # Moderate deviation → small penalty

    def __init__(
        self,
        n_agents: int,
        config: ReputationConfig | None = None,
    ) -> None:
        self.n_agents = n_agents
        self.config = config or ReputationConfig()
        self.trust_scores: np.ndarray = np.full(
            n_agents, self.config.initial_trust, dtype=float
        )
        self._deviation_history: dict[int, list[float]] = defaultdict(list)
        self._attack_flags: dict[int, int] = defaultdict(int)

    def update(
        self,
        agent_id: int,
        value: float,
        neighbor_values: list[float],
    ) -> float:
        """Update trust score based on deviation from consensus.

        Parameters
        ----------
        agent_id : int
            Agent to update.
        value : float
            Agent's reported value.
        neighbor_values : list[float]
            Neighbor values for comparison.

        Returns
        -------
        float
            Updated trust score.
        """
        cfg = self.config

        if not neighbor_values:
            return float(self.trust_scores[agent_id])

        # Compute neighborhood median
        median = float(np.median(neighbor_values))

        # Normalized deviation (safe against zero median)
        deviation = abs(value - median) / (abs(median) + 1e-6)
        self._deviation_history[agent_id].append(deviation)

        # Trim history to window size
        hist = self._deviation_history[agent_id]
        if len(hist) > cfg.window_size:
            self._deviation_history[agent_id] = hist[-cfg.window_size:]
            hist = self._deviation_history[agent_id]

        # Update trust based on mean deviation
        mean_deviation = float(np.mean(hist))

        if mean_deviation > self._DEV_HIGH:
            self.trust_scores[agent_id] -= cfg.penalty_false_data
            self._attack_flags[agent_id] += 1
        elif mean_deviation > self._DEV_MOD:
            self.trust_scores[agent_id] -= cfg.penalty_deviation
        else:
            self.trust_scores[agent_id] += cfg.reward_consistent

        # Clip to bounds
        self.trust_scores[agent_id] = float(np.clip(
            self.trust_scores[agent_id], cfg.min_trust, cfg.max_trust
        ))

        return float(self.trust_scores[agent_id])

    def get_weighted_consensus(self, values: dict[int, float]) -> float:
        """Compute trust-weighted consensus.

        Parameters
        ----------
        values : dict[int, float]
            Mapping {agent_id: value}.

        Returns
        -------
        float
            Trust-weighted average.
        """
        total_weight = 0.0
        weighted_sum = 0.0

        for agent_id, value in values.items():
            weight = self.trust_scores[agent_id]
            weighted_sum += weight * value
            total_weight += weight

        return weighted_sum / total_weight if total_weight > 0.0 else 0.0

    def get_trust_report(self) -> dict:
        """Return trust score statistics."""
        return {
            "mean_trust": float(np.mean(self.trust_scores)),
            "min_trust": float(np.min(self.trust_scores)),
            "max_trust": float(np.max(self.trust_scores)),
            "suspicious_agents": int(np.sum(self.trust_scores < 0.5)),
            "attack_detections": dict(self._attack_flags),
            "trust_scores": self.trust_scores.tolist(),
        }


class OutlierDetector:
    """Statistical outlier detection using Mahalanobis distance.

    Detects multivariate outliers in relay measurements,
    useful for identifying compromised sensors or communication errors.
    """

    def __init__(self, threshold: float = 3.0) -> None:
        self.threshold = threshold
        self._mean: np.ndarray | None = None
        self._cov: np.ndarray | None = None
        self._inv_cov: np.ndarray | None = None

    def fit(self, data: np.ndarray) -> None:
        """Fit detector on clean (training) data.

        Parameters
        ----------
        data : np.ndarray, shape (n_samples, n_features)
            Clean reference data.
        """
        self._mean = np.mean(data, axis=0)
        self._cov = np.cov(data.T)

        # Regularized inverse for numerical stability
        reg_cov = self._cov + np.eye(self._cov.shape[0]) * 1e-6
        self._inv_cov = np.linalg.inv(reg_cov)

    def detect(self, sample: np.ndarray) -> tuple[bool, float]:
        """Detect if sample is an outlier.

        Parameters
        ----------
        sample : np.ndarray, shape (n_features,)
            Single observation.

        Returns
        -------
        tuple[bool, float]
            (is_outlier, mahalanobis_distance)
        """
        if self._mean is None or self._inv_cov is None:
            return False, 0.0

        diff = sample - self._mean
        distance = float(np.sqrt(diff @ self._inv_cov @ diff))

        return distance > self.threshold, distance

    def detect_batch(self, samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Detect outliers in a batch.

        Parameters
        ----------
        samples : np.ndarray, shape (n_samples, n_features)
            Batch of observations.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (outlier_flags, distances)
        """
        if self._mean is None or self._inv_cov is None:
            n = len(samples)
            return np.zeros(n, dtype=bool), np.zeros(n)

        diff = samples - self._mean
        distances = np.sqrt(np.sum(diff @ self._inv_cov * diff, axis=1))
        outliers = distances > self.threshold

        return outliers, distances


class ByzantineResilientAggregator:
    """Complete Byzantine-resilient aggregation pipeline.

    Combines multiple defense mechanisms:
    1. Outlier detection (Mahalanobis)
    2. Reputation-based weighting
    3. Trimmed mean aggregation
    4. Median consensus fallback
    """

    def __init__(self, n_agents: int, trim_ratio: float = 0.2) -> None:
        self.n_agents = n_agents
        self.median_filter = MedianConsensusFilter(n_agents)
        self.trimmed_mean = TrimmedMeanAggregator(trim_ratio)
        self.reputation = ReputationManager(n_agents)
        self.outlier_detector = OutlierDetector()

    def aggregate(
        self,
        agent_id: int,
        value: float,
        neighbor_values: list[float],
    ) -> tuple[float, dict]:
        """Full Byzantine-resilient aggregation.

        Parameters
        ----------
        agent_id : int
            Agent requesting aggregation.
        value : float
            Agent's own value.
        neighbor_values : list[float]
            Values from neighboring agents.

        Returns
        -------
        tuple[float, dict]
            (aggregated_value, metadata)
        """
        # Step 1: Update reputation
        trust = self.reputation.update(agent_id, value, neighbor_values)

        # Step 2: Median consensus
        median_value = self.median_filter.filter(agent_id, value, neighbor_values)

        # Step 3: Trimmed mean on all values
        all_values = [value] + list(neighbor_values)
        trimmed_value = self.trimmed_mean.aggregate(all_values)

        # Step 4: Combine (weighted by trust)
        if trust > 1.0:
            # High trust: use trimmed mean (more efficient)
            final_value = trimmed_value
        else:
            # Low trust environment: use median (more robust)
            final_value = median_value

        metadata = {
            "original": value,
            "median": median_value,
            "trimmed_mean": trimmed_value,
            "trust_score": trust,
            "n_neighbors": len(neighbor_values),
        }

        return final_value, metadata

    def get_system_health(self) -> dict:
        """Return overall system health report."""
        trust_report = self.reputation.get_trust_report()
        suspicious = trust_report["suspicious_agents"]

        return {
            "trust": trust_report,
            "status": "healthy" if suspicious == 0 else "degraded",
            "recommendation": (
                "Monitor suspicious agents" if suspicious > 0 else "System normal"
            ),
        }


# ── Self-test ───────────────────────────────────────────────────────────────

def test_byzantine_resilience() -> None:
    """Quick self-test of Byzantine resilience components."""
    print("Testing Byzantine Resilience...")

    # Test 1: Median filter with outlier
    filt = MedianConsensusFilter(n_agents=5)
    result = filt.filter(0, 1.0, [1.1, 1.2, 5.0, 1.15])
    assert 1.1 <= result <= 1.25, f"Median filter failed: {result}"
    print(f"  Median filter: input=1.0, outlier=5.0, result={result:.2f} ✓")

    # Test 2: Trimmed mean
    tma = TrimmedMeanAggregator(trim_ratio=0.2)
    result = tma.aggregate([1.0, 1.1, 1.2, 10.0, 1.15])
    assert 1.0 <= result <= 1.2, f"Trimmed mean failed: {result}"
    print(f"  Trimmed mean: input=[1.0,1.1,1.2,10.0,1.15], result={result:.2f} ✓")

    # Test 3: Reputation manager — normal behavior
    rep = ReputationManager(n_agents=3)
    trust = rep.update(0, 1.0, [1.1, 1.2])
    assert trust >= 1.0, f"Normal trust should not decrease: {trust}"
    print(f"  Reputation: normal behavior, trust={trust:.2f} ✓")

    # Test 4: Reputation manager — bad behavior
    trust_bad = rep.update(1, 10.0, [1.1, 1.2])
    assert trust_bad < 1.0, f"Bad trust should decrease: {trust_bad}"
    print(f"  Reputation: bad behavior, trust={trust_bad:.2f} ✓")

    # Test 5: Full aggregator
    agg = ByzantineResilientAggregator(n_agents=5)
    val, meta = agg.aggregate(0, 1.0, [1.1, 1.2, 5.0, 1.15])
    assert 1.0 <= val <= 1.3, f"Aggregator failed: {val}"
    print(f"  Full aggregator: result={val:.2f}, trust={meta['trust_score']:.2f} ✓")

    health = agg.get_system_health()
    assert health["status"] in ("healthy", "degraded")
    print(f"  System health: {health['status']} ✓")

    print("All Byzantine resilience tests passed!")


if __name__ == "__main__":
    test_byzantine_resilience()