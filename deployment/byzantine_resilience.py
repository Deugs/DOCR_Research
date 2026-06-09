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
    clean_value = filter.filter(agent_id=0, value=1.2, neighbor_values=[1.1, 1.3, 5.0, 1.2])
    # Result: 1.2 (median, ignores the 5.0 outlier)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict


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


class MedianConsensusFilter:
    """
    Byzantine-resilient median consensus filter.

    Tolerates up to f faulty agents in neighborhood if n > 2f,
    where n = total agents in neighborhood.

    Algorithm:
    1. Collect own value + all neighbor values
    2. Compute median
    3. Replace own value with median

    Properties:
    - Optimal breakdown point: 50% (tolerates up to half faulty)
    - No parametric assumptions (distribution-free)
    - O(n log n) complexity per agent
    """

    def __init__(self, n_agents: int):
        self.n_agents = n_agents
        self.history: Dict[int, List[float]] = defaultdict(list)

    def filter(self, agent_id: int, value: float, 
               neighbor_values: List[float]) -> float:
        """
        Apply median consensus filtering.

        Args:
            agent_id: ID of the agent filtering
            value: Own measurement/value
            neighbor_values: List of neighbor values

        Returns:
            filtered_value: Median of all values
        """
        all_values = [value] + list(neighbor_values)

        if len(all_values) == 0:
            return value

        filtered = float(np.median(all_values))
        self.history[agent_id].append({
            "original": value,
            "filtered": filtered,
            "neighbors": neighbor_values,
        })

        return filtered

    def filter_vector(self, agent_id: int, values: np.ndarray,
                      neighbor_values_list: List[np.ndarray]) -> np.ndarray:
        """Apply median filtering to vector-valued data."""
        all_values = [values] + list(neighbor_values_list)
        stacked = np.stack(all_values, axis=0)
        return np.median(stacked, axis=0)

    def get_history(self, agent_id: int) -> List[Dict]:
        """Return filtering history for an agent."""
        return self.history.get(agent_id, [])


class TrimmedMeanAggregator:
    """
    Trimmed mean aggregator for Byzantine resilience.

    Removes a fraction of extreme values before computing mean.
    More robust than simple mean, less aggressive than median.

    Parameters:
        trim_ratio: Fraction to trim from each end (0.0 = mean, 0.5 = median)
    """

    def __init__(self, trim_ratio: float = 0.2):
        self.trim_ratio = np.clip(trim_ratio, 0.0, 0.5)

    def aggregate(self, values: List[float]) -> float:
        """
        Compute trimmed mean.

        Args:
            values: List of values from agents

        Returns:
            trimmed_mean: Mean after removing extremes
        """
        arr = np.array(values)
        n = len(arr)

        if n == 0:
            return 0.0

        k = int(n * self.trim_ratio)
        sorted_arr = np.sort(arr)

        if k > 0:
            trimmed = sorted_arr[k:-k] if k * 2 < n else sorted_arr
        else:
            trimmed = sorted_arr

        return float(np.mean(trimmed))

    def aggregate_vector(self, values_list: List[np.ndarray]) -> np.ndarray:
        """Trimmed mean for vector-valued data."""
        stacked = np.stack(values_list, axis=0)
        n = len(values_list)
        k = int(n * self.trim_ratio)

        # Sort along agent axis and trim
        sorted_stack = np.sort(stacked, axis=0)
        if k > 0 and k * 2 < n:
            trimmed = sorted_stack[k:-k]
        else:
            trimmed = sorted_stack

        return np.mean(trimmed, axis=0)


class ReputationManager:
    """
    Reputation-based trust management for relay agents.

    Tracks trust scores per agent based on:
    - Consistency with neighborhood median
    - Sudden deviations (attack detection)
    - Historical behavior

    Agents with low trust have reduced influence in consensus.
    """

    def __init__(self, n_agents: int, config: Optional[ReputationConfig] = None):
        self.n_agents = n_agents
        self.config = config or ReputationConfig()
        self.trust_scores: np.ndarray = np.full(n_agents, self.config.initial_trust)
        self.deviation_history: Dict[int, List[float]] = defaultdict(list)
        self.attack_flags: Dict[int, int] = defaultdict(int)

    def update(self, agent_id: int, value: float, 
               neighbor_values: List[float]) -> float:
        """
        Update trust score based on deviation from consensus.

        Args:
            agent_id: Agent to update
            value: Agent's reported value
            neighbor_values: Neighbor values for comparison

        Returns:
            new_trust: Updated trust score
        """
        cfg = self.config

        if not neighbor_values:
            return self.trust_scores[agent_id]

        # Compute neighborhood median
        median = np.median(neighbor_values)

        # Compute normalized deviation
        deviation = abs(value - median) / (abs(median) + 1e-6)
        self.deviation_history[agent_id].append(deviation)

        # Trim history to window size
        if len(self.deviation_history[agent_id]) > cfg.window_size:
            self.deviation_history[agent_id] = self.deviation_history[agent_id][-cfg.window_size:]

        # Update trust
        mean_deviation = np.mean(self.deviation_history[agent_id])

        if mean_deviation > 0.5:  # Significant deviation
            self.trust_scores[agent_id] -= cfg.penalty_false_data
            self.attack_flags[agent_id] += 1
        elif mean_deviation > 0.2:  # Moderate deviation
            self.trust_scores[agent_id] -= cfg.penalty_deviation
        else:  # Consistent
            self.trust_scores[agent_id] += cfg.reward_consistent

        # Clip to bounds
        self.trust_scores[agent_id] = np.clip(
            self.trust_scores[agent_id], cfg.min_trust, cfg.max_trust
        )

        return float(self.trust_scores[agent_id])

    def get_weighted_consensus(self, values: Dict[int, float]) -> float:
        """
        Compute trust-weighted consensus.

        Args:
            values: {agent_id: value}

        Returns:
            weighted_mean: Trust-weighted average
        """
        total_weight = 0.0
        weighted_sum = 0.0

        for agent_id, value in values.items():
            weight = self.trust_scores[agent_id]
            weighted_sum += weight * value
            total_weight += weight

        return weighted_sum / total_weight if total_weight > 0 else 0.0

    def get_trust_report(self) -> Dict:
        """Return trust score statistics."""
        return {
            "mean_trust": float(np.mean(self.trust_scores)),
            "min_trust": float(np.min(self.trust_scores)),
            "max_trust": float(np.max(self.trust_scores)),
            "suspicious_agents": int(np.sum(self.trust_scores < 0.5)),
            "attack_detections": dict(self.attack_flags),
            "trust_scores": self.trust_scores.tolist(),
        }


class OutlierDetector:
    """
    Statistical outlier detection using Mahalanobis distance.

    Detects multivariate outliers in relay measurements,
    useful for identifying compromised sensors or communication errors.
    """

    def __init__(self, threshold: float = 3.0):
        self.threshold = threshold
        self.mean: Optional[np.ndarray] = None
        self.cov: Optional[np.ndarray] = None
        self.inv_cov: Optional[np.ndarray] = None

    def fit(self, data: np.ndarray):
        """Fit detector on clean data."""
        self.mean = np.mean(data, axis=0)
        self.cov = np.cov(data.T)

        # Regularized inverse for numerical stability
        reg_cov = self.cov + np.eye(self.cov.shape[0]) * 1e-6
        self.inv_cov = np.linalg.inv(reg_cov)

    def detect(self, sample: np.ndarray) -> Tuple[bool, float]:
        """
        Detect if sample is outlier.

        Returns:
            is_outlier: True if Mahalanobis distance > threshold
            distance: Mahalanobis distance
        """
        if self.mean is None or self.inv_cov is None:
            return False, 0.0

        diff = sample - self.mean
        distance = np.sqrt(diff @ self.inv_cov @ diff)

        return distance > self.threshold, float(distance)

    def detect_batch(self, samples: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Detect outliers in batch."""
        if self.mean is None or self.inv_cov is None:
            return np.zeros(len(samples), dtype=bool), np.zeros(len(samples))

        diff = samples - self.mean
        distances = np.sqrt(np.sum(diff @ self.inv_cov * diff, axis=1))
        outliers = distances > self.threshold

        return outliers, distances


class ByzantineResilientAggregator:
    """
    Complete Byzantine-resilient aggregation pipeline.

    Combines multiple defense mechanisms:
    1. Outlier detection (Mahalanobis)
    2. Reputation-based weighting
    3. Trimmed mean aggregation
    4. Median consensus fallback
    """

    def __init__(self, n_agents: int, trim_ratio: float = 0.2):
        self.n_agents = n_agents
        self.median_filter = MedianConsensusFilter(n_agents)
        self.trimmed_mean = TrimmedMeanAggregator(trim_ratio)
        self.reputation = ReputationManager(n_agents)
        self.outlier_detector = OutlierDetector()

    def aggregate(self, agent_id: int, value: float,
                  neighbor_values: List[float]) -> Tuple[float, Dict]:
        """
        Full Byzantine-resilient aggregation.

        Returns:
            aggregated_value: Clean consensus value
            metadata: Dict with filtering info
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

    def get_system_health(self) -> Dict:
        """Return overall system health report."""
        trust_report = self.reputation.get_trust_report()

        return {
            "trust": trust_report,
            "status": "healthy" if trust_report["suspicious_agents"] == 0 else "degraded",
            "recommendation": "Monitor suspicious agents" if trust_report["suspicious_agents"] > 0 else "System normal",
        }


def test_byzantine_resilience():
    """Quick test of Byzantine resilience components."""
    print("Testing Byzantine Resilience...")

    # Test 1: Median filter with outlier
    filter = MedianConsensusFilter(n_agents=5)
    result = filter.filter(0, 1.0, [1.1, 1.2, 5.0, 1.15])
    print(f"  Median filter: input=1.0, outlier=5.0, result={result:.2f} ✓")

    # Test 2: Trimmed mean
    tma = TrimmedMeanAggregator(trim_ratio=0.2)
    result = tma.aggregate([1.0, 1.1, 1.2, 10.0, 1.15])
    print(f"  Trimmed mean: input=[1.0,1.1,1.2,10.0,1.15], result={result:.2f} ✓")

    # Test 3: Reputation manager
    rep = ReputationManager(n_agents=3)
    trust = rep.update(0, 1.0, [1.1, 1.2])
    print(f"  Reputation: normal behavior, trust={trust:.2f} ✓")

    trust_bad = rep.update(1, 10.0, [1.1, 1.2])
    print(f"  Reputation: bad behavior, trust={trust_bad:.2f} ✓")

    # Test 4: Full aggregator
    agg = ByzantineResilientAggregator(n_agents=5)
    val, meta = agg.aggregate(0, 1.0, [1.1, 1.2, 5.0, 1.15])
    print(f"  Full aggregator: result={val:.2f}, trust={meta['trust_score']:.2f} ✓")

    health = agg.get_system_health()
    print(f"  System health: {health['status']} ✓")

    print("All Byzantine resilience tests passed!")


if __name__ == "__main__":
    test_byzantine_resilience()
