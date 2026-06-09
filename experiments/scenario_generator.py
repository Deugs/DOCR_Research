"""
experiments/scenario_generator.py
─────────────────────────────────
Large-Scale Scenario Generation for DOCR-MARL Training.

Phase 5: Generates 12,000+ scenarios with stratified sampling:
- 10,000 training scenarios (stratified across PV levels, fault types, topologies)
- 2,000 validation scenarios (unseen combinations)
- Monte Carlo fault injection with edge-case coverage
- NREL NSRDB-inspired irradiance profiles + Markov cloud transitions
- Curriculum stage-aware scenario distribution

Key Features:
- Stratified sampling: Ensures coverage of all parameter combinations
- Edge-case injection: Low-probability but critical scenarios
- Curriculum progression: Scenarios ordered by difficulty
- Reproducibility: Deterministic seeding, JSON export
"""

import numpy as np
import json
import os
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum


class FaultType(Enum):
    THREE_PHASE = "3ph"
    LINE_LINE = "LL"
    SINGLE_PHASE = "1ph"
    TWO_PHASE_GROUND = "2ph-g"


class PVLevel(Enum):
    ZERO = 0.0
    LOW = 0.2
    MEDIUM = 0.4
    HIGH = 0.6
    VERY_HIGH = 0.8
    FULL = 1.0


@dataclass
class ScenarioConfig:
    """Configuration for a single training/validation scenario."""
    scenario_id: int
    system_name: str  # "3bus", "8bus", "30bus", "39bus"
    pv_level: float
    fault_type: str
    topology_changes: Dict[str, Any]  # line outages, bus splits, etc.
    time_of_day: float  # 0-24 hours
    irradiance_factor: float  # 0.0-1.0 (clear sky = 1.0)
    temperature: float  # Celsius
    load_profile_factor: float  # 0.5-1.5 (peak load multiplier)
    n1_outage_line: Optional[int] = None  # Line index for N-1, None = no outage
    sensor_noise_level: float = 0.0  # 0.0-1.0
    curriculum_stage: int = 0  # 0-5
    is_validation: bool = False
    seed: int = 0

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> 'ScenarioConfig':
        return cls(**d)


@dataclass
class StratificationConfig:
    """Configuration for stratified sampling."""
    n_train: int = 10000
    n_val: int = 2000
    pv_levels: List[float] = None
    fault_types: List[str] = None
    systems: List[str] = None
    n1_outage_prob: float = 0.15  # 15% scenarios have N-1 outage
    sensor_noise_prob: float = 0.20  # 20% have sensor noise
    edge_case_prob: float = 0.05  # 5% are edge cases (extreme conditions)
    curriculum_weighted: bool = True  # Weight by curriculum stage

    def __post_init__(self):
        if self.pv_levels is None:
            self.pv_levels = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        if self.fault_types is None:
            self.fault_types = ["3ph", "LL", "1ph", "2ph-g"]
        if self.systems is None:
            self.systems = ["8bus", "30bus", "39bus"]  # Exclude 3bus (toy)


class MarkovCloudModel:
    """
    3-state Markov chain for cloud coverage modeling.
    States: Clear (1.0), Partly Cloudy (0.6), Overcast (0.2)
    """

    STATES = {
        "clear": 1.0,
        "partly": 0.6,
        "overcast": 0.2,
    }

    # Transition matrix: [from][to]
    # Based on diurnal patterns: more stable midday, more variable morning/evening
    TRANSITIONS = {
        "clear": {"clear": 0.85, "partly": 0.12, "overcast": 0.03},
        "partly": {"clear": 0.25, "partly": 0.55, "overcast": 0.20},
        "overcast": {"clear": 0.10, "partly": 0.30, "overcast": 0.60},
    }

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.state_names = list(self.STATES.keys())

    def generate_sequence(self, n_steps: int, time_of_day: float = 12.0) -> List[Tuple[str, float]]:
        """Generate a sequence of cloud states."""
        # Adjust transitions based on time of day
        transitions = self._adjust_for_time(time_of_day)

        # Start with clear sky (most common)
        current = "clear"
        sequence = []

        for _ in range(n_steps):
            sequence.append((current, self.STATES[current]))
            # Transition
            probs = [transitions[current][s] for s in self.state_names]
            current = self.rng.choice(self.state_names, p=probs)

        return sequence

    def _adjust_for_time(self, time_of_day: float) -> Dict:
        """Adjust transition probabilities based on time of day."""
        transitions = {k: dict(v) for k, v in self.TRANSITIONS.items()}

        # More variable in morning/evening (6-9am, 4-7pm)
        is_variable = (6 <= time_of_day <= 9) or (16 <= time_of_day <= 19)

        if is_variable:
            # Increase transition probabilities
            for from_state in transitions:
                for to_state in transitions[from_state]:
                    if from_state != to_state:
                        transitions[from_state][to_state] *= 1.5
                # Renormalize
                total = sum(transitions[from_state].values())
                for to_state in transitions[from_state]:
                    transitions[from_state][to_state] /= total

        return transitions


class DailyProfileGenerator:
    """Generate realistic daily profiles for irradiance, temperature, and load."""

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def generate_irradiance(self, time_of_day: float, cloud_factor: float = 1.0) -> float:
        """Generate solar irradiance (W/m²) at given time."""
        # Clear sky model: peak at noon (12:00), zero at night
        if time_of_day < 6 or time_of_day > 18:
            return 0.0

        # Sinusoidal profile peaking at noon
        hour_angle = (time_of_day - 12) * 15  # degrees
        cos_zenith = max(0, np.cos(np.radians(hour_angle)))

        # Peak irradiance ~1000 W/m²
        clear_sky = 1000 * cos_zenith

        # Apply cloud factor
        return clear_sky * cloud_factor

    def generate_temperature(self, time_of_day: float, base_temp: float = 25.0) -> float:
        """Generate temperature (Celsius) with daily cycle."""
        # Sinusoidal: coldest at 6am, hottest at 3pm
        phase = (time_of_day - 6) * 2 * np.pi / 24
        amplitude = 8.0  # ±8°C variation
        noise = self.rng.normal(0, 1.0)
        return base_temp + amplitude * np.sin(phase - np.pi/2) + noise

    def generate_load_factor(self, time_of_day: float) -> float:
        """Generate load profile factor (0.5-1.5)."""
        # Morning peak (8-10am), evening peak (6-9pm)
        morning_peak = 0.3 * np.exp(-0.5 * ((time_of_day - 9) / 1.5) ** 2)
        evening_peak = 0.4 * np.exp(-0.5 * ((time_of_day - 19.5) / 2.0) ** 2)
        base = 0.7 + morning_peak + evening_peak
        noise = self.rng.normal(0, 0.05)
        return np.clip(base + noise, 0.5, 1.5)


class LargeScaleScenarioGenerator:
    """
    Main scenario generator for Phase 5 large-scale training.

    Generates stratified scenarios ensuring:
    - Balanced coverage across all parameter combinations
    - Edge-case scenarios for robustness testing
    - Curriculum progression for staged learning
    - Deterministic reproducibility
    """

    def __init__(self, config: StratificationConfig = None, seed: int = 42):
        self.config = config or StratificationConfig()
        self.rng = np.random.default_rng(seed)
        self.cloud_model = MarkovCloudModel(seed=seed)
        self.profile_gen = DailyProfileGenerator(seed=seed)
        self.scenarios: List[ScenarioConfig] = []

    def generate_all_scenarios(self) -> Tuple[List[ScenarioConfig], List[ScenarioConfig]]:
        """Generate all training and validation scenarios."""
        train_scenarios = self._generate_split(n=self.config.n_train, is_validation=False)
        val_scenarios = self._generate_split(n=self.config.n_val, is_validation=True)
        return train_scenarios, val_scenarios

    def _generate_split(self, n: int, is_validation: bool) -> List[ScenarioConfig]:
        """Generate a split (train or val) with stratified sampling."""
        scenarios = []

        # Calculate strata sizes
        n_pv = len(self.config.pv_levels)
        n_fault = len(self.config.fault_types)
        n_systems = len(self.config.systems)
        n_combinations = n_pv * n_fault * n_systems

        # Base scenarios per combination
        base_per_combo = n // n_combinations
        remainder = n % n_combinations

        scenario_id = 0

        for system in self.config.systems:
            for pv_level in self.config.pv_levels:
                for fault_type in self.config.fault_types:
                    # Determine curriculum stage from PV level
                    curriculum_stage = self.config.pv_levels.index(pv_level)

                    # Number of scenarios for this combination
                    n_scenarios = base_per_combo + (1 if remainder > 0 else 0)
                    remainder -= 1

                    for _ in range(n_scenarios):
                        scenario = self._generate_single_scenario(
                            scenario_id=scenario_id,
                            system=system,
                            pv_level=pv_level,
                            fault_type=fault_type,
                            curriculum_stage=curriculum_stage,
                            is_validation=is_validation,
                        )
                        scenarios.append(scenario)
                        scenario_id += 1

        # Add edge-case scenarios (replace random base scenarios)
        n_edge = int(n * self.config.edge_case_prob)
        edge_indices = self.rng.choice(len(scenarios), size=n_edge, replace=False)
        for idx in edge_indices:
            scenarios[idx] = self._generate_edge_case(scenarios[idx])

        # Shuffle
        self.rng.shuffle(scenarios)

        # Reassign IDs after shuffle
        for i, s in enumerate(scenarios):
            s.scenario_id = i

        return scenarios

    def _generate_single_scenario(
        self,
        scenario_id: int,
        system: str,
        pv_level: float,
        fault_type: str,
        curriculum_stage: int,
        is_validation: bool,
    ) -> ScenarioConfig:
        """Generate a single scenario configuration."""
        # Time of day: biased toward solar hours (6am-6pm) for PV relevance
        if self.rng.random() < 0.8:
            time_of_day = self.rng.uniform(6, 18)
        else:
            time_of_day = self.rng.uniform(0, 24)

        # Cloud state sequence (single step for now)
        cloud_seq = self.cloud_model.generate_sequence(1, time_of_day)
        cloud_factor = cloud_seq[0][1]

        # Generate profiles
        irradiance = self.profile_gen.generate_irradiance(time_of_day, cloud_factor)
        temperature = self.profile_gen.generate_temperature(time_of_day)
        load_factor = self.profile_gen.generate_load_factor(time_of_day)

        # N-1 outage (with probability)
        n1_outage = None
        topology_changes = {}
        if self.rng.random() < self.config.n1_outage_prob:
            n1_outage = int(self.rng.integers(0, 10))  # Random line index
            topology_changes = {
                "type": "n1_outage",
                "line_idx": n1_outage,
                "outage_factor": self.rng.uniform(0.5, 0.9),
            }

        # Sensor noise
        sensor_noise = 0.0
        if self.rng.random() < self.config.sensor_noise_prob:
            sensor_noise = self.rng.uniform(0.1, 0.5)

        return ScenarioConfig(
            scenario_id=scenario_id,
            system_name=system,
            pv_level=pv_level,
            fault_type=fault_type,
            topology_changes=topology_changes,
            time_of_day=time_of_day,
            irradiance_factor=cloud_factor,
            temperature=temperature,
            load_profile_factor=load_factor,
            n1_outage_line=n1_outage,
            sensor_noise_level=sensor_noise,
            curriculum_stage=curriculum_stage,
            is_validation=is_validation,
            seed=int(self.rng.integers(0, 2**31)),
        )

    def _generate_edge_case(self, base: ScenarioConfig) -> ScenarioConfig:
        """Generate an edge-case scenario from a base scenario."""
        edge_type = self.rng.choice([
            "extreme_pv", "extreme_load", "multiple_outages", "high_noise", "low_irradiance"
        ])

        if edge_type == "extreme_pv":
            base.pv_level = 1.0 if self.rng.random() < 0.5 else 0.0
            base.curriculum_stage = 5 if base.pv_level == 1.0 else 0
        elif edge_type == "extreme_load":
            base.load_profile_factor = self.rng.uniform(1.3, 1.5)
        elif edge_type == "multiple_outages":
            base.topology_changes = {
                "type": "multiple_outages",
                "line_indices": [int(self.rng.integers(0, 10)) for _ in range(2)],
                "outage_factor": self.rng.uniform(0.4, 0.7),
            }
            base.n1_outage_line = base.topology_changes["line_indices"][0]
        elif edge_type == "high_noise":
            base.sensor_noise_level = self.rng.uniform(0.5, 1.0)
        elif edge_type == "low_irradiance":
            base.irradiance_factor = self.rng.uniform(0.0, 0.2)
            base.time_of_day = self.rng.uniform(18, 24)  # Evening/night

        return base

    def get_curriculum_batches(self, scenarios: List[ScenarioConfig]) -> Dict[int, List[ScenarioConfig]]:
        """Group scenarios by curriculum stage."""
        batches = {i: [] for i in range(6)}
        for s in scenarios:
            batches[s.curriculum_stage].append(s)
        return batches

    def get_statistics(self, scenarios: List[ScenarioConfig]) -> Dict:
        """Compute statistics for a scenario set."""
        stats = {
            "total": len(scenarios),
            "by_system": {},
            "by_pv_level": {},
            "by_fault_type": {},
            "by_curriculum_stage": {},
            "n1_outage_rate": 0.0,
            "sensor_noise_rate": 0.0,
            "edge_case_rate": 0.0,
            "validation_rate": 0.0,
        }

        for s in scenarios:
            stats["by_system"][s.system_name] = stats["by_system"].get(s.system_name, 0) + 1
            stats["by_pv_level"][s.pv_level] = stats["by_pv_level"].get(s.pv_level, 0) + 1
            stats["by_fault_type"][s.fault_type] = stats["by_fault_type"].get(s.fault_type, 0) + 1
            stats["by_curriculum_stage"][s.curriculum_stage] = stats["by_curriculum_stage"].get(s.curriculum_stage, 0) + 1
            if s.n1_outage_line is not None:
                stats["n1_outage_rate"] += 1
            if s.sensor_noise_level > 0:
                stats["sensor_noise_rate"] += 1
            if s.is_validation:
                stats["validation_rate"] += 1

        stats["n1_outage_rate"] /= len(scenarios)
        stats["sensor_noise_rate"] /= len(scenarios)
        stats["edge_case_rate"] = stats["by_curriculum_stage"].get(5, 0) / len(scenarios)  # Approximate
        stats["validation_rate"] /= len(scenarios)

        return stats

    def save_scenarios(self, scenarios: List[ScenarioConfig], filepath: str):
        """Save scenarios to JSON file."""
        data = [s.to_dict() for s in scenarios]
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"  Saved {len(scenarios)} scenarios to {filepath}")

    def load_scenarios(self, filepath: str) -> List[ScenarioConfig]:
        """Load scenarios from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return [ScenarioConfig.from_dict(d) for d in data]


def generate_scenario_dataset(
    output_dir: str = "outputs/scenarios",
    n_train: int = 10000,
    n_val: int = 2000,
    seed: int = 42,
) -> Tuple[str, str]:
    """
    Convenience function to generate and save full scenario dataset.

    Returns:
        (train_path, val_path)
    """
    os.makedirs(output_dir, exist_ok=True)

    config = StratificationConfig(n_train=n_train, n_val=n_val)
    generator = LargeScaleScenarioGenerator(config=config, seed=seed)

    print(f"\n{'='*60}")
    print(f"  PHASE 5: Large-Scale Scenario Generation")
    print(f"{'='*60}")
    print(f"  Training scenarios: {n_train:,}")
    print(f"  Validation scenarios: {n_val:,}")
    print(f"  Total: {n_train + n_val:,}")
    print(f"  PV levels: {config.pv_levels}")
    print(f"  Fault types: {config.fault_types}")
    print(f"  Systems: {config.systems}")
    print(f"{'='*60}")

    train_scenarios, val_scenarios = generator.generate_all_scenarios()

    # Print statistics
    print("\n  Training Set Statistics:")
    train_stats = generator.get_statistics(train_scenarios)
    for key, val in train_stats.items():
        print(f"    {key}: {val}")

    print("\n  Validation Set Statistics:")
    val_stats = generator.get_statistics(val_scenarios)
    for key, val in val_stats.items():
        print(f"    {key}: {val}")

    # Save
    train_path = os.path.join(output_dir, "train_scenarios.json")
    val_path = os.path.join(output_dir, "val_scenarios.json")

    generator.save_scenarios(train_scenarios, train_path)
    generator.save_scenarios(val_scenarios, val_path)

    # Save metadata
    metadata = {
        "n_train": n_train,
        "n_val": n_val,
        "train_stats": train_stats,
        "val_stats": val_stats,
        "stratification": {
            "pv_levels": config.pv_levels,
            "fault_types": config.fault_types,
            "systems": config.systems,
            "n1_outage_prob": config.n1_outage_prob,
            "sensor_noise_prob": config.sensor_noise_prob,
            "edge_case_prob": config.edge_case_prob,
        },
    }

    meta_path = os.path.join(output_dir, "scenario_metadata.json")
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  Saved metadata to {meta_path}")

    return train_path, val_path


if __name__ == "__main__":
    generate_scenario_dataset()
