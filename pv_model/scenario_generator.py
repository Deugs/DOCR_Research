"""
scenario_generator.py
─────────────────────
Scenario generator for PV/load profiles using:
- NREL NSRDB irradiance data patterns (simplified)
- Markov chain cloud cover models
- Daily load profiles (IEEE RTS / residential)
- Fault injection scenarios

Phase 1: Data generation for training and testing.
"""

import numpy as np
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass


@dataclass
class DailyProfile:
    """24-hour profile for irradiance, temperature, and load."""
    hours: np.ndarray
    irradiance: np.ndarray   # W/m²
    temperature: np.ndarray  # °C
    load_pu: np.ndarray      # Per-unit load


class MarkovCloudModel:
    """
    Markov chain model for cloud cover dynamics.
    States: Clear (0), Partly Cloudy (1), Overcast (2)

    Transition matrix based on typical diurnal patterns.
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        # Transition probabilities [from][to]
        # Higher probability of staying in same state, clear→overcast rare
        self.P = np.array([
            [0.85, 0.12, 0.03],   # From Clear
            [0.20, 0.70, 0.10],   # From Partly Cloudy
            [0.10, 0.25, 0.65],   # From Overcast
        ])
        self.states = ["clear", "partly_cloudy", "overcast"]
        self.irradiance_factor = {"clear": 1.0, "partly_cloudy": 0.6, "overcast": 0.2}

    def generate_sequence(self, n_steps: int, initial_state: int = 0) -> List[int]:
        """Generate n_steps of cloud cover states."""
        sequence = [initial_state]
        for _ in range(n_steps - 1):
            current = sequence[-1]
            next_state = self.rng.choice(3, p=self.P[current])
            sequence.append(next_state)
        return sequence

    def generate_daily_profile(self, resolution_min: int = 15) -> DailyProfile:
        """Generate 24-hour profile with 15-minute resolution."""
        n_steps = 24 * 60 // resolution_min
        hours = np.arange(0, 24, resolution_min / 60)

        # Cloud states
        states = self.generate_sequence(n_steps)

        # Base solar irradiance (clear sky model)
        # Simplified: peak at noon, zero at night
        hour_angle = 2 * np.pi * (hours - 12) / 24
        cos_factor = np.maximum(0, np.cos(hour_angle))
        g_clear = 1000 * cos_factor  # W/m² peak

        # Apply cloud cover
        irradiance = np.array([
            g_clear[i] * self.irradiance_factor[self.states[s]]
            for i, s in enumerate(states)
        ])

        # Temperature (correlated with irradiance, lagged)
        temperature = 20 + 10 * cos_factor + self.rng.normal(0, 2, n_steps)

        # Load profile (IEEE RTS residential + commercial mix)
        # Morning peak, evening peak, night minimum
        load_base = (
            0.4 +  # Base load
            0.35 * np.exp(-((hours - 8) ** 2) / 8) +  # Morning peak
            0.25 * np.exp(-((hours - 19) ** 2) / 12)   # Evening peak
        )
        load_pu = np.clip(load_base + self.rng.normal(0, 0.05, n_steps), 0.2, 1.5)

        return DailyProfile(hours, irradiance, temperature, load_pu)


class ScenarioGenerator:
    """
    Generate training and validation scenarios for DOCR coordination.

    Produces:
    - PV penetration levels: 0%, 20%, 40%, 60%, 80%, 100%
    - Daily profiles with varying cloud cover
    - Fault scenarios (3ph primary, LL/1ph/2ph-g for generalization)
    - Topology changes (N-1 contingencies)
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.cloud_model = MarkovCloudModel(seed)
        self.pv_levels = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    def generate_scenarios(
        self,
        n_train: int = 10000,
        n_val: int = 2000,
        n_test: int = 1000,
    ) -> Dict[str, List[Dict]]:
        """
        Generate complete scenario dataset.

        Returns:
            Dict with 'train', 'val', 'test' lists of scenario dicts.
            Each scenario contains: pv_level, profile, fault_type, topology
        """
        scenarios = {"train": [], "val": [], "test": []}

        for split, n_scenarios in [("train", n_train), ("val", n_val), ("test", n_test)]:
            for _ in range(n_scenarios):
                scenario = self._generate_single_scenario(split)
                scenarios[split].append(scenario)

        return scenarios

    def _generate_single_scenario(self, split: str) -> Dict:
        """Generate one random scenario."""
        # PV penetration level (stratified sampling)
        if split == "train":
            pv_level = self.rng.choice(self.pv_levels)
        else:
            # Ensure all levels represented in val/test
            pv_level = self.rng.choice(self.pv_levels)

        # Daily profile
        profile = self.cloud_model.generate_daily_profile()

        # Fault type (3ph primary for training, mixed for test)
        if split == "train":
            fault_type = "3ph"  # Primary training on 3ph
        else:
            fault_type = self.rng.choice(["3ph", "LL", "1ph", "2ph-g"], p=[0.6, 0.2, 0.1, 0.1])

        # Topology changes (N-1 with 10% probability)
        topology = {"line_outages": [], "bus_splits": []}
        if self.rng.random() < 0.1:
            n_outages = self.rng.integers(1, 3)
            topology["line_outages"] = [
                int(self.rng.integers(0, 20)) for _ in range(n_outages)
            ]

        # Time of day for fault (affects irradiance and load)
        fault_time_idx = self.rng.integers(0, len(profile.hours))

        return {
            "pv_level": pv_level,
            "profile": profile,
            "fault_type": fault_type,
            "topology": topology,
            "fault_time_idx": fault_time_idx,
            "split": split,
        }

    def generate_curriculum_scenarios(
        self,
        stages: List[float] = None,
        episodes_per_stage: int = 2000,
    ) -> List[Dict]:
        """
        Generate curriculum learning scenarios.

        Stages: Static → PV@20% → PV@40% → PV@60% → PV@80% → PV@100% + topology
        """
        if stages is None:
            stages = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

        curriculum = []
        for stage_idx, pv_level in enumerate(stages):
            for _ in range(episodes_per_stage):
                scenario = self._generate_single_scenario("train")
                scenario["pv_level"] = pv_level
                scenario["curriculum_stage"] = stage_idx
                scenario["topology_complexity"] = min(stage_idx, 2)  # Increase topology changes

                # Add topology changes for later stages
                if stage_idx >= 4 and self.rng.random() < 0.3:
                    scenario["topology"]["line_outages"] = [
                        int(self.rng.integers(0, 20))
                    ]

                curriculum.append(scenario)

        return curriculum

    def get_static_scenario(self, pv_level: float = 0.0) -> Dict:
        """Generate a simple static scenario for baseline testing."""
        profile = self.cloud_model.generate_daily_profile()
        # Use noon (peak irradiance) for static case
        noon_idx = len(profile.hours) // 2

        return {
            "pv_level": pv_level,
            "profile": profile,
            "fault_type": "3ph",
            "topology": {"line_outages": [], "bus_splits": []},
            "fault_time_idx": noon_idx,
            "split": "test",
            "static": True,
        }


def generate_pv_profiles(output_dir: str = "data/pv_profiles", n_days: int = 365):
    """Generate and save PV profiles for a full year."""
    import os
    os.makedirs(output_dir, exist_ok=True)

    gen = ScenarioGenerator(seed=2024)

    for day in range(n_days):
        profile = gen.cloud_model.generate_daily_profile()
        np.savez(
            f"{output_dir}/day_{day:03d}.npz",
            hours=profile.hours,
            irradiance=profile.irradiance,
            temperature=profile.temperature,
            load_pu=profile.load_pu,
        )

    print(f"Generated {n_days} daily profiles in {output_dir}")


def generate_load_profiles(output_dir: str = "data/load_profiles", n_profiles: int = 100):
    """Generate and save load profiles."""
    import os
    os.makedirs(output_dir, exist_ok=True)

    gen = ScenarioGenerator(seed=42)

    for i in range(n_profiles):
        profile = gen.cloud_model.generate_daily_profile()
        np.savez(
            f"{output_dir}/load_profile_{i:03d}.npz",
            hours=profile.hours,
            load_pu=profile.load_pu,
        )

    print(f"Generated {n_profiles} load profiles in {output_dir}")
