"""
experiments/config_manager.py
─────────────────────────────
YAML Configuration Manager for DOCR-MARL Experiments.

Phase 5: Manages experiment configurations via YAML files:
- config_base.yaml: Default/base configuration
- config_high_pv.yaml: High PV penetration experiments
- config_comm_ablation.yaml: Communication ablation study
- config_curriculum.yaml: Curriculum learning experiments
- config_robustness.yaml: Robustness testing configuration

Features:
- YAML loading with validation
- Configuration inheritance (base + overrides)
- Experiment template generation
- Hyperparameter sweep support
"""

import yaml
import os
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict, field


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    name: str = "default_experiment"
    description: str = ""

    # Systems
    systems: List[str] = field(default_factory=lambda: ["8bus", "30bus"])

    # Methods
    methods: List[str] = field(default_factory=lambda: ["GA-SQP", "MAPPO", "MAPPO-Safe"])

    # PV levels
    pv_levels: List[float] = field(default_factory=lambda: [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

    # Fault types
    fault_types: List[str] = field(default_factory=lambda: ["3ph"])

    # Training parameters
    n_runs: int = 10
    fast_mode: bool = False

    # GA-SQP parameters
    ga_pop: int = 100
    ga_gen: int = 200

    # Metaheuristic parameters
    meta_pop: int = 30
    meta_iter: int = 500

    # MARL parameters
    marl_steps: int = 30000
    hidden_actor: int = 64
    hidden_critic: int = 128
    lr_actor: float = 0.02
    lr_critic: float = 0.01
    clip_eps: float = 0.2
    gamma: float = 0.99
    lam: float = 0.95
    rollout_len: int = 64
    n_epochs: int = 4

    # Curriculum learning
    use_curriculum: bool = False
    curriculum_stages: int = 6

    # Safe RL
    use_safe_rl: bool = False
    safe_rl_lambda_init: float = 1.0
    safe_rl_lambda_lr: float = 0.01
    safe_rl_rho: float = 10.0

    # EWC
    use_ewc: bool = False
    ewc_lambda: float = 100.0
    ewc_fisher_samples: int = 100

    # Communication
    use_comm: bool = False
    comm_msg_dim: int = 8
    comm_bandwidth_limit: float = 1.0  # kbps

    # Topology
    test_n1_outages: bool = False
    n1_outage_prob: float = 0.15

    # Sensor noise
    test_sensor_noise: bool = False
    sensor_noise_prob: float = 0.20

    # Scenario generation
    n_train_scenarios: int = 10000
    n_val_scenarios: int = 2000

    # Output
    output_dir: str = "outputs/phase5"
    save_checkpoints: bool = True
    checkpoint_interval: int = 100

    # Parallel execution
    n_workers: int = -1  # -1 = auto (CPU count - 1)

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> 'ExperimentConfig':
        return cls(**d)

    def to_yaml(self, filepath: str):
        """Save configuration to YAML file."""
        with open(filepath, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_yaml(cls, filepath: str) -> 'ExperimentConfig':
        """Load configuration from YAML file."""
        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)


class ConfigManager:
    """
    Manages experiment configurations with inheritance and validation.

    Usage:
        manager = ConfigManager("experiments/configs")
        config = manager.load("config_high_pv.yaml")
        config = manager.merge("config_base.yaml", overrides={"pv_levels": [0.8, 1.0]})
    """

    def __init__(self, config_dir: str = "experiments/configs"):
        self.config_dir = config_dir
        os.makedirs(config_dir, exist_ok=True)

    def load(self, filename: str) -> ExperimentConfig:
        """Load a configuration file."""
        filepath = os.path.join(self.config_dir, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Config file not found: {filepath}")
        return ExperimentConfig.from_yaml(filepath)

    def save(self, config: ExperimentConfig, filename: str):
        """Save a configuration file."""
        filepath = os.path.join(self.config_dir, filename)
        config.to_yaml(filepath)
        print(f"  Saved config to {filepath}")

    def merge(self, base_filename: str, overrides: Dict[str, Any]) -> ExperimentConfig:
        """Load base config and apply overrides."""
        config = self.load(base_filename)

        for key, value in overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)
            else:
                raise ValueError(f"Unknown config key: {key}")

        return config

    def create_default_configs(self):
        """Create default configuration files."""

        # 1. Base configuration
        base = ExperimentConfig(
            name="base",
            description="Default experiment configuration with all methods on standard test systems",
            systems=["8bus", "30bus"],
            methods=["GA-SQP", "MAPPO", "MAPPO-Safe", "MAPPO-Safe-EWC", "HWOA", "JAYA"],
            pv_levels=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            fault_types=["3ph"],
            n_runs=10,
            output_dir="outputs/phase5/base",
        )
        self.save(base, "config_base.yaml")

        # 2. High PV configuration
        high_pv = ExperimentConfig(
            name="high_pv",
            description="Focus on high PV penetration levels (60-100%)",
            systems=["8bus", "30bus", "39bus"],
            methods=["MAPPO-Safe", "MAPPO-Safe-EWC", "HWOA", "JAYA"],
            pv_levels=[0.6, 0.8, 1.0],
            fault_types=["3ph", "LL"],
            n_runs=15,
            use_safe_rl=True,
            output_dir="outputs/phase5/high_pv",
        )
        self.save(high_pv, "config_high_pv.yaml")

        # 3. Communication ablation
        comm_ablation = ExperimentConfig(
            name="comm_ablation",
            description="Ablation study: MAPPO with/without GNN communication",
            systems=["8bus", "30bus"],
            methods=["MAPPO", "MAPPO-Comm"],
            pv_levels=[0.0, 0.5, 1.0],
            fault_types=["3ph"],
            n_runs=20,
            use_comm=True,
            comm_bandwidth_limit=1.0,
            output_dir="outputs/phase5/comm_ablation",
        )
        self.save(comm_ablation, "config_comm_ablation.yaml")

        # 4. Curriculum learning
        curriculum = ExperimentConfig(
            name="curriculum",
            description="Curriculum learning: 6-stage PV progression",
            systems=["8bus", "30bus"],
            methods=["MAPPO-Safe-EWC"],
            pv_levels=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            fault_types=["3ph"],
            n_runs=10,
            use_curriculum=True,
            curriculum_stages=6,
            use_ewc=True,
            marl_steps=50000,
            output_dir="outputs/phase5/curriculum",
        )
        self.save(curriculum, "config_curriculum.yaml")

        # 5. Robustness testing
        robustness = ExperimentConfig(
            name="robustness",
            description="Robustness testing: N-1 outages + sensor noise",
            systems=["8bus", "30bus"],
            methods=["MAPPO-Safe", "GA-SQP"],
            pv_levels=[0.0, 0.5, 1.0],
            fault_types=["3ph", "LL", "1ph", "2ph-g"],
            n_runs=10,
            test_n1_outages=True,
            n1_outage_prob=0.3,
            test_sensor_noise=True,
            sensor_noise_prob=0.3,
            output_dir="outputs/phase5/robustness",
        )
        self.save(robustness, "config_robustness.yaml")

        # 6. Fast smoke test
        fast = ExperimentConfig(
            name="fast_smoke",
            description="Quick smoke test with reduced parameters",
            systems=["8bus"],
            methods=["GA-SQP", "JAYA", "MAPPO"],
            pv_levels=[0.0, 1.0],
            fault_types=["3ph"],
            n_runs=3,
            fast_mode=True,
            ga_pop=30,
            ga_gen=50,
            meta_pop=20,
            meta_iter=200,
            marl_steps=5000,
            output_dir="outputs/phase5/fast_smoke",
        )
        self.save(fast, "config_fast.yaml")

        print(f"\n  Created 6 default configuration files in {self.config_dir}")

    def list_configs(self) -> List[str]:
        """List available configuration files."""
        files = []
        for f in os.listdir(self.config_dir):
            if f.endswith('.yaml') or f.endswith('.yml'):
                files.append(f)
        return sorted(files)

    def validate(self, config: ExperimentConfig) -> List[str]:
        """Validate a configuration and return list of issues."""
        issues = []

        # Check systems
        valid_systems = ["3bus", "8bus", "30bus", "39bus"]
        for s in config.systems:
            if s not in valid_systems:
                issues.append(f"Unknown system: {s}")

        # Check methods
        valid_methods = [
            "GA-SQP", "Single-PPO", "IPPO", "MAPPO", "MAPPO-Comm",
            "MAPPO-Safe", "MAPPO-Safe-EWC", "HWOA", "HHO", "JAYA", "EEO"
        ]
        for m in config.methods:
            if m not in valid_methods:
                issues.append(f"Unknown method: {m}")

        # Check PV levels
        for pv in config.pv_levels:
            if not 0.0 <= pv <= 1.0:
                issues.append(f"PV level out of range [0,1]: {pv}")

        # Check fault types
        valid_faults = ["3ph", "LL", "1ph", "2ph-g"]
        for f in config.fault_types:
            if f not in valid_faults:
                issues.append(f"Unknown fault type: {f}")

        # Check positive values
        if config.n_runs <= 0:
            issues.append("n_runs must be positive")
        if config.ga_pop <= 0:
            issues.append("ga_pop must be positive")
        if config.marl_steps <= 0:
            issues.append("marl_steps must be positive")

        return issues


def create_experiment_from_config(config: ExperimentConfig) -> Dict:
    """
    Convert ExperimentConfig to a runnable experiment specification.

    Returns:
        Dictionary with all parameters for the training pipeline.
    """
    return {
        "name": config.name,
        "systems": config.systems,
        "methods": config.methods,
        "pv_levels": config.pv_levels,
        "fault_types": config.fault_types,
        "n_runs": config.n_runs,
        "fast": config.fast_mode,
        "ga_pop": config.ga_pop,
        "ga_gen": config.ga_gen,
        "meta_pop": config.meta_pop,
        "meta_iter": config.meta_iter,
        "marl_steps": config.marl_steps,
        "use_curriculum": config.use_curriculum,
        "use_safe_rl": config.use_safe_rl,
        "use_ewc": config.use_ewc,
        "use_comm": config.use_comm,
        "output_dir": config.output_dir,
        "n_workers": config.n_workers,
    }


if __name__ == "__main__":
    # Create default configurations
    manager = ConfigManager("experiments/configs")
    manager.create_default_configs()

    # List and validate
    print("\n  Available configurations:")
    for cfg_name in manager.list_configs():
        cfg = manager.load(cfg_name)
        issues = manager.validate(cfg)
        status = "✓" if not issues else "✗"
        print(f"    {status} {cfg_name}: {cfg.name}")
        if issues:
            for issue in issues:
                print(f"      - {issue}")
