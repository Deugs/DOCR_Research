#!/usr/bin/env python3
"""
DOCR-MARL Configuration System
═══════════════════════════════════════════════════════════════════════════════
Unified configuration management supporting:
  • YAML/JSON config files for reproducible experiments
  • CLI argument overrides (takes precedence over config file)
  • Environment variable fallbacks
  • Validation and sensible defaults
  • Multiple preset configs for common scenarios

Usage:
  # Use default config
  python run_thesis_experiments.py --all
  
  # Use custom config file
  python run_thesis_experiments.py --all --config experiments/8bus_full.yaml
  
  # Override specific parameters via CLI
  python run_thesis_experiments.py --all --system 14bus --n-runs 20 --episodes 2000
  
  # Fast mode with custom system
  python run_thesis_experiments.py --all --fast --system 3bus --n-runs 2
"""

import argparse
import os
import sys
import json
import yaml
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Union
from enum import Enum


# ═══════════════════════════════════════════════════════════════════════════════
# ENUMS & CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

class SystemType(Enum):
    """Supported IEEE test systems."""
    BUS_3 = "3bus"
    BUS_8 = "8bus"
    BUS_14 = "14bus"
    BUS_30 = "30bus"
    BUS_39 = "39bus"
    BUS_57 = "57bus"
    BUS_118 = "118bus"


class FaultType(Enum):
    """Supported fault types."""
    THREE_PHASE = "3ph"
    LINE_TO_LINE = "LL"
    SINGLE_PHASE = "1ph"
    TWO_PHASE_GROUND = "2ph-g"


# System metadata (relays, pairs, typical training times)
SYSTEM_METADATA = {
    # n_relays/n_pairs are static text and can drift from the actual
    # generated topology in network/network_data.py (a prior audit found
    # 14/39/57/118bus stale here) — verify against
    # NETWORKS[name]().n_pairs / Iload_max.size if these are ever
    # suspected stale again.
    "3bus":   {"n_relays": 6,   "n_pairs": 6,   "fast_episodes": 100,  "full_episodes": 1000},
    "8bus":   {"n_relays": 14,  "n_pairs": 20,  "fast_episodes": 500,  "full_episodes": 5000},
    "14bus":  {"n_relays": 40,  "n_pairs": 52,  "fast_episodes": 500,  "full_episodes": 5000},
    "30bus":  {"n_relays": 42,  "n_pairs": 70,  "fast_episodes": 500,  "full_episodes": 5000},
    "39bus":  {"n_relays": 46,  "n_pairs": 42,  "fast_episodes": 500,  "full_episodes": 5000},
    "57bus":  {"n_relays": 156, "n_pairs": 192, "fast_episodes": 500,  "full_episodes": 5000},
    "118bus": {"n_relays": 318, "n_pairs": 440, "fast_episodes": 500,  "full_episodes": 5000},
}


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION DATACLASSES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SystemConfig:
    """Power system configuration."""
    name: str = "8bus"                    # System type: 3bus, 8bus, 14bus, 30bus, 39bus, 57bus, 118bus
    pv_penetration: float = 0.6           # 0.0 to 1.0 — active sweep: 0.0, 0.2, 0.4, 0.6
    fault_type: str = "3ph"               # 3ph, LL, 1ph, 2ph-g
    use_pandapower: bool = False          # Use pandapower for dynamic simulation
    curriculum_stage: int = 3             # 0-5 (0%,20%,40%,60%,80%,100%) — 3 = 60% PV (current max)


@dataclass
class TrainingConfig:
    """Core training hyperparameters."""
    # Run counts
    n_runs: int = 50                      # Number of independent training runs
    # These are now only the MINIMUM floor for fast mode's n_runs -- Cores
    # A/B/C/D auto-scale n_runs at runtime via run_core_training.py's
    # _n_runs_for_wave(n_workers, jobs_per_run, minimum) to exactly fill
    # one worker-pool wave for whatever --n-workers was actually passed
    # (e.g. Core C uses n_runs=6 on a 32-worker pod, 12 on a 64-worker
    # one), so a fixed constant here would either waste a bigger pod's
    # capacity or force extra waves on a smaller one. This floor only
    # matters when n_workers is small enough that auto-scaling wouldn't
    # help (e.g. a n_workers=1 laptop run) -- kept at the original 2.
    n_runs_fast: int = 2                # Fast mode runs (floor; see above)
    n_runs_baseline: int = 10           # Baseline comparison runs
    n_runs_baseline_fast: int = 2       # Fast baseline runs (floor; see above)
    n_runs_hparam: int = 3              # Hyperparameter grid seeds
    n_runs_hparam_fast: int = 1         # Fast hparam seeds
    
    # Episode counts
    n_episodes: int = 5000              # Episodes per run (full)
    n_episodes_fast: int = 500          # Episodes per run (fast)
    n_episodes_hparam: int = 2000       # Episodes for hyperparameter search
    n_episodes_hparam_fast: int = 500   # Fast hparam episodes
    
    # Algorithm
    hidden_actor: int = 64              # Actor hidden layer size
    hidden_critic: int = 128            # Critic hidden layer size
    lr_actor: float = 0.02            # Actor learning rate
    lr_critic: float = 0.01           # Critic learning rate
    gamma: float = 0.99               # Discount factor
    lam: float = 0.95                 # GAE lambda
    clip_eps: float = 0.2             # PPO clip ratio
    
    # Features
    use_comm: bool = False              # Enable GNN communication
    use_safety: bool = True             # Enable safety projection layer
    use_ewc: bool = True                # Enable Elastic Weight Consolidation
    curriculum: bool = False            # Enable curriculum learning
    
    # EWC parameters
    lambda_ewc: float = 1e4             # EWC regularization strength
    alpha_lambda: float = 1e-4          # Lagrange multiplier learning rate
    
    # Misc
    seed: int = 42                      # Global random seed
    rollout_len: int = 64               # Steps per rollout
    n_epochs: int = 4                   # Update epochs per rollout
    action_scale: float = 0.05          # Action scaling factor
    max_steps: int = 300                # Max steps per episode


@dataclass
class ExperimentConfig:
    """Experiment orchestration settings."""
    # Execution
    fast_mode: bool = False             # Reduced iterations for testing
    use_synthetic: bool = False            # Enable synthetic fallback on failure
    parallel: bool = False                # Unused placeholder — see n_workers below for the real implementation
    n_workers: int = 1                  # Worker processes for each Core's independent per-run training (Cores A-F)
    seed: int = 42                       # Global random seed (also in TrainingConfig.seed for training)
    
    # Cores to run (A-K)
    cores: List[str] = field(default_factory=lambda: list("ABCDEFGHIJK"))
    
    # Output
    output_dir: str = "results"         # Base output directory
    save_checkpoints: bool = True       # Save model checkpoints
    checkpoint_interval: int = 1000     # Episodes between checkpoints
    
    # Logging
    log_level: str = "INFO"             # DEBUG, INFO, WARNING, ERROR
    log_every: int = 500                # Episodes between log prints
    
    # Cache
    clear_cache: bool = False           # Clear cache before running
    cache_dir: str = "results/core_cache"


@dataclass
class PostProcessingConfig:
    """Figure generation settings."""
    figures_dir: str = "results/figures"
    dpi: int = 200                      # Figure resolution
    fig_format: str = "pdf"           # pdf, png, svg
    generate_all: bool = True         # Generate all figures
    selected_figures: List[int] = field(default_factory=list)
    phases: List[int] = field(default_factory=list)


@dataclass
class DOCRConfig:
    """Master configuration container."""
    system: SystemConfig = field(default_factory=SystemConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    postprocessing: PostProcessingConfig = field(default_factory=PostProcessingConfig)
    
    # Metadata
    version: str = "2.0"
    description: str = ""
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DOCRConfig":
        """Create config from dictionary (e.g., loaded YAML/JSON)."""
        return cls(
            system=SystemConfig(**d.get("system", {})),
            training=TrainingConfig(**d.get("training", {})),
            experiment=ExperimentConfig(**d.get("experiment", {})),
            postprocessing=PostProcessingConfig(**d.get("postprocessing", {})),
            version=d.get("version", "2.0"),
            description=d.get("description", ""),
        )
    
    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "DOCRConfig":
        """Load config from YAML file."""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)
    
    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "DOCRConfig":
        """Load config from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    def to_yaml(self, path: Union[str, Path]):
        """Save config to YAML file."""
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)
    
    def to_json(self, path: Union[str, Path]):
        """Save config to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of errors."""
        errors = []
        
        # System validation
        if self.system.name not in SYSTEM_METADATA:
            errors.append(f"Unknown system: {self.system.name}. Available: {list(SYSTEM_METADATA.keys())}")
        if not 0.0 <= self.system.pv_penetration <= 1.0:
            errors.append(f"PV penetration must be in [0, 1], got {self.system.pv_penetration}")
        if self.system.fault_type not in [ft.value for ft in FaultType]:
            errors.append(f"Unknown fault type: {self.system.fault_type}")
        
        # Training validation
        if self.training.n_runs < 1:
            errors.append("n_runs must be >= 1")
        if self.training.n_episodes < 1:
            errors.append("n_episodes must be >= 1")
        if self.training.hidden_actor < 1:
            errors.append("hidden_actor must be >= 1")
        if self.training.hidden_critic < 1:
            errors.append("hidden_critic must be >= 1")
        
        # EWC validation
        if self.training.use_ewc and self.training.lambda_ewc <= 0:
            errors.append("lambda_ewc must be positive when EWC is enabled")
        
        return errors
    
    def apply_fast_mode(self):
        """Adjust parameters for fast mode testing."""
        self.experiment.fast_mode = True
        self.training.n_runs = self.training.n_runs_fast
        self.training.n_runs_baseline = self.training.n_runs_baseline_fast
        self.training.n_runs_hparam = self.training.n_runs_hparam_fast
        self.training.n_episodes = self.training.n_episodes_fast
        self.training.n_episodes_hparam = self.training.n_episodes_hparam_fast
        
        # Adjust system-specific episode counts
        meta = SYSTEM_METADATA.get(self.system.name, {})
        if meta:
            self.training.n_episodes = meta.get("fast_episodes", 500)
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get metadata for the configured system."""
        return SYSTEM_METADATA.get(self.system.name, {})


# ═══════════════════════════════════════════════════════════════════════════════
# PRESET CONFIGURATIONS
# ═══════════════════════════════════════════════════════════════════════════════

PRESETS = {
    "thesis_sweep": {
        "description": "Active thesis configuration: 8/14/39-bus, 0/20/40/60% PV",
        "system": {"name": "8bus", "pv_penetration": 0.6, "fault_type": "3ph", "curriculum_stage": 3},
        "training": {"n_runs": 50, "n_episodes": 5000, "use_safety": True, "use_ewc": True, "use_comm": True},
    },
    "thesis_sweep_fast": {
        "description": "Fast smoke test: 8-bus, 0/20/40/60% PV",
        "system": {"name": "8bus", "pv_penetration": 0.6, "fault_type": "3ph", "curriculum_stage": 3},
        "training": {"n_runs": 2, "n_episodes": 500, "use_safety": True, "use_ewc": True},
        "experiment": {"fast_mode": True},
    },
    "8bus_full": {
        "description": "Full 8-bus experiment — max PV 60% (active sweep)",
        "system": {"name": "8bus", "pv_penetration": 0.6, "fault_type": "3ph", "curriculum_stage": 3},
        "training": {"n_runs": 50, "n_episodes": 5000, "use_safety": True, "use_ewc": True},
    },
    "8bus_fast": {
        "description": "Fast 8-bus smoke test",
        "system": {"name": "8bus", "pv_penetration": 0.6, "fault_type": "3ph", "curriculum_stage": 3},
        "training": {"n_runs": 2, "n_episodes": 500, "use_safety": True, "use_ewc": True},
        "experiment": {"fast_mode": True},
    },
    "14bus_full": {
        "description": "Full 14-bus meshed network experiment — max PV 60% (active sweep)",
        "system": {"name": "14bus", "pv_penetration": 0.6, "fault_type": "3ph", "curriculum_stage": 3},
        "training": {"n_runs": 50, "n_episodes": 5000, "use_safety": True, "use_ewc": True},
    },
    "39bus_full": {
        "description": "Full 39-bus New England experiment — max PV 60% (active sweep)",
        "system": {"name": "39bus", "pv_penetration": 0.6, "fault_type": "3ph", "curriculum_stage": 3},
        "training": {"n_runs": 30, "n_episodes": 5000, "use_safety": True, "use_ewc": True},
    },
    "30bus_full": {
        "description": "Full 30-bus experiment (not in active thesis sweep — see thesis_sweep)",
        "system": {"name": "30bus", "pv_penetration": 1.0, "fault_type": "3ph"},
        "training": {"n_runs": 30, "n_episodes": 5000, "use_safety": True, "use_ewc": True},
    },
    "3bus_toy": {
        "description": "3-bus toy validation (quick test)",
        "system": {"name": "3bus", "pv_penetration": 0.0, "fault_type": "3ph"},
        "training": {"n_runs": 5, "n_episodes": 1000, "use_safety": True, "use_ewc": False},
        "experiment": {"fast_mode": True},
    },
    "scalability_sweep": {
        "description": "Scalability analysis across all systems",
        "system": {"name": "8bus"},
        "training": {"n_runs": 10, "n_episodes": 1000},
        "experiment": {"cores": ["F"]},
    },
    "curriculum_learning": {
        "description": "Curriculum learning 0%->60% PV on 8-bus",
        "system": {"name": "8bus", "pv_penetration": 0.0},
        "training": {"n_runs": 10, "n_episodes": 500, "curriculum": True},
        "experiment": {"cores": ["B"]},
    },
    "ablation_study": {
        "description": "Systematic ablation study",
        "system": {"name": "8bus", "pv_penetration": 0.6},
        "training": {"n_runs": 10, "n_episodes": 500},
        "experiment": {"cores": ["D"]},
    },
}


def get_preset(name: str) -> Dict[str, Any]:
    """Get a preset configuration by name."""
    if name not in PRESETS:
        available = ", ".join(PRESETS.keys())
        raise ValueError(f"Unknown preset '{name}'. Available: {available}")
    return PRESETS[name]


def list_presets() -> Dict[str, str]:
    """List all available presets with descriptions."""
    return {name: preset.get("description", "No description") 
            for name, preset in PRESETS.items()}


# ═══════════════════════════════════════════════════════════════════════════════
# CLI ARGUMENT PARSER
# ═══════════════════════════════════════════════════════════════════════════════

def build_config_parser() -> argparse.ArgumentParser:
    """Build argument parser with all configuration options."""
    parser = argparse.ArgumentParser(
        description="DOCR-MARL Experiment Configuration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Configuration Precedence (highest to lowest):
  1. CLI arguments (--system, --n-runs, etc.)
  2. Config file (--config)
  3. Preset (--preset)
  4. Environment variables (DOCR_SYSTEM, DOCR_N_RUNS, etc.)
  5. Default values

Examples:
  # Use preset
  python run_thesis_experiments.py --all --preset 8bus_fast
  
  # Custom config file
  python run_thesis_experiments.py --all --config my_config.yaml
  
  # CLI overrides
  python run_thesis_experiments.py --all --system 14bus --n-runs 20 --episodes 2000
  
  # Fast mode with specific system
  python run_thesis_experiments.py --all --fast --system 3bus
  
  # List presets
  python run_thesis_experiments.py --list-presets
  
  # Generate example config
  python run_thesis_experiments.py --generate-config example.yaml
        """
    )
    
    # Config file / preset
    parser.add_argument("--config", "-c", type=str, 
                        help="Path to YAML/JSON config file")
    parser.add_argument("--preset", "-p", type=str,
                        help=f"Use a preset configuration. Available: {', '.join(PRESETS.keys())}")
    parser.add_argument("--list-presets", action="store_true",
                        help="List all available presets")
    parser.add_argument("--generate-config", type=str, metavar="PATH",
                        help="Generate an example config file and exit")
    
    # System configuration
    sys_group = parser.add_argument_group("System Configuration")
    sys_group.add_argument("--system", "-s", type=str,
                           choices=list(SYSTEM_METADATA.keys()),
                           help="IEEE test system (active sweep: 8bus, 14bus, 39bus — others available)")
    sys_group.add_argument("--pv", type=float, metavar="[0-1]",
                           help="PV penetration level (0.0 to 1.0)")
    sys_group.add_argument("--fault-type", type=str,
                           choices=[ft.value for ft in FaultType],
                           help="Fault type for protection study")
    sys_group.add_argument("--use-pandapower", action="store_true", default=None,
                           help="Enable pandapower dynamic simulation")
    sys_group.add_argument("--curriculum-stage", type=int, choices=range(6),
                           help="Curriculum stage (0-5)")
    
    # Training configuration
    train_group = parser.add_argument_group("Training Configuration")
    train_group.add_argument("--n-runs", type=int, help="Number of independent runs")
    train_group.add_argument("--episodes", "-e", type=int, help="Episodes per run")
    train_group.add_argument("--n-workers", type=int,
                              help="Parallel worker processes for each Core's independent per-run "
                                   "training (Cores A-F). Default: 1 (sequential). Results are "
                                   "identical regardless of n_workers.")
    train_group.add_argument("--hidden-actor", type=int, help="Actor hidden layer size")
    train_group.add_argument("--hidden-critic", type=int, help="Critic hidden layer size")
    train_group.add_argument("--lr-actor", type=float, help="Actor learning rate")
    train_group.add_argument("--lr-critic", type=float, help="Critic learning rate")
    train_group.add_argument("--gamma", type=float, help="Discount factor")
    train_group.add_argument("--lam", type=float, help="GAE lambda")
    train_group.add_argument("--clip-eps", type=float, help="PPO clip ratio")
    
    # Feature toggles
    feat_group = parser.add_argument_group("Feature Toggles")
    # default=None (not the implicit False) on every store_true flag below
    # that feeds into config_from_args()'s cli_mappings loop: that loop
    # applies a CLI value whenever it "is not None", to distinguish "user
    # didn't mention this flag" from "user explicitly set it". store_true's
    # ordinary default of False doesn't satisfy that — it made every one of
    # these flags unconditionally overwrite whatever a --preset/--config set,
    # every single invocation, whether or not the flag was actually passed
    # (e.g. --preset thesis_sweep_fast's fast_mode=True was silently
    # clobbered back to False by the mere presence of --fast's untouched
    # default, and --no-synthetic's invert-lambda forced use_synthetic=True
    # on every run that didn't explicitly pass --no-synthetic).
    feat_group.add_argument("--use-comm", action="store_true", default=None, help="Enable GNN communication")
    feat_group.add_argument("--no-safety", action="store_true", default=None, help="Disable safety layer")
    feat_group.add_argument("--no-ewc", action="store_true", default=None, help="Disable EWC")
    feat_group.add_argument("--curriculum", action="store_true", default=None, help="Enable curriculum learning")
    
    # EWC parameters
    ewc_group = parser.add_argument_group("EWC Parameters")
    ewc_group.add_argument("--lambda-ewc", type=float, help="EWC regularization strength")
    ewc_group.add_argument("--alpha-lambda", type=float, help="Lagrange multiplier learning rate")
    
    # Experiment control
    exp_group = parser.add_argument_group("Experiment Control")
    exp_group.add_argument("--fast", action="store_true", default=None, help="Fast mode (reduced iterations)")
    exp_group.add_argument("--no-synthetic", action="store_true", default=None, help="Disable synthetic fallback")
    exp_group.add_argument("--cores", type=str, nargs="+", 
                           help="Specific cores to run (A-K)")
    exp_group.add_argument("--seed", type=int, help="Global random seed")
    exp_group.add_argument("--output-dir", type=str, help="Output directory")
    
    # Logging
    log_group = parser.add_argument_group("Logging")
    log_group.add_argument("--log-level", type=str, 
                           choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                           help="Logging level")
    log_group.add_argument("--log-every", type=int, help="Log interval (episodes)")
    
    # Cache
    cache_group = parser.add_argument_group("Cache")
    cache_group.add_argument("--clear-cache", action="store_true", default=None, help="Clear cache before running")
    cache_group.add_argument("--cache-dir", type=str, help="Cache directory")
    
    return parser


def config_from_args(args: argparse.Namespace) -> DOCRConfig:
    """
    Build DOCRConfig from parsed CLI arguments.
    
    Precedence:
      1. CLI args
      2. Config file
      3. Preset
      4. Environment variables
      5. Defaults
    """
    # Start with defaults
    config = DOCRConfig()
    
    # Apply preset if specified
    if args.preset:
        preset_data = get_preset(args.preset)
        config = DOCRConfig.from_dict(preset_data)
    
    # Apply config file if specified
    if args.config:
        path = Path(args.config)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        if path.suffix in ('.yaml', '.yml'):
            config = DOCRConfig.from_yaml(path)
        elif path.suffix == '.json':
            config = DOCRConfig.from_json(path)
        else:
            raise ValueError(f"Unsupported config format: {path.suffix}. Use .yaml, .yml, or .json")
    
    # Apply environment variables
    env_mappings = {
        'DOCR_SYSTEM': ('system', 'name', str),
        'DOCR_PV': ('system', 'pv_penetration', float),
        'DOCR_FAULT_TYPE': ('system', 'fault_type', str),
        'DOCR_N_RUNS': ('training', 'n_runs', int),
        'DOCR_EPISODES': ('training', 'n_episodes', int),
        'DOCR_HIDDEN_ACTOR': ('training', 'hidden_actor', int),
        'DOCR_HIDDEN_CRITIC': ('training', 'hidden_critic', int),
        'DOCR_SEED': ('training', 'seed', int),
        'DOCR_FAST': ('experiment', 'fast_mode', lambda x: x.lower() in ('1', 'true', 'yes')),
        'DOCR_USE_SYNTHETIC': ('experiment', 'use_synthetic', lambda x: x.lower() in ('1', 'true', 'yes')),
        'DOCR_OUTPUT_DIR': ('experiment', 'output_dir', str),
        'DOCR_LOG_LEVEL': ('experiment', 'log_level', str),
    }
    
    for env_var, (section, key, converter) in env_mappings.items():
        value = os.environ.get(env_var)
        if value is not None:
            try:
                converted = converter(value)
                section_obj = getattr(config, section)
                setattr(section_obj, key, converted)
            except (ValueError, TypeError) as e:
                print(f"Warning: Could not parse {env_var}={value}: {e}")
    
    # Apply CLI overrides (highest precedence)
    cli_mappings = {
        'system': ('system', 'name'),
        'pv': ('system', 'pv_penetration'),
        'fault_type': ('system', 'fault_type'),
        'use_pandapower': ('system', 'use_pandapower'),
        'curriculum_stage': ('system', 'curriculum_stage'),
        'n_runs': ('training', 'n_runs'),
        'episodes': ('training', 'n_episodes'),
        'hidden_actor': ('training', 'hidden_actor'),
        'hidden_critic': ('training', 'hidden_critic'),
        'lr_actor': ('training', 'lr_actor'),
        'lr_critic': ('training', 'lr_critic'),
        'gamma': ('training', 'gamma'),
        'lam': ('training', 'lam'),
        'clip_eps': ('training', 'clip_eps'),
        'use_comm': ('training', 'use_comm'),
        'no_safety': ('training', 'use_safety', lambda x: not x),  # Invert
        'no_ewc': ('training', 'use_ewc', lambda x: not x),  # Invert
        'curriculum': ('training', 'curriculum'),
        'lambda_ewc': ('training', 'lambda_ewc'),
        'alpha_lambda': ('training', 'alpha_lambda'),
        'seed': ('training', 'seed'),
        'fast': ('experiment', 'fast_mode'),
        'n_workers': ('experiment', 'n_workers'),
        'no_synthetic': ('experiment', 'use_synthetic', lambda x: not x),
        'cores': ('experiment', 'cores'),
        'output_dir': ('experiment', 'output_dir'),
        'log_level': ('experiment', 'log_level'),
        'log_every': ('experiment', 'log_every'),
        'clear_cache': ('experiment', 'clear_cache'),
        'cache_dir': ('experiment', 'cache_dir'),
    }
    
    for arg_name, mapping in cli_mappings.items():
        value = getattr(args, arg_name, None)
        if value is not None:
            section = mapping[0]
            key = mapping[1]
            # Apply transformer if provided
            if len(mapping) > 2:
                value = mapping[2](value)
            section_obj = getattr(config, section)
            setattr(section_obj, key, value)
    
    # Apply fast mode adjustments
    if config.experiment.fast_mode:
        config.apply_fast_mode()
    
    # Validate
    errors = config.validate()
    if errors:
        print("Configuration errors:")
        for err in errors:
            print(f"  ✗ {err}")
        raise ValueError("Invalid configuration")
    
    return config


def generate_example_config(path: str):
    """Generate an example configuration file."""
    example = DOCRConfig(
        system=SystemConfig(
            name="8bus",
            pv_penetration=0.6,          # Active sweep max: 0/20/40/60%
            fault_type="3ph",
            use_pandapower=False,
            curriculum_stage=3,          # 60% PV
        ),
        training=TrainingConfig(
            n_runs=50,
            n_episodes=5000,
            hidden_actor=64,
            hidden_critic=128,
            use_safety=True,
            use_ewc=True,
            lambda_ewc=1e4,
            alpha_lambda=1e-4,
        ),
        experiment=ExperimentConfig(
            fast_mode=False,
            use_synthetic=False,
            seed=42,
            cores=list("ABCDEFGHIJK"),
            output_dir="results",
            log_level="INFO",
        ),
        postprocessing=PostProcessingConfig(
            figures_dir="results/figures",
            dpi=200,
            fig_format="pdf",
        ),
        description="DOCR-MARL thesis configuration — 8/14/39-bus, 0/20/40/60% PV sweep",
    )
    
    path = Path(path)
    if path.suffix in ('.yaml', '.yml'):
        example.to_yaml(path)
    elif path.suffix == '.json':
        example.to_json(path)
    else:
        # Default to YAML
        path = path.with_suffix('.yaml')
        example.to_yaml(path)
    
    print(f"Example config saved to: {path}")
    return path


# ═══════════════════════════════════════════════════════════════════════════════
# INTEGRATION HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def print_config_summary(config: DOCRConfig):
    """Print a formatted summary of the current configuration."""
    sys_info = config.get_system_info()
    
    print("\n" + "=" * 70)
    print("DOCR-MARL CONFIGURATION SUMMARY")
    print("=" * 70)
    
    print(f"\n[SYSTEM] {config.system.name.upper()}")
    print(f"   - Relays: ...")
    print(f"   - Pandapower: {'ON' if config.system.use_pandapower else 'OFF'}")
    print(f"\n[TRAINING]")
    print(f"   - GNN Communication: {'ON' if config.training.use_comm else 'OFF'}")
    print(f"   - Safety Layer: {'ON' if config.training.use_safety else 'OFF'}")
    print(f"   - EWC: {'ON' if config.training.use_ewc else 'OFF'}")
    print(f"   - Curriculum: {'ON' if config.training.curriculum else 'OFF'}")
    print(f"\n[EXPERIMENT]")
    print(f"   - Synthetic Fallback: {'ON' if config.experiment.use_synthetic else 'OFF'}")
    print(f"\n[POST-PROCESSING]")
    
    if config.description:
        print(f"\n Description: {config.description}")
    
    print("=" * 70 + "\n")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN (for testing)
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = build_config_parser()
    args = parser.parse_args()
    
    if args.list_presets:
        print("\nAvailable Presets:")
        print("=" * 60)
        for name, desc in list_presets().items():
            print(f"  {name:<25} {desc}")
        print()
        sys.exit(0)
    
    if args.generate_config:
        generate_example_config(args.generate_config)
        sys.exit(0)
    
    # Build and display config
    config = config_from_args(args)
    print_config_summary(config)