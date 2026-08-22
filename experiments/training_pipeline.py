"""
experiments/training_pipeline.py
────────────────────────────────
Large-Scale Training Pipeline for DOCR-MARL.

Phase 5: Orchestrates large-scale experiments across:
- Multiple test systems (8-bus, 30-bus, 39-bus)
- Multiple PV penetration levels (0-100%)
- Multiple fault types (3ph, LL, 1ph, 2ph-g)
- Multiple topology configurations (N-1 contingencies)
- Multiple algorithms (GA-SQP, MARL variants, metaheuristics)

Extended to support all 17 stub figures (13,14,20,22,23,24,25,36,44,45,46,47,48,49,50,53,54,58).

Features:
- Parallel execution support (multiprocessing)
- Checkpoint/resume for long-running experiments
- Progress tracking and logging
- Automatic result aggregation
- Curriculum learning orchestration
- Figure-specific evaluation pipelines
"""

import numpy as np
import json
import os
import time
import pickle
import itertools
from network.relay_physics import check_constraints
from typing import List, Dict, Tuple, Optional, Callable, Any, Union
from dataclasses import dataclass, asdict, field
from multiprocessing import Pool, cpu_count
import warnings

# ═══════════════════════════════════════════════════════════════════════════════
#  PART 1: Extended Configuration & Result Dataclasses
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TrainingConfig:
    """
    Configuration for a single training run.
    Extended to support all 17 stub figures (13,14,20,22,23,24,25,36,44,45,46,47,48,49,50,53,54,58).
    """
    # ── Core identification ──
    method: str  # "GA-SQP", "IPPO", "MAPPO", "MAPPO-Comm", "MAPPO-Safe", "MAPPO-Safe-EWC",
                 # "HWOA", "HHO", "JAYA", "EEO", "Single-PPO"
    system: str  # "8bus", "14bus", "39bus"
    pv_level: float  # 0.0, 0.2, 0.4, 0.6
    fault_type: str  # "3ph", "LL", "1ph", "2ph-g"
    topology_changes: Optional[Dict] = None
    n_runs: int = 10
    seed_offset: int = 0

    # ── Method-specific parameters ──
    ga_pop: int = 100
    ga_gen: int = 200
    meta_pop: int = 30
    meta_iter: int = 500
    marl_steps: int = 30000

    # ── Curriculum & Safe RL ──
    use_curriculum: bool = False
    curriculum_stage: int = 0
    use_safe_rl: bool = False
    use_ewc: bool = False
    use_comm: bool = False

    # ═════════════════════════════════════════════════════════════════════════
    #  FIGURE-SPECIFIC CONFIGURATIONS (NEW)
    # ═════════════════════════════════════════════════════════════════════════

    # ── Fig 13: Communication Stress ──
    comm_stress_level: float = 0.0           # 0.0=none, 0.3=light, 0.6=moderate, 0.9=severe
    packet_loss_rate: float = 0.0              # Probability of message drop
    latency_ms: float = 0.0                    # Artificial delay in ms
    quantization_bits: int = 32                # Message quantization (32=full, 8, 4, 2)

    # ── Fig 14: N-1 Contingency ──
    n1_line_outage: Optional[int] = None       # Index of line to trip; None = base case
    n1_recovery_eval: bool = False             # Evaluate post-recovery performance

    # ── Fig 20: Byzantine Agents ──
    byzantine_agents: List[int] = field(default_factory=list)  # Agent IDs acting maliciously
    byzantine_strategy: str = "random"         # "random", "reverse", "colluding"

    # ── Fig 22: Bandwidth vs Performance ──
    gnn_depth: int = 2                         # Number of GNN layers
    compression_ratio: float = 1.0             # 1.0=none, 0.5=half, 0.25=quarter
    message_dim: int = 8                       # Communication message dimension

    # ── Fig 23: Unseen Fault Generalisation ──
    fault_impedance_ohm: float = 0.0           # Fault impedance (0=bolted, >0=high-Z)
    evolving_fault: bool = False               # Time-varying fault impedance
    unseen_fault_eval: bool = False            # Evaluate on fault types not seen in training
    unseen_fault_type: Optional[str] = None    # e.g. "LL", "1ph", "2ph-g" — actual env-supported axis

    # ── Fig 24: Sensor Robustness ──
    sensor_noise_std: float = 0.0              # Std dev of Gaussian noise on observations
    missing_sensor_rate: float = 0.0             # Probability of sensor dropout
    sensor_bias: float = 0.0                     # Constant offset on observations (systematic bias)

    # ── Fig 25: Zero-Shot Transfer ──
    train_system: Optional[str] = None         # System trained on (None = same as system)
    eval_system: Optional[str] = None          # System evaluated on (None = same as system)
    freeze_transfer_weights: bool = False      # Freeze weights during transfer eval

    # ── Fig 36: Action Space Geometry ──
    action_space_grid: bool = False            # Run grid search over TDS/PSM space
    grid_tds_range: Tuple[float, float] = (0.5, 2.0)
    grid_psm_range: Tuple[float, float] = (1.0, 5.0)
    grid_resolution: int = 20

    # ── Fig 44: Fault Current Contribution ──
    run_shortcircuit_study: bool = False       # Use pandapower shortcircuit

    # ── Fig 45: Daily PV & Adaptive Response ──
    daily_profile: bool = False                # Simulate 24-hour irradiance profile
    irradiance_profile: Optional[np.ndarray] = None  # Custom profile; None = synthetic

    # ── Fig 46: Relay Operating Curves ──
    relay_curve_eval: bool = False             # Compute t_op vs I_f for learned settings
    fault_current_range: Tuple[float, float] = (100.0, 10000.0)

    # ── Fig 47: Sympathetic Tripping ──
    sympathetic_tripping_eval: bool = False    # Check backup relay miscoordination

    # ── Fig 48: SHAP Feature Importance ──
    shap_eval: bool = False                    # Run SHAP analysis
    shap_n_background: int = 100               # Background samples for KernelExplainer

    # ── Fig 49: Decision Boundary (t-SNE) ──
    tsne_eval: bool = False                    # Collect obs/action pairs for t-SNE
    tsne_n_samples: int = 500

    # ── Fig 50: Critical P/B Pair Ranking ──
    critical_pair_eval: bool = False           # Extract Lagrange multipliers from Safe RL

    # ── Fig 53: Model Compression ──
    quantize_bits: Optional[int] = None        # None=full, 8=int8, 4=int4
    prune_ratio: float = 0.0                   # Fraction of weights pruned (0.0-0.9)
    compression_eval: bool = False           # Evaluate compressed model

    # ── Fig 54: Adversarial Robustness ──
    adversarial_attack: Optional[str] = None   # "FGSM", "PGD", "SPSA"
    attack_epsilon: float = 0.01               # Attack perturbation magnitude
    adversarial_eval: bool = False             # Run adversarial evaluation

    # ── Fig 58: Seasonal Generalisation ──
    season: Optional[str] = None               # "spring", "summer", "autumn", "winter"
    seasonal_temperature: Optional[float] = None  # Override ambient temperature

    def to_dict(self) -> Dict:
        d = asdict(self)
        # Convert numpy arrays to lists for JSON serialization
        for key in ["irradiance_profile"]:
            if isinstance(d.get(key), np.ndarray):
                d[key] = d[key].tolist()
        return d

    def __post_init__(self):
        """Validate configuration."""
        if self.comm_stress_level < 0 or self.comm_stress_level > 1:
            raise ValueError("comm_stress_level must be in [0, 1]")
        if self.packet_loss_rate < 0 or self.packet_loss_rate > 1:
            raise ValueError("packet_loss_rate must be in [0, 1]")
        if self.missing_sensor_rate < 0 or self.missing_sensor_rate > 1:
            raise ValueError("missing_sensor_rate must be in [0, 1]")
        if self.prune_ratio < 0 or self.prune_ratio > 0.99:
            raise ValueError("prune_ratio must be in [0, 0.99]")


@dataclass
class TrainingResult:
    """Result from a single training run with figure-specific metrics."""
    config: TrainingConfig
    run_id: int

    # ── Core performance ──
    best_of: float
    best_tms: np.ndarray
    best_ps: np.ndarray
    cti_satisfied: bool
    cti_violations: int
    cti_min: float
    elapsed_seconds: float
    history: List[float]

    # ── Method-specific ──
    comm_bandwidth: float = 0.0
    safe_rl_stats: Optional[Dict] = None
    ewc_stats: Optional[Dict] = None

    # ── Trained policy (REQUIRED for any downstream eval to be authentic) ──
    # Flat parameter vector from SharedGaussianActor.get_all_params(). Without
    # this, downstream _eval_* methods have no way to act other than random
    # noise or the single best (TMS, PS) point found during training, which
    # is not a "policy" in any reactive sense (e.g. cannot respond to sensor
    # noise, packet loss, Byzantine messages, or unseen faults at evaluation
    # time, since it's just two static arrays).
    actor_params: Optional[np.ndarray] = None
    obs_dim: Optional[int] = None
    act_dim: Optional[int] = None
    actor_hidden: int = 64

    # ── Communication module (REQUIRED if trained with use_comm=True) ──
    # When use_comm=True, the actor is sized for comm_graph.decoder.
    # enhanced_dim (obs_dim + msg_dim), not raw env.OBS_DIM, and every
    # observation must be passed through comm_graph.forward(obs, rng)
    # BEFORE being given to the actor. Persisting only actor_params and
    # treating obs_dim as env.OBS_DIM (the previous version of this fix)
    # silently feeds the actor the wrong input dimensionality whenever
    # communication was used during training — exactly the kind of bug
    # this rewrite exists to eliminate, just one level deeper.
    use_comm: bool = False
    comm_graph_params: Optional[np.ndarray] = None
    comm_n_agents: Optional[int] = None
    comm_msg_dim: int = 8

    # ═════════════════════════════════════════════════════════════════════════
    #  FIGURE-SPECIFIC RESULTS (NEW)
    # ═════════════════════════════════════════════════════════════════════════

    # ── Fig 13: Communication Stress ──
    comm_stress_of: Optional[float] = None           # OF under stress
    comm_stress_cti_ok: Optional[bool] = None        # CTI satisfied under stress
    comm_stress_latency: Optional[float] = None        # Measured latency

    # ── Fig 14: N-1 Contingency ──
    n1_of: Optional[float] = None                    # OF with line outage
    n1_cti_ok: Optional[bool] = None                 # CTI with line outage
    n1_recovery_of: Optional[float] = None           # OF after recovery
    n1_line_index: Optional[int] = None              # Which line was outaged
    n1_outage_factor: Optional[float] = None         # Fault-current scaling applied

    # ── Fig 20: Byzantine Agents ──
    byzantine_of: Optional[float] = None             # OF with Byzantine agents
    byzantine_detected: Optional[bool] = None        # Detection flag
    byzantine_trust_scores: Optional[np.ndarray] = None  # Per-agent trust

    # ── Fig 22: Bandwidth vs Performance ──
    bandwidth_kbps: Optional[float] = None           # Actual bandwidth used
    bandwidth_of: Optional[float] = None             # OF at this config
    compression_params: Optional[int] = None         # Remaining parameters

    # ── Fig 23: Unseen Faults ──
    unseen_fault_of: Optional[float] = None          # OF on unseen fault
    unseen_fault_cti_ok: Optional[bool] = None       # CTI on unseen fault
    fault_impedance: Optional[float] = None          # Impedance tested
    unseen_fault_type: Optional[str] = None          # Actual fault type used for the eval

    # ── Fig 24: Sensor Robustness ──
    noisy_of: Optional[float] = None                 # OF with sensor noise
    noisy_cti_ok: Optional[bool] = None              # CTI with sensor noise
    sensor_snr: Optional[float] = None               # Signal-to-noise ratio

    # ── Fig 25: Zero-Shot Transfer ──
    transfer_of: Optional[float] = None              # OF on eval system (no fine-tuning)
    transfer_cti_ok: Optional[bool] = None           # CTI on eval system
    transfer_system: Optional[str] = None            # Which system transferred to

    # ── Fig 36: Action Space Geometry ──
    action_space_grid_data: Optional[Dict] = None    # {tds_values, psm_values, cti_grid, of_grid}

    # ── Fig 44: Fault Current Contribution ──
    fault_contributions: Optional[Dict] = None       # {source_name: contribution_pct}

    # ── Fig 45: Daily PV Response ──
    daily_tms_trajectory: Optional[np.ndarray] = None   # (24, n_relays) TDS over day
    daily_ps_trajectory: Optional[np.ndarray] = None    # (24, n_relays) PS over day
    daily_of_trajectory: Optional[np.ndarray] = None    # (24,) OF over day
    daily_irradiance: Optional[np.ndarray] = None       # (24,) irradiance profile

    # ── Fig 46: Relay Curves ──
    relay_curves: Optional[Dict] = None              # {relay_id: {If_range, t_op_range}}

    # ── Fig 47: Sympathetic Tripping ──
    sympathetic_trips: Optional[int] = None          # Count of unnecessary trips
    traditional_trips: Optional[int] = None          # Count with traditional settings

    # ── Fig 48: SHAP ──
    shap_values: Optional[np.ndarray] = None        # (n_features, n_actions)
    shap_base_value: Optional[float] = None           # Expected value

    # ── Fig 49: Decision Boundary ──
    tsne_observations: Optional[np.ndarray] = None  # (n_samples, obs_dim)
    tsne_actions: Optional[np.ndarray] = None       # (n_samples, act_dim)
    tsne_embeddings: Optional[np.ndarray] = None     # (n_samples, 2) 2D projection

    # ── Fig 50: Critical Pairs ──
    critical_pairs: Optional[List[Dict]] = None      # [{primary, backup, lambda, violation_rate}]

    # ── Fig 53: Compression ──
    compressed_of: Optional[float] = None            # OF after compression
    compressed_size_kb: Optional[float] = None       # Model size in KB
    compression_accuracy_drop: Optional[float] = None  # OF increase from baseline

    # ── Fig 54: Adversarial ──
    adversarial_of: Optional[float] = None           # OF under attack
    adversarial_success_rate: Optional[float] = None  # Attack success rate
    attack_perturbation_norm: Optional[float] = None # L2 norm of perturbation

    # ── Fig 58: Seasonal ──
    seasonal_of: Optional[float] = None            # OF under seasonal conditions
    seasonal_cti_ok: Optional[bool] = None         # CTI under seasonal conditions

    def to_dict(self) -> Dict:
        """Serialize to dict, converting numpy arrays to lists."""
        d = self.config.to_dict()
        d.update({
            "run_id": self.run_id,
            "best_of": float(self.best_of),
            "cti_satisfied": self.cti_satisfied,
            "cti_violations": self.cti_violations,
            "cti_min": float(self.cti_min),
            "elapsed_seconds": self.elapsed_seconds,
            "history": [float(h) for h in self.history],
            "comm_bandwidth": self.comm_bandwidth,
            "safe_rl_stats": self.safe_rl_stats,
            "ewc_stats": self.ewc_stats,
        })

        # Add all figure-specific results
        for key, val in self.__dict__.items():
            if key in ("config", "best_tms", "best_ps", "byzantine_trust_scores",
                       "daily_tms_trajectory", "daily_ps_trajectory", "daily_of_trajectory",
                       "daily_irradiance", "shap_values", "tsne_observations",
                       "tsne_actions", "tsne_embeddings"):
                if isinstance(val, np.ndarray):
                    d[key] = val.tolist()
                else:
                    d[key] = val
            elif key not in d and not key.startswith("_"):
                d[key] = val

        # Handle numpy arrays explicitly
        d["best_tms"] = self.best_tms.tolist() if isinstance(self.best_tms, np.ndarray) else self.best_tms
        d["best_ps"] = self.best_ps.tolist() if isinstance(self.best_ps, np.ndarray) else self.best_ps

        return d


# ═══════════════════════════════════════════════════════════════════════════════
#  PART 2: EXTENDED TRAINING PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

class TrainingPipeline:
    """
    Orchestrates large-scale training experiments.
    Extended to support all 17 stub figures with real data collection.
    """

    def __init__(
        self,
        configs: List[TrainingConfig],
        n_workers: int = -1,
        checkpoint_dir: str = "outputs/checkpoints",
        results_dir: str = "outputs/results",
        verbose: bool = True,
    ):
        self.configs = configs
        self.n_workers = n_workers if n_workers > 0 else max(1, cpu_count() - 1)
        self.checkpoint_dir = checkpoint_dir
        self.results_dir = results_dir
        self.verbose = verbose
        self.results: List[TrainingResult] = []
        self.completed_configs: set = set()

        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)

    # ── Public API ──

    def run_all(self) -> List[TrainingResult]:
        """Run all training configurations with figure-specific evaluations."""
        print(f"\n{'='*70}")
        print(f"  PHASE 5: Large-Scale Training Pipeline (Extended)")
        print(f"  Supports figures: 13, 14, 20, 22, 23, 24, 25, 36, 44, 45, 46, 47, 48, 49, 50, 53, 54, 58")
        print(f"{'='*70}")
        print(f"  Total configurations: {len(self.configs)}")
        print(f"  Parallel workers: {self.n_workers}")
        print(f"  Checkpoint dir: {self.checkpoint_dir}")
        print(f"  Results dir: {self.results_dir}")
        print(f"{'='*70}")

        self._load_checkpoint()

        pending = [c for c in self.configs if self._config_key(c) not in self.completed_configs]
        print(f"  Pending: {len(pending)}  |  Completed: {len(self.completed_configs)}")

        if not pending:
            print("  All configurations already completed!")
            return self.results

        if self.n_workers == 1:
            for i, config in enumerate(pending):
                print(f"\n  [{i+1}/{len(pending)}] {config.method} on {config.system} "
                      f"(PV={config.pv_level}, fault={config.fault_type})")
                result = self._run_single_config(config)
                self.results.append(result)
                self.completed_configs.add(self._config_key(config))
                self._save_checkpoint()
        else:
            with Pool(processes=self.n_workers) as pool:
                for i, result in enumerate(pool.imap_unordered(self._run_single_config, pending)):
                    self.results.append(result)
                    self.completed_configs.add(self._config_key(result.config))
                    self._save_checkpoint()
                    if self.verbose:
                        print(f"  Completed {i+1}/{len(pending)}: "
                              f"{result.config.method} on {result.config.system} "
                              f"→ OF={result.best_of:.4f}, CTI_ok={result.cti_satisfied}")

        print(f"\n  All {len(self.results)} runs completed!")
        return self.results

    def save_results(self, filepath: Optional[str] = None):
        """Save all results to JSON. Auto-generates filename if not provided."""
        if filepath is None:
            filepath = os.path.join(self.results_dir, "phase5_results.json")
        data = [r.to_dict() for r in self.results]
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"  Saved {len(self.results)} results to {filepath}")

    def save_results_by_figure(self):
        """Save separate JSON files organized by figure for easy plotting."""
        figure_fields = {
            "fig13_comm_stress": ["comm_stress_of", "comm_stress_cti_ok", "comm_stress_latency"],
            "fig14_n1_contingency": ["n1_of", "n1_cti_ok", "n1_recovery_of", "n1_line_index"],
            "fig20_byzantine": ["byzantine_of", "byzantine_detected", "byzantine_trust_scores"],
            "fig22_bandwidth": ["bandwidth_kbps", "bandwidth_of", "compression_params"],
            "fig23_unseen_faults": ["unseen_fault_of", "unseen_fault_cti_ok", "fault_impedance"],
            "fig24_sensor_robustness": ["noisy_of", "noisy_cti_ok", "sensor_snr"],
            "fig25_zero_shot": ["transfer_of", "transfer_cti_ok", "transfer_system"],
            "fig36_action_space": ["action_space_grid_data"],
            "fig44_fault_contrib": ["fault_contributions"],
            "fig45_daily_pv": ["daily_tms_trajectory", "daily_ps_trajectory", "daily_of_trajectory", "daily_irradiance"],
            "fig46_relay_curves": ["relay_curves"],
            "fig47_sympathetic": ["sympathetic_trips", "traditional_trips"],
            "fig48_shap": ["shap_values", "shap_base_value"],
            "fig49_tsne": ["tsne_observations", "tsne_actions", "tsne_embeddings"],
            "fig50_critical_pairs": ["critical_pairs"],
            "fig53_compression": ["compressed_of", "compressed_size_kb", "compression_accuracy_drop"],
            "fig54_adversarial": ["adversarial_of", "adversarial_success_rate", "attack_perturbation_norm"],
            "fig58_seasonal": ["seasonal_of", "seasonal_cti_ok"],
        }

        for fig_name, fields in figure_fields.items():
            fig_data = []
            for r in self.results:
                entry = {"config": r.config.to_dict(), "run_id": r.run_id}
                has_data = False
                for f in fields:
                    val = getattr(r, f, None)
                    if val is not None:
                        has_data = True
                        if isinstance(val, np.ndarray):
                            entry[f] = val.tolist()
                        else:
                            entry[f] = val
                if has_data:
                    fig_data.append(entry)

            if fig_data:
                path = os.path.join(self.results_dir, f"{fig_name}.json")
                with open(path, 'w') as f:
                    json.dump(fig_data, f, indent=2)
                print(f"  Saved {len(fig_data)} entries to {fig_name}.json")

    # ── Core execution ──

    def _run_single_config(self, config: TrainingConfig) -> TrainingResult:
        """Run a single training configuration with all figure evaluations."""
        t0 = time.perf_counter()

        from network.network_data import NETWORKS
        from network.relay_physics import check_constraints, objective_function

        # Get network
        net_fn = NETWORKS.get(config.system)
        if net_fn is None:
            raise ValueError(f"Unknown system: {config.system}")
        net = net_fn()

        # Apply PV penetration
        if config.pv_level > 0:
            net = net.apply_pv_penetration(config.pv_level)

        # Apply seasonal temperature if specified (Fig 58)
        if config.seasonal_temperature is not None:
            net = self._apply_seasonal_temperature(net, config.seasonal_temperature)

        # Apply topology changes (e.g. run_topology_robustness's N-1 outage
        # configs) BEFORE training/dispatch, not just as the separate
        # post-hoc Fig-14 evaluation below. config.topology_changes used to
        # be parsed onto TrainingConfig and never read anywhere -- baseline
        # (topology_changes={}) and "with outage" configs trained on the
        # identical, unperturbed network, so ResultsAggregator.get_topology_robustness()
        # always saw ~0% degradation regardless of the real N-1 impact.
        # Reuses the same NetworkData.apply_n1_outage() physics as
        # _eval_n1_contingency below.
        if config.topology_changes and config.topology_changes.get("type") == "n1_outage":
            net, _outage_factor = net.apply_n1_outage(config.topology_changes.get("line_idx", 0))

        # Run base method
        result = self._dispatch_method(config, net)

        # ═════════════════════════════════════════════════════════════════════
        #  FIGURE-SPECIFIC EVALUATIONS (NEW)
        # ═════════════════════════════════════════════════════════════════════

        # Fig 13: Communication Stress
        if config.comm_stress_level > 0 or config.packet_loss_rate > 0:
            result = self._eval_comm_stress(config, net, result)

        # Fig 14: N-1 Contingency
        if config.n1_line_outage is not None:
            result = self._eval_n1_contingency(config, net, result)

        # Fig 20: Byzantine Agents
        if len(config.byzantine_agents) > 0:
            result = self._eval_byzantine(config, net, result)

        # Fig 22: Bandwidth vs Performance
        if config.compression_eval or config.gnn_depth != 2 or config.compression_ratio != 1.0:
            result = self._eval_bandwidth(config, net, result)

        # Fig 23: Unseen Faults
        if config.unseen_fault_eval or config.fault_impedance_ohm > 0:
            result = self._eval_unseen_faults(config, net, result)

        # Fig 24: Sensor Robustness
        if config.sensor_noise_std > 0 or config.missing_sensor_rate > 0 or config.sensor_bias != 0:
            result = self._eval_sensor_robustness(config, net, result)

        # Fig 25: Zero-Shot Transfer
        if config.eval_system is not None and config.eval_system != config.system:
            result = self._eval_zero_shot_transfer(config, net, result)

        # Fig 36: Action Space Geometry
        if config.action_space_grid:
            result = self._eval_action_space_grid(config, net, result)

        # Fig 44: Fault Current Contribution
        if config.run_shortcircuit_study:
            result = self._eval_fault_contributions(config, net, result)

        # Fig 45: Daily PV Response
        if config.daily_profile:
            result = self._eval_daily_pv(config, net, result)

        # Fig 46: Relay Operating Curves
        if config.relay_curve_eval:
            result = self._eval_relay_curves(config, net, result)

        # Fig 47: Sympathetic Tripping
        if config.sympathetic_tripping_eval:
            result = self._eval_sympathetic_tripping(config, net, result)

        # Fig 48: SHAP
        if config.shap_eval:
            result = self._eval_shap(config, net, result)

        # Fig 49: Decision Boundary (t-SNE)
        if config.tsne_eval:
            result = self._eval_tsne(config, net, result)

        # Fig 50: Critical Pairs
        if config.critical_pair_eval:
            result = self._eval_critical_pairs(config, net, result)

        # Fig 53: Model Compression
        if config.compression_eval or config.quantize_bits is not None or config.prune_ratio > 0:
            result = self._eval_compression(config, net, result)

        # Fig 54: Adversarial Robustness
        if config.adversarial_eval or config.adversarial_attack is not None:
            result = self._eval_adversarial(config, net, result)

        # Fig 58: Seasonal Generalisation
        if config.season is not None:
            result = self._eval_seasonal(config, net, result)

        result.elapsed_seconds = time.perf_counter() - t0
        return result

    # ── Method dispatch ──

    def _dispatch_method(self, config: TrainingConfig, net) -> TrainingResult:
        """Dispatch to appropriate method runner."""
        method = config.method

        if method == "GA-SQP":
            return self._run_ga_sqp(config, net)
        elif method in ["HWOA", "HHO", "JAYA", "EEO"]:
            return self._run_metaheuristic(config, net)
        elif method in ["IPPO", "MAPPO", "MAPPO-Comm", "MAPPO-Safe", "MAPPO-Safe-EWC"]:
            return self._run_marl(config, net)
        elif method == "Single-PPO":
            return self._run_single_ppo(config, net)
        else:
            raise ValueError(f"Unknown method: {method}")

    # ── Base method runners (unchanged from original) ──

    def _run_ga_sqp(self, config: TrainingConfig, net) -> TrainingResult:
        from baselines.ga_sqp import run_ga_sqp
        res = run_ga_sqp(
            net, ga_pop=config.ga_pop, ga_gen=config.ga_gen,
            seed=config.seed_offset, verbose=False,
            fault_type=config.fault_type, pv_level=config.pv_level,
        )
        constraints = check_constraints(net, res["TMS"], res["PS"], fault_type=config.fault_type)
        return TrainingResult(
            config=config, run_id=0, best_of=res["OF"],
            best_tms=res["TMS"], best_ps=res["PS"],
            cti_satisfied=constraints["cti_satisfied"],
            cti_violations=constraints["cti_violations"],
            cti_min=constraints["cti_min"],
            elapsed_seconds=0.0,
            history=res.get("history", [res["OF"]]),
        )

    def _run_metaheuristic(self, config: TrainingConfig, net) -> TrainingResult:
        from baselines.metaheuristics import run_hwoa, run_hho, run_jaya, run_eeo
        runners = {"HWOA": run_hwoa, "HHO": run_hho, "JAYA": run_jaya, "EEO": run_eeo}
        runner = runners[config.method]
        kw = {"seed": config.seed_offset, "verbose": False,
              "fault_type": config.fault_type, "pv_level": config.pv_level, "use_sqp": True}
        if config.method == "HWOA":
            kw.update({"n_whales": config.meta_pop, "max_iter": config.meta_iter})
        elif config.method == "HHO":
            kw.update({"n_hawks": config.meta_pop, "max_iter": config.meta_iter})
        elif config.method == "JAYA":
            kw.update({"n_pop": config.meta_pop, "max_iter": config.meta_iter})
        elif config.method == "EEO":
            kw.update({"n_particles": config.meta_pop, "max_iter": config.meta_iter})
        res = runner(net, **kw)
        constraints = check_constraints(net, res["TMS"], res["PS"], fault_type=config.fault_type)
        return TrainingResult(
            config=config, run_id=0, best_of=res["OF"],
            best_tms=res["TMS"], best_ps=res["PS"],
            cti_satisfied=constraints["cti_satisfied"],
            cti_violations=constraints["cti_violations"],
            cti_min=constraints["cti_min"],
            elapsed_seconds=0.0,
            history=res.get("history", [res["OF"]]),
        )

    def _run_marl(self, config: TrainingConfig, net) -> TrainingResult:
        from marl.marl_env import DynamicDOCREnv
        from marl.marl_agents import MAPPOMARLTrainer
        from marl.safe_rl import create_safe_rl_env, SafeRLConfig
        from marl.continual_trainer import create_ewc_trainer, EWCConfig

        env = DynamicDOCREnv(
            net, action_scale=0.05, max_steps=300,
            seed=config.seed_offset, use_pandapower=False,
            fault_type=config.fault_type,
        )
        if config.use_curriculum:
            env.set_curriculum_stage(config.curriculum_stage)

        # Sensor noise/missing-sensor robustness (Fig 24) is tested at
        # EVAL time only (_eval_sensor_robustness corrupts observations
        # explicitly there — see its docstring: does a policy trained
        # under CLEAN conditions degrade gracefully when its inputs are
        # corrupted, not whether training itself was noisy). This used
        # to set env.sensor_noise_std/env.missing_sensor_rate here, but
        # DynamicDOCREnv never defines or reads either attribute — a
        # dead assignment left over from an earlier design, removed.

        # Safe RL wrapper
        if config.use_safe_rl or "Safe" in config.method:
            safe_cfg = SafeRLConfig(
                project_actions=True, enforce_cti=True,
                lambda_init=1.0, lambda_lr=0.01, rho_penalty=10.0,
                use_pid=True, adaptive_rho=True,
            )
            env = create_safe_rl_env(env, safe_cfg)

        trainer = MAPPOMARLTrainer(
            env, use_comm=config.use_comm or "Comm" in config.method,
            hidden_actor=64, hidden_critic=128,
            lr_actor=0.02, lr_critic=0.01,
            clip_eps=0.2, gamma=0.99, lam=0.95,
            rollout_len=64, n_epochs=4,
            seed=config.seed_offset, curriculum=config.use_curriculum,
        )

        if config.use_ewc or "EWC" in config.method:
            ewc_cfg = EWCConfig(
                lambda_ewc=100.0, fisher_samples=100,
                use_curriculum_warmstart=True, max_tasks=6,
            )
            trainer = create_ewc_trainer(trainer, ewc_cfg)

        result = trainer.train(total_steps=config.marl_steps, verbose=False)
        constraints = check_constraints(net, result["TMS"], result["PS"], fault_type=config.fault_type)

        return TrainingResult(
            config=config, run_id=0, best_of=result["OF"],
            best_tms=result["TMS"], best_ps=result["PS"],
            cti_satisfied=constraints["cti_satisfied"],
            cti_violations=constraints["cti_violations"],
            cti_min=constraints["cti_min"],
            elapsed_seconds=result.get("elapsed", 0.0),
            history=result.get("history", [result["OF"]]),
            comm_bandwidth=result.get("comm_bandwidth", 0.0),
            safe_rl_stats=result.get("safe_rl_stats"),
            ewc_stats=result.get("ewc_stats"),
        )

    def _run_single_ppo(self, config: TrainingConfig, net) -> TrainingResult:
        from marl.single_agent_ppo import run_single_ppo
        res = run_single_ppo(
            net, total_steps=config.marl_steps,
            seed=config.seed_offset, verbose=False,
            use_pandapower=False, fault_type=config.fault_type,
        )
        constraints = check_constraints(net, res["TMS"], res["PS"], fault_type=config.fault_type)
        return TrainingResult(
            config=config, run_id=0, best_of=res["OF"],
            best_tms=res["TMS"], best_ps=res["PS"],
            cti_satisfied=constraints["cti_satisfied"],
            cti_violations=constraints["cti_violations"],
            cti_min=constraints["cti_min"],
            elapsed_seconds=res.get("elapsed", 0.0),
            history=res.get("history", [res["OF"]]),
        )

    # ── Checkpoint helpers ──

    def _config_key(self, config: TrainingConfig) -> str:
        """Generate unique key for a configuration including figure-specific fields."""
        parts = [
            config.method, config.system,
            f"PV{config.pv_level}", config.fault_type,
            f"r{config.seed_offset}",
        ]
        # Append figure-specific identifiers
        if config.comm_stress_level > 0:
            parts.append(f"stress{config.comm_stress_level}")
        if config.n1_line_outage is not None:
            parts.append(f"N1L{config.n1_line_outage}")
        if config.byzantine_agents:
            parts.append(f"byz{''.join(map(str, config.byzantine_agents))}")
        if config.gnn_depth != 2:
            parts.append(f"gnn{config.gnn_depth}")
        if config.compression_ratio != 1.0:
            parts.append(f"comp{config.compression_ratio}")
        if config.fault_impedance_ohm > 0:
            parts.append(f"Zf{config.fault_impedance_ohm}")
        if config.sensor_noise_std > 0:
            parts.append(f"noise{config.sensor_noise_std}")
        if config.eval_system:
            parts.append(f"transfer{config.eval_system}")
        if config.season:
            parts.append(config.season)
        return "_".join(parts)

    def _save_checkpoint(self):
        checkpoint = {
            "completed_configs": list(self.completed_configs),
            "n_results": len(self.results),
        }
        path = os.path.join(self.checkpoint_dir, "pipeline_checkpoint.json")
        with open(path, 'w') as f:
            json.dump(checkpoint, f)

    def _load_checkpoint(self):
        path = os.path.join(self.checkpoint_dir, "pipeline_checkpoint.json")
        if os.path.exists(path):
            with open(path, 'r') as f:
                checkpoint = json.load(f)
            self.completed_configs = set(checkpoint.get("completed_configs", []))
            print(f"  Loaded checkpoint: {len(self.completed_configs)} completed")

    def _apply_seasonal_temperature(self, net, temperature: float):
        """
        Override fault current capacity based on seasonal temperature.

        Previous version checked `hasattr(net, 'lines')`, which is
        always False — NetworkData has no .lines table at all (topology
        lives in .pairs and .line_data, confirmed in
        network/network_data.py), so this was a complete no-op:
        net_seasonal was always literally the same object as net, and
        "seasonal" evaluation always silently measured the exact same
        conditions regardless of which season was requested.

        Fixed version applies a standard linear thermal derating
        (consistent with IEC 60287 conductor resistance temperature
        scaling, R_T = R_20 * (1 + alpha*(T-20)) -- higher temperature
        means higher line resistance, which marginally increases
        source impedance and reduces available fault current) directly
        to the fault currents stored on each RelayPair, which IS what
        relay_physics.py's objective_function/check_constraints
        actually consume.
        """
        ALPHA_CU = 0.00393  # 1/degC, copper conductor temperature coefficient
        derating = 1.0 / (1.0 + ALPHA_CU * (temperature - 20.0))

        net_seasonal = net.copy()
        for pair in net_seasonal.pairs:
            pair.If_pri *= derating
            pair.If_bkp *= derating
        return net_seasonal


    # ═════════════════════════════════════════════════════════════════════════
    #  PART 3: FIGURE-SPECIFIC EVALUATION METHODS (Figures 13-25)
    # ═════════════════════════════════════════════════════════════════════════

    # ── Fig 13: Communication Stress ─────────────────────────────────────────

    def _get_actor_for_eval(self, env, result: TrainingResult, net=None):
        """
        Reconstruct a real, callable policy from the trained parameters
        persisted on `result`, returning (act_fn, raw_obs_dim) where
        act_fn(obs_i, rng) -> (action, log_prob) has the SAME signature
        as SharedGaussianActor.sample, but transparently applies GNN
        communication enhancement first if the policy was trained with
        use_comm=True.

        This closes a second, deeper instance of the same class of bug
        already fixed once in this file: simply restoring actor_params
        and calling actor.sample(raw_obs_i, rng) is WRONG whenever
        use_comm=True, because the actor was trained on enhanced_dim
        (raw_obs_dim + msg_dim) inputs produced by CommGraph.forward(),
        not on raw env observations. Skipping that step would make the
        actor receive an input of the wrong dimensionality (and the
        wrong semantic content) at evaluation time — see
        marl/marl_agents.py: MAPPOMARLTrainer._collect_rollout(), which
        always calls comm_graph.forward(obs, rng) before actor.sample()
        whenever self.use_comm is True.

        Every _eval_* method below MUST go through this rather than
        sampling np.random.normal(0, 0.1, ...) directly, AND must call
        the returned act_fn (not a raw SharedGaussianActor) so
        communication-enhanced policies are evaluated correctly.
        """
        from marl.marl_agents import SharedGaussianActor

        if result.actor_params is None:
            raise RuntimeError(
                "TrainingResult has no actor_params — this evaluation "
                "would otherwise fall back to random actions, which is "
                "not a valid test of policy robustness. Train a base "
                "policy with CoreEvalBridge._train_base_policy() (which "
                "now calls trainer.train() and persists weights) before "
                "running this evaluation."
            )

        raw_obs_dim = result.obs_dim or getattr(env, "OBS_DIM", 10)
        act_dim = result.act_dim or getattr(env, "ACT_DIM", 2)
        hidden = result.actor_hidden or 64

        comm_graph = None
        if result.use_comm:
            if result.comm_graph_params is None:
                raise RuntimeError(
                    "TrainingResult.use_comm=True but comm_graph_params "
                    "is missing — cannot reconstruct the GNN communication "
                    "module the actor was actually trained with. The actor "
                    "expects enhanced (obs + message) inputs and will "
                    "silently misbehave if fed raw observations instead."
                )
            from marl.comm_graph import CommGraph, build_relay_adjacency
            if net is None:
                net = getattr(env, "net", None)
            if net is None:
                raise RuntimeError(
                    "Cannot rebuild CommGraph adjacency without access to "
                    "the NetworkData instance (net=None and env.net missing)."
                )
            adj = build_relay_adjacency(net)
            comm_graph = CommGraph(
                n_agents=result.comm_n_agents or env.n_agents,
                obs_dim=raw_obs_dim,
                enhanced_dim=raw_obs_dim + result.comm_msg_dim,
                adjacency=adj,
                seed=0,
            )
            comm_graph.set_all_params(result.comm_graph_params)
            actor_obs_dim = raw_obs_dim + result.comm_msg_dim
        else:
            actor_obs_dim = raw_obs_dim

        actor = SharedGaussianActor(actor_obs_dim, act_dim, hidden=hidden, seed=0)
        actor.set_all_params(result.actor_params)

        if comm_graph is None:
            def act_fn(all_obs, agent_idx, rng):
                return actor.sample(all_obs[agent_idx], rng)

            def mean_fn(obs_i):
                return actor.get_mean(obs_i)
        else:
            def act_fn(all_obs, agent_idx, rng):
                enhanced_obs, _messages, _info = comm_graph.forward(all_obs, rng)
                return actor.sample(enhanced_obs[agent_idx], rng)

            def mean_fn(obs_i):
                # For single-observation callers (e.g. SHAP/t-SNE on a
                # per-agent obs slice) we cannot recompute the full
                # multi-agent attention pass, so we approximate the
                # enhanced input with a zero message vector appended —
                # equivalent to "no neighbor information available",
                # which is the best honest approximation without the
                # full batch of all agents' observations at this call
                # site. Callers needing the exact enhanced observation
                # for a full env step should use act_fn instead.
                padded = np.concatenate([obs_i, np.zeros(result.comm_msg_dim)])
                return actor.get_mean(padded)

        return act_fn, mean_fn, comm_graph, actor

    def _eval_comm_stress(self, config: TrainingConfig, net, result: TrainingResult) -> TrainingResult:
        """
        Evaluate policy under communication stress: packet loss, latency, quantization.

        Previous version checked `hasattr(env, 'comm_graph')`, but
        DynamicDOCREnv (marl/marl_env.py) never defines a `comm_graph`
        attribute at all — that object only exists on MAPPOMARLTrainer.
        So this entire method's body silently never executed; the
        condition was always False, and `result` was returned completely
        unmodified (no comm_stress_of, no comm_stress_cti_ok). This
        means every prior Fig 13 datapoint was either None or a
        synthetic fallback value, regardless of the (also-buggy) random
        action sampling inside the dead branch.

        Fixed version reconstructs the ACTUAL trained CommGraph via
        _get_actor_for_eval (raises clearly if the policy was not
        trained with use_comm=True, since "communication stress" is
        meaningless for a policy that never used communication), then
        genuinely perturbs that object's packet_loss_rate / latency /
        quantization before driving a real rollout through it.
        """
        from marl.marl_env import DynamicDOCREnv
        from network.relay_physics import objective_function

        env = DynamicDOCREnv(net, action_scale=0.05, max_steps=300,
                             seed=config.seed_offset + 999,
                             fault_type=config.fault_type)

        if not result.use_comm:
            warnings.warn(
                "Fig 13 (communication stress) requires a policy trained "
                "with use_comm=True; this TrainingResult has use_comm=False, "
                "so there is no communication channel to stress. Skipping."
            )
            return result

        act_fn, mean_fn, comm_graph, actor = self._get_actor_for_eval(env, result, net=net)

        # Apply the requested stress conditions to the REAL trained
        # comm graph, not a phantom env.comm_graph that never existed.
        if config.packet_loss_rate > 0 and hasattr(comm_graph, 'packet_loss_rate'):
            comm_graph.packet_loss_rate = config.packet_loss_rate
        if config.latency_ms > 0 and hasattr(comm_graph, 'latency_steps'):
            comm_graph.latency_steps = max(1, int(config.latency_ms / 100))
        if config.quantization_bits < 32 and hasattr(comm_graph, 'quantization_bits'):
            comm_graph.quantization_bits = config.quantization_bits

        rng = np.random.default_rng(config.seed_offset + 999)
        obs = env.reset()
        done = False
        total_reward = 0
        steps = 0
        while not done and steps < 300:
            actions = np.zeros((env.n_agents, env.ACT_DIM))
            for i in range(env.n_agents):
                a, _ = act_fn(obs, i, rng)
                actions[i] = a
            obs, reward, done, _ = env.step(actions)
            total_reward += reward.mean()
            steps += 1

        TMS, PS = env.get_state()
        try:
            of_stress, _, _ = objective_function(net, TMS, PS, fault_type=config.fault_type)
            constraints = check_constraints(net, TMS, PS, fault_type=config.fault_type)
            result.comm_stress_of = float(of_stress)
            result.comm_stress_cti_ok = constraints["cti_satisfied"]
            result.comm_stress_latency = config.latency_ms
        except Exception as e:
            warnings.warn(f"Comm stress eval failed: {e}")

        return result

    # ── Fig 14: N-1 Contingency ──────────────────────────────────────────────

    def _eval_n1_contingency(self, config: TrainingConfig, net, result: TrainingResult) -> TrainingResult:
        """
        Evaluate policy with one line genuinely outaged.

        Previous version called `net.lines.at[line_idx, 'in_service'] =
        False`, but NetworkData has no `.lines` table — that line silently
        no-op'd (caught by the bare except), so every "N-1" result was
        actually the BASE-CASE objective with no topology change at all.
        It also never let the policy act post-outage; it read off the
        objective from a static (TMS, PS) snapshot taken before any
        interaction with the new topology.

        Fixed version:
          1. Genuinely mutates fault currents via NetworkData.apply_n1_outage
             (reuses the same TopologyManager outage-factor physics as the
             Fig 15 scalability sweep).
          2. Lets the trained policy actually step through the post-outage
             environment to reach its own settling point, rather than
             reusing the pre-outage (TMS, PS).
          3. If n1_recovery_eval is requested, runs a genuine short
             additional rollout on the outaged network so "recovery" means
             the policy adapting online, not re-evaluating the same point.
        """
        from marl.marl_env import DynamicDOCREnv
        from network.relay_physics import objective_function

        line_idx = config.n1_line_outage
        if line_idx is None:
            line_idx = 0

        try:
            net_n1, outage_factor = net.apply_n1_outage(line_idx)

            env_n1 = DynamicDOCREnv(
                net_n1, action_scale=0.05, max_steps=300,
                seed=config.seed_offset + 500,
                fault_type=config.fault_type,
            )
            act_fn, mean_fn, comm_graph, actor = self._get_actor_for_eval(env_n1, result, net=net_n1)
            rng = np.random.default_rng(config.seed_offset + 500)

            # Seed the policy from its pre-outage settings, then let it
            # actually act under the new topology rather than freezing at
            # the pre-outage operating point. reset() unconditionally
            # re-randomizes TMS/PS, so set_state() must come AFTER it
            # (setting the attributes directly before reset(), as a
            # previous version of this method did, gets silently
            # discarded) — and obs must be recomputed post-set_state()
            # since the obs reset() returned reflects the now-stale
            # randomized TMS/PS, not the just-restored best_tms/best_ps.
            obs = env_n1.reset(seed=config.seed_offset + 500)
            env_n1.set_state(result.best_tms.copy(), result.best_ps.copy())
            obs = env_n1._get_obs()
            done = False
            steps = 0
            while not done and steps < env_n1.max_steps:
                actions = np.zeros((env_n1.n_agents, env_n1.ACT_DIM))
                for i in range(env_n1.n_agents):
                    a, _ = act_fn(obs, i, rng)
                    actions[i] = a
                obs, _, done, _ = env_n1.step(actions)
                steps += 1

            TMS, PS = env_n1.get_state()
            of_n1, _, _ = objective_function(net_n1, TMS, PS, fault_type=config.fault_type)
            constraints_n1 = check_constraints(net_n1, TMS, PS, fault_type=config.fault_type)

            result.n1_of = float(of_n1)
            result.n1_cti_ok = constraints_n1["cti_satisfied"]
            result.n1_line_index = line_idx
            result.n1_outage_factor = outage_factor

            if config.n1_recovery_eval:
                # Genuine recovery: continue rolling out on the SAME
                # outaged network/env (not a fresh copy) for additional
                # steps so the policy has more interaction to settle
                # into a better post-contingency operating point.
                # reset() is needed to clear step_count/done, but it
                # unconditionally re-randomizes TMS/PS too — restore the
                # settled (TMS, PS) captured just above right after, or
                # "recovery" would silently restart from a fresh random
                # point instead of continuing from where the first
                # rollout left off.
                recovery_steps = 100
                obs = env_n1.reset(seed=config.seed_offset + 600)
                env_n1.set_state(TMS, PS)
                obs = env_n1._get_obs()
                done = False
                steps = 0
                while not done and steps < recovery_steps:
                    actions = np.zeros((env_n1.n_agents, env_n1.ACT_DIM))
                    for i in range(env_n1.n_agents):
                        a, _ = act_fn(obs, i, rng)
                        actions[i] = a
                    obs, _, done, _ = env_n1.step(actions)
                    steps += 1

                TMS_rec, PS_rec = env_n1.get_state()
                of_rec, _, _ = objective_function(net_n1, TMS_rec, PS_rec, fault_type=config.fault_type)
                result.n1_recovery_of = float(of_rec)

        except Exception as e:
            warnings.warn(f"N-1 eval failed for line {line_idx}: {e}")

        return result

    # ── Fig 20: Byzantine Agents ─────────────────────────────────────────────

    def _eval_byzantine(self, config: TrainingConfig, net, result: TrainingResult) -> TrainingResult:
        """
        Evaluate with malicious agents. Detect and measure impact.
        """
        from marl.marl_env import DynamicDOCREnv
        from network.relay_physics import objective_function

        env = DynamicDOCREnv(net, action_scale=0.05, max_steps=300,
                             seed=config.seed_offset + 700,
                             fault_type=config.fault_type)

        act_fn, mean_fn, comm_graph, actor = self._get_actor_for_eval(env, result, net=net)
        rng = np.random.default_rng(config.seed_offset + 700)

        obs = env.reset()
        trust_scores = np.ones(env.n_agents)

        done = False
        steps = 0
        while not done and steps < 300:
            actions = np.zeros((env.n_agents, env.ACT_DIM))

            # Honest agents act first so "reverse" Byzantine agents have
            # a real honest action to invert, rather than inverting zeros
            # (the previous bug: actions[i] = -actions[i] before any
            # honest action had been written into the array).
            for i in range(env.n_agents):
                if i not in config.byzantine_agents:
                    a, _ = act_fn(obs, i, rng)
                    actions[i] = a

            for i in config.byzantine_agents:
                if config.byzantine_strategy == "random":
                    actions[i] = rng.uniform(-1, 1, env.ACT_DIM)
                elif config.byzantine_strategy == "reverse":
                    honest_a, _ = act_fn(obs, i, rng)
                    actions[i] = -honest_a
                elif config.byzantine_strategy in ("colluding", "collusion"):
                    actions[i] = np.ones(env.ACT_DIM) * 0.9
                elif config.byzantine_strategy == "fdi":
                    # False Data Injection: a compromised agent reports a
                    # small, deliberately-wrong but plausible-looking
                    # action (a fixed, modest bias) rather than either
                    # colluding agents' extreme saturated push (0.9) or
                    # pure random noise — the point of FDI is to evade
                    # detection by staying inside the normal operating
                    # range while still being systematically wrong.
                    actions[i] = np.full(env.ACT_DIM, 0.3)
                else:
                    raise ValueError(
                        f"Unknown byzantine_strategy {config.byzantine_strategy!r}; "
                        f"expected one of 'random', 'reverse', 'colluding'/'collusion', 'fdi'."
                    )

            obs, reward, done, info = env.step(actions)

            if 'per_agent_reward' in info:
                for i in range(env.n_agents):
                    if info['per_agent_reward'][i] < -0.5:
                        trust_scores[i] *= 0.9

            steps += 1

        TMS, PS = env.get_state()
        try:
            of_byz, _, _ = objective_function(net, TMS, PS, fault_type=config.fault_type)
            constraints = check_constraints(net, TMS, PS, fault_type=config.fault_type)
            result.byzantine_of = float(of_byz)
            result.byzantine_trust_scores = trust_scores
            result.byzantine_detected = np.any(trust_scores[config.byzantine_agents] < 0.5)
        except Exception as e:
            warnings.warn(f"Byzantine eval failed: {e}")

        return result

    # ── Fig 22: Bandwidth vs Performance ────────────────────────────────────

    def _eval_bandwidth(self, config: TrainingConfig, net, result: TrainingResult) -> TrainingResult:
        """
        Evaluate bandwidth usage and OF at different GNN depths/compression levels.

        Previous version checked `hasattr(env, 'comm_graph')`, which is
        always False (DynamicDOCREnv never defines this attribute — see
        the identical bug already fixed in _eval_comm_stress), so this
        method's body never executed and Fig 22 always returned an
        unmodified result. Even granting that the condition somehow
        passed, `result.bandwidth_of = result.best_of` hardcoded the
        objective to be identical regardless of gnn_depth or
        compression_ratio — meaning compression would always have
        appeared completely free of any performance cost, which is not
        a credible claim for a genuine bandwidth/performance trade-off
        figure.

        Fixed version reconstructs the real trained comm graph, applies
        compression as message quantization/sparsification (reducing
        the effective precision of the GNN message vector the policy
        actually receives, proportional to compression_ratio), and
        measures the resulting OF from a real rollout — so the
        trade-off curve reflects an actual performance cost.
        """
        from marl.marl_env import DynamicDOCREnv
        from network.relay_physics import objective_function

        env = DynamicDOCREnv(net, action_scale=0.05, max_steps=300,
                             seed=config.seed_offset + 800,
                             fault_type=config.fault_type)

        if not result.use_comm:
            warnings.warn(
                "Fig 22 (bandwidth trade-off) requires a policy trained "
                "with use_comm=True; this TrainingResult has use_comm=False. "
                "Skipping — there is no communication channel to compress."
            )
            return result

        act_fn, mean_fn, comm_graph, actor = self._get_actor_for_eval(env, result, net=net)
        rng = np.random.default_rng(config.seed_offset + 800)

        # Raw (uncompressed) bandwidth: msg_dim floats/agent/step at 4 bytes
        # each (float32), reduced by compression_ratio via lossy
        # quantization below.
        msg_dim = result.comm_msg_dim
        bytes_per_step = env.n_agents * msg_dim * 4 * config.compression_ratio
        kbps = bytes_per_step * 8 / 1000.0

        total_params = comm_graph.n_params() if hasattr(comm_graph, 'n_params') else 0
        compressed_params = int(total_params * config.compression_ratio)

        # Genuinely degrade the message the policy receives: simulate
        # quantization/sparsification by zeroing out a fraction of each
        # message vector's components proportional to how aggressively
        # it's being compressed (compression_ratio=1.0 -> no loss;
        # compression_ratio=0.1 -> only 10% of message content survives).
        original_forward = comm_graph.forward
        drop_fraction = max(0.0, 1.0 - config.compression_ratio)

        def degraded_forward(obs, rng_):
            enhanced_obs, messages, info = original_forward(obs, rng_)
            if drop_fraction > 0:
                mask = rng_.random(enhanced_obs.shape) < drop_fraction
                enhanced_obs = enhanced_obs.copy()
                enhanced_obs[mask] = 0.0
            return enhanced_obs, messages, info

        comm_graph.forward = degraded_forward

        obs = env.reset()
        done = False
        steps = 0
        while not done and steps < 300:
            actions = np.zeros((env.n_agents, env.ACT_DIM))
            for i in range(env.n_agents):
                a, _ = act_fn(obs, i, rng)
                actions[i] = a
            obs, _, done, _ = env.step(actions)
            steps += 1

        comm_graph.forward = original_forward  # restore, in case comm_graph is reused

        TMS, PS = env.get_state()
        try:
            of_compressed, _, _ = objective_function(net, TMS, PS, fault_type=config.fault_type)
            result.bandwidth_kbps = float(kbps)
            result.compression_params = compressed_params
            result.bandwidth_of = float(of_compressed)
        except Exception as e:
            warnings.warn(f"Bandwidth eval failed: {e}")

        return result

    # ── Fig 23: Unseen Fault Generalisation ──────────────────────────────────

    def _eval_unseen_faults(self, config: TrainingConfig, net, result: TrainingResult) -> TrainingResult:
        """
        Evaluate on faults with impedance or evolving characteristics not seen in training.

        Previous version checked `hasattr(env, 'fault_impedance')`,
        which DynamicDOCREnv never defines (it only has `self.fault_type`,
        which the real env DOES consume in step()/compute_operating_times
        via FAULT_TYPE_WEIGHTS scaling), so the requested fault severity
        never reached the simulation. It also never let the policy act —
        TMS/PS were pasted directly from the training-time best values
        and the objective was read off immediately, which tests whether
        a frozen operating point happens to be acceptable under a
        different fault scaling, not whether the POLICY can react to an
        unfamiliar fault. Finally, `objective_function(net, TMS, PS)`
        omitted fault_type, so it silently scored everything as if it
        were a plain 3ph fault regardless of what was configured.
        """
        from marl.marl_env import DynamicDOCREnv
        from network.relay_physics import objective_function

        # config.fault_impedance_ohm doesn't correspond to anything the
        # physics model (relay_physics.py) consumes directly — the
        # nearest real generalisation axis the environment supports is
        # fault TYPE (3ph/LL/1ph/2ph-g), which genuinely changes fault
        # current magnitude via FAULT_TYPE_WEIGHTS. We use
        # config.fault_type as the actual unseen condition, and record
        # the requested impedance value for bookkeeping/plotting only.
        eval_fault_type = config.unseen_fault_type or config.fault_type

        env = DynamicDOCREnv(net, action_scale=0.05, max_steps=300,
                             seed=config.seed_offset + 900,
                             fault_type=eval_fault_type)

        act_fn, mean_fn, comm_graph, actor = self._get_actor_for_eval(env, result, net=net)
        rng = np.random.default_rng(config.seed_offset + 900)

        # reset() unconditionally re-randomizes TMS/PS, so set_state()
        # must come AFTER it (setting the attributes directly before
        # reset(), as a previous version of this method did, gets
        # silently discarded) — and obs must be recomputed post-
        # set_state() since the obs reset() returned reflects the
        # now-stale randomized TMS/PS, not the just-restored
        # best_tms/best_ps.
        obs = env.reset(seed=config.seed_offset + 900, fault_type=eval_fault_type)
        env.set_state(result.best_tms.copy(), result.best_ps.copy())
        obs = env._get_obs()
        done = False
        steps = 0
        while not done and steps < env.max_steps:
            actions = np.zeros((env.n_agents, env.ACT_DIM))
            for i in range(env.n_agents):
                a, _ = act_fn(obs, i, rng)
                actions[i] = a
            obs, _, done, _ = env.step(actions)
            steps += 1

        try:
            TMS, PS = env.get_state()
            of_unseen, _, _ = objective_function(net, TMS, PS, fault_type=eval_fault_type)
            constraints = check_constraints(net, TMS, PS, fault_type=eval_fault_type)
            result.unseen_fault_of = float(of_unseen)
            result.unseen_fault_cti_ok = constraints["cti_satisfied"]
            result.fault_impedance = config.fault_impedance_ohm
            result.unseen_fault_type = eval_fault_type
        except Exception as e:
            warnings.warn(f"Unseen fault eval failed: {e}")

        return result

    # ── Fig 24: Sensor Robustness ────────────────────────────────────────────

    def _eval_sensor_robustness(self, config: TrainingConfig, net, result: TrainingResult) -> TrainingResult:
        """
        Evaluate under sensor noise and missing data.
        """
        from marl.marl_env import DynamicDOCREnv
        from network.relay_physics import objective_function

        env = DynamicDOCREnv(net, action_scale=0.05, max_steps=300,
                             seed=config.seed_offset + 1000,
                             fault_type=config.fault_type)

        # NOTE: DynamicDOCREnv only consumes a boolean `sensor_noise` flag
        # internally (see environment/dynamic_network.py); it has no
        # consumer for sensor_noise_std or missing_sensor_rate, so setting
        # those attributes directly on the env was a dead assignment that
        # never reached the observation the policy actually sees. We
        # instead corrupt the observation explicitly, here, before it's
        # passed to the policy — this is what "sensor robustness" has to
        # mean: does the TRAINED policy degrade gracefully when its inputs
        # are corrupted, not whether an untouched observation looks noisy.
        act_fn, mean_fn, comm_graph, actor = self._get_actor_for_eval(env, result, net=net)
        rng = np.random.default_rng(config.seed_offset + 1000)

        def _corrupt(o):
            o = o.copy()
            if config.sensor_noise_std > 0:
                o = o + rng.normal(0, config.sensor_noise_std, o.shape)
            if config.sensor_bias != 0:
                # Systematic bias: unlike Gaussian noise, a constant
                # offset every step (a miscalibrated CT/PT, not random
                # measurement error).
                o = o + config.sensor_bias
            if config.missing_sensor_rate > 0:
                mask = rng.random(o.shape) < config.missing_sensor_rate
                o[mask] = 0.0
            return o

        obs = env.reset()
        done = False
        steps = 0
        while not done and steps < 300:
            # Corrupt EVERY agent's raw observation before any comm-graph
            # forward pass, since act_fn (when use_comm=True) computes
            # GNN attention over the full multi-agent observation array —
            # corrupting only obs[i] in isolation would leave clean
            # neighbor messages flowing into a noisy agent's decision,
            # understating how sensor faults actually propagate through
            # the learned communication channel.
            noisy_obs = np.array([_corrupt(obs[i]) for i in range(env.n_agents)])
            actions = np.zeros((env.n_agents, env.ACT_DIM))
            for i in range(env.n_agents):
                a, _ = act_fn(noisy_obs, i, rng)
                actions[i] = a
            obs, reward, done, _ = env.step(actions)
            steps += 1

        TMS, PS = env.get_state()
        try:
            of_noisy, _, _ = objective_function(net, TMS, PS, fault_type=config.fault_type)
            constraints = check_constraints(net, TMS, PS, fault_type=config.fault_type)
            result.noisy_of = float(of_noisy)
            result.noisy_cti_ok = constraints["cti_satisfied"]
            signal_power = np.mean(obs ** 2) if hasattr(obs, '__len__') else 1.0
            result.sensor_snr = float(10 * np.log10(signal_power / (config.sensor_noise_std ** 2 + 1e-10)))
        except Exception as e:
            warnings.warn(f"Sensor robustness eval failed: {e}")

        return result

    # ── Fig 25: Zero-Shot Transfer ───────────────────────────────────────────

    def _eval_zero_shot_transfer(self, config: TrainingConfig, net, result: TrainingResult) -> TrainingResult:
        """
        Transfer trained policy to different system without fine-tuning.

        Previous version pasted result.best_tms/best_ps directly onto the
        target environment's state and read off objective_function
        immediately, without the policy ever acting. That tests whether
        a FROZEN numeric operating point transfers, not whether the
        POLICY transfers — a materially different and weaker claim than
        "zero-shot transfer", which should mean: does the trained actor,
        when it observes the target system's own state, produce
        reasonable actions on a topology it never trained on?

        Note: if n_target != n_source (e.g. transferring 8-bus -> 30-bus),
        the policy is a SHARED per-agent network (SharedGaussianActor),
        so it can be applied to any number of agents without
        architecture changes — that part of the original code's
        assumption was directionally correct, just never exercised
        because the policy never got to act.
        """
        from network.network_data import NETWORKS
        from marl.marl_env import DynamicDOCREnv
        from network.relay_physics import objective_function

        target_system = config.eval_system
        target_fn = NETWORKS.get(target_system)
        if target_fn is None:
            warnings.warn(f"Unknown eval system: {target_system}")
            return result

        target_net = target_fn()
        if config.pv_level > 0:
            target_net = target_net.apply_pv_penetration(config.pv_level)

        env = DynamicDOCREnv(target_net, action_scale=0.05, max_steps=300,
                             seed=config.seed_offset + 1100,
                             fault_type=config.fault_type)

        try:
            # use_comm policies need comm_n_agents to match the TARGET
            # system's agent count for the GNN adjacency to be valid;
            # rebuilding via _get_actor_for_eval with net=target_net
            # ensures build_relay_adjacency() reflects the target
            # topology, not the source one the policy was trained on.
            act_fn, mean_fn, comm_graph, actor = self._get_actor_for_eval(
                env, result, net=target_net
            )
            rng = np.random.default_rng(config.seed_offset + 1100)

            obs = env.reset(seed=config.seed_offset + 1100)
            done = False
            steps = 0
            while not done and steps < env.max_steps:
                actions = np.zeros((env.n_agents, env.ACT_DIM))
                for i in range(env.n_agents):
                    a, _ = act_fn(obs, i, rng)
                    actions[i] = a
                obs, _, done, _ = env.step(actions)
                steps += 1

            TMS, PS = env.get_state()
            of_transfer, _, _ = objective_function(target_net, TMS, PS, fault_type=config.fault_type)
            constraints = check_constraints(target_net, TMS, PS, fault_type=config.fault_type)

            result.transfer_of = float(of_transfer)
            result.transfer_cti_ok = constraints["cti_satisfied"]
            result.transfer_system = target_system
        except Exception as e:
            warnings.warn(f"Zero-shot transfer failed: {e}")

        return result


    # ═════════════════════════════════════════════════════════════════════════
    #  PART 4: FIGURE-SPECIFIC EVALUATION METHODS (Figures 36-58)
    # ═════════════════════════════════════════════════════════════════════════

    # ── Fig 36: Action Space Geometry ───────────────────────────────────────

    def _eval_action_space_grid(self, config: TrainingConfig, net, result: TrainingResult) -> TrainingResult:
        """
        Grid search over TDS/PSM space to visualize CTI and OF landscape.
        """
        from network.relay_physics import check_constraints, objective_function

        tds_vals = np.linspace(config.grid_tds_range[0], config.grid_tds_range[1], config.grid_resolution)
        psm_vals = np.linspace(config.grid_psm_range[0], config.grid_psm_range[1], config.grid_resolution)

        cti_grid = np.zeros((config.grid_resolution, config.grid_resolution))
        of_grid = np.zeros((config.grid_resolution, config.grid_resolution))

        n_relays = len(result.best_tms)

        for i, tds in enumerate(tds_vals):
            for j, psm in enumerate(psm_vals):
                TMS_test = np.full(n_relays, tds)
                PS_test = np.full(n_relays, psm)

                try:
                    of_val, _, _ = objective_function(net, TMS_test, PS_test, fault_type=config.fault_type)
                    constraints = check_constraints(net, TMS_test, PS_test, fault_type=config.fault_type)
                    of_grid[i, j] = of_val
                    cti_grid[i, j] = 1.0 if constraints["cti_satisfied"] else 0.0
                except Exception:
                    of_grid[i, j] = np.nan
                    cti_grid[i, j] = 0.0

        result.action_space_grid_data = {
            "tds_values": tds_vals.tolist(),
            "psm_values": psm_vals.tolist(),
            "cti_grid": cti_grid.tolist(),
            "of_grid": of_grid.tolist(),
        }
        return result

    # ── Fig 44: Fault Current Contribution ────────────────────────────────────

    def _eval_fault_contributions(self, config: TrainingConfig, net, result: TrainingResult) -> TrainingResult:
        """
        Use pandapower short-circuit study to compute fault current contributions.

        Previous version called `pandapower.shortcircuit.calc_sc(net, ...)`
        directly on `net`, which is a NetworkData dataclass instance, not
        a pandapower network object — calc_sc expects a pandapower net
        (with .bus, .line, .sgen tables etc.) and would raise immediately
        on any NetworkData input, landing in the except branch every
        time and silently recording {"error": ...} as the "result".

        Fixed version builds the actual PandapowerNetwork wrapper (which
        DOES expose a real .pp_net), applies the configured PV
        penetration to the DG/sgen elements first (so the SC study
        reflects the PV level actually being evaluated, not whatever
        default penetration the network was built with), then runs the
        short-circuit study on pp_net.
        """
        try:
            from network.pandapower_network import build_pandapower_network, PANDAPOWER_AVAILABLE

            if not PANDAPOWER_AVAILABLE:
                warnings.warn("pandapower not installed; fault contribution eval requires it")
                result.fault_contributions = {"error": "pandapower not installed"}
                return result

            pp_wrapper = build_pandapower_network(net)
            pp_wrapper.update_pv_penetration(config.pv_level)
            pp_wrapper.run_power_flow()

            import pandapower.shortcircuit as sc
            sc.calc_sc(pp_wrapper.pp_net, fault=config.fault_type, case="max")

            contributions = {}
            pp_net = pp_wrapper.pp_net
            if hasattr(pp_net, 'res_bus_sc') and len(pp_net.res_bus_sc) > 0:
                total_ik = pp_net.res_bus_sc['ikss_ka'].sum()
                for bus_idx, ik in pp_net.res_bus_sc['ikss_ka'].items():
                    bus_name = pp_net.bus.name.at[bus_idx] if hasattr(pp_net.bus, 'name') else f"bus_{bus_idx}"
                    contributions[str(bus_name)] = float(ik / total_ik * 100) if total_ik > 0 else 0.0

            result.fault_contributions = contributions
        except Exception as e:
            warnings.warn(f"Fault contribution eval failed: {e}")
            result.fault_contributions = {"error": str(e)}

        return result

    # ── Fig 45: Daily PV & Adaptive Relay Response ────────────────────────────

    def _eval_daily_pv(self, config: TrainingConfig, net, result: TrainingResult) -> TrainingResult:
        """
        Simulate 24-hour operation with varying irradiance.

        Previous version sampled actions via
        np.random.normal(0, 0.05, ...) instead of the trained policy,
        so "adaptive relay response" was actually a random walk's
        response. It also set `env.current_pv_level` mid-episode without
        ever calling env.reset(pv_level=...) again — but
        current_pv_level only feeds into the fault-current/observation
        pipeline THROUGH dist_net.reset() (see marl/marl_env.py:
        DynamicDOCREnv.reset), which step() never calls. So changing
        current_pv_level between steps had no effect on the actual
        fault-current physics for that simulated hour; only the next
        full reset() would have picked it up, and that never happened
        inside the loop.
        """
        from marl.marl_env import DynamicDOCREnv
        from network.relay_physics import objective_function

        if config.irradiance_profile is not None:
            irradiance = config.irradiance_profile
        else:
            hours = np.arange(24)
            irradiance = np.maximum(0, np.sin(np.pi * hours / 12) * 1000)

        env = DynamicDOCREnv(net, action_scale=0.05, max_steps=50,
                             seed=config.seed_offset + 1200,
                             fault_type=config.fault_type)

        act_fn, mean_fn, comm_graph, actor = self._get_actor_for_eval(env, result, net=net)
        rng = np.random.default_rng(config.seed_offset + 1200)

        tms_traj = []
        ps_traj = []
        of_traj = []

        for h in range(24):
            pv_factor = irradiance[h] / 1000.0
            hour_pv_level = float(np.clip(config.pv_level * pv_factor, 0.0, 1.0))

            # Genuinely apply the hour's PV level: reset() is the only
            # path that feeds pv_level into dist_net.reset(), which
            # recomputes fault currents for the new generation mix.
            obs = env.reset(seed=config.seed_offset + 1200 + h, pv_level=hour_pv_level)

            done = False
            steps = 0
            while not done and steps < env.max_steps:
                actions = np.zeros((env.n_agents, env.ACT_DIM))
                for i in range(env.n_agents):
                    a, _ = act_fn(obs, i, rng)
                    actions[i] = a
                obs, _, done, _ = env.step(actions)
                steps += 1

            TMS, PS = env.get_state()
            tms_traj.append(TMS.copy())
            ps_traj.append(PS.copy())

            try:
                of_h, _, _ = objective_function(net, TMS, PS, fault_type=config.fault_type)
                of_traj.append(float(of_h))
            except Exception:
                of_traj.append(np.nan)

        result.daily_tms_trajectory = np.array(tms_traj)
        result.daily_ps_trajectory = np.array(ps_traj)
        result.daily_of_trajectory = np.array(of_traj)
        result.daily_irradiance = irradiance

        return result

    # ── Fig 46: Relay Operating Characteristic Curves ───────────────────────

    def _eval_relay_curves(self, config: TrainingConfig, net, result: TrainingResult) -> TrainingResult:
        """
        Compute operating time t_op for range of fault currents using learned settings.
        """
        n_relays = len(result.best_tms)
        If_range = np.linspace(config.fault_current_range[0],
                               config.fault_current_range[1], 100)

        curves = {}
        for relay_id in range(n_relays):
            tds = result.best_tms[relay_id]
            psm = result.best_ps[relay_id]
            t_op = []
            for If in If_range:
                if If / psm > 1.0:
                    t = tds * 0.14 / ((If / psm) ** 0.02 - 1)
                else:
                    t = np.inf
                t_op.append(float(t))

            curves[f"relay_{relay_id}"] = {
                "If_range_A": If_range.tolist(),
                "t_op_s": t_op,
                "TDS": float(tds),
                "PSM": float(psm),
            }

        result.relay_curves = curves
        return result

    # ── Fig 47: Sympathetic Tripping ──────────────────────────────────────────

    def _eval_sympathetic_tripping(self, config: TrainingConfig, net, result: TrainingResult) -> TrainingResult:
        """
        Compare MARL vs traditional settings for sympathetic tripping.

        Previous version had two independent bugs that together meant
        sympathetic_trips and traditional_trips were ALWAYS 0:
          1. `env.TMS = result.best_tms.copy()` was set BEFORE
             `env.reset()`, but DynamicDOCREnv.reset() unconditionally
             re-randomizes TMS/PS internally (see marl_env.py:
             `self.TMS = self.rng.uniform(...)`), silently discarding
             the assignment on every one of the 10 episodes.
          2. `info.get('backup_trip', False)` and
             `hasattr(env, 'inject_fault')` reference an info key and a
             method that do not exist anywhere in DynamicDOCREnv — the
             real step() info dict never contains 'backup_trip', and
             there is no inject_fault() method, so the condition was
             always False and the hasattr check always failed silently.

        Fixed version uses the real, physically meaningful signal
        already computed every step: CTI margin (t_bkp - t_pri). A
        sympathetic/uncoordinated trip is exactly the case where the
        backup relay's margin over the primary collapses below the CTI
        threshold (it would trip at the same time as, or before, the
        primary) — this is what `check_constraints()['cti_violations']`
        already counts, so we use it directly rather than inventing a
        new ad hoc signal.
        """
        from marl.marl_env import DynamicDOCREnv

        n_trips_marl = 0
        n_trips_trad = 0
        n_episodes = 10

        # With MARL (trained policy) settings — let the policy actually act.
        env_marl = DynamicDOCREnv(net, action_scale=0.05, max_steps=50,
                                  seed=config.seed_offset + 1300,
                                  fault_type=config.fault_type)
        act_fn, mean_fn, comm_graph, actor = self._get_actor_for_eval(env_marl, result, net=net)
        rng = np.random.default_rng(config.seed_offset + 1300)

        for ep in range(n_episodes):
            obs = env_marl.reset(seed=config.seed_offset + 1300 + ep)
            done = False
            steps = 0
            while not done and steps < env_marl.max_steps:
                actions = np.zeros((env_marl.n_agents, env_marl.ACT_DIM))
                for i in range(env_marl.n_agents):
                    a, _ = act_fn(obs, i, rng)
                    actions[i] = a
                obs, _, done, info = env_marl.step(actions)
                steps += 1
            TMS, PS = env_marl.get_state()
            c = check_constraints(net, TMS, PS, fault_type=config.fault_type)
            n_trips_marl += c["cti_violations"]

        # With traditional (fixed, non-adaptive) settings — zero action
        # every step is the correct way to represent "no adaptation",
        # since DynamicDOCREnv actions are deltas, not absolute settings.
        env_trad = DynamicDOCREnv(net, action_scale=0.05, max_steps=50,
                                  seed=config.seed_offset + 1300,
                                  fault_type=config.fault_type)

        for ep in range(n_episodes):
            obs = env_trad.reset(seed=config.seed_offset + 1300 + ep)
            env_trad.TMS = np.ones(env_trad.n_agents)
            env_trad.PS = np.ones(env_trad.n_agents) * 2.0
            done = False
            steps = 0
            while not done and steps < env_trad.max_steps:
                obs, _, done, info = env_trad.step(np.zeros((env_trad.n_agents, env_trad.ACT_DIM)))
                steps += 1
            TMS, PS = env_trad.get_state()
            c = check_constraints(net, TMS, PS, fault_type=config.fault_type)
            n_trips_trad += c["cti_violations"]

        result.sympathetic_trips = n_trips_marl
        result.traditional_trips = n_trips_trad

        return result

    # ── Fig 48: SHAP Feature Importance ───────────────────────────────────────

    def _eval_shap(self, config: TrainingConfig, net, result: TrainingResult) -> TrainingResult:
        """
        Compute SHAP values for actor policy decisions.

        Previous version's `actor_predict(X)` returned
        `np.random.randn(len(X), env.ACT_DIM)` — a function with NO
        dependence on its input X at all. SHAP attributing "feature
        importance" to a function that ignores its input is meaningless;
        every attribution would be sampling noise. Background collection
        also drove the env with random actions rather than the trained
        policy, so even the input DISTRIBUTION being explained was wrong
        (states a random policy visits, not states the trained policy
        actually visits).
        """
        try:
            import shap
            from marl.marl_env import DynamicDOCREnv

            env = DynamicDOCREnv(net, action_scale=0.05, max_steps=300,
                                 seed=config.seed_offset + 1400,
                                 fault_type=config.fault_type)

            act_fn, mean_fn, comm_graph, actor = self._get_actor_for_eval(env, result, net=net)
            rng = np.random.default_rng(config.seed_offset + 1400)

            background = []
            obs = env.reset()
            for _ in range(config.shap_n_background):
                actions = np.zeros((env.n_agents, env.ACT_DIM))
                for i in range(env.n_agents):
                    a, _ = act_fn(obs, i, rng)
                    actions[i] = a
                # Record the per-agent observation actually consumed by
                # the policy (not the flattened multi-agent obs), since
                # the actor is a per-agent function of a single obs_dim
                # vector and that's what we want feature attribution over.
                for i in range(env.n_agents):
                    background.append(obs[i].copy())
                obs, _, done, _ = env.step(actions)
                if done:
                    obs = env.reset()

            background = np.array(background)

            def actor_predict(X):
                # Real forward pass through the trained policy's mean
                # action (via mean_fn, which correctly approximates the
                # comm-enhanced input when use_comm=True) — deterministic,
                # so SHAP's perturbation analysis reflects the actual
                # function being explained.
                return np.array([mean_fn(x) for x in X])

            n_bg = min(50, len(background))
            n_explain = min(20, len(background))
            explainer = shap.KernelExplainer(actor_predict, background[:n_bg])
            shap_vals = explainer.shap_values(background[:n_explain])

            result.shap_values = np.array(shap_vals)
            result.shap_base_value = float(np.mean(explainer.expected_value))

        except ImportError:
            warnings.warn("shap library not installed; skipping SHAP eval")
        except Exception as e:
            warnings.warn(f"SHAP eval failed: {e}")

        return result

    # ── Fig 49: Decision Boundary (t-SNE) ───────────────────────────────────

    def _eval_tsne(self, config: TrainingConfig, net, result: TrainingResult) -> TrainingResult:
        """
        Collect observations and actions for t-SNE visualization.

        This figure's entire purpose is to show the TRAINED policy's
        decision boundary in observation space. Sampling from
        np.random.normal(0, 0.1, ...) instead collects the decision
        boundary of independent Gaussian noise, which is uninformative
        about the policy and will produce an embedding that's just an
        isotropic blob with no structure tied to relay physics.
        """
        from marl.marl_env import DynamicDOCREnv

        env = DynamicDOCREnv(net, action_scale=0.05, max_steps=300,
                             seed=config.seed_offset + 1500,
                             fault_type=config.fault_type)

        act_fn, mean_fn, comm_graph, actor = self._get_actor_for_eval(env, result, net=net)
        rng = np.random.default_rng(config.seed_offset + 1500)

        observations = []
        actions = []

        obs = env.reset()
        for _ in range(config.tsne_n_samples):
            act = np.zeros((env.n_agents, env.ACT_DIM))
            for i in range(env.n_agents):
                a, _ = act_fn(obs, i, rng)
                act[i] = a

            observations.append(obs.copy())
            actions.append(act.copy())

            obs, _, done, _ = env.step(act)
            if done:
                obs = env.reset()

        obs_array = np.array(observations)
        act_array = np.array(actions)

        obs_flat = obs_array.reshape(-1, obs_array.shape[-1])
        act_flat = act_array.reshape(-1, act_array.shape[-1])

        features = np.concatenate([obs_flat, act_flat], axis=-1)

        try:
            from sklearn.manifold import TSNE
            embeddings = TSNE(n_components=2, random_state=config.seed_offset).fit_transform(features)
            result.tsne_embeddings = embeddings
        except ImportError:
            warnings.warn("sklearn not installed; returning raw features only")
            result.tsne_embeddings = features[:, :2]

        result.tsne_observations = obs_flat
        result.tsne_actions = act_flat

        return result

    # ── Fig 50: Critical P/B Pair Ranking ─────────────────────────────────────

    def _eval_critical_pairs(self, config: TrainingConfig, net, result: TrainingResult) -> TrainingResult:
        """
        Extract Lagrange multipliers and violation margins for each P/B pair.

        Previous version's fallback branch checked for a 'cti_details'
        key in check_constraints()'s return dict — that key does not
        exist (see network/relay_physics.py: check_constraints returns
        cti_satisfied, cti_violations, cti_min, cti_mean, t_pri, t_bkp,
        delta_cti, ... but never cti_details), so the fallback was dead
        code. Combined with safe_rl_stats['lagrange_multipliers'] never
        being populated for any CoreEvalBridge-sourced TrainingResult,
        critical_pairs was always an empty list regardless of how
        constrained any given pair actually was.

        Fixed version uses the real delta_cti array (t_bkp - t_pri per
        pair) that check_constraints already computes, and reports an
        honest zero Lagrange multiplier with a note when no Safe RL
        dual variables are available, rather than silently fabricating
        structure via a dead key lookup.
        """
        critical_pairs = []

        if result.safe_rl_stats and 'lagrange_multipliers' in result.safe_rl_stats:
            lambdas = result.safe_rl_stats['lagrange_multipliers']
            for pair_idx, (pair, lam) in enumerate(zip(net.pairs, lambdas)):
                critical_pairs.append({
                    "primary": pair.primary,
                    "backup": pair.backup,
                    "lambda": float(lam),
                    "violation_rate": result.safe_rl_stats.get('violation_rates', {}).get(pair_idx, 0.0),
                    "distance_to_violation": result.safe_rl_stats.get('distances', {}).get(pair_idx, 0.0),
                    "lambda_source": "safe_rl_dual_variable",
                })
        else:
            from network.relay_physics import check_constraints, CTI
            constraints = check_constraints(net, result.best_tms, result.best_ps,
                                            fault_type=config.fault_type)
            delta_cti = constraints.get("delta_cti")
            if delta_cti is not None:
                for pair_idx, pair in enumerate(net.pairs):
                    margin = float(delta_cti[pair_idx]) - CTI
                    critical_pairs.append({
                        "primary": pair.primary,
                        "backup": pair.backup,
                        "lambda": 0.0,
                        "violation_rate": 1.0 if margin < 0 else 0.0,
                        "distance_to_violation": margin,
                        "lambda_source": "no_safe_rl_duals_available",
                    })

        result.critical_pairs = critical_pairs
        return result

    # ── Fig 53: Model Compression ─────────────────────────────────────────────

    def _eval_compression(self, config: TrainingConfig, net, result: TrainingResult) -> TrainingResult:
        """
        Evaluate policy after quantization and/or pruning of its weights.

        Previous version set env.TMS/env.PS to the trained best values,
        immediately read back objective_function on that SAME unchanged
        TMS/PS, and reported `compressed_of` — which is therefore
        mathematically guaranteed to equal best_of exactly (since
        nothing about the network OR the policy's actual weights ever
        changed), making compression_accuracy_drop always identically
        zero regardless of quantize_bits or prune_ratio. That is not
        evidence of "compression-robust" performance; it's an artifact
        of never having compressed anything that affects behavior.

        Fixed version actually quantizes and prunes the ACTOR's weight
        vector (the only artifact "compression" can meaningfully apply
        to in this numpy-only architecture), then lets the genuinely
        degraded policy act in the environment.
        """
        from marl.marl_env import DynamicDOCREnv
        from network.relay_physics import objective_function

        base_params = len(result.best_tms) + len(result.best_ps)
        if config.quantize_bits:
            bytes_per_param = config.quantize_bits / 8
        else:
            bytes_per_param = 4.0

        pruned_params = int(base_params * (1 - config.prune_ratio))
        size_kb = pruned_params * bytes_per_param / 1024.0
        result.compressed_size_kb = float(size_kb)

        env = DynamicDOCREnv(net, action_scale=0.05, max_steps=300,
                             seed=config.seed_offset + 1600,
                             fault_type=config.fault_type)

        if result.actor_params is None:
            warnings.warn("Compression eval requires a trained policy (actor_params is None)")
            return result

        # Reconstruct the ACTUAL trained actor via _get_actor_for_eval —
        # hand-building a fresh SharedGaussianActor sized for the raw obs
        # dimension (the previous version of this method) silently
        # misaligned MLP.set_params's parameter-chunk splitting whenever
        # result.use_comm=True, since actor_params is then sized for
        # raw_obs_dim + comm_msg_dim, not the raw dimension alone.
        act_fn, _mean_fn, _comm_graph, actor = self._get_actor_for_eval(env, result, net=net)

        # Genuinely compress the actor's weights:
        #   - Quantization: round to the requested bit depth's effective
        #     resolution over the weight's observed range.
        #   - Pruning: zero out the smallest-magnitude weights up to
        #     prune_ratio, the standard magnitude-pruning heuristic.
        compressed_params = actor.get_all_params().copy()

        if config.quantize_bits and config.quantize_bits < 32:
            levels = 2 ** config.quantize_bits
            w_min, w_max = compressed_params.min(), compressed_params.max()
            w_range = max(w_max - w_min, 1e-9)
            step = w_range / levels
            compressed_params = np.round((compressed_params - w_min) / step) * step + w_min

        if config.prune_ratio > 0:
            n_prune = int(len(compressed_params) * config.prune_ratio)
            if n_prune > 0:
                prune_idx = np.argsort(np.abs(compressed_params))[:n_prune]
                compressed_params[prune_idx] = 0.0

        actor.set_all_params(compressed_params)

        rng = np.random.default_rng(config.seed_offset + 1600)
        obs = env.reset()
        done = False
        steps = 0
        while not done and steps < 300:
            actions = np.zeros((env.n_agents, env.ACT_DIM))
            for i in range(env.n_agents):
                # Note: this evaluates compression of the ACTOR only.
                # If result.use_comm, the comm graph is left
                # uncompressed here — Fig 22 (bandwidth) already covers
                # comm-channel compression specifically, so Fig 53
                # isolates actor compression to avoid conflating the
                # two effects in one figure. act_fn (from
                # _get_actor_for_eval) still routes obs through the
                # (uncompressed) comm graph first when use_comm=True,
                # matching how the actor was actually trained.
                a, _ = act_fn(obs, i, rng)
                actions[i] = a
            obs, _, done, _ = env.step(actions)
            steps += 1

        TMS, PS = env.get_state()
        try:
            of_comp, _, _ = objective_function(net, TMS, PS, fault_type=config.fault_type)
            result.compressed_of = float(of_comp)
            result.compression_accuracy_drop = float(of_comp - result.best_of)
        except Exception as e:
            warnings.warn(f"Compression eval failed: {e}")

        return result

    # ── Fig 54: Adversarial Robustness ────────────────────────────────────────

    def _eval_adversarial(self, config: TrainingConfig, net, result: TrainingResult) -> TrainingResult:
        """
        Evaluate under FGSM/PGD adversarial perturbations on the trained policy.

        Previous version had three independent bugs that together made
        the figure meaningless:
          1. "FGSM" (Fast GRADIENT Sign Method) used
             np.sign(np.random.randn(...)) — a random sign, not the sign
             of any gradient of any loss with respect to the observation.
          2. The adversarial observation (`adv_obs`) was computed but
             never passed to the policy — actions were instead sampled
             from np.random.normal(0, 0.1, ...), so the attack never
             reached the thing it was supposed to be attacking.
          3. `perturbation` referenced after the loop only holds the
             value from the LAST of 10 iterations, reported as if it
             characterised the whole run.

        Fixed version computes a true finite-difference gradient of a
        scalar "harm" loss (negative norm of the mean action vector,
        a reasonable proxy for "make the relay setting change as
        differently as possible from its undisturbed decision") with
        respect to the input observation, through the real actor's
        forward pass — the closest defensible analogue to backprop-FGSM
        available without an autodiff framework in this numpy-only
        actor. The perturbed observation is then the ACTUAL input to
        env.step(), so the attack is real.
        """
        from marl.marl_env import DynamicDOCREnv
        from network.relay_physics import objective_function

        env = DynamicDOCREnv(net, action_scale=0.05, max_steps=300,
                             seed=config.seed_offset + 1700,
                             fault_type=config.fault_type)

        act_fn, mean_fn, comm_graph, actor = self._get_actor_for_eval(env, result, net=net)
        rng = np.random.default_rng(config.seed_offset + 1700)

        def _grad_sign(obs_i, h=1e-3):
            """
            Finite-difference estimate of d(||mean_action||)/d(obs_i),
            then take its sign. This is the numpy-only analogue of
            FGSM's sign(grad_x L) when no autodiff is available.
            Uses mean_fn (not the raw actor) so the gradient is taken
            with respect to the same effective input the trained policy
            actually consumes, including its comm-enhanced approximation
            when use_comm=True.
            """
            base = mean_fn(obs_i)
            base_loss = np.linalg.norm(base)
            grad = np.zeros_like(obs_i)
            for k in range(len(obs_i)):
                bump = obs_i.copy()
                bump[k] += h
                loss_k = np.linalg.norm(mean_fn(bump))
                grad[k] = (loss_k - base_loss) / h
            return np.sign(grad)

        obs = env.reset()
        n_success = 0
        n_total = 10
        perturbation_norms = []

        for _ in range(n_total):
            clean_obs = obs.copy()
            adv_obs = clean_obs.copy()

            for i in range(env.n_agents):
                if config.adversarial_attack == "FGSM":
                    grad_sign = _grad_sign(clean_obs[i])
                    perturbation_i = config.attack_epsilon * grad_sign
                elif config.adversarial_attack == "PGD":
                    # Multi-step PGD: iterate FGSM-style steps and project
                    # back into the epsilon-ball each step.
                    perturbed = clean_obs[i].copy()
                    step_size = config.attack_epsilon / 4.0
                    for _pgd_step in range(4):
                        grad_sign = _grad_sign(perturbed)
                        perturbed = perturbed + step_size * grad_sign
                        delta = np.clip(perturbed - clean_obs[i],
                                         -config.attack_epsilon, config.attack_epsilon)
                        perturbed = clean_obs[i] + delta
                    perturbation_i = perturbed - clean_obs[i]
                else:
                    perturbation_i = config.attack_epsilon * rng.standard_normal(clean_obs[i].shape)

                adv_obs[i] = clean_obs[i] + perturbation_i
                perturbation_norms.append(float(np.linalg.norm(perturbation_i)))

            # The policy now actually acts on the ADVERSARIAL observation —
            # this is the step the previous version skipped entirely.
            actions = np.zeros((env.n_agents, env.ACT_DIM))
            for i in range(env.n_agents):
                a, _ = act_fn(adv_obs, i, rng)
                actions[i] = a

            obs, _, done, _ = env.step(actions)
            if done:
                obs = env.reset()

            TMS, PS = env.get_state()
            try:
                of_adv, _, _ = objective_function(net, TMS, PS, fault_type=config.fault_type)
                constraints = check_constraints(net, TMS, PS, fault_type=config.fault_type)
                if not constraints["cti_satisfied"] or of_adv > result.best_of * 1.5:
                    n_success += 1
            except Exception:
                pass

        TMS, PS = env.get_state()
        try:
            of_adv, _, _ = objective_function(net, TMS, PS, fault_type=config.fault_type)
            result.adversarial_of = float(of_adv)
            result.adversarial_success_rate = n_success / n_total
            result.attack_perturbation_norm = float(np.mean(perturbation_norms)) if perturbation_norms else 0.0
        except Exception as e:
            warnings.warn(f"Adversarial eval failed: {e}")

        return result

    # ── Fig 58: Seasonal Generalisation ───────────────────────────────────────

    def _eval_seasonal(self, config: TrainingConfig, net, result: TrainingResult) -> TrainingResult:
        """
        Evaluate under seasonal irradiance/temperature conditions.

        Previous version's temperature derating was a complete no-op
        (see _apply_seasonal_temperature, now fixed), and even granting
        a genuine net_seasonal, this method pasted result.best_tms/
        best_ps directly onto env and read off objective_function
        without the policy ever acting — so "seasonal generalisation"
        never actually exercised the trained policy's behaviour under
        the changed conditions, only a frozen number's compatibility
        with them.
        """
        from marl.marl_env import DynamicDOCREnv
        from network.relay_physics import objective_function

        seasonal_profiles = {
            "spring": {"irradiance_factor": 0.7, "temp": 15.0},
            "summer": {"irradiance_factor": 1.0, "temp": 35.0},
            "autumn": {"irradiance_factor": 0.5, "temp": 10.0},
            "winter": {"irradiance_factor": 0.3, "temp": 0.0},
        }

        profile = seasonal_profiles.get(config.season, {"irradiance_factor": 1.0, "temp": 20.0})
        net_seasonal = self._apply_seasonal_temperature(net, profile["temp"])
        seasonal_pv_level = float(np.clip(config.pv_level * profile["irradiance_factor"], 0.0, 1.0))
        if seasonal_pv_level > 0 and hasattr(net_seasonal, 'apply_pv_penetration'):
            net_seasonal = net_seasonal.apply_pv_penetration(seasonal_pv_level)

        env = DynamicDOCREnv(net_seasonal, action_scale=0.05, max_steps=300,
                             seed=config.seed_offset + 1800,
                             fault_type=config.fault_type)

        act_fn, mean_fn, comm_graph, actor = self._get_actor_for_eval(env, result, net=net_seasonal)
        rng = np.random.default_rng(config.seed_offset + 1800)

        obs = env.reset(seed=config.seed_offset + 1800, pv_level=seasonal_pv_level)
        done = False
        steps = 0
        while not done and steps < env.max_steps:
            actions = np.zeros((env.n_agents, env.ACT_DIM))
            for i in range(env.n_agents):
                a, _ = act_fn(obs, i, rng)
                actions[i] = a
            obs, _, done, _ = env.step(actions)
            steps += 1

        try:
            TMS, PS = env.get_state()
            of_seasonal, _, _ = objective_function(net_seasonal, TMS, PS, fault_type=config.fault_type)
            constraints = check_constraints(net_seasonal, TMS, PS, fault_type=config.fault_type)

            result.seasonal_of = float(of_seasonal)
            result.seasonal_cti_ok = constraints["cti_satisfied"]
        except Exception as e:
            warnings.warn(f"Seasonal eval failed: {e}")

        return result


# ═══════════════════════════════════════════════════════════════════════════════
#  PART 5: EXPERIMENT MATRIX BUILDERS & AGGREGATION
# ═══════════════════════════════════════════════════════════════════════════════

def create_experiment_matrix(
    systems: List[str] = None,
    methods: List[str] = None,
    pv_levels: List[float] = None,
    fault_types: List[str] = None,
    n_runs: int = 10,
    fast: bool = False,
) -> List[TrainingConfig]:
    """Create base experiment configuration matrix (unchanged from original)."""
    if systems is None:
        systems = ["8bus", "14bus", "39bus"]
    if methods is None:
        methods = ["GA-SQP", "MAPPO", "MAPPO-Safe", "MAPPO-Safe-EWC", "HWOA", "JAYA"]
    if pv_levels is None:
        pv_levels = [0.0, 0.2, 0.4, 0.6]
    if fault_types is None:
        fault_types = ["3ph"]

    ga_pop = 30 if fast else 100
    ga_gen = 50 if fast else 200
    meta_pop = 20 if fast else 30
    meta_iter = 200 if fast else 500
    marl_steps = 5000 if fast else 30000

    configs = []
    for system in systems:
        for method in methods:
            for pv_level in pv_levels:
                for fault_type in fault_types:
                    for run in range(n_runs):
                        config = TrainingConfig(
                            method=method,
                            system=system,
                            pv_level=pv_level,
                            fault_type=fault_type,
                            n_runs=n_runs,
                            seed_offset=run * 1000,
                            ga_pop=ga_pop,
                            ga_gen=ga_gen,
                            meta_pop=meta_pop,
                            meta_iter=meta_iter,
                            marl_steps=marl_steps,
                            use_safe_rl="Safe" in method,
                            use_ewc="EWC" in method,
                            use_comm="Comm" in method,
                        )
                        configs.append(config)

    print(f"Created {len(configs)} base experiment configurations:")
    print(f"  Systems: {systems}")
    print(f"  Methods: {methods}")
    print(f"  PV levels: {pv_levels}")
    print(f"  Fault types: {fault_types}")
    print(f"  Runs per config: {n_runs}")

    return configs


# ── Figure-Specific Experiment Matrices ────────────────────────────────────

def create_fig13_comm_stress_matrix(
    base_configs: List[TrainingConfig],
    stress_levels: List[float] = None,
    packet_loss_rates: List[float] = None,
    quantization_bits: List[int] = None,
) -> List[TrainingConfig]:
    """Create configurations for Fig 13: Communication Stress."""
    if stress_levels is None:
        stress_levels = [0.0, 0.3, 0.6, 0.9]
    if packet_loss_rates is None:
        packet_loss_rates = [0.0, 0.1, 0.3, 0.5]
    if quantization_bits is None:
        quantization_bits = [32, 16, 8, 4]

    configs = []
    for base in base_configs:
        if not base.use_comm and "Comm" not in base.method:
            continue
        for stress in stress_levels:
            for plr in packet_loss_rates:
                for qb in quantization_bits:
                    cfg = TrainingConfig(**base.to_dict())
                    cfg.comm_stress_level = stress
                    cfg.packet_loss_rate = plr
                    cfg.quantization_bits = qb
                    cfg.latency_ms = stress * 100
                    configs.append(cfg)

    print(f"Fig 13: Created {len(configs)} comm stress configurations")
    return configs


def create_fig14_n1_contingency_matrix(
    base_configs: List[TrainingConfig],
    line_indices: List[int] = None,
    include_recovery: bool = True,
) -> List[TrainingConfig]:
    """Create configurations for Fig 14: N-1 Contingency Impact & Recovery."""
    if line_indices is None:
        line_indices = list(range(6))

    configs = []
    for base in base_configs:
        for line_idx in line_indices:
            cfg = TrainingConfig(**base.to_dict())
            cfg.n1_line_outage = line_idx
            cfg.n1_recovery_eval = include_recovery
            configs.append(cfg)

    print(f"Fig 14: Created {len(configs)} N-1 contingency configurations")
    return configs


def create_fig20_byzantine_matrix(
    base_configs: List[TrainingConfig],
    n_agents: int = 6,
    byzantine_counts: List[int] = None,
    strategies: List[str] = None,
) -> List[TrainingConfig]:
    """Create configurations for Fig 20: Byzantine Agent Detection."""
    if byzantine_counts is None:
        byzantine_counts = [0, 1, 2, 3]
    if strategies is None:
        strategies = ["random", "reverse", "colluding"]

    configs = []
    for base in base_configs:
        if not base.use_comm and "MAPPO" not in base.method:
            continue
        for count in byzantine_counts:
            for strategy in strategies:
                if count == 0:
                    cfg = TrainingConfig(**base.to_dict())
                    cfg.byzantine_agents = []
                    cfg.byzantine_strategy = strategy
                    configs.append(cfg)
                else:
                    for agents in itertools.combinations(range(n_agents), count):
                        cfg = TrainingConfig(**base.to_dict())
                        cfg.byzantine_agents = list(agents)
                        cfg.byzantine_strategy = strategy
                        configs.append(cfg)

    print(f"Fig 20: Created {len(configs)} Byzantine configurations")
    return configs


def create_fig22_bandwidth_matrix(
    base_configs: List[TrainingConfig],
    gnn_depths: List[int] = None,
    compression_ratios: List[float] = None,
    message_dims: List[int] = None,
) -> List[TrainingConfig]:
    """Create configurations for Fig 22: Bandwidth vs. Performance Trade-off."""
    if gnn_depths is None:
        gnn_depths = [1, 2, 3, 4]
    if compression_ratios is None:
        compression_ratios = [1.0, 0.75, 0.5, 0.25]
    if message_dims is None:
        message_dims = [4, 8, 16, 32]

    configs = []
    for base in base_configs:
        if not base.use_comm and "Comm" not in base.method:
            continue
        for depth in gnn_depths:
            for comp in compression_ratios:
                for msg_dim in message_dims:
                    cfg = TrainingConfig(**base.to_dict())
                    cfg.gnn_depth = depth
                    cfg.compression_ratio = comp
                    cfg.message_dim = msg_dim
                    cfg.compression_eval = True
                    configs.append(cfg)

    print(f"Fig 22: Created {len(configs)} bandwidth configurations")
    return configs


def create_fig23_unseen_fault_matrix(
    base_configs: List[TrainingConfig],
    fault_impedances: List[float] = None,
    evolving_faults: List[bool] = None,
) -> List[TrainingConfig]:
    """Create configurations for Fig 23: Unseen Fault Generalisation."""
    if fault_impedances is None:
        fault_impedances = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]
    if evolving_faults is None:
        evolving_faults = [False, True]

    configs = []
    for base in base_configs:
        for zf in fault_impedances:
            for evol in evolving_faults:
                cfg = TrainingConfig(**base.to_dict())
                cfg.fault_impedance_ohm = zf
                cfg.evolving_fault = evol
                cfg.unseen_fault_eval = True
                configs.append(cfg)

    print(f"Fig 23: Created {len(configs)} unseen fault configurations")
    return configs


def create_fig24_sensor_robustness_matrix(
    base_configs: List[TrainingConfig],
    noise_stds: List[float] = None,
    missing_rates: List[float] = None,
) -> List[TrainingConfig]:
    """Create configurations for Fig 24: Sensor Robustness."""
    if noise_stds is None:
        noise_stds = [0.0, 0.01, 0.05, 0.1, 0.2]
    if missing_rates is None:
        missing_rates = [0.0, 0.05, 0.1, 0.2, 0.3]

    configs = []
    for base in base_configs:
        for noise in noise_stds:
            for miss in missing_rates:
                cfg = TrainingConfig(**base.to_dict())
                cfg.sensor_noise_std = noise
                cfg.missing_sensor_rate = miss
                configs.append(cfg)

    print(f"Fig 24: Created {len(configs)} sensor robustness configurations")
    return configs


def create_fig25_zero_shot_matrix(
    base_configs: List[TrainingConfig],
    train_systems: List[str] = None,
    eval_systems: List[str] = None,
) -> List[TrainingConfig]:
    """Create configurations for Fig 25: Zero-Shot Transfer."""
    if train_systems is None:
        train_systems = ["8bus"]
    if eval_systems is None:
        eval_systems = ["14bus", "39bus"]

    configs = []
    for base in base_configs:
        for train_sys in train_systems:
            for eval_sys in eval_systems:
                if train_sys == eval_sys:
                    continue
                cfg = TrainingConfig(**base.to_dict())
                cfg.system = train_sys
                cfg.train_system = train_sys
                cfg.eval_system = eval_sys
                cfg.freeze_transfer_weights = True
                configs.append(cfg)

    print(f"Fig 25: Created {len(configs)} zero-shot transfer configurations")
    return configs


def create_fig36_action_space_matrix(
    base_configs: List[TrainingConfig],
    tds_ranges: List[Tuple[float, float]] = None,
    psm_ranges: List[Tuple[float, float]] = None,
    resolutions: List[int] = None,
) -> List[TrainingConfig]:
    """Create configurations for Fig 36: Action Space Geometry."""
    if tds_ranges is None:
        tds_ranges = [(0.5, 2.0)]
    if psm_ranges is None:
        psm_ranges = [(1.0, 5.0)]
    if resolutions is None:
        resolutions = [20, 50]

    configs = []
    for base in base_configs:
        for tds_r in tds_ranges:
            for psm_r in psm_ranges:
                for res in resolutions:
                    cfg = TrainingConfig(**base.to_dict())
                    cfg.action_space_grid = True
                    cfg.grid_tds_range = tds_r
                    cfg.grid_psm_range = psm_r
                    cfg.grid_resolution = res
                    configs.append(cfg)

    print(f"Fig 36: Created {len(configs)} action space configurations")
    return configs


def create_fig44_fault_contrib_matrix(
    base_configs: List[TrainingConfig],
) -> List[TrainingConfig]:
    """Create configurations for Fig 44: Fault Current Contribution."""
    configs = []
    for base in base_configs:
        cfg = TrainingConfig(**base.to_dict())
        cfg.run_shortcircuit_study = True
        configs.append(cfg)

    print(f"Fig 44: Created {len(configs)} fault contribution configurations")
    return configs


def create_fig45_daily_pv_matrix(
    base_configs: List[TrainingConfig],
) -> List[TrainingConfig]:
    """Create configurations for Fig 45: Daily PV & Adaptive Relay Response."""
    configs = []
    for base in base_configs:
        if base.pv_level <= 0:
            continue
        cfg = TrainingConfig(**base.to_dict())
        cfg.daily_profile = True
        configs.append(cfg)

    print(f"Fig 45: Created {len(configs)} daily PV configurations")
    return configs


def create_fig46_relay_curves_matrix(
    base_configs: List[TrainingConfig],
    current_ranges: List[Tuple[float, float]] = None,
) -> List[TrainingConfig]:
    """Create configurations for Fig 46: Relay Operating Characteristic Curves."""
    if current_ranges is None:
        current_ranges = [(100.0, 10000.0)]

    configs = []
    for base in base_configs:
        for cr in current_ranges:
            cfg = TrainingConfig(**base.to_dict())
            cfg.relay_curve_eval = True
            cfg.fault_current_range = cr
            configs.append(cfg)

    print(f"Fig 46: Created {len(configs)} relay curve configurations")
    return configs


def create_fig47_sympathetic_matrix(
    base_configs: List[TrainingConfig],
) -> List[TrainingConfig]:
    """Create configurations for Fig 47: Sympathetic Tripping."""
    configs = []
    for base in base_configs:
        cfg = TrainingConfig(**base.to_dict())
        cfg.sympathetic_tripping_eval = True
        configs.append(cfg)

    print(f"Fig 47: Created {len(configs)} sympathetic tripping configurations")
    return configs


def create_fig48_shap_matrix(
    base_configs: List[TrainingConfig],
    n_backgrounds: List[int] = None,
) -> List[TrainingConfig]:
    """Create configurations for Fig 48: SHAP Feature Importance."""
    if n_backgrounds is None:
        n_backgrounds = [50, 100]

    configs = []
    for base in base_configs:
        for nb in n_backgrounds:
            cfg = TrainingConfig(**base.to_dict())
            cfg.shap_eval = True
            cfg.shap_n_background = nb
            configs.append(cfg)

    print(f"Fig 48: Created {len(configs)} SHAP configurations")
    return configs


def create_fig49_tsne_matrix(
    base_configs: List[TrainingConfig],
    n_samples_list: List[int] = None,
) -> List[TrainingConfig]:
    """Create configurations for Fig 49: Decision Boundary (t-SNE)."""
    if n_samples_list is None:
        n_samples_list = [500, 1000]

    configs = []
    for base in base_configs:
        for ns in n_samples_list:
            cfg = TrainingConfig(**base.to_dict())
            cfg.tsne_eval = True
            cfg.tsne_n_samples = ns
            configs.append(cfg)

    print(f"Fig 49: Created {len(configs)} t-SNE configurations")
    return configs


def create_fig50_critical_pairs_matrix(
    base_configs: List[TrainingConfig],
) -> List[TrainingConfig]:
    """Create configurations for Fig 50: Critical P/B Pair Ranking."""
    configs = []
    for base in base_configs:
        if not base.use_safe_rl and "Safe" not in base.method:
            continue
        cfg = TrainingConfig(**base.to_dict())
        cfg.critical_pair_eval = True
        configs.append(cfg)

    print(f"Fig 50: Created {len(configs)} critical pair configurations")
    return configs


def create_fig53_compression_matrix(
    base_configs: List[TrainingConfig],
    quantize_bits_list: List[Optional[int]] = None,
    prune_ratios: List[float] = None,
) -> List[TrainingConfig]:
    """Create configurations for Fig 53: Model Compression."""
    if quantize_bits_list is None:
        quantize_bits_list = [None, 8, 4]
    if prune_ratios is None:
        prune_ratios = [0.0, 0.3, 0.5, 0.7]

    configs = []
    for base in base_configs:
        for qb in quantize_bits_list:
            for pr in prune_ratios:
                cfg = TrainingConfig(**base.to_dict())
                cfg.quantize_bits = qb
                cfg.prune_ratio = pr
                cfg.compression_eval = True
                configs.append(cfg)

    print(f"Fig 53: Created {len(configs)} compression configurations")
    return configs


def create_fig54_adversarial_matrix(
    base_configs: List[TrainingConfig],
    attacks: List[str] = None,
    epsilons: List[float] = None,
) -> List[TrainingConfig]:
    """Create configurations for Fig 54: Adversarial Robustness."""
    if attacks is None:
        attacks = ["FGSM", "PGD", "SPSA"]
    if epsilons is None:
        epsilons = [0.001, 0.01, 0.05, 0.1]

    configs = []
    for base in base_configs:
        for attack in attacks:
            for eps in epsilons:
                cfg = TrainingConfig(**base.to_dict())
                cfg.adversarial_attack = attack
                cfg.attack_epsilon = eps
                cfg.adversarial_eval = True
                configs.append(cfg)

    print(f"Fig 54: Created {len(configs)} adversarial configurations")
    return configs


def create_fig58_seasonal_matrix(
    base_configs: List[TrainingConfig],
    seasons: List[str] = None,
) -> List[TrainingConfig]:
    """Create configurations for Fig 58: Seasonal Generalisation."""
    if seasons is None:
        seasons = ["spring", "summer", "autumn", "winter"]

    configs = []
    for base in base_configs:
        for season in seasons:
            cfg = TrainingConfig(**base.to_dict())
            cfg.season = season
            temp_map = {"spring": 15.0, "summer": 35.0, "autumn": 10.0, "winter": 0.0}
            cfg.seasonal_temperature = temp_map.get(season, 20.0)
            configs.append(cfg)

    print(f"Fig 58: Created {len(configs)} seasonal configurations")
    return configs


# ── Master Experiment Builder ────────────────────────────────────────────────

def create_full_experiment_suite(
    systems: List[str] = None,
    methods: List[str] = None,
    pv_levels: List[float] = None,
    fault_types: List[str] = None,
    n_runs: int = 10,
    fast: bool = False,
    figures: List[str] = None,
) -> Dict[str, List[TrainingConfig]]:
    """
    Create a complete suite of experiments for all figures.

    Returns:
        Dictionary mapping figure name to list of configurations.
    """
    if figures is None:
        figures = ["base", "13", "14", "20", "22", "23", "24", "25",
                   "36", "44", "45", "46", "47", "48", "49", "50", "53", "54", "58"]

    base_configs = create_experiment_matrix(
        systems=systems, methods=methods, pv_levels=pv_levels,
        fault_types=fault_types, n_runs=n_runs, fast=fast,
    )

    suite = {"base": base_configs}

    builders = {
        "13": create_fig13_comm_stress_matrix,
        "14": create_fig14_n1_contingency_matrix,
        "20": create_fig20_byzantine_matrix,
        "22": create_fig22_bandwidth_matrix,
        "23": create_fig23_unseen_fault_matrix,
        "24": create_fig24_sensor_robustness_matrix,
        "25": create_fig25_zero_shot_matrix,
        "36": create_fig36_action_space_matrix,
        "44": create_fig44_fault_contrib_matrix,
        "45": create_fig45_daily_pv_matrix,
        "46": create_fig46_relay_curves_matrix,
        "47": create_fig47_sympathetic_matrix,
        "48": create_fig48_shap_matrix,
        "49": create_fig49_tsne_matrix,
        "50": create_fig50_critical_pairs_matrix,
        "53": create_fig53_compression_matrix,
        "54": create_fig54_adversarial_matrix,
        "58": create_fig58_seasonal_matrix,
    }

    for fig in figures:
        if fig == "base":
            continue
        if fig in builders:
            try:
                suite[f"fig{fig}"] = builders[fig](base_configs)
            except Exception as e:
                warnings.warn(f"Failed to build fig{fig} matrix: {e}")

    total = sum(len(v) for v in suite.values())
    print(f"\n{'='*70}")
    print(f"  TOTAL EXPERIMENT SUITE: {total} configurations across {len(suite)} figure groups")
    print(f"{'='*70}")
    for name, configs in suite.items():
        print(f"  {name}: {len(configs)} configs")

    return suite


# ── Aggregation Methods ──────────────────────────────────────────────────────

def aggregate_by_method(results: List[TrainingResult]) -> Dict[str, Dict]:
    """Aggregate results by method (from original, extended with new fields)."""
    agg = {}
    for r in results:
        method = r.config.method
        if method not in agg:
            agg[method] = {
                "count": 0, "of_values": [], "cti_ok_count": 0,
                "elapsed_times": [],
                "comm_stress_ofs": [], "n1_ofs": [], "byzantine_ofs": [],
                "bandwidth_kbpcs": [], "unseen_fault_ofs": [],
                "noisy_ofs": [], "transfer_ofs": [],
            }
        agg[method]["count"] += 1
        agg[method]["of_values"].append(r.best_of)
        if r.cti_satisfied:
            agg[method]["cti_ok_count"] += 1
        agg[method]["elapsed_times"].append(r.elapsed_seconds)

        if r.comm_stress_of is not None:
            agg[method]["comm_stress_ofs"].append(r.comm_stress_of)
        if r.n1_of is not None:
            agg[method]["n1_ofs"].append(r.n1_of)
        if r.byzantine_of is not None:
            agg[method]["byzantine_ofs"].append(r.byzantine_of)
        if r.bandwidth_kbps is not None:
            agg[method]["bandwidth_kbpcs"].append(r.bandwidth_kbps)
        if r.unseen_fault_of is not None:
            agg[method]["unseen_fault_ofs"].append(r.unseen_fault_of)
        if r.noisy_of is not None:
            agg[method]["noisy_ofs"].append(r.noisy_of)
        if r.transfer_of is not None:
            agg[method]["transfer_ofs"].append(r.transfer_of)

    for method in agg:
        ofs = agg[method]["of_values"]
        times = agg[method]["elapsed_times"]
        agg[method]["of_mean"] = float(np.mean(ofs))
        agg[method]["of_std"] = float(np.std(ofs))
        agg[method]["of_min"] = float(np.min(ofs))
        agg[method]["of_max"] = float(np.max(ofs))
        agg[method]["cti_ok_rate"] = agg[method]["cti_ok_count"] / agg[method]["count"]
        agg[method]["time_mean"] = float(np.mean(times))
        agg[method]["time_std"] = float(np.std(times))

        for key in ["comm_stress_ofs", "n1_ofs", "byzantine_ofs",
                    "bandwidth_kbpcs", "unseen_fault_ofs", "noisy_ofs", "transfer_ofs"]:
            if agg[method][key]:
                agg[method][f"{key}_mean"] = float(np.mean(agg[method][key]))
                agg[method][f"{key}_std"] = float(np.std(agg[method][key]))

    return agg


def aggregate_by_pv_level(results: List[TrainingResult]) -> Dict[float, Dict]:
    """Aggregate results by PV penetration level (from original)."""
    agg = {}
    for r in results:
        pv = r.config.pv_level
        if pv not in agg:
            agg[pv] = {"count": 0, "of_values": [], "cti_ok_count": 0}
        agg[pv]["count"] += 1
        agg[pv]["of_values"].append(r.best_of)
        if r.cti_satisfied:
            agg[pv]["cti_ok_count"] += 1

    for pv in agg:
        ofs = agg[pv]["of_values"]
        agg[pv]["of_mean"] = float(np.mean(ofs))
        agg[pv]["of_std"] = float(np.std(ofs))
        agg[pv]["cti_ok_rate"] = agg[pv]["cti_ok_count"] / agg[pv]["count"]

    return agg


def aggregate_by_figure(results: List[TrainingResult], figure_field: str) -> Dict[Any, Dict]:
    """
    Generic aggregation by any figure-specific field.

    Example: aggregate_by_figure(results, "comm_stress_level") 
             -> {0.0: {...}, 0.3: {...}, 0.6: {...}, 0.9: {...}}
    """
    agg = {}
    for r in results:
        key = getattr(r.config, figure_field, None)
        if key is None:
            continue

        if key not in agg:
            agg[key] = {"count": 0, "of_values": [], "cti_ok_count": 0,
                       "figure_ofs": [], "figure_cti_oks": []}

        agg[key]["count"] += 1
        agg[key]["of_values"].append(r.best_of)

        result_field = figure_field.replace("_level", "_of").replace("_rate", "_of")
        fig_of = getattr(r, result_field, None)
        if fig_of is not None:
            agg[key]["figure_ofs"].append(fig_of)

        fig_cti = getattr(r, result_field.replace("_of", "_cti_ok"), None)
        if fig_cti is not None:
            agg[key]["figure_cti_oks"].append(fig_cti)

    for key in agg:
        if agg[key]["figure_ofs"]:
            agg[key]["figure_of_mean"] = float(np.mean(agg[key]["figure_ofs"]))
            agg[key]["figure_of_std"] = float(np.std(agg[key]["figure_ofs"]))
        if agg[key]["figure_cti_oks"]:
            agg[key]["figure_cti_ok_rate"] = np.mean(agg[key]["figure_cti_oks"])

    return agg


# ═══════════════════════════════════════════════════════════════════════════════
#  PART 6: RESULT ANALYSIS & PLOTTING UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════
# (The original `if __name__ == "__main__":` block that used to live here --
#  an unconditional "Example 1 + Example 2" run -- was removed: it predated
#  and was fully superseded by the "UPDATED __main__ WITH ALL EXAMPLES"
#  block below (example_1_basic_usage()/example_2_full_suite() reproduce
#  both examples, selectable by CLI arg, alongside 3 more this block never
#  had). Since Python has no early-return between two top-level
#  `if __name__ == "__main__":` blocks, running this file directly used to
#  execute both -- the old unconditional example run, THEN the real
#  argv-driven dispatcher -- rather than either being disabled.

class ResultAnalyzer:
    """
    Post-processing utilities for analyzing pipeline results.
    Generates tables, statistics, and plotting-ready data structures.
    """

    def __init__(self, results: List[TrainingResult]):
        self.results = results
        self.df = self._to_dataframe()

    def _to_dataframe(self):
        """Convert results to a pandas-like dictionary structure."""
        try:
            import pandas as pd
            records = []
            for r in self.results:
                d = r.to_dict()
                records.append(d)
            return pd.DataFrame(records)
        except ImportError:
            warnings.warn("pandas not installed; using dict-based analysis")
            return None

    def get_best_result(self, method: str = None, system: str = None) -> Optional[TrainingResult]:
        """Get the result with lowest OF, optionally filtered by method/system."""
        filtered = self.results
        if method:
            filtered = [r for r in filtered if r.config.method == method]
        if system:
            filtered = [r for r in filtered if r.config.system == system]
        if not filtered:
            return None
        return min(filtered, key=lambda r: r.best_of)

    def get_pareto_front(self, objectives: List[str] = None) -> List[TrainingResult]:
        """
        Extract Pareto-optimal results for multi-objective analysis.
        Default objectives: best_of (minimize), cti_min (maximize).
        """
        if objectives is None:
            objectives = [("best_of", -1), ("cti_min", 1)]  # (field, direction)

        pareto = []
        for r in self.results:
            dominated = False
            for other in self.results:
                if r is other:
                    continue
                better_or_equal = True
                strictly_better = False
                for field, direction in objectives:
                    r_val = getattr(r, field, getattr(r.config, field, 0))
                    o_val = getattr(other, field, getattr(other.config, field, 0))
                    if direction * r_val > direction * o_val:
                        better_or_equal = False
                        break
                    elif direction * r_val < direction * o_val:
                        strictly_better = True
                if better_or_equal and strictly_better:
                    dominated = True
                    break
            if not dominated:
                pareto.append(r)
        return pareto

    def compare_methods(self, metric: str = "best_of") -> Dict[str, Dict]:
        """Statistical comparison of methods using Wilcoxon rank-sum test."""
        try:
            from scipy import stats
        except ImportError:
            warnings.warn("scipy not installed; returning descriptive stats only")
            return aggregate_by_method(self.results)

        methods = list(set(r.config.method for r in self.results))
        comparison = {}

        for i, m1 in enumerate(methods):
            vals1 = [getattr(r, metric) for r in self.results if r.config.method == m1]
            comparison[m1] = {
                "mean": float(np.mean(vals1)),
                "std": float(np.std(vals1)),
                "median": float(np.median(vals1)),
                "vs_others": {}
            }
            for m2 in methods[i+1:]:
                vals2 = [getattr(r, metric) for r in self.results if r.config.method == m2]
                if len(vals1) > 0 and len(vals2) > 0:
                    stat, pval = stats.ranksums(vals1, vals2)
                    comparison[m1]["vs_others"][m2] = {
                        "statistic": float(stat),
                        "p_value": float(pval),
                        "significant": pval < 0.05
                    }

        return comparison

    def figure_summary(self, figure_name: str) -> Dict:
        """Generate summary statistics for a specific figure's data."""
        field_map = {
            "fig13": ("comm_stress_level", "comm_stress_of"),
            "fig14": ("n1_line_outage", "n1_of"),
            "fig20": ("byzantine_strategy", "byzantine_of"),
            "fig22": ("compression_ratio", "bandwidth_of"),
            "fig23": ("fault_impedance_ohm", "unseen_fault_of"),
            "fig24": ("sensor_noise_std", "noisy_of"),
            "fig25": ("eval_system", "transfer_of"),
            "fig53": ("prune_ratio", "compressed_of"),
            "fig54": ("adversarial_attack", "adversarial_of"),
            "fig58": ("season", "seasonal_of"),
        }

        if figure_name not in field_map:
            return {"error": f"Unknown figure: {figure_name}"}

        config_field, result_field = field_map[figure_name]
        groups = {}

        for r in self.results:
            key = getattr(r.config, config_field, None)
            val = getattr(r, result_field, None)
            if key is not None and val is not None:
                if key not in groups:
                    groups[key] = []
                groups[key].append(val)

        summary = {}
        for key, vals in sorted(groups.items()):
            summary[str(key)] = {
                "count": len(vals),
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
            }

        return summary

    def export_latex_table(self, filepath: str, metrics: List[str] = None):
        """Export aggregated results as a LaTeX table for thesis."""
        if metrics is None:
            metrics = ["best_of", "cti_min", "elapsed_seconds"]

        agg = aggregate_by_method(self.results)
        methods = sorted(agg.keys())

        lines = [
            "\\begin{table}[htbp]",
            "\\centering",
            "\\caption{Performance Comparison of DOCR Methods}",
            "\\label{tab:method_comparison}",
            "\\begin{tabular}{l" + "c" * len(metrics) + "}",
            "\\toprule",
            "Method & " + " & ".join(metrics) + " \\\\",
            "\\midrule",
        ]

        for method in methods:
            row = [method]
            for metric in metrics:
                if metric in agg[method]:
                    val = agg[method][metric]
                    if isinstance(val, float):
                        row.append(f"{val:.4f}")
                    else:
                        row.append(str(val))
                else:
                    row.append("--")
            lines.append(" & ".join(row) + " \\\\")

        lines.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
        ])

        with open(filepath, 'w') as f:
            f.write("".join(lines))

        print(f"LaTeX table exported to {filepath}")


# ═══════════════════════════════════════════════════════════════════════════════
#  PART 7: CONFIGURATION PRESETS FOR COMMON EXPERIMENTS
# ═══════════════════════════════════════════════════════════════════════════════

class ExperimentPresets:
    """Quick configuration presets for common experimental setups."""

    @staticmethod
    def thesis_chapter5_validation(fast: bool = False) -> List[TrainingConfig]:
        """
        Preset for Chapter 5 validation experiments.
        Core results: method comparison across PV levels.
        """
        return create_experiment_matrix(
            systems=["8bus", "14bus", "39bus"],
            methods=["GA-SQP", "HWOA", "JAYA", "MAPPO", "MAPPO-Comm", "MAPPO-Safe"],
            pv_levels=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            fault_types=["3ph"],
            n_runs=10 if not fast else 3,
            fast=fast,
        )

    @staticmethod
    def thesis_chapter6_robustness(fast: bool = False) -> Dict[str, List[TrainingConfig]]:
        """
        Preset for Chapter 6 robustness experiments.
        Includes: N-1, comm stress, sensor robustness, unseen faults.
        """
        base = create_experiment_matrix(
            systems=["8bus", "14bus", "39bus"],
            methods=["MAPPO", "MAPPO-Comm", "MAPPO-Safe"],
            pv_levels=[0.0, 0.5, 1.0],
            fault_types=["3ph"],
            n_runs=5 if not fast else 2,
            fast=fast,
        )
        return {
            "base": base,
            "fig14_n1": create_fig14_n1_contingency_matrix(base, line_indices=list(range(6))),
            "fig13_comm": create_fig13_comm_stress_matrix(base),
            "fig24_sensor": create_fig24_sensor_robustness_matrix(base),
            "fig23_faults": create_fig23_unseen_fault_matrix(base),
        }

    @staticmethod
    def thesis_chapter7_transfer(fast: bool = False) -> Dict[str, List[TrainingConfig]]:
        """
        Preset for Chapter 7 transfer learning experiments.
        Includes: zero-shot transfer, compression, adversarial.
        """
        base = create_experiment_matrix(
            systems=["8bus"],
            methods=["MAPPO", "MAPPO-Safe-EWC"],
            pv_levels=[0.5],
            fault_types=["3ph"],
            n_runs=5 if not fast else 2,
            fast=fast,
        )
        return {
            "base": base,
            "fig25_transfer": create_fig25_zero_shot_matrix(base, train_systems=["8bus"], eval_systems=["14bus", "39bus"]),
            "fig53_compression": create_fig53_compression_matrix(base),
            "fig54_adversarial": create_fig54_adversarial_matrix(base),
        }

    @staticmethod
    def ieee_paper_experiments(fast: bool = True) -> Dict[str, List[TrainingConfig]]:
        """
        Preset for IEEE paper submission experiments.
        Focused subset with statistical significance.
        """
        base = create_experiment_matrix(
            systems=["8bus", "14bus", "39bus"],
            methods=["GA-SQP", "MAPPO", "MAPPO-Comm", "MAPPO-Safe", "MAPPO-Safe-EWC"],
            pv_levels=[0.0, 0.5, 1.0],
            fault_types=["3ph", "LL"],
            n_runs=30 if not fast else 5,
            fast=fast,
        )
        return {
            "base": base,
            "fig14_n1": create_fig14_n1_contingency_matrix(base, line_indices=[0, 1, 2]),
            "fig13_comm": create_fig13_comm_stress_matrix(base, stress_levels=[0.0, 0.5]),
            "fig25_transfer": create_fig25_zero_shot_matrix(base, eval_systems=["14bus"]),
        }

    @staticmethod
    def quick_demo() -> List[TrainingConfig]:
        """Minimal demo for testing the pipeline."""
        return create_experiment_matrix(
            systems=["8bus"],
            methods=["GA-SQP", "JAYA"],
            pv_levels=[0.0, 1.0],
            fault_types=["3ph"],
            n_runs=1,
            fast=True,
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  PART 8: PROGRESS TRACKING & LOGGING UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

class ExperimentLogger:
    """Structured logging for long-running experiments."""

    def __init__(self, log_dir: str = "outputs/logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.start_time = time.time()
        self.events = []

    def log_event(self, event_type: str, config: TrainingConfig = None, message: str = ""):
        """Log an experiment event."""
        entry = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "elapsed": time.time() - self.start_time,
            "type": event_type,
            "message": message,
        }
        if config:
            entry["config_key"] = f"{config.method}_{config.system}_PV{config.pv_level}"
        self.events.append(entry)

        # Write to file immediately
        log_file = os.path.join(self.log_dir, "experiment.log")
        with open(log_file, 'a') as f:
            f.write(f"[{entry['timestamp']}] ({entry['elapsed']:.1f}s) {event_type}: {message}")

    def log_result(self, result: TrainingResult):
        """Log a completed result."""
        self.log_event(
            "RESULT",
            result.config,
            f"OF={result.best_of:.4f}, CTI_ok={result.cti_satisfied}, "
            f"time={result.elapsed_seconds:.1f}s"
        )

    def log_error(self, config: TrainingConfig, error: Exception):
        """Log an error."""
        self.log_event("ERROR", config, str(error))

    def get_summary(self) -> Dict:
        """Get experiment summary."""
        total = len([e for e in self.events if e["type"] == "RESULT"])
        errors = len([e for e in self.events if e["type"] == "ERROR"])
        return {
            "total_results": total,
            "errors": errors,
            "elapsed_hours": (time.time() - self.start_time) / 3600,
            "events": self.events,
        }


# ═══════════════════════════════════════════════════════════════════════════════
#  PART 9: RESUME & RECOVERY UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

class ExperimentRecovery:
    """Utilities for recovering from interrupted experiments."""

    @staticmethod
    def find_incomplete_runs(checkpoint_dir: str = "outputs/checkpoints") -> List[str]:
        """Find configuration keys that were interrupted."""
        checkpoint_file = os.path.join(checkpoint_dir, "pipeline_checkpoint.json")
        if not os.path.exists(checkpoint_file):
            return []

        with open(checkpoint_file, 'r') as f:
            checkpoint = json.load(f)

        completed = set(checkpoint.get("completed_configs", []))

        # If we have a results file, cross-check
        results_file = os.path.join("outputs/results", "phase5_results.json")
        if os.path.exists(results_file):
            with open(results_file, 'r') as f:
                results = json.load(f)
            result_keys = {r["config_key"] for r in results if "config_key" in r}
            # Find keys in completed but not in results (interrupted after checkpoint)
            interrupted = completed - result_keys
            return list(interrupted)

        return []

    @staticmethod
    def merge_result_files(results_dir: str = "outputs/results") -> List[Dict]:
        """Merge all figure-specific JSON files into one master file."""
        all_results = []

        for filename in os.listdir(results_dir):
            if filename.endswith(".json") and filename != "phase5_results.json":
                filepath = os.path.join(results_dir, filename)
                with open(filepath, 'r') as f:
                    data = json.load(f)
                all_results.extend(data)

        # Deduplicate by config key
        seen = set()
        unique = []
        for r in all_results:
            key = r.get("config_key", json.dumps(r.get("config", {}), sort_keys=True))
            if key not in seen:
                seen.add(key)
                unique.append(r)

        output_file = os.path.join(results_dir, "merged_results.json")
        with open(output_file, 'w') as f:
            json.dump(unique, f, indent=2)

        print(f"Merged {len(unique)} unique results into {output_file}")
        return unique


# ═══════════════════════════════════════════════════════════════════════════════
#  PART 10: QUICK START EXAMPLES
# ═══════════════════════════════════════════════════════════════════════════════

def example_1_basic_usage():
    """Example 1: Run a basic experiment matrix."""
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Basic Usage")
    print("=" * 70)

    configs = ExperimentPresets.quick_demo()
    pipeline = TrainingPipeline(configs, n_workers=1)
    results = pipeline.run_all()
    pipeline.save_results("outputs/results/demo.json")

    print(f"Completed {len(results)} runs")
    best = min(results, key=lambda r: r.best_of)
    print(f"Best OF: {best.best_of:.4f} ({best.config.method} on {best.config.system})")


def example_2_full_suite():
    """Example 2: Run full experiment suite with figure-specific evaluations."""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Full Suite with Figures")
    print("=" * 70)

    suite = create_full_experiment_suite(
        systems=["8bus", "14bus", "39bus"],
        methods=["MAPPO", "MAPPO-Comm", "MAPPO-Safe"],
        pv_levels=[0.0, 0.5, 1.0],
        figures=["base", "13", "14", "22", "24", "25"],
        fast=True,
    )

    all_configs = []
    for configs in suite.values():
        all_configs.extend(configs)

    pipeline = TrainingPipeline(all_configs, n_workers=2)
    results = pipeline.run_all()

    # Save organized by figure
    pipeline.save_results_by_figure()

    # Analyze
    analyzer = ResultAnalyzer(results)
    print("\nMethod Comparison:")
    comparison = analyzer.compare_methods()
    for method, stats in comparison.items():
        print(f"  {method}: {stats['mean']:.4f} +- {stats['std']:.4f}")


def example_3_thesis_chapter():
    """Example 3: Run experiments for a specific thesis chapter."""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Thesis Chapter 6 - Robustness")
    print("=" * 70)

    suite = ExperimentPresets.thesis_chapter6_robustness(fast=True)

    all_configs = []
    for configs in suite.values():
        all_configs.extend(configs)

    pipeline = TrainingPipeline(all_configs, n_workers=2)
    results = pipeline.run_all()

    # Generate LaTeX table
    analyzer = ResultAnalyzer(results)
    analyzer.export_latex_table("outputs/results/chapter6_table.tex")

    # Figure summaries
    for fig in ["fig13", "fig14", "fig24", "fig23"]:
        summary = analyzer.figure_summary(fig)
        print(f"\n{fig} summary: {len(summary)} groups")


def example_4_resume_interrupted():
    """Example 4: Resume an interrupted experiment."""
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Resume Interrupted Experiment")
    print("=" * 70)

    # Check for incomplete runs
    interrupted = ExperimentRecovery.find_incomplete_runs()
    if interrupted:
        print(f"Found {len(interrupted)} interrupted runs")
        # Re-run with same configs - checkpoint will skip completed ones
        configs = ExperimentPresets.quick_demo()
        pipeline = TrainingPipeline(configs, n_workers=1)
        results = pipeline.run_all()  # Automatically resumes
    else:
        print("No interrupted runs found")


def example_5_custom_figure_sweep():
    """Example 5: Custom parameter sweep for a specific figure."""
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Custom Figure Sweep (Fig 22 - Bandwidth)")
    print("=" * 70)

    base = create_experiment_matrix(
        systems=["8bus"],
        methods=["MAPPO-Comm"],
        pv_levels=[0.5],
        n_runs=3,
        fast=True,
    )

    # Custom sweep: test extreme compression
    configs = create_fig22_bandwidth_matrix(
        base,
        gnn_depths=[1, 2, 3],
        compression_ratios=[1.0, 0.5, 0.1],
        message_dims=[2, 4, 8, 16],
    )

    pipeline = TrainingPipeline(configs, n_workers=2)
    results = pipeline.run_all()

    # Analyze bandwidth vs performance trade-off
    analyzer = ResultAnalyzer(results)
    summary = analyzer.figure_summary("fig22")
    for ratio, stats in sorted(summary.items()):
        print(f"  Compression {ratio}: OF={stats['mean']:.4f} +- {stats['std']:.4f}")


# ═══════════════════════════════════════════════════════════════════════════════
#  UPDATED __main__ WITH ALL EXAMPLES
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    # Parse command-line argument for example selection
    example_map = {
        "1": example_1_basic_usage,
        "2": example_2_full_suite,
        "3": example_3_thesis_chapter,
        "4": example_4_resume_interrupted,
        "5": example_5_custom_figure_sweep,
    }

    if len(sys.argv) > 1 and sys.argv[1] in example_map:
        example_map[sys.argv[1]]()
    else:
        print("Usage: python training_pipeline.py <example_number>")
        print("Examples:")
        print("  1 - Basic usage (quick demo)")
        print("  2 - Full suite with figures")
        print("  3 - Thesis Chapter 6 robustness")
        print("  4 - Resume interrupted experiment")
        print("  5 - Custom figure sweep (bandwidth)")
        print("\nRunning default: Example 1 (Basic Usage)")
        example_1_basic_usage()