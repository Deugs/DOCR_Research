#!/usr/bin/env python3
"""
core_eval_bridge.py — Bridge between core training runs and training_pipeline evals
═══════════════════════════════════════════════════════════════════════════════════════
Adapts the core run workflow (UnifiedTrainingLogger, cache-based) to use
real evaluation methods from training_pipeline.py.

This module:
1. Loads a trained policy from Core A cache (or trains one if missing)
2. Wraps it into a TrainingResult-compatible object
3. Calls training_pipeline.py's _eval_* methods with proper context
4. Returns results in the format expected by post-processing

Usage:
    from core_eval_bridge import CoreEvalBridge
    bridge = CoreEvalBridge(seed=42, fast_mode=False)
    results = bridge.run_evaluations(fig_nums=[13, 20, 22, 24, 54])
"""

import os
import sys
import json
import time
import warnings
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field

# ──────────────────────────────────────────────────────────────────
# Add parent dir to path for imports
# ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.absolute()
sys.path.insert(0, str(BASE_DIR))

# ──────────────────────────────────────────────────────────────────
# Import training_pipeline components
# ──────────────────────────────────────────────────────────────────
try:
    from experiments.training_pipeline import (
        TrainingConfig, TrainingResult, TrainingPipeline,
        create_experiment_matrix
    )
    TP_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import training_pipeline: {e}")
    TP_AVAILABLE = False

# ──────────────────────────────────────────────────────────────────
# Import core training components
# ──────────────────────────────────────────────────────────────────
try:
    from docl_config import DOCRConfig
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False

# ──────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger(__name__)


def _json_fallback(obj):
    """json.dump default= hook for values TrainingResult attaches (numpy
    arrays/scalars). `default=str` on an ndarray previously called
    numpy's own str(), which whitespace-separates rows and truncates
    large arrays with '...' — the cached JSON looked fine but the array
    data (e.g. fig48 SHAP values, fig49 t-SNE embeddings) was silently
    unusable/lossy on reload. tolist()/item() round-trip exactly."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer, np.bool_)):
        return obj.item()
    return str(obj)


# ═══════════════════════════════════════════════════════════════════════════════
#  BRIDGE RESULT WRAPPER
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class BridgeResult:
    """Wrapper that converts training_pipeline results to core-run-compatible format."""
    fig_num: int
    raw_data: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_core_format(self) -> Dict:
        """Convert to the dictionary format expected by post-processing."""
        return {
            "fig_num": self.fig_num,
            "data": self.raw_data,
            "metadata": self.metadata,
            "timestamp": time.time(),
        }


# ═══════════════════════════════════════════════════════════════════════════════
#  CORE EVAL BRIDGE
# ═══════════════════════════════════════════════════════════════════════════════

class CoreEvalBridge:
    """
    Bridges core training runs with training_pipeline evaluation methods.
    
    Flow:
        1. Train or load base policy (MAPPO-Safe-EWC on 8-bus, 100% PV)
        2. For each requested figure:
           a. Create TrainingConfig with figure-specific flags
           b. Call training_pipeline._eval_* method
           c. Convert result to core-run-compatible format
        3. Cache results for post-processing
    """

    def __init__(
        self,
        seed: int = 42,
        fast_mode: bool = False,
        system: str = "8bus",
        pv_level: float = 0.6,
        fault_type: str = "3ph",
        cache_dir: Optional[Path] = None,
        allow_synthetic_fallback: bool = False,
    ):
        self.seed = seed
        self.fast_mode = fast_mode
        self.system = system
        self.pv_level = pv_level
        self.fault_type = fault_type
        self.cache_dir = cache_dir or (BASE_DIR / "results" / "core_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Default False: a failed real evaluation should raise, not
        # silently substitute fabricated numbers. See run_evaluation().
        self.allow_synthetic_fallback = allow_synthetic_fallback

        self._base_result: Optional[TrainingResult] = None
        self._net = None
        self._pipeline: Optional[Any] = None
        
        if not TP_AVAILABLE:
            raise RuntimeError("training_pipeline.py not available. Cannot run real evaluations.")

    def _get_network_factory(self):
        """
        Resolve self.system to the correct NetworkData factory.

        Previously _train_base_policy() and _get_net() both hardcoded
        `from network.network_data import get_8bus`, so passing
        system="30bus" or system="39bus" silently had no effect — every
        evaluation ran on the 8-bus network regardless of what was
        requested, while still being LABELLED with the requested system
        name in TrainingConfig. This made every result for non-8-bus
        systems mislabeled rather than simply missing.
        """
        from network.network_data import NETWORKS
        if self.system not in NETWORKS:
            raise ValueError(
                f"Unknown system '{self.system}'. Available: {sorted(NETWORKS.keys())}"
            )
        return NETWORKS[self.system]

    # ── Base Policy Training ──

    def _train_base_policy(self) -> TrainingResult:
        """
        Train MAPPO-Safe-EWC on 8-bus, 100% PV to get a real policy.

        Previous version looped over episodes calling only
        `trainer.actor.sample()` and recording the best (TMS, PS) ever
        observed — it never called `trainer.train()` or any update
        method, so the actor's weights were frozen at random
        initialization for the entire run. The "trained" result was
        really just the best output of a fixed random policy across
        N episodes: random search, not MARL, despite MAPPOMARLTrainer
        existing specifically to perform ES-based parameter updates
        (see marl/marl_agents.py: _update_actor_es / _update_critic_es,
        invoked inside trainer.train()).

        Fixed version actually calls trainer.train(total_steps=...), then
        persists the actor's learned weight vector (actor.get_all_params())
        onto the TrainingResult so every downstream _eval_* method can
        reconstruct and use the REAL trained policy, not random noise or
        a frozen actor.
        """
        logger.info(f"Training base policy (MAPPO-Safe-EWC, {self.system}, "
                    f"{int(self.pv_level*100)}% PV)...")

        net_factory = self._get_network_factory()
        from marl.marl_env import DynamicDOCREnv
        from marl.marl_agents import MAPPOMARLTrainer
        from marl.safe_rl import create_safe_rl_env, SafeRLConfig
        from marl.continual_trainer import create_ewc_trainer, EWCConfig
        from network.relay_physics import check_constraints, objective_function

        net = net_factory()
        if self.pv_level > 0 and hasattr(net, 'apply_pv_penetration'):
            net = net.apply_pv_penetration(self.pv_level)

        env = DynamicDOCREnv(
            net, seed=self.seed, use_pandapower=False,
            fault_type=self.fault_type, curriculum_stage=5
        )

        # total_steps here means environment steps consumed by
        # trainer.train(), mirroring the step-budget convention used by
        # every exp_fig*.py script in marl/experiments/. fast_mode trims
        # this for iteration speed, matching the prior n_episodes scaling.
        total_steps = 5000 if self.fast_mode else 50000

        # "Safe"/"EWC" were previously dead post-hoc attribute sets
        # (trainer.use_safety/use_ewc are not real MAPPOMARLTrainer
        # attributes — the hasattr guards were always False), so this
        # base policy — which every downstream Core G/H/I/J/K evaluation
        # is built on — trained as plain MAPPO+GNN despite being labeled
        # "MAPPO-Safe-EWC". Safe RL is a real env wrapper and must wrap
        # env before the trainer is constructed.
        safe_env = create_safe_rl_env(env, SafeRLConfig(
            project_actions=True, enforce_cti=True,
            lambda_init=1.0, lambda_lr=0.01, rho_penalty=10.0,
            use_pid=True, adaptive_rho=True,
        ))
        trainer = MAPPOMARLTrainer(safe_env, use_comm=True, seed=self.seed)
        ewc_trainer = create_ewc_trainer(trainer, EWCConfig())

        # This is the actual fix: previously this call never happened.
        # trainer.train() runs the ES rollout + _update_actor_es +
        # _update_critic_es loop, genuinely moving actor.net's weights
        # away from their random initialization.
        history_obj = trainer.train(total_steps=total_steps, verbose=self.fast_mode is False)
        history = list(history_obj) if history_obj is not None else []

        # Register this task with EWC so the apparatus genuinely runs
        # (n_tasks_learned reflects reality) rather than staying entirely
        # inert. A lone task has nothing to protect against forgetting —
        # expected, not a bug — but the machinery should reflect that it
        # actually ran (same rationale as run_core_training.py's Core A).
        ewc_trainer.ewc_regularizer.register_task(
            f"{self.system}_{int(self.pv_level*100)}pv", trainer.actor.get_all_params())

        # Evaluate the TRAINED (post-update) policy to find its best
        # observed operating point — this is now a real evaluation of a
        # learned policy, not a random-search artifact. Uses safe_env
        # (not the raw env) so action projection and the Lagrangian CTI
        # penalty are exercised during evaluation too, matching training.
        rng = np.random.default_rng(self.seed + 999)
        n_eval_episodes = 20 if self.fast_mode else 100
        best_of = float('inf')
        best_tms, best_ps = None, None

        for ep in range(n_eval_episodes):
            obs = safe_env.reset(seed=self.seed + 10_000 + ep)
            done = False
            while not done:
                actions = np.zeros((env.n_agents, env.ACT_DIM))
                if trainer.use_comm and trainer.comm_graph is not None:
                    policy_obs, _, _ = trainer.comm_graph.forward(obs, rng)
                else:
                    policy_obs = obs
                for i in range(env.n_agents):
                    a, _ = trainer.actor.sample(policy_obs[i], rng)
                    actions[i] = a
                obs, reward, done, info = safe_env.step(actions)

            TMS, PS = safe_env.get_state()
            try:
                of_val, _, _ = objective_function(net, TMS, PS, fault_type=self.fault_type)
                if of_val < best_of:
                    best_of = of_val
                    best_tms = TMS.copy()
                    best_ps = PS.copy()
                if not history or ep > 0:
                    history.append(float(of_val))
            except Exception:
                pass

        if best_tms is None:
            # Defensive fallback only — should not trigger in practice
            # since n_eval_episodes >= 1, but avoids a hard crash if the
            # objective function raised on every episode.
            best_tms, best_ps = env.get_state()
            best_of = float(objective_function(net, best_tms, best_ps, fault_type=self.fault_type)[0])

        constraints = check_constraints(net, best_tms, best_ps, fault_type=self.fault_type)

        actor_params = trainer.actor.get_all_params()

        # Persist the trained communication module too, if one exists.
        # Without this, _get_actor_for_eval has no way to reconstruct the
        # GNN encoder/attention/decoder the actor was actually trained
        # against, and any downstream eval that feeds raw (rather than
        # comm-enhanced) observations into the actor would silently use
        # the wrong input dimensionality and content.
        use_comm = bool(getattr(trainer, 'use_comm', False) and trainer.comm_graph is not None)
        comm_graph_params = trainer.comm_graph.get_all_params() if use_comm else None
        comm_msg_dim = getattr(trainer.comm_graph, 'msg_dim', 8) if use_comm else 8

        # Create TrainingResult-compatible object
        result = TrainingResult(
            config=TrainingConfig(
                method="MAPPO-Safe-EWC",
                system=self.system,
                pv_level=self.pv_level,
                fault_type=self.fault_type,
            ),
            run_id=0,
            best_of=best_of,
            best_tms=best_tms,
            best_ps=best_ps,
            cti_satisfied=constraints["cti_satisfied"],
            cti_violations=constraints["cti_violations"],
            cti_min=constraints["cti_min"],
            elapsed_seconds=0.0,
            history=history,
            actor_params=actor_params,
            obs_dim=env.OBS_DIM,
            act_dim=env.ACT_DIM,
            actor_hidden=64,
            use_comm=use_comm,
            comm_graph_params=comm_graph_params,
            comm_n_agents=env.n_agents,
            comm_msg_dim=comm_msg_dim,
        )

        self._base_result = result
        self._net = net
        logger.info(
            f"Base policy trained ({total_steps} steps). "
            f"Best OF={best_of:.4f}, CTI_ok={constraints['cti_satisfied']}, "
            f"actor_params persisted ({actor_params.size} weights)."
        )
        return result


    def _get_base_result(self) -> TrainingResult:
        """Get or train base policy."""
        if self._base_result is None:
            self._base_result = self._train_base_policy()
        return self._base_result

    def _get_net(self):
        """Get or create network."""
        if self._net is None:
            net_factory = self._get_network_factory()
            self._net = net_factory()
            if self.pv_level > 0 and hasattr(self._net, 'apply_pv_penetration'):
                self._net = self._net.apply_pv_penetration(self.pv_level)
        return self._net

    # ── Eval Method Dispatch ──

    def _get_pipeline(self) -> TrainingPipeline:
        """Get or create pipeline instance for eval methods."""
        if self._pipeline is None:
            # Create minimal pipeline for eval methods
            dummy_configs = [TrainingConfig(
                method="MAPPO-Safe-EWC",
                system=self.system,
                pv_level=self.pv_level,
                fault_type=self.fault_type,
            )]
            self._pipeline = TrainingPipeline(dummy_configs, n_workers=1, verbose=False)
        return self._pipeline

    def run_evaluation(self, fig_num: int, **eval_kwargs) -> BridgeResult:
        """
        Run a single figure evaluation using real methods.
        
        Args:
            fig_num: Figure number (13, 14, 20, 22, 23, 24, 25, 36, 44, 45, 46, 47, 48, 49, 50, 53, 54, 58)
            **eval_kwargs: Additional arguments for the eval method
        
        Returns:
            BridgeResult with real evaluation data
        """
        result = self._get_base_result()
        net = self._get_net()
        pipeline = self._get_pipeline()
        
        # Map figure number to eval method and config builder
        eval_map = {
            13: ("_eval_comm_stress", self._build_config_fig13),
            14: ("_eval_n1_contingency", self._build_config_fig14),
            20: ("_eval_byzantine", self._build_config_fig20),
            22: ("_eval_bandwidth", self._build_config_fig22),
            23: ("_eval_unseen_faults", self._build_config_fig23),
            24: ("_eval_sensor_robustness", self._build_config_fig24),
            25: ("_eval_zero_shot_transfer", self._build_config_fig25),
            36: ("_eval_action_space_grid", self._build_config_fig36),
            44: ("_eval_fault_contributions", self._build_config_fig44),
            45: ("_eval_daily_pv", self._build_config_fig45),
            46: ("_eval_relay_curves", self._build_config_fig46),
            47: ("_eval_sympathetic_tripping", self._build_config_fig47),
            48: ("_eval_shap", self._build_config_fig48),
            49: ("_eval_tsne", self._build_config_fig49),
            50: ("_eval_critical_pairs", self._build_config_fig50),
            53: ("_eval_compression", self._build_config_fig53),
            54: ("_eval_adversarial", self._build_config_fig54),
            58: ("_eval_seasonal", self._build_config_fig58),
        }
        
        if fig_num not in eval_map:
            raise ValueError(f"Figure {fig_num} not supported. Supported: {list(eval_map.keys())}")
        
        method_name, config_builder = eval_map[fig_num]
        config = config_builder(**eval_kwargs)
        
        logger.info(f"[Fig {fig_num}] Running real evaluation: {method_name}")
        t0 = time.time()
        
        try:
            # Call the eval method
            eval_method = getattr(pipeline, method_name)
            updated_result = eval_method(config, net, result)
            
            # Extract figure-specific data from updated_result
            raw_data = self._extract_fig_data(fig_num, updated_result)
            
            metadata = {
                "method": method_name,
                "duration": time.time() - t0,
                "config": config.to_dict(),
                "base_of": result.best_of,
            }
            
            bridge_result = BridgeResult(fig_num=fig_num, raw_data=raw_data, metadata=metadata)
            
            # Cache the result
            self._cache_result(fig_num, bridge_result)
            
            logger.info(f"[Fig {fig_num}] ✓ Real evaluation complete in {metadata['duration']:.2f}s")
            return bridge_result
            
        except Exception as e:
            logger.error(f"[Fig {fig_num}] Real evaluation failed: {e}", exc_info=True)
            if not self.allow_synthetic_fallback:
                # Re-raise by default. A silent substitution of fabricated
                # numbers for a genuine failure is worse than a crash: it
                # produces a figure that looks identical to a real result
                # while reporting nothing about the actual trained policy.
                # Set allow_synthetic_fallback=True explicitly (e.g. for
                # CI smoke tests where only file-shape matters) to opt
                # back into the old behaviour.
                raise
            warnings.warn(
                f"Real eval failed for Fig {fig_num}, using SYNTHETIC fallback. "
                f"This result is NOT derived from the trained policy and must "
                f"not be reported as an experimental finding."
            )
            return self._synthetic_fallback(fig_num, result)

    # ── Config Builders ──

    def _build_config_fig13(self, **kwargs) -> TrainingConfig:
        """Config for Fig 13: Communication Stress."""
        return TrainingConfig(
            method="MAPPO-Safe-EWC", system=self.system, pv_level=self.pv_level,
            fault_type=self.fault_type,
            comm_stress_level=kwargs.get("stress_level", 0.5),
            packet_loss_rate=kwargs.get("packet_loss", 0.1),
            latency_ms=kwargs.get("latency_ms", 100),
            quantization_bits=kwargs.get("quantization", 32),
            use_comm=True, use_safe_rl=True, use_ewc=True,
        )

    def _build_config_fig14(self, **kwargs) -> TrainingConfig:
        """Config for Fig 14: N-1 Contingency."""
        return TrainingConfig(
            method="MAPPO-Safe-EWC", system=self.system, pv_level=self.pv_level,
            fault_type=self.fault_type,
            n1_line_outage=kwargs.get("line_outage", 0),
            n1_recovery_eval=kwargs.get("recovery", True),
        )

    def _build_config_fig20(self, **kwargs) -> TrainingConfig:
        """Config for Fig 20: Byzantine Agents."""
        return TrainingConfig(
            method="MAPPO-Safe-EWC", system=self.system, pv_level=self.pv_level,
            fault_type=self.fault_type,
            byzantine_agents=kwargs.get("byzantine_agents", [1]),
            byzantine_strategy=kwargs.get("strategy", "random"),
            use_comm=True,
        )

    def _build_config_fig22(self, **kwargs) -> TrainingConfig:
        """Config for Fig 22: Bandwidth vs Performance."""
        return TrainingConfig(
            method="MAPPO-Safe-EWC", system=self.system, pv_level=self.pv_level,
            fault_type=self.fault_type,
            gnn_depth=kwargs.get("gnn_depth", 2),
            compression_ratio=kwargs.get("compression", 1.0),
            message_dim=kwargs.get("message_dim", 8),
            use_comm=True,
        )

    def _build_config_fig23(self, **kwargs) -> TrainingConfig:
        """Config for Fig 23: Unseen Faults."""
        return TrainingConfig(
            method="MAPPO-Safe-EWC", system=self.system, pv_level=self.pv_level,
            fault_type=self.fault_type,
            fault_impedance_ohm=kwargs.get("impedance", 1.0),
            evolving_fault=kwargs.get("evolving", False),
            unseen_fault_eval=True,
        )

    def _build_config_fig24(self, **kwargs) -> TrainingConfig:
        """Config for Fig 24: Sensor Robustness."""
        return TrainingConfig(
            method="MAPPO-Safe-EWC", system=self.system, pv_level=self.pv_level,
            fault_type=self.fault_type,
            sensor_noise_std=kwargs.get("noise_std", 0.0),
            missing_sensor_rate=kwargs.get("missing_rate", 0.0),
            sensor_bias=kwargs.get("bias", 0.0),
        )

    def _build_config_fig25(self, **kwargs) -> TrainingConfig:
        """Config for Fig 25: Zero-Shot Transfer."""
        return TrainingConfig(
            method="MAPPO-Safe-EWC", system=self.system, pv_level=self.pv_level,
            fault_type=self.fault_type,
            train_system=self.system,
            eval_system=kwargs.get("eval_system", "14bus"),
            freeze_transfer_weights=True,
        )

    def _build_config_fig36(self, **kwargs) -> TrainingConfig:
        """Config for Fig 36: Action Space Geometry."""
        return TrainingConfig(
            method="MAPPO-Safe-EWC", system=self.system, pv_level=self.pv_level,
            fault_type=self.fault_type,
            action_space_grid=True,
            grid_tds_range=kwargs.get("tds_range", (0.5, 2.0)),
            grid_psm_range=kwargs.get("psm_range", (1.0, 5.0)),
            grid_resolution=kwargs.get("resolution", 20),
        )

    def _build_config_fig44(self, **kwargs) -> TrainingConfig:
        """Config for Fig 44: Fault Current Contribution."""
        return TrainingConfig(
            method="MAPPO-Safe-EWC", system=self.system, pv_level=self.pv_level,
            fault_type=self.fault_type,
            run_shortcircuit_study=True,
        )

    def _build_config_fig45(self, **kwargs) -> TrainingConfig:
        """Config for Fig 45: Daily PV Profile."""
        return TrainingConfig(
            method="MAPPO-Safe-EWC", system=self.system, pv_level=self.pv_level,
            fault_type=self.fault_type,
            daily_profile=True,
            irradiance_profile=kwargs.get("irradiance", None),
        )

    def _build_config_fig46(self, **kwargs) -> TrainingConfig:
        """Config for Fig 46: Relay Operating Curves."""
        return TrainingConfig(
            method="MAPPO-Safe-EWC", system=self.system, pv_level=self.pv_level,
            fault_type=self.fault_type,
            relay_curve_eval=True,
            fault_current_range=kwargs.get("current_range", (100.0, 10000.0)),
        )

    def _build_config_fig47(self, **kwargs) -> TrainingConfig:
        """Config for Fig 47: Sympathetic Tripping."""
        return TrainingConfig(
            method="MAPPO-Safe-EWC", system=self.system, pv_level=self.pv_level,
            fault_type=self.fault_type,
            sympathetic_tripping_eval=True,
        )

    def _build_config_fig48(self, **kwargs) -> TrainingConfig:
        """Config for Fig 48: SHAP Feature Importance."""
        return TrainingConfig(
            method="MAPPO-Safe-EWC", system=self.system, pv_level=self.pv_level,
            fault_type=self.fault_type,
            shap_eval=True,
            shap_n_background=kwargs.get("n_background", 100),
        )

    def _build_config_fig49(self, **kwargs) -> TrainingConfig:
        """Config for Fig 49: Decision Boundary (t-SNE)."""
        return TrainingConfig(
            method="MAPPO-Safe-EWC", system=self.system, pv_level=self.pv_level,
            fault_type=self.fault_type,
            tsne_eval=True,
            tsne_n_samples=kwargs.get("n_samples", 500),
        )

    def _build_config_fig50(self, **kwargs) -> TrainingConfig:
        """Config for Fig 50: Critical P/B Pairs."""
        return TrainingConfig(
            method="MAPPO-Safe-EWC", system=self.system, pv_level=self.pv_level,
            fault_type=self.fault_type,
            critical_pair_eval=True,
            use_safe_rl=True,
        )

    def _build_config_fig53(self, **kwargs) -> TrainingConfig:
        """Config for Fig 53: Model Compression."""
        return TrainingConfig(
            method="MAPPO-Safe-EWC", system=self.system, pv_level=self.pv_level,
            fault_type=self.fault_type,
            compression_eval=True,
            quantize_bits=kwargs.get("quantize_bits", None),
            prune_ratio=kwargs.get("prune_ratio", 0.0),
        )

    def _build_config_fig54(self, **kwargs) -> TrainingConfig:
        """Config for Fig 54: Adversarial Robustness."""
        return TrainingConfig(
            method="MAPPO-Safe-EWC", system=self.system, pv_level=self.pv_level,
            fault_type=self.fault_type,
            adversarial_eval=True,
            adversarial_attack=kwargs.get("attack", "FGSM"),
            attack_epsilon=kwargs.get("epsilon", 0.01),
        )

    def _build_config_fig58(self, **kwargs) -> TrainingConfig:
        """Config for Fig 58: Seasonal Generalization."""
        return TrainingConfig(
            method="MAPPO-Safe-EWC", system=self.system, pv_level=self.pv_level,
            fault_type=self.fault_type,
            season=kwargs.get("season", "summer"),
            seasonal_temperature=kwargs.get("temperature", 35.0),
        )

    # ── Data Extraction ──

    def _extract_fig_data(self, fig_num: int, result: TrainingResult) -> Dict:
        """Extract figure-specific data from TrainingResult."""
        extractors = {
            13: lambda r: {"packet_loss": r.comm_stress_of, "latency": r.comm_stress_latency, "cti_ok": r.comm_stress_cti_ok},
            14: lambda r: {"n1_of": r.n1_of, "n1_cti_ok": r.n1_cti_ok, "recovery_of": r.n1_recovery_of, "line": r.n1_line_index},
            20: lambda r: {"byzantine_of": r.byzantine_of, "detected": r.byzantine_detected, "trust": r.byzantine_trust_scores},
            22: lambda r: {"bandwidth": r.bandwidth_kbps, "of": r.bandwidth_of, "params": r.compression_params},
            23: lambda r: {"unseen_of": r.unseen_fault_of, "cti_ok": r.unseen_fault_cti_ok, "impedance": r.fault_impedance},
            24: lambda r: {"noisy_of": r.noisy_of, "cti_ok": r.noisy_cti_ok, "snr": r.sensor_snr},
            25: lambda r: {"transfer_of": r.transfer_of, "cti_ok": r.transfer_cti_ok, "system": r.transfer_system},
            36: lambda r: r.action_space_grid_data or {},
            44: lambda r: r.fault_contributions or {},
            45: lambda r: {"tms": r.daily_tms_trajectory, "ps": r.daily_ps_trajectory, "of": r.daily_of_trajectory, "irradiance": r.daily_irradiance},
            46: lambda r: r.relay_curves or {},
            47: lambda r: {"sympathetic": r.sympathetic_trips, "traditional": r.traditional_trips},
            48: lambda r: {"shap": r.shap_values, "base": r.shap_base_value},
            49: lambda r: {"obs": r.tsne_observations, "actions": r.tsne_actions, "embeddings": r.tsne_embeddings},
            50: lambda r: r.critical_pairs or [],
            53: lambda r: {"of": r.compressed_of, "size_kb": r.compressed_size_kb, "drop": r.compression_accuracy_drop},
            54: lambda r: {"of": r.adversarial_of, "success": r.adversarial_success_rate, "norm": r.attack_perturbation_norm},
            58: lambda r: {"of": r.seasonal_of, "cti_ok": r.seasonal_cti_ok},
        }
        
        extractor = extractors.get(fig_num, lambda r: {})
        return extractor(result)

    # ── Synthetic Fallback ──

    def _synthetic_fallback(self, fig_num: int, result: TrainingResult) -> BridgeResult:
        """Generate synthetic data as fallback."""
        rng = np.random.default_rng(self.seed + fig_num)
        
        fallbacks = {
            13: {"packet_loss": {pl: max(0, 0.85 - pl * 0.8 + rng.normal(0, 0.05)) for pl in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]},
                 "latency": {lat: max(0, 0.85 - lat / 1000 * 0.3 + rng.normal(0, 0.05)) for lat in [0, 100, 200, 300, 400, 500]}},
            20: {"random": max(0, 0.85 - 0.15 + rng.normal(0, 0.05)),
                 "collusion": max(0, 0.85 - 0.25 + rng.normal(0, 0.05)),
                 "fdi": max(0, 0.85 - 0.20 + rng.normal(0, 0.05))},
            22: {d: {"performance": max(0, 0.85 - (d - 2) * 0.05), "bandwidth_mb": d * 12} for d in [1, 2, 3, 4]},
            23: {"bolted": 0.85, "high-impedance": 0.78, "evolving": 0.72, "simultaneous": 0.68},
            24: {"noise": {n: max(0, 0.85 - n * 2 + rng.normal(0, 0.03)) for n in [0.0, 0.05, 0.1, 0.15, 0.2]},
                 "bias": {b: max(0, 0.85 - b * 1.5 + rng.normal(0, 0.03)) for b in [0.0, 0.05, 0.1]},
                 "missing": {m: max(0, 0.85 - m * 0.5 + rng.normal(0, 0.03)) for m in [0.0, 0.1, 0.2]}},
            25: {"warm_start": {"performance": 0.75, "convergence_ep": 500},
                 "cold_start": {"performance": 0.20, "convergence_ep": 2000}},
            36: {"tds_values": [0.5, 1.0, 1.5, 2.0], "psm_values": [1.0, 2.0, 3.0, 4.0, 5.0],
                 "cti_grid": [[1, 1, 0, 0], [1, 1, 1, 0], [0, 1, 1, 1], [0, 0, 1, 1]]},
            44: {"PV": 35, "Grid": 45, "Synchronous": 20},
            45: {"hours": list(range(24)), "pv_generation": [max(0, np.sin(np.pi * h / 12) * 100) for h in range(24)],
                 "relay_response": [0.2 + 0.1 * (1 - max(0, np.sin(np.pi * h / 12))) for h in range(24)]},
            46: {"current_multiples": list(np.linspace(1, 20, 100)),
                 "operating_time": [0.5 * 0.14 / ((x / 2.0) ** 0.02 - 1) for x in np.linspace(1, 20, 100)],
                 "TDS": 0.5, "PSM": 2.0},
            47: {"cases_prevented": 15, "cases_total": 20, "prevention_rate": 0.75},
            48: {"features": ["V_local", "I_branch", "TMS", "PS", "PV_gen"], "shap": [0.1, 0.08, 0.05, 0.03, 0.02]},
            49: {"x": rng.standard_normal(500), "y": rng.standard_normal(500),
                 "labels": (rng.standard_normal(500) > 0).astype(int)},
            50: [{"primary": i, "backup": j, "lambda": 0.5, "violation_rate": 0.1} for i in range(1, 8) for j in range(i + 1, 8)],
            53: {"original": {"size_mb": 45, "performance": 0.85}, "int8": {"size_mb": 12, "performance": 0.84},
                 "pruning": {"size_mb": 22, "performance": 0.83}, "distillation": {"size_mb": 18, "performance": 0.82}},
            54: {"fgsm": {eps: max(0, 0.85 - eps * 3 + rng.normal(0, 0.03)) for eps in [0.01, 0.05, 0.1, 0.2]},
                 "pgd": {eps: max(0, 0.85 - eps * 4 + rng.normal(0, 0.03)) for eps in [0.01, 0.05, 0.1, 0.2]}},
            58: {"spring": {"performance": 0.83, "pv_variability": 0.3},
                 "summer": {"performance": 0.80, "pv_variability": 0.5},
                 "autumn": {"performance": 0.84, "pv_variability": 0.25},
                 "winter": {"performance": 0.82, "pv_variability": 0.15}},
        }
        
        raw_data = fallbacks.get(fig_num, {})
        metadata = {"fallback": True, "reason": "Real eval failed"}
        return BridgeResult(fig_num=fig_num, raw_data=raw_data, metadata=metadata)

    # ── Caching ──

    def _cache_result(self, fig_num: int, result: BridgeResult):
        """Cache result for post-processing."""
        cache_file = self.cache_dir / f"core_eval_fig{fig_num:02d}.json"
        with open(cache_file, 'w') as f:
            json.dump(result.to_core_format(), f, indent=2, default=_json_fallback)
        logger.info(f"  Cached to {cache_file}")

    def load_cached_result(self, fig_num: int) -> Optional[BridgeResult]:
        """Load cached result if available."""
        cache_file = self.cache_dir / f"core_eval_fig{fig_num:02d}.json"
        if cache_file.exists():
            with open(cache_file, 'r') as f:
                data = json.load(f)
            return BridgeResult(
                fig_num=data["fig_num"],
                raw_data=data["data"],
                metadata=data.get("metadata", {})
            )
        return None

    # ── Batch Evaluation ──

    def run_evaluations(self, fig_nums: List[int], **kwargs) -> Dict[int, BridgeResult]:
        """
        Run multiple figure evaluations.

        Failures are collected and reported at the end rather than
        silently papered over with synthetic data for each figure that
        happened to fail — a batch run that quietly downgrades half its
        figures to fabricated numbers is far more dangerous than one that
        reports clearly which figures did not produce real results.
        """
        results = {}
        failed: List[int] = []
        for fig_num in fig_nums:
            # Check cache first
            cached = self.load_cached_result(fig_num)
            if cached and not kwargs.get("force_recompute", False):
                logger.info(f"[Fig {fig_num}] Using cached result")
                results[fig_num] = cached
                continue

            try:
                results[fig_num] = self.run_evaluation(fig_num, **kwargs)
            except Exception as e:
                logger.error(f"[Fig {fig_num}] Evaluation failed: {e}", exc_info=True)
                failed.append(fig_num)
                if self.allow_synthetic_fallback:
                    results[fig_num] = self._synthetic_fallback(fig_num, self._get_base_result())
                # else: fig_num is simply absent from results, which the
                # caller must handle explicitly rather than unknowingly
                # receiving fabricated data.

        if failed:
            logger.error(
                f"{len(failed)} figure(s) FAILED real evaluation and were "
                f"{'replaced with synthetic fallback' if self.allow_synthetic_fallback else 'omitted'}: "
                f"{failed}"
            )

        return results


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run real evaluations for thesis figures")
    parser.add_argument("--figs", type=int, nargs="+", default=[13, 14, 20, 22, 23, 24, 25, 36, 44, 45, 46, 47, 48, 49, 50, 53, 54, 58],
                        help="Figure numbers to evaluate")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fast", action="store_true", help="Fast mode (fewer episodes)")
    parser.add_argument("--force", action="store_true", help="Force recompute (ignore cache)")
    args = parser.parse_args()
    
    bridge = CoreEvalBridge(seed=args.seed, fast_mode=args.fast)
    results = bridge.run_evaluations(args.figs, force_recompute=args.force)
    
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    for fig_num, result in sorted(results.items()):
        is_real = not result.metadata.get("fallback", False)
        status = "✓ REAL" if is_real else "✗ SYNTHETIC (fallback)"
        print(f"  Fig {fig_num:2d}: {status}")


if __name__ == "__main__":
    main()