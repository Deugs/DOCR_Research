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

Features:
- Parallel execution support (multiprocessing)
- Checkpoint/resume for long-running experiments
- Progress tracking and logging
- Automatic result aggregation
- Curriculum learning orchestration
"""

import numpy as np
import json
import os
import time
import pickle
from typing import List, Dict, Tuple, Optional, Callable, Any
from dataclasses import dataclass, asdict
from multiprocessing import Pool, cpu_count
import warnings


@dataclass
class TrainingConfig:
    """Configuration for a single training run."""
    method: str  # "GA-SQP", "IPPO", "MAPPO", "MAPPO-Comm", "MAPPO-Safe", "MAPPO-Safe-EWC",
                 # "HWOA", "HHO", "JAYA", "EEO", "Single-PPO"
    system: str  # "8bus", "30bus", "39bus"
    pv_level: float  # 0.0, 0.2, 0.4, 0.6, 0.8, 1.0
    fault_type: str  # "3ph", "LL", "1ph", "2ph-g"
    topology_changes: Optional[Dict] = None
    n_runs: int = 10  # Independent runs for statistical significance
    seed_offset: int = 0

    # Method-specific parameters
    ga_pop: int = 100
    ga_gen: int = 200
    meta_pop: int = 30
    meta_iter: int = 500
    marl_steps: int = 30000

    # Curriculum
    use_curriculum: bool = False
    curriculum_stage: int = 0

    # Safe RL / EWC
    use_safe_rl: bool = False
    use_ewc: bool = False
    use_comm: bool = False

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class TrainingResult:
    """Result from a single training run."""
    config: TrainingConfig
    run_id: int

    # Performance metrics
    best_of: float
    best_tms: np.ndarray
    best_ps: np.ndarray

    # Constraint metrics
    cti_satisfied: bool
    cti_violations: int
    cti_min: float

    # Timing
    elapsed_seconds: float

    # Convergence
    history: List[float]

    # Method-specific
    comm_bandwidth: float = 0.0
    safe_rl_stats: Optional[Dict] = None
    ewc_stats: Optional[Dict] = None

    def to_dict(self) -> Dict:
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
        return d


class TrainingPipeline:
    """
    Orchestrates large-scale training experiments.

    Usage:
        pipeline = TrainingPipeline(configs, n_workers=4)
        results = pipeline.run_all()
        pipeline.save_results("outputs/phase5_results.json")
    """

    def __init__(
        self,
        configs: List[TrainingConfig],
        n_workers: int = -1,
        checkpoint_dir: str = "outputs/checkpoints",
        verbose: bool = True,
    ):
        self.configs = configs
        self.n_workers = n_workers if n_workers > 0 else max(1, cpu_count() - 1)
        self.checkpoint_dir = checkpoint_dir
        self.verbose = verbose
        self.results: List[TrainingResult] = []
        self.completed_configs: set = set()

        os.makedirs(checkpoint_dir, exist_ok=True)

    def run_all(self) -> List[TrainingResult]:
        """Run all training configurations."""
        print(f"\n{'='*70}")
        print(f"  PHASE 5: Large-Scale Training Pipeline")
        print(f"{'='*70}")
        print(f"  Total configurations: {len(self.configs)}")
        print(f"  Parallel workers: {self.n_workers}")
        print(f"  Checkpoint dir: {self.checkpoint_dir}")
        print(f"{'='*70}")

        # Load checkpoint if exists
        self._load_checkpoint()

        # Filter out completed configs
        pending = [c for c in self.configs if self._config_key(c) not in self.completed_configs]
        print(f"  Pending configurations: {len(pending)}")
        print(f"  Already completed: {len(self.completed_configs)}")

        if not pending:
            print("  All configurations already completed!")
            return self.results

        # Run configurations
        if self.n_workers == 1:
            # Sequential execution
            for i, config in enumerate(pending):
                print(f"\n  [{i+1}/{len(pending)}] {config.method} on {config.system} "
                      f"(PV={config.pv_level}, fault={config.fault_type})")
                result = self._run_single_config(config)
                self.results.append(result)
                self.completed_configs.add(self._config_key(config))
                self._save_checkpoint()
        else:
            # Parallel execution
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

    def _run_single_config(self, config: TrainingConfig) -> TrainingResult:
        """Run a single training configuration."""
        t0 = time.perf_counter()

        # Import here to avoid serialization issues in multiprocessing
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

        # Run method
        result = self._dispatch_method(config, net)

        elapsed = time.perf_counter() - t0
        result.elapsed_seconds = elapsed

        return result

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

    def _run_ga_sqp(self, config: TrainingConfig, net) -> TrainingResult:
        """Run GA-SQP baseline."""
        from baselines.ga_sqp import run_ga_sqp

        res = run_ga_sqp(
            net,
            ga_pop=config.ga_pop,
            ga_gen=config.ga_gen,
            seed=config.seed_offset,
            verbose=False,
            fault_type=config.fault_type,
            pv_level=config.pv_level,
        )

        return TrainingResult(
            config=config,
            run_id=0,
            best_of=res["OF"],
            best_tms=res["TMS"],
            best_ps=res["PS"],
            cti_satisfied=res["constraints"]["cti_satisfied"],
            cti_violations=res["constraints"]["cti_violations"],
            cti_min=res["constraints"]["cti_min"],
            elapsed_seconds=0.0,
            history=res.get("history", [res["OF"]]),
        )

    def _run_metaheuristic(self, config: TrainingConfig, net) -> TrainingResult:
        """Run metaheuristic algorithm."""
        from baselines.metaheuristics import (
            run_hwoa, run_hho, run_jaya, run_eeo
        )

        runners = {
            "HWOA": run_hwoa,
            "HHO": run_hho,
            "JAYA": run_jaya,
            "EEO": run_eeo,
        }

        runner = runners[config.method]

        # Map generic param names to method-specific names
        if config.method == "HWOA":
            res = runner(net, n_whales=config.meta_pop, max_iter=config.meta_iter,
                        seed=config.seed_offset, verbose=False,
                        fault_type=config.fault_type, pv_level=config.pv_level, use_sqp=True)
        elif config.method == "HHO":
            res = runner(net, n_hawks=config.meta_pop, max_iter=config.meta_iter,
                        seed=config.seed_offset, verbose=False,
                        fault_type=config.fault_type, pv_level=config.pv_level, use_sqp=True)
        elif config.method == "JAYA":
            res = runner(net, n_pop=config.meta_pop, max_iter=config.meta_iter,
                        seed=config.seed_offset, verbose=False,
                        fault_type=config.fault_type, pv_level=config.pv_level, use_sqp=True)
        elif config.method == "EEO":
            res = runner(net, n_particles=config.meta_pop, max_iter=config.meta_iter,
                        seed=config.seed_offset, verbose=False,
                        fault_type=config.fault_type, pv_level=config.pv_level, use_sqp=True)

        return TrainingResult(
            config=config,
            run_id=0,
            best_of=res["OF"],
            best_tms=res["TMS"],
            best_ps=res["PS"],
            cti_satisfied=res["constraints"]["cti_satisfied"],
            cti_violations=res["constraints"]["cti_violations"],
            cti_min=res["constraints"]["cti_min"],
            elapsed_seconds=0.0,
            history=res.get("history", [res["OF"]]),
        )

    def _run_marl(self, config: TrainingConfig, net) -> TrainingResult:
        """Run MARL algorithm."""
        from marl.marl_env import DynamicDOCREnv
        from marl.marl_agents import MAPPOMARLTrainer
        from marl.safe_rl import create_safe_rl_env, SafeRLConfig
        from marl.continual_trainer import create_ewc_trainer, EWCConfig

        env = DynamicDOCREnv(
            net,
            action_scale=0.05,
            max_steps=300,
            seed=config.seed_offset,
            use_pandapower=False,
            fault_type=config.fault_type,
        )

        # Apply curriculum stage
        if config.use_curriculum:
            env.set_curriculum_stage(config.curriculum_stage)

        # Wrap with Safe RL if needed
        if config.use_safe_rl or config.method in ["MAPPO-Safe", "MAPPO-Safe-EWC"]:
            safe_config = SafeRLConfig(
                project_actions=True,
                enforce_cti=True,
                lambda_init=1.0,
                lambda_lr=0.01,
                rho_penalty=10.0,
                use_pid=True,
                adaptive_rho=True,
            )
            env = create_safe_rl_env(env, safe_config)

        # Create trainer
        trainer = MAPPOMARLTrainer(
            env,
            use_comm=config.use_comm or config.method == "MAPPO-Comm",
            hidden_actor=64,
            hidden_critic=128,
            lr_actor=0.02,
            lr_critic=0.01,
            clip_eps=0.2,
            gamma=0.99,
            lam=0.95,
            rollout_len=64,
            n_epochs=4,
            seed=config.seed_offset,
            curriculum=config.use_curriculum,
        )

        # Wrap with EWC if needed
        if config.use_ewc or config.method == "MAPPO-Safe-EWC":
            ewc_config = EWCConfig(
                lambda_ewc=100.0,
                fisher_samples=100,
                use_curriculum_warmstart=True,
                max_tasks=6,
            )
            trainer = create_ewc_trainer(trainer, ewc_config)

        # Train
        result = trainer.train(total_steps=config.marl_steps, verbose=False)

        # Extract metrics
        constraints = check_constraints(net, result["TMS"], result["PS"], fault_type=config.fault_type)

        return TrainingResult(
            config=config,
            run_id=0,
            best_of=result["OF"],
            best_tms=result["TMS"],
            best_ps=result["PS"],
            cti_satisfied=constraints["cti_satisfied"],
            cti_violations=constraints["cti_violations"],
            cti_min=constraints["cti_min"],
            elapsed_seconds=result["elapsed"],
            history=result.get("history", [result["OF"]]),
            comm_bandwidth=result.get("comm_bandwidth", 0.0),
            safe_rl_stats=result.get("safe_rl_stats"),
            ewc_stats=result.get("ewc_stats"),
        )

    def _run_single_ppo(self, config: TrainingConfig, net) -> TrainingResult:
        """Run Single-PPO baseline."""
        from marl.single_agent_ppo import run_single_ppo

        res = run_single_ppo(
            net,
            total_steps=config.marl_steps,
            seed=config.seed_offset,
            verbose=False,
            use_pandapower=False,
            fault_type=config.fault_type,
        )

        constraints = check_constraints(net, res["TMS"], res["PS"], fault_type=config.fault_type)

        return TrainingResult(
            config=config,
            run_id=0,
            best_of=res["OF"],
            best_tms=res["TMS"],
            best_ps=res["PS"],
            cti_satisfied=constraints["cti_satisfied"],
            cti_violations=constraints["cti_violations"],
            cti_min=constraints["cti_min"],
            elapsed_seconds=res.get("elapsed", 0.0),
            history=res.get("history", [res["OF"]]),
        )

    def _config_key(self, config: TrainingConfig) -> str:
        """Generate unique key for a configuration."""
        return f"{config.method}_{config.system}_PV{config.pv_level}_{config.fault_type}_r{config.seed_offset}"

    def _save_checkpoint(self):
        """Save checkpoint of completed configurations."""
        checkpoint = {
            "completed_configs": list(self.completed_configs),
            "n_results": len(self.results),
        }
        path = os.path.join(self.checkpoint_dir, "pipeline_checkpoint.json")
        with open(path, 'w') as f:
            json.dump(checkpoint, f)

    def _load_checkpoint(self):
        """Load checkpoint if exists."""
        path = os.path.join(self.checkpoint_dir, "pipeline_checkpoint.json")
        if os.path.exists(path):
            with open(path, 'r') as f:
                checkpoint = json.load(f)
            self.completed_configs = set(checkpoint.get("completed_configs", []))
            print(f"  Loaded checkpoint: {len(self.completed_configs)} completed")

    def save_results(self, filepath: str):
        """Save all results to JSON."""
        data = [r.to_dict() for r in self.results]
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"  Saved {len(self.results)} results to {filepath}")

    def aggregate_by_method(self) -> Dict[str, Dict]:
        """Aggregate results by method."""
        agg = {}
        for r in self.results:
            method = r.config.method
            if method not in agg:
                agg[method] = {
                    "count": 0,
                    "of_values": [],
                    "cti_ok_count": 0,
                    "elapsed_times": [],
                }
            agg[method]["count"] += 1
            agg[method]["of_values"].append(r.best_of)
            if r.cti_satisfied:
                agg[method]["cti_ok_count"] += 1
            agg[method]["elapsed_times"].append(r.elapsed_seconds)

        # Compute statistics
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

        return agg

    def aggregate_by_pv_level(self) -> Dict[float, Dict]:
        """Aggregate results by PV penetration level."""
        agg = {}
        for r in self.results:
            pv = r.config.pv_level
            if pv not in agg:
                agg[pv] = {
                    "count": 0,
                    "of_values": [],
                    "cti_ok_count": 0,
                }
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


def create_experiment_matrix(
    systems: List[str] = None,
    methods: List[str] = None,
    pv_levels: List[float] = None,
    fault_types: List[str] = None,
    n_runs: int = 10,
    fast: bool = False,
) -> List[TrainingConfig]:
    """
    Create a full experiment configuration matrix.

    Args:
        systems: Test systems (default: ["8bus", "30bus"])
        methods: Algorithms to test (default: all)
        pv_levels: PV penetration levels (default: [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        fault_types: Fault types (default: ["3ph"])
        n_runs: Independent runs per configuration
        fast: Reduced parameters for quick testing

    Returns:
        List of TrainingConfig objects
    """
    if systems is None:
        systems = ["8bus", "30bus"]
    if methods is None:
        methods = ["GA-SQP", "MAPPO", "MAPPO-Safe", "MAPPO-Safe-EWC", "HWOA", "JAYA"]
    if pv_levels is None:
        pv_levels = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    if fault_types is None:
        fault_types = ["3ph"]

    # Adjust parameters for fast mode
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

    print(f"Created {len(configs)} experiment configurations:")
    print(f"  Systems: {systems}")
    print(f"  Methods: {methods}")
    print(f"  PV levels: {pv_levels}")
    print(f"  Fault types: {fault_types}")
    print(f"  Runs per config: {n_runs}")

    return configs


if __name__ == "__main__":
    # Quick test
    configs = create_experiment_matrix(
        systems=["8bus"],
        methods=["GA-SQP", "JAYA"],
        pv_levels=[0.0, 1.0],
        n_runs=2,
        fast=True,
    )

    pipeline = TrainingPipeline(configs, n_workers=1, verbose=True)
    results = pipeline.run_all()
    print(f"\nCompleted {len(results)} runs")
