"""
continual_trainer.py
────────────────────
Elastic Weight Consolidation (EWC) for Continual Adaptation in DOCR-MARL.

Phase 3: Continual learning to prevent catastrophic forgetting when adapting
to changing PV penetration levels, topology changes, and fault types.

Design:
- EWC computes Fisher Information Matrix (FIM) diagonal approximation
- FIM measures parameter importance for current task
- When adapting to new conditions, EWC penalizes changes to important parameters
- Combined with curriculum learning for smooth task transitions

Mathematical formulation:
    L_EWC(θ) = L_new(θ) + (λ/2) * Σ F_i * (θ_i - θ*_i)²

    where:
    - θ*: Optimal parameters for previous task
    - F_i: Fisher Information (diagonal) for parameter i
    - λ: EWC penalty coefficient (importance of old task)

    F_i = E[(∂log p(a|o,θ)/∂θ_i)²] ≈ (1/N) Σ (∂log π(a_t|o_t)/∂θ_i)²

References:
- Kirkpatrick et al., "Overcoming catastrophic forgetting in neural networks", PNAS 2017
- Schwarz et al., "Progress & Compress: A scalable framework for continual learning", ICML 2018
- Huszár, "Note on the quadratic penalties in elastic weight consolidation", 2018
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass, field
import copy


@dataclass
class EWCConfig:
    """Configuration for EWC continual learning."""

    # EWC penalty coefficient
    lambda_ewc: float = 100.0       # Importance weight for old task
    lambda_decay: float = 0.99        # Decay λ over time (reduce penalty as new task is learned)
    lambda_min: float = 10.0          # Minimum λ value

    # Fisher Information computation
    fisher_samples: int = 1000        # Number of samples for FIM estimation
    fisher_batch_size: int = 64       # Batch size for FIM estimation
    fisher_damping: float = 1e-4      # Damping factor for numerical stability

    # Task management
    max_tasks: int = 10               # Maximum number of tasks to remember
    consolidation_interval: int = 5000  # Steps between consolidation

    # Curriculum integration
    use_curriculum_warmstart: bool = True  # Warm-start from previous task
    curriculum_overlap: int = 1000    # Overlap steps between curriculum stages

    # Adaptive EWC
    adaptive_lambda: bool = True      # Adapt λ based on task similarity
    similarity_threshold: float = 0.5  # Task similarity threshold

    # Parameter importance
    importance_threshold: float = 1e-3  # Minimum FIM value for parameter to be protected
    sparsify_fisher: bool = True      # Only protect top-k important parameters
    top_k_fraction: float = 0.5       # Fraction of parameters to protect


class FisherInformationEstimator:
    """
    Estimate Fisher Information Matrix diagonal for neural network parameters.

    For a policy π(a|o; θ), the Fisher Information is:
        F_i = E_{o~D, a~π} [(∂log π(a|o)/∂θ_i)²]

    We approximate this by sampling from the current policy.
    """

    def __init__(self, config: EWCConfig = None):
        self.config = config or EWCConfig()

    def estimate_fisher_diagonal(self,
                                  get_policy_gradient: Callable,
                                  n_params: int,
                                  n_samples: int = None) -> np.ndarray:
        """
        Estimate diagonal Fisher Information Matrix.

        Parameters:
            get_policy_gradient: Function that returns gradient of log π(a|o) w.r.t. params
                                 Signature: get_policy_gradient() -> np.ndarray (n_params,)
            n_params: Total number of parameters
            n_samples: Number of gradient samples (default: config.fisher_samples)

        Returns:
            fisher_diagonal: (n_params,) estimated FIM diagonal
        """
        n_samples = n_samples or self.config.fisher_samples

        fisher = np.zeros(n_params)

        for _ in range(n_samples):
            # Sample gradient from current policy
            grad = get_policy_gradient()

            # Accumulate squared gradients
            fisher += grad ** 2

        # Average over samples
        fisher /= n_samples

        # Add damping for numerical stability
        fisher += self.config.fisher_damping

        # Sparsify: only protect most important parameters
        if self.config.sparsify_fisher:
            threshold = np.percentile(fisher, (1 - self.config.top_k_fraction) * 100)
            fisher = np.where(fisher >= threshold, fisher, 0.0)

        # Apply importance threshold
        fisher = np.where(fisher >= self.config.importance_threshold, fisher, 0.0)

        return fisher

    def estimate_fisher_from_trajectories(self,
                                          trajectories: List[Dict],
                                          param_indices: List[int] = None) -> np.ndarray:
        """
        Estimate Fisher Information from collected trajectories.

        Parameters:
            trajectories: List of trajectory dicts with 'obs', 'actions', 'log_probs'
            param_indices: Optional subset of parameters to estimate

        Returns:
            fisher_diagonal: (n_params,) or (len(param_indices),)
        """
        if not trajectories:
            return np.array([])

        # Extract log probability gradients
        all_grads = []
        for traj in trajectories:
            if 'log_prob_grads' in traj:
                all_grads.extend(traj['log_prob_grads'])

        if not all_grads:
            return np.zeros(len(param_indices) if param_indices else 0)

        grads = np.array(all_grads)
        if param_indices:
            grads = grads[:, param_indices]

        fisher = np.mean(grads ** 2, axis=0)
        fisher += self.config.fisher_damping

        return fisher


class EWCRegularizer:
    """
    EWC regularization term for preventing catastrophic forgetting.

    Maintains a list of (optimal_params, fisher_info) pairs for each task.
    The regularization term is:
        R_EWC(θ) = (λ/2) * Σ_task Σ_i F_task,i * (θ_i - θ*_task,i)²
    """

    def __init__(self, config: EWCConfig = None):
        self.config = config or EWCConfig()

        # Task memory: list of (task_name, optimal_params, fisher_diagonal, lambda_weight)
        self.task_memory: List[Tuple[str, np.ndarray, np.ndarray, float]] = []

        # Current task
        self.current_task_name: Optional[str] = None
        self.current_task_params: Optional[np.ndarray] = None

        # Statistics
        self.n_tasks_learned = 0
        self.task_similarities = []

    def register_task(self, task_name: str, optimal_params: np.ndarray):
        """
        Register a new task. This should be called when switching to a new task.

        Parameters:
            task_name: Unique identifier for the task (e.g., "PV_0%", "PV_50%")
            optimal_params: Optimal parameters for the current task (before switching)
        """
        # If we have a current task, consolidate it
        if self.current_task_name is not None and self.current_task_params is not None:
            self._consolidate_current_task()

        self.current_task_name = task_name
        self.current_task_params = optimal_params.copy()
        self.n_tasks_learned += 1

    def _consolidate_current_task(self):
        """Consolidate current task into memory with Fisher Information."""
        # For now, use uniform Fisher (will be updated with actual estimation)
        # In practice, this should be called after Fisher estimation
        n_params = len(self.current_task_params)
        fisher = np.ones(n_params) * 0.1  # Placeholder - will be updated

        # Add to memory
        self.task_memory.append((
            self.current_task_name,
            self.current_task_params.copy(),
            fisher,
            self.config.lambda_ewc
        ))

        # Limit memory size
        if len(self.task_memory) > self.config.max_tasks:
            # Remove oldest task (or least important - could be improved)
            self.task_memory.pop(0)

    def update_fisher(self, task_name: str, fisher_diagonal: np.ndarray):
        """
        Update Fisher Information for a registered task.

        Parameters:
            task_name: Name of the task to update
            fisher_diagonal: Estimated Fisher Information diagonal
        """
        for i, (name, params, _, lambda_w) in enumerate(self.task_memory):
            if name == task_name:
                self.task_memory[i] = (name, params, fisher_diagonal, lambda_w)
                return

        # If task not found, it might be the current task
        if self.current_task_name == task_name:
            # Store for later consolidation
            self._pending_fisher = fisher_diagonal

    def compute_ewc_penalty(self, current_params: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Compute EWC penalty and gradient.

        Parameters:
            current_params: Current parameter values

        Returns:
            penalty: Scalar EWC penalty value
            gradient: (n_params,) gradient of penalty w.r.t. parameters
        """
        if not self.task_memory:
            return 0.0, np.zeros_like(current_params)

        penalty = 0.0
        gradient = np.zeros_like(current_params)

        for task_name, optimal_params, fisher, lambda_w in self.task_memory:
            # Parameter difference
            delta = current_params - optimal_params

            # EWC penalty for this task
            task_penalty = 0.5 * lambda_w * np.sum(fisher * (delta ** 2))
            penalty += task_penalty

            # Gradient: λ * F * (θ - θ*)
            task_gradient = lambda_w * fisher * delta
            gradient += task_gradient

        return penalty, gradient

    def compute_ewc_loss(self, 
                         current_params: np.ndarray,
                         new_task_loss: float,
                         new_task_gradient: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Compute combined loss: L_new + L_EWC.

        Parameters:
            current_params: Current parameter values
            new_task_loss: Loss on new task
            new_task_gradient: Gradient of new task loss

        Returns:
            total_loss: Combined loss value
            total_gradient: Combined gradient
        """
        ewc_penalty, ewc_gradient = self.compute_ewc_penalty(current_params)

        total_loss = new_task_loss + ewc_penalty
        total_gradient = new_task_gradient + ewc_gradient

        return total_loss, total_gradient

    def adapt_lambda(self, task_similarity: float):
        """
        Adapt EWC penalty based on task similarity.

        Higher similarity → lower λ (less penalty, tasks are similar)
        Lower similarity → higher λ (more penalty, protect old knowledge)

        Parameters:
            task_similarity: Similarity between current and new task (0-1)
        """
        if not self.config.adaptive_lambda:
            return

        # Adjust λ based on similarity
        if task_similarity > self.config.similarity_threshold:
            # Similar tasks - reduce penalty
            new_lambda = max(self.config.lambda_min, 
                           self.config.lambda_ewc * (1 - task_similarity))
        else:
            # Different tasks - increase penalty
            new_lambda = self.config.lambda_ewc * (1 + (1 - task_similarity))

        # Update λ for all tasks (or could be per-task)
        for i, (name, params, fisher, _) in enumerate(self.task_memory):
            self.task_memory[i] = (name, params, fisher, new_lambda)

        self.task_similarities.append(task_similarity)

    def get_task_stats(self) -> Dict:
        """Return statistics about learned tasks."""
        return {
            "n_tasks_learned": self.n_tasks_learned,
            "n_tasks_consolidated": len(self.task_memory),
            "task_names": [t[0] for t in self.task_memory],
            "current_task": self.current_task_name,
            "avg_lambda": float(np.mean([t[3] for t in self.task_memory])) if self.task_memory else 0.0,
            "avg_task_similarity": float(np.mean(self.task_similarities)) if self.task_similarities else 0.0,
        }

    def reset(self):
        """Reset all task memory."""
        self.task_memory = []
        self.current_task_name = None
        self.current_task_params = None
        self.n_tasks_learned = 0
        self.task_similarities = []


class ContinualMAPPOTrainer:
    """
    MAPPO trainer with EWC continual adaptation.

    Extends standard MAPPO with:
    1. Task registration when conditions change (PV level, topology, fault type)
    2. Fisher Information estimation after each task
    3. EWC penalty applied during policy updates
    4. Curriculum warm-start from previous task

    Task boundaries are detected by:
    - PV penetration level changes (0% → 20% → 40% ...)
    - Topology changes (N-1 contingencies)
    - Fault type changes (3ph → LL → 1ph)
    """

    def __init__(self,
                 base_trainer,
                 ewc_config: EWCConfig = None,
                 task_detection_threshold: float = 0.1):
        """
        Parameters:
            base_trainer: Base MAPPO trainer (MAPPOMARLTrainer instance)
            ewc_config: EWC configuration
            task_detection_threshold: Threshold for detecting task change
        """
        self.base_trainer = base_trainer
        self.ewc_config = ewc_config or EWCConfig()
        self.task_detection_threshold = task_detection_threshold

        # EWC components
        self.fisher_estimator = FisherInformationEstimator(self.ewc_config)
        self.ewc_regularizer = EWCRegularizer(self.ewc_config)

        # Task tracking
        self.current_task_conditions: Dict = {}
        self.task_history: List[Dict] = []
        self.task_transitions: List[Tuple[str, str]] = []

        # Adaptation metrics
        self.adaptation_times: List[float] = []
        self.performance_after_adaptation: List[float] = []

        # Step counter for consolidation
        self.steps_since_consolidation = 0

    def detect_task_change(self, env_conditions: Dict) -> Tuple[bool, str]:
        """
        Detect if environment conditions have changed enough to constitute a new task.

        Parameters:
            env_conditions: Dict with 'pv_level', 'topology_flag', 'fault_type', etc.

        Returns:
            is_new_task: Whether this is a new task
            task_name: Name for the task
        """
        # Generate task name from conditions
        pv_level = env_conditions.get('pv_level', 0.0)
        topology = env_conditions.get('topology_flag', 0.0)
        fault_type = env_conditions.get('fault_type', '3ph')

        task_name = f"PV{int(pv_level*100):03d}_TOP{int(topology)}_{fault_type}"

        if not self.current_task_conditions:
            # First task
            self.current_task_conditions = env_conditions.copy()
            return True, task_name

        # Check for significant changes
        pv_change = abs(pv_level - self.current_task_conditions.get('pv_level', 0.0))
        topo_change = abs(topology - self.current_task_conditions.get('topology_flag', 0.0))
        fault_change = fault_type != self.current_task_conditions.get('fault_type', '3ph')

        is_new_task = (
            pv_change > self.task_detection_threshold or
            topo_change > 0.5 or  # Binary topology change
            fault_change
        )

        if is_new_task:
            # Record transition
            old_task = self._task_name_from_conditions(self.current_task_conditions)
            self.task_transitions.append((old_task, task_name))
            self.current_task_conditions = env_conditions.copy()

        return is_new_task, task_name

    def _task_name_from_conditions(self, conditions: Dict) -> str:
        """Generate task name from conditions dict."""
        pv = conditions.get('pv_level', 0.0)
        topo = conditions.get('topology_flag', 0.0)
        fault = conditions.get('fault_type', '3ph')
        return f"PV{int(pv*100):03d}_TOP{int(topo)}_{fault}"

    def on_task_transition(self, 
                          old_task_name: str,
                          new_task_name: str,
                          current_params: np.ndarray):
        """
        Handle task transition: consolidate old task, prepare for new task.

        Parameters:
            old_task_name: Name of the task being left
            new_task_name: Name of the new task
            current_params: Current model parameters (optimal for old task)
        """
        # Register old task with current parameters
        self.ewc_regularizer.register_task(old_task_name, current_params)

        # Estimate Fisher Information for old task (async or batched)
        # In practice, this would use collected trajectories
        # For now, we'll estimate during training

        # Log transition
        self.task_history.append({
            'task_name': old_task_name,
            'params': current_params.copy(),
            'timestamp': len(self.task_history),
        })

        # Warm-start: optionally initialize new task from old task parameters
        if self.ewc_config.use_curriculum_warmstart:
            # Keep current parameters as initialization (already done)
            pass

    def estimate_fisher_for_current_task(self,
                                          get_policy_gradient: Callable,
                                          n_params: int) -> np.ndarray:
        """
        Estimate Fisher Information for the current task.

        Parameters:
            get_policy_gradient: Function to get policy gradients
            n_params: Number of parameters

        Returns:
            fisher_diagonal: Estimated Fisher Information
        """
        fisher = self.fisher_estimator.estimate_fisher_diagonal(
            get_policy_gradient, n_params
        )

        # Update Fisher for the most recent task in memory
        if self.ewc_regularizer.task_memory:
            last_task_name = self.ewc_regularizer.task_memory[-1][0]
            self.ewc_regularizer.update_fisher(last_task_name, fisher)

        return fisher

    def compute_ewc_penalty(self, current_params: np.ndarray) -> Tuple[float, np.ndarray]:
        """Compute EWC penalty for current parameters."""
        return self.ewc_regularizer.compute_ewc_penalty(current_params)

    def apply_ewc_to_gradient(self,
                              current_params: np.ndarray,
                              new_gradient: np.ndarray) -> np.ndarray:
        """
        Apply EWC regularization to gradient.

        Parameters:
            current_params: Current parameter values
            new_gradient: Gradient from new task

        Returns:
            regularized_gradient: Gradient with EWC penalty
        """
        ewc_penalty, ewc_gradient = self.ewc_regularizer.compute_ewc_penalty(current_params)
        # ── Tier-2: log penalty for Fig 29 curve ──────────────────────────────
        self._log_ewc_penalty(ewc_penalty)
        return new_gradient + ewc_gradient

    def get_adaptation_stats(self) -> Dict:
        """Get statistics about continual adaptation."""
        stats = self.ewc_regularizer.get_task_stats()
        stats.update({
            'n_task_transitions':          len(self.task_transitions),
            'task_transitions':            self.task_transitions,
            'avg_adaptation_time':         float(np.mean(self.adaptation_times)) if self.adaptation_times else 0.0,
            'performance_after_adaptation': self.performance_after_adaptation,
        })
        return stats

    # ── Tier-2: Fig 29 data extraction ────────────────────────────────────────
    def get_fisher_diagonals(self) -> Dict[str, np.ndarray]:
        """
        Return the Fisher Information diagonal for every consolidated task.

        Returns
        -------
        dict  {task_name: fisher_ndarray}  — one entry per remembered task.
        Used for Fig 29 (EWC Fisher diagonal visualisation).
        """
        return {
            name: fisher.copy()
            for name, _params, fisher, _lw in self.ewc_regularizer.task_memory
        }

    def get_parameter_drift(self) -> Dict[str, float]:
        """
        Compute ||theta - theta*_task||^2 for every remembered task
        using the *current* policy parameters.

        Returns
        -------
        dict  {task_name: drift_scalar}
        Used for Fig 29 (parameter drift across task transitions).
        """
        if not hasattr(self, 'base_trainer') or not hasattr(self.base_trainer, 'actor'):
            return {}
        current_params = self.base_trainer.actor.get_all_params()
        drift = {}
        for name, opt_params, fisher, _lw in self.ewc_regularizer.task_memory:
            delta = current_params[:len(opt_params)] - opt_params
            weighted = float(np.sum(fisher * delta**2))
            unweighted = float(np.sum(delta**2))
            drift[name] = {"weighted_l2": weighted, "raw_l2": unweighted}
        return drift

    def get_ewc_loss_history(self) -> List[float]:
        """
        Return per-step EWC penalty history.

        The ContinualMAPPOTrainer logs EWC penalty each time
        apply_ewc_to_gradient() is called.  Returns the full list
        for Fig 29 (EWC loss curve during curriculum training).
        """
        return list(getattr(self, '_ewc_loss_log', []))

    def _log_ewc_penalty(self, penalty: float):
        """Internal: accumulate EWC penalty for get_ewc_loss_history()."""
        if not hasattr(self, '_ewc_loss_log'):
            self._ewc_loss_log: List[float] = []
        self._ewc_loss_log.append(float(penalty))

    def should_consolidate(self, step: int) -> bool:
        """Check if it's time to consolidate current task."""
        self.steps_since_consolidation += 1
        if self.steps_since_consolidation >= self.ewc_config.consolidation_interval:
            self.steps_since_consolidation = 0
            return True
        return False


class TaskSimilarityEstimator:
    """
    Estimate similarity between tasks for adaptive EWC.

    Uses parameter-space similarity and performance correlation.
    """

    def __init__(self):
        self.task_embeddings: Dict[str, np.ndarray] = {}

    def compute_task_embedding(self, 
                               env_conditions: Dict,
                               policy_params: np.ndarray) -> np.ndarray:
        """
        Compute a task embedding from environment conditions and policy.

        Parameters:
            env_conditions: Dict with task conditions
            policy_params: Policy parameters

        Returns:
            embedding: Task embedding vector
        """
        # Simple embedding: concatenate normalized conditions and parameter statistics
        pv = env_conditions.get('pv_level', 0.0)
        topo = env_conditions.get('topology_flag', 0.0)

        # One-hot encode fault type
        fault_types = ['3ph', 'LL', '1ph', '2ph-g']
        fault_onehot = np.zeros(len(fault_types))
        fault = env_conditions.get('fault_type', '3ph')
        if fault in fault_types:
            fault_onehot[fault_types.index(fault)] = 1.0

        # Parameter statistics
        param_mean = np.mean(policy_params)
        param_std = np.std(policy_params)
        param_norm = np.linalg.norm(policy_params)

        embedding = np.concatenate([
            [pv, topo],
            fault_onehot,
            [param_mean, param_std, param_norm],
        ])

        return embedding

    def estimate_similarity(self, 
                           task1_conditions: Dict, 
                           task1_params: np.ndarray,
                           task2_conditions: Dict,
                           task2_params: np.ndarray) -> float:
        """
        Estimate similarity between two tasks.

        Returns:
            similarity: Float in [0, 1] (1 = identical tasks)
        """
        emb1 = self.compute_task_embedding(task1_conditions, task1_params)
        emb2 = self.compute_task_embedding(task2_conditions, task2_params)

        # Cosine similarity
        dot = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        similarity = dot / (norm1 * norm2)

        # Map to [0, 1]
        similarity = (similarity + 1) / 2

        return float(np.clip(similarity, 0.0, 1.0))


def create_ewc_trainer(base_trainer, ewc_config: EWCConfig = None):
    """
    Factory function to wrap any MAPPO trainer with EWC.

    Usage:
        base_trainer = MAPPOMARLTrainer(env, ...)
        ewc_trainer = create_ewc_trainer(base_trainer)

        # During training, detect task changes
        is_new, task_name = ewc_trainer.detect_task_change(env_conditions)
        if is_new:
            ewc_trainer.on_task_transition(old_task, task_name, params)
    """
    return ContinualMAPPOTrainer(base_trainer, ewc_config)
