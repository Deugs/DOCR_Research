"""
safe_rl.py
──────────
Safe Reinforcement Learning for DOCR-MARL.

Phase 3: Augmented Lagrangian + Action Projection for hard constraint satisfaction.

Design:
- Hard constraints: CTI ≥ 0.3s, TMS ∈ [0.01, 1.10], PS ∈ [PS_min, PS_max]
- Action projection: Clips actions to ensure TMS/PS bounds never violated
- Augmented Lagrangian: Dual ascent on Lagrange multipliers for CTI constraints
- Combined with PPO/MAPPO for policy optimization

Mathematical formulation:
    min_π  E[Σ r_t] + Σ λ_i * g_i(s,a) + (ρ/2) * Σ max(0, g_i(s,a))²

    where g_i(s,a) = CTI - (t_bkp_i - t_pri_i) ≤ 0  (CTI margin constraint)

    λ_{k+1} = max(0, λ_k + α * g_i(s,a))  (dual ascent)

References:
- Stooke & Abbeel, "Responsive Safety in Reinforcement Learning by PID Lagrangian Methods", ICML 2020
- Ray et al., "Benchmarking Safe Exploration in Deep Reinforcement Learning", 2019
- Achiam et al., "Constrained Policy Optimization", ICML 2017
"""

import numpy as np
from typing import Tuple, List, Dict, Optional, Callable
from dataclasses import dataclass

from network.network_data import NetworkData
from network.relay_physics import (
    objective_function, check_constraints,
    compute_operating_times,
    TMS_MIN, TMS_MAX, CTI, TOP_MIN, TOP_MAX,
)


@dataclass
class SafeRLConfig:
    """Configuration for Safe RL."""
    # Augmented Lagrangian parameters
    lambda_init: float = 1.0          # Initial Lagrange multiplier
    lambda_lr: float = 0.01           # Dual ascent learning rate
    rho_penalty: float = 10.0         # Augmented penalty coefficient
    rho_max: float = 1000.0           # Max penalty (prevent numerical issues)

    # PID Lagrangian (for stability)
    use_pid: bool = True              # Use PID controller for λ updates
    Kp: float = 0.01                  # Proportional gain
    Ki: float = 0.001                 # Integral gain
    Kd: float = 0.005                 # Derivative gain

    # Action projection
    project_actions: bool = True      # Clip actions to maintain bounds
    projection_margin: float = 0.01   # Safety margin for projection

    # Constraint types
    enforce_cti: bool = True          # Enforce CTI ≥ 0.3s
    enforce_tms_bounds: bool = True   # Enforce TMS bounds
    enforce_ps_bounds: bool = True    # Enforce PS bounds

    # Cost shaping
    cost_scale: float = 1.0           # Scale factor for constraint cost
    reward_penalty_coef: float = 5.0  # Coefficient for reward penalty

    # Adaptive penalty
    adaptive_rho: bool = True         # Adapt ρ based on violation rate
    rho_adapt_rate: float = 1.5       # Multiplicative factor for ρ adaptation
    rho_adapt_threshold: float = 0.1  # Violation rate threshold for adaptation


class ActionProjector:
    """
    Action projection to ensure relay settings remain within bounds.

    Given current TMS/PS and proposed delta actions, project to feasible set:
    - TMS ∈ [TMS_MIN, TMS_MAX]
    - PS ∈ [PS_min, PS_max] (per relay)

    Also ensures CTI ≥ 0.3s by limiting action magnitudes when near violation.
    """

    def __init__(self, net: NetworkData, config: SafeRLConfig = None):
        self.net = net
        self.config = config or SafeRLConfig()
        self.ps_min, self.ps_max = net.ps_bounds()
        self.tms_range = TMS_MAX - TMS_MIN
        self.ps_range = np.maximum(self.ps_max - self.ps_min, 1e-3)

    def project(self, 
                TMS: np.ndarray, 
                PS: np.ndarray,
                delta_tms: np.ndarray, 
                delta_ps: np.ndarray,
                action_scale: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
        """
        Project delta actions to maintain feasibility.

        Parameters:
            TMS: Current TMS values (n_relays,)
            PS: Current PS values (n_relays,)
            delta_tms: Proposed TMS changes (n_relays,) in [-1, 1]
            delta_ps: Proposed PS changes (n_relays,) in [-1, 1]
            action_scale: Scaling factor for actions

        Returns:
            projected_delta_tms, projected_delta_ps: Feasible actions
        """
        if not self.config.project_actions:
            return delta_tms, delta_ps

        proj_dt = delta_tms.copy()
        proj_dp = delta_ps.copy()

        margin = self.config.projection_margin

        # Project TMS to maintain bounds
        for i in range(len(TMS)):
            proposed_tms = TMS[i] + delta_tms[i] * action_scale * self.tms_range

            if proposed_tms < TMS_MIN + margin:
                # Scale down to not violate lower bound
                max_delta = (TMS_MIN + margin - TMS[i]) / (action_scale * self.tms_range)
                proj_dt[i] = np.clip(delta_tms[i], max_delta, 1.0)

            elif proposed_tms > TMS_MAX - margin:
                # Scale down to not violate upper bound
                max_delta = (TMS_MAX - margin - TMS[i]) / (action_scale * self.tms_range)
                proj_dt[i] = np.clip(delta_tms[i], -1.0, max_delta)

        # Project PS to maintain bounds
        for i in range(len(PS)):
            proposed_ps = PS[i] + delta_ps[i] * action_scale * self.ps_range[i]

            if proposed_ps < self.ps_min[i] + margin:
                max_delta = (self.ps_min[i] + margin - PS[i]) / (action_scale * self.ps_range[i])
                proj_dp[i] = np.clip(delta_ps[i], max_delta, 1.0)

            elif proposed_ps > self.ps_max[i] - margin:
                max_delta = (self.ps_max[i] - margin - PS[i]) / (action_scale * self.ps_range[i])
                proj_dp[i] = np.clip(delta_ps[i], -1.0, max_delta)

        return proj_dt, proj_dp

    def project_with_cti_lookahead(self,
                                    TMS: np.ndarray,
                                    PS: np.ndarray,
                                    delta_tms: np.ndarray,
                                    delta_ps: np.ndarray,
                                    action_scale: float = 0.05,
                                    cti_margin: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
        """
        Advanced projection that also checks CTI constraints.

        If proposed action would cause CTI violation, scale it down.
        Uses a fast analytical check (not full power flow).

        Parameters:
            cti_margin: Additional safety margin on CTI (e.g., 0.05 = 0.35s effective)

        Returns:
            projected_delta_tms, projected_delta_ps
        """
        proj_dt, proj_dp = self.project(TMS, PS, delta_tms, delta_ps, action_scale)

        if not self.config.enforce_cti:
            return proj_dt, proj_dp

        # Test proposed settings
        proposed_TMS = np.clip(
            TMS + proj_dt * action_scale * self.tms_range,
            TMS_MIN, TMS_MAX
        )
        proposed_PS = np.clip(
            PS + proj_dp * action_scale * self.ps_range,
            self.ps_min, self.ps_max
        )

        # Quick CTI check
        t_pri, t_bkp = compute_operating_times(self.net, proposed_TMS, proposed_PS)
        delta_cti = t_bkp - t_pri

        # cti_margin is an EXTRA safety margin on top of the CTI requirement
        # (per the docstring, "0.05 = 0.35s effective") — the threshold must
        # be CTI + cti_margin, not CTI - cti_margin, or this check silently
        # accepts actions that only just clear the bare 0.3s requirement
        # with none of the intended margin, and even lets some genuine
        # violations (delta_cti in (CTI - cti_margin, CTI)) through
        # unactioned.
        if np.any(delta_cti < CTI + cti_margin):
            # CTI would be violated - scale actions down
            scale = 1.0
            for _ in range(10):  # Binary search for feasible scale
                scale *= 0.7
                test_TMS = TMS + proj_dt * action_scale * self.tms_range * scale
                test_PS = PS + proj_dp * action_scale * self.ps_range * scale
                test_TMS = np.clip(test_TMS, TMS_MIN, TMS_MAX)
                test_PS = np.clip(test_PS, self.ps_min, self.ps_max)

                t_pri_test, t_bkp_test = compute_operating_times(self.net, test_TMS, test_PS)
                if np.all(t_bkp_test - t_pri_test >= CTI + cti_margin):
                    break

            proj_dt *= scale
            proj_dp *= scale

        return proj_dt, proj_dp


class AugmentedLagrangian:
    """
    Augmented Lagrangian method for CTI constraint satisfaction.

    Maintains Lagrange multipliers λ_i for each P/B pair constraint:
        g_i(TMS, PS) = CTI - (t_bkp_i - t_pri_i) ≤ 0

    The augmented Lagrangian is:
        L = OF + Σ λ_i * g_i + (ρ/2) * Σ max(0, g_i)²

    Dual ascent updates:
        λ_{k+1} = max(0, λ_k + α * g_i)
    """

    def __init__(self, n_pairs: int, config: SafeRLConfig = None, seed: int = 0):
        self.n_pairs = n_pairs
        self.config = config or SafeRLConfig()
        self.rng = np.random.default_rng(seed)

        # Lagrange multipliers (one per P/B pair)
        self.lambdas = np.full(n_pairs, self.config.lambda_init)

        # PID controller state
        self.error_integral = np.zeros(n_pairs)
        self.error_prev = np.zeros(n_pairs)

        # Penalty coefficient
        self.rho = self.config.rho_penalty

        # Tracking
        self.violation_history = []
        self.lambda_history = []
        self.rho_history = []
        self.constraint_cost_history = []

    def compute_constraint_cost(self, 
                               TMS: np.ndarray, 
                               PS: np.ndarray,
                               net: NetworkData,
                               fault_type: str = "3ph",
                               topology_state = None) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Compute constraint violation cost and gradients.

        Returns:
            total_cost: Scalar augmented Lagrangian cost
            g_values: (n_pairs,) constraint values g_i = CTI - (t_bkp - t_pri)
            cti_violations: (n_pairs,) binary violation flags
        """
        # Convert topology_state to proper format if needed
        topo_dict = None
        if topology_state is not None:
            if hasattr(topology_state, 'fault_currents'):
                # It's a DynamicState object or similar
                topo_dict = {"fault_currents": topology_state.fault_currents}
            elif isinstance(topology_state, dict):
                topo_dict = topology_state

        t_pri, t_bkp = compute_operating_times(
            net, TMS, PS, fault_type=fault_type, topology_state=topo_dict
        )
        delta_cti = t_bkp - t_pri

        # Constraint: g_i = CTI - delta_cti ≤ 0
        # If delta_cti < CTI, then g_i > 0 (violation)
        g_values = CTI - delta_cti

        # Augmented Lagrangian cost
        # L = λ^T g + (ρ/2) ||max(0, g)||²
        violation_mask = g_values > 0

        linear_term = np.sum(self.lambdas * np.maximum(0, g_values))
        quadratic_term = 0.5 * self.rho * np.sum(np.maximum(0, g_values) ** 2)

        total_cost = linear_term + quadratic_term

        cti_violations = violation_mask.astype(float)

        return total_cost * self.config.cost_scale, g_values, cti_violations

    def update_lagrangian(self, 
                          g_values: np.ndarray,
                          episode_violation_rate: float = None):
        """
        Update Lagrange multipliers via dual ascent.

        Parameters:
            g_values: (n_pairs,) constraint values from current policy
            episode_violation_rate: Fraction of steps with violations (for adaptive ρ)
        """
        if self.config.use_pid:
            # PID controller for stable λ updates
            error = g_values  # Positive = violation

            # Proportional
            P = self.config.Kp * error

            # Integral (with anti-windup)
            self.error_integral += error
            self.error_integral = np.clip(self.error_integral, -100, 100)
            I = self.config.Ki * self.error_integral

            # Derivative
            D = self.config.Kd * (error - self.error_prev)
            self.error_prev = error.copy()

            # PID update
            delta_lambda = P + I + D
        else:
            # Simple gradient ascent
            delta_lambda = self.config.lambda_lr * g_values

        # Update λ (project to non-negative)
        self.lambdas = np.maximum(0, self.lambdas + delta_lambda)

        # Adaptive penalty coefficient
        if self.config.adaptive_rho and episode_violation_rate is not None:
            if episode_violation_rate > self.config.rho_adapt_threshold:
                # Increase penalty if violations persist
                self.rho = min(self.rho * self.config.rho_adapt_rate, self.config.rho_max)
            elif episode_violation_rate < self.config.rho_adapt_threshold / 2:
                # Decrease penalty if constraints are satisfied
                self.rho = max(self.rho / self.config.rho_adapt_rate, 1.0)

        # Track history
        self.violation_history.append(float(np.mean(g_values > 0)))
        self.lambda_history.append(self.lambdas.copy())
        self.rho_history.append(self.rho)
        self.constraint_cost_history.append(float(np.sum(np.maximum(0, g_values) ** 2)))

    def get_lagrangian_reward_penalty(self,
                                       TMS: np.ndarray,
                                       PS: np.ndarray,
                                       net: NetworkData,
                                       base_reward: float,
                                       fault_type: str = "3ph",
                                       topology_state = None) -> float:
        """
        Compute penalty to subtract from reward.

        Returns:
            penalized_reward: base_reward - penalty
        """
        cost, g_values, violations = self.compute_constraint_cost(
            TMS, PS, net, fault_type, topology_state
        )

        # Convert cost to reward penalty
        penalty = self.config.reward_penalty_coef * cost

        return base_reward - penalty

    def get_constraint_stats(self) -> Dict:
        """Return current constraint statistics."""
        return {
            "lambda_mean":     float(np.mean(self.lambdas)),
            "lambda_max":      float(np.max(self.lambdas)),
            "lambda_min":      float(np.min(self.lambdas)),
            "lambda_per_pair": self.lambdas.tolist(),
            "rho":             self.rho,
            "violation_rate":  float(np.mean(self.violation_history[-100:])) if self.violation_history else 0.0,
            "avg_cost":        float(np.mean(self.constraint_cost_history[-100:])) if self.constraint_cost_history else 0.0,
        }

    def get_lambda_trajectories(self) -> np.ndarray:
        """Return full per-pair lambda history shape (n_updates, n_pairs).
        Used for Figs 9 and 10 (CTI violation dynamics, per-pair lambda evolution)."""
        if not self.lambda_history:
            return np.zeros((0, self.n_pairs))
        return np.stack(self.lambda_history, axis=0)

    def get_projection_stats(self) -> Dict:
        """Summary stats: n_dual_updates, lambda_final, violation_trend."""
        return {
            "n_dual_updates":  len(self.lambda_history),
            "n_pairs":         self.n_pairs,
            "lambda_final":    self.lambdas.tolist(),
            "violation_trend": self.violation_history[-20:] if self.violation_history else [],
        }

    def reset(self):
        """Reset Lagrangian state."""
        self.lambdas = np.full(self.n_pairs, self.config.lambda_init)
        self.error_integral = np.zeros(self.n_pairs)
        self.error_prev = np.zeros(self.n_pairs)
        self.rho = self.config.rho_penalty
        self.violation_history = []
        self.lambda_history = []
        self.rho_history = []
        self.constraint_cost_history = []


class SafeRLWrapper:
    """
    Wrapper that adds Safe RL to any MARL trainer.

    Composes:
    1. ActionProjector: Ensures actions maintain hard bounds
    2. AugmentedLagrangian: Dual ascent for CTI constraints

    Usage:
        safe_config = SafeRLConfig()
        projector = ActionProjector(net, safe_config)
        lagrangian = AugmentedLagrangian(net.n_pairs, safe_config)

        wrapper = SafeRLWrapper(env, projector, lagrangian)
        # Use wrapper like regular env but with safe actions
    """

    def __init__(self, 
                 env,
                 projector: ActionProjector,
                 lagrangian: AugmentedLagrangian,
                 config: SafeRLConfig = None):
        self.env = env
        self.projector = projector
        self.lagrangian = lagrangian
        self.config = config or SafeRLConfig()

        # Pass through attributes
        self.n_agents = env.n_agents
        self.OBS_DIM = env.OBS_DIM
        self.ACT_DIM = env.ACT_DIM
        self.net = env.net
        self.action_scale = getattr(env, 'action_scale', 0.05)

        # Pass through TMS/PS for compatibility
        self.TMS = getattr(env, 'TMS', None)
        self.PS = getattr(env, 'PS', None)
        self.max_steps = getattr(env, 'max_steps', 300)
        self.step_count = getattr(env, 'step_count', 0)
        self.current_state = getattr(env, 'current_state', None)
        self.fault_type = getattr(env, 'fault_type', '3ph')
        self.current_pv_level = getattr(env, 'current_pv_level', 0.0)

        # Tracking
        self.n_updates = 0
        self.episode_violations = 0
        self.episode_steps = 0

    def reset(self, *args, **kwargs):
        """Reset environment and safe RL state."""
        self.episode_violations = 0
        self.episode_steps = 0
        obs = self.env.reset(*args, **kwargs)

        # Update passthrough attributes
        self.TMS = getattr(self.env, 'TMS', None)
        self.PS = getattr(self.env, 'PS', None)
        self.step_count = getattr(self.env, 'step_count', 0)
        self.current_state = getattr(self.env, 'current_state', None)

        return obs

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, bool, Dict]:
        """
        Step with safe action projection and Lagrangian penalty.

        Parameters:
            actions: (n_agents, 2) raw actions from policy

        Returns:
            obs, rewards, done, info (with safe RL metrics)
        """
        # Get current state
        TMS, PS = self.env.get_state()

        # Project actions to maintain bounds
        delta_tms = actions[:, 0]
        delta_ps = actions[:, 1]

        proj_dt, proj_dp = self.projector.project(
            TMS, PS, delta_tms, delta_ps, 
            action_scale=self.action_scale
        )

        # Optional: CTI lookahead projection
        if self.config.enforce_cti:
            proj_dt, proj_dp = self.projector.project_with_cti_lookahead(
                TMS, PS, proj_dt, proj_dp,
                action_scale=self.action_scale,
                cti_margin=0.05
            )

        safe_actions = np.stack([proj_dt, proj_dp], axis=1)

        # Step environment with safe actions
        obs, rewards, done, info = self.env.step(safe_actions)

        # Apply Lagrangian penalty to rewards
        if self.config.enforce_cti:
            new_TMS, new_PS = self.env.get_state()

            # Get topology state - handle DynamicState object
            topology_state = getattr(self.env, 'current_state', None)

            # Compute constraint cost
            cost, g_values, violations = self.lagrangian.compute_constraint_cost(
                new_TMS, new_PS, self.env.net,
                fault_type=getattr(self.env, 'fault_type', '3ph'),
                topology_state=topology_state
            )

            # Track violations
            n_viol = int(np.sum(violations))
            self.episode_violations += n_viol
            self.episode_steps += 1

            # Apply penalty to all agents equally (shared constraint)
            penalty_per_agent = self.config.reward_penalty_coef * cost / self.n_agents
            rewards -= penalty_per_agent

            # ── Tier-2: projection magnitude (Fig 21) ──────────────────────
            proj_delta_l2 = float(np.linalg.norm(safe_actions - actions))
            proj_applied  = bool(proj_delta_l2 > 1e-6)

            # Add safe RL info
            info["safe_rl"] = {
                "constraint_cost":       float(cost),
                "cti_violations":        n_viol,
                "g_mean":                float(np.mean(g_values)),
                "g_max":                 float(np.max(g_values)),
                "lambda_mean":           float(np.mean(self.lagrangian.lambdas)),
                "lambda_per_pair":       self.lagrangian.lambdas.tolist(),   # Fig 10
                "rho":                   self.lagrangian.rho,
                "projected_actions":     proj_applied,
                "projection_magnitude":  proj_delta_l2,                      # Fig 21
                "action_projection_ratio": float(
                    np.mean(np.abs(safe_actions - actions) < 1e-6)
                ),
            }
        else:
            info["safe_rl"] = {"projected_actions": True}

        # Update passthrough attributes
        self.TMS = getattr(self.env, 'TMS', None)
        self.PS = getattr(self.env, 'PS', None)
        self.step_count = getattr(self.env, 'step_count', 0)
        self.current_state = getattr(self.env, 'current_state', None)

        return obs, rewards, done, info

    def update_lagrangian(self):
        """Update Lagrangian multipliers after episode/rollout."""
        if not self.config.enforce_cti:
            return

        # Get current constraint state
        TMS, PS = self.env.get_state()

        # Get topology state
        topology_state = getattr(self.env, 'current_state', None)

        _, g_values, _ = self.lagrangian.compute_constraint_cost(
            TMS, PS, self.env.net,
            fault_type=getattr(self.env, 'fault_type', '3ph'),
            topology_state=topology_state
        )

        # Compute violation rate for this episode
        violation_rate = (self.episode_violations / max(self.episode_steps, 1))

        # Update Lagrangian
        self.lagrangian.update_lagrangian(g_values, violation_rate)
        self.n_updates += 1

        # Reset episode counters
        self.episode_violations = 0
        self.episode_steps = 0

    def get_state(self):
        return self.env.get_state()

    def set_state(self, TMS, PS):
        return self.env.set_state(TMS, PS)

    def evaluate_current_settings(self):
        return self.env.evaluate_current_settings()

    def set_curriculum_stage(self, stage):
        if hasattr(self.env, 'set_curriculum_stage'):
            self.env.set_curriculum_stage(stage)

    @property
    def curriculum_stage(self):
        # Callers (Core B/D's curriculum loops) read/write this directly
        # as a plain attribute rather than through set_curriculum_stage() —
        # without this property, `wrapped_env.curriculum_stage = x` would
        # silently shadow-set an attribute on the wrapper instead of
        # reaching the real environment the wrapper actually steps.
        return self.env.curriculum_stage

    @curriculum_stage.setter
    def curriculum_stage(self, stage):
        self.env.curriculum_stage = stage

    def get_safe_rl_stats(self) -> Dict:
        """Get comprehensive safe RL statistics."""
        stats = self.lagrangian.get_constraint_stats()
        stats["n_updates"] = self.n_updates
        stats["projected_actions"] = self.config.project_actions
        return stats

    def _get_obs(self):
        """Pass through to env._get_obs if available."""
        if hasattr(self.env, '_get_obs'):
            return self.env._get_obs()
        return self.env.reset()

    def get_neighbor_observations(self, obs):
        """Pass through to env.get_neighbor_observations if available."""
        if hasattr(self.env, 'get_neighbor_observations'):
            return self.env.get_neighbor_observations(obs)
        return np.zeros((self.n_agents, 1, self.OBS_DIM))


def create_safe_rl_env(env, safe_config: SafeRLConfig = None):
    """
    Factory function to wrap any DOCR environment with Safe RL.

    Usage:
        env = DynamicDOCREnv(net)
        safe_env = create_safe_rl_env(env)
        # Use safe_env in any trainer
    """
    safe_config = safe_config or SafeRLConfig()

    projector = ActionProjector(env.net, safe_config)
    lagrangian = AugmentedLagrangian(env.net.n_pairs, safe_config)

    return SafeRLWrapper(env, projector, lagrangian, safe_config)
