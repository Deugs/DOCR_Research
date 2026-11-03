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

2026-11-02 torch-port note: this module's compute-heavy step --
compute_operating_times, called by both ActionProjector's CTI lookahead
and AugmentedLagrangian's constraint-cost computation -- is already real
torch (device-gated by the project's DOCR_USE_GPU=1 policy) as of the
2026-10-26 relay_physics.py port; that benefit already flows through here
transparently, no further change needed. ActionProjector.project()'s two
per-relay Python loops are now vectorized (NumPy array ops, not torch --
see that method's docstring for why tensors buy nothing at this size).
AugmentedLagrangian's own PID/lambda bookkeeping (self.lambdas, self.rho,
self.error_integral) deliberately stays plain NumPy, not torch tensors:
it's accessed directly as a NumPy array (not through a method) by code
outside this module -- experiments/exp_fig50_critical_pairs.py,
marl/utils/exp_fig10_lagrange_evolution.py, and tests/test_phase3.py all
call `.copy()` on AugmentedLagrangian.lambdas directly, which doesn't
exist on a torch tensor -- so converting it would be a confirmed breaking
change across multiple files, for an (n_pairs,)-sized array (14-92
elements) with no plausible GPU benefit regardless. A deliberate exception
to "move everything to torch", not an oversight -- flagged here the same
way evaluation/stat_analysis.py was flagged in the project's torch-port
plan.
"""

import numpy as np
from typing import Tuple, List, Dict, Optional, Callable
from dataclasses import dataclass

from network.network_data import NetworkData
from network.relay_physics import (
    objective_function, check_constraints,
    compute_operating_times, compute_operating_times_batch,
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

        margin = self.config.projection_margin

        # 2026-11-02: vectorized across all relays at once -- same
        # boundary-clamping formula as before (if the proposed setting
        # would cross TMS_MIN/TMS_MAX or this relay's PS bounds, shrink
        # that relay's delta just enough to land on the margin instead),
        # just via np.where masking instead of a Python
        # `for i in range(len(TMS))` loop recomputing the same scalar
        # formula index by index. TMS/PS/delta_* here are tiny (n_relays,)
        # arrays (14-92 elements) -- this isn't torch-tensor territory
        # (no GPU-relevant compute exists at this size), just removing
        # real per-element Python-loop overhead with plain NumPy array ops.
        proposed_tms = TMS + delta_tms * action_scale * self.tms_range
        max_delta_lo = (TMS_MIN + margin - TMS) / (action_scale * self.tms_range)
        max_delta_hi = (TMS_MAX - margin - TMS) / (action_scale * self.tms_range)
        proj_dt = np.where(
            proposed_tms < TMS_MIN + margin, np.clip(delta_tms, max_delta_lo, 1.0),
            np.where(proposed_tms > TMS_MAX - margin, np.clip(delta_tms, -1.0, max_delta_hi),
                     delta_tms)
        )

        proposed_ps = PS + delta_ps * action_scale * self.ps_range
        max_delta_lo_ps = (self.ps_min + margin - PS) / (action_scale * self.ps_range)
        max_delta_hi_ps = (self.ps_max - margin - PS) / (action_scale * self.ps_range)
        proj_dp = np.where(
            proposed_ps < self.ps_min + margin, np.clip(delta_ps, max_delta_lo_ps, 1.0),
            np.where(proposed_ps > self.ps_max - margin, np.clip(delta_ps, -1.0, max_delta_hi_ps),
                     delta_ps)
        )

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

    def project_batch(self, TMS_batch: np.ndarray, PS_batch: np.ndarray,
                       delta_tms_batch: np.ndarray, delta_ps_batch: np.ndarray,
                       action_scale: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
        """Batched counterpart of project() (Phase V8 of the
        GPU-vectorized-environment redesign): TMS_batch/PS_batch/
        delta_*_batch all (n_envs, n_agents). Identical boundary-clamping
        formula, broadcasting self.ps_min/ps_max/ps_range (static, shared
        across every environment -- one network's per-relay PS bounds don't
        vary by environment) against the added n_envs axis. No torch here,
        same reasoning as project() itself: these are 14-92-element-per-env
        arrays, not GPU-relevant compute even batched across hundreds of
        envs."""
        if not self.config.project_actions:
            return delta_tms_batch, delta_ps_batch

        margin = self.config.projection_margin

        proposed_tms = TMS_batch + delta_tms_batch * action_scale * self.tms_range
        max_delta_lo = (TMS_MIN + margin - TMS_batch) / (action_scale * self.tms_range)
        max_delta_hi = (TMS_MAX - margin - TMS_batch) / (action_scale * self.tms_range)
        proj_dt = np.where(
            proposed_tms < TMS_MIN + margin, np.clip(delta_tms_batch, max_delta_lo, 1.0),
            np.where(proposed_tms > TMS_MAX - margin, np.clip(delta_tms_batch, -1.0, max_delta_hi),
                     delta_tms_batch)
        )

        ps_min, ps_max, ps_range = self.ps_min[None, :], self.ps_max[None, :], self.ps_range[None, :]
        proposed_ps = PS_batch + delta_ps_batch * action_scale * ps_range
        max_delta_lo_ps = (ps_min + margin - PS_batch) / (action_scale * ps_range)
        max_delta_hi_ps = (ps_max - margin - PS_batch) / (action_scale * ps_range)
        proj_dp = np.where(
            proposed_ps < ps_min + margin, np.clip(delta_ps_batch, max_delta_lo_ps, 1.0),
            np.where(proposed_ps > ps_max - margin, np.clip(delta_ps_batch, -1.0, max_delta_hi_ps),
                     delta_ps_batch)
        )

        return proj_dt, proj_dp

    def project_with_cti_lookahead_batch(
        self, TMS_batch: np.ndarray, PS_batch: np.ndarray,
        delta_tms_batch: np.ndarray, delta_ps_batch: np.ndarray,
        net: NetworkData, action_scale: float = 0.05, cti_margin: float = 0.05,
        fault_type: str = "3ph", fault_currents_batch=None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Batched counterpart of project_with_cti_lookahead(): each
        environment gets its OWN independent scalar `scale` and its own
        independent binary-search stopping point (an env with no violation
        never enters the reduction loop at all, exactly matching the
        single-env version's `if np.any(violation)` skip-path; an env that
        becomes feasible mid-loop stops reducing further while others
        continue) -- via `needs_scaling` masking rather than a Python loop
        over environments, so a batch of thousands of envs costs the same
        10 iterations regardless of batch size.

        `fault_currents_batch`: optional (If_pri_batch, If_bkp_batch), each
        (n_envs, n_pairs) -- the per-env dynamic fault currents the
        calling VectorizedDynamicDOCREnv already cached for this episode
        (see that class's `_recompute_episode_cache`), passed straight
        through to `compute_operating_times_batch`. None uses the static
        shared fault currents (matches project_with_cti_lookahead()'s own
        default, single-candidate behavior).
        """
        n_envs = TMS_batch.shape[0]
        proj_dt, proj_dp = self.project_batch(TMS_batch, PS_batch, delta_tms_batch,
                                               delta_ps_batch, action_scale)

        if not self.config.enforce_cti:
            return proj_dt, proj_dp

        ps_min, ps_max, ps_range = self.ps_min[None, :], self.ps_max[None, :], self.ps_range[None, :]
        proposed_TMS = np.clip(TMS_batch + proj_dt * action_scale * self.tms_range, TMS_MIN, TMS_MAX)
        proposed_PS = np.clip(PS_batch + proj_dp * action_scale * ps_range, ps_min, ps_max)

        t_pri, t_bkp = compute_operating_times_batch(
            net, proposed_TMS, proposed_PS, fault_type=fault_type,
            fault_currents_batch=fault_currents_batch,
        )
        delta_cti = t_bkp - t_pri  # (n_envs, n_pairs)

        needs_scaling = np.any(delta_cti < CTI + cti_margin, axis=1)  # (n_envs,)
        scale = np.ones(n_envs)

        for _ in range(10):
            if not np.any(needs_scaling):
                break
            trial_scale = np.where(needs_scaling, scale * 0.7, scale)
            test_TMS = np.clip(TMS_batch + proj_dt * action_scale * self.tms_range * trial_scale[:, None],
                                TMS_MIN, TMS_MAX)
            test_PS = np.clip(PS_batch + proj_dp * action_scale * ps_range * trial_scale[:, None],
                               ps_min, ps_max)
            t_pri_t, t_bkp_t = compute_operating_times_batch(
                net, test_TMS, test_PS, fault_type=fault_type,
                fault_currents_batch=fault_currents_batch,
            )
            feasible_now = np.all(t_bkp_t - t_pri_t >= CTI + cti_margin, axis=1)  # (n_envs,)
            scale = trial_scale
            needs_scaling = needs_scaling & ~feasible_now

        proj_dt = proj_dt * scale[:, None]
        proj_dp = proj_dp * scale[:, None]
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

    def compute_constraint_cost_batch(
        self, TMS_batch: np.ndarray, PS_batch: np.ndarray, net: NetworkData,
        fault_type: str = "3ph", fault_currents_batch=None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Batched counterpart of compute_constraint_cost() (Phase V8):
        TMS_batch/PS_batch (n_envs, n_agents). `self.lambdas`/`self.rho`
        stay a single (n_pairs,) array / scalar, SHARED across every
        environment in the batch -- confirmed correct during planning, not
        a gap: all envs in a vectorized rollout train one shared policy, so
        there is exactly one set of dual variables for it, the same way
        there is one shared actor/critic rather than one per environment.

        Returns (total_cost, g_values, cti_violations), each (n_envs,) /
        (n_envs, n_pairs) / (n_envs, n_pairs) -- the per-env batch
        counterpart of compute_constraint_cost()'s (scalar, (n_pairs,),
        (n_pairs,)) return.
        """
        t_pri, t_bkp = compute_operating_times_batch(
            net, TMS_batch, PS_batch, fault_type=fault_type,
            fault_currents_batch=fault_currents_batch,
        )
        delta_cti = t_bkp - t_pri  # (n_envs, n_pairs)

        g_values = CTI - delta_cti  # (n_envs, n_pairs)
        violation_mask = g_values > 0

        linear_term = np.sum(self.lambdas[None, :] * np.maximum(0, g_values), axis=1)
        quadratic_term = 0.5 * self.rho * np.sum(np.maximum(0, g_values) ** 2, axis=1)
        total_cost = linear_term + quadratic_term

        cti_violations = violation_mask.astype(float)
        return total_cost * self.config.cost_scale, g_values, cti_violations

    def update_lagrangian_batch(self, g_values_batch: np.ndarray,
                                 episode_violation_rate: float = None):
        """Batch-reduce g_values_batch (n_envs, n_pairs) to a single
        (n_pairs,) vector by averaging across environments, then delegate to
        the existing scalar update_lagrangian() -- since `self.lambdas` is
        one shared set of dual variables for the whole batch's policy (see
        compute_constraint_cost_batch's docstring), the dual-ascent update
        itself is not re-derived here, just fed a batch-pooled constraint
        signal instead of one environment's. Because all environments in
        this redesign reset in lockstep (fixed max_steps, Phase V6), this
        fires once per rollout/episode boundary for the whole batch at
        once, with no per-env staggering to account for."""
        g_values_mean = g_values_batch.mean(axis=0)
        self.update_lagrangian(g_values_mean, episode_violation_rate)

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
                # Fraction of action COMPONENTS actually altered by
                # projection — was inverted (measuring the unchanged
                # fraction instead) despite the name; unused by any
                # current consumer, but fixed so it means what it says.
                "action_projection_ratio": float(
                    np.mean(np.abs(safe_actions - actions) >= 1e-6)
                ),
            }
        else:
            info["safe_rl"] = {"projected_actions": True}

        # Update passthrough attributes
        self.TMS = getattr(self.env, 'TMS', None)
        self.PS = getattr(self.env, 'PS', None)
        self.step_count = getattr(self.env, 'step_count', 0)
        self.current_state = getattr(self.env, 'current_state', None)

        # Dual-ascent update at episode boundaries. This used to require an
        # external caller to invoke update_lagrangian() explicitly — no
        # real training path (Core A-D, mains/main3-6.py) ever did, so
        # lambda/rho stayed frozen at their init values for the whole run
        # despite lambda_lr/adaptive_rho being configured. Triggering it
        # here makes the wrapper self-contained, matching its own
        # docstring ("Use wrapper like regular env but with safe actions")
        # — no calling code needs Safe-RL-specific awareness.
        if done and self.config.enforce_cti:
            self.update_lagrangian()

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


class VectorizedSafeRLWrapper:
    """
    GPU-vectorized counterpart of SafeRLWrapper (Phase V12 of the
    GPU-vectorized-environment redesign): wraps a VectorizedDynamicDOCREnv
    with Safe RL action projection + Augmented Lagrangian dual ascent,
    batched across n_envs, using the already-verified Phase V8 batched
    primitives (ActionProjector.project_batch/.project_with_cti_lookahead_batch,
    AugmentedLagrangian.compute_constraint_cost_batch/.update_lagrangian_batch).

    Faithfully mirrors SafeRLWrapper.step()'s REAL behavior, not an
    idealized one: project_with_cti_lookahead() doesn't accept dynamic
    fault currents at all (it always uses compute_operating_times()'s
    static fallback, a deliberate "fast analytical check, not full power
    flow" per its own docstring), so project_with_cti_lookahead_batch() is
    called here the same way, with fault_currents_batch=None.
    compute_constraint_cost(), by contrast, DOES receive real per-step
    dynamic fault currents in the single-env wrapper, via
    `topology_state=env.current_state` -> `current_state.fault_currents`'s
    "pair_i_pri"/"pair_i_bkp" dict (built by the same
    compute_fault_currents_dynamic() Phase V6 already verified
    VectorizedDynamicDOCREnv's own cached `_If_pri_dyn`/`_If_bkp_dyn`
    arrays against) -- so compute_constraint_cost_batch() is called here
    WITH fault_currents_batch=(env._If_pri_dyn, env._If_bkp_dyn), to match.

    Unlike SafeRLWrapper, which snapshots a handful of the wrapped env's
    attributes once at __init__/reset() time (current_pv_level, fault_type,
    TMS, PS, step_count, current_state) as plain instance attributes -- a
    real staleness risk for any attribute the underlying env mutates after
    construction without a matching wrapper-side refresh (confirmed:
    SafeRLWrapper.current_pv_level/fault_type are set once in __init__ and
    never updated on reset(), so any PV-level/fault-type change after the
    wrapper is built goes unseen by anything reading it through the
    wrapper) -- this class forwards every attribute it doesn't explicitly
    define straight through to the live wrapped env via __getattr__, so
    VectorizedMAPPOTrainer always sees the environment's actual current
    state, not a frozen copy. This is a deliberate improvement made only in
    this new, parallel class -- the pre-existing SafeRLWrapper staleness is
    a separate, already-production single-env class this redesign's "never
    modify the verified original" rule keeps untouched; flagged, not fixed,
    there.

    Lagrangian state (self.lagrangian.lambdas/.rho) is one shared set of
    dual variables for the whole n_envs batch -- confirmed correct in
    Phase V8's own design note (all envs in a vectorized rollout train one
    shared policy) -- and since VectorizedDynamicDOCREnv.step() returns one
    shared `done` bool across the whole lockstep batch, the dual-ascent
    update fires once per rollout boundary for the whole batch, exactly
    mirroring SafeRLWrapper's single done-triggered call.
    """

    def __init__(self, env, projector: ActionProjector, lagrangian: AugmentedLagrangian,
                 config: SafeRLConfig = None):
        self.env = env
        self.projector = projector
        self.lagrangian = lagrangian
        self.config = config or SafeRLConfig()
        self.action_scale = getattr(env, 'action_scale', 0.05)

        self.n_updates = 0
        self.episode_violations = np.zeros(env.n_envs)
        self.episode_steps = 0

    def __getattr__(self, name):
        # Only invoked when normal attribute lookup fails on this wrapper
        # itself -- forwards to the live wrapped env (see class docstring).
        return getattr(self.env, name)

    def reset(self, *args, **kwargs):
        """Reset environment and safe RL episode bookkeeping."""
        self.episode_violations = np.zeros(self.env.n_envs)
        self.episode_steps = 0
        return self.env.reset(*args, **kwargs)

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, bool, Dict]:
        """
        Step with safe action projection and Lagrangian penalty, batched
        across n_envs.

        Parameters:
            actions: (n_envs, n_agents, 2) raw actions from policy

        Returns:
            obs, rewards, done, info (with batched safe RL metrics)
        """
        TMS, PS = self.env.get_state()  # each (n_envs, n_agents)

        delta_tms = actions[:, :, 0]
        delta_ps = actions[:, :, 1]

        proj_dt, proj_dp = self.projector.project_batch(
            TMS, PS, delta_tms, delta_ps, action_scale=self.action_scale,
        )

        if self.config.enforce_cti:
            proj_dt, proj_dp = self.projector.project_with_cti_lookahead_batch(
                TMS, PS, proj_dt, proj_dp, self.env.net,
                action_scale=self.action_scale, cti_margin=0.05,
                fault_type=self.env.fault_type, fault_currents_batch=None,
            )

        safe_actions = np.stack([proj_dt, proj_dp], axis=2)  # (n_envs, n_agents, 2)

        obs, rewards, done, info = self.env.step(safe_actions)

        if self.config.enforce_cti:
            new_TMS, new_PS = self.env.get_state()

            cost, g_values, violations = self.lagrangian.compute_constraint_cost_batch(
                new_TMS, new_PS, self.env.net, fault_type=self.env.fault_type,
                fault_currents_batch=(self.env._If_pri_dyn, self.env._If_bkp_dyn),
            )

            n_viol_per_env = violations.sum(axis=1)  # (n_envs,)
            self.episode_violations = self.episode_violations + n_viol_per_env
            self.episode_steps += 1

            # Apply penalty to all agents equally within each environment
            # (shared constraint per env), same broadcast rule as the
            # single-env wrapper's `penalty_per_agent = ... ; rewards -=
            # penalty_per_agent`, now with an added n_envs axis.
            penalty_per_agent = self.config.reward_penalty_coef * cost / self.n_agents  # (n_envs,)
            rewards = rewards - penalty_per_agent[:, None]

            # Tier-2: projection magnitude (Fig 21), per environment.
            proj_delta_l2 = np.linalg.norm(
                (safe_actions - actions).reshape(self.env.n_envs, -1), axis=1)  # (n_envs,)
            proj_applied = proj_delta_l2 > 1e-6

            info["safe_rl"] = {
                "constraint_cost":       cost,                                   # (n_envs,)
                "cti_violations":        n_viol_per_env,                         # (n_envs,)
                "g_mean":                g_values.mean(axis=1),                  # (n_envs,)
                "g_max":                 g_values.max(axis=1),                   # (n_envs,)
                "lambda_mean":           float(np.mean(self.lagrangian.lambdas)),  # shared
                "lambda_per_pair":       self.lagrangian.lambdas.tolist(),         # shared
                "rho":                   self.lagrangian.rho,                      # shared
                "projected_actions":     proj_applied,                          # (n_envs,)
                "projection_magnitude":  proj_delta_l2,                         # (n_envs,)
                "action_projection_ratio": np.mean(
                    np.abs(safe_actions - actions) >= 1e-6, axis=(1, 2)),       # (n_envs,)
            }
        else:
            info["safe_rl"] = {"projected_actions": True}

        if done and self.config.enforce_cti:
            self.update_lagrangian()

        return obs, rewards, done, info

    def update_lagrangian(self):
        """Update Lagrangian multipliers after episode/rollout, batch-pooled
        across all n_envs (see AugmentedLagrangian.update_lagrangian_batch's
        own docstring for why pooling, not per-env state, is correct here)."""
        if not self.config.enforce_cti:
            return

        TMS, PS = self.env.get_state()
        _, g_values, _ = self.lagrangian.compute_constraint_cost_batch(
            TMS, PS, self.env.net, fault_type=self.env.fault_type,
            fault_currents_batch=(self.env._If_pri_dyn, self.env._If_bkp_dyn),
        )

        violation_rate = float(np.mean(self.episode_violations / max(self.episode_steps, 1)))
        self.lagrangian.update_lagrangian_batch(g_values, violation_rate)
        self.n_updates += 1

        self.episode_violations = np.zeros(self.env.n_envs)
        self.episode_steps = 0

    def get_safe_rl_stats(self) -> Dict:
        """Get comprehensive safe RL statistics."""
        stats = self.lagrangian.get_constraint_stats()
        stats["n_updates"] = self.n_updates
        stats["projected_actions"] = self.config.project_actions
        return stats


def create_vectorized_safe_rl_env(env, safe_config: SafeRLConfig = None):
    """
    Factory function mirroring create_safe_rl_env() for a
    VectorizedDynamicDOCREnv.

    Usage:
        env = VectorizedDynamicDOCREnv(net, n_envs=64)
        safe_env = create_vectorized_safe_rl_env(env)
        trainer = VectorizedMAPPOTrainer(safe_env, ...)
    """
    safe_config = safe_config or SafeRLConfig()

    projector = ActionProjector(env.net, safe_config)
    lagrangian = AugmentedLagrangian(env.net.n_pairs, safe_config)

    return VectorizedSafeRLWrapper(env, projector, lagrangian, safe_config)


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
