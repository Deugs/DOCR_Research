"""
marl_env.py
───────────
DOCR Coordination as a Decentralised Multi-Agent RL Environment.

Phase 1 Refactor: DynamicDOCREnv
──────────────────────────────────
- Integrates DistributionNetwork for dynamic power flow
- Supports 6 PV penetration levels (0, 20, 40, 60, 80, 100%)
- Multiple fault types (3ph primary, LL/1ph/2ph-g for generalization)
- Topology-aware observations (N-1 contingencies)
- Sensor noise simulation (CT/PT errors)
- Curriculum learning support (6 stages)

Design (CTDE — Centralised Training, Decentralised Execution)
──────────────────────────────────────────────────────────────
- One agent per relay.
- Agent i controls (TMS_i, PS_i) via continuous Δ-actions in [-1, 1].
- Observation: [TMS_i, PS_i, If_norm, cti_slack_norm, viol_flag, top_norm, ps_norm,
                pv_level, topology_flag, sensor_quality]  (10 features — Phase 1)
- Reward: shaped local + global penalty signal + PV adaptation bonus.
- Shared centralised critic sees full [TMS_all || PS_all || PV_level || topology] during training.
"""

import numpy as np
from typing import Tuple, List, Optional, Dict
from network.network_data import NetworkData
from network.relay_physics import (
    objective_function, check_constraints,
    compute_operating_times,
    TMS_MIN, TMS_MAX, CTI, TOP_MIN, TOP_MAX,
)
from environment.dynamic_network import DistributionNetwork, DynamicState


class DynamicDOCREnv:

    OBS_DIM = 10   # Phase 1: expanded from 7 to 10 features
    ACT_DIM = 2    # [ΔTMS, ΔPS] normalised to [-1, 1]

    def __init__(
        self,
        net:          NetworkData,
        action_scale: float = 0.05,
        max_steps:    int   = 300,
        reward_scale: float = 1.0,
        seed:         int   = 0,
        use_pandapower: bool = False,  # Phase 1: optional pandapower
        curriculum_stage: int = 0,    # Phase 1: curriculum learning stage (0-5)
        sensor_noise: bool = False,     # Phase 1: enable sensor noise
        fault_type:   str = "3ph",      # Phase 1: fault type for generalization
    ):
        self.net          = net
        self.n_agents     = net.n_relays
        self.action_scale = action_scale
        self.max_steps    = max_steps
        self.reward_scale = reward_scale
        self.rng          = np.random.default_rng(seed)

        # Phase 1 additions
        self.use_pandapower = use_pandapower
        self.curriculum_stage = curriculum_stage
        self.sensor_noise = sensor_noise
        self.fault_type = fault_type

        # Create distribution network wrapper
        self.dist_net = DistributionNetwork(
            net_data=net,
            use_pandapower=use_pandapower,
            sensor_seed=seed,
        )

        # PV penetration levels (6 stages: 0, 20, 40, 60, 80, 100%)
        self.pv_levels = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        self.current_pv_level = self.pv_levels[min(curriculum_stage, 5)]

        self.ps_min, self.ps_max = net.ps_bounds()
        self.ps_range  = np.maximum(self.ps_max - self.ps_min, 1e-3)
        self.tms_range = TMS_MAX - TMS_MIN

        self._build_adjacency()
        self.TMS = np.zeros(self.n_agents)
        self.PS  = np.zeros(self.n_agents)
        self.step_count = 0
        self.current_state = None

        # Reset to initialize
        self.reset()

    def _build_adjacency(self):
        """Map relay index → list of pair indices it appears in."""
        self.relay_pairs: List[List[int]] = [[] for _ in range(self.n_agents)]
        for idx, pair in enumerate(self.net.pairs):
            self.relay_pairs[pair.primary].append(idx)
            self.relay_pairs[pair.backup].append(idx)

    def reset(
        self,
        seed: Optional[int] = None,
        pv_level: Optional[float] = None,
        fault_type: Optional[str] = None,
        topology_changes: Optional[Dict] = None,
    ) -> np.ndarray:
        """
        Phase 1: Enhanced reset with PV level, fault type, and topology.

        Parameters:
            seed: Random seed
            pv_level: PV penetration (0.0-1.0), uses curriculum stage if None
            fault_type: Fault type ("3ph", "LL", "1ph", "2ph-g"), uses default if None
            topology_changes: Dict with line_outages, bus_splits, etc.
        """
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        # Determine PV level
        if pv_level is not None:
            self.current_pv_level = pv_level
        else:
            self.current_pv_level = self.pv_levels[min(self.curriculum_stage, 5)]

        # Determine fault type
        if fault_type is not None:
            self.fault_type = fault_type

        # Random initialization of relay settings
        self.TMS = self.rng.uniform(TMS_MIN + 0.02, TMS_MAX * 0.5, self.n_agents)
        self.PS  = (self.rng.uniform(0, 1, self.n_agents)
                    * self.ps_range + self.ps_min)
        self.PS  = np.clip(self.PS, self.ps_min, self.ps_max)
        self.step_count = 0

        # Reset distribution network
        time_of_day = self.rng.uniform(6, 18)  # Daytime for PV generation
        self.current_state = self.dist_net.reset(
            pv_level=self.current_pv_level,
            fault_type=self.fault_type,
            topology_changes=topology_changes,
            time_of_day=time_of_day,
        )

        # Update distribution network with current relay settings
        self.dist_net.set_relay_settings(self.TMS, self.PS)

        return self._get_obs()

    def _get_obs(self) -> np.ndarray:
        """
        Phase 1: Enhanced observation with PV level, topology, and sensor quality.

        10-dimensional observation per agent:
        [tms_n, ps_n, If_norm, cti_slack_norm, viol_flag, top_norm, ps_norm,
         pv_level, topology_flag, sensor_quality]
        """
        t_pri, t_bkp = compute_operating_times(
            self.net, self.TMS, self.PS, 
            fault_type=self.fault_type,
            topology_state={"fault_currents": self.current_state.fault_currents} if self.current_state else None,
        )
        delta = t_bkp - t_pri

        obs = np.zeros((self.n_agents, self.OBS_DIM))

        # Topology flag: 1 if any topology changes active
        topology_flag = 1.0 if any(
            not v for v in self.current_state.topology_state.line_status.values()
        ) else 0.0 if self.current_state else 0.0

        # Sensor quality (simulated)
        sensor_quality = 1.0
        if self.sensor_noise:
            # Reduce quality with more topology changes
            n_outages = sum(1 for v in self.current_state.topology_state.line_status.values() if not v) if self.current_state else 0
            sensor_quality = max(0.5, 1.0 - 0.1 * n_outages)

        for i in range(self.n_agents):
            tms_n = (self.TMS[i] - TMS_MIN) / self.tms_range
            ps_n  = (self.PS[i]  - self.ps_min[i]) / self.ps_range[i]

            if_vals = []
            for pidx in self.relay_pairs[i]:
                p = self.net.pairs[pidx]
                if_vals.append(p.If_pri if p.primary == i else p.If_bkp)
            If_norm = (np.max(if_vals) / 15000.0) if if_vals else 0.0

            slacks = [delta[pidx] - CTI for pidx in self.relay_pairs[i]]
            cti_slack_norm = np.clip(
                float(np.mean(slacks)) / 2.0 if slacks else 0.0,
                -1.0, 1.0
            )
            viol = float(
                any(delta[pidx] < CTI - 1e-6 for pidx in self.relay_pairs[i])
            )
            pri_pairs = [pidx for pidx in self.relay_pairs[i]
                         if self.net.pairs[pidx].primary == i]
            top_own = (float(np.mean([t_pri[pidx] for pidx in pri_pairs]))
                       if pri_pairs else TOP_MAX)
            top_norm = np.clip(top_own / TOP_MAX, 0.0, 1.0)

            # Phase 1 additions
            pv_level_obs = self.current_pv_level  # 0.0 to 1.0

            obs[i] = [
                tms_n,           # 0: Normalized TMS
                ps_n,            # 1: Normalized PS
                If_norm,         # 2: Normalized fault current
                cti_slack_norm,  # 3: CTI slack
                viol,            # 4: Violation flag
                top_norm,        # 5: Operating time norm
                ps_n,            # 6: PS norm (duplicate for consistency)
                pv_level_obs,    # 7: PV penetration level (NEW)
                topology_flag,   # 8: Topology change flag (NEW)
                sensor_quality,  # 9: Sensor quality estimate (NEW)
            ]

        return obs

    def step(
        self,
        actions: np.ndarray,   # (n_agents, ACT_DIM) in [-1, 1]
    ) -> Tuple[np.ndarray, np.ndarray, bool, dict]:
        """
        Phase 1: Enhanced step with dynamic network updates.
        """
        delta_tms = actions[:, 0] * self.action_scale * self.tms_range
        delta_ps  = actions[:, 1] * self.action_scale * self.ps_range

        self.TMS = np.clip(self.TMS + delta_tms, TMS_MIN, TMS_MAX)
        self.PS  = np.clip(self.PS  + delta_ps,  self.ps_min, self.ps_max)

        # Update distribution network relay settings
        self.dist_net.set_relay_settings(self.TMS, self.PS)

        # Compute objective with dynamic fault currents
        topology_state = {"fault_currents": self.current_state.fault_currents} if self.current_state else None
        OF, sum_top, penalty = objective_function(
            self.net, self.TMS, self.PS,
            fault_type=self.fault_type,
            topology_state=topology_state,
        )
        t_pri, t_bkp = compute_operating_times(
            self.net, self.TMS, self.PS,
            fault_type=self.fault_type,
            topology_state=topology_state,
        )
        delta_cti = t_bkp - t_pri

        rewards = self._compute_rewards(t_pri, t_bkp, delta_cti, OF)

        self.step_count += 1
        done = self.step_count >= self.max_steps

        # Step distribution network forward (time evolution)
        if not done:
            self.current_state = self.dist_net.step(time_delta=0.25)

        info = {
            "OF":      OF,
            "sum_top": sum_top,
            "penalty": penalty,
            "cti_min": float(np.min(delta_cti)),
            "cti_ok":  bool(np.all(delta_cti >= CTI - 1e-6)),
            "step":    self.step_count,
            "pv_level": self.current_pv_level,
            "fault_type": self.fault_type,
            "curriculum_stage": self.curriculum_stage,
        }

        # Add sensor measurements if enabled
        if self.sensor_noise:
            measured_i, measured_v = self.dist_net.get_sensor_measurements(
                fault_condition=True
            )
            info["measured_currents"] = measured_i
            info["measured_voltages"] = measured_v

        return self._get_obs(), rewards * self.reward_scale, done, info

    def _compute_rewards(self, t_pri, t_bkp, delta_cti, OF):
        """
        Phase 1: Enhanced reward with PV adaptation and topology awareness.
        """
        rewards    = np.zeros(self.n_agents)
        lam_coord  = 5.0
        bonus_cti  = 0.5

        # PV adaptation bonus: higher reward for maintaining coordination at high PV
        pv_adaptation_bonus = 0.0
        if self.current_pv_level > 0.5:
            pv_adaptation_bonus = 0.3 * self.current_pv_level

        for i in range(self.n_agents):
            my_pairs = self.relay_pairs[i]
            if not my_pairs:
                continue
            pri_idxs = [p for p in my_pairs
                        if self.net.pairs[p].primary == i]
            r_speed = (-float(np.mean([t_pri[p] for p in pri_idxs])) / TOP_MAX
                       if pri_idxs else 0.0)
            r_coord = -lam_coord * np.sum(
                np.maximum(0, CTI - delta_cti[my_pairs]))
            r_bonus = bonus_cti if all(
                delta_cti[p] >= CTI - 1e-6 for p in my_pairs) else 0.0

            # Topology awareness: bonus for maintaining performance during N-1
            topology_bonus = 0.0
            if self.current_state and any(
                not v for v in self.current_state.topology_state.line_status.values()
            ):
                topology_bonus = 0.2 if all(
                    delta_cti[p] >= CTI - 1e-6 for p in my_pairs
                ) else -0.1

            rewards[i] = r_speed + r_coord + r_bonus + pv_adaptation_bonus + topology_bonus

        rewards += 0.3 * (-OF / (self.n_agents * TOP_MAX))
        return rewards

    def get_state(self):
        return self.TMS.copy(), self.PS.copy()

    def set_state(self, TMS, PS):
        self.TMS = TMS.copy()
        self.PS  = PS.copy()
        self.dist_net.set_relay_settings(TMS, PS)

    def set_curriculum_stage(self, stage: int):
        """Phase 1: Update curriculum learning stage (0-5)."""
        self.curriculum_stage = stage
        self.current_pv_level = self.pv_levels[min(stage, 5)]

    def evaluate_current_settings(self) -> Dict:
        """Phase 1: Evaluate current relay settings using dynamic network."""
        return self.dist_net.evaluate_relay_settings(self.TMS, self.PS)


# Backward compatibility alias
DOCREnv = DynamicDOCREnv