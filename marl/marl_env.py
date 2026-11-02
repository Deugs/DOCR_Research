"""
marl_env.py
───────────
DOCR Coordination as a Decentralised Multi-Agent RL Environment.

Phase 2 Refactor: Enhanced with neighbor observations and communication interface.
────────────────────────────────────────────────────────────────────────────────
- Integrates DistributionNetwork for dynamic power flow
- Supports 6 PV penetration levels (0, 20, 40, 60, 80, 100%)
- Multiple fault types (3ph primary, LL/1ph/2ph-g for generalization)
- Topology-aware observations (N-1 contingencies)
- Sensor noise simulation (CT/PT errors)
- Curriculum learning support (6 stages)
- NEW Phase 2: Neighbor observations for GNN communication (get_neighbor_observations())

Design (CTDE — Centralised Training, Decentralised Execution)
──────────────────────────────────────────────────────────────
- One agent per relay.
- Agent i controls (TMS_i, PS_i) via continuous Delta-actions in [-1, 1].
- Observation: [TMS_i, PS_i, If_norm, cti_slack_norm, viol_flag, top_norm, ps_norm,
                pv_level, topology_flag, sensor_quality, v_norm, p_dir]  (12 features).
  v_norm/p_dir (Objective 2's local V/P_dir measurements) are real under
  use_pandapower=True (genuine AC power-flow solve); an analytical proxy
  otherwise -- see _get_obs()'s v_pu/p_dir computation for why (a real
  per-step solve is ~28x slower, prohibitive for full-campaign training).
  An earlier "Phase 2 Enhanced" pass computed additional neighbor_TMS/PS/
  viol/slack statistics here intending to fold them into this vector, but
  never actually did — they were computed every step and discarded
  without affecting OBS_DIM or agent behavior. Removed; real neighbor
  conditioning for GNN communication happens separately via
  get_neighbor_observations(), used by MAPPOMARLTrainer when use_comm=True.
- Reward: shaped local + global penalty signal + PV adaptation bonus.
  (A "communication efficiency penalty" term exists in _compute_rewards
  but is hardcoded to 0.0 — never implemented; see that method.)
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

    # Objective 2 (thesis) states each relay acts from local measurements
    # (V, I, P_dir, I_fault). Only I_fault (If_norm below) was literally
    # present before this -- expanded 10 -> 12 to add real per-agent
    # voltage (V) and power-flow direction (P_dir). See _get_obs()'s
    # v_pu/p_dir computation for how each is sourced.
    OBS_DIM = 12   # Phase 1: expanded from 7 to 10; Objective 2: 10 to 12 (V, P_dir)
    ACT_DIM = 2    # [DeltaTMS, DeltaPS] normalised to [-1, 1]

    def __init__(
        self,
        net:          NetworkData,
        action_scale: float = 0.05,
        max_steps:    int   = 300,
        reward_scale: float = 1.0,
        seed:         int   = 0,
        use_pandapower: bool = False,
        curriculum_stage: int = 0,
        sensor_noise: bool = False,
        fault_type:   str = "3ph",
    ):
        self.net          = net
        self.n_agents     = net.n_relays
        self.action_scale = action_scale
        self.max_steps    = max_steps
        self.reward_scale = reward_scale
        self.rng          = np.random.default_rng(seed)

        self.use_pandapower = use_pandapower
        self.curriculum_stage = curriculum_stage
        self.sensor_noise = sensor_noise
        self.fault_type = fault_type

        self.dist_net = DistributionNetwork(
            net_data=net,
            use_pandapower=use_pandapower,
            sensor_seed=seed,
        )

        self.pv_levels = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        self.current_pv_level = self.pv_levels[min(curriculum_stage, 5)]

        self.ps_min, self.ps_max = net.ps_bounds()
        self.ps_range  = np.maximum(self.ps_max - self.ps_min, 1e-3)
        self.tms_range = TMS_MAX - TMS_MIN

        self._build_adjacency()
        # 0-indexed set of buses hosting DG/PV (DGConfig.bus_idx is
        # 1-indexed, matching RelayPair.bus_pri/bus_bkp's convention) --
        # precomputed once rather than rebuilt every _get_obs() call.
        # Used by the P_dir/voltage analytical proxies below.
        self._dg_bus_set = {dg.bus_idx - 1 for dg in self.net.dg_buses}
        # Depends on _dg_bus_set (just above) and _relay_bus_arr (built in
        # _build_adjacency(), which runs before _dg_bus_set exists) -- built
        # here, not in _build_adjacency(), so both prerequisites are ready.
        self._is_dg_bus_arr = np.array([
            1.0 if bus in self._dg_bus_set else 0.0 for bus in self._relay_bus_arr
        ])
        self.TMS = np.zeros(self.n_agents)
        self.PS  = np.zeros(self.n_agents)
        self.step_count = 0
        self.current_state = None

        self.reset()

    def _build_adjacency(self):
        """Map relay index -> list of pair indices it appears in."""
        self.relay_pairs: List[List[int]] = [[] for _ in range(self.n_agents)]
        for idx, pair in enumerate(self.net.pairs):
            self.relay_pairs[pair.primary].append(idx)
            self.relay_pairs[pair.backup].append(idx)

        # Build neighbor list for each relay (Phase 2)
        self.neighbors: List[List[int]] = [[] for _ in range(self.n_agents)]
        for i in range(self.n_agents):
            neighbor_set = set()
            for pidx in self.relay_pairs[i]:
                pair = self.net.pairs[pidx]
                if pair.primary == i:
                    neighbor_set.add(pair.backup)
                else:
                    neighbor_set.add(pair.primary)
            # Also add relays at same bus. Use net.get_relay_bus(), which
            # resolves the authoritative relay_bus array (falling back to
            # checking both primary AND backup roles in pairs) — the
            # previous primary-only pair scan silently missed relay i's
            # same-bus neighbors whenever i (or the candidate neighbor)
            # only ever appeared as a backup.
            bus_i = self.net.get_relay_bus(i)
            for j in range(self.n_agents):
                if j != i and self.net.get_relay_bus(j) == bus_i:
                    neighbor_set.add(j)
            self.neighbors[i] = sorted(list(neighbor_set))

        self._build_vectorized_pair_maps()
        self._relay_bus_arr = np.array(
            [self.net.get_relay_bus(i) for i in range(self.n_agents)]
        )

    def _build_vectorized_pair_maps(self):
        """Precompute dense, padded (n_agents, max_degree) index/mask arrays
        from the ragged self.relay_pairs[i] lists, computed once here rather
        than rebuilt implicitly every _get_obs()/_compute_rewards() call via
        a Python `for i in range(n_agents)` loop over a list-of-lists.
        Padding uses index 0 with an accompanying boolean mask (standard
        ragged-to-dense technique) so downstream masked reductions
        (max/sum/mean/any/all over "pairs touching relay i") can run as
        single vectorized NumPy ops across all agents at once instead of a
        Python loop gathering each agent's variable-length pair list.

        Two groupings, matching the two ways _get_obs()/_compute_rewards()
        index into relay_pairs: the UNION membership (pairs where relay i is
        either primary or backup -- self.relay_pairs[i] itself) and the
        PRIMARY-ONLY membership (pairs where relay i is specifically the
        primary relay -- used for If_norm's primary/backup value selection,
        top_own, and r_speed, each of which only wants the primary-side
        value or only this relay's own primary pairs)."""
        n = self.n_agents

        max_deg = max((len(p) for p in self.relay_pairs), default=0)
        max_deg = max(max_deg, 1)  # avoid a zero-width array if n==0 or edge case
        pair_idx = np.zeros((n, max_deg), dtype=np.int64)
        pair_mask = np.zeros((n, max_deg), dtype=bool)
        is_primary_role = np.zeros((n, max_deg), dtype=bool)
        for i in range(n):
            pairs_i = self.relay_pairs[i]
            for slot, pidx in enumerate(pairs_i):
                pair_idx[i, slot] = pidx
                pair_mask[i, slot] = True
                is_primary_role[i, slot] = (self.net.pairs[pidx].primary == i)
        self._pair_idx_padded = pair_idx
        self._pair_mask = pair_mask
        self._is_primary_role = is_primary_role

        pri_lists = [
            [p for p in self.relay_pairs[i] if self.net.pairs[p].primary == i]
            for i in range(n)
        ]
        max_deg_pri = max((len(p) for p in pri_lists), default=0)
        max_deg_pri = max(max_deg_pri, 1)
        pri_idx = np.zeros((n, max_deg_pri), dtype=np.int64)
        pri_mask = np.zeros((n, max_deg_pri), dtype=bool)
        for i in range(n):
            for slot, pidx in enumerate(pri_lists[i]):
                pri_idx[i, slot] = pidx
                pri_mask[i, slot] = True
        self._pri_idx_padded = pri_idx
        self._pri_mask = pri_mask

        # Static per-pair base fault currents (net.pairs[pidx].If_pri/
        # If_bkp, not the dynamic topology_state-adjusted values used
        # inside compute_operating_times) -- _get_obs()'s If_norm selects
        # between these two per (agent, pair-slot) based on is_primary_role.
        self._if_pri_base = np.array([p.If_pri for p in self.net.pairs])
        self._if_bkp_base = np.array([p.If_bkp for p in self.net.pairs])

        # Same ragged-to-padded technique for self.neighbors (used by
        # get_neighbor_observations()).
        max_nbr = max((len(nb) for nb in self.neighbors), default=0)
        max_nbr = max(max_nbr, 1)
        nbr_idx = np.zeros((n, max_nbr), dtype=np.int64)
        nbr_mask = np.zeros((n, max_nbr), dtype=bool)
        for i in range(n):
            nbrs_i = self.neighbors[i]
            for slot, nidx in enumerate(nbrs_i):
                nbr_idx[i, slot] = nidx
                nbr_mask[i, slot] = True
        self._neighbor_idx_padded = nbr_idx
        self._neighbor_mask = nbr_mask

    def reset(
        self,
        seed: Optional[int] = None,
        pv_level: Optional[float] = None,
        fault_type: Optional[str] = None,
        topology_changes: Optional[Dict] = None,
    ) -> np.ndarray:
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        if pv_level is not None:
            self.current_pv_level = pv_level
        else:
            self.current_pv_level = self.pv_levels[min(self.curriculum_stage, 5)]

        if fault_type is not None:
            self.fault_type = fault_type

        self.TMS = self.rng.uniform(TMS_MIN + 0.02, TMS_MAX * 0.5, self.n_agents)
        self.PS  = (self.rng.uniform(0, 1, self.n_agents)
                    * self.ps_range + self.ps_min)
        self.PS  = np.clip(self.PS, self.ps_min, self.ps_max)
        self.step_count = 0

        time_of_day = self.rng.uniform(6, 18)
        self.current_state = self.dist_net.reset(
            pv_level=self.current_pv_level,
            fault_type=self.fault_type,
            topology_changes=topology_changes,
            time_of_day=time_of_day,
        )

        self.dist_net.set_relay_settings(self.TMS, self.PS)

        return self._get_obs()

    def _get_obs(self) -> np.ndarray:
        t_pri, t_bkp = compute_operating_times(
            self.net, self.TMS, self.PS,
            fault_type=self.fault_type,
            topology_state={"fault_currents": self.current_state.fault_currents} if self.current_state else None,
        )
        delta = t_bkp - t_pri

        # Guard on self.current_state BEFORE dereferencing it — the
        # previous ternary evaluated the `any(...)` condition
        # unconditionally and only guarded the else-branch, so it would
        # raise AttributeError (not return 0.0) if _get_obs() were ever
        # called before current_state is set. Currently unreachable
        # since reset() always populates current_state first, but fixed
        # for correctness rather than relying on that call order.
        topology_flag = 0.0
        if self.current_state:
            topology_flag = 1.0 if any(
                not v for v in self.current_state.topology_state.line_status.values()
            ) else 0.0

        sensor_quality = 1.0
        if self.sensor_noise:
            n_outages = sum(1 for v in self.current_state.topology_state.line_status.values() if not v) if self.current_state else 0
            sensor_quality = max(0.5, 1.0 - 0.1 * n_outages)

        # 2026-11-02: vectorized across all agents at once, replacing the
        # per-agent `for i in range(self.n_agents)` loop. The real
        # difficulty here is that self.relay_pairs[i] is a RAGGED
        # structure (each relay appears in a different number of P/B
        # pairs), which doesn't vectorize via plain reshaping -- instead
        # uses the padded index/mask arrays precomputed once in
        # _build_vectorized_pair_maps() (ragged-to-dense + boolean mask,
        # the standard technique for this), so each "for pidx in
        # relay_pairs[i]: ..." becomes one masked reduction (max/mean/any)
        # across all agents simultaneously. Verified bit-for-bit identical
        # to the pre-vectorization per-agent loop (see test additions) --
        # this directly shapes the RL observation every agent trains on,
        # so unlike the metaheuristics' RNG-order-tolerant vectorization,
        # this one has no tolerance for even statistical drift.
        tms_n = (self.TMS - TMS_MIN) / self.tms_range
        ps_n = (self.PS - self.ps_min) / self.ps_range

        if_pri_padded = self._if_pri_base[self._pair_idx_padded]
        if_bkp_padded = self._if_bkp_base[self._pair_idx_padded]
        if_vals_padded = np.where(self._is_primary_role, if_pri_padded, if_bkp_padded)
        if_vals_for_max = np.where(self._pair_mask, if_vals_padded, -np.inf)
        has_any_pair = self._pair_mask.any(axis=1)
        If_norm = np.where(has_any_pair, np.max(if_vals_for_max, axis=1) / 15000.0, 0.0)

        delta_padded = delta[self._pair_idx_padded]
        slacks_padded = np.where(self._pair_mask, delta_padded - CTI, 0.0)
        pair_counts = self._pair_mask.sum(axis=1)
        slack_mean = np.where(has_any_pair, slacks_padded.sum(axis=1) / np.maximum(pair_counts, 1), 0.0)
        cti_slack_norm = np.clip(slack_mean / 2.0, -1.0, 1.0)

        viol = np.any(self._pair_mask & (delta_padded < CTI - 1e-6), axis=1).astype(float)

        t_pri_pri_padded = t_pri[self._pri_idx_padded]
        has_pri = self._pri_mask.any(axis=1)
        pri_sum = np.where(self._pri_mask, t_pri_pri_padded, 0.0).sum(axis=1)
        pri_counts = self._pri_mask.sum(axis=1)
        top_own = np.where(has_pri, pri_sum / np.maximum(pri_counts, 1), TOP_MAX)
        top_norm = np.clip(top_own / TOP_MAX, 0.0, 1.0)

        pv_level_obs = np.full(self.n_agents, self.current_pv_level)

        # Objective 2: real local voltage (V) and power-flow
        # direction (P_dir) at this relay's bus. Under
        # use_pandapower=True, V comes from a genuine AC power-flow
        # solve (self.current_state.bus_voltages, populated by
        # environment/dynamic_network.py's real PandapowerNetwork
        # path). That solve is ~28x slower per step than the
        # analytical fallback (measured directly: 84ms vs 3ms/step)
        # -- prohibitive for the full training campaign (would turn
        # hours of training into days) -- so use_pandapower=False
        # (the vast majority of training) uses a fast analytical
        # proxy instead, the same "real physics under pandapower,
        # cheap approximation for training speed otherwise" pattern
        # this codebase already uses for If_pri/If_bkp
        # (network/pandapower_network.py's _analytical_fault_currents).
        # P_dir has no real per-step signal available at all under
        # the analytical path (no power-flow solve happens there),
        # so it is always the analytical proxy: local DG injection
        # (scaled by the current PV level) versus local load,
        # bounded to [-1, 1] with -1 = normal substation-to-load
        # flow, +1 = fully reversed by local generation.
        is_dg_bus = self._is_dg_bus_arr
        if self.use_pandapower and self.current_state and self.current_state.bus_voltages:
            # Dict-keyed-by-bus lookup, not a (relay,) array -- multiple
            # relays can share a bus, so this stays a small per-agent loop.
            # Left as-is deliberately: this branch is the rare, already-
            # ~28x-more-expensive opt-in path (real AC power-flow solve
            # dominates its cost regardless), not the default training hot
            # path the rest of this vectorization targets.
            v_pu = np.array([
                self.current_state.bus_voltages.get(bus, 1.0)
                for bus in self._relay_bus_arr
            ])
        else:
            v_pu = np.clip(
                1.0 - 0.05 * If_norm + 0.05 * pv_level_obs * is_dg_bus, 0.85, 1.10
            )
        v_norm = np.clip((v_pu - 0.85) / 0.25, 0.0, 1.0)
        p_dir = np.clip(2.0 * pv_level_obs * is_dg_bus - 1.0, -1.0, 1.0)

        obs = np.stack([
            tms_n,
            ps_n,
            If_norm,
            cti_slack_norm,
            viol,
            top_norm,
            ps_n,
            pv_level_obs,
            np.full(self.n_agents, topology_flag),
            np.full(self.n_agents, sensor_quality),
            v_norm,
            p_dir,
        ], axis=1)

        return obs

    def get_neighbor_observations(self, obs: np.ndarray) -> np.ndarray:
        """
        Phase 2: Get neighbor observations for each agent.

        Returns:
            neighbor_obs: (n_agents, max_neighbors, OBS_DIM) padded
        """
        max_neighbors = max(len(n) for n in self.neighbors) if self.neighbors else 0
        if max_neighbors == 0:
            return np.zeros((self.n_agents, 1, self.OBS_DIM))

        # 2026-11-02: vectorized gather using the padded neighbor index/mask
        # arrays precomputed once in _build_vectorized_pair_maps(), instead
        # of a nested per-agent/per-neighbor Python loop. obs[neighbor_idx]
        # at padding slots reads an arbitrary (index-0) observation, but
        # those slots are then zeroed via the mask -- matching the
        # original's padded-with-zero behavior exactly, since neighbor_obs
        # there started as all-zeros and only real (i, j<len(neighbors[i]))
        # slots were ever written.
        neighbor_obs = obs[self._neighbor_idx_padded] * self._neighbor_mask[:, :, None]
        return neighbor_obs

    def step(
        self,
        actions: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, bool, dict]:
        delta_tms = actions[:, 0] * self.action_scale * self.tms_range
        delta_ps  = actions[:, 1] * self.action_scale * self.ps_range

        self.TMS = np.clip(self.TMS + delta_tms, TMS_MIN, TMS_MAX)
        self.PS  = np.clip(self.PS  + delta_ps,  self.ps_min, self.ps_max)

        self.dist_net.set_relay_settings(self.TMS, self.PS)

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
            "neighbor_counts": [len(n) for n in self.neighbors],
        }

        if self.sensor_noise:
            measured_i, measured_v = self.dist_net.get_sensor_measurements(
                fault_condition=True
            )
            info["measured_currents"] = measured_i
            info["measured_voltages"] = measured_v

        return self._get_obs(), rewards * self.reward_scale, done, info

    def _compute_rewards(self, t_pri, t_bkp, delta_cti, OF):
        lam_coord  = 5.0
        bonus_cti  = 0.5

        pv_adaptation_bonus = 0.0
        if self.current_pv_level > 0.5:
            pv_adaptation_bonus = 0.3 * self.current_pv_level

        # 2026-11-02: vectorized across all agents at once, replacing the
        # per-agent `for i in range(self.n_agents)` loop -- same padded
        # index/mask arrays as _get_obs() (see _build_vectorized_pair_maps
        # and that method's docstring for why relay_pairs' ragged structure
        # needs this rather than a plain reshape). Verified bit-for-bit
        # identical to the pre-vectorization per-agent loop.
        has_pairs = self._pair_mask.any(axis=1)

        t_pri_pri_padded = t_pri[self._pri_idx_padded]
        has_pri = self._pri_mask.any(axis=1)
        pri_sum = np.where(self._pri_mask, t_pri_pri_padded, 0.0).sum(axis=1)
        pri_counts = self._pri_mask.sum(axis=1)
        r_speed = np.where(has_pri, -(pri_sum / np.maximum(pri_counts, 1)) / TOP_MAX, 0.0)

        delta_cti_padded = delta_cti[self._pair_idx_padded]
        viol_amt = np.where(self._pair_mask, np.maximum(0, CTI - delta_cti_padded), 0.0)
        r_coord = -lam_coord * viol_amt.sum(axis=1)

        satisfied = delta_cti_padded >= (CTI - 1e-6)
        # "all satisfied among my_pairs" -- padding slots must count as
        # satisfied (True) so they never flip an otherwise-all-satisfied
        # agent to False, matching Python's all() over only the real pairs.
        all_satisfied = np.all(np.where(self._pair_mask, satisfied, True), axis=1)
        r_bonus = np.where(all_satisfied, bonus_cti, 0.0)

        topology_active = bool(self.current_state and any(
            not v for v in self.current_state.topology_state.line_status.values()
        ))
        if topology_active:
            topology_bonus = np.where(all_satisfied, 0.2, -0.1)
        else:
            topology_bonus = np.zeros(self.n_agents)

        # Communication efficiency penalty: not implemented (always 0).
        # Kept as an explicit term so a future real implementation has
        # an obvious place to plug in, rather than silently claiming
        # (as an earlier docstring did) that comm cost already
        # factors into the reward.
        comm_penalty = 0.0

        rewards = r_speed + r_coord + r_bonus + pv_adaptation_bonus + topology_bonus + comm_penalty
        # Agents with no pairs at all got no reward contribution in the
        # original (the `for` loop's `if not my_pairs: continue` skipped
        # straight past them, leaving rewards[i] at its zero-initialized
        # value) -- only the shared OF term below ever reached them.
        rewards = np.where(has_pairs, rewards, 0.0)

        rewards += 0.3 * (-OF / (self.n_agents * TOP_MAX))
        return rewards

    def get_state(self):
        return self.TMS.copy(), self.PS.copy()

    def set_state(self, TMS, PS):
        self.TMS = TMS.copy()
        self.PS  = PS.copy()
        self.dist_net.set_relay_settings(TMS, PS)

    def set_curriculum_stage(self, stage: int):
        self.curriculum_stage = stage
        self.current_pv_level = self.pv_levels[min(stage, 5)]

    def evaluate_current_settings(self) -> Dict:
        return self.dist_net.evaluate_relay_settings(self.TMS, self.PS)


# Backward compatibility alias
DOCREnv = DynamicDOCREnv
