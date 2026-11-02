"""
vectorized_marl_env.py
───────────────────────
VectorizedDynamicDOCREnv: the central integration class of the GPU-native
vectorized-environment redesign (Phase V6 -- see elegant-rolling-map.md for
full context). Runs `n_envs` parallel copies of `marl_env.py`'s
`DynamicDOCREnv` as one batched computation per step/reset, composing:

  - Phase V1 (`environment.vectorized_topology_manager.VectorizedTopologyManager`)
    for per-env line-outage state.
  - Phase V-PF (`network.vectorized_power_flow.VectorizedPowerFlowSolver`)
    for real per-env bus voltages, replacing `_get_obs()`'s old
    flat/analytical `v_pu` heuristic with a genuine batched AC power-flow
    solve -- a real fidelity upgrade, not just a parity port (see that
    module's docstring).
  - `network.relay_physics.compute_operating_times_batch`/
    `objective_function_batch` (Phase V3's `fault_currents_batch` extension)
    for per-env dynamic fault currents and the batched IDMT/OF core.

Scope, confirmed via code reading before writing this (same "port only
what's load-bearing" discipline as every other phase):

- Real Core training (`run_core_training.py`) always calls `env.reset(seed=
  ...)`/`env.step(actions)` with `topology_changes=None` -- grep-confirmed
  across every call site. So N-1 outages are never actually injected via
  this path during real training; `TopologyManager`'s outage machinery only
  matters for the separate `get_scenario_batch`/`get_curriculum_batch`
  dataset-generation helpers (out of scope here, same as Phase V1's own
  scoping). `topology_changes` is still accepted here (as a single dict
  applied uniformly across the whole batch, mirroring DynamicDOCREnv's
  single-dict signature) for API parity and testing, not because real
  training exercises it.
- `DynamicState.irradiance`/`temperature`/`load_pu` (the scenario
  generator's output, Phase V2) are consumed ONLY by
  `get_sensor_measurements()`, itself gated behind `sensor_noise=True` --
  never true in any real Core training call (grep-confirmed). They play NO
  role in `_get_obs()`/`_compute_rewards()`'s default-path computation.
  Consequently `sensor_noise=True` (and therefore Phase V2's scenario
  generator) is explicitly OUT OF SCOPE for this class for now -- a
  documented limitation, not an oversight, the same category as
  `use_pandapower=True` being permanently out of scope for the vectorized
  path (V-PF's own docstring).
- Within one episode, `pv_level`/`fault_type`/topology never change
  (confirmed: `DynamicDOCREnv.step()` calls `dist_net.step(time_delta=0.25)`
  with no `topology_changes`, and nothing rescales `pv_level`/`fault_type`
  mid-episode either) -- so the per-env dynamic fault currents and the
  power-flow solve are each computed ONCE per `reset()` and cached, not
  recomputed every step. This matches real single-env behavior exactly
  (`DynamicState.fault_currents` is likewise frozen for the whole episode
  under the default `use_pandapower=False` path) while avoiding needless
  repeated solves.
- `fault_type` is a single value shared across the whole `n_envs` batch per
  episode (not one value per environment) -- a deliberate, documented scope
  simplification, the same "coarse, shared across envs" pattern already
  established for curriculum stage elsewhere in this plan.
- `p_dir` is kept bit-identical to `DynamicDOCREnv`'s existing formula (pure
  function of `pv_level`/`is_dg_bus`, confirmed via code reading to have NO
  dependency on real bus voltages even under `use_pandapower=True` in the
  original) rather than exploiting the plan's stated allowance for it to
  diverge -- there is nothing for it to diverge TO given the current
  formula, so bit-identical is simply the correct outcome here, a stricter
  result than the plan's verification bar requires, not a gap.
- The real power-flow solve's only effect on observations is `v_norm`
  (`V-PF`'s bus voltages). It deliberately does NOT feed back into the
  fault-current PV-contribution term inside `_compute_fault_currents()`'s
  original formula: that term is computed there against `bus_voltages`,
  which under the single-env default (`use_pandapower=False`) path is
  always a flat `{bus: 1.0}` dict (confirmed by reading
  `environment/dynamic_network.py`'s `reset()`) -- and since
  `PVPlant.get_fault_current_contribution()` returns exactly 0.0 whenever
  `v_terminal_pu > 0.9`, that whole term is provably always zero under the
  real training path regardless of PV level, and is therefore omitted here
  (not computed as "zero", simply never added) rather than wired to V-PF's
  real voltages -- doing the latter would make fault currents (and hence
  every reward) diverge from `DynamicDOCREnv`'s verified default-path
  output, which the plan's own verification bar does not sanction (only
  `v_norm`/`p_dir` may diverge, not the reward-driving fault-current path).

State stays NumPy at this class's own boundary (TMS/PS/obs/rewards), matching
the NumPy-in/NumPy-out convention every `relay_physics.py` batched function
already uses -- the actual batched tensor math inside
`compute_operating_times_batch`/`objective_function_batch`/V-PF's solver
still runs on whichever device `DOCR_USE_GPU=1` selects; only the env's own
glue code (index-array gathers, reward-formula arithmetic) stays NumPy,
consistent with this project's established division of labor.

Only systems with real per-branch impedance data (8/14/39-bus) are
supported -- same restriction as `VectorizedPowerFlowSolver` (raises the
same `ValueError` for anything else), since V-PF is a mandatory component
here, not optional.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from network.network_data import NetworkData
from network.relay_physics import (
    objective_function_batch, compute_operating_times_batch,
    TMS_MIN, TMS_MAX, CTI, TOP_MAX, FAULT_TYPE_WEIGHTS,
)
from marl.marl_env import DynamicDOCREnv
from environment.vectorized_topology_manager import VectorizedTopologyManager
from network.vectorized_power_flow import VectorizedPowerFlowSolver


def _default_device():
    """Reuses the project's DOCR_USE_GPU=1 opt-in policy."""
    try:
        from marl.marl_agents import _default_device as _marl_default_device
        return _marl_default_device()
    except ImportError:
        import os
        import torch
        cuda_present = torch.cuda.is_available()
        gpu_available = cuda_present and os.environ.get("DOCR_USE_GPU") == "1"
        return torch.device("cuda") if gpu_available else torch.device("cpu")


class VectorizedDynamicDOCREnv:

    OBS_DIM = DynamicDOCREnv.OBS_DIM   # 12 -- identical observation layout
    ACT_DIM = DynamicDOCREnv.ACT_DIM   # 2

    def __init__(
        self,
        net: NetworkData,
        n_envs: int,
        action_scale: float = 0.05,
        max_steps: int = 300,
        reward_scale: float = 1.0,
        seed: int = 0,
        curriculum_stage: int = 0,
        fault_type: str = "3ph",
        device=None,
    ):
        self.net = net
        self.n_envs = n_envs
        self.n_agents = net.n_relays
        self.action_scale = action_scale
        self.max_steps = max_steps
        self.reward_scale = reward_scale
        self.curriculum_stage = curriculum_stage
        self.fault_type = fault_type
        self.device = device if device is not None else _default_device()
        self.rng = np.random.default_rng(seed)

        # Borrow DynamicDOCREnv's already-verified ragged-pair/neighbor
        # padded-index-array construction (_build_adjacency /
        # _build_vectorized_pair_maps) rather than re-deriving it a third
        # time -- this is a structure donor only (never stepped/reset after
        # this point), guaranteeing byte-for-byte identical pair/neighbor
        # indexing to the single-env path with zero duplication risk.
        ref_env = DynamicDOCREnv(
            net, action_scale=action_scale, max_steps=max_steps,
            reward_scale=reward_scale, seed=seed, use_pandapower=False,
            curriculum_stage=curriculum_stage, sensor_noise=False,
            fault_type=fault_type,
        )
        self._pair_idx_padded = ref_env._pair_idx_padded
        self._pair_mask = ref_env._pair_mask
        self._is_primary_role = ref_env._is_primary_role
        self._pri_idx_padded = ref_env._pri_idx_padded
        self._pri_mask = ref_env._pri_mask
        self._if_pri_base = ref_env._if_pri_base
        self._if_bkp_base = ref_env._if_bkp_base
        self._neighbor_idx_padded = ref_env._neighbor_idx_padded
        self._neighbor_mask = ref_env._neighbor_mask
        self._relay_bus_arr = ref_env._relay_bus_arr
        self._is_dg_bus_arr = ref_env._is_dg_bus_arr
        self.neighbors = ref_env.neighbors

        self.pv_levels = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

        self.ps_min, self.ps_max = net.ps_bounds()
        self.ps_range = np.maximum(self.ps_max - self.ps_min, 1e-3)
        self.tms_range = TMS_MAX - TMS_MIN

        # V-PF is mandatory here (not optional) -- raises the same
        # ValueError as VectorizedPowerFlowSolver itself for systems without
        # real per-branch impedance data (anything but 8/14/39-bus).
        self.power_flow = VectorizedPowerFlowSolver(net, n_envs, device=self.device)
        n_lines = self.power_flow.n_lines
        self.topology = VectorizedTopologyManager(
            n_envs, n_buses=net.n_buses, n_lines=n_lines,
            n_switches=max(0, n_lines - 3), device=self.device,
        )

        self.TMS = np.zeros((n_envs, self.n_agents))
        self.PS = np.zeros((n_envs, self.n_agents))
        self.step_count = 0
        self.current_pv_level = np.full(n_envs, self.pv_levels[min(curriculum_stage, 5)])

        self.reset()

    def reset(
        self,
        seed: Optional[int] = None,
        pv_level=None,
        fault_type: Optional[str] = None,
        topology_changes: Optional[Dict] = None,
    ) -> np.ndarray:
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        if pv_level is not None:
            self.current_pv_level = np.broadcast_to(
                np.asarray(pv_level, dtype=np.float64), (self.n_envs,)
            ).copy()
        else:
            self.current_pv_level = np.full(
                self.n_envs, self.pv_levels[min(self.curriculum_stage, 5)]
            )

        if fault_type is not None:
            self.fault_type = fault_type

        self.TMS = self.rng.uniform(TMS_MIN + 0.02, TMS_MAX * 0.5, (self.n_envs, self.n_agents))
        self.PS = (self.rng.uniform(0, 1, (self.n_envs, self.n_agents))
                   * self.ps_range[None, :] + self.ps_min[None, :])
        self.PS = np.clip(self.PS, self.ps_min[None, :], self.ps_max[None, :])
        self.step_count = 0

        # Topology: reset to all-in-service, then apply the (single,
        # whole-batch-shared) requested outage set -- see module docstring
        # on why real training never exercises this branch.
        self.topology.reset()
        if topology_changes:
            import torch
            line_outages = topology_changes.get("line_outages", [])
            if line_outages:
                mask = torch.zeros((self.n_envs, self.topology.n_lines),
                                    dtype=torch.bool, device=self.device)
                idx = torch.as_tensor([i for i in line_outages if 0 <= i < self.topology.n_lines],
                                       dtype=torch.long, device=self.device)
                if idx.numel() > 0:
                    mask[:, idx] = True
                self.topology.apply_multiple_outages(line_outage_mask=mask)

        self._recompute_episode_cache()
        return self._get_obs()

    def _recompute_episode_cache(self) -> None:
        """Dynamic fault currents and the power-flow solve, both frozen for
        the rest of the episode (see module docstring: neither pv_level,
        fault_type, nor topology change mid-episode under this class's
        supported usage) -- computed once here instead of every step()."""
        import torch

        outage_factor = self.topology.get_outage_factor().cpu().numpy()  # (n_envs,)
        ft_scale = FAULT_TYPE_WEIGHTS.get(self.fault_type, 1.0)
        pv_contribution_factor = 1.0 + 0.3 * self.current_pv_level  # (n_envs,)
        scale = ft_scale * pv_contribution_factor * outage_factor  # (n_envs,)
        self._If_pri_dyn = self._if_pri_base[None, :] * scale[:, None]
        self._If_bkp_dyn = self._if_bkp_base[None, :] * scale[:, None]

        pv_level_t = torch.as_tensor(self.current_pv_level, dtype=self.power_flow.rdtype,
                                      device=self.device)
        vm, _va, converged = self.power_flow.solve(self.topology.line_status, pv_level_t)
        self._vm_batch = vm.detach().cpu().numpy()
        self.power_flow_converged = converged.detach().cpu().numpy()

    def _get_obs(self) -> np.ndarray:
        n_envs, n_agents = self.n_envs, self.n_agents
        t_pri, t_bkp = compute_operating_times_batch(
            self.net, self.TMS, self.PS, fault_type=self.fault_type,
            fault_currents_batch=(self._If_pri_dyn, self._If_bkp_dyn),
        )  # each (n_envs, n_pairs)
        delta = t_bkp - t_pri

        tms_n = (self.TMS - TMS_MIN) / self.tms_range
        ps_n = (self.PS - self.ps_min[None, :]) / self.ps_range[None, :]

        # If_norm uses the STATIC base fault currents (matching
        # DynamicDOCREnv._get_obs() exactly) -- env-independent, so computed
        # once and broadcast, not a per-env quantity.
        if_pri_padded = self._if_pri_base[self._pair_idx_padded]
        if_bkp_padded = self._if_bkp_base[self._pair_idx_padded]
        if_vals_padded = np.where(self._is_primary_role, if_pri_padded, if_bkp_padded)
        if_vals_for_max = np.where(self._pair_mask, if_vals_padded, -np.inf)
        has_any_pair = self._pair_mask.any(axis=1)
        If_norm_static = np.where(has_any_pair, np.max(if_vals_for_max, axis=1) / 15000.0, 0.0)
        If_norm = np.broadcast_to(If_norm_static[None, :], (n_envs, n_agents))

        delta_padded = delta[:, self._pair_idx_padded]  # (n_envs, n_agents, max_deg)
        mask_b = self._pair_mask[None, :, :]
        slacks_padded = np.where(mask_b, delta_padded - CTI, 0.0)
        pair_counts = self._pair_mask.sum(axis=1)
        slack_mean = np.where(has_any_pair[None, :],
                               slacks_padded.sum(axis=2) / np.maximum(pair_counts, 1)[None, :], 0.0)
        cti_slack_norm = np.clip(slack_mean / 2.0, -1.0, 1.0)

        viol = np.any(mask_b & (delta_padded < CTI - 1e-6), axis=2).astype(float)

        t_pri_pri_padded = t_pri[:, self._pri_idx_padded]
        pri_mask_b = self._pri_mask[None, :, :]
        has_pri = self._pri_mask.any(axis=1)
        pri_sum = np.where(pri_mask_b, t_pri_pri_padded, 0.0).sum(axis=2)
        pri_counts = self._pri_mask.sum(axis=1)
        top_own = np.where(has_pri[None, :], pri_sum / np.maximum(pri_counts, 1)[None, :], TOP_MAX)
        top_norm = np.clip(top_own / TOP_MAX, 0.0, 1.0)

        pv_level_obs = np.broadcast_to(self.current_pv_level[:, None], (n_envs, n_agents))

        is_dg_bus = self._is_dg_bus_arr
        v_pu = self._vm_batch[:, self._relay_bus_arr]  # real V-PF voltages, (n_envs, n_agents)
        v_norm = np.clip((v_pu - 0.85) / 0.25, 0.0, 1.0)
        # Bit-identical to DynamicDOCREnv's formula (no real-voltage
        # dependency in the original -- see module docstring).
        p_dir = np.clip(2.0 * pv_level_obs * is_dg_bus[None, :] - 1.0, -1.0, 1.0)

        topology_active = (~self.topology.line_status).any(dim=1).cpu().numpy()  # (n_envs,)
        topology_flag = np.broadcast_to(topology_active.astype(float)[:, None], (n_envs, n_agents))
        sensor_quality = np.ones((n_envs, n_agents))  # sensor_noise out of scope (module docstring)

        obs = np.stack([
            tms_n, ps_n, If_norm, cti_slack_norm, viol, top_norm, ps_n,
            pv_level_obs, topology_flag, sensor_quality, v_norm, p_dir,
        ], axis=2)
        return obs

    def get_neighbor_observations(self, obs: np.ndarray) -> np.ndarray:
        """Batched counterpart of DynamicDOCREnv.get_neighbor_observations():
        (n_envs, n_agents, max_neighbors, OBS_DIM)."""
        max_neighbors = max((len(n) for n in self.neighbors), default=0)
        if max_neighbors == 0:
            return np.zeros((self.n_envs, self.n_agents, 1, self.OBS_DIM))
        neighbor_obs = obs[:, self._neighbor_idx_padded, :] * self._neighbor_mask[None, :, :, None]
        return neighbor_obs

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, bool, dict]:
        delta_tms = actions[:, :, 0] * self.action_scale * self.tms_range
        delta_ps = actions[:, :, 1] * self.action_scale * self.ps_range[None, :]

        self.TMS = np.clip(self.TMS + delta_tms, TMS_MIN, TMS_MAX)
        self.PS = np.clip(self.PS + delta_ps, self.ps_min[None, :], self.ps_max[None, :])

        OF = objective_function_batch(
            self.net, self.TMS, self.PS, fault_type=self.fault_type,
            fault_currents_batch=(self._If_pri_dyn, self._If_bkp_dyn),
        )
        t_pri, t_bkp = compute_operating_times_batch(
            self.net, self.TMS, self.PS, fault_type=self.fault_type,
            fault_currents_batch=(self._If_pri_dyn, self._If_bkp_dyn),
        )
        delta_cti = t_bkp - t_pri

        rewards = self._compute_rewards(t_pri, t_bkp, delta_cti, OF)

        self.step_count += 1
        done = self.step_count >= self.max_steps
        # Fixed max_steps shared across every environment in the batch --
        # all n_envs finish together (see plan's "synchronized batch reset"
        # design note), so `done` is one shared bool, not an (n_envs,) array.
        # pv_level/fault_type/topology are frozen for the episode (module
        # docstring), so no re-solve/dist_net-style step() is needed here.

        info = {
            "OF": OF,
            "cti_min": np.min(delta_cti, axis=1),
            "cti_ok": np.all(delta_cti >= CTI - 1e-6, axis=1),
            "step": self.step_count,
            "pv_level": self.current_pv_level,
            "fault_type": self.fault_type,
            "curriculum_stage": self.curriculum_stage,
            "power_flow_converged": self.power_flow_converged,
        }

        return self._get_obs(), rewards * self.reward_scale, done, info

    def _compute_rewards(self, t_pri, t_bkp, delta_cti, OF) -> np.ndarray:
        lam_coord = 5.0
        bonus_cti = 0.5
        n_envs, n_agents = self.n_envs, self.n_agents

        pv_adaptation_bonus = np.where(self.current_pv_level > 0.5,
                                        0.3 * self.current_pv_level, 0.0)  # (n_envs,)

        has_pairs = self._pair_mask.any(axis=1)  # static, (n_agents,)
        pri_mask_b = self._pri_mask[None, :, :]
        has_pri = self._pri_mask.any(axis=1)
        pri_counts = self._pri_mask.sum(axis=1)

        t_pri_pri_padded = t_pri[:, self._pri_idx_padded]
        pri_sum = np.where(pri_mask_b, t_pri_pri_padded, 0.0).sum(axis=2)
        r_speed = np.where(has_pri[None, :], -(pri_sum / np.maximum(pri_counts, 1)[None, :]) / TOP_MAX, 0.0)

        delta_cti_padded = delta_cti[:, self._pair_idx_padded]
        mask_b = self._pair_mask[None, :, :]
        viol_amt = np.where(mask_b, np.maximum(0, CTI - delta_cti_padded), 0.0)
        r_coord = -lam_coord * viol_amt.sum(axis=2)

        satisfied = delta_cti_padded >= (CTI - 1e-6)
        all_satisfied = np.all(np.where(mask_b, satisfied, True), axis=2)
        r_bonus = np.where(all_satisfied, bonus_cti, 0.0)

        topology_active = (~self.topology.line_status).any(dim=1).cpu().numpy()  # (n_envs,)
        topology_bonus = np.where(
            topology_active[:, None],
            np.where(all_satisfied, 0.2, -0.1),
            0.0,
        )

        comm_penalty = 0.0  # not implemented -- same as DynamicDOCREnv

        rewards = (r_speed + r_coord + r_bonus + pv_adaptation_bonus[:, None]
                   + topology_bonus + comm_penalty)
        rewards = np.where(has_pairs[None, :], rewards, 0.0)
        rewards += 0.3 * (-OF[:, None] / (n_agents * TOP_MAX))
        return rewards

    def get_state(self):
        return self.TMS.copy(), self.PS.copy()

    def set_state(self, TMS, PS):
        self.TMS = TMS.copy()
        self.PS = PS.copy()

    def set_curriculum_stage(self, stage: int):
        self.curriculum_stage = stage
