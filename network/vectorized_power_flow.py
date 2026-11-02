"""
vectorized_power_flow.py
──────────────────────────
Batched Newton-Raphson AC power flow for the GPU-vectorized-environment
redesign (Phase V-PF of the vectorized-env plan -- see plan file for full
context and the fidelity-vs-effort decision that led here).

Replaces pandapower for the vectorized training path ONLY -- pandapower
itself has no batch API (confirmed: it solves exactly one network per
call), so it cannot be used inside a GPU-batched environment. The existing
single-environment `environment/pandapower_network.py` path is untouched
and remains the unvectorized ground-truth/reference implementation this
module is verified against.

Scope, confirmed via code search before writing this:
- Pandapower's only real, consumed output anywhere in this codebase is bus
  voltage magnitude and line loading, feeding exactly two MARL observation
  features (`v_norm`/`p_dir` in `marl/marl_env.py`) -- fault currents (the
  quantity that drives every reward) are ALWAYS computed by
  `relay_physics.compute_fault_currents_dynamic()`, a separate analytical
  heuristic, regardless of pandapower. This module's job is narrow: produce
  real bus voltages, not reimplement fault-current physics.
- Within one episode, pandapower's power-flow INPUTS never actually change
  after `reset()` (loads are static; PV output and topology are set once at
  reset; `step()` calls `run_power_flow()` again but with identical inputs
  every time, since nothing rescales loads per step) -- confirmed via direct
  code reading of `environment/dynamic_network.py`. This solver exploits the
  same property deliberately: solve once per `reset()` per environment
  batch, not every step.
- Only 8/14/39-bus have real per-branch impedance data (`NetworkData.
  line_data["lines"]`'s `r_pu`/`x_pu`/`b_pu`) -- the systems without it
  (3/30/57/118-bus) are not supported by this solver and must keep using
  the existing single-environment path if real power flow is ever needed
  for them.
- Bus types mirror `pandapower_network.py`'s own classification exactly:
  one slack bus (`ext_grid`), PV buses (always-on generators with a fixed
  voltage setpoint -- 39-bus's synchronous machines), PQ buses (loads and
  distributed PV/sgen injections, including non-always-on DG that scales
  with `pv_penetration`). An all-PQ simplification was tried and rejected
  by this project already (see `pandapower_network.py`'s comment: 39-bus's
  real reference case `case39()` does not converge without its 9 generator
  buses modeled as PV).

Jacobian: computed via `torch.func.jacrev` + `vmap` (automatic
differentiation through the batched complex power-injection formula,
`S = Vc * conj(Ybus @ Vc)`) rather than the textbook closed-form analytical
Jacobian (`dSbus_dV`-style formulas). This is a deliberate choice: this
project's own history (the pandapower wrapper's many bug-fix comments: tap
ratio off-by-ones, bus-indexing mismatches, sign errors) shows how easy
sign/indexing mistakes are in hand-derived power-flow math. Autograd
removes an entire category of that risk at the cost of model simplicity --
reasonable given this is itself the one phase where "trust but verify
against an external ground truth" is correctness-critical (see this
module's test file).

float64/`torch.complex128` throughout (matching this project's established
precision precedent: the 2026-10-26 relay_physics.py rewrite found and
fixed a real float32 precision regression). `import torch` stays
module-level here deliberately (NOT lazy): this module is never imported
by the baselines/ga_sqp.py or baselines/metaheuristics.py call chains the
import-order segfault risk applies to (see relay_physics.py's module
docstring) -- it's only ever reached via the vectorized-env training path,
which already requires torch to be imported first regardless.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch

from network.network_data import NetworkData


def _default_device():
    """Reuses the project's DOCR_USE_GPU=1 opt-in policy."""
    try:
        from marl.marl_agents import _default_device as _marl_default_device
        return _marl_default_device()
    except ImportError:
        import os
        cuda_present = torch.cuda.is_available()
        gpu_available = cuda_present and os.environ.get("DOCR_USE_GPU") == "1"
        return torch.device("cuda") if gpu_available else torch.device("cpu")


class VectorizedPowerFlowSolver:
    """Batched Newton-Raphson AC power flow for `n_envs` parallel copies of
    one network topology (same bus/branch/generator structure -- i.e. the
    same `NetworkData`), each with potentially different active line
    outages and PV penetration level.

    Usage:
        solver = VectorizedPowerFlowSolver(net_data, n_envs=256)
        vm, va = solver.solve(line_in_service, pv_level)  # (n_envs, n_bus) each
    """

    def __init__(self, net_data: NetworkData, n_envs: int, device=None,
                 max_iter: int = 20, dtype=torch.complex128):
        if not (net_data.line_data and net_data.line_data.get("lines")):
            raise ValueError(
                f"VectorizedPowerFlowSolver requires real per-branch impedance "
                f"data (NetworkData.line_data['lines']) -- network '{net_data.name}' "
                f"has none. Only 8/14/39-bus are supported; other systems must use "
                f"the existing single-environment pandapower path."
            )
        self.net_data = net_data
        self.n_envs = n_envs
        self.device = device if device is not None else _default_device()
        self.max_iter = max_iter
        self.cdtype = dtype
        self.rdtype = torch.float64 if dtype == torch.complex128 else torch.float32
        self._build_static_structure()

    def _build_static_structure(self) -> None:
        nd = self.net_data
        n = nd.n_buses
        self.n_bus = n
        S_base = nd.base_mva

        slack_bus0 = (nd.line_data.get("slack_bus", 1) - 1) if nd.line_data else 0
        slack_vm_pu = (nd.line_data.get("slack_vm_pu", 1.0)) if nd.line_data else 1.0
        self.slack_bus = int(slack_bus0)
        self.slack_vm = float(slack_vm_pu)

        # Branch list: (from0, to0, r_pu, x_pu, b_pu, tap). Index k in this
        # list MUST correspond to index k in VectorizedTopologyManager's
        # line_status columns -- both are built by iterating
        # net_data.line_data["lines"] in the same order.
        branches = []
        for ln in nd.line_data["lines"]:
            f = ln["from_bus"] - 1
            t = ln["to_bus"] - 1
            if not (0 <= f < n and 0 <= t < n):
                continue
            r = float(ln["r_pu"])
            x = float(ln["x_pu"])
            b = float(ln.get("b_pu", 0.0))
            tap = float(ln.get("ratio", 0.0) or 1.0)
            branches.append((f, t, r, x, b, tap))
        self.branches = branches
        self.n_lines = len(branches)

        # Static (non-outage-dependent) per-branch series admittance and
        # shunt susceptance, precomputed once -- only the in-service mask
        # varies per environment/call.
        ys = np.array([1.0 / complex(r, x) for (_, _, r, x, _, _) in branches], dtype=np.complex128)
        b_arr = np.array([b for (_, _, _, _, b, _) in branches], dtype=np.float64)
        tap_arr = np.array([tap for (_, _, _, _, _, tap) in branches], dtype=np.float64)
        self._ys = torch.as_tensor(ys, dtype=self.cdtype, device=self.device)
        self._b = torch.as_tensor(b_arr, dtype=self.rdtype, device=self.device)
        self._tap = torch.as_tensor(tap_arr, dtype=self.rdtype, device=self.device)
        self._from_idx = torch.as_tensor([f for f, t, *_ in branches], dtype=torch.long, device=self.device)
        self._to_idx = torch.as_tensor([t for f, t, *_ in branches], dtype=torch.long, device=self.device)

        # Static loads (confirmed: never rescaled per-step by the real
        # pandapower path either -- see module docstring).
        bus_load_p = np.zeros(n)
        bus_load_q = np.zeros(n)
        if nd.line_data.get("loads"):
            for ld in nd.line_data["loads"]:
                b0 = ld["bus"] - 1
                if 0 <= b0 < n:
                    bus_load_p[b0] += ld["p_mw"]
                    bus_load_q[b0] += ld["q_mvar"]
        self.bus_load_p_pu = torch.as_tensor(bus_load_p / S_base, dtype=self.rdtype, device=self.device)
        self.bus_load_q_pu = torch.as_tensor(bus_load_q / S_base, dtype=self.rdtype, device=self.device)

        # Generator classification, mirroring pandapower_network.py exactly:
        # always_on + vm_pu set -> PV bus; everything else -> PQ injection.
        pv_bus0, pv_p_mw, pv_vm = [], [], []
        pq_bus0, pq_p_mw, pq_q_mvar, pq_scales_with_pv = [], [], [], []
        for dg in nd.dg_buses:
            b0 = dg.bus_idx - 1
            if not (0 <= b0 < n):
                continue
            if dg.always_on and dg.vm_pu is not None:
                pv_bus0.append(b0)
                pv_p_mw.append(dg.p_mw)
                pv_vm.append(dg.vm_pu)
            else:
                pq_bus0.append(b0)
                pq_p_mw.append(dg.p_mw)
                pq_q_mvar.append(dg.q_mvar)
                pq_scales_with_pv.append(not dg.always_on)

        bus_type = np.full(n, 2, dtype=np.int64)  # 2 = PQ (default)
        for b0 in pv_bus0:
            bus_type[b0] = 1  # 1 = PV
        bus_type[self.slack_bus] = 0  # 0 = slack
        self.bus_type = bus_type  # plain numpy, used for Python-side indexing only

        self.pv_bus0 = np.array(pv_bus0, dtype=np.int64)
        self.pv_p_mw = np.array(pv_p_mw, dtype=np.float64)
        self.pv_vm = torch.as_tensor(np.array(pv_vm, dtype=np.float64), dtype=self.rdtype, device=self.device)
        self.pq_gen_bus0 = np.array(pq_bus0, dtype=np.int64)
        self.pq_gen_p_mw = np.array(pq_p_mw, dtype=np.float64)
        self.pq_gen_q_mvar = np.array(pq_q_mvar, dtype=np.float64)
        self.pq_gen_scales_with_pv = np.array(pq_scales_with_pv, dtype=bool)
        self.S_base = S_base

        # Index sets for the NR state vector (fixed across all envs --
        # which buses are slack/PV/PQ never changes, only values do).
        non_slack = [i for i in range(n) if i != self.slack_bus]
        pq_idx = [i for i in range(n) if bus_type[i] == 2]
        self.non_slack_idx = torch.as_tensor(non_slack, dtype=torch.long, device=self.device)
        self.pq_idx = torch.as_tensor(pq_idx, dtype=torch.long, device=self.device)
        self.n_theta = len(non_slack)
        self.n_v = len(pq_idx)

    def build_ybus_batch(self, line_in_service: torch.Tensor) -> torch.Tensor:
        """line_in_service: (n_envs, n_lines) bool (or float 0/1), ordered to
        match self.branches (see VectorizedTopologyManager -- both built
        from the same net_data.line_data['lines'] iteration order).
        Returns complex Ybus, (n_envs, n_bus, n_bus)."""
        n_envs = line_in_service.shape[0]
        active = line_in_service.to(self.cdtype)  # (n_envs, n_lines)

        Yff = active * (self._ys / (self._tap.to(self.cdtype) ** 2) + 1j * self._b.to(self.cdtype) / 2)
        Ytt = active * (self._ys + 1j * self._b.to(self.cdtype) / 2)
        Yft = active * (-self._ys / self._tap.to(self.cdtype))
        Ytf = Yft  # tap is real here (no phase-shifting transformers in this data)

        Ybus = torch.zeros((n_envs, self.n_bus, self.n_bus), dtype=self.cdtype, device=self.device)
        env_idx = torch.arange(n_envs, device=self.device).unsqueeze(1).expand(-1, self.n_lines)
        f_idx = self._from_idx.unsqueeze(0).expand(n_envs, -1)
        t_idx = self._to_idx.unsqueeze(0).expand(n_envs, -1)

        Ybus.index_put_((env_idx, f_idx, f_idx), Yff, accumulate=True)
        Ybus.index_put_((env_idx, t_idx, t_idx), Ytt, accumulate=True)
        Ybus.index_put_((env_idx, f_idx, t_idx), Yft, accumulate=True)
        Ybus.index_put_((env_idx, t_idx, f_idx), Ytf, accumulate=True)
        return Ybus

    def build_power_injections(self, pv_level: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """pv_level: (n_envs,) current PV penetration per environment.
        Returns (P_spec, Q_spec), each (n_envs, n_bus), in per-unit,
        generation-minus-load convention (negative at pure load buses)."""
        n_envs = pv_level.shape[0]
        P = -self.bus_load_p_pu.unsqueeze(0).expand(n_envs, -1).clone()
        Q = -self.bus_load_q_pu.unsqueeze(0).expand(n_envs, -1).clone()

        for i, b0 in enumerate(self.pq_gen_bus0):
            scale = pv_level if self.pq_gen_scales_with_pv[i] else torch.ones_like(pv_level)
            P[:, b0] += (self.pq_gen_p_mw[i] / self.S_base) * scale
            Q[:, b0] += (self.pq_gen_q_mvar[i] / self.S_base) * scale
        # PV-bus P is specified too (only Q is solved for); always-on, so
        # no pv_level scaling (matches pandapower_network.py's dg.always_on
        # branch: output_p_mw = dg.p_mw, not dg.p_mw * pv_penetration).
        for i, b0 in enumerate(self.pv_bus0):
            P[:, b0] += self.pv_p_mw[i] / self.S_base

        return P, Q

    def _mismatch_single(self, x: torch.Tensor, Ybus_re: torch.Tensor, Ybus_im: torch.Tensor,
                          P_spec: torch.Tensor, Q_spec: torch.Tensor) -> torch.Tensor:
        """Single-environment (no batch dim) mismatch function -- vmapped
        over the env dimension by solve(). x packs [theta[non_slack],
        V[pq_idx]] (both real). Returns the real mismatch vector
        [P_calc - P_spec at non-slack buses, Q_calc - Q_spec at PQ buses].

        Ybus is split into separate real/imaginary tensors (not one complex
        tensor) because `torch.func.jacrev` requires every argument of the
        function it wraps to be real, even arguments outside `argnums` that
        aren't being differentiated -- Ybus is reconstructed as a complex
        tensor internally, which is fine, since only the function's
        top-level inputs/outputs need to be real, not its intermediate
        values.
        """
        theta = torch.zeros(self.n_bus, dtype=self.rdtype, device=self.device)
        theta = theta.index_copy(0, self.non_slack_idx, x[: self.n_theta])

        V = torch.zeros(self.n_bus, dtype=self.rdtype, device=self.device)
        V[self.slack_bus] = self.slack_vm
        if len(self.pv_bus0) > 0:
            pv_bus_t = torch.as_tensor(self.pv_bus0, dtype=torch.long, device=self.device)
            V = V.index_copy(0, pv_bus_t, self.pv_vm)
        V = V.index_copy(0, self.pq_idx, x[self.n_theta:])

        Ybus = torch.complex(Ybus_re, Ybus_im)
        Vc = (V * torch.exp(1j * theta)).to(self.cdtype)
        S = Vc * torch.conj(Ybus @ Vc)  # complex power injection, all buses
        P_calc = S.real.to(self.rdtype)
        Q_calc = S.imag.to(self.rdtype)

        dP = (P_calc - P_spec).index_select(0, self.non_slack_idx)
        dQ = (Q_calc - Q_spec).index_select(0, self.pq_idx)
        return torch.cat([dP, dQ])

    def solve(self, line_in_service: torch.Tensor, pv_level: torch.Tensor,
              x0: Optional[torch.Tensor] = None, damping: float = 1e-8) -> Tuple[torch.Tensor, torch.Tensor]:
        """Batched Newton-Raphson solve. line_in_service: (n_envs, n_lines)
        bool. pv_level: (n_envs,). Returns (vm, va, converged): `vm`/`va`
        are (n_envs, n_bus) voltage magnitude (pu) and angle (rad), with
        slack/PV magnitudes and the slack angle set exactly to their
        specified values; `converged` is an (n_envs,) bool, False where the
        final mismatch residual exceeds 1e-6 (see the residual-computation
        comment below for why callers should check this rather than trust
        every environment's output unconditionally).

        Fixed iteration count (not per-environment adaptive convergence
        checking) -- simpler, and cheap to over-iterate at this problem
        size; tune `max_iter` upward if validation against pandapower
        (see this module's test file) shows residual mismatch too large.

        `damping`: small Levenberg-Marquardt-style ridge term added to the
        Jacobian diagonal (`J + damping*I`) before each solve. Not a
        correctness compromise on well-posed systems (see this module's
        test file -- negligible effect, well below pandapower-agreement
        tolerance, on every normal scenario tested) -- it exists specifically
        for a real degenerate case found during verification: a random line/
        transformer outage can genuinely ISLAND a generator bus (no
        remaining path to the rest of the network), which is mathematically
        singular for ANY power-flow method, not a bug in this solver.
        Pandapower's own behavior on that exact case is to silently drop the
        isolated bus and report NaN for it while still claiming
        `converged=True` for the rest -- this solver's equivalent response
        is to keep running (via damping, rather than a hard
        `torch.linalg.solve` crash that would take down an entire batch of
        otherwise-healthy environments over one degenerate one) but the
        islanded bus's resulting voltage should be treated with the same
        suspicion as pandapower's own NaN would be, not trusted as a real
        physical answer. A per-environment topological-islanding detector
        would be the more rigorous fix if this turns out to matter in
        practice; not implemented here since this is a rare, randomly-
        triggered edge case rather than a normal operating condition.
        """
        n_envs = line_in_service.shape[0]
        Ybus = self.build_ybus_batch(line_in_service)
        Ybus_re, Ybus_im = Ybus.real.contiguous(), Ybus.imag.contiguous()
        P_spec, Q_spec = self.build_power_injections(pv_level)

        n_x = self.n_theta + self.n_v
        if x0 is not None:
            x = x0.clone()
        else:
            x = torch.zeros((n_envs, n_x), dtype=self.rdtype, device=self.device)
            x[:, self.n_theta:] = 1.0  # flat-start V=1.0pu for PQ buses

        jac_fn = torch.func.vmap(torch.func.jacrev(self._mismatch_single), in_dims=(0, 0, 0, 0, 0))
        mismatch_fn = torch.func.vmap(self._mismatch_single, in_dims=(0, 0, 0, 0, 0))
        eye = torch.eye(n_x, dtype=self.rdtype, device=self.device).unsqueeze(0)

        for _ in range(self.max_iter):
            mismatch = mismatch_fn(x, Ybus_re, Ybus_im, P_spec, Q_spec)  # (n_envs, n_x)
            J = jac_fn(x, Ybus_re, Ybus_im, P_spec, Q_spec)  # (n_envs, n_x, n_x)
            J = J + damping * eye
            dx = torch.linalg.solve(J, -mismatch.unsqueeze(-1)).squeeze(-1)
            x = x + dx

        # Final residual, per environment -- confirmed during verification
        # (see this module's test file) that non-convergence shows up
        # clearly here (residual ~10-200) vs. a healthy solve (~1e-10):
        # cases where real pandapower itself fails to converge (its own
        # internal NR diverging, e.g. an extreme/unrealistic load-vs-
        # generation imbalance from a random scenario) make THIS solver's
        # voltages go visibly unphysical too (negative magnitudes), not
        # silently plausible-looking-but-wrong. `converged` lets callers
        # (e.g. Phase V6's VectorizedDynamicDOCREnv) detect this
        # per-environment and fall back rather than feed a physically
        # meaningless voltage into an observation vector.
        final_mismatch = mismatch_fn(x, Ybus_re, Ybus_im, P_spec, Q_spec)
        residual = final_mismatch.abs().amax(dim=1)
        converged = residual < 1e-6

        theta = torch.zeros((n_envs, self.n_bus), dtype=self.rdtype, device=self.device)
        theta[:, self.non_slack_idx] = x[:, : self.n_theta]

        V = torch.zeros((n_envs, self.n_bus), dtype=self.rdtype, device=self.device)
        V[:, self.slack_bus] = self.slack_vm
        if len(self.pv_bus0) > 0:
            V[:, self.pv_bus0] = self.pv_vm.unsqueeze(0).expand(n_envs, -1)
        V[:, self.pq_idx] = x[:, self.n_theta:]

        return V, theta, converged
