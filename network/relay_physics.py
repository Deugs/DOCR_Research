"""
relay_physics.py
────────────────
IDMT relay model (IEC 60255-151 / IEEE C37.112-2018)
Objective function, penalty computation, and constraint checking
shared by both GA-SQP and MARL.

Phase 1 Enhancement: Added bidirectional fault current support,
dynamic fault current calculation hooks, and topology-aware constraint checking.

2026-10-26 torch port: `compute_operating_times` previously called the
scalar, branching `idmt_time()` once per relay pair in a Python list
comprehension -- the actual bottleneck objective_function_batch_gpu already
worked around for its own population-batched case. This file now routes
that same formula through one torch core (`_idmt_time_vec_t`) for every
caller, single-candidate or batched, under `torch.no_grad()`, NumPy in/out
so none of this module's ~40+ call sites need to change. Device selection
reuses marl.marl_agents' DOCR_USE_GPU=1 opt-in policy (GPU does not
auto-activate -- see that module's docstring for why: this project has
twice measured GPU as *slower* than CPU for small per-call workloads like
this one, so the policy is "available, not assumed"). `import torch` stays
lazy (inside functions), never module-level -- this file is sometimes
imported before pandapower in real call chains (baselines/ga_sqp.py and
baselines/metaheuristics.py import only network.network_data +
network.relay_physics, never pandapower), and torch-before-pandapower
segfaults with no diagnostic on at least one dev machine used on this
project (see marl/marl_agents.py's module docstring).
"""

import numpy as np
from typing import Tuple, Dict, Optional
from network.network_data import NetworkData


def _torch_device():
    """Resolve the compute device under the project's DOCR_USE_GPU=1
    opt-in policy. Delegates to marl.marl_agents._default_device() when
    importable (the single source of truth for this policy); falls back to
    an inline equivalent for standalone contexts where importing the marl
    package isn't possible (e.g. a bare network-only script) or would risk
    a circular import. `import torch` here is still lazy -- this function
    is only ever called from inside other lazy-import functions below."""
    import torch
    try:
        from marl.marl_agents import _default_device
        return _default_device()
    except ImportError:
        import os
        cuda_present = torch.cuda.is_available()
        gpu_available = cuda_present and os.environ.get("DOCR_USE_GPU") == "1"
        return torch.device("cuda") if gpu_available else torch.device("cpu")


def _pair_index_arrays(net: NetworkData):
    """Cache (primary_idx, backup_idx, If_pri_base, If_bkp_base) on the
    NetworkData instance itself. Safe to cache: net.pairs is never mutated
    in place on an existing instance (confirmed 2026-10-26 -- the only
    in-place If_pri/If_bkp rescaling, in NetworkData.apply_n1_outage(),
    operates on a fresh self.copy() built via the constructor, which starts
    without this cache attribute and so recomputes it correctly from the
    mutated copy's pairs on first use). Replaces the per-call Python list
    comprehension that used to rebuild these same arrays from net.pairs on
    every single call to compute_operating_times."""
    cache = getattr(net, "_relay_physics_index_cache", None)
    if cache is not None:
        return cache
    primary = np.array([p.primary for p in net.pairs], dtype=np.int64)
    backup = np.array([p.backup for p in net.pairs], dtype=np.int64)
    if_pri_base = np.array([p.If_pri for p in net.pairs], dtype=np.float64)
    if_bkp_base = np.array([p.If_bkp for p in net.pairs], dtype=np.float64)
    cache = (primary, backup, if_pri_base, if_bkp_base)
    net._relay_physics_index_cache = cache
    return cache


# ── IEC Standard Inverse (IDMT) constants ────────────────────────────────────
A_COEFF = 0.14
B_COEFF = 0.02

TMS_MIN = 0.01
TMS_MAX = 1.10
CTI     = 0.30  # Phase 1: Unified stricter standard (0.3s)
TOP_MIN = 0.05
TOP_MAX = 10.0

# Fault types for generalization testing
FAULT_TYPES = ["3ph", "LL", "1ph", "2ph-g"]
FAULT_TYPE_WEIGHTS = {"3ph": 1.0, "LL": 0.87, "1ph": 0.58, "2ph-g": 0.75}


def idmt_time(If: float, PS: float, TMS: float) -> float:
    """
    t = A * TMS / ((If/PS)^B - 1)
    Returns TOP_MAX when relay would not trip (If <= PS), and never
    returns a value outside [TOP_MIN, TOP_MAX] — the standard's floor
    on realistic operate time. This scalar version previously enforced
    no floor at all (only its unused vectorized twin idmt_time_vec()
    did, via np.clip), so unconstrained TMS/PS at their legal
    RL-action-space boundary values could — and empirically did, on
    100% of pairs at TMS_MIN/ps_min on 8bus and 39bus — drive computed
    operating times below what IEC 60255-151/IEEE C37.112 permits,
    letting training/optimization chase infeasible sub-floor "operating
    times" through every objective_function/CTI-margin computation that
    calls this.

    Kept as a plain Python scalar (not torch) -- tests/test_phase1.py calls
    it directly for exact-formula regression checks, and compute_operating_times
    below no longer calls it in a per-pair loop (see _idmt_time_vec_t), so
    there's no hot-path reason to change it.
    """
    ratio = If / PS
    if ratio <= 1.0:
        return TOP_MAX
    t = A_COEFF * TMS / (ratio ** B_COEFF - 1.0)
    return min(max(t, TOP_MIN), TOP_MAX)


def _idmt_time_tensor(If_t, PS_t, TMS_t):
    """Pure-tensor IDMT formula, no numpy conversion -- shared core for
    both the NumPy-in/NumPy-out single-candidate path (_idmt_time_vec_t
    below) and the batched population path (objective_function_batch),
    which chains further tensor ops (sum/penalty reduction) on top of this
    before converting back to NumPy once at the very end, instead of
    round-tripping through NumPy after every intermediate step -- the
    difference matters for the batched path specifically, since repeated
    host-device syncs are exactly the kind of overhead that determines
    whether GPU can ever win for a workload this size (see module
    docstring's note on this project's prior GPU-slower-than-CPU findings).
    Broadcasts normally (e.g. a (n_pairs,) fault-current array against a
    (pop_size, n_pairs) PS/TMS array), same as plain NumPy broadcasting.
    """
    import torch
    ratio = If_t / PS_t
    safe = torch.where(ratio > 1.0, ratio, torch.full_like(ratio, 1.0 + 1e-9))
    t = A_COEFF * TMS_t / (safe ** B_COEFF - 1.0)
    t = torch.where(ratio > 1.0, t, torch.full_like(t, TOP_MAX))
    return torch.clamp(t, TOP_MIN, TOP_MAX)


def _idmt_time_vec_t(If_batch, PS_batch, TMS_batch, device) -> np.ndarray:
    """NumPy-in/NumPy-out wrapper around _idmt_time_tensor, for the
    single-candidate hot path used by compute_operating_times (and the
    standalone idmt_time_vec() below). If_batch/PS_batch/TMS_batch: numpy
    arrays of identical shape, typically (n_pairs,). `import torch` is
    lazy (module docstring).

    float64, not float32: the pre-port NumPy implementation ran in
    float64 throughout. A first equivalence pass at float32 measured up
    to 0.14 absolute error on objective_function's penalty term on 39-bus
    (alpha2=100/beta=50 amplify tiny near-the-CTI-boundary float32
    rounding after squaring) -- a real precision regression, not just
    noise, given this value drives RL reward and metaheuristic fitness
    ranking. float64 on CPU costs nothing meaningful here (these are
    14-92 element arrays); on GPU it's slower than float32, but per the
    module docstring this workload isn't expected to benefit from GPU
    regardless, so correctness wins the tradeoff outright.
    """
    import torch
    If_t = torch.as_tensor(If_batch, dtype=torch.float64, device=device)
    PS_t = torch.as_tensor(PS_batch, dtype=torch.float64, device=device)
    TMS_t = torch.as_tensor(TMS_batch, dtype=torch.float64, device=device)
    with torch.no_grad():
        t = _idmt_time_tensor(If_t, PS_t, TMS_t)
    return t.cpu().numpy()


def idmt_time_vec(If: np.ndarray, PS: np.ndarray,
                  TMS: np.ndarray) -> np.ndarray:
    """Vectorised IDMT over N relays. No real callers found in this
    codebase as of 2026-10-26 (grep confirmed) -- kept for API
    compatibility, now routed through the same torch core as every other
    function in this module instead of its own separate NumPy formula."""
    return _idmt_time_vec_t(If, PS, TMS, _torch_device())


def compute_operating_times(
    net: NetworkData,
    TMS: np.ndarray,
    PS:  np.ndarray,
    fault_type: str = "3ph",
    topology_state: Optional[Dict] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (t_pri, t_bkp) for every P/B pair.

    Phase 1: Supports multiple fault types and topology-aware fault currents.
    If topology_state is provided, uses dynamic fault currents from pandapower.
    Otherwise falls back to static values from NetworkData.

    2026-10-26: previously built t_pri/t_bkp via a Python list comprehension
    calling the scalar, branching idmt_time() once per relay pair -- the
    real per-call bottleneck this module's torch port targets. Now gathers
    PS/TMS by cached primary/backup index arrays (_pair_index_arrays) and
    evaluates the whole pair array in one torch call (_idmt_time_vec_t).
    The topology_state branch still does one small Python loop over
    net.pairs purely to pull values out of the fault_currents dict (dicts
    can't be vectorized away) -- but no longer calls idmt_time() inside
    that loop, which was the actual cost.
    """
    ft_scale = FAULT_TYPE_WEIGHTS.get(fault_type, 1.0)
    primary, backup, if_pri_base, if_bkp_base = _pair_index_arrays(net)
    device = _torch_device()

    if topology_state and "fault_currents" in topology_state:
        # Dynamic fault currents from pandapower short-circuit analysis.
        fc = topology_state["fault_currents"]
        If_pri = np.array([
            fc.get(f"pair_{i}_pri", if_pri_base[i] * ft_scale)
            for i in range(len(net.pairs))
        ])
        If_bkp = np.array([
            fc.get(f"pair_{i}_bkp", if_bkp_base[i] * ft_scale)
            for i in range(len(net.pairs))
        ])
    else:
        # Static fault currents (backward compatible).
        If_pri = if_pri_base * ft_scale
        If_bkp = if_bkp_base * ft_scale

    t_pri = _idmt_time_vec_t(If_pri, PS[primary], TMS[primary], device)
    t_bkp = _idmt_time_vec_t(If_bkp, PS[backup], TMS[backup], device)
    return t_pri, t_bkp


def objective_function(
    net:    NetworkData,
    TMS:    np.ndarray,
    PS:     np.ndarray,
    alpha1: float = 1.0,
    alpha2: float = 100.0,
    beta:   float = 50.0,
    fault_type: str = "3ph",
    topology_state: Optional[Dict] = None,
) -> Tuple[float, float, float]:
    """
    Eq. (9): OF = Σ T_ik  +  α1 * Σ Penalty²
    Returns (OF, sum_top, total_penalty).
    """
    t_pri, t_bkp = compute_operating_times(net, TMS, PS, fault_type, topology_state)
    sum_top = float(np.sum(t_pri))

    delta        = t_bkp - t_pri - CTI
    viol_coord   = np.sum(np.minimum(delta, 0.0) ** 2)
    viol_mintime = np.sum(np.minimum(t_bkp - CTI, 0.0) ** 2)

    penalty = alpha2 * viol_coord + beta * viol_mintime
    OF = sum_top + alpha1 * penalty
    return OF, sum_top, penalty


def objective_function_batch(net, TMS_batch, PS_batch, alpha1=1.0, alpha2=100.0,
                              beta=50.0, fault_type="3ph", device=None,
                              fault_currents_batch=None):
    """Batched counterpart of objective_function(): evaluates a whole
    POPULATION of (TMS, PS) candidates at once via torch tensors, for
    population-based metaheuristics (baselines/ga_sqp.py,
    baselines/metaheuristics.py) and (2026-11-xx, Phase V3 of the
    GPU-vectorized-environment plan) for per-environment dynamic fault
    currents in the vectorized training path. 2026-10-26: unifies what was
    previously a separate, GPU-only function (objective_function_batch_gpu,
    kept below as a thin backward-compatible alias) with the rest of this
    module's torch core (_idmt_time_tensor) -- same math, now device-gated
    via the project's DOCR_USE_GPU=1 opt-in policy by default instead of
    hardcoding CPU, consistent with every other torch-backed function here.

    `fault_currents_batch`: optional `(If_pri_batch, If_bkp_batch)` pair,
    each `(P, n_pairs)` -- one row per candidate/environment's own dynamic
    fault currents (e.g. topology/PV-state-dependent), overriding the
    default shared static `(n_pairs,)` vector broadcast to every row. When
    given, these values are used AS-IS, with no additional `ft_scale`
    multiplication -- matching compute_operating_times()'s single-candidate
    dynamic-fault-current path, where fault_type scaling is already baked
    into whatever produced the dict values (relay_physics.
    compute_fault_currents_dynamic()), not reapplied on top. When None
    (the original, still-default behavior), every row shares one static
    `(n_pairs,)` vector scaled by `ft_scale` -- unchanged from before this
    parameter existed, so every existing caller (GA-SQP, metaheuristics)
    needs no changes.

    TMS_batch/PS_batch: (pop_size, n_relays) numpy arrays. Returns
    (pop_size,) numpy array of OF values.

    Stays in torch tensors from input to final reduction, converting to
    NumPy only once at the end -- unlike compute_operating_times' NumPy-out
    _idmt_time_vec_t wrapper, repeated host-device round trips here would
    undercut the whole point of batching a population for GPU (see
    _idmt_time_tensor's docstring). `import torch` is lazy (module
    docstring's import-order note).
    """
    import torch
    primary, backup, if_pri_base, if_bkp_base = _pair_index_arrays(net)
    ft_scale = FAULT_TYPE_WEIGHTS.get(fault_type, 1.0)
    dev = torch.device(device) if device is not None else _torch_device()

    # float64, not float32 -- see _idmt_time_vec_t's docstring: alpha2/beta
    # (default 100/50) amplify tiny float32 rounding in the squared penalty
    # terms into real precision loss on the returned OF value.
    TMS_t = torch.as_tensor(TMS_batch, dtype=torch.float64, device=dev)  # (P, n_relays)
    PS_t = torch.as_tensor(PS_batch, dtype=torch.float64, device=dev)
    primary_t = torch.as_tensor(primary, dtype=torch.long, device=dev)
    backup_t = torch.as_tensor(backup, dtype=torch.long, device=dev)

    if fault_currents_batch is not None:
        If_pri_dyn_batch, If_bkp_dyn_batch = fault_currents_batch
        If_pri_t = torch.as_tensor(If_pri_dyn_batch, dtype=torch.float64, device=dev)  # (P, n_pairs)
        If_bkp_t = torch.as_tensor(If_bkp_dyn_batch, dtype=torch.float64, device=dev)
    else:
        If_pri_t = torch.as_tensor(if_pri_base * ft_scale, dtype=torch.float64, device=dev)  # (n_pairs,)
        If_bkp_t = torch.as_tensor(if_bkp_base * ft_scale, dtype=torch.float64, device=dev)

    with torch.no_grad():
        # If_pri_t/If_bkp_t broadcast against (P, n_pairs) PS/TMS
        # selections -- either a shared (n_pairs,) vector (static path,
        # same semantics as plain NumPy broadcasting) or an already
        # per-row (P, n_pairs) tensor (dynamic path), both handled
        # identically by ordinary broadcasting/indexing rules.
        t_pri = _idmt_time_tensor(If_pri_t, PS_t[:, primary_t], TMS_t[:, primary_t])
        t_bkp = _idmt_time_tensor(If_bkp_t, PS_t[:, backup_t], TMS_t[:, backup_t])

        sum_top = t_pri.sum(dim=1)  # (P,)
        delta = t_bkp - t_pri - CTI
        viol_coord = torch.clamp(delta, max=0.0).pow(2).sum(dim=1)
        viol_mintime = torch.clamp(t_bkp - CTI, max=0.0).pow(2).sum(dim=1)
        penalty = alpha2 * viol_coord + beta * viol_mintime
        OF = sum_top + alpha1 * penalty
    return OF.cpu().numpy()


def objective_function_batch_gpu(net, TMS_batch, PS_batch, alpha1=1.0, alpha2=100.0,
                                  beta=50.0, fault_type="3ph", device=None):
    """Backward-compatible alias for objective_function_batch() -- kept
    under its original name since any existing caller/test may still
    reference it directly by this name. New code should call
    objective_function_batch() instead. Unlike the original version of
    this function (pre-2026-10-26), device=None now resolves via the
    project's DOCR_USE_GPU=1 opt-in policy instead of hardcoding CPU."""
    return objective_function_batch(net, TMS_batch, PS_batch, alpha1, alpha2,
                                     beta, fault_type, device)


def check_constraints(
    net: NetworkData,
    TMS: np.ndarray,
    PS:  np.ndarray,
    fault_type: str = "3ph",
    topology_state: Optional[Dict] = None,
) -> dict:
    """Return constraint satisfaction metrics dict."""
    t_pri, t_bkp = compute_operating_times(net, TMS, PS, fault_type, topology_state)
    delta = t_bkp - t_pri

    cti_ok = delta >= CTI - 1e-6
    ps_min, ps_max = net.ps_bounds()

    return {
        "cti_satisfied":  bool(np.all(cti_ok)),
        "cti_violations": int(np.sum(~cti_ok)),
        "cti_min":        float(np.min(delta)),
        "cti_mean":       float(np.mean(delta)),
        "top_pri_mean":   float(np.mean(t_pri)),
        "top_pri_sum":    float(np.sum(t_pri)),
        "tms_ok":         bool(np.all((TMS >= TMS_MIN-1e-6) & (TMS <= TMS_MAX+1e-6))),
        "ps_ok":          bool(np.all((PS >= ps_min-1e-6) & (PS <= ps_max+1e-6))),
        "t_pri":          t_pri,
        "t_bkp":          t_bkp,
        "delta_cti":      delta,
        "fault_type":     fault_type,
    }


def compute_fault_currents_dynamic(
    net: NetworkData,
    pv_penetration: float = 0.0,
    fault_type: str = "3ph",
    topology_changes: Optional[Dict] = None,
) -> Dict[str, float]:
    """
    Phase 1: Compute dynamic fault currents considering PV inverter contribution.

    This is a hook for pandapower short-circuit calculation.
    When pandapower is available, it will compute actual fault currents.
    Otherwise, uses analytical approximation with PV scaling.

    Parameters:
        net: NetworkData instance
        pv_penetration: 0.0 to 1.0 (fraction of installed PV capacity)
        fault_type: Type of fault (3ph, LL, 1ph, 2ph-g)
        topology_changes: Dict with line outages, bus splits, etc.

    Returns:
        Dict mapping fault current identifiers to values
    """
    fc = {}
    ft_scale = FAULT_TYPE_WEIGHTS.get(fault_type, 1.0)

    # PV inverter fault current contribution model.
    # This is a simplified linear approximation of PV fault-current contribution
    # during low-voltage ride-through (LVRT), loosely informed by the general
    # shape of grid-code fault ride-through requirements (see e.g. IEEE 1547 /
    # IEC 61727 for the standards that actually govern grid-connected PV fault
    # behavior). It is NOT a literal implementation of any single standard's
    # current-injection formula.
    pv_contribution_factor = 1.0 + 0.3 * pv_penetration  # 1.0 to 1.3

    for i, p in enumerate(net.pairs):
        # Base fault current (from network data)
        If_pri_base = p.If_pri * ft_scale
        If_bkp_base = p.If_bkp * ft_scale

        # Add PV contribution if DG buses are near the fault location
        # Simplified: assume PV increases fault current by factor
        If_pri_dyn = If_pri_base * pv_contribution_factor
        If_bkp_dyn = If_bkp_base * pv_contribution_factor

        # Apply topology changes (line outages reduce fault current)
        if topology_changes:
            outage_factor = topology_changes.get("outage_factor", 1.0)
            If_pri_dyn *= outage_factor
            If_bkp_dyn *= outage_factor

        fc[f"pair_{i}_pri"] = If_pri_dyn
        fc[f"pair_{i}_bkp"] = If_bkp_dyn

    return fc
