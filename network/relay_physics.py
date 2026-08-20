"""
relay_physics.py
────────────────
IDMT relay model (IEC 60255-151 / IEEE C37.112-2018)
Objective function, penalty computation, and constraint checking
shared by both GA-SQP and MARL.

Phase 1 Enhancement: Added bidirectional fault current support,
dynamic fault current calculation hooks, and topology-aware constraint checking.
"""

import numpy as np
from typing import Tuple, Dict, Optional
from network.network_data import NetworkData


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
    """
    ratio = If / PS
    if ratio <= 1.0:
        return TOP_MAX
    t = A_COEFF * TMS / (ratio ** B_COEFF - 1.0)
    return min(max(t, TOP_MIN), TOP_MAX)


def idmt_time_vec(If: np.ndarray, PS: np.ndarray,
                  TMS: np.ndarray) -> np.ndarray:
    """Vectorised IDMT over N relays."""
    ratio = If / PS
    safe  = np.where(ratio > 1.0, ratio, 1.0 + 1e-9)
    t     = A_COEFF * TMS / (safe ** B_COEFF - 1.0)
    t     = np.where(ratio > 1.0, t, TOP_MAX)
    return np.clip(t, TOP_MIN, TOP_MAX)


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
    """
    # Apply fault type scaling factor
    ft_scale = FAULT_TYPE_WEIGHTS.get(fault_type, 1.0)

    if topology_state and "fault_currents" in topology_state:
        # Dynamic fault currents from pandapower short-circuit analysis
        fc = topology_state["fault_currents"]
        t_pri = np.array([
            idmt_time(fc.get(f"pair_{i}_pri", p.If_pri * ft_scale), 
                     PS[p.primary], TMS[p.primary])
            for i, p in enumerate(net.pairs)
        ])
        t_bkp = np.array([
            idmt_time(fc.get(f"pair_{i}_bkp", p.If_bkp * ft_scale), 
                     PS[p.backup], TMS[p.backup])
            for i, p in enumerate(net.pairs)
        ])
    else:
        # Static fault currents (backward compatible)
        t_pri = np.array([
            idmt_time(p.If_pri * ft_scale, PS[p.primary], TMS[p.primary])
            for p in net.pairs
        ])
        t_bkp = np.array([
            idmt_time(p.If_bkp * ft_scale, PS[p.backup], TMS[p.backup])
            for p in net.pairs
        ])

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
