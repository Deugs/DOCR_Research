"""
experiments/exp_reverse_power_flow.py
──────────────────────────────────────
Objective 1: quantify reverse power flow induced by PV penetration, on the
active thesis scope (8bus/14bus/39bus, 0/20/40/60% PV).

This was previously entirely absent from the codebase -- no module computed
or reported power-flow direction anywhere; "reverse power flow" appeared
only as a plot-label string. This is a standalone AC power-flow analysis
(genuinely solves pandapower's power flow at each PV level, not a live RL
observation feature -- see marl_env.py for that separate concern), reusing
PandapowerNetwork's existing update_pv_penetration()/run_power_flow()/
get_line_flows() rather than adding new physics.

Usage:
    python -m experiments.exp_reverse_power_flow
    # writes results/reverse_power_flow_analysis.json and prints a summary

    from experiments.exp_reverse_power_flow import analyze_system, run_all
    report = run_all()  # {system: {...}}, in memory
"""
import json
import logging
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent

PV_LEVELS = [0.0, 0.2, 0.4, 0.6]
SYSTEMS = ["8bus", "14bus", "39bus"]

# A "non-trivial" flow, below which a line is considered too lightly loaded
# for its sign (direction) to mean anything -- noise around zero on a
# near-unloaded line shouldn't be reported as a reversal, nor used as a
# baseline denominator for flow_ratio_vs_baseline (which would blow up
# toward +-inf for no physical reason). Single source of truth for both the
# existing reversal-detection check below and the new flow-ratio/closest-
# to-reversal instrumentation, which needs the exact same notion of
# "this baseline flow is real, not near-zero noise".
NONTRIVIAL_FLOW_MW = 0.01


def _model_fidelity(system: str, net_data) -> Dict:
    """Per-system disclosure block: where this system's power-flow model's
    line impedance/rating and load data actually came from, and whether
    loading %/voltage numbers are calibrated against a cited/real source or
    just an internally-consistent placeholder. Read once per system (not
    per PV level -- the underlying line_data doesn't change with PV) so a
    reader of the JSON doesn't have to cross-reference network_data.py's
    comments to know whether a given system's numbers are trustworthy."""
    line_data = net_data.line_data

    if not line_data or not line_data.get("lines"):
        # No line_data at all -- PandapowerNetwork._build_network() falls
        # back to a generic std_type cable + relay-adjacency/ring topology
        # (see its own inline comments). Topology/direction is still
        # meaningful (it comes from real relay-pair adjacency), but loading
        # %/flow magnitudes are not calibrated against any real line rating
        # or bus demand.
        return {
            "has_real_line_data": False,
            "line_data_source": ("none -- generic std_type cable proxy + "
                                  "relay-adjacency/ring topology fallback "
                                  "(PandapowerNetwork._build_network())"),
            "load_data_source": ("none -- Iload_max (a per-relay CT/"
                                  "protection pickup rating, not real bus "
                                  "demand) derived heuristic"),
            "data_provenance": None,
            "loading_percent_calibrated": False,
            "known_gaps": [
                "no real per-line impedance/rating data for this system",
                "no real per-bus load data for this system",
                "loading %/flow magnitudes are not calibrated -- only "
                "topology/direction structure should be trusted",
            ],
        }

    provenance = line_data.get("data_provenance")
    fidelity = {
        "has_real_line_data": True,
        "line_data_source": "real per-line impedance/rating table (NetworkData.line_data['lines'])",
        "load_data_source": ("real per-bus load table (NetworkData.line_data['loads'])"
                              if line_data.get("loads") else "none"),
        "data_provenance": provenance,
    }

    if provenance:
        # Explicit provenance metadata exists on this system's line_data
        # (currently 8bus) -- surface it directly rather than re-deriving
        # it, so this JSON can't drift out of sync with network_data.py.
        fidelity["loading_percent_calibrated"] = bool(
            provenance.get("loading_percent_calibrated", True)
        )
        known_gaps = []
        mva_prov = provenance.get("mva_limit")
        if mva_prov and mva_prov != "cited_verified":
            known_gaps.append(
                f"mva_limit provenance is '{mva_prov}', not a cited source -- "
                "line thermal ratings (and therefore loading %) are an "
                "engineering estimate, not published data."
            )
        fidelity["known_gaps"] = known_gaps
    else:
        # line_data is real (real impedance/rating/load), but this system
        # has no explicit data_provenance dict yet (currently 14bus/39bus).
        # Do NOT claim "verified" here -- sourcing is asserted in
        # network_data.py's code comments/citations only, not machine-
        # checkable from this JSON.
        fidelity["loading_percent_calibrated"] = None
        known_gaps = [
            "no explicit data_provenance dict on this system's line_data -- "
            "line/load sourcing is asserted in network_data.py's code "
            "comments/citations, not machine-verifiable from this JSON.",
        ]
        if system == "39bus":
            known_gaps.append(
                "per network_data.py's own comments, this system's solved "
                "power flow closely matches pandapower.networks.case39()'s "
                "reference solve (~73.4% max loading, ~0.982 min V pu) -- "
                "the one active-scope system with an independent reference "
                "cross-check, though that check is not re-run by this script."
            )
        if system == "14bus":
            known_gaps.extend([
                "empty net.gen -- generator buses are modeled as fixed PQ "
                "injections (via sgen), not voltage-controlled generation, "
                "so there's no real reactive/voltage support at gen buses",
                "missing the real IEEE 14-bus case's bus-9 shunt capacitor",
                "the 3 transformer branches (buses 4-7, 4-9, 5-6) use "
                "ratio=0.0 (nominal tap only) -- no off-nominal tap modeled",
                "mva_limit values for this system's lines are invented, "
                "not from a cited source",
                "known to produce a min voltage of ~0.71pu vs the real "
                "system's ~1.01pu -- loading %/voltage are NOT "
                "independently validated for this system (see Part 2e).",
            ])
        fidelity["known_gaps"] = known_gaps

    return fidelity


def analyze_system(system: str) -> Dict:
    """Sweep PV penetration on `system`, solving real AC power flow at
    each level, and report which lines reverse direction relative to the
    0%-PV baseline (a sign flip in p_from_mw)."""
    from network.network_data import NETWORKS
    from network.pandapower_network import PandapowerNetwork, PANDAPOWER_AVAILABLE

    if not PANDAPOWER_AVAILABLE:
        raise RuntimeError(
            "pandapower not installed -- reverse-power-flow analysis "
            "requires a real AC power-flow solve, there is no analytical "
            "fallback for flow direction (only for fault-current magnitude)."
        )

    net_data = NETWORKS[system]()
    pp_net = PandapowerNetwork(net_data)

    # Real pandapower branch counts, straight from the solved model --
    # topology is fixed once built (doesn't change across the PV sweep),
    # so read it once here rather than per PV level. Kept alongside (not
    # instead of) net_data.n_lines: for 8bus specifically, the relay-
    # coordination graph (net_data.n_lines) is deliberately smaller than
    # the real line count pandapower actually solves power flow on
    # (net_data.line_data is None for 8bus today, so PandapowerNetwork
    # falls back to a relay-adjacency-derived topology that doesn't equal
    # the real 7-line published system) -- a reader relying on
    # net_data.n_lines alone would be misled about what line_flows_mw/
    # line_loadings_pct below are actually indexed against.
    n_lines_pandapower = len(pp_net.pp_net.line) if pp_net.pp_net is not None else None
    n_trafo_pandapower = len(pp_net.pp_net.trafo) if pp_net.pp_net is not None else None

    # Reverse lookup (pandapower internal bus id -> this network's own
    # 0-indexed bus numbering) so per-DG-bus local load can be read back
    # off the solved net.load table by real bus number.
    pp_bus_to_bus0 = {pp_bus: bus0 for bus0, pp_bus in pp_net.bus_map.items()}

    per_pv = {}
    baseline_flows = None
    for pv in PV_LEVELS:
        pp_net.update_pv_penetration(pv)
        converged = pp_net.run_power_flow()
        if not converged:
            logger.warning(f"{system}: power flow did not converge at PV={pv:.0%}")
            per_pv[pv] = {"converged": False}
            continue

        flows = pp_net.get_line_flows()
        voltages = pp_net.get_bus_voltages()
        loadings = pp_net.get_line_loadings()

        if pv == 0.0:
            baseline_flows = flows

        reversed_lines: List[int] = []
        flow_ratio_vs_baseline: Dict[int, float] = {}
        if baseline_flows is not None:
            for line_idx, p in flows.items():
                base_p = baseline_flows.get(line_idx)
                # A reversal is a genuine sign flip, not noise around zero
                # on a lightly-loaded line -- require both sides to carry
                # a non-trivial flow before calling it a reversal.
                if base_p is not None and abs(base_p) > NONTRIVIAL_FLOW_MW:
                    if abs(p) > NONTRIVIAL_FLOW_MW and (base_p > 0) != (p > 0):
                        reversed_lines.append(line_idx)
                    # Ratio of this level's flow magnitude to baseline's,
                    # for every line with a non-trivial baseline (same
                    # threshold as above) -- not gated on abs(p) also being
                    # non-trivial, since a ratio trending toward zero *is*
                    # the signal that a line is approaching reversal, even
                    # before it actually crosses zero.
                    flow_ratio_vs_baseline[line_idx] = abs(p) / abs(base_p)

        min_flow_ratio = None
        closest_to_reversal_line = None
        if flow_ratio_vs_baseline:
            closest_idx = min(flow_ratio_vs_baseline, key=flow_ratio_vs_baseline.get)
            min_flow_ratio = flow_ratio_vs_baseline[closest_idx]
            closest_to_reversal_line = {
                "line_idx": closest_idx,
                "flow_ratio": min_flow_ratio,
                "flow_mw_at_this_level": flows.get(closest_idx),
                "flow_mw_at_baseline": baseline_flows.get(closest_idx),
            }

        # System-wide PV/load totals at this level, read straight off the
        # solved pandapower tables (not re-derived from NetworkData) so
        # they reflect exactly what power flow actually saw -- including
        # 8bus's fallback per-relay load-heuristic path, which has no
        # line_data["loads"] table to read from at all.
        total_load_mw = (float(pp_net.pp_net.load["p_mw"].sum())
                          if len(pp_net.pp_net.load) else 0.0)
        # Only genuine distributed-PV sgens (always_on=False DGs) land in
        # net.sgen -- always_on generators (e.g. 39bus's real synchronous
        # machines) are modeled as net.gen instead and are intentionally
        # excluded here: they don't scale with pv_level, and counting them
        # as "PV" would misrepresent the actual distributed-PV sweep this
        # analysis is about.
        total_pv_mw = (float(pp_net.pp_net.sgen["p_mw"].sum())
                        if len(pp_net.pp_net.sgen) else 0.0)
        pv_to_load_ratio = (total_pv_mw / total_load_mw) if total_load_mw > 0 else None

        # Per-bus load, aggregated straight from the solved net.load table
        # (handles both the real line_data["loads"] path and 8bus's
        # multi-row-per-bus fallback path uniformly) and mapped back to
        # this network's own 0-indexed bus numbering.
        local_load_by_bus0: Dict[int, float] = {}
        if len(pp_net.pp_net.load):
            for pp_bus, p_mw in pp_net.pp_net.load.groupby("bus")["p_mw"].sum().items():
                bus0 = pp_bus_to_bus0.get(pp_bus)
                if bus0 is not None:
                    local_load_by_bus0[bus0] = float(p_mw)

        per_dg_bus_pv_to_local_load: Dict[int, float] = {}
        for dg in net_data.dg_buses:
            if dg.always_on:
                continue  # not distributed PV -- see total_pv_mw's comment above
            local_load = local_load_by_bus0.get(dg.bus_idx - 1)
            if local_load:  # skip DG buses with no (or zero) matching local load
                per_dg_bus_pv_to_local_load[dg.bus_idx] = (dg.p_mw * pv) / local_load

        # Real power supplied by the external grid / slack bus at this PV
        # level. No existing PandapowerNetwork accessor exposes this, so
        # read pandapower's own results table directly.
        slack_p_mw = (float(pp_net.pp_net.res_ext_grid["p_mw"].sum())
                      if pp_net.pp_net is not None and len(pp_net.pp_net.res_ext_grid)
                      else None)

        per_pv[pv] = {
            "converged": True,
            "line_flows_mw": flows,
            "bus_voltages_pu": voltages,
            "line_loadings_pct": loadings,
            "reversed_lines_vs_baseline": reversed_lines,
            "n_reversed": len(reversed_lines),
            "max_loading_pct": max(loadings.values()) if loadings else None,
            "min_voltage_pu": min(voltages.values()) if voltages else None,
            "flow_ratio_vs_baseline": flow_ratio_vs_baseline,
            "min_flow_ratio": min_flow_ratio,
            "closest_to_reversal_line": closest_to_reversal_line,
            "slack_p_mw": slack_p_mw,
            "total_pv_mw": total_pv_mw,
            "total_load_mw": total_load_mw,
            "pv_to_load_ratio": pv_to_load_ratio,
            "per_dg_bus_pv_to_local_load": per_dg_bus_pv_to_local_load,
        }

    return {
        "system": system,
        "n_lines": net_data.n_lines,
        "n_lines_pandapower": n_lines_pandapower,
        "n_trafo_pandapower": n_trafo_pandapower,
        "model_fidelity": _model_fidelity(system, net_data),
        "pv_levels": PV_LEVELS,
        "per_pv": per_pv,
    }


def run_all() -> Dict[str, Dict]:
    return {system: analyze_system(system) for system in SYSTEMS}


def write_report(output_path: str = None) -> str:
    if output_path is None:
        output_path = str(REPO_ROOT / "results" / "reverse_power_flow_analysis.json")
    report = run_all()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info(f"Reverse power flow analysis: wrote {out}")
    return str(out)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    path = write_report()
    print(f"\nWrote {path}\n")

    with open(path) as f:
        report = json.load(f)

    print(f"{'System':<8} {'PV':>5} {'Lines Reversed':>16} {'Max Loading %':>15} {'Min V (pu)':>12}")
    print("-" * 62)
    for system, data in report.items():
        for pv_str, pv_data in data["per_pv"].items():
            if not pv_data.get("converged"):
                print(f"{system:<8} {float(pv_str):>4.0%}  DID NOT CONVERGE")
                continue
            print(f"{system:<8} {float(pv_str):>4.0%} {pv_data['n_reversed']:>16d} "
                  f"{pv_data['max_loading_pct']:>15.1f} {pv_data['min_voltage_pu']:>12.4f}")
