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
        if baseline_flows is not None:
            for line_idx, p in flows.items():
                base_p = baseline_flows.get(line_idx)
                # A reversal is a genuine sign flip, not noise around zero
                # on a lightly-loaded line -- require both sides to carry
                # a non-trivial flow (>10 kW) before calling it a reversal.
                if base_p is not None and abs(base_p) > 0.01 and abs(p) > 0.01:
                    if (base_p > 0) != (p > 0):
                        reversed_lines.append(line_idx)

        per_pv[pv] = {
            "converged": True,
            "line_flows_mw": flows,
            "bus_voltages_pu": voltages,
            "line_loadings_pct": loadings,
            "reversed_lines_vs_baseline": reversed_lines,
            "n_reversed": len(reversed_lines),
            "max_loading_pct": max(loadings.values()) if loadings else None,
            "min_voltage_pu": min(voltages.values()) if voltages else None,
        }

    return {
        "system": system,
        "n_lines": net_data.n_lines,
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
