"""
tests/test_pandapower_model.py
-------------------------------
First use_pandapower=True test coverage in this suite: exercises
PandapowerNetwork directly (real AC power flow / model construction),
rather than the analytical fallback path every other test file targets.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import numpy as np

from network.network_data import NETWORKS, get_8bus, get_14bus, get_39bus
from network.pandapower_network import PandapowerNetwork, PANDAPOWER_AVAILABLE

pytestmark = pytest.mark.skipif(not PANDAPOWER_AVAILABLE, reason="pandapower not installed")


# 118bus is deliberately excluded from every fallback-path check below:
# pandapower's runpp() on it is noticeably slower than the other systems
# (a real, meaningful fraction of this unit-test suite's total runtime)
# and its own dedicated coverage lives in the broader experiment scripts,
# not this fast unit-test file.
FALLBACK_PATH_SYSTEMS = ["3bus", "8bus", "30bus", "57bus"]


@pytest.mark.parametrize("system", FALLBACK_PATH_SYSTEMS)
def test_no_duplicate_load_rows_per_bus(system):
    """2a's regression guard: the fallback (no line_data["loads"]) load-
    construction path must aggregate by bus (max() across the relays
    located there), not create one pp.create_load() row per relay -- the
    original bug summed several relays' independent CT/protection
    pickup ratings at one bus as if they were additive loads, multiply-
    counting demand at any bus with more than one incident relay
    (verified pre-fix: 3 of 8bus's 8 buses were triple-counted this way,
    part of what pushed max line loading to 865-886%)."""
    net_data = NETWORKS[system]()
    pn = PandapowerNetwork(net_data)
    assert pn.pp_net is not None
    assert pn.pp_net.load.bus.is_unique, (
        f"{system}: duplicate load rows on the same bus -- the fallback "
        f"load-construction path is not aggregating by bus"
    )
    if system == "8bus":
        # 8bus now has real line_data with exactly 4 real load buses
        # (2/3/4/5) -- it no longer reaches the fallback path at all, so
        # this also locks in that it takes the real-load-table path.
        assert len(pn.pp_net.load) == 4, (
            f"8bus: expected 4 real load rows (buses 2/3/4/5), got "
            f"{len(pn.pp_net.load)} -- is it still on the real-data path?"
        )
    print(f"  {system}: {len(pn.pp_net.load)} load row(s), one per bus")


def test_8bus_line_ratings_are_plausible():
    """8bus must be on the real-parameter power-flow path
    (create_line_from_parameters with explicit r/x/max_i_ka), not the
    generic voltage-threshold std_type cable fallback that previously
    modeled its 150kV subtransmission lines with an 11-20kV-rated cable
    -- the root cause of the 865-886% loading figures this fix
    addresses."""
    net_data = get_8bus()
    pn = PandapowerNetwork(net_data)
    assert len(pn.pp_net.line) > 0
    assert pn.pp_net.line.std_type.isna().all(), (
        "8bus lines have a std_type set -- they are on the generic "
        "cable-type fallback, not the real-parameter path"
    )
    print(f"  8bus: {len(pn.pp_net.line)} lines on the real-parameter "
          f"path (std_type unset for all)")


@pytest.mark.parametrize("system", [
    "8bus",
    "39bus",
    pytest.param(
        "14bus",
        marks=pytest.mark.xfail(
            strict=True,
            reason=(
                "known pre-existing model gap (out of scope for this pass): "
                "empty net.gen (generator buses are fixed-PQ sgens, not "
                "voltage-controlled generation), the real system's bus-9 "
                "shunt capacitor is not modeled, and the 3 transformer "
                "branches (buses 4-7, 4-9, 5-6) use ratio=0.0 (nominal tap "
                "only, no off-nominal tap) -- together these produce a min "
                "voltage of ~0.71pu vs the real system's ~1.01pu."
            ),
        ),
    ),
])
def test_power_flow_converges_and_loading_bounded(system):
    """At PV=0% (fresh construction, no update_pv_penetration() call),
    power flow must converge with a physically sensible max line
    loading and min bus voltage. 14bus's min-voltage bound is a
    documented pre-existing gap (see the xfail reason above), not
    loosened or skipped."""
    net_data = NETWORKS[system]()
    pn = PandapowerNetwork(net_data)
    converged = pn.run_power_flow()
    assert converged, f"{system}: power flow did not converge at PV=0%"

    max_loading = float(pn.pp_net.res_line.loading_percent.max())
    assert max_loading < 150.0, f"{system}: max loading {max_loading:.1f}% >= 150%"

    min_v = float(pn.pp_net.res_bus.vm_pu.min())
    assert min_v > 0.90, f"{system}: min voltage {min_v:.4f}pu <= 0.90pu"
    print(f"  {system}: converged, max_loading={max_loading:.1f}%, min_v={min_v:.4f}pu")


@pytest.mark.parametrize("system", ["39bus", "14bus"])
def test_dg_sizing_is_sane(system):
    """Guard against ever silently inflating distributed-PV nameplate
    capacity to manufacture reversals (rather than a precise physical
    claim): every genuine distributed-PV DG bus (always_on=False) with a
    real matching local-load entry must stay well below that bus's own
    real load. Current data peaks at ~18.4% of local load (14bus bus 2,
    ~10% for 39bus); 50% is a round threshold with comfortable margin
    above today's real values."""
    net_data = NETWORKS[system]()
    assert net_data.line_data and net_data.line_data.get("loads")
    loads_by_bus = {ld["bus"]: ld["p_mw"] for ld in net_data.line_data["loads"]}

    checked = 0
    for dg in net_data.dg_buses:
        if dg.always_on:
            continue  # not distributed PV -- a real, fixed-dispatch generator
        local_load = loads_by_bus.get(dg.bus_idx)
        if not local_load:
            continue
        checked += 1
        assert dg.p_mw <= 0.5 * local_load, (
            f"{system}: DG at bus {dg.bus_idx} ({dg.p_mw} MW) exceeds 50% "
            f"of its local load ({local_load} MW)"
        )
    assert checked > 0, (
        f"{system}: no distributed-PV DG bus had a matching local load "
        f"entry to check -- this test would otherwise pass vacuously"
    )
    print(f"  {system}: {checked} distributed-PV DG bus(es), all <= 50% of local load")


def test_14bus_dg_not_on_slack_bus():
    """14bus's distributed PV was relocated off the slack bus (Part 2c
    -- it used to sit there and be invisibly absorbed by the ideal ext_
    grid source). Resolve the slack bus exactly as
    PandapowerNetwork._build_network() does: line_data is present but
    carries no explicit "slack_bus" key, so it defaults to bus 0
    (1-indexed bus 1). Should now PASS (not xfail) -- confirmed by
    running it, not assumed from the plan text."""
    net_data = get_14bus()
    assert net_data.line_data is not None
    slack_bus_1indexed = net_data.line_data.get("slack_bus", 1)

    assert len(net_data.dg_buses) > 0
    for dg in net_data.dg_buses:
        assert dg.bus_idx != slack_bus_1indexed, (
            f"DG at bus {dg.bus_idx} sits on the resolved slack bus "
            f"{slack_bus_1indexed}"
        )
    print(f"  14bus: resolved slack bus={slack_bus_1indexed}, no DG bus "
          f"matches it ({len(net_data.dg_buses)} DG buses checked)")


def test_39bus_matches_pandapower_reference():
    """39bus's solved power flow should closely match pandapower's own
    built-in case39() reference solve -- independently cross-checked:
    both converge to ~73.4% max line loading and ~0.982pu min bus
    voltage. Skips gracefully if the installed pandapower version has no
    networks.case39()."""
    try:
        import pandapower as pp
        import pandapower.networks as pn_nets
    except ImportError:
        pytest.skip("pandapower not installed")
    if not hasattr(pn_nets, "case39"):
        pytest.skip("installed pandapower version has no networks.case39()")

    ref_net = pn_nets.case39()
    pp.runpp(ref_net)
    ref_max_loading = float(ref_net.res_line.loading_percent.max())
    ref_min_v = float(ref_net.res_bus.vm_pu.min())

    net_data = get_39bus()
    pn = PandapowerNetwork(net_data)
    assert pn.run_power_flow()
    max_loading = float(pn.pp_net.res_line.loading_percent.max())
    min_v = float(pn.pp_net.res_bus.vm_pu.min())

    assert abs(max_loading - ref_max_loading) < 2.0, (
        f"max loading {max_loading:.2f}% vs pandapower's own reference "
        f"{ref_max_loading:.2f}% (tolerance: 2 percentage points)"
    )
    assert abs(min_v - ref_min_v) < 0.01, (
        f"min voltage {min_v:.4f}pu vs pandapower's own reference "
        f"{ref_min_v:.4f}pu (tolerance: 0.01pu)"
    )
    print(f"  39bus vs pandapower.networks.case39(): max_loading="
          f"{max_loading:.2f}% (ref {ref_max_loading:.2f}%), "
          f"min_v={min_v:.4f} (ref {ref_min_v:.4f})")


def test_reverse_power_flow_report_shape():
    """First test of exp_reverse_power_flow.py's analyze_system() --
    exercised in-memory only (never write_report(), which would
    overwrite the committed results/reverse_power_flow_analysis.json).
    Asserts against the function's ACTUAL current return shape (read
    directly from source), including the disclosure/instrumentation
    fields added alongside the 39bus/14bus/8bus data-layer fix:
    n_lines_pandapower/n_trafo_pandapower and model_fidelity at the
    top level, plus flow_ratio_vs_baseline/min_flow_ratio/
    closest_to_reversal_line/slack_p_mw/total_pv_mw/total_load_mw/
    pv_to_load_ratio/per_dg_bus_pv_to_local_load per PV level."""
    try:
        from experiments.exp_reverse_power_flow import analyze_system
    except Exception as e:
        pytest.skip(f"exp_reverse_power_flow.py not safely importable: {e}")

    report = analyze_system("8bus")

    assert set(report.keys()) == {
        "system", "n_lines", "n_lines_pandapower", "n_trafo_pandapower",
        "pv_levels", "per_pv", "model_fidelity",
    }
    assert report["system"] == "8bus"
    assert isinstance(report["pv_levels"], list) and len(report["pv_levels"]) > 0
    assert report["model_fidelity"]["loading_percent_calibrated"] is False

    expected_per_pv_keys = {
        "converged", "line_flows_mw", "bus_voltages_pu", "line_loadings_pct",
        "reversed_lines_vs_baseline", "n_reversed", "max_loading_pct",
        "min_voltage_pu", "flow_ratio_vs_baseline", "min_flow_ratio",
        "closest_to_reversal_line", "slack_p_mw", "total_pv_mw",
        "total_load_mw", "pv_to_load_ratio", "per_dg_bus_pv_to_local_load",
    }
    for pv in report["pv_levels"]:
        pv_data = report["per_pv"][pv]
        assert pv_data["converged"] is True, (
            f"8bus power flow did not converge at PV={pv:.0%}"
        )
        assert set(pv_data.keys()) == expected_per_pv_keys
    print(f"  analyze_system('8bus'): all {len(report['pv_levels'])} PV "
          f"levels converged with the expected report shape")
