"""
tests/test_phase1.py
--------------------
Unit tests for Phase 1 components.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from network.network_data import (
    get_3bus, get_8bus, get_30bus, get_39bus, get_14bus, get_57bus, get_118bus,
    NetworkData, RelayPair, _ensure_feasible_ps_bounds, NETWORKS,
)
from network.relay_physics import (
    idmt_time, compute_operating_times, objective_function, 
    check_constraints, CTI, TMS_MIN, TMS_MAX
)
from pv_model.pv_inverter import PVInverter, PVPlant
from pv_model.scenario_generator import MarkovCloudModel, ScenarioGenerator
from environment.topology_manager import TopologyManager
from environment.relay_sensor import RelaySensor, CTParameters
from environment.dynamic_network import DistributionNetwork, create_network


def test_network_data():
    """Test IEEE test system creation."""
    for get_net in [get_3bus, get_8bus, get_30bus, get_39bus]:
        net = get_net()
        assert net.n_relays > 0
        assert net.n_pairs > 0
        assert len(net.dg_buses) >= 0
        ps_min, ps_max = net.ps_bounds()
        assert np.all(ps_min < ps_max)
        print(f"  {net.name}: {net.n_relays} relays, {net.n_pairs} pairs, {len(net.dg_buses)} DG buses")


def test_get_14bus_full_topology():
    """Round-5 regression: get_14bus() force-overrode n_relays to 20 after
    already computing it correctly as 2*n_lines=40, so every
    "if r_from/r_to < n_relays" guard in the relay-array construction
    silently dropped lines 10-19 (half the 20-line topology) from
    Iload_max, If_min, relay_bus, and every P/B pair — those relays kept
    their zero-initialized defaults instead of real per-line values."""
    net = get_14bus()
    assert net.n_relays == 40, f"expected 40 relays (2 per line x 20 lines), got {net.n_relays}"
    assert net.n_pairs > 0
    assert len(net.relay_bus) == net.n_relays
    # No relay should be left at its zero-initialized default — every
    # relay must have a real (non-zero) load rating and fault-current
    # estimate from its own line.
    assert np.all(net.Iload_max > 0), "some relays have zero Iload_max (dropped by the n_relays bug)"
    assert np.all(net.If_min > 0), "some relays have zero If_min (dropped by the n_relays bug)"
    print(f"  14bus: {net.n_relays} relays, {net.n_pairs} pairs, all relays populated")


def test_relay_bus_mapping():
    """get_relay_bus() must return the authoritative relay_bus array
    (captured from each line's from/to bus at construction time) for
    get_14/39/57/118bus, covering every relay including ones absent
    from every RelayPair -- these previously fell through to a
    placeholder min(relay_idx, n_buses-1) mapping that silently
    mis-located loads/sensor readings for 8.5-30% of relays depending
    on the system (see get_relay_bus()'s docstring)."""
    for get_net in [get_14bus, get_39bus, get_57bus, get_118bus]:
        net = get_net()
        assert net.relay_bus is not None, f"{net.name}: relay_bus array should be populated"
        assert len(net.relay_bus) == net.n_relays

        for relay_idx in range(net.n_relays):
            bus = net.get_relay_bus(relay_idx)
            assert 0 <= bus < net.n_buses, \
                f"{net.name}: relay {relay_idx} mapped to out-of-range bus {bus}"
            # get_relay_bus() must actually read the authoritative array,
            # not silently fall through to a different code path.
            assert bus == int(net.relay_bus[relay_idx])

        print(f"  {net.name}: all {net.n_relays} relays have a valid relay_bus mapping")


def test_relay_bus_survives_apply_pv_penetration_and_copy():
    """
    Regression test: apply_pv_penetration() and copy() used to construct a
    new NetworkData without passing relay_bus=self.relay_bus, silently
    reverting it to None. Every real PV-penetration experiment calls
    apply_pv_penetration() before building the environment, and
    apply_n1_outage() calls copy() -- both would then reintroduce the
    "wrong bus" bug get_relay_bus() exists to fix (falling back to the old
    min(relay_idx, n_buses-1) placeholder) for any relay absent from every
    RelayPair, on exactly the systems (14/39/57/118-bus) and conditions
    (PV penetration > 0, post-N-1) this thesis actually studies.
    """
    for get_net in [get_14bus, get_39bus, get_57bus, get_118bus]:
        net = get_net()

        scaled = net.apply_pv_penetration(0.5)
        assert scaled.relay_bus is not None, f"{net.name}: apply_pv_penetration() dropped relay_bus"
        copied = net.copy()
        assert copied.relay_bus is not None, f"{net.name}: copy() dropped relay_bus"

        for relay_idx in range(net.n_relays):
            expected = net.get_relay_bus(relay_idx)
            assert scaled.get_relay_bus(relay_idx) == expected, \
                f"{net.name}: relay {relay_idx} bus changed after apply_pv_penetration()"
            assert copied.get_relay_bus(relay_idx) == expected, \
                f"{net.name}: relay {relay_idx} bus changed after copy()"

    print("  relay_bus survives apply_pv_penetration()/copy() on all line-based systems")


def test_relay_bus_mapping_fallback_paths():
    """get_relay_bus()'s two fallback paths (no relay_bus array; relay
    absent from every pair) on a minimal hand-built NetworkData, since
    get_14/39/57/118bus always populate relay_bus and so never exercise
    them on their own."""
    pairs = [RelayPair(primary=0, backup=1, If_pri=1000.0, If_bkp=800.0,
                        bus_pri=2, bus_bkp=5)]  # 1-indexed bus_pri/bus_bkp
    net = NetworkData(
        name="fallback_test", n_relays=3, pairs=pairs,
        Iload_max=np.array([10.0, 10.0, 10.0]),
        If_min=np.array([1000.0, 1000.0, 1000.0]),
        n_buses=10, relay_bus=None,
    )
    # relay 0 is the primary of the only pair -> bus_pri - 1
    assert net.get_relay_bus(0) == 2 - 1
    # relay 1 is the backup -> bus_bkp - 1
    assert net.get_relay_bus(1) == 5 - 1
    # relay 2 appears in no pair at all -> old placeholder fallback
    assert net.get_relay_bus(2) == min(2, net.n_buses - 1)
    print("  get_relay_bus() fallback paths (no relay_bus array) behave as documented")


def test_ensure_feasible_ps_bounds():
    """PS bounds (ps_min <= ps_max) must hold for every relay on the
    systems _ensure_feasible_ps_bounds was introduced to fix -- without
    it, a per-line fault-current estimate below the required overload
    margin makes ps_min > ps_max, which np.clip then silently resolves
    by always returning ps_max regardless of input (see
    _ensure_feasible_ps_bounds()'s docstring)."""
    for get_net in [get_14bus, get_57bus, get_118bus]:
        net = get_net()
        ps_min, ps_max = net.ps_bounds()
        assert np.all(ps_min <= ps_max), \
            f"{net.name}: ps_min > ps_max for at least one relay despite the feasibility floor"
        print(f"  {net.name}: ps_min <= ps_max holds for all {net.n_relays} relays")

    # Directly exercise the floor: an If_min deliberately below the
    # required margin for one relay must get raised, and none of the
    # already-feasible relays should be touched.
    iload_max = np.array([100.0, 100.0])
    olf, margin = 1.25, 1.1
    required = margin * 1.5 * olf * iload_max
    if_min = np.array([required[0] * 0.5, required[1] * 2.0])  # relay 0 infeasible, relay 1 fine
    fixed = _ensure_feasible_ps_bounds(if_min, iload_max, "unit_test_system", olf=olf, margin=margin)
    assert fixed[0] >= required[0] - 1e-9
    assert fixed[1] == if_min[1]  # untouched: was already feasible
    print("  _ensure_feasible_ps_bounds() floors only the infeasible relay")


def test_relay_graph_matches_powerflow_graph():
    """Regression guard for the 39bus/14bus relay-topology gap: for any
    network whose line_data carries a real branch table AND whose relay
    graph was generated from that same table -- the discriminator
    NetworkData.apply_n1_outage() already uses internally to decide
    whether its real-branch outage path applies: n_relays ==
    2 * len(line_data["lines"]) -- the relay/pair graph must actually
    describe the same topology PandapowerNetwork solves power flow on:
    one relay at each end of every branch, covering exactly the
    branches' own endpoints.

    8bus also carries line_data["lines"] (the real 7-line + 2-transformer
    published system, for pandapower's power-flow use only) but
    DELIBERATELY keeps a separate, smaller 14-relay/20-pair coordination
    graph (see get_8bus()'s own comment on the two coexisting
    structures) -- its n_relays (14) != 2*len(line_data["lines"]) (18),
    so it fails this same discriminator and must NOT be held to the
    invariant below.
    """
    checked = []
    excluded = []
    for name, get_net in NETWORKS.items():
        net = get_net()
        if not (net.line_data and net.line_data.get("lines")):
            continue
        branches = net.line_data["lines"]
        if net.n_relays == 2 * len(branches):
            checked.append(name)
            expected_endpoints = []
            for ln in branches:
                expected_endpoints.append(int(ln["from_bus"]) - 1)
                expected_endpoints.append(int(ln["to_bus"]) - 1)
            actual = sorted(int(b) for b in net.relay_bus)
            assert actual == sorted(expected_endpoints), \
                f"{name}: relay_bus multiset doesn't match branch endpoints"
        else:
            excluded.append(name)

    assert "14bus" in checked, "14bus should satisfy n_relays == 2*n_branches"
    assert "39bus" in checked, "39bus should satisfy n_relays == 2*n_branches"
    assert "8bus" in excluded, (
        "8bus's relay/pair graph is deliberately smaller than its "
        "line_data branch table (separate coordination vs power-flow "
        "topologies, see get_8bus()) -- it should NOT satisfy this "
        "invariant"
    )
    print(f"  relay graph matches line_data branch endpoints for {checked}; "
          f"8bus correctly excluded (deliberately separate graphs): {excluded}")


def test_39bus_full_bus_coverage():
    """get_39bus()'s relay graph, rebuilt from the full 46-branch
    topology, must cover all 39 buses -- the abbreviated 23-line graph
    it replaces (now get_39bus_reduced()) left 17 of 39 buses, including
    all 10 real generator buses, with zero relay coverage, so a
    "N-1 contingency on 39-bus" sweep could never outage a single
    generator-connected branch."""
    net = get_39bus()
    assert set(int(b) for b in net.relay_bus) == set(range(39))
    print("  get_39bus(): all 39 buses have relay coverage")


def test_generator_buses_have_relays():
    """Every DG bus, and every explicit line_data slack bus, must have at
    least one relay physically located there -- a generator (or slack)
    bus with zero relay coverage can never be N-1-outaged or sensed,
    defeating the point of adding it to the topology in the first
    place. Generic across every network that has both dg_buses and a
    real relay-location mapping."""
    for get_net in [get_8bus, get_14bus, get_39bus]:
        net = get_net()
        if net.relay_bus is not None:
            relay_buses = set(int(b) for b in net.relay_bus)
        else:
            relay_buses = set(net.get_relay_bus(i) for i in range(net.n_relays))

        for dg in net.dg_buses:
            dg_bus0 = int(dg.bus_idx) - 1
            assert dg_bus0 in relay_buses, \
                f"{net.name}: DG bus {dg.bus_idx} (0-indexed {dg_bus0}) has no relay"

        if net.line_data and "slack_bus" in net.line_data:
            slack0 = int(net.line_data["slack_bus"]) - 1
            assert slack0 in relay_buses, \
                f"{net.name}: slack bus {net.line_data['slack_bus']} has no relay"

        print(f"  {net.name}: all {len(net.dg_buses)} DG buses "
              f"(and slack bus, where set) have relay coverage")


def test_39bus_ps_bounds_and_n1_after_expansion():
    """After the 39bus relay-graph expansion (46->92 relays), PS bounds
    must remain feasible for every relay, and apply_n1_outage() must
    succeed (real single-branch contingency, via
    NetworkData.apply_n1_outage()'s real-branch path) for every one of
    the network's 46 real branches, returning a sensible outage
    factor."""
    net = get_39bus()
    ps_min, ps_max = net.ps_bounds()
    assert len(ps_min) == 92
    assert np.all(ps_min <= ps_max), \
        "ps_min > ps_max for at least one of the 92 relays"

    n_lines = net.n_lines
    assert n_lines == 46
    for k in range(n_lines):
        new_net, outage_factor = net.apply_n1_outage(k)
        assert 0.0 < outage_factor <= 1.0, \
            f"line {k}: outage_factor {outage_factor} is not a sensible (0, 1] value"
        assert new_net.n_relays == net.n_relays
    print(f"  get_39bus(): ps_bounds feasible for all 92 relays; "
          f"apply_n1_outage() succeeded for all {n_lines} branches")


def test_reduced_variant_preserved():
    """get_39bus_reduced() must reproduce the pre-expansion 39bus relay
    graph exactly (46 relays, 42 pairs, 22-of-39 bus coverage) so the
    already-published reduced-graph results stay reproducible and
    citable as a named provenance variant."""
    net = NETWORKS["39bus_reduced"]()
    assert net.n_relays == 46
    assert net.n_pairs == 42
    buses_covered = len(set(int(b) for b in net.relay_bus))
    assert buses_covered == 22
    print(f"  39bus_reduced: {net.n_relays} relays, {net.n_pairs} pairs, "
          f"{buses_covered}/39 buses covered")


def test_topology_fingerprint_changes_with_topology():
    """topology_fingerprint() must be a stable, deterministic function of
    a network's structural identity -- CacheManager relies on it to
    avoid returning a policy trained on a DIFFERENT topology as a valid
    cache hit after a topology change."""
    net_full = get_39bus()
    net_full_again = get_39bus()
    net_reduced = NETWORKS["39bus_reduced"]()

    fp1 = net_full.topology_fingerprint()
    fp2 = net_full_again.topology_fingerprint()
    fp_reduced = net_reduced.topology_fingerprint()

    assert fp1 == fp2, "topology_fingerprint() is not stable across repeated calls"
    assert fp1 != fp_reduced, (
        "39bus (92 relays/86 pairs) and 39bus_reduced (46/42) have "
        "genuinely different topologies but produced the same fingerprint"
    )

    fp8_a = get_8bus().topology_fingerprint()
    fp8_b = get_8bus().topology_fingerprint()
    assert fp8_a == fp8_b, "8bus topology_fingerprint() is not deterministic"
    print(f"  topology_fingerprint(): 39bus={fp1[:8]}... != "
          f"39bus_reduced={fp_reduced[:8]}..., stable across repeated calls")


def test_14bus_dg_placement_excludes_condensers_and_slack():
    """get_14bus()'s DG placement must never include the 3 zero-real-
    power synchronous-condenser buses (3/6/8, deliberately dropped by
    the `if p > 0` filter -- see get_14bus()'s comment) or the resolved
    slack bus (bus 1: 14bus's line_data carries no explicit "slack_bus"
    key, so PandapowerNetwork defaults the ext_grid to bus 0 / 1-indexed
    bus 1). A DG entry on either would be either inert (a p_mw=0
    reactive-only condenser under this codebase's P-scaled DG model) or
    invisibly absorbed by the ideal slack source -- the exact bug
    get_14bus()'s PV was relocated to fix. NOTE: this replaces the plan
    document's originally-specified `test_14bus_dg_filter_is_deliberate`
    (which pinned `len(dg_buses) == 2`); the actual current PV placement
    relocated PV onto 9 load buses, not 2, so that assertion no longer
    reflects reality."""
    net = get_14bus()
    condenser_buses_1indexed = {3, 6, 8}
    slack_bus_1indexed = 1  # line_data has no "slack_bus" key -> default bus 1

    assert len(net.dg_buses) > 0
    for dg in net.dg_buses:
        assert dg.bus_idx not in condenser_buses_1indexed, (
            f"DG at condenser bus {dg.bus_idx} -- these are reactive-only "
            f"and inert under the current P-scaled DG model"
        )
        assert dg.bus_idx != slack_bus_1indexed, (
            f"DG at the resolved slack bus {dg.bus_idx} would be "
            f"invisibly absorbed by the ideal ext_grid source"
        )
    print(f"  get_14bus(): {len(net.dg_buses)} DG buses, none on condenser "
          f"buses {condenser_buses_1indexed} or slack bus {slack_bus_1indexed}")


def test_idmt_relay():
    """Test IDMT relay operating time calculation."""
    t = idmt_time(If=2000, PS=500, TMS=0.5)
    assert t > 0
    assert t < 10.0

    # No trip if If <= PS
    t_no_trip = idmt_time(If=400, PS=500, TMS=0.5)
    assert t_no_trip >= 10.0 - 1e-6
    print(f"  IDMT: t={t:.4f}s (If=2000, PS=500, TMS=0.5)")


def test_idmt_relay_enforces_floor_and_ceiling():
    """Round-5 regression: idmt_time()'s scalar path previously enforced
    no TOP_MIN/TOP_MAX clamp at all (only its unused vectorized twin
    idmt_time_vec() did), so legal boundary TMS/PS values could compute a
    physically-unrealistic sub-floor operating time — e.g. a large
    fault-current ratio at TMS_MIN drove t well below TOP_MIN=0.05s."""
    from network.relay_physics import TOP_MIN, TOP_MAX, TMS_MIN, TMS_MAX

    # Large ratio + minimum TMS: unclamped formula gives ~0.007s, far
    # below the standard's floor.
    t_low = idmt_time(If=100000, PS=10, TMS=TMS_MIN)
    assert t_low == TOP_MIN, f"expected floor {TOP_MIN}, got {t_low}"

    # Ratio just above 1.0 + maximum TMS: unclamped formula gives
    # thousands of seconds, far above the standard's ceiling.
    t_high = idmt_time(If=501, PS=500, TMS=TMS_MAX)
    assert t_high == TOP_MAX, f"expected ceiling {TOP_MAX}, got {t_high}"
    print(f"  IDMT floor/ceiling: t_low={t_low}, t_high={t_high}")


def test_objective_function():
    """Test objective function and constraints."""
    net = get_3bus()
    n = net.n_relays
    TMS = np.full(n, 0.5)
    PS = (net.ps_bounds()[0] + net.ps_bounds()[1]) / 2
    
    OF, sum_top, penalty = objective_function(net, TMS, PS)
    assert OF > 0
    
    constraints = check_constraints(net, TMS, PS)
    assert "cti_satisfied" in constraints
    assert "cti_violations" in constraints
    print(f"  OF={OF:.4f}, CTI_ok={constraints['cti_satisfied']}, violations={constraints['cti_violations']}")


def test_pv_inverter():
    """Test PV inverter fault ride-through."""
    inv = PVInverter(rated_power_mw=1.0, rated_voltage_kv=0.4)
    
    # Normal operation
    p, q, i = inv.calculate_output(irradiance=800, temperature=25, v_terminal_pu=1.0)
    assert p > 0
    assert i > 0
    
    # Fault condition (voltage dip)
    p_f, q_f, i_f = inv.calculate_output(
        irradiance=800, temperature=25, v_terminal_pu=0.5, 
        fault_triggered=True, time=0.0
    )
    assert i_f > i  # Higher current during fault (ride-through)
    print(f"  PV: Normal P={p:.3f}MW, Fault P={p_f:.3f}MW, I_f={i_f:.3f}kA")


def test_scenario_generator():
    """Test scenario generation."""
    gen = ScenarioGenerator(seed=42)
    scenarios = gen.generate_scenarios(n_train=10, n_val=2, n_test=2)
    assert len(scenarios["train"]) == 10
    assert len(scenarios["val"]) == 2

    # Check PV levels
    pv_levels = [s["pv_level"] for s in scenarios["train"]]
    assert all(0 <= p <= 1.0 for p in pv_levels)
    print(f"  Generated {len(scenarios['train'])} train, {len(scenarios['val'])} val scenarios")


def test_pv_inverter_trip_latches():
    """Round-5 regression: calculate_output() cleared fault_active on trip
    but never latched a "stayed tripped" state. The very next call (with
    fault_triggered still True) re-entered `if fault_triggered and not
    self.fault_active:`, restarting the ride-through clock and resuming
    real fault-current injection instead of remaining latched at zero
    output for the rest of the fault."""
    from pv_model.pv_inverter import PVRideThroughCurve

    inv = PVInverter(
        rated_power_mw=1.0, rated_voltage_kv=0.4,
        frt_curve=PVRideThroughCurve(v_low=0.3, v_high=1.1, t_ride=0.1),
    )
    # Fault starts: within ride-through window, still injecting current.
    _, _, i0 = inv.calculate_output(irradiance=800, temperature=25,
                                     v_terminal_pu=0.1, fault_triggered=True, time=0.0)
    assert i0 > 0
    assert not inv.tripped

    # Ride-through exceeded with voltage still below v_low: trips.
    p1, q1, i1 = inv.calculate_output(irradiance=800, temperature=25,
                                       v_terminal_pu=0.1, fault_triggered=True, time=0.2)
    assert (p1, q1, i1) == (0.0, 0.0, 0.0)
    assert inv.tripped

    # Fault condition still present on the NEXT call: must stay latched
    # at zero output, not re-inject fault current.
    p2, q2, i2 = inv.calculate_output(irradiance=800, temperature=25,
                                       v_terminal_pu=0.1, fault_triggered=True, time=0.3)
    assert (p2, q2, i2) == (0.0, 0.0, 0.0), "inverter re-armed instead of staying latched tripped"
    assert inv.tripped

    # Fault genuinely clears: latch releases into post-fault recovery.
    inv.calculate_output(irradiance=800, temperature=25,
                          v_terminal_pu=1.0, fault_triggered=False, time=0.5)
    assert not inv.tripped
    print("  PV trip latch: stays tripped while fault persists, clears once fault_triggered=False")


def test_topology_manager():
    """Test topology management."""
    top = TopologyManager(n_buses=8, n_lines=7)
    
    # Base state
    assert top.get_outage_factor() == 1.0
    
    # N-1 outage
    state = top.apply_n1_outage("line", 0)
    factor = top.get_outage_factor(state)
    assert factor < 1.0
    assert factor >= 0.5
    print(f"  Topology: outage_factor={factor:.3f}")


def test_relay_sensor():
    """Test sensor simulation."""
    sensor = RelaySensor(seed=42)
    
    # Normal measurement
    measured = sensor.measure_current(true_current_a=1000.0)
    assert measured > 0
    assert abs(measured - 1000.0) < 100  # Within 10% error
    
    # Fault measurement with saturation
    measured_sat = sensor.measure_current(true_current_a=50000.0, fault_condition=True)
    assert measured_sat > 0
    print(f"  Sensor: Normal={measured:.1f}A, Saturated={measured_sat:.1f}A")


def test_distribution_network():
    """Test unified DistributionNetwork."""
    net = get_8bus()
    dist = DistributionNetwork(net, use_pandapower=False)
    
    # Reset with PV
    state = dist.reset(pv_level=0.5, fault_type="3ph")
    assert state.pv_penetration == 0.5
    assert state.fault_type == "3ph"
    
    # Evaluate settings
    TMS = np.full(net.n_relays, 0.5)
    PS = (net.ps_bounds()[0] + net.ps_bounds()[1]) / 2
    dist.set_relay_settings(TMS, PS)
    result = dist.evaluate_relay_settings(TMS, PS)
    assert "OF" in result
    assert "constraints" in result
    print(f"  DistributionNetwork: OF={result['OF']:.4f}, CTI_ok={result['constraints']['cti_satisfied']}")


def test_dynamic_doc_env():
    """Test DynamicDOCREnv."""
    from marl.marl_env import DynamicDOCREnv
    
    net = get_3bus()
    env = DynamicDOCREnv(net, use_pandapower=False, curriculum_stage=0)
    
    obs = env.reset(seed=42)
    assert obs.shape == (net.n_relays, env.OBS_DIM)
    assert env.OBS_DIM == 12  # Objective 2: 10 -> 12 (added real V, P_dir)
    
    actions = np.zeros((net.n_relays, env.ACT_DIM))
    next_obs, rewards, done, info = env.step(actions)
    assert next_obs.shape == obs.shape
    assert rewards.shape == (net.n_relays,)
    assert "pv_level" in info
    print(f"  DynamicDOCREnv: obs_shape={obs.shape}, reward_mean={rewards.mean():.4f}")


def run_all_tests():
    """Run all Phase 1 unit tests."""
    print("\n" + "="*55)
    print("  Phase 1 Unit Tests")
    print("="*55)
    
    tests = [
        test_network_data,
        test_get_14bus_full_topology,
        test_relay_bus_mapping,
        test_relay_bus_survives_apply_pv_penetration_and_copy,
        test_relay_bus_mapping_fallback_paths,
        test_ensure_feasible_ps_bounds,
        test_relay_graph_matches_powerflow_graph,
        test_39bus_full_bus_coverage,
        test_generator_buses_have_relays,
        test_39bus_ps_bounds_and_n1_after_expansion,
        test_reduced_variant_preserved,
        test_topology_fingerprint_changes_with_topology,
        test_14bus_dg_placement_excludes_condensers_and_slack,
        test_idmt_relay,
        test_idmt_relay_enforces_floor_and_ceiling,
        test_objective_function,
        test_pv_inverter,
        test_pv_inverter_trip_latches,
        test_scenario_generator,
        test_topology_manager,
        test_relay_sensor,
        test_distribution_network,
        test_dynamic_doc_env,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
            print(f"  [PASS] {test.__name__}")
        except Exception as e:
            failed += 1
            print(f"  [FAIL] {test.__name__}: {e}")
    
    print(f"\n  Results: {passed} passed, {failed} failed")
    print("="*55)
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
