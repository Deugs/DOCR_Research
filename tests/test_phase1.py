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
    NetworkData, RelayPair, _ensure_feasible_ps_bounds,
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


def test_idmt_relay():
    """Test IDMT relay operating time calculation."""
    t = idmt_time(If=2000, PS=500, TMS=0.5)
    assert t > 0
    assert t < 10.0
    
    # No trip if If <= PS
    t_no_trip = idmt_time(If=400, PS=500, TMS=0.5)
    assert t_no_trip >= 10.0 - 1e-6
    print(f"  IDMT: t={t:.4f}s (If=2000, PS=500, TMS=0.5)")


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
    assert env.OBS_DIM == 10  # Phase 1: 10 features
    
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
        test_relay_bus_mapping,
        test_relay_bus_mapping_fallback_paths,
        test_ensure_feasible_ps_bounds,
        test_idmt_relay,
        test_objective_function,
        test_pv_inverter,
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
