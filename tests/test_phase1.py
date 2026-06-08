"""
tests/test_phase1.py
--------------------
Unit tests for Phase 1 components.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from network.network_data import get_3bus, get_8bus, get_30bus, get_39bus
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
    result = dist.evaluate_current_settings()
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
