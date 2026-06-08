"""
tests/test_phase3.py
--------------------
Unit and integration tests for Phase 3 components.

Tests:
1. Action projection (TMS/PS bounds)
2. Action projection with CTI lookahead
3. Augmented Lagrangian constraint cost
4. Lagrange multiplier updates
5. SafeRLWrapper integration with environment
6. EWC Fisher Information estimation
7. EWC penalty computation
8. EWC regularizer with multiple tasks
9. Task similarity estimation
10. ContinualMAPPOTrainer task detection
11. Safe RL + MAPPO integration
12. Safe RL + EWC combined
13. Backward compatibility with Phase 1/2
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from network.network_data import get_3bus, get_8bus, get_30bus, get_39bus
from network.relay_physics import CTI, TMS_MIN, TMS_MAX


def test_action_projection():
    """Test action projection to maintain TMS/PS bounds."""
    from marl.safe_rl import ActionProjector, SafeRLConfig
    from network.network_data import get_3bus

    net = get_3bus()
    config = SafeRLConfig(project_actions=True)
    projector = ActionProjector(net, config)

    TMS = np.full(net.n_relays, 0.5)
    PS = (net.ps_bounds()[0] + net.ps_bounds()[1]) / 2

    # Test: action that would violate lower bound
    delta_tms = np.full(net.n_relays, -0.9)  # Strong negative
    delta_ps = np.zeros(net.n_relays)

    proj_dt, proj_dp = projector.project(TMS, PS, delta_tms, delta_ps, action_scale=0.05)

    # Verify projected actions don't violate bounds
    new_TMS = TMS + proj_dt * 0.05 * projector.tms_range
    new_PS = PS + proj_dp * 0.05 * projector.ps_range

    assert np.all(new_TMS >= TMS_MIN - 1e-6), f"TMS below min: {new_TMS.min()}"
    assert np.all(new_TMS <= TMS_MAX + 1e-6), f"TMS above max: {new_TMS.max()}"
    assert np.all(new_PS >= net.ps_bounds()[0] - 1e-6), f"PS below min"
    assert np.all(new_PS <= net.ps_bounds()[1] + 1e-6), f"PS above max"

    print(f"  Action Projection: TMS_range=[{new_TMS.min():.3f}, {new_TMS.max():.3f}], PS_bounds_ok")


def test_action_projection_cti_lookahead():
    """Test action projection with CTI lookahead."""
    from marl.safe_rl import ActionProjector, SafeRLConfig
    from network.network_data import get_3bus

    net = get_3bus()
    config = SafeRLConfig(project_actions=True, enforce_cti=True)
    projector = ActionProjector(net, config)

    # Initialize with settings that satisfy CTI
    TMS = np.full(net.n_relays, 0.8)
    PS = net.ps_bounds()[0] + 0.3 * (net.ps_bounds()[1] - net.ps_bounds()[0])

    # Action that might violate CTI
    delta_tms = np.full(net.n_relays, 0.5)
    delta_ps = np.zeros(net.n_relays)

    proj_dt, proj_dp = projector.project_with_cti_lookahead(
        TMS, PS, delta_tms, delta_ps, action_scale=0.05, cti_margin=0.05
    )

    new_TMS = TMS + proj_dt * 0.05 * projector.tms_range
    new_PS = PS + proj_dp * 0.05 * projector.ps_range
    new_TMS = np.clip(new_TMS, TMS_MIN, TMS_MAX)
    new_PS = np.clip(new_PS, net.ps_bounds()[0], net.ps_bounds()[1])

    # Check CTI
    from network.relay_physics import compute_operating_times
    t_pri, t_bkp = compute_operating_times(net, new_TMS, new_PS)
    delta_cti = t_bkp - t_pri

    # Should have fewer violations than without projection
    print(f"  CTI Lookahead: min_delta={delta_cti.min():.3f}, violations={int(np.sum(delta_cti < CTI - 0.05))}")


def test_augmented_lagrangian_cost():
    """Test Augmented Lagrangian constraint cost computation."""
    from marl.safe_rl import AugmentedLagrangian, SafeRLConfig
    from network.network_data import get_3bus

    net = get_3bus()
    config = SafeRLConfig()
    al = AugmentedLagrangian(net.n_pairs, config)

    # Settings with some CTI violations
    TMS = np.full(net.n_relays, 0.3)
    PS = net.ps_bounds()[0] + 0.1 * (net.ps_bounds()[1] - net.ps_bounds()[0])

    cost, g_values, violations = al.compute_constraint_cost(TMS, PS, net)

    assert cost >= 0, "Cost should be non-negative"
    assert g_values.shape == (net.n_pairs,), f"Expected ({net.n_pairs},), got {g_values.shape}"
    assert violations.shape == (net.n_pairs,)

    print(f"  AL Cost: {cost:.4f}, violations={int(np.sum(violations))}, g_mean={g_values.mean():.4f}")


def test_lagrange_multiplier_updates():
    """Test Lagrange multiplier dual ascent updates."""
    from marl.safe_rl import AugmentedLagrangian, SafeRLConfig
    from network.network_data import get_3bus

    net = get_3bus()
    config = SafeRLConfig(lambda_lr=0.1, use_pid=False)
    al = AugmentedLagrangian(net.n_pairs, config)

    # Initial lambdas
    lambda_init = al.lambdas.copy()

    # Simulate violations
    g_values = np.full(net.n_pairs, 0.5)  # All violating
    al.update_lagrangian(g_values)

    # Lambdas should increase
    assert np.all(al.lambdas >= lambda_init), "Lambdas should increase with violations"

    # Now simulate no violations
    g_values = np.full(net.n_pairs, -0.1)  # All satisfied
    al.update_lagrangian(g_values)

    # Lambdas should stay at 0 (projected)
    assert np.all(al.lambdas >= 0), "Lambdas should be non-negative"

    print(f"  Lagrange Update: λ_init={lambda_init.mean():.3f}, λ_after_viol={al.lambdas.mean():.3f}")


def test_pid_lagrangian():
    """Test PID-controlled Lagrange multiplier updates."""
    from marl.safe_rl import AugmentedLagrangian, SafeRLConfig
    from network.network_data import get_3bus

    net = get_3bus()
    config = SafeRLConfig(use_pid=True, Kp=0.01, Ki=0.001, Kd=0.005)
    al = AugmentedLagrangian(net.n_pairs, config)

    # Oscillating violations (test PID stability)
    for i in range(10):
        g = np.full(net.n_pairs, 0.3 * (-1) ** i)
        al.update_lagrangian(g)

    # Should be stable (not diverging)
    assert np.all(np.isfinite(al.lambdas)), "Lambdas should remain finite"
    assert np.all(al.lambdas >= 0), "Lambdas should be non-negative"

    print(f"  PID Lagrangian: λ_mean={al.lambdas.mean():.3f}, λ_max={al.lambdas.max():.3f}")


def test_safe_rl_wrapper():
    """Test SafeRLWrapper integration with environment."""
    from marl.marl_env import DynamicDOCREnv
    from marl.safe_rl import SafeRLWrapper, ActionProjector, AugmentedLagrangian, SafeRLConfig, create_safe_rl_env
    from network.network_data import get_3bus

    net = get_3bus()
    env = DynamicDOCREnv(net, use_pandapower=False, seed=42)
    safe_env = create_safe_rl_env(env)

    obs = safe_env.reset(seed=42)
    assert obs.shape == (net.n_relays, env.OBS_DIM)

    # Step with actions
    actions = np.zeros((net.n_relays, env.ACT_DIM))
    next_obs, rewards, done, info = safe_env.step(actions)

    assert "safe_rl" in info, "Safe RL info should be present"
    assert next_obs.shape == obs.shape
    assert rewards.shape == (net.n_relays,)

    # Update Lagrangian
    safe_env.update_lagrangian()
    stats = safe_env.get_safe_rl_stats()
    assert "lambda_mean" in stats
    assert "rho" in stats

    print(f"  SafeRL Wrapper: rewards_mean={rewards.mean():.4f}, λ_mean={stats['lambda_mean']:.3f}")


def test_adaptive_rho():
    """Test adaptive penalty coefficient."""
    from marl.safe_rl import AugmentedLagrangian, SafeRLConfig

    config = SafeRLConfig(adaptive_rho=True, rho_penalty=10.0, rho_adapt_rate=1.5)
    al = AugmentedLagrangian(5, config)

    initial_rho = al.rho

    # High violation rate → increase rho
    al.update_lagrangian(np.full(5, 0.5), episode_violation_rate=0.5)
    assert al.rho > initial_rho, "Rho should increase with high violations"

    # Low violation rate → decrease rho
    high_rho = al.rho
    al.update_lagrangian(np.full(5, -0.1), episode_violation_rate=0.0)
    assert al.rho < high_rho, "Rho should decrease with low violations"

    print(f"  Adaptive Rho: initial={initial_rho:.1f}, high_viol={high_rho:.1f}, low_viol={al.rho:.1f}")


def test_ewc_fisher_estimation():
    """Test Fisher Information estimation."""
    from marl.continual_trainer import FisherInformationEstimator, EWCConfig

    config = EWCConfig(fisher_samples=100, fisher_damping=1e-4)
    estimator = FisherInformationEstimator(config)

    n_params = 50

    def mock_gradient():
        return np.random.randn(n_params) * 0.1

    fisher = estimator.estimate_fisher_diagonal(mock_gradient, n_params)

    assert fisher.shape == (n_params,), f"Expected ({n_params},), got {fisher.shape}"
    assert np.all(fisher >= 0), "Fisher should be non-negative"
    assert np.all(np.isfinite(fisher)), "Fisher should be finite"

    # Check sparsification
    non_zero = np.sum(fisher > 1e-6)
    expected_non_zero = int(n_params * config.top_k_fraction)
    assert non_zero <= expected_non_zero + 5, "Should sparsify Fisher"

    print(f"  Fisher Estimation: shape={fisher.shape}, non_zero={non_zero}, mean={fisher.mean():.6f}")


def test_ewc_penalty():
    """Test EWC penalty computation."""
    from marl.continual_trainer import EWCRegularizer, EWCConfig

    config = EWCConfig(lambda_ewc=100.0)
    ewc = EWCRegularizer(config)

    n_params = 20
    old_params = np.random.randn(n_params)
    current_params = old_params + np.random.randn(n_params) * 0.1

    # Register task
    ewc.register_task("task1", old_params)

    # Update Fisher (mock)
    fisher = np.random.rand(n_params) * 0.5
    ewc.update_fisher("task1", fisher)

    # Compute penalty
    penalty, gradient = ewc.compute_ewc_penalty(current_params)

    assert penalty >= 0, "Penalty should be non-negative"
    assert gradient.shape == (n_params,)
    assert np.all(np.isfinite(gradient)), "Gradient should be finite"

    print(f"  EWC Penalty: {penalty:.4f}, grad_norm={np.linalg.norm(gradient):.4f}")


def test_ewc_multiple_tasks():
    """Test EWC with multiple tasks."""
    from marl.continual_trainer import EWCRegularizer, EWCConfig

    config = EWCConfig(lambda_ewc=50.0, max_tasks=3)
    ewc = EWCRegularizer(config)

    n_params = 10

    # Register multiple tasks
    for i in range(4):
        params = np.random.randn(n_params)
        ewc.register_task(f"task{i}", params)
        fisher = np.random.rand(n_params) * 0.3
        if ewc.task_memory:
            ewc.update_fisher(ewc.task_memory[-1][0], fisher)

    # Should only keep max_tasks
    assert len(ewc.task_memory) <= config.max_tasks, "Should limit task memory"

    # Compute penalty
    current = np.random.randn(n_params)
    penalty, _ = ewc.compute_ewc_penalty(current)

    assert penalty >= 0, "Penalty should be non-negative"

    stats = ewc.get_task_stats()
    assert stats["n_tasks_consolidated"] <= config.max_tasks

    print(f"  EWC Multi-Task: n_tasks={stats['n_tasks_consolidated']}, penalty={penalty:.4f}")


def test_task_similarity():
    """Test task similarity estimation."""
    from marl.continual_trainer import TaskSimilarityEstimator

    estimator = TaskSimilarityEstimator()

    cond1 = {"pv_level": 0.0, "topology_flag": 0.0, "fault_type": "3ph"}
    params1 = np.random.randn(100)

    cond2 = {"pv_level": 0.2, "topology_flag": 0.0, "fault_type": "3ph"}
    params2 = params1 + np.random.randn(100) * 0.1  # Similar params

    cond3 = {"pv_level": 1.0, "topology_flag": 1.0, "fault_type": "LL"}
    params3 = np.random.randn(100)  # Different params

    sim_12 = estimator.estimate_similarity(cond1, params1, cond2, params2)
    sim_13 = estimator.estimate_similarity(cond1, params1, cond3, params3)

    assert 0 <= sim_12 <= 1, "Similarity should be in [0, 1]"
    assert 0 <= sim_13 <= 1, "Similarity should be in [0, 1]"
    assert sim_12 > sim_13, "Similar tasks should have higher similarity"

    print(f"  Task Similarity: sim_12={sim_12:.3f}, sim_13={sim_13:.3f}")


def test_continual_trainer_task_detection():
    """Test ContinualMAPPOTrainer task detection."""
    from marl.continual_trainer import ContinualMAPPOTrainer, EWCConfig
    from marl.marl_env import DynamicDOCREnv
    from marl.marl_agents import MAPPOMARLTrainer
    from network.network_data import get_3bus

    net = get_3bus()
    env = DynamicDOCREnv(net, use_pandapower=False, seed=42)
    base_trainer = MAPPOMARLTrainer(
        env, use_comm=False, hidden_actor=16, hidden_critic=32,
        rollout_len=16, n_epochs=2, seed=42
    )

    ewc_config = EWCConfig()
    continual = ContinualMAPPOTrainer(base_trainer, ewc_config)

    # Detect first task
    cond1 = {"pv_level": 0.0, "topology_flag": 0.0, "fault_type": "3ph"}
    is_new, name1 = continual.detect_task_change(cond1)
    assert is_new, "First task should be new"

    # Same conditions → not new
    is_new, name2 = continual.detect_task_change(cond1)
    assert not is_new, "Same conditions should not be new task"
    assert name1 == name2, "Same task name"

    # Different PV level → new task
    cond2 = {"pv_level": 0.5, "topology_flag": 0.0, "fault_type": "3ph"}
    is_new, name3 = continual.detect_task_change(cond2)
    assert is_new, "Different PV should be new task"

    print(f"  Task Detection: task1={name1}, task2={name3}, transitions={len(continual.task_transitions)}")


def test_safe_rl_mappo_integration():
    """Test Safe RL integrated with MAPPO training."""
    from marl.marl_env import DynamicDOCREnv
    from marl.marl_agents import MAPPOMARLTrainer
    from marl.safe_rl import create_safe_rl_env, SafeRLConfig
    from network.network_data import get_3bus

    net = get_3bus()
    env = DynamicDOCREnv(net, use_pandapower=False, seed=42)
    safe_env = create_safe_rl_env(env, SafeRLConfig(enforce_cti=True, project_actions=True))

    trainer = MAPPOMARLTrainer(
        safe_env, use_comm=False, hidden_actor=16, hidden_critic=32,
        rollout_len=16, n_epochs=2, seed=42
    )

    # Quick training run
    result = trainer.train(total_steps=64, verbose=False, log_every=1000)

    assert "TMS" in result
    assert "PS" in result
    assert "OF" in result

    # Check safe RL stats
    stats = safe_env.get_safe_rl_stats()
    assert "lambda_mean" in stats
    assert "rho" in stats

    print(f"  Safe+MAPPO: OF={result['OF']:.4f}, λ={stats['lambda_mean']:.3f}, ρ={stats['rho']:.1f}")


def test_ewc_mappo_integration():
    """Test EWC with MAPPO trainer."""
    from marl.marl_env import DynamicDOCREnv
    from marl.marl_agents import MAPPOMARLTrainer
    from marl.continual_trainer import create_ewc_trainer, EWCConfig
    from network.network_data import get_3bus

    net = get_3bus()
    env = DynamicDOCREnv(net, use_pandapower=False, seed=42)
    base_trainer = MAPPOMARLTrainer(
        env, use_comm=False, hidden_actor=16, hidden_critic=32,
        rollout_len=16, n_epochs=2, seed=42
    )

    ewc_config = EWCConfig(lambda_ewc=10.0, fisher_samples=50)
    continual = create_ewc_trainer(base_trainer, ewc_config)

    # Simulate task transitions
    cond1 = {"pv_level": 0.0, "topology_flag": 0.0, "fault_type": "3ph"}
    is_new, name1 = continual.detect_task_change(cond1)

    # Mock parameters
    mock_params = np.random.randn(100)
    continual.on_task_transition("none", name1, mock_params)

    # Compute EWC penalty
    current = mock_params + np.random.randn(100) * 0.05
    penalty, grad = continual.compute_ewc_penalty(current)

    assert penalty >= 0, "Penalty should be non-negative"
    assert grad.shape == (100,)

    stats = continual.get_adaptation_stats()
    assert stats["n_tasks_learned"] >= 1

    print(f"  EWC+MAPPO: penalty={penalty:.4f}, n_tasks={stats['n_tasks_learned']}")


def test_backward_compatibility_phase3():
    """Test Phase 3 is backward compatible with Phases 1 and 2."""
    from marl.marl_env import DynamicDOCREnv
    from marl.marl_agents import MAPPOMARLTrainer
    from marl.safe_rl import create_safe_rl_env
    from marl.continual_trainer import create_ewc_trainer, EWCConfig
    from network.network_data import get_3bus

    net = get_3bus()
    env = DynamicDOCREnv(net, use_pandapower=False, seed=42)

    # Phase 1 compatibility: basic env works
    obs = env.reset(seed=42)
    assert obs.shape == (net.n_relays, env.OBS_DIM)
    assert env.OBS_DIM == 10

    # Phase 2 compatibility: MAPPO works
    trainer = MAPPOMARLTrainer(
        env, use_comm=False, hidden_actor=16, hidden_critic=32,
        rollout_len=16, n_epochs=2, seed=42
    )
    result = trainer.train(total_steps=64, verbose=False)
    assert "OF" in result

    # Phase 3 compatibility: Safe RL wrapper
    safe_env = create_safe_rl_env(env)
    obs = safe_env.reset(seed=42)
    actions = np.zeros((net.n_relays, env.ACT_DIM))
    next_obs, rewards, done, info = safe_env.step(actions)
    assert "safe_rl" in info

    # Phase 3 compatibility: EWC trainer
    ewc_trainer = create_ewc_trainer(trainer, EWCConfig())
    cond = {"pv_level": 0.0, "topology_flag": 0.0, "fault_type": "3ph"}
    is_new, name = ewc_trainer.detect_task_change(cond)
    assert is_new

    print(f"  Backward Compatible: Phase1/2/3 all functional, OF={result['OF']:.4f}")


def run_all_tests():
    """Run all Phase 3 unit tests."""
    print("\n" + "=" * 60)
    print("  Phase 3 Unit Tests: Safe RL + EWC Continual Adaptation")
    print("=" * 60)

    tests = [
        test_action_projection,
        test_action_projection_cti_lookahead,
        test_augmented_lagrangian_cost,
        test_lagrange_multiplier_updates,
        test_pid_lagrangian,
        test_safe_rl_wrapper,
        test_adaptive_rho,
        test_ewc_fisher_estimation,
        test_ewc_penalty,
        test_ewc_multiple_tasks,
        test_task_similarity,
        test_continual_trainer_task_detection,
        test_safe_rl_mappo_integration,
        test_ewc_mappo_integration,
        test_backward_compatibility_phase3,
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
            import traceback
            traceback.print_exc()

    print(f"\n  Results: {passed} passed, {failed} failed")
    print("=" * 60)
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
