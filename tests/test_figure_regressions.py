"""
tests/test_figure_regressions.py
─────────────────────────────────
Regression tests for standalone experiment/figure scripts under
experiments/ whose bugs don't fit naturally into the phaseN test files
(each covers one specific, previously-broken function in one script).

Tests:
1. exp_fig14_n1_contingency.py::fine_tune_policy fully restores the base
   trainer's actor/critic/comm_graph/env after fine-tuning.
2. exp_fig50_critical_pairs.py::extract_lagrange_multipliers real (safe-RL
   wrapped env) and zero-fallback (unwrapped env) branches.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from network.network_data import get_8bus
from marl.marl_env import DynamicDOCREnv
from marl.marl_agents import MAPPOMARLTrainer
from marl.safe_rl import create_safe_rl_env, SafeRLConfig


def test_fine_tune_policy_restores_actor_critic_comm_graph_and_env():
    """fine_tune_policy() must leave the passed-in trainer bit-identical
    to how it found it (actor, critic, comm_graph params, and trainer.env)
    once it returns -- main()'s per-line loop starts every line's
    fine-tuning from the same base policy, so any leaked mutation from
    line N would silently contaminate line N+1's "pre-trained, no
    adaptation" baseline. An earlier version only snapshotted/restored
    the actor and env, leaking critic/comm_graph adaptation."""
    from experiments.exp_fig14_n1_contingency import fine_tune_policy

    net = get_8bus()
    env = DynamicDOCREnv(net, use_pandapower=False, fault_type="3ph",
                          curriculum_stage=5, seed=1)
    trainer = MAPPOMARLTrainer(env, use_comm=True, seed=1)

    actor_before = trainer.actor.get_all_params().copy()
    critic_before = trainer.critic.get_params().copy()
    comm_before = trainer.comm_graph.get_all_params().copy()
    env_before = trainer.env

    result = fine_tune_policy(trainer, net, line_idx=0, n_finetune_episodes=2, seed=1)

    assert result is not None and "mean_OF" in result and np.isfinite(result["mean_OF"])

    assert np.array_equal(trainer.actor.get_all_params(), actor_before), \
        "actor params were not restored after fine_tune_policy"
    assert np.array_equal(trainer.critic.get_params(), critic_before), \
        "critic params were not restored after fine_tune_policy"
    assert np.array_equal(trainer.comm_graph.get_all_params(), comm_before), \
        "comm_graph params were not restored after fine_tune_policy"
    assert trainer.env is env_before, \
        "trainer.env was not restored to the original (non-fine-tune) env"

    print(f"  fine_tune_policy restoration OK: mean_OF={result['mean_OF']:.4f}")


def test_extract_lagrange_multipliers_real_and_fallback_branches():
    """extract_lagrange_multipliers() must read the real per-pair
    multipliers off a safe-RL-wrapped env's AugmentedLagrangian, and fall
    back to zeros (not crash) for a plain, unwrapped env. Previously it
    checked `trainer.lagrange_multipliers`/`safety_layer`/
    `constraint_multipliers`, none of which ever existed on
    MAPPOMARLTrainer -- both "real" and "fallback" paths silently
    returned the same all-zeros array."""
    from experiments.exp_fig50_critical_pairs import extract_lagrange_multipliers

    net = get_8bus()

    # Fallback branch: a plain env has no `lagrangian` attribute.
    raw_env = DynamicDOCREnv(net, use_pandapower=False, fault_type="3ph",
                              curriculum_stage=5, seed=1)
    trainer_unsafe = MAPPOMARLTrainer(raw_env, use_comm=False, seed=1)

    fallback = extract_lagrange_multipliers(trainer_unsafe, raw_env)
    assert fallback.shape == (net.n_pairs,)
    assert np.all(fallback == 0.0)

    # Real branch: a safe_rl-wrapped env exposes a real AugmentedLagrangian
    # whose multipliers start at SafeRLConfig.lambda_init (1.0), not zero.
    env2 = DynamicDOCREnv(net, use_pandapower=False, fault_type="3ph",
                           curriculum_stage=5, seed=2)
    safe_env = create_safe_rl_env(env2, SafeRLConfig(enforce_cti=True, lambda_init=1.0))
    trainer_safe = MAPPOMARLTrainer(safe_env, use_comm=False, seed=2)

    real = extract_lagrange_multipliers(trainer_safe, safe_env)
    assert real.shape == (net.n_pairs,)
    assert np.all(real == 1.0), f"expected all multipliers at lambda_init=1.0, got {real}"
    assert np.array_equal(real, safe_env.lagrangian.lambdas), \
        "extract_lagrange_multipliers() did not read the real AugmentedLagrangian.lambdas"

    print(f"  extract_lagrange_multipliers: fallback={fallback[:3]}, real={real[:3]}")


def run_all_tests():
    """Run all figure-script regression tests."""
    print("\n" + "=" * 60)
    print("  Figure Script Regression Tests")
    print("=" * 60)

    tests = [
        test_fine_tune_policy_restores_actor_critic_comm_graph_and_env,
        test_extract_lagrange_multipliers_real_and_fallback_branches,
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
