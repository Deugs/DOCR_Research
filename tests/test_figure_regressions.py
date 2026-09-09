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
3. exp_fig43_coordination_diagrams.py::relay_operating_time /
   compute_coordination_curve -- squared-PS (PSM=PS[idx] AND
   I_pickup=PS[idx]) regression, which previously saturated every ratio
   below 1.0 and returned inf for every fault position.
4. exp_fig50_critical_pairs.py::compute_criticality_score -- distance_norm
   must be computed against the network's own max observed distance, not
   a hardcoded /1.0 that saturates at 1.0 for any multi-bus network.
5. baselines/metaheuristics.py::run_eeo -- Eq. 9's exploitation term F
   must scale by a1 (not a2, which is already used elsewhere in the same
   iteration for t_vec per Eq. 8).
6. pv_model/pv_inverter.py::PVInverter.get_fault_current_contribution --
   fault_type must actually scale the returned contribution via
   FAULT_TYPE_WEIGHTS.
7. deployment/robustness_tests.py::RobustnessTester.test_method --
   delegates to test_method_with_stress() (real network perturbation)
   instead of calling method_runner with an unsupported `config` kwarg.
8. mains/main6.py::main() -- --phase7/--robustness/--deployment/
   --phase8/--validate dispatch to their handler functions instead of
   falling through to the default per-method Phase 4 loop.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import inspect
from unittest.mock import patch

import numpy as np

from network.network_data import get_8bus, get_3bus
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


def test_coordination_curve_not_all_inf_and_psm_is_one():
    """relay_operating_time()/compute_coordination_curve(): PS[idx] is
    already this codebase's full per-relay pickup current (see
    network/relay_physics.py's idmt_time: ratio = If/PS, no separate PSM
    multiplier). Passing PS[idx] as BOTH the PSM arg and the I_pickup arg
    squares it, saturating every ratio = I/(PSM*I_pickup) below 1.0 and
    returning inf for every fault position on every line, regardless of
    distance. compute_coordination_curve() must call
    relay_operating_time(TMS, PSM=1.0, I_fault, I_pickup=PS[idx])."""
    from experiments.exp_fig43_coordination_diagrams import (
        relay_operating_time, compute_coordination_curve,
    )

    net = get_8bus()
    n = net.n_relays
    TMS = np.full(n, 0.2)
    PS = np.full(n, 600.0)  # realistic pickup current (net.ps_bounds() ~ 350-1333)
    fault_positions = np.array([0.1, 0.5, 0.9])

    result = compute_coordination_curve(net, TMS, PS, pair_idx=0,
                                         fault_positions=fault_positions,
                                         I_base=5000.0)

    assert np.all(np.isfinite(result["t_primary"])), \
        f"t_primary is not all finite -- squared-PS regression: {result['t_primary']}"
    assert np.all(np.isfinite(result["t_backup"])), \
        f"t_backup is not all finite -- squared-PS regression: {result['t_backup']}"

    # Direct check on the underlying formula: with PSM correctly fixed at
    # 1.0, a realistic PS/I_fault ratio must NOT collapse the exponent
    # below 1.0 (which squaring PS would do for any PS > 1).
    t_correct = relay_operating_time(TDS=0.2, PSM=1.0, I_fault=5000.0, I_pickup=600.0)
    t_squared = relay_operating_time(TDS=0.2, PSM=600.0, I_fault=5000.0, I_pickup=600.0)
    assert np.isfinite(t_correct)
    assert t_squared == float("inf"), \
        "sanity check on the squared-PS bug itself failed to reproduce inf"

    print(f"  ✓ test_coordination_curve_not_all_inf_and_psm_is_one "
          f"(t_primary={result['t_primary']})")


def test_criticality_score_uses_network_max_distance_not_hardcoded_one():
    """compute_criticality_score(): distance_norm must be normalized
    against the network's own max observed electrical_distance
    (max_distance param), not a hardcoded /1.0. compute_electrical_distance()
    returns bus-index differences that routinely exceed 1.0 on any network
    with more than a couple of buses, so normalizing by a fixed 1.0
    saturated distance_norm at 1.0 for nearly every pair, collapsing
    "distance" into a near-constant/binary factor instead of a continuous
    differentiator."""
    from experiments.exp_fig50_critical_pairs import compute_criticality_score

    # A genuinely far pair under a realistic max_distance must NOT
    # saturate distance_norm to 1.0 the way it would under the old,
    # hardcoded /1.0 normalization.
    score_far_realistic_max = compute_criticality_score(
        lambda_val=2.0, violation_rate=0.1, electrical_distance=4.0, max_distance=10.0)
    score_far_unit_max = compute_criticality_score(
        lambda_val=2.0, violation_rate=0.1, electrical_distance=4.0, max_distance=1.0)

    assert score_far_realistic_max != score_far_unit_max, (
        "criticality score is insensitive to max_distance -- distance_norm "
        "regression to a hardcoded /1.0 normalization"
    )

    # A near pair (small electrical_distance) must score as MORE critical
    # (closer = more critical, per the (1 - distance_norm) term) than a
    # far pair under the same realistic max_distance.
    score_near = compute_criticality_score(
        lambda_val=2.0, violation_rate=0.1, electrical_distance=0.5, max_distance=10.0)
    assert score_near > score_far_realistic_max, \
        "closer pair did not score as more critical than a farther pair"

    print(f"  ✓ test_criticality_score_uses_network_max_distance_not_hardcoded_one "
          f"(near={score_near:.4f}, far@max=10={score_far_realistic_max:.4f}, "
          f"far@max=1={score_far_unit_max:.4f})")


def test_eeo_exploitation_term_uses_a1_not_a2():
    """run_eeo()'s Eq. 9 exploitation term
    `F = a1 * sign_vec * (exp(-lambda_vec*t_vec) - 1)` previously reused
    `a2` a second time here (a2 is already used above for `t_vec` per
    Eq. 8) -- leaving a1 computed every iteration but never referenced.

    Constructs a same-seed "a2-substituted" variant directly from the
    live source (so this test tracks the real function, not a copy that
    could drift) and asserts the actual run_eeo() output differs from it
    -- a1(t)=2*exp(-t/max_iter) and a2(t)=2*(1-t/max_iter) coincide only
    at t=0, so any run with max_iter>1 must diverge from the a2-only
    variant if F genuinely uses a1.
    """
    import baselines.metaheuristics as mh

    src = inspect.getsource(mh.run_eeo)
    target = "F = a1 * sign_vec * (np.exp(-lambda_vec * t_vec) - 1.0)"
    assert target in src, "expected F-update line not found -- test is stale, update it"
    buggy_src = src.replace(target, "F = a2 * sign_vec * (np.exp(-lambda_vec * t_vec) - 1.0)")
    assert buggy_src != src

    buggy_ns = {}
    exec(compile(buggy_src, "<run_eeo_a2_variant>", "exec"), vars(mh), buggy_ns)
    run_eeo_a2_variant = buggy_ns["run_eeo"]

    net = get_3bus()
    for seed in (0, 1, 2):
        real = mh.run_eeo(net, n_particles=15, max_iter=80, seed=seed,
                           use_sqp=False, verbose=False)
        a2_variant = run_eeo_a2_variant(net, n_particles=15, max_iter=80, seed=seed,
                                         use_sqp=False, verbose=False)
        assert real["OF"] != a2_variant["OF"], (
            f"seed={seed}: run_eeo() output is identical to the a2-substituted "
            f"variant -- F may have regressed to reusing a2 instead of a1"
        )

    print("  ✓ test_eeo_exploitation_term_uses_a1_not_a2")


def test_pv_inverter_fault_current_scales_by_fault_type():
    """PVInverter.get_fault_current_contribution(): `fault_type` must
    actually scale the returned current via FAULT_TYPE_WEIGHTS (the same
    weighting relay_physics.py's own fault-current calculation uses).
    Previously accepted but never applied, so 3ph/LL/1ph/2ph-g all
    returned the identical contribution."""
    from pv_model.pv_inverter import PVInverter
    from network.relay_physics import FAULT_TYPE_WEIGHTS

    inv = PVInverter(rated_power_mw=0.5)
    v_terminal_pu = 0.3  # deep voltage dip -> nonzero contribution

    contributions = {
        ft: inv.get_fault_current_contribution(v_terminal_pu, fault_type=ft)
        for ft in FAULT_TYPE_WEIGHTS
    }

    assert all(c > 0 for c in contributions.values()), \
        f"expected nonzero contribution for every fault type: {contributions}"

    distinct_weights = len(set(FAULT_TYPE_WEIGHTS.values()))
    distinct_contributions = len(set(round(c, 9) for c in contributions.values()))
    if distinct_weights > 1:
        assert distinct_contributions > 1, (
            f"fault_type has no effect on the returned contribution "
            f"(all equal: {contributions}) despite FAULT_TYPE_WEIGHTS differing"
        )

    # Contribution must be directly proportional to FAULT_TYPE_WEIGHTS.
    base_ft = next(iter(FAULT_TYPE_WEIGHTS))
    base_contribution = contributions[base_ft]
    base_weight = FAULT_TYPE_WEIGHTS[base_ft]
    for ft, weight in FAULT_TYPE_WEIGHTS.items():
        expected = base_contribution * (weight / base_weight)
        assert abs(contributions[ft] - expected) < 1e-9, (
            f"fault_type={ft}: contribution {contributions[ft]} is not "
            f"proportional to FAULT_TYPE_WEIGHTS[{ft}]={weight} "
            f"(expected {expected})"
        )

    print(f"  ✓ test_pv_inverter_fault_current_scales_by_fault_type ({contributions})")


def test_robustness_tester_test_method_delegates_to_stress_and_perturbs_net():
    """RobustnessTester.test_method() previously called
    ``method_runner(net, seed=seed, config=cond_config)`` directly on the
    UNPERTURBED net -- no real baseline runner accepts a `config` kwarg,
    so every call silently raised inside a try/except and results stayed
    empty regardless of condition. It must now delegate to
    test_method_with_stress(), which applies each condition via
    apply_stress_to_net() before calling ``method_runner(stressed_net, seed=seed)``."""
    from deployment.robustness_tests import RobustnessTester

    net = get_3bus()
    tester = RobustnessTester()

    seen_nets = []

    def fake_runner(stressed_net, seed=0):
        seen_nets.append(stressed_net)
        return {"OF": 1.0, "TMS": np.full(net.n_relays, 0.2),
                "PS": np.full(net.n_relays, 1.5), "elapsed": 0.01}

    results = tester.test_method("FakeMethod", fake_runner, net, n_runs=1, verbose=False)

    assert isinstance(results, dict) and len(results) > 0, \
        "test_method() returned empty results -- delegation to test_method_with_stress regressed"
    assert len(seen_nets) > 0, "method_runner was never called"
    # method_runner must be called with a `(net, seed)`-only signature --
    # if it had been called with an unsupported `config` kwarg it would
    # have raised inside test_method()'s own try/except instead. At
    # least one condition (a real stress condition, not "baseline") must
    # have actually perturbed the network's fault currents relative to
    # the original -- confirming apply_stress_to_net() ran for real.
    base_if_pri = [p.If_pri for p in net.pairs]
    any_perturbed = any(
        [p.If_pri for p in n.pairs] != base_if_pri for n in seen_nets
    )
    assert any_perturbed, \
        "no stressed net differed from the original -- conditions were not really applied"

    print(f"  ✓ test_robustness_tester_test_method_delegates_to_stress_and_perturbs_net "
          f"({len(results)} conditions, {len(seen_nets)} calls)")


def test_main6_phase7_robustness_deployment_phase8_dispatch():
    """mains/main6.py::main()'s --phase7/--robustness/--deployment/
    --phase8/--validate dispatch block was previously pasted into the
    module docstring instead of into main() -- a syntactically valid but
    inert string literal, never executed. Those flags silently fell
    through to the default per-method Phase 4 loop instead of dispatching
    to run_phase7_robustness/run_phase7_deployment/run_phase8_validation.
    Verifies each flag now reaches its real handler (handlers themselves
    are mocked out -- this tests dispatch wiring, not the handlers)."""
    import mains.main6 as main6

    cases = [
        (["main6.py", "--robustness"], "run_phase7_robustness"),
        (["main6.py", "--phase7"], "run_phase7_robustness"),
        (["main6.py", "--deployment"], "run_phase7_deployment"),
        (["main6.py", "--phase8"], "run_phase8_validation"),
        (["main6.py", "--validate"], "run_phase8_validation"),
    ]

    for argv, expected_handler in cases:
        with patch.object(sys, "argv", argv), \
             patch.object(main6, "run_phase7_robustness") as m_robust, \
             patch.object(main6, "run_phase7_deployment") as m_deploy, \
             patch.object(main6, "run_phase8_validation") as m_val:
            m_robust.return_value = 0
            m_deploy.return_value = 0
            m_val.return_value = 0
            main6.main()

            handlers = {"run_phase7_robustness": m_robust,
                        "run_phase7_deployment": m_deploy,
                        "run_phase8_validation": m_val}
            for name, mock in handlers.items():
                if name == expected_handler:
                    assert mock.call_count == 1, \
                        f"argv={argv}: expected {expected_handler} to be called once, " \
                        f"got {mock.call_count}"
                else:
                    assert mock.call_count == 0, \
                        f"argv={argv}: unexpected call to {name} " \
                        f"(dead-flag fallthrough regression?)"

    print("  ✓ test_main6_phase7_robustness_deployment_phase8_dispatch")


def test_fig48_shap_locates_feature_axis_by_size_not_fixed_position():
    """
    run_post_processing.py::fig_48_shap() previously assumed
    shap.KernelExplainer's returned array was shaped
    (n_actions, n_explain, n_features) and averaged over
    `range(shap_arr.ndim - 1)` -- i.e. every axis except the LAST.
    Verified directly against this repo's installed shap==0.51.0 (the
    numpy-2.x-compatible version this session upgraded to from the
    broken 0.41.0 pin): KernelExplainer.shap_values() on a multi-output
    function actually returns (n_explain, n_features, n_actions) --
    n_features is the MIDDLE axis, not the last. Averaging over the
    wrong axes silently collapsed the real per-feature signal down to
    n_actions=2 "features", which the pre-existing shape-mismatch
    fallback then blindly np.resize()'d up to the real feature count --
    every SHAP figure generated since the shap upgrade plotted 2
    meaningless bars stretched over 12 feature labels, with only a
    warning log (not a watermark) to notice.

    This constructs a synthetic SHAP array in the REAL (n_explain,
    n_features, n_actions) shape, with each feature's magnitude
    distinctly and monotonically increasing by feature index, and
    verifies fig_48_shap() correctly identifies the top feature (highest
    index) rather than being confused by the wrong axis order.
    """
    import matplotlib
    matplotlib.use("Agg")
    import run_post_processing as pp

    n_explain, n_features, n_actions = 5, 12, 2
    # Each feature i's two action columns are BOTH positive but on very
    # different scales ((i+1)*10 and (i+1)*1) -- per feature, averaging
    # over the (correct) sample+action axes still increases monotonically
    # with i (5.5, 11, 16.5, ..., 66), so the correct computation must
    # rank the LAST feature ("Power-flow direction") first. But averaging
    # over the WRONG axes (samples+features, as the old ndim-1 assumption
    # does) collapses each action column to a single overall scale ([65,
    # 6.5] here) -- tiling that 2-element result up to 12 via
    # np.resize() interleaves those two very different values by
    # even/odd position, completely uncorrelated with the real
    # per-feature ranking, so the buggy computation's top feature is
    # essentially arbitrary and reliably NOT index 11. This is a
    # deliberately strong, deterministic case (no noise) -- a milder
    # per-feature-only signal was tried first and passed even with the
    # bug reintroduced, due to the wrong axis's near-tied values
    # producing a top rank that coincidentally still looked plausible.
    shap_arr = np.zeros((n_explain, n_features, n_actions))
    for i in range(n_features):
        shap_arr[:, i, 0] = (i + 1) * 10.0
        shap_arr[:, i, 1] = (i + 1) * 1.0
    assert shap_arr.shape == (n_explain, n_features, n_actions)

    fake_bridge_cache = {"data": {"shap": shap_arr.tolist(), "base": 0.0}}

    captured = {}

    class _FakeAx:
        def barh(self, positions, values, **kwargs):
            captured["values"] = list(values)
            return None
        def set_yticks(self, *a, **k): pass
        def set_yticklabels(self, labels, **k): captured["labels"] = list(labels)
        def set_xlabel(self, *a, **k): pass
        def set_title(self, *a, **k): pass
        def invert_yaxis(self): pass
        def grid(self, *a, **k): pass

    class _FakeFig:
        def savefig(self, *a, **k): pass

    with patch.object(pp, "load_bridge_cache", lambda fig_num: fake_bridge_cache["data"]), \
         patch.object(pp.plt, "subplots", lambda *a, **k: (_FakeFig(), _FakeAx())), \
         patch.object(pp, "save_fig", lambda *a, **k: None):
        pp.fig_48_shap()

    assert "labels" in captured and "values" in captured
    # Highest-index feature (feature 12, "Power-flow direction") must
    # rank first -- if the wrong axis were averaged over instead, the
    # per-feature ordering would collapse to noise.
    assert captured["labels"][0] == "Power-flow direction", (
        f"expected the highest-magnitude feature (last, index 11) to rank "
        f"first, got ordering starting with {captured['labels'][:3]} -- "
        f"looks like the wrong array axis was averaged over"
    )
    assert captured["values"][0] > captured["values"][-1], \
        "expected descending order by |SHAP value|"

    print("  ✓ test_fig48_shap_locates_feature_axis_by_size_not_fixed_position")


def run_all_tests():
    """Run all figure-script regression tests."""
    print("\n" + "=" * 60)
    print("  Figure Script Regression Tests")
    print("=" * 60)

    tests = [
        test_fine_tune_policy_restores_actor_critic_comm_graph_and_env,
        test_extract_lagrange_multipliers_real_and_fallback_branches,
        test_coordination_curve_not_all_inf_and_psm_is_one,
        test_criticality_score_uses_network_max_distance_not_hardcoded_one,
        test_eeo_exploitation_term_uses_a1_not_a2,
        test_pv_inverter_fault_current_scales_by_fault_type,
        test_robustness_tester_test_method_delegates_to_stress_and_perturbs_net,
        test_main6_phase7_robustness_deployment_phase8_dispatch,
        test_fig48_shap_locates_feature_axis_by_size_not_fixed_position,
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
