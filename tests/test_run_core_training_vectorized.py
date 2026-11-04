"""
tests/test_run_core_training_vectorized.py
-------------------------------------------
Phase V14.1 of the GPU-vectorized-environment redesign: run_core_training_
vectorized.py's Core A replication. Verifies the produced cache entry is
schema-compatible with the real Core A cache (run_core_training.py's
_aggregate_runs() output) -- same key set, same shapes, loadable through
run_post_processing.py's real load_cache() -- at a tiny, fast budget (not a
claim of converged/real-scale results, same "crash/correctness gate, not a
performance claim" bar every phase in this redesign uses before real-scale
validation).
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from network.network_data import load_network
from run_core_training import CacheManager, CACHE_DIR
from run_core_training_vectorized import (
    run_core_A_vectorized, _run_core_A_vectorized_single,
    run_core_B_vectorized, _run_curriculum_single_vectorized,
    run_core_C_vectorized, _run_baseline_single_vectorized,
    _evaluate_traditional_vectorized, _CORE_C_METHODS,
    run_core_D_vectorized, _run_ablation_single_vectorized, _CORE_D_CONFIGS,
    run_core_E_vectorized, _run_hparam_single_vectorized,
    run_core_F_vectorized, _run_scalability_single_vectorized,
    _train_base_policy_vectorized, _get_eval_pipeline,
    run_core_G_vectorized, _eval_comm_stress_vectorized, _eval_n1_contingency_vectorized,
    _eval_byzantine_vectorized, _eval_bandwidth_vectorized, _eval_sensor_robustness_vectorized,
    _eval_adversarial_vectorized,
    run_core_H_vectorized, _eval_unseen_faults_vectorized, _eval_zero_shot_transfer_vectorized,
    run_core_I_vectorized, _measure_latency_vectorized, _eval_compression_vectorized,
    run_core_J_vectorized, _eval_daily_pv_vectorized, _eval_sympathetic_tripping_vectorized,
    run_core_K_vectorized, _eval_seasonal_vectorized,
)


def test_run_core_A_vectorized_single_schema():
    """One run's raw result must carry every field UnifiedTrainingLogger.
    to_dict() produces, with the shapes _aggregate_runs() expects to stack."""
    result = _run_core_A_vectorized_single(n_episodes=2, seed=777)

    for key in ["episodes", "global_returns", "per_agent_returns",
                "cti_violation_counts", "cti_violation_rates", "lagrange_norms",
                "safety_activations", "action_projections", "cumulative_violations",
                "attention_snapshots", "attention_entropies", "n_agents", "n_pairs"]:
        assert key in result, f"missing key: {key}"

    n_eps = len(result["episodes"])
    assert n_eps == 2
    assert result["global_returns"].shape == (n_eps,)
    assert result["per_agent_returns"].shape == (n_eps, result["n_agents"])
    assert np.all(np.isfinite(result["global_returns"]))
    assert np.all(result["cti_violation_rates"] >= 0.0)
    assert "action_projection_samples" in result
    assert set(result["action_projection_samples"].keys()) == {"raw", "projected"}


def test_run_core_A_vectorized_matches_real_aggregation_schema():
    """The cached, aggregated result must carry the exact key set the real
    Core A's _aggregate_runs() produces (confirmed against a real cache
    entry during development -- see elegant-rolling-map.md's Phase V14.1
    notes), and must be loadable through run_post_processing.py's real
    load_cache() under its own (non-colliding) cache name."""
    cache = CacheManager(CACHE_DIR)
    result = run_core_A_vectorized(n_runs=1, n_episodes=2, seed=4242, cache=cache)

    expected_keys = {
        "n_runs", "n_agents", "n_pairs", "episodes",
        "global_returns_mean", "global_returns_std", "global_returns_all",
        "cti_violation_rates_mean", "cti_violation_rates_std", "cti_violation_rates_all",
        "lagrange_norms_mean", "lagrange_norms_std", "lagrange_norms_all",
        "safety_activations_mean", "safety_activations_std", "safety_activations_all",
        "action_projections_mean", "action_projections_std", "action_projections_all",
        "cumulative_violations_mean", "cumulative_violations_std", "cumulative_violations_all",
        "per_agent_returns_mean", "per_agent_returns_std", "per_agent_returns_all",
        "config", "checkpoint_path",
    }
    assert expected_keys.issubset(set(result.keys())), (
        f"missing keys: {expected_keys - set(result.keys())}"
    )
    assert result["n_runs"] == 1
    assert result["config"]["name"] == "core_A_mappo_safe_ewc_8bus_60pv_vectorized"
    assert result["config"]["name"] != "core_A_mappo_safe_ewc_8bus_60pv", (
        "must never collide with the real Core A cache name"
    )

    # Round-trip through the real figure-pipeline loader. load_cache()'s own
    # "most recent wins" lookup (run_post_processing.py) resolves by name
    # across EVERY entry ever cached under that name, not scoped to this
    # test call -- repeated test-suite runs over time leave multiple
    # differently-seeded entries sharing this one name, so whichever was
    # created most recently overall (not necessarily this call's `result`)
    # is what comes back. Checking schema equivalence (same keys/shapes),
    # not bit-for-bit equality against `result`, is the correct and robust
    # thing to assert here -- it is what "genuinely schema-compatible, not
    # just shape-compatible on paper" actually means for this test.
    import run_post_processing as rpp
    loaded = rpp.load_cache("core_A_mappo_safe_ewc_8bus_60pv_vectorized")
    assert loaded is not None
    assert expected_keys.issubset(set(loaded.keys()))
    assert loaded["global_returns_mean"].shape == result["global_returns_mean"].shape
    assert np.all(np.isfinite(loaded["global_returns_mean"]))


def test_run_core_A_vectorized_cache_hit_on_second_call():
    """Calling with an identical config must hit the cache, not retrain."""
    cache = CacheManager(CACHE_DIR)
    first = run_core_A_vectorized(n_runs=1, n_episodes=2, seed=5151, cache=cache)
    hits_before = cache.get_stats()["hits"]
    second = run_core_A_vectorized(n_runs=1, n_episodes=2, seed=5151, cache=cache)
    assert cache.get_stats()["hits"] == hits_before + 1
    assert np.allclose(first["global_returns_mean"], second["global_returns_mean"])


# ═════════════════════════════════════════════════════════════════
# Core B (Phase V14.2)
# ═════════════════════════════════════════════════════════════════
# n_eval_episodes is overridden way down from the real Core B's
# hardcoded 30 (run_core_training.py:1551,1601) purely for test runtime --
# see _run_curriculum_single_vectorized's docstring. episodes_per_task=1
# and n_eval_episodes=2 together keep each full with_ewc/without_ewc pair
# under a few minutes instead of the ~1.5h a faithful n_eval_episodes=30
# smoke test took during development.

def test_run_core_B_vectorized_single_schema():
    """One curriculum run's raw result must carry every field the real
    _run_curriculum_single() returns, with the shapes run_core_B_vectorized
    expects to collect into with_ewc/without_ewc/task_evaluations."""
    result = _run_curriculum_single_vectorized(
        use_ewc=True, seed=6161, episodes_per_task=1, system="8bus", n_eval_episodes=2)

    for key in ["task_performances", "use_ewc", "final_checkpoint", "pv_evaluation", "fisher", "history"]:
        assert key in result, f"missing key: {key}"

    TASKS = [0.0, 0.20, 0.40, 0.60]
    assert result["task_performances"].shape == (len(TASKS), len(TASKS))
    assert np.all(np.isfinite(result["task_performances"]))
    assert result["use_ewc"] is True
    assert result["fisher"] is not None
    assert set(result["pv_evaluation"].keys()) == set(TASKS)
    for eval_pv in TASKS:
        entry = result["pv_evaluation"][eval_pv]
        assert set(entry.keys()) == {"objective", "cti_rate"}
        assert np.isfinite(entry["objective"])
        assert 0.0 <= entry["cti_rate"] <= 1.0

    no_ewc = _run_curriculum_single_vectorized(
        use_ewc=False, seed=16161, episodes_per_task=1, system="8bus", n_eval_episodes=2)
    assert no_ewc["fisher"] is None
    assert no_ewc["use_ewc"] is False


def test_run_core_B_vectorized_matches_real_aggregation_schema():
    """The cached result must carry the exact key set the real run_core_B()
    produces (with_ewc/without_ewc/task_evaluations), loadable through
    run_post_processing.py's real load_cache() under its own
    (non-colliding) cache name."""
    cache = CacheManager(CACHE_DIR)
    result = run_core_B_vectorized(
        n_runs=1, episodes_per_task=1, seed=7171, system="8bus",
        cache=cache, n_eval_episodes=2)

    assert set(result.keys()) == {"with_ewc", "without_ewc", "task_evaluations"}
    assert len(result["with_ewc"]) == 1
    assert len(result["without_ewc"]) == 1
    assert len(result["task_evaluations"]) == 1
    assert result["task_evaluations"][0] == result["with_ewc"][0]["pv_evaluation"]

    expected_name = "core_B_curriculum_vectorized"
    assert expected_name != "core_B_curriculum", (
        "must never collide with the real Core B cache name"
    )

    # Same test-isolation lesson as Core A's equivalent test (see its
    # docstring): load_cache()'s "most recent wins" lookup resolves across
    # every entry ever cached under this shared name, not scoped to this
    # call -- assert schema/shape, not bit-for-bit equality against `result`.
    import run_post_processing as rpp
    loaded = rpp.load_cache(expected_name)
    assert loaded is not None
    assert set(loaded.keys()) == {"with_ewc", "without_ewc", "task_evaluations"}
    assert loaded["with_ewc"][0]["task_performances"].shape == (4, 4)


def test_run_core_B_vectorized_cache_hit_on_second_call():
    """Calling with an identical config must hit the cache, not retrain."""
    cache = CacheManager(CACHE_DIR)
    first = run_core_B_vectorized(
        n_runs=1, episodes_per_task=1, seed=8181, system="8bus",
        cache=cache, n_eval_episodes=2)
    hits_before = cache.get_stats()["hits"]
    second = run_core_B_vectorized(
        n_runs=1, episodes_per_task=1, seed=8181, system="8bus",
        cache=cache, n_eval_episodes=2)
    assert cache.get_stats()["hits"] == hits_before + 1
    assert np.allclose(
        first["with_ewc"][0]["task_performances"],
        second["with_ewc"][0]["task_performances"],
    )


# ═════════════════════════════════════════════════════════════════
# Core C (Phase V14.3) -- 5-method baseline comparison
# ═════════════════════════════════════════════════════════════════

def test_run_core_C_vectorized_traditional_schema():
    """The zero-action "Traditional" baseline has no trainer at all --
    verify its own dedicated function's schema directly."""
    returns, metrics = _evaluate_traditional_vectorized(
        load_network("8bus"), n_episodes=2, seed=3030)
    assert returns.shape == (2,)
    assert np.all(np.isfinite(returns))
    assert set(metrics.keys()) == {
        "cti_satisfaction_rate", "convergence_ep", "sample_efficiency",
        "robustness", "top_pri_mean", "delta_cti_mean", "final_of",
    }
    assert metrics["convergence_ep"] == 0
    assert metrics["robustness"] == 1.0
    assert metrics["sample_efficiency"] is None
    assert metrics["final_of"] is None
    assert np.isfinite(metrics["top_pri_mean"])
    assert np.isfinite(metrics["delta_cti_mean"])


def test_run_core_C_vectorized_single_schema_per_method():
    """Every non-Traditional method must carry the same real metrics
    schema, with real (non-None) convergence_ep/sample_efficiency/
    robustness/top_pri_mean/delta_cti_mean/final_of -- confirming the
    real boolean table (use_comm/use_safe/use_ewc per method) and the
    EWC path (MAPPO-Safe-EWC only) all actually ran rather than silently
    no-op'ing."""
    expected_keys = {
        "cti_satisfaction_rate", "convergence_ep", "sample_efficiency",
        "robustness", "top_pri_mean", "delta_cti_mean", "final_of",
    }
    for method in ["MAPPO-Safe-EWC", "MAPPO-Safe", "MAPPO", "IPPO"]:
        returns, metrics = _run_baseline_single_vectorized(
            method, seed=4040, n_episodes=2, system="8bus")
        assert returns.shape == (2,), f"{method}: wrong returns shape"
        assert np.all(np.isfinite(returns)), f"{method}: non-finite returns"
        assert set(metrics.keys()) == expected_keys, f"{method}: wrong metric keys"
        for key in ["convergence_ep", "sample_efficiency", "robustness",
                    "top_pri_mean", "delta_cti_mean", "final_of"]:
            assert metrics[key] is not None, f"{method}: {key} unexpectedly None"
            assert np.isfinite(metrics[key]), f"{method}: {key} non-finite"


def test_run_core_C_vectorized_matches_real_aggregation_schema():
    """The cached result must carry one np.ndarray per method (the exact
    shape the real run_core_C() returns), plus a separately-cached
    per-method metrics entry under its own (non-colliding) name --
    loadable through run_post_processing.py's real load_cache()."""
    cache = CacheManager(CACHE_DIR)
    result = run_core_C_vectorized(n_runs=1, n_episodes=2, seed=5050, system="8bus", cache=cache)

    assert set(result.keys()) == set(_CORE_C_METHODS)
    for method, returns in result.items():
        assert returns.shape == (1, 2), f"{method}: wrong shape {returns.shape}"
        assert np.all(np.isfinite(returns))

    expected_name = "core_C_baselines_8bus_vectorized"
    assert expected_name != "core_C_baselines_8bus", (
        "must never collide with the real Core C cache name"
    )

    import run_post_processing as rpp
    loaded = rpp.load_cache(expected_name)
    assert loaded is not None
    assert set(loaded.keys()) == set(_CORE_C_METHODS)
    assert loaded["Traditional"].shape[1] == 2

    metrics_loaded = rpp.load_cache("core_C_metrics_8bus_vectorized")
    assert metrics_loaded is not None
    # Traditional's metrics are real (non-None values throughout), so it
    # must be present alongside the 4 trainer-based methods.
    assert set(metrics_loaded.keys()) == set(_CORE_C_METHODS)
    assert "per_run" in metrics_loaded["MAPPO-Safe-EWC"]


def test_run_core_C_vectorized_cache_hit_on_second_call():
    """Calling with an identical config must hit the cache, not retrain."""
    cache = CacheManager(CACHE_DIR)
    first = run_core_C_vectorized(n_runs=1, n_episodes=2, seed=6060, system="8bus", cache=cache)
    hits_before = cache.get_stats()["hits"]
    second = run_core_C_vectorized(n_runs=1, n_episodes=2, seed=6060, system="8bus", cache=cache)
    assert cache.get_stats()["hits"] == hits_before + 1
    for method in _CORE_C_METHODS:
        assert np.allclose(first[method], second[method])


# ═════════════════════════════════════════════════════════════════
# Core D (Phase V14.4) -- 5-arm ablation study
# ═════════════════════════════════════════════════════════════════

def test_run_core_D_vectorized_single_arm_schema():
    """Every ablation arm must return a finite (OF, CTI) scalar pair --
    covering the plain arm (no comm/safe/EWC/curriculum) and the fully-
    loaded arm (comm+safe+EWC+curriculum, exercising the 4-stage
    curriculum-with-EWC-task-transition branch) confirms both code paths
    inside _run_ablation_single_vectorized actually run end to end."""
    for config_name in ["MAPPO", "MAPPO+GNN+Safe+EWC+Curr"]:
        of_val, cti_val = _run_ablation_single_vectorized(config_name, seed=8080, n_episodes=1)
        assert np.isfinite(of_val), f"{config_name}: non-finite OF"
        assert np.isfinite(cti_val), f"{config_name}: non-finite CTI"
        assert cti_val >= 0.0


def test_run_core_D_vectorized_matches_real_aggregation_schema():
    """The cached result must carry the exact key set the real
    run_core_D() produces: one list of {"OF", "CTI"} dicts per ablation
    config, loadable through run_post_processing.py's real
    load_cache()."""
    cache = CacheManager(CACHE_DIR)
    result = run_core_D_vectorized(n_runs=1, n_episodes=1, seed=9090, cache=cache)

    assert set(result.keys()) == set(_CORE_D_CONFIGS)
    for cfg_name, runs in result.items():
        assert len(runs) == 1, f"{cfg_name}: wrong n_runs"
        assert set(runs[0].keys()) == {"OF", "CTI"}, f"{cfg_name}: wrong keys"
        assert np.isfinite(runs[0]["OF"])
        assert np.isfinite(runs[0]["CTI"])

    expected_name = "core_D_ablation_vectorized"
    assert expected_name != "core_D_ablation", (
        "must never collide with the real Core D cache name"
    )

    import run_post_processing as rpp
    loaded = rpp.load_cache(expected_name)
    assert loaded is not None
    assert set(loaded.keys()) == set(_CORE_D_CONFIGS)
    assert set(loaded["MAPPO"][0].keys()) == {"OF", "CTI"}


def test_run_core_D_vectorized_cache_hit_on_second_call():
    """Calling with an identical config must hit the cache, not retrain."""
    cache = CacheManager(CACHE_DIR)
    first = run_core_D_vectorized(n_runs=1, n_episodes=1, seed=10100, cache=cache)
    hits_before = cache.get_stats()["hits"]
    second = run_core_D_vectorized(n_runs=1, n_episodes=1, seed=10100, cache=cache)
    assert cache.get_stats()["hits"] == hits_before + 1
    for cfg_name in _CORE_D_CONFIGS:
        assert first[cfg_name][0]["OF"] == second[cfg_name][0]["OF"]


# ═════════════════════════════════════════════════════════════════
# Core E (Phase V14.5) -- hyperparameter grid
# ═════════════════════════════════════════════════════════════════
# The real grid is 5x4x4=80 points x n_seeds -- grid overrides below use a
# single (lambda_ewc, alpha_lambda, hidden_dim) point purely for test
# runtime (each point is a full two-phase training run + fixed 30-episode
# eval loop, ~3 minutes even at this tiny n_steps) -- see
# run_core_E_vectorized's docstring.

def test_run_core_E_vectorized_single_point_schema():
    """One grid point must return a finite (OF, CTI) scalar pair --
    confirming the warm-start -> EWC-correction -> target-task training
    structure and the fixed 30-episode eval loop both run end to end."""
    of_val, cti_val = _run_hparam_single_vectorized(
        lambda_ewc=1e4, alpha_lambda=1e-3, hidden_dim=16, seed=12120, n_steps=4)
    assert np.isfinite(of_val)
    assert np.isfinite(cti_val)
    assert cti_val >= 0.0


def test_run_core_E_vectorized_matches_real_aggregation_schema():
    """The cached result must carry the exact structure the real
    run_core_E() produces: a flat list of per-grid-point dicts, each with
    the same keys the real _core_E_job() returns, loadable through
    run_post_processing.py's real load_cache()."""
    cache = CacheManager(CACHE_DIR)
    result = run_core_E_vectorized(
        n_seeds=1, n_steps=4, seed=13130, cache=cache,
        lambda_ewc_grid=[1e4], alpha_lambda_grid=[1e-3], hidden_dim_grid=[16])

    assert isinstance(result, list)
    assert len(result) == 1
    entry = result[0]
    assert set(entry.keys()) == {"lambda_ewc", "alpha_lambda", "hidden_dim", "seed", "OF", "CTI"}
    assert entry["lambda_ewc"] == 1e4
    assert entry["alpha_lambda"] == 1e-3
    assert entry["hidden_dim"] == 16
    assert np.isfinite(entry["OF"])
    assert np.isfinite(entry["CTI"])

    expected_name = "core_E_hparam_vectorized"
    assert expected_name != "core_E_hparam", (
        "must never collide with the real Core E cache name"
    )

    import run_post_processing as rpp
    loaded = rpp.load_cache(expected_name)
    assert loaded is not None
    assert isinstance(loaded, list)
    assert set(loaded[0].keys()) == {"lambda_ewc", "alpha_lambda", "hidden_dim", "seed", "OF", "CTI"}


def test_run_core_E_vectorized_cache_hit_on_second_call():
    """Calling with an identical config must hit the cache, not retrain."""
    cache = CacheManager(CACHE_DIR)
    first = run_core_E_vectorized(
        n_seeds=1, n_steps=4, seed=14140, cache=cache,
        lambda_ewc_grid=[1e4], alpha_lambda_grid=[1e-3], hidden_dim_grid=[16])
    hits_before = cache.get_stats()["hits"]
    second = run_core_E_vectorized(
        n_seeds=1, n_steps=4, seed=14140, cache=cache,
        lambda_ewc_grid=[1e4], alpha_lambda_grid=[1e-3], hidden_dim_grid=[16])
    assert cache.get_stats()["hits"] == hits_before + 1
    assert first[0]["OF"] == second[0]["OF"]


# ═════════════════════════════════════════════════════════════════
# Core F (Phase V14.6) -- scalability analysis
# ═════════════════════════════════════════════════════════════════
# Each system's run has a FIXED cost (100*300=30000 training steps + 10
# eval episodes, not parametrized by any n_episodes-style override -- see
# run_core_F_vectorized's docstring) -- tests use the smallest supported
# system (8bus) to keep runtime down, not a reduced-budget variant.

def test_run_core_F_vectorized_unsupported_system_raises():
    """VectorizedPowerFlowSolver requires real per-branch impedance data,
    which only 8bus/14bus/39bus have -- confirms the real, pre-existing
    Phase V-PF limitation surfaces as a clear error rather than silently
    producing wrong results for systems like 3bus/30bus/57bus/118bus."""
    with pytest.raises(ValueError, match="per-branch impedance data"):
        _run_scalability_single_vectorized("3bus", seed=16160)


def test_run_core_F_vectorized_single_system_schema():
    """One system's raw result must carry every field the real
    _run_scalability_single() returns, with finite values."""
    result = _run_scalability_single_vectorized("8bus", seed=17170)
    assert set(result.keys()) == {"n_relays", "train_time", "final_performance"}
    assert result["n_relays"] > 0
    assert result["train_time"] > 0.0
    assert np.isfinite(result["final_performance"])


def test_run_core_F_vectorized_matches_real_aggregation_schema():
    """The cached result must carry the exact {sys_name: {...}} structure
    the real run_core_F() produces, loadable through
    run_post_processing.py's real load_cache()."""
    cache = CacheManager(CACHE_DIR)
    result = run_core_F_vectorized(systems=["8bus"], seed=18180, cache=cache)

    assert set(result.keys()) == {"8bus"}
    assert set(result["8bus"].keys()) == {"n_relays", "train_time", "final_performance"}

    expected_name = "core_F_scalability_vectorized"
    assert expected_name != "core_F_scalability", (
        "must never collide with the real Core F cache name"
    )

    import run_post_processing as rpp
    loaded = rpp.load_cache(expected_name)
    assert loaded is not None
    assert "8bus" in loaded
    assert set(loaded["8bus"].keys()) == {"n_relays", "train_time", "final_performance"}


def test_run_core_F_vectorized_cache_hit_on_second_call():
    """Calling with an identical config must hit the cache, not retrain."""
    cache = CacheManager(CACHE_DIR)
    first = run_core_F_vectorized(systems=["8bus"], seed=19190, cache=cache)
    hits_before = cache.get_stats()["hits"]
    second = run_core_F_vectorized(systems=["8bus"], seed=19190, cache=cache)
    assert cache.get_stats()["hits"] == hits_before + 1
    assert first["8bus"]["final_performance"] == second["8bus"]["final_performance"]


# ═════════════════════════════════════════════════════════════════
# Core G (Phase V14.7) -- robustness evaluation
# ═════════════════════════════════════════════════════════════════
# Every _eval_*_vectorized function is pure inference against a FROZEN,
# already-trained base policy -- so each test below trains exactly ONE
# base policy (fast_mode=True) and exercises several eval functions
# against it, rather than retraining per assertion.

def test_core_G_eval_functions_schema():
    """Every Core G eval function must return finite, correctly-keyed
    output when run against a real (fast_mode) trained base policy."""
    net = load_network("8bus")
    result = _train_base_policy_vectorized(system="8bus", seed=21210, fast_mode=True)
    assert result.use_comm is True
    assert result.actor_params is not None
    assert result.comm_graph_params is not None

    d = _eval_comm_stress_vectorized(net, result, packet_loss_rate=0.2, seed=21210)
    assert np.isfinite(d["packet_loss"])

    d = _eval_n1_contingency_vectorized(net, result, line_idx=0, recovery=True, seed=21210)
    assert np.isfinite(d["n1_of"])
    assert np.isfinite(d["n1_recovery_of"])

    d = _eval_byzantine_vectorized(net, result, byzantine_agents=[1], strategy="fdi", seed=21210)
    assert np.isfinite(d["byzantine_of"])
    assert d["byzantine_trust_scores"].shape == (net.n_relays,)

    d = _eval_bandwidth_vectorized(net, result, compression_ratio=0.5, seed=21210)
    assert np.isfinite(d["of"])
    assert d["bandwidth"] > 0

    d = _eval_sensor_robustness_vectorized(net, result, noise_std=0.1, seed=21210)
    assert np.isfinite(d["noisy_of"])

    d = _eval_adversarial_vectorized(net, result, attack="FGSM", epsilon=0.05, seed=21210)
    assert np.isfinite(d["of"])
    assert 0.0 <= d["success"] <= 1.0


def test_run_core_G_vectorized_matches_schema():
    """The cached result must carry the real run_core_G()'s stress-type
    key set, loadable through run_post_processing.py's real load_cache()."""
    cache = CacheManager(CACHE_DIR)
    result = run_core_G_vectorized(seed=22220, cache=cache, fast_mode=True)

    assert set(result.keys()) == {"communication", "byzantine", "bandwidth", "sensor",
                                   "adversarial", "n1_contingency"}
    assert set(result["communication"].keys()) == {"packet_loss", "latency"}

    expected_name = "core_G_robustness_vectorized"
    assert expected_name != "core_G_robustness"

    import run_post_processing as rpp
    loaded = rpp.load_cache(expected_name)
    assert loaded is not None
    assert set(loaded.keys()) == set(result.keys())


def test_run_core_G_vectorized_cache_hit_on_second_call():
    cache = CacheManager(CACHE_DIR)
    first = run_core_G_vectorized(seed=23230, cache=cache, fast_mode=True)
    hits_before = cache.get_stats()["hits"]
    second = run_core_G_vectorized(seed=23230, cache=cache, fast_mode=True)
    assert cache.get_stats()["hits"] == hits_before + 1
    assert first["byzantine"] == second["byzantine"]


# ═════════════════════════════════════════════════════════════════
# Core H (Phase V14.7) -- transfer learning
# ═════════════════════════════════════════════════════════════════

def test_core_H_eval_functions_schema():
    """_eval_unseen_faults_vectorized/_eval_zero_shot_transfer_vectorized
    must return finite output against a real (fast_mode) trained policy."""
    net = load_network("8bus")
    result = _train_base_policy_vectorized(system="8bus", seed=24240, fast_mode=True)

    d = _eval_unseen_faults_vectorized(net, result, unseen_fault_type="LL", seed=24240)
    assert np.isfinite(d["unseen_of"])

    d = _eval_zero_shot_transfer_vectorized(result, eval_system="14bus", seed=24240)
    assert np.isfinite(d["transfer_of"])
    assert d["transfer_system"] == "14bus"


def test_run_core_H_vectorized_matches_schema():
    """The cached result must carry the real run_core_H()'s key set
    (unseen_faults + zero_shot, including the cold-start baseline),
    loadable through run_post_processing.py's real load_cache()."""
    cache = CacheManager(CACHE_DIR)
    result = run_core_H_vectorized(seed=25250, n_episodes=2, cache=cache, fast_mode=True)

    assert set(result.keys()) == {"unseen_faults", "zero_shot"}
    assert set(result["zero_shot"].keys()) >= {
        "transfer_of", "transfer_cti_ok", "transfer_system", "base_of", "cold_start"}
    assert set(result["zero_shot"]["cold_start"].keys()) == {"performance", "convergence_ep"}

    expected_name = "core_H_transfer_vectorized"
    assert expected_name != "core_H_transfer"

    import run_post_processing as rpp
    loaded = rpp.load_cache(expected_name)
    assert loaded is not None
    assert set(loaded.keys()) == set(result.keys())


def test_run_core_H_vectorized_cache_hit_on_second_call():
    cache = CacheManager(CACHE_DIR)
    first = run_core_H_vectorized(seed=26260, n_episodes=2, cache=cache, fast_mode=True)
    hits_before = cache.get_stats()["hits"]
    second = run_core_H_vectorized(seed=26260, n_episodes=2, cache=cache, fast_mode=True)
    assert cache.get_stats()["hits"] == hits_before + 1
    assert first["unseen_faults"] == second["unseen_faults"]


# ═════════════════════════════════════════════════════════════════
# Core I (Phase V14.7) -- deployment benchmark
# ═════════════════════════════════════════════════════════════════

def test_core_I_eval_functions_schema():
    """_measure_latency_vectorized/_eval_compression_vectorized must
    return finite output against a real (fast_mode) trained policy."""
    net = load_network("8bus")
    result = _train_base_policy_vectorized(system="8bus", seed=27270, fast_mode=True)

    latency = _measure_latency_vectorized(net, result, seed=27270)
    assert set(latency.keys()) == {"latencies_ms", "mean_ms", "p50_ms", "p99_ms"}
    assert latency["mean_ms"] > 0

    comp = _eval_compression_vectorized(net, result, quantize_bits=8, prune_ratio=0.0, seed=27270)
    assert np.isfinite(comp["of"])
    assert comp["size_kb"] > 0


def test_run_core_I_vectorized_matches_schema():
    """The cached result must carry the real run_core_I()'s key set
    (latency, latency_39bus, compression with 3 methods), loadable
    through run_post_processing.py's real load_cache()."""
    cache = CacheManager(CACHE_DIR)
    result = run_core_I_vectorized(seed=28280, cache=cache, fast_mode=True)

    assert set(result.keys()) == {"latency", "latency_39bus", "compression"}
    assert set(result["compression"].keys()) == {"original", "int8", "pruning"}
    for method_result in result["compression"].values():
        assert "size_mb" in method_result

    expected_name = "core_I_deployment_vectorized"
    assert expected_name != "core_I_deployment"

    import run_post_processing as rpp
    loaded = rpp.load_cache(expected_name)
    assert loaded is not None
    assert set(loaded.keys()) == set(result.keys())


def test_run_core_I_vectorized_cache_hit_on_second_call():
    cache = CacheManager(CACHE_DIR)
    first = run_core_I_vectorized(seed=29290, cache=cache, fast_mode=True)
    hits_before = cache.get_stats()["hits"]
    second = run_core_I_vectorized(seed=29290, cache=cache, fast_mode=True)
    assert cache.get_stats()["hits"] == hits_before + 1
    assert first["compression"]["int8"]["size_mb"] == second["compression"]["int8"]["size_mb"]


# ═════════════════════════════════════════════════════════════════
# Core J (Phase V14.7) -- protection engineering
# ═════════════════════════════════════════════════════════════════

def test_core_J_eval_functions_schema():
    """_eval_daily_pv_vectorized/_eval_sympathetic_tripping_vectorized
    must return finite, correctly-shaped output against a real
    (fast_mode) trained policy. Fig 44/46 (_eval_fault_contributions/
    _eval_relay_curves) are exercised via run_core_J_vectorized below,
    since they are reused UNMODIFIED from the real TrainingPipeline, not
    reimplemented here."""
    net = load_network("8bus")
    result = _train_base_policy_vectorized(system="8bus", seed=30300, fast_mode=True)

    daily = _eval_daily_pv_vectorized(net, result, seed=30300)
    assert daily["tms"].shape == (24, net.n_relays)
    assert daily["of"].shape == (24,)

    sym = _eval_sympathetic_tripping_vectorized(net, result, seed=30300)
    assert set(sym["by_fault_type"].keys()) == {"1ph", "LL", "3ph"}
    for counts in sym["by_fault_type"].values():
        assert counts["marl"] >= 0 and counts["traditional"] >= 0


def test_run_core_J_vectorized_matches_schema():
    """The cached result must carry the real run_core_J()'s 5-analysis
    key set, loadable through run_post_processing.py's real
    load_cache(). fault_contribution/relay_curves come from the real,
    unmodified TrainingPipeline._eval_fault_contributions/_eval_relay_curves."""
    cache = CacheManager(CACHE_DIR)
    result = run_core_J_vectorized(seed=31310, cache=cache, fast_mode=True)

    assert set(result.keys()) == {"coordination", "fault_contribution", "daily_profile",
                                   "relay_curves", "sympathetic"}
    assert "error" not in result["fault_contribution"], result["fault_contribution"]
    assert len(result["relay_curves"]) > 0

    expected_name = "core_J_protection_vectorized"
    assert expected_name != "core_J_protection"

    import run_post_processing as rpp
    loaded = rpp.load_cache(expected_name)
    assert loaded is not None
    assert set(loaded.keys()) == set(result.keys())


def test_run_core_J_vectorized_cache_hit_on_second_call():
    cache = CacheManager(CACHE_DIR)
    first = run_core_J_vectorized(seed=32320, cache=cache, fast_mode=True)
    hits_before = cache.get_stats()["hits"]
    second = run_core_J_vectorized(seed=32320, cache=cache, fast_mode=True)
    assert cache.get_stats()["hits"] == hits_before + 1
    assert first["sympathetic"] == second["sympathetic"]


# ═════════════════════════════════════════════════════════════════
# Core K (Phase V14.7) -- seasonal generalization
# ═════════════════════════════════════════════════════════════════

def test_core_K_eval_seasonal_schema():
    """_eval_seasonal_vectorized must return finite output for every
    season, against a real (fast_mode) trained policy."""
    net = load_network("8bus")
    result = _train_base_policy_vectorized(system="8bus", seed=33330, fast_mode=True)

    for season in ["spring", "summer", "autumn", "winter"]:
        d = _eval_seasonal_vectorized(net, result, season=season, seed=33330)
        assert np.isfinite(d["of"])
        assert np.isfinite(d["pv_variability"])


def test_run_core_K_vectorized_matches_schema():
    """The cached result must carry the real run_core_K()'s 4-season
    key set, loadable through run_post_processing.py's real load_cache()."""
    cache = CacheManager(CACHE_DIR)
    result = run_core_K_vectorized(seed=34340, cache=cache, fast_mode=True)

    assert set(result.keys()) == {"spring", "summer", "autumn", "winter"}
    for season_result in result.values():
        assert "performance" in season_result
        assert np.isfinite(season_result["of"])

    expected_name = "core_K_seasonal_vectorized"
    assert expected_name != "core_K_seasonal"

    import run_post_processing as rpp
    loaded = rpp.load_cache(expected_name)
    assert loaded is not None
    assert set(loaded.keys()) == set(result.keys())


def test_run_core_K_vectorized_cache_hit_on_second_call():
    cache = CacheManager(CACHE_DIR)
    first = run_core_K_vectorized(seed=35350, cache=cache, fast_mode=True)
    hits_before = cache.get_stats()["hits"]
    second = run_core_K_vectorized(seed=35350, cache=cache, fast_mode=True)
    assert cache.get_stats()["hits"] == hits_before + 1
    assert first["summer"]["of"] == second["summer"]["of"]
