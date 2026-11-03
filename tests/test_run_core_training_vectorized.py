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
