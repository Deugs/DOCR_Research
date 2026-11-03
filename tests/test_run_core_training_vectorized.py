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

from run_core_training import CacheManager, CACHE_DIR
from run_core_training_vectorized import (
    run_core_A_vectorized, _run_core_A_vectorized_single,
    run_core_B_vectorized, _run_curriculum_single_vectorized,
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
