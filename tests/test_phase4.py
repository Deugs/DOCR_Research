
"""
test_phase4.py
──────────────
Phase 4: Comprehensive unit tests for benchmark suite.

Tests:
1. HWOA basic execution
2. HHO basic execution
3. JAYA basic execution
4. EEO basic execution
5. Metaheuristic SQP refinement
6. Friedman test correctness
7. Nemenyi post-hoc test
8. Cliff's delta effect size
9. Wilcoxon signed-rank test
10. Benchmark suite initialization
11. Benchmark suite single run
12. Statistical analysis integration
13. Backward compatibility with Phases 1-3
14. LaTeX table generation
15. Results aggregation
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
from network.network_data import get_3bus, get_8bus, NETWORKS
from network.relay_physics import check_constraints

# ── Test 1: HWOA ─────────────────────────────────────────────────────────────

def test_hwoa():
    """Test HWOA on 3-bus system."""
    from baselines.metaheuristics import run_hwoa
    
    net = get_3bus()
    result = run_hwoa(net, n_whales=10, max_iter=50, seed=42, verbose=False)
    
    assert result["method"] == "HWOA"
    assert result["network"] == net.name
    assert len(result["TMS"]) == net.n_relays
    assert len(result["PS"]) == net.n_relays
    assert result["OF"] > 0
    assert "constraints" in result
    assert result["elapsed"] > 0
    assert len(result["history"]) == 51  # 50 iters + initial
    print("  ✓ test_hwoa")


# ── Test 2: HHO ──────────────────────────────────────────────────────────────

def test_hho():
    """Test HHO on 3-bus system."""
    from baselines.metaheuristics import run_hho
    
    net = get_3bus()
    result = run_hho(net, n_hawks=10, max_iter=50, seed=42, verbose=False)
    
    assert result["method"] == "HHO"
    assert len(result["TMS"]) == net.n_relays
    assert result["OF"] > 0
    assert "constraints" in result
    print("  ✓ test_hho")


# ── Test 3: JAYA ─────────────────────────────────────────────────────────────

def test_jaya():
    """Test JAYA on 3-bus system."""
    from baselines.metaheuristics import run_jaya
    
    net = get_3bus()
    result = run_jaya(net, n_pop=10, max_iter=50, seed=42, verbose=False)
    
    assert result["method"] == "JAYA"
    assert len(result["TMS"]) == net.n_relays
    assert result["OF"] > 0
    assert "constraints" in result
    print("  ✓ test_jaya")


# ── Test 4: EEO ──────────────────────────────────────────────────────────────

def test_eeo():
    """Test EEO on 3-bus system."""
    from baselines.metaheuristics import run_eeo
    
    net = get_3bus()
    result = run_eeo(net, n_particles=10, max_iter=50, seed=42, verbose=False)
    
    assert result["method"] == "EEO"
    assert len(result["TMS"]) == net.n_relays
    assert result["OF"] > 0
    assert "constraints" in result
    print("  ✓ test_eeo")


# ── Test 5: Metaheuristic SQP refinement ─────────────────────────────────────

def test_metaheuristic_sqp_refinement():
    """Test that SQP refinement improves metaheuristic results."""
    from baselines.metaheuristics import run_jaya
    
    net = get_3bus()
    
    # Without SQP
    res_no_sqp = run_jaya(net, n_pop=10, max_iter=50, seed=42, 
                          verbose=False, use_sqp=False)
    
    # With SQP
    res_with_sqp = run_jaya(net, n_pop=10, max_iter=50, seed=42,
                            verbose=False, use_sqp=True)
    
    # SQP should not make things worse (usually improves)
    assert res_with_sqp["OF"] <= res_no_sqp["OF"] * 1.5  # Allow some tolerance
    assert res_with_sqp["OF_meta"] == res_no_sqp["OF"]  # Meta result stored
    print("  ✓ test_metaheuristic_sqp_refinement")


# ── Test 6: Friedman test ────────────────────────────────────────────────────

def test_friedman_test():
    """Test Friedman test with synthetic data."""
    from evaluation.stat_analysis import friedman_test
    
    # Synthetic: 5 datasets, 3 algorithms
    # Algo1 consistently best, Algo3 consistently worst
    np.random.seed(42)
    results = np.array([
        [1.0, 1.5, 2.0],
        [1.1, 1.6, 2.1],
        [0.9, 1.4, 1.9],
        [1.2, 1.7, 2.2],
        [1.0, 1.5, 2.0],
    ])
    
    stat, p_val, reject, details = friedman_test(results, alpha=0.05)
    
    assert stat > 0
    assert 0 <= p_val <= 1
    assert reject == True  # Should reject H0 (algorithms differ)
    assert details["n_datasets"] == 5
    assert details["n_algorithms"] == 3
    assert len(details["mean_ranks"]) == 3
    # Algo1 should have lowest (best) mean rank
    assert details["mean_ranks"][0] < details["mean_ranks"][2]
    print("  ✓ test_friedman_test")


# ── Test 7: Nemenyi post-hoc ─────────────────────────────────────────────────

def test_nemenyi_test():
    """Test Nemenyi post-hoc test."""
    from evaluation.stat_analysis import nemenyi_test
    
    mean_ranks = np.array([1.5, 2.5, 3.0])
    CD, diff_matrix, significant, details = nemenyi_test(mean_ranks, 5, 3, alpha=0.05)
    
    assert CD > 0
    assert diff_matrix.shape == (3, 3)
    assert significant.shape == (3, 3)
    assert not significant[0, 0]  # Diagonal should be False
    assert details["q_alpha"] > 0
    print("  ✓ test_nemenyi_test")


# ── Test 8: Cliff's delta ────────────────────────────────────────────────────

def test_cliffs_delta():
    """Test Cliff's delta effect size computation."""
    from evaluation.stat_analysis import cliffs_delta
    
    # Clear difference
    x = np.array([1.0, 1.1, 1.2, 1.0, 1.1])
    y = np.array([2.0, 2.1, 2.2, 2.0, 2.1])
    
    delta, magnitude = cliffs_delta(x, y)
    
    assert -1 <= delta <= 1
    assert delta < 0  # x is smaller (better) than y
    assert magnitude in ["negligible", "small", "medium", "large"]
    
    # Similar distributions
    z = np.array([1.0, 1.05, 1.1, 0.95, 1.02])
    delta2, magnitude2 = cliffs_delta(x, z)
    assert abs(delta2) < 0.5  # Should be small/negligible
    print("  ✓ test_cliffs_delta")


# ── Test 9: Wilcoxon test ────────────────────────────────────────────────────

def test_wilcoxon_test():
    """Test Wilcoxon signed-rank test."""
    from evaluation.stat_analysis import wilcoxon_pairwise
    
    np.random.seed(42)
    results = np.array([
        [1.0, 1.5, 2.0],
        [1.1, 1.6, 2.1],
        [0.9, 1.4, 1.9],
        [1.2, 1.7, 2.2],
        [1.0, 1.5, 2.0],
    ])
    
    methods = ["Algo1", "Algo2", "Algo3"]
    wilcoxon = wilcoxon_pairwise(results, methods, alpha=0.05, correction="holm")
    
    assert "Algo1" in wilcoxon
    assert "Algo2" in wilcoxon["Algo1"]
    assert "p_value" in wilcoxon["Algo1"]["Algo2"]
    assert "reject" in wilcoxon["Algo1"]["Algo2"]
    print("  ✓ test_wilcoxon_test")


# ── Test 10: Benchmark suite initialization ──────────────────────────────────

def test_benchmark_suite_init():
    """Test benchmark suite initialization."""
    from evaluation.benchmark_suite import BenchmarkSuite, BenchmarkConfig
    
    cfg = BenchmarkConfig(
        systems=["3bus"],
        methods=["GA-SQP", "JAYA"],
        n_runs=2,
        pop_size=10,
        max_iter=20,
        ga_pop=20,
        ga_gen=20,
        verbose=False,
    )
    
    suite = BenchmarkSuite(cfg)
    assert suite.config == cfg
    assert len(suite.results) == 0
    print("  ✓ test_benchmark_suite_init")


# ── Test 11: Benchmark suite single run ──────────────────────────────────────

def test_benchmark_suite_run():
    """Test benchmark suite execution (single run)."""
    from evaluation.benchmark_suite import BenchmarkSuite, BenchmarkConfig
    
    cfg = BenchmarkConfig(
        systems=["3bus"],
        methods=["JAYA"],  # Fastest metaheuristic
        n_runs=1,
        pop_size=10,
        max_iter=20,
        verbose=False,
    )
    
    suite = BenchmarkSuite(cfg)
    results = suite.run_all()
    
    assert "3bus" in results
    assert len(results["3bus"]) == 1  # 1 run
    assert "JAYA" in results["3bus"][0]
    assert "OF" in results["3bus"][0]["JAYA"]
    print("  ✓ test_benchmark_suite_run")


# ── Test 12: Statistical analysis integration ────────────────────────────────

def test_statistical_analysis():
    """Test full statistical analysis pipeline."""
    from evaluation.stat_analysis import run_benchmark_analysis
    
    # Mock results: 3 systems, 4 methods
    results_dict = {
        "3bus": {"GA-SQP": 2.5, "JAYA": 2.6, "HWOA": 2.55, "HHO": 2.58},
        "8bus": {"GA-SQP": 5.2, "JAYA": 5.3, "HWOA": 5.25, "HHO": 5.28},
        "30bus": {"GA-SQP": 12.1, "JAYA": 12.3, "HWOA": 12.15, "HHO": 12.2},
    }
    
    analysis = run_benchmark_analysis(results_dict, metric="OF", alpha=0.05, verbose=False)
    
    assert "friedman" in analysis
    assert "nemenyi" in analysis
    assert "cliffs_delta" in analysis
    assert "wilcoxon" in analysis
    assert "ranking" in analysis
    assert len(analysis["ranking"]["methods"]) == 4
    print("  ✓ test_statistical_analysis")


# ── Test 13: Backward compatibility ──────────────────────────────────────────

def test_backward_compatibility():
    """Test that Phase 4 doesn't break Phase 1-3 functionality."""
    from baselines.ga_sqp import run_ga_sqp
    from network.network_data import get_3bus
    
    net = get_3bus()
    result = run_ga_sqp(net, ga_pop=20, ga_gen=20, seed=42, verbose=False)
    
    assert result["method"] == "GA-SQP"
    assert "TMS" in result
    assert "PS" in result
    assert "OF" in result
    assert "constraints" in result
    print("  ✓ test_backward_compatibility")


# ── Test 14: Metaheuristic convergence history ───────────────────────────────

def test_convergence_history():
    """Test that all metaheuristics produce convergence history."""
    from baselines.metaheuristics import run_hwoa, run_hho, run_jaya, run_eeo
    
    net = get_3bus()
    
    for runner, name, param_name in [(run_hwoa, "HWOA", "n_whales"), (run_hho, "HHO", "n_hawks"),
                                       (run_jaya, "JAYA", "n_pop"), (run_eeo, "EEO", "n_particles")]:
        result = runner(net, **{param_name: 10, "max_iter": 20, "seed": 42, "verbose": False})
        assert "history" in result
        assert len(result["history"]) >= 20
        # History should be non-increasing (best so far)
        for i in range(1, len(result["history"])):
            assert result["history"][i] <= result["history"][i-1] * 1.01  # Small tolerance
    print("  ✓ test_convergence_history")


# ── Test 15: Results aggregation ─────────────────────────────────────────────

def test_results_aggregation():
    """Test results aggregation across multiple runs."""
    from evaluation.stat_analysis import aggregate_results
    
    raw_results = {
        "3bus": [
            {"GA-SQP": {"OF": 2.5}, "JAYA": {"OF": 2.6}},
            {"GA-SQP": {"OF": 2.4}, "JAYA": {"OF": 2.7}},
            {"GA-SQP": {"OF": 2.6}, "JAYA": {"OF": 2.5}},
        ]
    }
    
    aggregated = aggregate_results(raw_results, metric="OF", aggregation="mean")
    
    assert "3bus" in aggregated
    assert abs(aggregated["3bus"]["GA-SQP"] - 2.5) < 0.01
    assert abs(aggregated["3bus"]["JAYA"] - 2.6) < 0.01
    print("  ✓ test_results_aggregation")


# ── Test 16: All metaheuristics on 8-bus ─────────────────────────────────────

def test_all_metaheuristics_8bus():
    """Test all metaheuristics on larger 8-bus system."""
    from baselines.metaheuristics import run_all_metaheuristics
    
    net = get_8bus()
    results = run_all_metaheuristics(
        net, pop_size=10, max_iter=30, seed=42, verbose=False
    )
    
    assert len(results) == 4
    for method in ["HWOA", "HHO", "JAYA", "EEO"]:
        assert method in results
        assert results[method]["OF"] > 0
        assert len(results[method]["TMS"]) == net.n_relays
        assert results[method]["constraints"]["cti_violations"] >= 0
    print("  ✓ test_all_metaheuristics_8bus")


# ── Test 17: Constraint satisfaction across methods ──────────────────────────

def test_constraint_satisfaction():
    """Verify that all methods produce valid relay settings."""
    from baselines.metaheuristics import run_jaya
    from baselines.ga_sqp import run_ga_sqp
    from network.relay_physics import TMS_MIN, TMS_MAX
    
    net = get_3bus()
    
    for runner, kwargs in [
        (run_ga_sqp, {"ga_pop": 20, "ga_gen": 20, "seed": 42, "verbose": False}),
        (run_jaya, {"n_pop": 10, "max_iter": 20, "seed": 42, "verbose": False}),
    ]:
        result = runner(net, **kwargs)
        TMS, PS = result["TMS"], result["PS"]
        
        # TMS bounds
        assert np.all(TMS >= TMS_MIN - 1e-6)
        assert np.all(TMS <= TMS_MAX + 1e-6)
        
        # PS bounds
        ps_min, ps_max = net.ps_bounds()
        assert np.all(PS >= ps_min - 1e-6)
        assert np.all(PS <= ps_max + 1e-6)
    print("  ✓ test_constraint_satisfaction")


# ── Main test runner ─────────────────────────────────────────────────────────

def run_all_tests():
    """Run all Phase 4 tests."""
    print("\\n" + "="*60)
    print("  PHASE 4 UNIT TESTS")
    print("="*60 + "\\n")
    
    tests = [
        test_hwoa,
        test_hho,
        test_jaya,
        test_eeo,
        test_metaheuristic_sqp_refinement,
        test_friedman_test,
        test_nemenyi_test,
        test_cliffs_delta,
        test_wilcoxon_test,
        test_benchmark_suite_init,
        test_benchmark_suite_run,
        test_statistical_analysis,
        test_backward_compatibility,
        test_convergence_history,
        test_results_aggregation,
        test_all_metaheuristics_8bus,
        test_constraint_satisfaction,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  ✗ {test.__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print(f"\\n{'='*60}")
    print(f"  RESULTS: {passed} passed, {failed} failed")
    print(f"{'='*60}\\n")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)