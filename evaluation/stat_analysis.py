"""
stat_analysis.py
────────────────
Phase 4: Statistical validation for DOCR benchmark comparison.

Implements:
1. Friedman Test      — Non-parametric ANOVA for multiple algorithms
2. Nemenyi Post-Hoc   — Pairwise comparison after Friedman
3. Cliff's Delta      — Effect size measure for pairwise comparisons
4. Wilcoxon Signed-Rank — Alternative pairwise test (optional)
5. Holm-Bonferroni    — p-value correction for multiple comparisons
6. Ranking tables     — Mean rank computation
7. CD diagrams        — Critical Difference diagrams for visualization

All tests are non-parametric (distribution-free), appropriate for
stochastic optimization algorithms.

References:
- Demšar, "Statistical Comparisons of Classifiers over Multiple Data Sets", JMLR 2006
- García & Herrera, "An Extension on Statistical Comparisons of Classifiers", 2008
- Cliff, "Ordinal Methods for Behavioral Data Analysis", 1996
"""

import numpy as np
from scipy import stats
from scipy.special import gammaln
from typing import Dict, List, Tuple, Optional
import warnings


# ── Friedman Test ────────────────────────────────────────────────────────────

def friedman_test(
    results: np.ndarray,
    alpha: float = 0.05,
) -> Tuple[float, float, bool, Dict]:
    """
    Friedman test for comparing multiple algorithms across multiple datasets.
    
    Parameters:
        results: (n_datasets, n_algorithms) array of performance values
                 Lower is better (e.g., objective function values)
        alpha: Significance level (default 0.05)
    
    Returns:
        statistic: Chi-square statistic
        p_value: p-value
        reject_null: True if null hypothesis rejected
        details: Dict with ranks, mean ranks, etc.
    
    Null hypothesis H0: All algorithms perform equivalently.
    Alternative H1: At least one algorithm performs differently.
    """
    n_datasets, n_algorithms = results.shape
    
    if n_datasets < 2 or n_algorithms < 2:
        raise ValueError("Need at least 2 datasets and 2 algorithms")
    
    # Rank algorithms per dataset (1 = best, n_algorithms = worst)
    ranks = np.empty_like(results, dtype=float)
    for i in range(n_datasets):
        # argsort of argsort gives ranks (0-based, so add 1)
        ranks[i] = np.argsort(np.argsort(results[i])) + 1
    
    mean_ranks = np.mean(ranks, axis=0)
    
    # Friedman statistic
    # χ²_F = (12N / k(k+1)) * [ΣR_j² - k(k+1)²/4]
    # where N = n_datasets, k = n_algorithms, R_j = sum of ranks for algo j
    N, k = n_datasets, n_algorithms
    sum_ranks = np.sum(ranks, axis=0)
    
    chi2_stat = (12.0 * N / (k * (k + 1))) * (
        np.sum(sum_ranks ** 2) - N * N * k * (k + 1) ** 2 / 4.0
    )
    
    # Iman-Davenport correction (more accurate for small N)
    # F_F = ((N - 1) * χ²_F) / (N * (k - 1) - χ²_F)
    if chi2_stat < N * (k - 1):
        f_stat = ((N - 1) * chi2_stat) / (N * (k - 1) - chi2_stat)
        df1, df2 = k - 1, (k - 1) * (N - 1)
        p_value = 1.0 - stats.f.cdf(f_stat, df1, df2)
        statistic = f_stat
        test_name = "Iman-Davenport F"
    else:
        # Fallback to chi-square
        p_value = 1.0 - stats.chi2.cdf(chi2_stat, k - 1)
        statistic = chi2_stat
        test_name = "Friedman chi-square"
    
    reject_null = p_value < alpha
    
    details = {
        "test_name": test_name,
        "n_datasets": N,
        "n_algorithms": k,
        "ranks": ranks,
        "mean_ranks": mean_ranks,
        "sum_ranks": sum_ranks,
        "alpha": alpha,
    }
    
    return statistic, p_value, reject_null, details


# ── Nemenyi Post-Hoc Test ──────────────────────────────────────────────────

def nemenyi_test(
    mean_ranks: np.ndarray,
    n_datasets: int,
    n_algorithms: int,
    alpha: float = 0.05,
) -> Tuple[float, np.ndarray, np.ndarray, Dict]:
    """
    Nemenyi post-hoc test for pairwise comparison after Friedman.
    
    Parameters:
        mean_ranks: Array of mean ranks per algorithm
        n_datasets: Number of datasets/problems
        n_algorithms: Number of algorithms
        alpha: Significance level
    
    Returns:
        critical_difference: CD value
        diff_matrix: Pairwise rank differences
        significant: Boolean matrix (True if significantly different)
        details: Dict with q_alpha, CD, etc.
    
    Two algorithms are significantly different if their rank difference
    exceeds the critical difference.
    """
    k = n_algorithms
    N = n_datasets
    
    # Critical value q_alpha for Studentized range distribution
    # Approximate values for alpha = 0.05 and 0.10
    q_alpha_table = {
        2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850,
        7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164
    }
    
    if k in q_alpha_table:
        q_alpha = q_alpha_table[k]
    else:
        # Approximation for k > 10
        q_alpha = 2.0 + 0.3 * np.log(k)
    
    # Critical Difference
    CD = q_alpha * np.sqrt(k * (k + 1) / (6.0 * N))
    
    # Pairwise differences
    diff_matrix = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            diff_matrix[i, j] = abs(mean_ranks[i] - mean_ranks[j])
    
    significant = diff_matrix > CD
    
    details = {
        "q_alpha": q_alpha,
        "critical_difference": CD,
        "alpha": alpha,
        "n_datasets": N,
        "n_algorithms": k,
    }
    
    return CD, diff_matrix, significant, details


# ── Cliff's Delta (Effect Size) ──────────────────────────────────────────────

def cliffs_delta(
    x: np.ndarray,
    y: np.ndarray,
) -> Tuple[float, str]:
    """
    Cliff's delta effect size for two independent samples.
    
    Parameters:
        x: Sample 1 (e.g., algorithm A results across datasets)
        y: Sample 2 (e.g., algorithm B results across datasets)
    
    Returns:
        delta: Cliff's delta (-1 to 1)
        magnitude: "negligible", "small", "medium", "large"
    
    Interpretation:
        delta > 0: x tends to be larger than y
        delta < 0: x tends to be smaller than y (x is better if minimizing)
        |delta| > 0.147: small effect
        |delta| > 0.33: medium effect
        |delta| > 0.474: large effect
    """
    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()
    
    n_x, n_y = len(x), len(y)
    
    # Count comparisons
    greater = 0  # x > y
    less = 0     # x < y
    
    for xi in x:
        greater += np.sum(xi > y)
        less += np.sum(xi < y)
    
    delta = (greater - less) / (n_x * n_y)
    
    # Magnitude interpretation (Romano et al., 2006 thresholds)
    abs_delta = abs(delta)
    if abs_delta < 0.147:
        magnitude = "negligible"
    elif abs_delta < 0.33:
        magnitude = "small"
    elif abs_delta < 0.474:
        magnitude = "medium"
    else:
        magnitude = "large"
    
    return delta, magnitude


def pairwise_cliffs_delta(
    results: np.ndarray,
    algorithm_names: Optional[List[str]] = None,
) -> Dict[str, Dict[str, Tuple[float, str]]]:
    """
    Compute Cliff's delta for all algorithm pairs.
    
    Parameters:
        results: (n_datasets, n_algorithms) array
        algorithm_names: List of algorithm names
    
    Returns:
        Nested dict: cliffs_delta[algo_i][algo_j] = (delta, magnitude)
    """
    n_datasets, n_algorithms = results.shape
    
    if algorithm_names is None:
        algorithm_names = [f"Algo{i+1}" for i in range(n_algorithms)]
    
    cliffs = {}
    for i in range(n_algorithms):
        cliffs[algorithm_names[i]] = {}
        for j in range(n_algorithms):
            if i != j:
                delta, magnitude = cliffs_delta(results[:, i], results[:, j])
                cliffs[algorithm_names[i]][algorithm_names[j]] = (delta, magnitude)
    
    return cliffs


# ── Wilcoxon Signed-Rank Test ────────────────────────────────────────────────

def wilcoxon_pairwise(
    results: np.ndarray,
    algorithm_names: Optional[List[str]] = None,
    alpha: float = 0.05,
    correction: str = "holm",
) -> Dict[str, Dict[str, Dict]]:
    """
    Pairwise Wilcoxon signed-rank test with p-value correction.
    
    Parameters:
        results: (n_datasets, n_algorithms) array
        algorithm_names: List of algorithm names
        alpha: Significance level
        correction: "holm", "bonferroni", or "none"
    
    Returns:
        Nested dict with statistic, p_value, reject for each pair
    """
    n_datasets, n_algorithms = results.shape
    
    if algorithm_names is None:
        algorithm_names = [f"Algo{i+1}" for i in range(n_algorithms)]
    
    # Collect all p-values for correction
    p_values = []
    pairs = []
    
    for i in range(n_algorithms):
        for j in range(i + 1, n_algorithms):
            try:
                statistic, p_value = stats.wilcoxon(results[:, i], results[:, j])
                p_values.append(p_value)
                pairs.append((i, j, statistic))
            except ValueError:
                # All values identical
                p_values.append(1.0)
                pairs.append((i, j, 0.0))
    
    # Apply correction
    p_values = np.array(p_values)
    if correction == "holm":
        corrected_p = _holm_bonferroni(p_values)
    elif correction == "bonferroni":
        corrected_p = np.minimum(p_values * len(p_values), 1.0)
    else:
        corrected_p = p_values
    
    # Build result dict
    results_dict = {}
    for idx, (i, j, stat) in enumerate(pairs):
        name_i, name_j = algorithm_names[i], algorithm_names[j]
        
        if name_i not in results_dict:
            results_dict[name_i] = {}
        if name_j not in results_dict:
            results_dict[name_j] = {}
        
        entry = {
            "statistic": stat,
            "p_value": p_values[idx],
            "p_value_corrected": corrected_p[idx],
            "reject": corrected_p[idx] < alpha,
            "alpha": alpha,
        }
        
        results_dict[name_i][name_j] = entry
        # Symmetric entry (negate statistic for reverse)
        entry_rev = entry.copy()
        entry_rev["statistic"] = -stat
        results_dict[name_j][name_i] = entry_rev
    
    return results_dict


def _holm_bonferroni(p_values: np.ndarray) -> np.ndarray:
    """Holm-Bonferroni step-down correction."""
    m = len(p_values)
    sorted_idx = np.argsort(p_values)
    sorted_p = p_values[sorted_idx]
    
    corrected = np.zeros(m)
    for i, p in enumerate(sorted_p):
        corrected[sorted_idx[i]] = min(p * (m - i), 1.0)
    
    # Ensure monotonicity
    for i in range(1, m):
        idx = sorted_idx[i]
        prev_idx = sorted_idx[i - 1]
        corrected[idx] = max(corrected[idx], corrected[prev_idx])
    
    return corrected


# ── Comprehensive Benchmark Analysis ─────────────────────────────────────────

def run_benchmark_analysis(
    results_dict: Dict[str, Dict[str, np.ndarray]],
    metric: str = "OF",
    alpha: float = 0.05,
    verbose: bool = True,
) -> Dict:
    """
    Run full statistical analysis on benchmark results.
    
    Parameters:
        results_dict: {system_name: {method_name: [run1, run2, ...]}}
        metric: Metric to analyze ("OF", "elapsed", etc.)
        alpha: Significance level
        verbose: Print results
    
    Returns:
        analysis: Dict with all test results
    """
    # Build results matrix: (n_systems, n_methods)
    systems = sorted(results_dict.keys())
    methods = None
    
    # Find common methods across all systems
    for sys_name, sys_results in results_dict.items():
        if methods is None:
            methods = sorted(sys_results.keys())
        else:
            methods = [m for m in methods if m in sys_results]
    
    n_systems = len(systems)
    n_methods = len(methods)
    
    results_matrix = np.zeros((n_systems, n_methods))
    for i, sys_name in enumerate(systems):
        for j, method in enumerate(methods):
            values = results_dict[sys_name][method]
            if isinstance(values, list):
                results_matrix[i, j] = np.mean(values)
            else:
                results_matrix[i, j] = values
    
    if verbose:
        print(f"\\n{'='*70}")
        print(f"  BENCHMARK STATISTICAL ANALYSIS")
        print(f"  Metric: {metric}  |  Alpha: {alpha}")
        print(f"  Systems: {n_systems}  |  Methods: {n_methods}")
        print(f"{'='*70}")
        
        print(f"\\n  Results Matrix (mean {metric}):")
        print(f"  {'System':<12}", end="")
        for m in methods:
            print(f"{m:>12}", end="")
        print()
        for i, sys_name in enumerate(systems):
            print(f"  {sys_name:<12}", end="")
            for j in range(n_methods):
                print(f"{results_matrix[i, j]:>12.4f}", end="")
            print()
    
    # 1. Friedman Test
    stat, p_val, reject, friedman_details = friedman_test(results_matrix, alpha)
    
    if verbose:
        print(f"\\n{'─'*70}")
        print(f"  1. FRIEDMAN TEST")
        print(f"{'─'*70}")
        print(f"  Statistic: {stat:.4f}")
        print(f"  p-value:   {p_val:.6f}")
        print(f"  Result:    {'REJECT H0' if reject else 'FAIL TO REJECT H0'}")
        print(f"  (H0: All algorithms perform equivalently)")
        print(f"\\n  Mean Ranks:")
        for j, method in enumerate(methods):
            print(f"    {method:<15} {friedman_details['mean_ranks'][j]:.3f}")
    
    # 2. Nemenyi Post-Hoc
    CD, diff_matrix, significant, nemenyi_details = nemenyi_test(
        friedman_details["mean_ranks"], n_systems, n_methods, alpha
    )
    
    if verbose:
        print(f"\\n{'─'*70}")
        print(f"  2. NEMENYI POST-HOC TEST")
        print(f"{'─'*70}")
        print(f"  Critical Difference (CD): {CD:.4f}")
        print(f"  q_alpha: {nemenyi_details['q_alpha']:.4f}")
        print(f"\\n  Pairwise Rank Differences:")
        print(f"  {'':15}", end="")
        for m in methods:
            print(f"{m:>10}", end="")
        print()
        for i, mi in enumerate(methods):
            print(f"  {mi:<15}", end="")
            for j, mj in enumerate(methods):
                if i == j:
                    print(f"{'—':>10}", end="")
                elif significant[i, j]:
                    print(f"{diff_matrix[i, j]:>9.3f}*", end="")
                else:
                    print(f"{diff_matrix[i, j]:>10.3f}", end="")
            print()
        print(f"  (* = significant difference, p < {alpha})")
    
    # 3. Cliff's Delta (Effect Size)
    cliffs = pairwise_cliffs_delta(results_matrix, methods)
    
    if verbose:
        print(f"\\n{'─'*70}")
        print(f"  3. CLIFF'S DELTA (Effect Size)")
        print(f"{'─'*70}")
        print(f"  {'Method A':<12} {'Method B':<12} {'Delta':>10} {'Magnitude':>12}")
        for i, mi in enumerate(methods):
            for j, mj in enumerate(methods):
                if i < j and mi in cliffs and mj in cliffs[mi]:
                    delta, magnitude = cliffs[mi][mj]
                    print(f"  {mi:<12} {mj:<12} {delta:>+10.4f} {magnitude:>12}")
    
    # 4. Wilcoxon Signed-Rank (with Holm correction)
    wilcoxon = wilcoxon_pairwise(results_matrix, methods, alpha, correction="holm")
    
    if verbose:
        print(f"\\n{'─'*70}")
        print(f"  4. WILCOXON SIGNED-RANK TEST (Holm-corrected)")
        print(f"{'─'*70}")
        print(f"  {'Method A':<12} {'Method B':<12} {'p-value':>12} {'Significant?':>14}")
        for i, mi in enumerate(methods):
            for j, mj in enumerate(methods):
                if i < j and mi in wilcoxon and mj in wilcoxon[mi]:
                    p = wilcoxon[mi][mj]["p_value_corrected"]
                    sig = "YES" if wilcoxon[mi][mj]["reject"] else "NO"
                    print(f"  {mi:<12} {mj:<12} {p:>12.6f} {sig:>14}")
    
    # 5. Summary ranking
    sorted_methods = [methods[i] for i in np.argsort(friedman_details["mean_ranks"])]
    
    if verbose:
        print(f"\\n{'='*70}")
        print(f"  SUMMARY RANKING (Best to Worst)")
        print(f"{'='*70}")
        for rank, method in enumerate(sorted_methods, 1):
            idx = methods.index(method)
            mr = friedman_details["mean_ranks"][idx]
            mean_perf = np.mean(results_matrix[:, idx])
            print(f"  {rank}. {method:<15}  Mean Rank: {mr:.3f}  |  Avg {metric}: {mean_perf:.4f}")
        print(f"{'='*70}\\n")
    
    return {
        "friedman": {
            "statistic": stat,
            "p_value": p_val,
            "reject_null": reject,
            "details": friedman_details,
        },
        "nemenyi": {
            "critical_difference": CD,
            "diff_matrix": diff_matrix,
            "significant": significant,
            "details": nemenyi_details,
        },
        "cliffs_delta": cliffs,
        "wilcoxon": wilcoxon,
        "ranking": {
            "methods": sorted_methods,
            "mean_ranks": friedman_details["mean_ranks"],
        },
        "results_matrix": results_matrix,
        "systems": systems,
        "methods": methods,
        "alpha": alpha,
    }


# ── Helper: Aggregate results across multiple runs ───────────────────────────

def aggregate_results(
    raw_results: Dict[str, List[Dict]],
    metric: str = "OF",
    aggregation: str = "mean",
) -> Dict[str, Dict[str, float]]:
    """
    Aggregate results from multiple independent runs.
    
    Parameters:
        raw_results: {system_name: [run1_dict, run2_dict, ...]}
                     where each run_dict is {method_name: result_dict}
        metric: Metric to extract ("OF", "elapsed", etc.)
        aggregation: "mean", "median", "best", "worst"
    
    Returns:
        {system_name: {method_name: aggregated_value}}
    """
    aggregated = {}
    
    for sys_name, runs in raw_results.items():
        aggregated[sys_name] = {}
        
        # Find all methods
        methods = set()
        for run in runs:
            methods.update(run.keys())
        
        for method in methods:
            values = []
            for run in runs:
                if method in run and metric in run[method]:
                    val = run[method][metric]
                    if isinstance(val, (int, float)):
                        values.append(val)
            
            if not values:
                continue
            
            if aggregation == "mean":
                aggregated[sys_name][method] = np.mean(values)
            elif aggregation == "median":
                aggregated[sys_name][method] = np.median(values)
            elif aggregation == "best":
                aggregated[sys_name][method] = np.min(values)
            elif aggregation == "worst":
                aggregated[sys_name][method] = np.max(values)
            else:
                aggregated[sys_name][method] = np.mean(values)
    
    return aggregated