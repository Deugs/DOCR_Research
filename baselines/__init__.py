"""
baselines
─────────
Metaheuristic optimization baselines for DOCR coordination.

Provides unified interface to GA-SQP and 4 nature-inspired metaheuristics
(HWOA, HHO, JAYA, EEO) with optional SQP local refinement.

All functions accept the same NetworkData + fault_type + topology_state API
and return standardized result dictionaries for fair comparison.

Example:
    from baselines.metaheuristics import run_ga_sqp, run_metaheuristic

    # GA-SQP hybrid
    result = run_ga_sqp(net, ga_pop=100, ga_gen=200, seed=42, verbose=True)

    # Single metaheuristic
    result = run_metaheuristic("HWOA", net, pop_size=30, max_iter=500, seed=42)

    # All metaheuristics at once
    from baselines.metaheuristics import run_all_metaheuristics
    results = run_all_metaheuristics(net, pop_size=30, max_iter=500, seed=42)
"""

from .ga_sqp import run_ga_sqp
from .metaheuristics import (
    run_hwoa,
    run_hho,
    run_jaya,
    run_eeo,
    run_metaheuristic,
    run_all_metaheuristics,
    METAHEURISTICS,
)

__all__ = [
    "run_ga_sqp",
    "run_hwoa",
    "run_hho",
    "run_jaya",
    "run_eeo",
    "run_metaheuristic",
    "run_all_metaheuristics",
    "METAHEURISTICS",
]