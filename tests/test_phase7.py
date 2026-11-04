"""
tests/test_phase7.py
─────────────────────
Phase 7: Tests for Protection Engineering & Interpretability
(Figures 43-50, 58).

This file previously was a byte-for-byte copy of tests/test_phase8.py
(same 372 lines, same thesis-validation test bodies) — meaning "Phase 7"
had zero real test coverage despite PHASE6_7_8_COMPLETE.md claiming full
coverage, and test_phase8.py's actual subject (evaluation/thesis_validation.py)
was being exercised twice under two different filenames.

Policy update (2026-11-04): run_post_processing.py no longer fabricates
synthetic/hardcoded data when a cache is missing — every fig_NN_*()
function now raises MissingRealDataError instead (see the data provenance
policy note at the top of run_post_processing.py). Test 2 below exercises
each real fig_NN_* function against minimal but real-schema-shaped fixtures
(this repo has no real core_J_protection/core_K_seasonal/core_eval_fig*
caches for most of these figures) rather than relying on fallback data to
produce a figure unconditionally.

Tests:
1. PHASE_FIGURES[7] matches the documented figure set (43-50, 58)
2. Each real fig_NN_* function in run_post_processing.py for this phase
   runs end-to-end against a real-schema-shaped fixture and produces a file
3. Each of the standalone "CLI Adapter" scripts for this phase (which plot
   hand-typed synthetic data — see EXPERIMENT_GUIDE.md section 2.2.1) still
   emits its synthetic-data warning, locking in the fix applied alongside
   this test file so a future edit can't silently drop it
"""

import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import run_thesis_experiments as rte

REPO_ROOT = Path(os.path.dirname(os.path.dirname(__file__)))

PHASE7_POST_PROCESSING_FUNCS = [
    "fig_43_coordination_diagram",
    "fig_44_fault_contribution",
    "fig_45_daily_profile",
    "fig_46_relay_curves",
    "fig_47_sympathetic_tripping",
    "fig_48_shap",
    "fig_49_decision_boundary",
    "fig_50_critical_pairs",
    "fig_58_seasonal",
]

# The standalone synthetic-data CLI adapters covering the same figures
# (see EXPERIMENT_GUIDE.md section 2.2.1 for why these are separate from
# the run_post_processing.py functions above).
PHASE7_CLI_ADAPTERS = [
    "network/utils/coordination_diagram.py",
    "network/utils/fault_contribution.py",
    "data/experiments/daily_adaptive_response.py",
    "network/utils/relay_curves.py",
    "network/utils/sympathetic_trip_cases.py",
    "network/utils/critical_pairs.py",
    "evaluation/experiments/shap_analysis.py",
    "evaluation/experiments/decision_boundary.py",
    "evaluation/experiments/seasonal_generalization.py",
]


def _load_post_processing():
    pp_path = str(REPO_ROOT / "run_post_processing.py")
    spec = importlib.util.spec_from_file_location("post_processing_test7", pp_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_phase7_figure_set_matches_guide():
    """Test 1: PHASE_FIGURES[7] is [43-50, 58], matching EXPERIMENT_GUIDE.md."""
    print("  Test 1: Phase 7 figure set...")
    assert rte.PHASE_FIGURES[7] == [43, 44, 45, 46, 47, 48, 49, 50, 58]
    print("    OK")


def _fake_core_J_protection_cache():
    """Minimal but real-schema-shaped core_J_protection fixture covering
    figs 43 (coordination), 44 (fault_contribution), 45 (daily_profile),
    46 (relay_curves) — field names match run_post_processing.py's own
    MissingRealDataError messages for each figure."""
    import numpy as np
    n = 24
    return {
        "coordination": {
            "distances": list(np.linspace(0, 10, 10)),
            "primary": list(0.1 + 0.05 * np.linspace(0, 10, 10)),
            "backup": list(0.3 + 0.08 * np.linspace(0, 10, 10)),
        },
        "fault_contribution": {"PV": 35.0, "Grid": 45.0, "Synchronous": 20.0},
        "daily_profile": {
            "irradiance": list(np.maximum(0, np.sin(np.pi * np.arange(n) / 12))),
            "of": list(np.full(n, 1.0)),
        },
        "relay_curves": {
            "relay_0": {
                "TDS": 0.5, "PSM": 2.0,
                "If_range_A": list(np.linspace(2.0, 40.0, 20)),
                "t_op_s": list(np.full(20, 0.5)),
            },
        },
    }


def _fake_core_K_seasonal_cache():
    """Minimal but real-schema-shaped core_K_seasonal fixture (fig 58)."""
    return {
        season: {"performance": 0.9, "pv_variability": 0.2, "of": 100.0, "cti_ok": True}
        for season in ("spring", "summer", "autumn", "winter")
    }


def _fake_bridge_cache(fig_num: int):
    """Minimal but real-schema-shaped core_eval_fig{NN}.json fixtures for
    figs 47 (sympathetic_tripping), 48 (shap), 49 (decision_boundary),
    50 (critical_pairs) — these are read via load_bridge_cache(fig_num),
    not load_cache(name)."""
    import numpy as np
    if fig_num == 47:
        return {
            "sympathetic": 5, "traditional": 20,
            "by_fault_type": {
                "1ph": {"traditional": 10, "marl": 2},
                "LL": {"traditional": 8, "marl": 1},
                "3ph": {"traditional": 5, "marl": 1},
            },
        }
    if fig_num == 48:
        # 12 real obs features (see fig_48_shap's own `features` list);
        # shape (n_explain, n_features, n_actions).
        return {"shap": np.random.default_rng(0).random((5, 12, 2)).tolist()}
    if fig_num == 49:
        rng = np.random.default_rng(0)
        return {
            "embeddings": rng.standard_normal((50, 2)).tolist(),
            "actions": rng.standard_normal((50, 2)).tolist(),
        }
    if fig_num == 50:
        return [
            {"primary": 1, "backup": 2, "distance_to_violation": 0.1},
            {"primary": 2, "backup": 3, "distance_to_violation": -0.05},
        ]
    return None


def test_post_processing_functions_exist_and_run():
    """Test 2: every fig_NN_* function for Phase 7 runs and saves a file,
    against real-schema-shaped fixtures (this repo has no real
    core_J_protection/core_K_seasonal/core_eval_fig* caches for most of
    these figures, and the functions now raise rather than fabricate data
    on a genuine miss)."""
    print("  Test 2: Phase 7 post-processing functions...")
    pp = _load_post_processing()
    tmp_dir = tempfile.mkdtemp()

    core_j = _fake_core_J_protection_cache()
    core_k = _fake_core_K_seasonal_cache()

    def fake_load_cache(name):
        if name == "core_J_protection":
            return core_j
        if name == "core_K_seasonal":
            return core_k
        return None

    orig_load_cache = pp.load_cache
    orig_load_bridge_cache = pp.load_bridge_cache
    pp.load_cache = fake_load_cache
    pp.load_bridge_cache = _fake_bridge_cache
    try:
        pp._config = None
        orig_get_figures_dir = pp.get_figures_dir
        pp.get_figures_dir = lambda: Path(tmp_dir)
        try:
            for func_name in PHASE7_POST_PROCESSING_FUNCS:
                assert hasattr(pp, func_name), f"missing function: {func_name}"
                before = set(Path(tmp_dir).rglob("*"))
                getattr(pp, func_name)()
                after = set(Path(tmp_dir).rglob("*"))
                assert after > before, f"{func_name} did not produce a new file"
        finally:
            pp.get_figures_dir = orig_get_figures_dir
    finally:
        pp.load_cache = orig_load_cache
        pp.load_bridge_cache = orig_load_bridge_cache
        shutil.rmtree(tmp_dir, ignore_errors=True)
    print(f"    OK ({len(PHASE7_POST_PROCESSING_FUNCS)} functions)")


def test_fig58_raises_on_missing_cache():
    """Test 2b: fig_58_seasonal() raises MissingRealDataError when
    core_K_seasonal is genuinely absent, instead of fabricating placeholder
    seasonal data (2026-11-04 policy change)."""
    print("  Test 2b: Fig 58 raises on missing cache...")
    pp = _load_post_processing()
    orig_load_cache = pp.load_cache
    pp.load_cache = lambda name: None
    try:
        raised = False
        try:
            pp.fig_58_seasonal()
        except pp.MissingRealDataError:
            raised = True
        assert raised, "expected MissingRealDataError when core_K_seasonal cache is missing"
    finally:
        pp.load_cache = orig_load_cache
    print("    OK")


def test_cli_adapters_flag_synthetic_data():
    """Test 3: each Phase 7 CLI adapter still emits its synthetic-data warning."""
    print("  Test 3: Phase 7 CLI adapter synthetic-data warnings...")
    tmp_dir = tempfile.mkdtemp()
    try:
        for rel_path in PHASE7_CLI_ADAPTERS:
            out_path = Path(tmp_dir) / (Path(rel_path).stem + ".pdf")
            result = subprocess.run(
                [sys.executable, rel_path, "--output", str(out_path)],
                cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=60,
            )
            assert result.returncode == 0, (
                f"{rel_path} failed:\nstdout={result.stdout}\nstderr={result.stderr}"
            )
            combined = result.stdout + result.stderr
            assert "SYNTHETIC DATA" in combined, (
                f"{rel_path} no longer emits the synthetic-data warning"
            )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    print(f"    OK ({len(PHASE7_CLI_ADAPTERS)} adapters)")


if __name__ == "__main__":
    test_phase7_figure_set_matches_guide()
    test_post_processing_functions_exist_and_run()
    test_fig58_raises_on_missing_cache()
    test_cli_adapters_flag_synthetic_data()
    print("All Phase 7 tests passed.")
