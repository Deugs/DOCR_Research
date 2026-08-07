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

Tests:
1. PHASE_FIGURES[7] matches the documented figure set (43-50, 58)
2. Each real fig_NN_* function in run_post_processing.py for this phase
   runs end-to-end and produces a file
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


def test_post_processing_functions_exist_and_run():
    """Test 2: every fig_NN_* function for Phase 7 runs and saves a file."""
    print("  Test 2: Phase 7 post-processing functions...")
    pp = _load_post_processing()
    tmp_dir = tempfile.mkdtemp()
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
        shutil.rmtree(tmp_dir, ignore_errors=True)
    print(f"    OK ({len(PHASE7_POST_PROCESSING_FUNCS)} functions)")


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
    test_cli_adapters_flag_synthetic_data()
    print("All Phase 7 tests passed.")
