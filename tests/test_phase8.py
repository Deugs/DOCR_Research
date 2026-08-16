"""
tests/test_phase8.py
────────────────────
Phase 8: Tests for thesis validation, reproducibility, and statistical framework.

Tests:
1. Novelty claim initialization
2. Claim validation (C1: MAPPO-Comm performance)
3. Claim validation (C2: Safe RL zero violations)
4. Claim validation (C3: EWC forgetting)
5. Claim validation (C4: Curriculum efficiency)
6. Claim validation (C5: Scalability)
7. Validation report generation
8. LaTeX table generation
9. Reproducibility checklist
10. Docker file generation
11. GitHub Actions CI config
12. Statistical test battery
13. Effect size computation
14. Full validation pipeline
"""

import numpy as np
import sys, os
import tempfile
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from evaluation.thesis_validation import (
    ThesisValidator, NoveltyClaim, ValidationResult,
    ReproducibilityChecklist, generate_dockerfile, generate_github_actions,
)
from evaluation.stat_analysis import friedman_test, cliffs_delta


def test_claim_initialization():
    """Test 1: Novelty claim initialization."""
    print("  Test 1: Claim initialization...")

    claim = NoveltyClaim(
        id="C1",
        title="Test Claim",
        description="A test claim",
        hypothesis_h0="H0",
        hypothesis_h1="H1",
        metrics=["OF"],
        pass_criteria={"OF": (0.05, "<")},
        required_methods=["GA-SQP"],
        systems=["8bus"],
    )

    assert claim.id == "C1"
    assert "OF" in claim.metrics
    assert claim.pass_criteria["OF"] == (0.05, "<")

    print("    ✓ PASS: Novelty claim initialized correctly")


def test_claim_c1_validation():
    """Test 2: Validate C1 (MAPPO-Comm outperforms baselines)."""
    print("  Test 2: C1 validation...")

    validator = ThesisValidator()

    # Create mock results with ALL required methods for C1
    validator.results = {
        "results": [
            {"method": "GA-SQP", "system": "8bus", "best_of": 5.2, "cti_satisfied": True},
            {"method": "Single-PPO", "system": "8bus", "best_of": 5.0, "cti_satisfied": True},
            {"method": "IPPO", "system": "8bus", "best_of": 5.1, "cti_satisfied": True},
            {"method": "MAPPO", "system": "8bus", "best_of": 4.8, "cti_satisfied": True},
            {"method": "MAPPO-Comm", "system": "8bus", "best_of": 4.5, "cti_satisfied": True},
            {"method": "GA-SQP", "system": "30bus", "best_of": 12.1, "cti_satisfied": True},
            {"method": "Single-PPO", "system": "30bus", "best_of": 12.0, "cti_satisfied": True},
            {"method": "IPPO", "system": "30bus", "best_of": 11.9, "cti_satisfied": True},
            {"method": "MAPPO", "system": "30bus", "best_of": 11.5, "cti_satisfied": True},
            {"method": "MAPPO-Comm", "system": "30bus", "best_of": 11.0, "cti_satisfied": True},
        ]
    }

    claim = ThesisValidator.CLAIMS[0]  # C1
    result = validator.validate_claim(claim)

    assert result.claim_id == "C1"
    assert "MAPPO-Comm_OF_mean" in result.details
    assert "GA-SQP_OF_mean" in result.details
    assert result.details["MAPPO-Comm_OF_mean"] < result.details["GA-SQP_OF_mean"]

    print(f"    ✓ PASS: C1 validated (MAPPO-Comm OF={result.details['MAPPO-Comm_OF_mean']:.2f} < GA-SQP OF={result.details['GA-SQP_OF_mean']:.2f})")


def test_claim_c2_validation():
    """Test 3: Validate C2 (Safe RL zero violations)."""
    print("  Test 3: C2 validation...")

    validator = ThesisValidator()
    validator.results = {
        "results": [
            {"method": "MAPPO", "system": "8bus", "cti_violations": 2, "cti_satisfied": False},
            {"method": "MAPPO-Safe", "system": "8bus", "cti_violations": 0, "cti_satisfied": True},
            {"method": "MAPPO", "system": "30bus", "cti_violations": 3, "cti_satisfied": False},
            {"method": "MAPPO-Safe", "system": "30bus", "cti_violations": 0, "cti_satisfied": True},
            {"method": "MAPPO", "system": "39bus", "cti_violations": 4, "cti_satisfied": False},
            {"method": "MAPPO-Safe", "system": "39bus", "cti_violations": 0, "cti_satisfied": True},
        ]
    }

    claim = ThesisValidator.CLAIMS[1]  # C2
    result = validator.validate_claim(claim)

    assert result.claim_id == "C2"
    assert "MAPPO-Safe_cti_satisfied_rate" in result.details
    assert "MAPPO_cti_satisfied_rate" in result.details
    assert result.details["MAPPO-Safe_cti_satisfied_rate"] > result.details["MAPPO_cti_satisfied_rate"]

    print(f"    ✓ PASS: C2 validated (Safe RL satisfaction={result.details['MAPPO-Safe_cti_satisfied_rate']:.2f})")


def test_claim_c3_validation():
    """Test 4: Validate C3 (EWC forgetting)."""
    print("  Test 4: C3 validation...")

    validator = ThesisValidator()
    validator.results = {
        "results": [
            {"method": "MAPPO-Safe", "system": "8bus", "best_of": 5.1},
            {"method": "MAPPO-Safe-EWC", "system": "8bus", "best_of": 5.0},
            {"method": "MAPPO-Safe", "system": "30bus", "best_of": 12.0},
            {"method": "MAPPO-Safe-EWC", "system": "30bus", "best_of": 12.1},
        ]
    }

    claim = ThesisValidator.CLAIMS[2]  # C3
    result = validator.validate_claim(claim)

    assert result.claim_id == "C3"
    assert "forgetting_measure" in result.details

    print(f"    ✓ PASS: C3 validated (forgetting={result.details['forgetting_measure']:.4f})")


def test_claim_c5_validation():
    """Test 5: Validate C5 (Scalability)."""
    print("  Test 5: C5 validation...")

    validator = ThesisValidator()
    validator.results = {
        "results": [
            {"method": "MAPPO-Comm", "system": "39bus", "inference_latency_ms": 50.0, "comm_bandwidth": 0.5},
            {"method": "MAPPO-Safe-EWC", "system": "39bus", "inference_latency_ms": 80.0, "comm_bandwidth": 0.8},
        ]
    }

    claim = ThesisValidator.CLAIMS[4]  # C5
    result = validator.validate_claim(claim)

    assert result.claim_id == "C5"
    assert result.details["inference_latency_ms"] < 100.0
    assert result.details["bandwidth_kbps"] < 1.0

    print(f"    ✓ PASS: C5 validated (latency={result.details['inference_latency_ms']:.1f}ms, BW={result.details['bandwidth_kbps']:.2f}kbps)")


def test_validation_report():
    """Test 6: Generate validation report."""
    print("  Test 6: Validation report...")

    validator = ThesisValidator()
    validator.results = {
        "results": [
            {"method": "GA-SQP", "system": "8bus", "best_of": 5.2},
            {"method": "MAPPO-Comm", "system": "8bus", "best_of": 4.9},
        ]
    }

    validator.validate_all_claims()
    report = validator.generate_validation_report()

    assert "PHASE 8: THESIS VALIDATION REPORT" in report
    assert "C1" in report
    assert "C5" in report

    print("    ✓ PASS: Validation report generated")


def test_latex_table_generation(tmp_path):
    """Test 7: LaTeX table generation."""
    print("  Test 7: LaTeX table generation...")

    validator = ThesisValidator()
    validator.results = {"results": []}
    validator.validate_all_claims()

    output_path = tmp_path / "test_thesis_table.tex"
    latex = validator.generate_latex_tables(str(output_path))

    assert "\\begin{table}" in latex
    assert "\\end{table}" in latex
    assert "C1" in latex

    assert output_path.exists()

    print("    ✓ PASS: LaTeX table generated")


def test_reproducibility_checklist(tmp_path):
    """Test 8: Reproducibility checklist."""
    print("  Test 8: Reproducibility checklist...")

    # Build a fake repo tree with a known, deliberate mix of present/absent
    # markers so every check's actual boolean VALUE (not just its presence
    # as a key) can be verified against what run_check() should compute.
    (tmp_path / ".git").mkdir()                       # Code availability     -> True
    (tmp_path / "requirements.txt").write_text("numpy\n")  # Environment spec -> True
    (tmp_path / "configs").mkdir()                     # Hyperparameters       -> True
    (tmp_path / "evaluation").mkdir()
    (tmp_path / "evaluation" / "stat_analysis.py").write_text("# stub\n")  # Statistical tests -> True
    # Deliberately NOT created: "data", "Dockerfile", "outputs",
    # "evaluation/evaluation.py" -> their checks must come back False.

    checklist = ReproducibilityChecklist(repo_dir=str(tmp_path))
    checks = checklist.run_all_checks()

    expected = {
        "Code availability": True,
        "Data availability": False,
        "Environment specification": True,
        "Docker container": False,
        "Random seeds": True,             # always documented in code
        "Hyperparameters": True,
        "Hardware specification": True,   # always documented in thesis
        "Runtime logs": False,
        "Statistical tests": True,
        "Plot generation": False,
    }

    assert len(checks) == 10
    assert checks == expected, f"Checklist values diverged from expected: {checks}"

    expected_passed = sum(expected.values())
    assert expected_passed == 6
    assert sum(checks.values()) == expected_passed

    report = checklist.generate_report()
    assert "REPRODUCIBILITY CHECKLIST" in report
    assert f"Score: {expected_passed}/10 (60%)" in report
    # Every individual check result must be reflected in the report with the
    # correct pass/fail marker, not just the aggregate score.
    for name, passed_flag in expected.items():
        marker = "✓" if passed_flag else "✗"
        assert f"{marker} {name}:" in report, f"Report missing '{marker} {name}:'"

    print(f"    ✓ PASS: Reproducibility checklist ({sum(checks.values())}/{len(checks)} checks)")


def test_dockerfile_generation(tmp_path):
    """Test 9: Dockerfile generation."""
    print("  Test 9: Dockerfile generation...")

    output_path = tmp_path / "test_dockerfile"
    generate_dockerfile(str(output_path))

    assert output_path.exists()

    content = output_path.read_text()

    assert "FROM python:3.10-slim" in content
    assert "requirements.txt" in content

    print("    ✓ PASS: Dockerfile generated")


def test_github_actions_generation(tmp_path):
    """Test 10: GitHub Actions CI config."""
    print("  Test 10: GitHub Actions CI config...")

    output_path = tmp_path / "test_ci.yml"
    generate_github_actions(str(output_path))

    assert output_path.exists()

    content = output_path.read_text()

    assert "name: DOCR-MARL CI" in content
    assert "test_phase1.py" in content
    assert "test_phase7.py" in content

    print("    ✓ PASS: GitHub Actions CI config generated")


def test_statistical_battery():
    """Test 11: Statistical test battery."""
    print("  Test 11: Statistical test battery...")

    # Create test data
    np.random.seed(42)
    results = np.random.randn(10, 4)  # 10 datasets, 4 methods

    # Friedman test
    stat, p, reject, details = friedman_test(results, alpha=0.05)

    assert p > 0  # Valid p-value
    assert "mean_ranks" in details
    assert len(details["mean_ranks"]) == 4

    # Cliff's delta
    delta, magnitude = cliffs_delta(results[:, 0], results[:, 1])
    assert -1 <= delta <= 1
    assert magnitude in ["negligible", "small", "medium", "large"]

    print(f"    ✓ PASS: Statistical battery (Friedman p={p:.4f}, Cliff's δ={delta:.4f})")


def test_full_validation_pipeline():
    """Test 12: Full validation pipeline."""
    print("  Test 12: Full validation pipeline...")

    validator = ThesisValidator()

    # Comprehensive mock results with all required methods and systems
    validator.results = {
        "results": [
            # C1: Performance comparison - all required methods
            {"method": "GA-SQP", "system": "8bus", "best_of": 5.2, "cti_satisfied": True, "cti_violations": 0},
            {"method": "Single-PPO", "system": "8bus", "best_of": 5.0, "cti_satisfied": True, "cti_violations": 0},
            {"method": "IPPO", "system": "8bus", "best_of": 5.1, "cti_satisfied": True, "cti_violations": 0},
            {"method": "MAPPO", "system": "8bus", "best_of": 4.8, "cti_satisfied": True, "cti_violations": 0},
            {"method": "MAPPO-Comm", "system": "8bus", "best_of": 4.5, "cti_satisfied": True, "cti_violations": 0},
            {"method": "GA-SQP", "system": "30bus", "best_of": 12.1, "cti_satisfied": True, "cti_violations": 0},
            {"method": "Single-PPO", "system": "30bus", "best_of": 12.0, "cti_satisfied": True, "cti_violations": 0},
            {"method": "IPPO", "system": "30bus", "best_of": 11.9, "cti_satisfied": True, "cti_violations": 0},
            {"method": "MAPPO", "system": "30bus", "best_of": 11.5, "cti_satisfied": True, "cti_violations": 0},
            {"method": "MAPPO-Comm", "system": "30bus", "best_of": 11.0, "cti_satisfied": True, "cti_violations": 0},

            # C2: Safe RL - all required systems
            {"method": "MAPPO", "system": "8bus", "best_of": 5.0, "cti_satisfied": False, "cti_violations": 3},
            {"method": "MAPPO-Safe", "system": "8bus", "best_of": 4.9, "cti_satisfied": True, "cti_violations": 0},
            {"method": "MAPPO", "system": "30bus", "best_of": 12.0, "cti_satisfied": False, "cti_violations": 3},
            {"method": "MAPPO-Safe", "system": "30bus", "best_of": 11.8, "cti_satisfied": True, "cti_violations": 0},
            {"method": "MAPPO", "system": "39bus", "best_of": 15.0, "cti_satisfied": False, "cti_violations": 4},
            {"method": "MAPPO-Safe", "system": "39bus", "best_of": 14.8, "cti_satisfied": True, "cti_violations": 0},

            # C3: EWC
            {"method": "MAPPO-Safe", "system": "8bus", "best_of": 5.1},
            {"method": "MAPPO-Safe-EWC", "system": "8bus", "best_of": 5.0},
            {"method": "MAPPO-Safe", "system": "30bus", "best_of": 12.0},
            {"method": "MAPPO-Safe-EWC", "system": "30bus", "best_of": 12.1},

            # C5: Scalability
            {"method": "MAPPO-Comm", "system": "39bus", "inference_latency_ms": 50.0, "comm_bandwidth": 0.5},
            {"method": "MAPPO-Safe-EWC", "system": "39bus", "inference_latency_ms": 80.0, "comm_bandwidth": 0.8},
        ]
    }

    results = validator.validate_all_claims()

    assert len(results) == 5
    assert all(isinstance(r, ValidationResult) for r in results)

    report = validator.generate_validation_report()
    assert "PHASE 8" in report

    print(f"    ✓ PASS: Full pipeline validated {len(results)} claims")


def run_all_tests():
    """Run all Phase 8 tests."""
    print("\n" + "=" * 70)
    print("  PHASE 8 TEST SUITE: Thesis Validation & Reproducibility")
    print("=" * 70 + "\n")

    tests = [
        test_claim_initialization,
        test_claim_c1_validation,
        test_claim_c2_validation,
        test_claim_c3_validation,
        test_claim_c5_validation,
        test_validation_report,
        test_latex_table_generation,
        test_reproducibility_checklist,
        test_dockerfile_generation,
        test_github_actions_generation,
        test_statistical_battery,
        test_full_validation_pipeline,
    ]

    # These tests take a `tmp_path`-style fixture when run under pytest.
    # When this module is executed directly (not via pytest), give each of
    # them its own throwaway temp directory instead.
    tests_needing_tmp_path = {
        test_latex_table_generation,
        test_reproducibility_checklist,
        test_dockerfile_generation,
        test_github_actions_generation,
    }

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test in tests_needing_tmp_path:
                with tempfile.TemporaryDirectory() as tmpdir:
                    test(Path(tmpdir))
            else:
                test()
            passed += 1
        except Exception as e:
            print(f"    ✗ FAIL: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 70)
    print(f"  RESULTS: {passed} passed, {failed} failed out of {len(tests)} tests")
    print("=" * 70 + "\n")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)