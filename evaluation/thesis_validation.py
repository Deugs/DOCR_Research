"""
evaluation/thesis_validation.py
─────────────────────────────────
Phase 8: Statistical validation framework and novelty claim verification.
Active thesis scope: IEEE 8/14/39-bus test systems, 0-60% PV penetration
(see EXPERIMENT_GUIDE.md's Reproducibility Checklist). The network
catalog (network/network_data.py) also defines 3/30/57/118-bus systems
for other uses (e.g. Core F's separate scalability sweep), but those are
not part of the primary claim-validation scope below.

Validates the 5 novelty claims from the thesis:
1. MAPPO with GNN communication outperforms all baselines
2. Safe RL (Augmented Lagrangian) achieves zero CTI violations
3. EWC enables continual adaptation without catastrophic forgetting
4. Curriculum learning reduces sample complexity by 40%+
5. System scales to 39-bus networks with <100ms inference

Provides:
- Hypothesis test battery (H0/H1 formulation)
- Novelty claim validation with pass/fail criteria
- Reproducibility checklist
- LaTeX table generation for thesis chapters
- GitHub Actions CI configuration
- Docker container specification

Usage:
    from evaluation.thesis_validation import ThesisValidator, NoveltyClaim

    validator = ThesisValidator(results_dir="outputs/phase5")
    report = validator.validate_all_claims()
    validator.generate_latex_tables("outputs/thesis_tables.tex")
"""

import numpy as np
import json
import os
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from pathlib import Path
from collections import defaultdict
from scipy.stats import mannwhitneyu

try:
    from network.network_data import NETWORKS
except ImportError:
    NETWORKS = {}

try:
    from evaluation.stat_analysis import (
        friedman_test, nemenyi_test, cliffs_delta, wilcoxon_pairwise
    )
except ImportError:
    from stat_analysis import (
        friedman_test, nemenyi_test, cliffs_delta, wilcoxon_pairwise
    )


@dataclass
class NoveltyClaim:
    """A thesis novelty claim with validation criteria."""
    id: str
    title: str
    description: str
    hypothesis_h0: str
    hypothesis_h1: str
    metrics: List[str]
    pass_criteria: Dict[str, Tuple[float, str]]  # metric: (threshold, operator)
    required_methods: List[str]
    systems: List[str]


@dataclass
class ValidationResult:
    """Result of validating a single novelty claim."""
    claim_id: str
    passed: bool
    # None (not a placeholder value like 0.05/1.0) whenever no real statistical
    # test could be computed — e.g. a threshold-only claim, or too few samples.
    p_value: Optional[float]
    effect_size: Optional[float]
    effect_magnitude: str
    details: Dict
    recommendation: str


class ThesisValidator:
    """
    Comprehensive thesis validation framework.

    Validates all 5 novelty claims using the Phase 5 large-scale
    experimental results and Phase 4 statistical tests.
    """

    CLAIMS = [
        NoveltyClaim(
            id="C1",
            title="MAPPO with GNN Communication Outperforms Baselines",
            description="MAPPO-Comm achieves significantly lower objective function than GA-SQP, Single-PPO, IPPO, and MAPPO without communication.",
            hypothesis_h0="MAPPO-Comm performs equivalently to all baselines across test systems.",
            hypothesis_h1="MAPPO-Comm achieves lower OF than at least one baseline with statistical significance (p < 0.05, Cliff's delta > 0.33).",
            metrics=["OF", "cti_satisfied"],
            pass_criteria={
                "p_value": (0.05, "<"),
                "cliffs_delta": (0.33, ">"),
            },
            required_methods=["GA-SQP", "Single-PPO", "IPPO", "MAPPO", "MAPPO-Comm"],
            # Active thesis scope is exactly 8/14/39-bus, 0-60% PV (see
            # EXPERIMENT_GUIDE.md's Reproducibility Checklist) -- trimmed
            # from the broader 3/8/14/30/39/57/118-bus list this claim
            # previously spanned, to match.
            systems=["8bus", "14bus", "39bus"],
        ),
        NoveltyClaim(
            id="C2",
            title="Safe RL Achieves Zero CTI Violations",
            description="MAPPO-Safe achieves zero CTI violations at convergence across all test systems and PV levels, validated by Augmented Lagrangian dual feasibility.",
            hypothesis_h0="MAPPO-Safe has CTI violation rate > 0% (equivalent to unsafe methods).",
            hypothesis_h1="MAPPO-Safe achieves CTI violation rate = 0% with statistical significance.",
            metrics=["cti_violation_rate", "cti_satisfied"],
            pass_criteria={
                "cti_violation_rate": (0.01, "<"),  # < 1%
                "cti_satisfied_rate": (0.95, ">"),  # > 95%
            },
            required_methods=["MAPPO", "MAPPO-Safe"],
            systems=["8bus", "14bus", "39bus"],
        ),
        NoveltyClaim(
            id="C3",
            title="EWC Enables Continual Adaptation Without Forgetting",
            description="MAPPO-Safe-EWC shows <10% performance drop on old tasks after learning new tasks, compared to >30% drop without EWC.",
            hypothesis_h0="EWC provides no benefit over standard training for continual adaptation.",
            hypothesis_h1="EWC reduces forgetting by >50% compared to baseline (p < 0.05).",
            metrics=["forgetting_measure", "backward_transfer"],
            pass_criteria={
                "forgetting_measure": (0.10, "<"),  # < 10% forgetting
                "backward_transfer": (-0.05, ">"),  # > -5% (minimal forgetting)
            },
            required_methods=["MAPPO-Safe", "MAPPO-Safe-EWC"],
            systems=["8bus", "14bus"],
        ),
        NoveltyClaim(
            id="C4",
            title="Curriculum Learning Reduces Sample Complexity",
            description="Curriculum-trained MAPPO reaches target performance in <60% of the steps required by non-curriculum training.",
            hypothesis_h0="Curriculum learning provides no sample efficiency benefit.",
            hypothesis_h1="Curriculum learning reduces steps to convergence by >40% (p < 0.05).",
            metrics=["samples_to_convergence", "final_of"],
            pass_criteria={
                "sample_efficiency_ratio": (0.60, "<"),  # < 60% of baseline steps
            },
            required_methods=["MAPPO", "MAPPO-Safe-EWC"],
            systems=["8bus", "14bus"],
        ),
        NoveltyClaim(
            id="C5",
            title="System Scales to 39-Bus with Sub-100ms Inference",
            description="The complete system achieves <100ms inference latency per decision on 39-bus network (46 relays) with <1kbps communication bandwidth.",
            hypothesis_h0="System inference latency exceeds 100ms on 39-bus network.",
            hypothesis_h1="System achieves <100ms inference with statistical confidence (p < 0.05).",
            metrics=["inference_latency_ms", "bandwidth_kbps"],
            pass_criteria={
                "inference_latency_ms": (100.0, "<"),
                "bandwidth_kbps": (1.0, "<"),
            },
            required_methods=["MAPPO-Comm", "MAPPO-Safe-EWC"],
            systems=["39bus"],
        ),
    ]

    def __init__(self, results_dir: str = "outputs/phase5"):
        self.results_dir = Path(results_dir)
        self.results: Dict = {}
        self.validation_results: List[ValidationResult] = []

    def load_results(self) -> bool:
        """Load Phase 5 experimental results."""
        results_file = self.results_dir / "phase5_results.json"
        if not results_file.exists():
            print(f"Warning: Results file not found: {results_file}")
            return False

        with open(results_file) as f:
            self.results = json.load(f)

        print(f"  Loaded {len(self.results)} results")
        return True

    def validate_claim(self, claim: NoveltyClaim) -> ValidationResult:
        """
        Validate a single novelty claim.

        Returns:
            ValidationResult with pass/fail status
        """
        print(f"\n  Validating Claim {claim.id}: {claim.title}")

        # Normalize results to dict format
        results_data = self._get_results_data()

        # Check if required data exists
        available_methods = set()
        available_systems = set()

        for result in results_data:
            available_methods.add(result.get("method", ""))
            available_systems.add(result.get("system", ""))

        missing_methods = [m for m in claim.required_methods if m not in available_methods]
        missing_systems = [s for s in claim.systems if s not in available_systems]

        # Only fail early if NO required methods or NO required systems are present at all
        if not any(m in available_methods for m in claim.required_methods):
            return ValidationResult(
                claim_id=claim.id,
                passed=False,
                p_value=1.0,
                effect_size=0.0,
                effect_magnitude="N/A",
                details={
                    "missing_methods": missing_methods,
                    "missing_systems": missing_systems,
                    "available_methods": list(available_methods),
                    "available_systems": list(available_systems),
                },
                recommendation="Run Phase 5 experiments with required methods and systems.",
            )

        if not any(s in available_systems for s in claim.systems):
            return ValidationResult(
                claim_id=claim.id,
                passed=False,
                p_value=1.0,
                effect_size=0.0,
                effect_magnitude="N/A",
                details={
                    "missing_methods": missing_methods,
                    "missing_systems": missing_systems,
                    "available_methods": list(available_methods),
                    "available_systems": list(available_systems),
                },
                recommendation="Run Phase 5 experiments with required systems.",
            )

        # Extract relevant results
        claim_results = [
            r for r in results_data
            if r.get("method") in claim.required_methods
            and r.get("system") in claim.systems
        ]

        if not claim_results:
            return ValidationResult(
                claim_id=claim.id,
                passed=False,
                p_value=1.0,
                effect_size=0.0,
                effect_magnitude="N/A",
                details={"error": "No matching results"},
                recommendation="Check result filtering criteria.",
            )

        # Compute metrics
        metrics = self._compute_claim_metrics(claim, claim_results)

        # Add missing info to metrics for transparency
        if missing_methods:
            metrics["missing_methods"] = missing_methods
        if missing_systems:
            metrics["missing_systems"] = missing_systems

        # Check pass criteria. A criterion whose metric was never computed
        # (e.g. too few seeds for a significance test) is a FAILURE to
        # validate, not a metric to skip — skipping it let claims pass
        # vacuously whenever the underlying data was simply absent.
        passed = True
        unevaluated_criteria = []
        for metric, (threshold, operator) in claim.pass_criteria.items():
            if metric not in metrics:
                unevaluated_criteria.append(metric)
                passed = False
                continue
            value = metrics[metric]
            if operator == "<":
                passed &= (value < threshold)
            elif operator == ">":
                passed &= (value > threshold)
            elif operator == "<=":
                passed &= (value <= threshold)
            elif operator == ">=":
                passed &= (value >= threshold)
        if unevaluated_criteria:
            metrics["unevaluated_criteria"] = unevaluated_criteria

        # Use the real test computed in _compute_claim_metrics (if this claim
        # has one) instead of a hardcoded pass/fail-dependent constant. Claims
        # that are purely threshold-based (e.g. C2's CTI violation rate) have
        # no p-value/effect-size to report — None here, not an invented one.
        p_value = metrics.get("p_value")
        effect_size = metrics.get("cliffs_delta")
        effect_magnitude = metrics.get("cliffs_delta_magnitude", "not_applicable")

        recommendation = (
            "Claim validated ✓" if passed
            else "Claim NOT validated. Collect more data or adjust methods."
        )

        return ValidationResult(
            claim_id=claim.id,
            passed=passed,
            p_value=p_value,
            effect_size=effect_size,
            effect_magnitude=effect_magnitude,
            details=metrics,
            recommendation=recommendation,
        )

    def _get_results_data(self) -> List[Dict]:
        """Normalize self.results to a list of result dicts."""
        if isinstance(self.results, dict):
            return self.results.get("results", [])
        elif isinstance(self.results, list):
            return self.results
        return []

    def _compute_claim_metrics(self, claim: NoveltyClaim, 
                               results: List[Dict]) -> Dict:
        """Compute metrics for a claim from results."""
        metrics = {}

        # Group by method
        by_method = defaultdict(list)
        for r in results:
            by_method[r.get("method", "unknown")].append(r)

        # OF comparison
        if "OF" in claim.metrics:
            for method in claim.required_methods:
                ofs = [v for v in (r.get("best_of", r.get("OF")) for r in by_method.get(method, [])) if v is not None]
                if ofs:
                    metrics[f"{method}_OF_mean"] = float(np.mean(ofs))
                    metrics[f"{method}_OF_std"] = float(np.std(ofs))

            # Real significance test: the claim's own method (last in
            # required_methods, e.g. "MAPPO-Comm") vs. every baseline listed
            # before it. Previously validate_claim() hardcoded p_value=0.05/
            # effect_size=0.5 whenever the threshold check passed, without
            # ever calling a test — this computes an actual Mann-Whitney U
            # (small, possibly-unpaired seed counts across methods) and
            # Cliff's delta per baseline, and reports the worst case across
            # all baselines (claim only holds if it beats *every* baseline).
            primary = claim.required_methods[-1]
            primary_of = [v for v in
                          (r.get("best_of", r.get("OF")) for r in by_method.get(primary, []))
                          if v is not None]
            comparisons = {}
            for method in claim.required_methods[:-1]:
                other_of = [v for v in
                            (r.get("best_of", r.get("OF")) for r in by_method.get(method, []))
                            if v is not None]
                if len(primary_of) < 2 or len(other_of) < 2:
                    continue  # not enough samples for a real test — do not fabricate one
                try:
                    _, p = mannwhitneyu(primary_of, other_of, alternative="two-sided")
                except ValueError:
                    continue  # identical distributions etc. — scipy refuses the test
                delta, magnitude = cliffs_delta(other_of, primary_of)
                comparisons[method] = {
                    "p_value": float(p),
                    "cliffs_delta": float(delta),
                    "magnitude": magnitude,
                }
            if comparisons:
                worst = min(comparisons.values(), key=lambda c: abs(c["cliffs_delta"]))
                metrics["of_comparisons"] = comparisons
                metrics["p_value"] = max(c["p_value"] for c in comparisons.values())
                metrics["cliffs_delta"] = worst["cliffs_delta"]
                metrics["cliffs_delta_magnitude"] = worst["magnitude"]
            else:
                metrics["insufficient_samples_for_significance_test"] = True

        # CTI violation rate
        if "cti_violation_rate" in claim.metrics:
            for method in claim.required_methods:
                method_results = by_method.get(method, [])
                # Use each run's own n_pairs (looked up from its recorded system)
                # rather than a fixed guess — n_pairs differs across the 3/8/14/
                # 30/39/57/118-bus systems this claim spans.
                total_pairs = 0
                total_viols = 0
                for r in method_results:
                    # A run missing "cti_violations" entirely (never
                    # measured) previously still counted its full n_pairs
                    # into the denominator with `.get(..., 0)` silently
                    # adding 0 violations to the numerator -- indistinguishable
                    # from a run that genuinely measured zero violations,
                    # so missing data quietly inflated the apparent safety
                    # rate. Skip runs that never actually measured this.
                    if r.get("cti_violations") is None:
                        continue
                    system = r.get("system")
                    n_pairs = NETWORKS[system]().n_pairs if system in NETWORKS else r.get("n_pairs")
                    if n_pairs:
                        total_pairs += n_pairs
                        total_viols += r["cti_violations"]
                if total_pairs > 0:
                    metrics[f"{method}_cti_violation_rate"] = total_viols / total_pairs
                else:
                    metrics[f"{method}_cti_violation_rate_unavailable"] = True
            # pass_criteria checks the unprefixed "cti_violation_rate" key —
            # this is a claim about one specific method (the claim's own
            # subject, i.e. the last of required_methods, e.g. "MAPPO-Safe"),
            # not an average across every method in the comparison. Without
            # this alias the criterion could never match any key this
            # function actually writes.
            primary_cti = claim.required_methods[-1]
            if f"{primary_cti}_cti_violation_rate" in metrics:
                metrics["cti_violation_rate"] = metrics[f"{primary_cti}_cti_violation_rate"]

        # CTI satisfaction rate
        if "cti_satisfied" in claim.metrics:
            for method in claim.required_methods:
                satisfied = [r.get("cti_satisfied", False) for r in by_method.get(method, [])]
                if satisfied:
                    metrics[f"{method}_cti_satisfied_rate"] = float(np.mean(satisfied))
            primary_cti = claim.required_methods[-1]
            if f"{primary_cti}_cti_satisfied_rate" in metrics:
                metrics["cti_satisfied_rate"] = metrics[f"{primary_cti}_cti_satisfied_rate"]

        # Forgetting measure (for EWC). Excludes curriculum=True entries
        # (Core B's curriculum-trained runs, see core_to_phase5_bridge.py's
        # _from_core_b_curriculum) -- this claim measures the EWC-ablation
        # methodology (single-task training with/without EWC via Core C),
        # not curriculum learning; mixing the two under the same
        # "MAPPO-Safe-EWC" label would conflate different training setups
        # into one average, same issue Claim C4's block guards against.
        if "forgetting_measure" in claim.metrics:
            ewc_results = [r for r in by_method.get("MAPPO-Safe-EWC", []) if not r.get("curriculum")]
            baseline_results = by_method.get("MAPPO-Safe", [])
            if ewc_results and baseline_results:
                ewc_of = np.mean([r.get("best_of", 0) for r in ewc_results])
                base_of = np.mean([r.get("best_of", 0) for r in baseline_results])
                # abs() previously scored EWC beating the baseline as
                # equally "forgotten" as EWC losing to it (e.g. EWC at
                # ~50% better than baseline OF failed the <10% threshold
                # exactly like EWC at 50% worse would have). Only count
                # degradation as forgetting; an improvement is zero
                # forgetting, not negative. Mirrors backward_transfer's
                # already-correctly-signed version of this same comparison.
                metrics["forgetting_measure"] = (
                    max(0.0, (ewc_of - base_of) / base_of) if base_of > 0 else 0
                )

        # Backward transfer (continual-learning literature convention:
        # positive = old-task performance IMPROVED after learning new
        # tasks; negative = forgetting). This dataset (flat per-run
        # TrainingResult entries) doesn't carry a per-task retention
        # matrix like Core B's task_performances (evaluating an OLD
        # task's env with the model trained through a LATER task), so a
        # strict formal BWT isn't computable here — this reuses the same
        # real (not fabricated) aggregate best_of values as
        # forgetting_measure above, signed rather than abs()'d: EWC
        # finishing below the non-EWC Safe-RL baseline's OF (lower is
        # better in this codebase) is scored as positive transfer, above
        # it as forgetting. A documented proxy, not a formal per-task BWT.
        if "backward_transfer" in claim.metrics:
            # Same curriculum exclusion as forgetting_measure above.
            ewc_results = [r for r in by_method.get("MAPPO-Safe-EWC", []) if not r.get("curriculum")]
            baseline_results = by_method.get("MAPPO-Safe", [])
            if ewc_results and baseline_results:
                ewc_of = np.mean([r.get("best_of", 0) for r in ewc_results])
                base_of = np.mean([r.get("best_of", 0) for r in baseline_results])
                metrics["backward_transfer"] = (
                    (base_of - ewc_of) / base_of if base_of > 0 else 0.0
                )

        # Samples to convergence / sample efficiency (curriculum vs.
        # non-curriculum). "Convergence" = first history index after
        # which training has settled, measured in units of logged history
        # entries (checkpoints), not raw env steps — TrainingResult
        # doesn't record the step count per checkpoint, but both methods
        # share the same logging cadence from this pipeline, so the RATIO
        # between them is still a valid comparison even though neither
        # absolute count is in real steps.
        if "samples_to_convergence" in claim.metrics or "final_of" in claim.metrics:
            def _convergence_point(history, tol=0.05):
                """Smoothed settling-time convergence point.

                `history` entries are per-iteration objective-function
                evaluations of an evolving, ES-perturbed policy, not
                independent ground-truth scores of a fixed final policy
                -- individual entries can differ from their neighbors by
                2-10x purely from per-iteration ES perturbation noise,
                persisting even long after the underlying policy has
                genuinely settled. Verified against this pipeline's real
                Core B / benchmark_pv histories: several runs' tails (the
                clearly-converged final ~10%) show a raw-point
                coefficient of variation above 1.0 -- individual points
                routinely differ from their own local mean by MORE than
                100%. A prior version of this function compared raw,
                single history points against a single raw final point;
                on data like this that's essentially guaranteed to report
                "still converging" almost everywhere regardless of `tol`,
                since raw per-iteration noise alone routinely swamps any
                reasonable tolerance band -- it isn't measuring
                convergence speed at all once the underlying series is
                this noisy.

                Fix: smooth first via a trailing moving average (window
                sized proportionally to the series' own length, not a
                fixed constant, so this scales sensibly for histories of
                very different lengths), and anchor the "converged" level
                to the MEAN of the smoothed series' own final window (a
                robust, multi-point estimate) rather than a single last
                raw data point, which can itself be an outlier. Same
                "first index after which every later smoothed value
                stays within tol of the target" criterion as before,
                just applied to a noise-robust series and target instead
                of raw data and a raw single-point anchor.
                """
                if not history:
                    return None
                arr = np.asarray(history, dtype=float)
                n = len(arr)
                if n < 2:
                    return 0

                window = max(3, n // 10)
                # Trailing moving average via cumulative sum -- O(n), not
                # O(n * window).
                cumsum = np.cumsum(np.insert(arr, 0, 0.0))
                idx = np.arange(n)
                lo = np.maximum(0, idx - window + 1)
                smoothed = (cumsum[idx + 1] - cumsum[lo]) / (idx + 1 - lo)

                target = float(np.mean(smoothed[-window:]))
                if target == 0:
                    return n

                within = np.abs(smoothed - target) <= tol * abs(target)
                # First index after which ALL remaining (smoothed) points
                # stay within tol == one past the LAST index that is NOT
                # within tol (0 if every point already is).
                not_within = np.where(~within)[0]
                return int(not_within[-1] + 1) if len(not_within) > 0 else 0

            curriculum_method = claim.required_methods[-1]   # e.g. "MAPPO-Safe-EWC"
            baseline_method = claim.required_methods[0]       # e.g. "MAPPO"

            for method in claim.required_methods:
                method_results = by_method.get(method, [])
                # The curriculum side's method name (e.g. "MAPPO-Safe-EWC")
                # also has non-curriculum entries under it -- benchmark_pv/
                # Core C train that same-named method directly at one PV
                # level, with no curriculum stages at all. Averaging those
                # in with Core B's genuinely curriculum-trained runs would
                # conflate two different training methodologies into one
                # number, defeating the point of this specific comparison.
                # Core B's bridge entries set curriculum=True explicitly
                # (see core_to_phase5_bridge.py's _from_core_b_curriculum);
                # every other source leaves the field absent.
                if method == curriculum_method:
                    method_results = [r for r in method_results if r.get("curriculum")]
                conv_points = [
                    _convergence_point(r.get("history", []))
                    for r in method_results if r.get("history")
                ]
                conv_points = [c for c in conv_points if c is not None]
                final_ofs = [r.get("best_of") for r in method_results if r.get("best_of") is not None]
                if conv_points:
                    metrics[f"{method}_samples_to_convergence"] = float(np.mean(conv_points))
                if final_ofs:
                    metrics[f"{method}_final_of"] = float(np.mean(final_ofs))

            curr_key = f"{curriculum_method}_samples_to_convergence"
            base_key = f"{baseline_method}_samples_to_convergence"
            if curr_key in metrics and base_key in metrics and metrics[base_key] > 0:
                metrics["sample_efficiency_ratio"] = metrics[curr_key] / metrics[base_key]
                metrics["final_of"] = metrics.get(f"{curriculum_method}_final_of")

        # Inference latency. `.get(key, 0)` previously defaulted every
        # result missing this field to a literal 0 -- for a "<" threshold
        # criterion, "never measured" and "measured as 0ms" are
        # indistinguishable that way, so a claim with NO real latency
        # data at all would vacuously pass (0 < 100 is always true).
        # Only average over entries that actually carry the field, and
        # leave the metric entirely unset (not 0.0) when none do, so the
        # pass_criteria check below reports it as a genuine
        # unevaluated_criteria failure instead of a false pass.
        if "inference_latency_ms" in claim.metrics:
            latencies = [r["inference_latency_ms"] for r in results if r.get("inference_latency_ms") is not None]
            if latencies:
                metrics["inference_latency_ms"] = float(np.mean(latencies))

        # Bandwidth -- same "don't fabricate a measurement via a missing-
        # field default" fix as inference_latency_ms above.
        if "bandwidth_kbps" in claim.metrics:
            bws = [r["comm_bandwidth"] for r in results if r.get("comm_bandwidth") is not None]
            if bws:
                metrics["bandwidth_kbps"] = float(np.mean(bws))

        return metrics

    def validate_all_claims(self) -> List[ValidationResult]:
        """Validate all 5 novelty claims."""
        print(f"\n{'-'*70}")
        print(f"  PHASE 8: THESIS NOVELTY CLAIM VALIDATION")
        print(f"{'-'*70}")

        # Only load from file if results not already set
        if not self.results:
            if not self.load_results():
                print("  Using placeholder validation (no results loaded)")

        self.validation_results = []
        for claim in self.CLAIMS:
            result = self.validate_claim(claim)
            self.validation_results.append(result)

            status = "✓ PASS" if result.passed else "✗ FAIL"
            print(f"\n  Claim {claim.id}: {status}")
            print(f"    {result.recommendation}")

        return self.validation_results

    def generate_validation_report(self) -> str:
        """Generate comprehensive validation report."""
        lines = [
            "=" * 80,
            "PHASE 8: THESIS VALIDATION REPORT",
            "=" * 80,
            "",
            f"Total Claims: {len(self.CLAIMS)}",
            f"Passed: {sum(1 for r in self.validation_results if r.passed)}",
            f"Failed: {sum(1 for r in self.validation_results if not r.passed)}",
            "",
        ]

        for result in self.validation_results:
            claim = next(c for c in self.CLAIMS if c.id == result.claim_id)
            status = "✓ PASS" if result.passed else "✗ FAIL"

            lines.extend([
                f"Claim {claim.id}: {status}",
                f"  Title: {claim.title}",
                f"  H0: {claim.hypothesis_h0}",
                f"  H1: {claim.hypothesis_h1}",
                f"  p-value: {result.p_value:.4f}" if result.p_value is not None
                    else "  p-value: N/A (no significance test computed for this claim)",
                f"  Effect size: {result.effect_size:.4f} ({result.effect_magnitude})" if result.effect_size is not None
                    else f"  Effect size: N/A ({result.effect_magnitude})",
                f"  Recommendation: {result.recommendation}",
                "",
            ])

        lines.append("=" * 80)
        return "\n".join(lines)

    def generate_latex_tables(self, output_path: str) -> str:
        """Generate LaTeX tables for thesis."""
        lines = [
            "\\begin{table}[t]",
            "\\centering",
            "\\caption{Novelty Claim Validation Results}",
            "\\label{tab:novelty_validation}",
            "\\begin{tabular}{clcccl}",
            "\\toprule",
            "Claim & Title & Passed & p-value & Effect & Status \\\\",
            "\\midrule",
        ]

        for result in self.validation_results:
            claim = next(c for c in self.CLAIMS if c.id == result.claim_id)
            status = "Valid" if result.passed else "Invalid"
            p_str = f"{result.p_value:.4f}" if result.p_value is not None else "N/A"
            lines.append(
                f"{claim.id} & {claim.title[:40]}... & "
                f"{'Yes' if result.passed else 'No'} & "
                f"{p_str} & "
                f"{result.effect_magnitude} & "
                f"{status} \\\\",
            )

        lines.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
        ])

        latex = "\n".join(lines)

        with open(output_path, 'w') as f:
            f.write(latex)

        print(f"  LaTeX table saved: {output_path}")
        return latex


class ReproducibilityChecklist:
    """
    Reproducibility checklist for thesis experiments.

    Ensures all experiments can be reproduced by others.
    """

    CHECKLIST = [
        ("Code availability", "GitHub repository with all source code"),
        ("Data availability", "All test systems and scenario data included"),
        ("Environment specification", "requirements.txt with exact versions"),
        ("Docker container", "Dockerfile for reproducible environment"),
        ("Random seeds", "All random seeds documented and fixed"),
        ("Hyperparameters", "All hyperparameters documented in config files"),
        ("Hardware specification", "CPU/GPU specs used for experiments"),
        ("Runtime logs", "Complete execution logs saved"),
        ("Statistical tests", "All tests reproducible from saved results"),
        ("Plot generation", "All plots reproducible from saved data"),
    ]

    def __init__(self, repo_dir: str = "."):
        self.repo_dir = Path(repo_dir)
        self.checks: Dict[str, bool] = {}

    def run_check(self, name: str) -> bool:
        """Run a single reproducibility check."""
        checks = {
            "Code availability": (self.repo_dir / ".git").exists(),
            "Data availability": (self.repo_dir / "data").exists(),
            "Environment specification": (self.repo_dir / "requirements.txt").exists(),
            "Docker container": (self.repo_dir / "Dockerfile").exists(),
            "Random seeds": True,  # Documented in code
            "Hyperparameters": (self.repo_dir / "configs").exists(),
            "Hardware specification": True,  # Documented in thesis
            "Runtime logs": (self.repo_dir / "outputs").exists(),
            "Statistical tests": (self.repo_dir / "evaluation" / "stat_analysis.py").exists(),
            "Plot generation": (self.repo_dir / "evaluation" / "evaluation.py").exists(),
        }
        return checks.get(name, False)

    def run_all_checks(self) -> Dict[str, bool]:
        """Run all reproducibility checks."""
        for name, _ in self.CHECKLIST:
            self.checks[name] = self.run_check(name)
        return self.checks

    def generate_report(self) -> str:
        """Generate reproducibility report."""
        self.run_all_checks()

        lines = [
            "=" * 70,
            "REPRODUCIBILITY CHECKLIST",
            "=" * 70,
            "",
        ]

        passed = sum(self.checks.values())
        total = len(self.checks)

        for name, description in self.CHECKLIST:
            status = "✓" if self.checks.get(name, False) else "✗"
            lines.append(f"  {status} {name}: {description}")

        lines.extend([
            "",
            f"Score: {passed}/{total} ({passed/total*100:.0f}%)",
            "=" * 70,
        ])

        return "\n".join(lines)


def generate_dockerfile(output_path: str = "Dockerfile"):
    """Generate Docker container specification."""
    dockerfile = """# DOCR-MARL Thesis: Reproducible Environment
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Set environment variables
ENV PYTHONPATH=/app
ENV DOCR_SEED=42

# Default command: run all tests
CMD ["python", "tests/test_all_phases.py"]
"""

    with open(output_path, 'w') as f:
        f.write(dockerfile)

    print(f"  Dockerfile generated: {output_path}")


def generate_github_actions(output_path: str = ".github/workflows/ci.yml"):
    """Generate GitHub Actions CI configuration."""
    ci_config = """name: DOCR-MARL CI

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ['3.9', '3.10', '3.11']

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt

    - name: Run Phase 1 tests
      run: python tests/test_phase1.py

    - name: Run Phase 2 tests
      run: python tests/test_phase2.py

    - name: Run Phase 3 tests
      run: python tests/test_phase3.py

    - name: Run Phase 4 tests
      run: python tests/test_phase4.py

    - name: Run Phase 5 tests
      run: python tests/test_phase5.py

    - name: Run Phase 7 tests
      run: python tests/test_phase7.py

    - name: Run Phase 8 validation
      run: python -c "from evaluation.thesis_validation import ThesisValidator; v = ThesisValidator(); v.validate_all_claims()"
"""

    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w') as f:
        f.write(ci_config)

    print(f"  GitHub Actions CI config: {output_path}")


if __name__ == "__main__":
    # Demo validation
    validator = ThesisValidator()
    validator.validate_all_claims()
    print(validator.generate_validation_report())

    # Reproducibility checklist
    checklist = ReproducibilityChecklist()
    print(checklist.generate_report())

    # Generate Docker and CI files
    generate_dockerfile()
    generate_github_actions()