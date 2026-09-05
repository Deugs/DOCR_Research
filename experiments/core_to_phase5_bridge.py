"""
experiments/core_to_phase5_bridge.py
─────────────────────────────────────
Bridges the real results this repo's ACTIVE pipeline already produces
(results/benchmark_pv*.pkl and results/core_cache/*.pkl, written by
run_thesis_experiments.py -> run_core_training.py's Cores A-K, the
pipeline behind all 58 thesis figures) into the flat per-run result list
evaluation/thesis_validation.py's ThesisValidator expects to load from
outputs/phase5/phase5_results.json.

Why this exists
----------------
ThesisValidator's 5 novelty claims (C1-C5) were never wired to the
Core A-K pipeline. They read a file only written by a separate, legacy
entry point (`mains/main5.py --phase5` -> experiments/training_pipeline.py's
TrainingPipeline.save_results()) that has never actually been run in this
project. Running that second, redundant large-scale pipeline from scratch
would take days; this bridge instead lets the SAME results already
generated/validated for the 58 figures also feed the formal statistical
claim validation, in exactly the shape TrainingPipeline.save_results()
writes: a bare JSON list of flat per-run dicts (see that method + this
module's build_phase5_results() below).

Scope (matches EXPERIMENT_GUIDE.md's Reproducibility Checklist)
-----------------------------------------------------------------
Active systems: 8bus, 14bus, 39bus. PV sweep: 0%, 20%, 40%, 60%.

Method-name disambiguation (read this before trusting a "MAPPO*" entry)
-------------------------------------------------------------------------
The SAME method names mean DIFFERENT trained configurations depending on
source:
  - results/benchmark_pv*.pkl (evaluation/benchmark_suite.py's
    BenchmarkSuite, run alongside Core A-K training by
    run_thesis_experiments.py's Phase 2 "Objective 4" step) trains
    "MAPPO", "MAPPO-Safe", "MAPPO-Safe-EWC" WITHOUT GNN communication
    (BenchmarkSuite._run_marl_method: `use_comm = "Comm" in method`,
    and none of these names contain "Comm"). It covers exactly this
    scope's 3 systems x 4 PV levels x >=3 runs.
  - run_core_training.py's Core C (core_C_baselines_{system}) trains its
    own "MAPPO"/"MAPPO-Safe"/"MAPPO-Safe-EWC" WITH GNN communication
    (_run_baseline_single: `use_comm = method in [...]` includes
    "MAPPO"). Core C's "MAPPO" is therefore relabeled "MAPPO-Comm" below
    -- it IS the comm-enabled variant the claims call that, just not
    literally named that in the cache. Core C only covers 8bus.

Because of this, benchmark_pv is used as the PRIMARY source for
"MAPPO"/"MAPPO-Safe"/"MAPPO-Safe-EWC"/"GA-SQP"/"HWOA"/"HHO"/"JAYA"/"EEO"
(it already spans the full scope with real multi-run samples); Core C
only supplies "IPPO", "Traditional", and "MAPPO-Comm" (all absent from
benchmark_pv), each 8bus-only.

Honesty note: several claim/method/system combinations have no real data
anywhere in the current pipeline. This bridge does not fabricate values
to fill those gaps -- it leaves the field absent so ThesisValidator's own
"insufficient data"/"Run Phase 5 experiments" disclosures fire honestly
rather than silently passing on invented numbers. See
_known_gaps() and each _from_* function's docstring for exactly what's
missing and why:
  - "Single-PPO" (C1's required_methods): not run anywhere in the active
    pipeline (benchmark_pv's obj4_methods list omits it; Core C doesn't
    train it either).
  - Claim C5 (39bus inference latency + bandwidth): Core I's latency
    measurement and Core G's bandwidth sweep are both 8bus-only. No Core
    measures either quantity on 39bus.
  - Claim C4's "samples_to_convergence" needs a per-checkpoint `history`
    array with a comparable logging cadence between the curriculum and
    non-curriculum method. Core B (the only curriculum=True path) logs
    per-PV-stage task performance, not a per-episode timeline comparable
    to benchmark_pv's/Core C's per-episode returns -- forcing a
    comparison across those mismatched granularities would be
    methodologically misleading, so `history` is intentionally left
    unset here. ThesisValidator will correctly report this criterion as
    unevaluated rather than silently fabricating a ratio.

Usage
-----
    python -m experiments.core_to_phase5_bridge
    # writes outputs/phase5/phase5_results.json and immediately runs
    # ThesisValidator against it, printing which claims can/can't be
    # validated with the bridged data.

    from experiments.core_to_phase5_bridge import build_phase5_results
    results = build_phase5_results()  # list of flat dicts, in memory
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent

# run_post_processing.py's load_cache() (name-based cache_index.json
# lookup, preferring the most-recently-created entry under that name) is
# the single already-battle-tested way to load a Core's cached result --
# reused here rather than reimplementing cache-index parsing.
import sys
sys.path.insert(0, str(REPO_ROOT))
import run_post_processing as _pp  # noqa: E402


def _load_core_cache(name: str) -> Optional[Dict]:
    try:
        return _pp.load_cache(name)
    except Exception as e:
        logger.warning(f"Bridge: failed to load core cache '{name}': {e}")
        return None


def _n_pairs(system: str) -> int:
    from network.network_data import NETWORKS
    try:
        return NETWORKS[system]().n_pairs
    except Exception:
        return 20  # 8bus's real value, a reasonable fallback


def _entry(method: str, system: str, pv_level: float, run_id: int, **fields) -> Dict:
    """A minimal, honestly-partial result entry -- only fields actually
    known are included; everything else is left absent (not zero/None-
    padded) so ThesisValidator's own missing-data handling stays accurate
    instead of silently treating "unmeasured" as "measured as zero"."""
    entry = {"method": method, "system": system, "pv_level": pv_level,
              "fault_type": "3ph", "run_id": run_id}
    entry.update({k: v for k, v in fields.items() if v is not None})
    return entry


def _from_benchmark_pv() -> List[Dict]:
    """results/benchmark_pv{000,020,040,060}.pkl -- the Objective-4
    comparison run_thesis_experiments.py's Phase 2 produces alongside
    Core A-K training (evaluation/benchmark_suite.py's BenchmarkSuite).
    Primary source for classical/metaheuristic baselines and the
    no-comm MAPPO family; see this module's docstring for why.

    Shape (BenchmarkSuite.run_all()'s own docstring):
        {system_name: [run1, run2, ...]}, each run = {method_name: result_dict}
    """
    out = []
    for pv_pct, pv_level in [(0, 0.0), (20, 0.2), (40, 0.4), (60, 0.6)]:
        f = REPO_ROOT / "results" / f"benchmark_pv{pv_pct:03d}.pkl"
        if not f.exists():
            continue
        try:
            with open(f, "rb") as fh:
                suite_results = pickle.load(fh)
        except Exception as e:
            logger.warning(f"Bridge: failed to load {f}: {e}")
            continue

        for system, run_list in suite_results.items():
            for run_idx, run_dict in enumerate(run_list):
                for method, result in run_dict.items():
                    if not isinstance(result, dict):
                        continue
                    of_val = result.get("OF")
                    if of_val is None:
                        continue
                    c = result.get("constraints", {}) or {}
                    out.append(_entry(
                        method, system, pv_level, run_id=run_idx,
                        best_of=float(of_val), OF=float(of_val),
                        cti_satisfied=c.get("cti_satisfied"),
                        cti_violations=c.get("cti_violations"),
                        elapsed_seconds=result.get("elapsed"),
                    ))
    return out


def _from_core_c(system: str = "8bus") -> List[Dict]:
    """run_core_training.py's Core C (core_C_metrics_{system}) -- supplies
    "IPPO", "Traditional", and "MAPPO-Comm" (Core C's own comm-enabled
    "MAPPO", relabeled -- see module docstring). Uses the real per-run
    metrics run_core_training.py's Group-D fix added (final_of,
    cti_satisfaction_rate under each method's "per_run" list) so multiple
    independent samples are available for a real significance test, not
    just the cross-run mean.
    """
    metrics = _load_core_cache(f"core_C_metrics_{system}")
    if not metrics:
        return []

    # Only the names benchmark_pv doesn't already cover -- avoid double-
    # counting "MAPPO"/"MAPPO-Safe"/"MAPPO-Safe-EWC" under two different
    # real configurations (comm vs no-comm) as if they were one method.
    LABEL = {"IPPO": "IPPO", "Traditional": "Traditional", "MAPPO": "MAPPO-Comm"}

    out = []
    n_pairs = _n_pairs(system)
    for core_method, claim_method in LABEL.items():
        m = metrics.get(core_method)
        if not m:
            continue
        for run_idx, run in enumerate(m.get("per_run", [])):
            final_of = run.get("final_of")
            if final_of is None:
                continue
            cti_rate = run.get("cti_satisfaction_rate")
            out.append(_entry(
                claim_method, system, 0.6, run_id=run_idx,
                best_of=final_of, OF=final_of,
                cti_satisfied=(cti_rate is not None and cti_rate > 0.5),
                # cti_satisfaction_rate is a per-run rate over ~200 eval
                # episodes each checking whole-network CTI satisfaction,
                # not a literal per-pair violation count -- approximated
                # back into a pair-count for the cti_violation_rate
                # metric's total_viols/total_pairs formula. A documented
                # proxy, not a direct per-pair measurement.
                cti_violations=(None if cti_rate is None else round((1 - cti_rate) * n_pairs)),
            ))
    return out


def _from_core_i_and_g(system: str = "8bus") -> List[Dict]:
    """Core I's real inference-latency measurement and Core G's real
    bandwidth sweep, both {system}-only -- feeds Claim C5's
    inference_latency_ms/comm_bandwidth. C5 formally requires
    system="39bus" specifically; neither Core measures either quantity
    on 39bus, so this cannot make C5 pass on its own system requirement
    -- included anyway (labeled with whatever system was actually
    measured) since it's the only real data available, and
    ThesisValidator's own system-match check will correctly still report
    the 39bus requirement as unmet.
    """
    out = []

    deployment = _load_core_cache("core_I_deployment")
    latency = (deployment or {}).get("latency", {})
    if latency.get("mean_ms") is not None:
        out.append(_entry(
            "MAPPO-Safe-EWC", system, 0.6, run_id=0,
            inference_latency_ms=latency["mean_ms"],
        ))

    robustness = _load_core_cache("core_G_robustness")
    bandwidth = (robustness or {}).get("bandwidth", {})
    for depth, point in bandwidth.items():
        if not isinstance(point, dict) or point.get("bandwidth_kbps") is None:
            continue
        out.append(_entry(
            "MAPPO-Comm", system, 0.6,
            run_id=int(depth) if str(depth).isdigit() else 0,
            comm_bandwidth=point["bandwidth_kbps"],
        ))
    return out


def _known_gaps() -> List[str]:
    """Structural gaps in this bridge's coverage -- printed by __main__
    and returned by build_phase5_results()'s caller for visibility.
    Nothing here is fabricated to close these; see module docstring."""
    return [
        "Single-PPO: not trained anywhere in the active pipeline "
        "(benchmark_pv's obj4_methods omits it; Core C doesn't run it "
        "either) -- Claim C1's comparison against it is structurally "
        "unavailable, not just under-sampled.",
        "Claim C5 (39bus inference latency < 100ms, bandwidth < 1kbps): "
        "Core I/Core G only measure these on 8bus. No Core measures "
        "either quantity on 39bus.",
        "Claim C4 (samples_to_convergence / sample_efficiency_ratio): "
        "no comparable per-episode `history` exists between a "
        "curriculum-trained (Core B, per-stage logging) and non-"
        "curriculum (benchmark_pv/Core C, per-episode logging) run -- "
        "left unset rather than compared across mismatched granularities.",
    ]


def build_phase5_results() -> List[Dict]:
    """Assemble the full bridged result list from every available real
    source. See each _from_* function's docstring for exactly what's
    real vs. structurally unavailable."""
    results: List[Dict] = []
    results += _from_benchmark_pv()
    results += _from_core_c("8bus")
    results += _from_core_i_and_g("8bus")
    logger.info(f"Bridge: assembled {len(results)} result entries "
                f"(benchmark_pv + core_C_metrics + core_I/core_G)")
    return results


def write_phase5_results(output_path: str = None) -> str:
    """Write build_phase5_results()'s output to outputs/phase5/phase5_results.json
    (or `output_path`), in the exact bare-list shape
    TrainingPipeline.save_results() writes."""
    if output_path is None:
        output_path = str(REPO_ROOT / "outputs" / "phase5" / "phase5_results.json")
    results = build_phase5_results()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Bridge: wrote {len(results)} entries to {out}")
    return str(out)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    path = write_phase5_results()
    print(f"\nWrote {path}")

    print("\nKnown coverage gaps (not fabricated, left absent):")
    for gap in _known_gaps():
        print(f"  - {gap}")

    # Report right away which claims can/can't validate with the bridged
    # data, rather than only surfacing this inside a separate
    # `python -m evaluation.thesis_validation` run.
    from evaluation.thesis_validation import ThesisValidator
    validator = ThesisValidator(results_dir=str(Path(path).parent))
    validator.results = build_phase5_results()
    validator.validate_all_claims()
    print()
    print(validator.generate_validation_report())
