"""
experiments/build_results_registry.py
──────────────────────────────────────
Consolidates this repo's real (non-synthetic) multi-system experiment
outputs into one queryable store, via ResultsAggregator.

Data sources scanned (only what genuinely varies method x system x PV
level -- see ResultsAggregator's own docstring; the other Core training
runs feed single-condition figures with bespoke per-figure schemas that
belong in run_post_processing.py, not here):

  outputs/{system}/fig05_pv_sweep.npz       -- OF vs PV penetration (Fig 5)
  outputs/{system}/fig06_cti_violations.npz -- CTI violation rate vs PV (Fig 6)
  results/core_cache/*.pkl named "core_C_baselines_<system>"
                                             -- MARL-vs-baseline comparison (Core C)

All three are pre-aggregated at the source -- the Fig 5/6 scripts save only
mean/std per (method, pv_level, system); the underlying per-seed values are
computed for significance testing (see the *_stats.txt they also write) but
never persisted to disk. So this script feeds
ResultsAggregator.ingest_preaggregated() rather than the raw-per-run
aggregate() path -- see that method's docstring for why the distinction
matters (it's what keeps std/n_runs honest instead of silently reporting
std=0.0 for a single already-averaged data point).

Usage:
    python3 experiments/build_results_registry.py
    python3 experiments/build_results_registry.py --outputs-dir outputs \\
        --cache-dir results/core_cache --output-dir results/reports

Run this after any Fig 5 / Fig 6 / Core C run to fold the new data in --
it's additive across systems (each outputs/{system}/ directory contributes
its own rows) but a full rebuild each time, not an incremental append.
"""
import argparse
import json
import os
import pickle
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.results_aggregator import ResultsAggregator

# Core C's config never varies PV level -- it runs once at the codebase's
# documented "active scope" PV (see CoreEvalBridge's pv_level default and
# EXPERIMENT_GUIDE.md). Recorded explicitly here rather than defaulting
# pv_level to 0.0, which would misrepresent it as a 0%-PV run.
CORE_C_PV_LEVEL = 0.6

_N_SEEDS_RE = re.compile(r"across (\d+) seeds")


def _infer_n_seeds(stats_path: Path) -> int:
    """n_seeds was only added to the npz itself after the 8bus/14bus runs in
    this repo's history were produced, so for those, fall back to parsing
    the "across N seeds" line the same run wrote to its stats.txt."""
    if stats_path.exists():
        m = _N_SEEDS_RE.search(stats_path.read_text())
        if m:
            return int(m.group(1))
    return 0


def load_fig05(system_dir: Path, system: str) -> list:
    npz_path = system_dir / "fig05_pv_sweep.npz"
    if not npz_path.exists():
        return []
    data = np.load(npz_path)
    pv_levels = data["pv_levels"]
    n_seeds = int(data["n_seeds"]) if "n_seeds" in data.files else _infer_n_seeds(
        system_dir / "fig05_pv_sweep_stats.txt")
    methods = sorted({k.rsplit("_mean_OF", 1)[0] for k in data.files if k.endswith("_mean_OF")})

    records = []
    for method in methods:
        mean_of = data[f"{method}_mean_OF"]
        std_of = data[f"{method}_std_OF"]
        mean_cti = data[f"{method}_mean_cti"]  # mean CTI *violation count*, not a margin
        for i, pv in enumerate(pv_levels):
            record = {
                "method": method, "system": system,
                "pv_level": round(float(pv), 4), "fault_type": "3ph",
                "of_mean": float(mean_of[i]), "of_std": float(std_of[i]),
                "cti_violations_mean": float(mean_cti[i]),
                "source": f"{npz_path} (pre-aggregated; raw per-seed values not persisted)",
            }
            # Omit n_runs entirely when unknown (0), rather than writing a
            # literal 0 -- ingest_preaggregated() merges field-by-field, so a
            # false "known" 0 here would silently clobber a real n_runs
            # value contributed by a companion Fig 6 record for the same key.
            if n_seeds:
                record["n_runs"] = n_seeds
            records.append(record)
    return records


def load_fig06(system_dir: Path, system: str) -> list:
    npz_path = system_dir / "fig06_cti_violations.npz"
    if not npz_path.exists():
        return []
    data = np.load(npz_path)
    pv_levels = data["pv_levels"]
    n_seeds = int(data["n_seeds"]) if "n_seeds" in data.files else 0
    methods = sorted({
        k.rsplit("_violation_rate", 1)[0] for k in data.files
        if k.endswith("_violation_rate") and not k.endswith("_violation_rate_std")
    })

    records = []
    for method in methods:
        # evaluate_cti_severity() in this script computes this as
        # 100 * violations / n_checks -- a 0-100 PERCENTAGE, not a 0-1
        # fraction, and not the same quantity as Fig 5's cti_violations_mean
        # (a raw violation count). Kept in its own field (cti_violation_rate_pct)
        # rather than merged into cti_violations_mean to avoid conflating
        # the two units.
        rate_pct = data[f"{method}_violation_rate"]
        for i, pv in enumerate(pv_levels):
            record = {
                "method": method, "system": system,
                "pv_level": round(float(pv), 4), "fault_type": "3ph",
                "cti_violation_rate_pct": float(rate_pct[i]),
                "cti_ok_rate": float((100.0 - rate_pct[i]) / 100.0),
                "source": f"{npz_path} (pre-aggregated; raw per-seed values not persisted)",
            }
            if n_seeds:  # see the matching comment in load_fig05
                record["n_runs"] = n_seeds
            records.append(record)
    return records


def load_core_c(cache_dir: Path) -> list:
    """
    Only ingests cache entries literally named "core_C_baselines_<system>" --
    the naming convention Core C adopted once it was parametrized by system
    (see run_core_training.py's CacheManager usage). The pre-parametrization
    orphan entry named plain "core_C_baselines" (implicitly 8-bus, left over
    from before that fix) is intentionally excluded: it never declared which
    system it was for, and guessing would repeat exactly the kind of
    undisclosed-provenance bug this repo's audit history has repeatedly had
    to fix.
    """
    index_path = cache_dir / "cache_index.json"
    if not index_path.exists():
        return []
    index = json.loads(index_path.read_text())

    records = []
    for key, meta in index.items():
        m = re.match(r"^core_C_baselines_(.+)$", meta.get("name", ""))
        if not m:
            continue
        system = m.group(1)
        pkl_path = cache_dir / f"{key}.pkl"
        if not pkl_path.exists():
            continue
        with open(pkl_path, "rb") as f:
            payload = pickle.load(f)
        for method, arr in payload.items():
            arr = np.asarray(arr)
            # Same reduction run_post_processing.py's Figs 11/31/32/33 use to
            # turn Core C's raw (n_runs, n_episodes) curves into one scalar
            # per method: mean of the last 100 episodes (or all, if fewer).
            window = arr[..., -100:] if arr.ndim >= 1 and arr.shape[-1] > 100 else arr
            records.append({
                "method": method, "system": system,
                "pv_level": CORE_C_PV_LEVEL, "fault_type": "3ph",
                "n_runs": arr.shape[0] if arr.ndim > 1 else 1,
                "of_mean": float(np.mean(window)), "of_std": float(np.std(window)),
                "source": f"results/core_cache/{key}.pkl [{meta['name']}]",
            })
    return records


def build_registry(outputs_dir: Path, cache_dir: Path) -> ResultsAggregator:
    agg = ResultsAggregator([])
    if outputs_dir.exists():
        for system_dir in sorted(outputs_dir.iterdir()):
            if not system_dir.is_dir():
                continue
            system = system_dir.name
            agg.ingest_preaggregated(load_fig05(system_dir, system))
            agg.ingest_preaggregated(load_fig06(system_dir, system))
    agg.ingest_preaggregated(load_core_c(cache_dir))
    return agg


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--outputs-dir", default="outputs")
    parser.add_argument("--cache-dir", default="results/core_cache")
    parser.add_argument("--output-dir", default="results/reports")
    args = parser.parse_args()

    outputs_dir = Path(args.outputs_dir)
    cache_dir = Path(args.cache_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    agg = build_registry(outputs_dir, cache_dir)
    n = len(agg.aggregated)
    systems = sorted({v.system for v in agg.aggregated.values()})
    print(f"Consolidated {n} (method, system, pv_level, fault_type) rows "
          f"across systems: {systems or '(none found)'}")
    if n == 0:
        print("Nothing found -- run the Fig 5/6 sweep scripts or Core C first (see README.md).")

    agg.export_aggregated_json(str(output_dir / "results_registry.json"))
    agg.export_aggregated_csv(str(output_dir / "results_registry.csv"))


if __name__ == "__main__":
    main()
