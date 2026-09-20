#!/usr/bin/env python3
"""
run_full_campaign.py — Orchestrates the full multi-system DOCR-MARL
training campaign (default: 8bus + 14bus + 39bus), replacing the ad-hoc
shell loop used to drive this project's Runpod campaigns, with two speed
improvements over a plain sequential `for system in ...` loop:

1. Cores A/B/C/D's fast-mode n_runs auto-scale to whatever --n-workers is
   actually passed (see run_core_training.py's _n_runs_for_wave) --
   nothing to do here, that happens automatically inside run_core_training.py.

2. Only Core B and Core C actually vary per system -- every other Core
   (A/D/E/F/G/H/I/J/K) is either 8bus-only by design or already internally
   spans all 3 systems in one call, with cache keys that don't include
   "system" at all. So after the first ("base") system's full pipeline
   populates every one of those shared caches, the remaining systems only
   need Core B and Core C -- and since those two DO key their cache by
   system, running them concurrently for the remaining systems is safe AS
   LONG AS each concurrent process writes to its own --cache-dir (avoids
   a lost-update race on the shared cache_index.json; the .pkl payloads
   themselves would never collide anyway since "system" is baked into
   their config hash, but the index file has no locking).

   Figure/report generation is NOT parallelized across systems: reading
   run_thesis_experiments.py and run_post_processing.py directly found
   that results/figures/, results/reports/compilation_report.json, and
   results/benchmark_pv*.pkl are all shared, non-per-system paths that
   two concurrent full `run_thesis_experiments.py --system X` runs would
   silently overwrite or race on. So this script keeps that phase
   strictly sequential, one system at a time -- it's the cheap part
   anyway once every system's Core data is already cached.

Usage:
    python run_full_campaign.py --n-workers 32 --fast
    python run_full_campaign.py --n-workers 64 --fast --pv 0.6 --episodes 500
    python run_full_campaign.py --n-workers 32 --fast --no-concurrent-tail
    python run_full_campaign.py --n-workers 32 --fast --skip-figures
"""
import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import List

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_CACHE_DIR = REPO_ROOT / "results" / "core_cache"


def _status_path(cache_dir: Path) -> Path:
    """Campaign checkpoint file, written next to the cache so both live on
    the same storage -- if --cache-root points at a mounted network
    volume, this survives pod loss right along with the cached results.
    Not required for resumption (CacheManager's content-addressed cache
    already makes re-running the same command a fast cache-hit replay),
    but gives a watchdog or a human a real signal for "is this campaign
    actually done" instead of a guessed sleep duration."""
    return cache_dir.parent / "campaign_status.json"


def _write_status(cache_dir: Path, phase: str, **extra) -> None:
    status_path = _status_path(cache_dir)
    status = json.loads(status_path.read_text()) if status_path.exists() else {"phases": []}
    status["phases"].append({"phase": phase, "timestamp": time.time(), **extra})
    status["last_phase"] = phase
    status["complete"] = phase == "campaign_complete"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps(status, indent=2))


def _base_flags(args: argparse.Namespace) -> List[str]:
    flags = []
    if args.fast:
        flags.append("--fast")
    if args.pv is not None:
        flags += ["--pv", str(args.pv)]
    if args.episodes is not None:
        flags += ["--episodes", str(args.episodes)]
    if args.fault_type is not None:
        flags += ["--fault-type", args.fault_type]
    return flags


def _run_blocking(cmd: List[str], description: str) -> None:
    print(f"\n{'=' * 70}\n>>> {description}\nCMD: {' '.join(cmd)}\n{'=' * 70}", flush=True)
    t0 = time.time()
    result = subprocess.run(cmd, cwd=REPO_ROOT)
    elapsed = time.time() - t0
    if result.returncode != 0:
        raise RuntimeError(
            f"{description} failed (exit {result.returncode}) after {elapsed:.1f}s: {' '.join(cmd)}")
    print(f">>> {description} done in {elapsed:.1f}s", flush=True)


def phase_a_train_base_system(base_system: str, n_workers: int, cache_dir: Path,
                               args: argparse.Namespace) -> None:
    """Base system alone, full workers -- populates every shared/8bus-only
    Core cache (A, D, E, F, G, H, I, J, K) plus this system's own Core
    B/C. --n-runs is deliberately never passed here: leaving it unset
    lets run_core_training.py's _n_runs_for_wave auto-scale to n_workers
    instead of a fixed constant (fast mode only; full mode uses its own
    fixed statistical-power constants regardless)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cmd = ([sys.executable, "run_core_training.py", "--all",
            "--system", base_system, "--n-workers", str(n_workers),
            "--cache-dir", str(cache_dir)]
           + _base_flags(args))
    _run_blocking(cmd, f"Phase A: train {base_system} (base system, full workers)")


def phase_b_train_remaining_concurrently(systems: List[str], n_workers: int, cache_dir: Path,
                                          args: argparse.Namespace) -> List[Path]:
    """Core B/C training for the remaining systems, launched at the same
    time, each in its own --cache-dir. Returns the temporary cache dirs
    for phase_c to merge. Each process's output is redirected to its own
    log file (concurrent stdout from 2 long-running processes interleaved
    on one terminal is unreadable otherwise)."""
    log_dir = cache_dir.parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    per_system_workers = max(1, n_workers // len(systems))
    cache_dirs = []
    procs = []
    for system in systems:
        sub_cache_dir = cache_dir.parent / f"core_cache_{system}"
        sub_cache_dir.mkdir(parents=True, exist_ok=True)
        cache_dirs.append(sub_cache_dir)
        log_path = log_dir / f"campaign_phaseB_{system}_{int(time.time())}.log"
        cmd = ([sys.executable, "run_core_training.py", "--all",
                "--cores", "B", "C", "--system", system,
                "--n-workers", str(per_system_workers),
                "--cache-dir", str(sub_cache_dir)]
               + _base_flags(args))
        print(f"\n{'=' * 70}\n>>> Phase B: launching {system} "
              f"(cache-dir={sub_cache_dir}, n-workers={per_system_workers})\n"
              f"CMD: {' '.join(cmd)}\nLOG: {log_path}\n{'=' * 70}", flush=True)
        log_file = open(log_path, "w", encoding="utf-8")
        proc = subprocess.Popen(cmd, cwd=REPO_ROOT, stdout=log_file, stderr=subprocess.STDOUT)
        procs.append((system, proc, log_file, log_path))

    t0 = time.time()
    failures = []
    for system, proc, log_file, log_path in procs:
        ret = proc.wait()
        log_file.close()
        status = "OK" if ret == 0 else f"FAILED (exit {ret})"
        print(f">>> Phase B: {system} {status} -- see {log_path}", flush=True)
        if ret != 0:
            failures.append((system, ret))
    elapsed = time.time() - t0
    if failures:
        raise RuntimeError(f"Phase B: {failures} failed after {elapsed:.1f}s -- check the logs above")
    print(f">>> Phase B: all {len(systems)} systems done concurrently in {elapsed:.1f}s", flush=True)
    return cache_dirs


def phase_b_train_remaining_sequentially(systems: List[str], n_workers: int,
                                          args: argparse.Namespace) -> None:
    """--no-concurrent-tail escape hatch: old sequential behavior, full
    workers each, writing straight into the shared default cache dir (no
    merge needed -- nothing else is writing there at the same time)."""
    for system in systems:
        cmd = ([sys.executable, "run_core_training.py", "--all",
                "--cores", "B", "C", "--system", system,
                "--n-workers", str(n_workers)]
               + _base_flags(args))
        _run_blocking(cmd, f"Phase B (sequential): train {system}")


def phase_c_merge_caches(main_cache_dir: Path, cache_dirs: List[Path]) -> None:
    """Union-merge each temporary cache dir's cache_index.json + .pkl
    files into the main cache dir. A plain union is safe here (not the
    "prefer newer" conflict resolution this session's earlier, unrelated
    cache-pull needed for genuinely overlapping stale/fresh entries):
    Core B/C's cache config always includes "system", so entries from
    different systems can never collide with each other or with the base
    system's own -- a duplicate key showing up below would mean something
    violated that assumption, not an expected merge conflict to
    resolve."""
    main_cache_dir.mkdir(parents=True, exist_ok=True)
    index_path = main_cache_dir / "cache_index.json"
    index = json.loads(index_path.read_text()) if index_path.exists() else {}

    merged_count = 0
    for cache_dir in cache_dirs:
        sub_index_path = cache_dir / "cache_index.json"
        if not sub_index_path.exists():
            print(f"  (no cache_index.json in {cache_dir}, nothing to merge)")
            continue
        sub_index = json.loads(sub_index_path.read_text())
        for key, meta in sub_index.items():
            if key in index:
                print(f"  WARNING: key {key} ({meta.get('name')}) already in the main "
                      f"index -- skipping rather than overwriting (should not happen "
                      f"by construction; investigate if this fires)")
                continue
            pkl_src = cache_dir / f"{key}.pkl"
            if not pkl_src.exists():
                print(f"  WARNING: {pkl_src} referenced by {sub_index_path} but missing -- skipping")
                continue
            shutil.copy2(pkl_src, main_cache_dir / f"{key}.pkl")
            index[key] = meta
            merged_count += 1

    index_path.write_text(json.dumps(index, indent=2))
    print(f">>> Phase C: merged {merged_count} cache entries into {index_path}")

    for cache_dir in cache_dirs:
        shutil.rmtree(cache_dir, ignore_errors=True)
    print(f">>> Phase C: removed {len(cache_dirs)} temporary cache dirs")


def phase_d_figures_sequential(systems: List[str], n_workers: int, args: argparse.Namespace) -> None:
    """Figure/report generation, strictly sequential per system -- see
    module docstring for why this can't be parallelized (shared,
    non-per-system output paths). By this point every system's Core data
    is already cached, so each pass here is fast: the core-training
    subprocess this internally launches is a near-total cache hit, the
    Objective-4 benchmark computes once (on whichever system runs first
    here) and is a cache hit for the rest, and the only genuinely new
    work per pass is that system's own figures."""
    if args.skip_figures:
        return
    for system in systems:
        cmd = ([sys.executable, "run_thesis_experiments.py", "--all",
                "--system", system, "--n-workers", str(n_workers)]
               + _base_flags(args))
        _run_blocking(cmd, f"Phase D: figures/report for {system}")


def phase_e_reverse_power_flow(args: argparse.Namespace) -> None:
    """Objective 1's reverse-power-flow analysis
    (experiments/exp_reverse_power_flow.py) -- genuinely real (solves
    actual AC power flow via pandapower at each PV level, no synthetic
    fallback), but was never wired into any orchestrator until now; only
    reachable via a manual `python -m experiments.exp_reverse_power_flow`.
    Runs once (it already internally loops over all 3 systems), not
    per-system like Phase D, and degrades gracefully -- not by crashing
    the whole campaign -- if pandapower isn't installed, matching how
    every other pandapower-dependent path in this codebase treats it as
    optional."""
    if args.skip_reverse_power_flow:
        return
    print(f"\n{'=' * 70}\n>>> Phase E: Objective 1 reverse-power-flow analysis\n{'=' * 70}", flush=True)
    t0 = time.time()
    try:
        from experiments.exp_reverse_power_flow import write_report
        path = write_report()
        print(f">>> Phase E: wrote {path} in {time.time() - t0:.1f}s", flush=True)
    except RuntimeError as e:
        print(f">>> Phase E: skipped -- {e}", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--systems", type=str, default="8bus,14bus,39bus",
                        help="Comma-separated systems to run; the first is the 'base' "
                             "system trained alone in Phase A, the rest run concurrently "
                             "in Phase B")
    parser.add_argument("--n-workers", type=int, required=True,
                        help="Total CPU workers available on this pod")
    parser.add_argument("--fast", action="store_true", help="Fast mode")
    parser.add_argument("--pv", type=float, default=None)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--fault-type", type=str, default=None)
    parser.add_argument("--no-concurrent-tail", action="store_true",
                        help="Run systems after the base one sequentially with full "
                             "workers each, instead of concurrently with split workers")
    parser.add_argument("--skip-figures", action="store_true",
                        help="Skip Phase D (figure/report generation) -- just get "
                             "Core-level training data cached for every system")
    parser.add_argument("--skip-reverse-power-flow", action="store_true",
                        help="Skip Phase E (Objective 1's reverse-power-flow analysis)")
    args = parser.parse_args()

    systems = [s.strip() for s in args.systems.split(",") if s.strip()]
    if not systems:
        parser.error("--systems must list at least one system")
    base_system, remaining = systems[0], systems[1:]

    cache_dir = DEFAULT_CACHE_DIR
    status_path = _status_path(cache_dir)
    if status_path.exists():
        prior = json.loads(status_path.read_text())
        print(f">>> Found existing {status_path} (last phase: {prior.get('last_phase')}, "
              f"complete={prior.get('complete')}). Re-running is safe -- every phase below "
              f"re-invokes run_core_training.py's content-addressed cache, so already-cached "
              f"work replays as a near-instant cache hit instead of retraining.", flush=True)

    t_start = time.time()
    phase_a_train_base_system(base_system, args.n_workers, cache_dir, args)
    _write_status(cache_dir, "phase_a", system=base_system)

    if remaining:
        if args.no_concurrent_tail:
            phase_b_train_remaining_sequentially(remaining, args.n_workers, args)
        else:
            cache_dirs = phase_b_train_remaining_concurrently(remaining, args.n_workers, cache_dir, args)
            phase_c_merge_caches(cache_dir, cache_dirs)
        _write_status(cache_dir, "phase_b_c", systems=remaining)

    phase_d_figures_sequential(systems, args.n_workers, args)
    _write_status(cache_dir, "phase_d")
    phase_e_reverse_power_flow(args)
    _write_status(cache_dir, "phase_e")

    total = time.time() - t_start
    _write_status(cache_dir, "campaign_complete", total_seconds=total)
    print(f"\n{'=' * 70}\nCAMPAIGN COMPLETE in {total / 60:.1f} min ({total:.0f}s)\n{'=' * 70}")


if __name__ == "__main__":
    main()
