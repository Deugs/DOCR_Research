"""
test_core_training_parallel.py
───────────────────────────────
Verifies the run_core_training.py parallelism restructuring (Cores G/H/I/K
now pool their eval sweeps / independent trainings via _run_pool's new
initializer/initargs support, instead of running fully sequentially).

Tests:
1. A trained CoreEvalBridge round-trips through pickle and still works
   post-unpickle -- the assumption the whole "train once in the main
   process, hand the trained bridge to pool workers" design depends on.
2. _run_pool's new shared-state initializer mechanism (initializer/
   initargs, chained after the existing BLAS-thread-pinning init) is
   picklable and dispatches correctly -- verified two ways:
   (a) always, via the sequential (n_workers=1) path against real
       run_core_training helpers, which exercises the exact same
       initializer/job-routing code without spawning a new process;
   (b) when RAM allows (see _enough_ram_for_pool_tests below), via a
       real multi-worker mp.Pool, confirming the initializer this file
       passes is genuinely picklable under Windows' spawn start method
       -- this is exactly the bug _combined_worker_init's rewrite fixed
       (mp.Pool's initializer must be a top-level function, not a
       closure; the original _run_pool had a `def _combined_init` nested
       inside itself, which fails with "Can't get local object" the
       moment n_workers>1 for ANY caller).
3. The GPU-disable worker init (_disable_gpu_in_worker) flips
   marl.marl_agents._GPU_AVAILABLE to False -- verified via the
   sequential path (same-process monkeypatch, so no additional import
   cost) plus, when RAM allows, inside a real pool worker process.

A note on why (b)/pool-worker variants are RAM-gated rather than always
run: every one of run_core_training.py's own worker functions needs each
worker process to import run_core_training itself, which transitively
pulls in torch + pandapower + matplotlib + seaborn + ipywidgets (a very
heavy chain). On a memory-constrained dev box, spawning even 2-4
concurrent workers that each need to load that whole chain can push the
machine into a genuine crash-respawn spiral: a worker fails an import
with MemoryError, multiprocessing.Pool sees the worker died and spawns a
replacement to maintain pool size, the replacement ALSO fails (system
memory is now even more pressured), and so on -- this was observed
directly during development (worker processes numbered past 30 before
the OS's paging file was exhausted, for a pool that only ever asked for
4). None of that is a correctness bug in this file's pooling logic (a
fully isolated, dependency-free reproduction of the exact
_combined_worker_init pattern was verified separately to work correctly
under real mp.Pool) -- it's a resource ceiling specific to running many
heavy-import Python processes concurrently on one machine. The Runpod
pods this project's real campaigns run on (100+ GB RAM, already proven
to sustain --n-workers 32 for Cores A-F all session) don't hit this
ceiling, so the pool-worker variants below run for real there and on any
other adequately-resourced machine/CI runner; they skip cleanly (not a
failure) wherever available RAM looks too tight to risk it.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pickle
import pytest

import run_core_training as rct

# Rough, deliberately conservative threshold -- each heavy-import worker
# process observed to need on the order of 1+ GB just to complete its
# import chain before doing any real work; require enough headroom for a
# small pool of them plus normal OS/IDE overhead before risking it.
_MIN_AVAILABLE_GB_FOR_POOL_TESTS = 6.0


def _enough_ram_for_pool_tests() -> bool:
    try:
        import psutil
        return psutil.virtual_memory().available / (1024 ** 3) >= _MIN_AVAILABLE_GB_FOR_POOL_TESTS
    except Exception:
        # No psutil, or the check itself failed -- don't risk it.
        return False


_skip_if_low_ram = pytest.mark.skipif(
    not _enough_ram_for_pool_tests(),
    reason=f"available RAM below {_MIN_AVAILABLE_GB_FOR_POOL_TESTS}GB threshold; "
           f"skipping real multi-worker Pool spawn to avoid the heavy-import "
           f"crash-respawn spiral documented in this file's module docstring",
)


def test_core_eval_bridge_picklable():
    """A trained CoreEvalBridge must survive a pickle round-trip and still
    work afterward -- this is what lets the main process train once and
    hand the bridge to pool workers via _run_pool's initargs, instead of
    each worker retraining its own copy. Single-process only (pickle
    round-trip in memory, no Pool spawn), so this is always safe to run."""
    from core_eval_bridge import CoreEvalBridge

    bridge = CoreEvalBridge(seed=1, fast_mode=True)
    base = bridge._get_base_result()

    data = pickle.dumps(bridge)
    bridge2 = pickle.loads(data)

    assert bridge2._get_base_result().best_of == base.best_of

    r = bridge2.run_evaluation(58, season="summer")
    assert "of" in r.raw_data


class _FakeResult:
    def __init__(self, raw_data):
        self.raw_data = raw_data


class _FakeBridge:
    """Deterministic stand-in for CoreEvalBridge: run_evaluation's output
    depends only on (fig_num, kwargs), not on any trained state, so
    identical inputs must produce identical outputs regardless of which
    worker process (or none) executes the call."""
    def __init__(self, tag):
        self.tag = tag

    def run_evaluation(self, fig_num, **kwargs):
        return _FakeResult({"tag": self.tag, "fig_num": fig_num, **kwargs})


def _fake_job(job):
    job_key, bridge_name, fig_num, kwargs = job
    bridge = rct._WORKER_BRIDGES[bridge_name]
    r = bridge.run_evaluation(fig_num, **kwargs)
    return job_key, r.raw_data


def _build_fake_jobs():
    bridges = {"main": _FakeBridge("main"), "alt": _FakeBridge("alt")}
    jobs = [
        (("a", i), "main", 99, {"x": i}) for i in range(6)
    ] + [
        (("b", i), "alt", 99, {"x": i}) for i in range(6)
    ]
    return bridges, jobs


def test_run_pool_shared_initializer_sequential_path():
    """_run_pool's initializer/initargs correctly seed _WORKER_BRIDGES and
    route each job to the right fake bridge -- via the sequential
    (n_workers=1) path, which runs _init_core_bridge_workers/_fake_job in
    THIS process (no new process spawned, no additional import cost), so
    always safe to run regardless of available RAM."""
    bridges, jobs = _build_fake_jobs()
    seq = dict(rct._run_pool(jobs, _fake_job, 1,
                              initializer=rct._init_core_bridge_workers,
                              initargs=(bridges,)))

    assert len(seq) == 12
    assert seq[("a", 3)] == {"tag": "main", "fig_num": 99, "x": 3}
    assert seq[("b", 5)] == {"tag": "alt", "fig_num": 99, "x": 5}


@_skip_if_low_ram
def test_run_pool_shared_initializer_matches_real_pool():
    """Same assertion as the sequential-path test above, but dispatched
    through a real mp.Pool(4) -- confirms _init_core_bridge_workers (and,
    transitively, _combined_worker_init) is genuinely picklable under
    Windows' spawn start method, not just importable in-process. This is
    exactly what caught the original bug: a nested `_combined_init`
    closure inside _run_pool raised "Can't get local object" the instant
    n_workers>1, for every caller (A-F included), not only G/H/I/K."""
    bridges, jobs = _build_fake_jobs()
    seq = dict(rct._run_pool(jobs, _fake_job, 1,
                              initializer=rct._init_core_bridge_workers,
                              initargs=(bridges,)))

    bridges2, jobs2 = _build_fake_jobs()
    par = dict(rct._run_pool(jobs2, _fake_job, 4,
                              initializer=rct._init_core_bridge_workers,
                              initargs=(bridges2,)))

    assert seq == par


def _gpu_flag_job(_):
    import marl.marl_agents as ma
    return None, ma._GPU_AVAILABLE


def test_disable_gpu_in_worker_sequential_path():
    """_disable_gpu_in_worker must flip marl.marl_agents._GPU_AVAILABLE to
    False -- verified via the sequential path (monkeypatches the already-
    imported module in this same process, so no additional heavy import
    is triggered), always safe to run."""
    results = list(rct._run_pool([1], _gpu_flag_job, 1,
                                  initializer=rct._disable_gpu_in_worker))
    assert results[0][1] is False


@_skip_if_low_ram
def test_disable_gpu_in_worker_real_pool_workers():
    """Same assertion, but inside real pool worker processes -- confirms
    the monkeypatch (run once per worker via _init_core_bridge_workers or
    directly for Core I's training pool) takes effect in a fresh
    interpreter that has to import marl.marl_agents from scratch, not
    just in a process that already had it imported."""
    results = list(rct._run_pool([1, 2, 3], _gpu_flag_job, 3,
                                  initializer=rct._disable_gpu_in_worker))
    flags = [flag for _, flag in results]
    assert flags == [False, False, False]


def test_max_safe_gpu_workers_sane_bounds():
    """_max_safe_gpu_workers() must return 0 whenever GPU use hasn't been
    explicitly opted into (DOCR_USE_GPU=1) regardless of whether CUDA
    hardware is physically present (see marl/marl_agents.py's
    _GPU_AVAILABLE policy -- auto-enabling GPU was removed after being
    measured as a net slowdown for the real backprop training path), and
    must also be 0 when CUDA genuinely isn't available at all. This test
    runs without DOCR_USE_GPU set (the realistic default for a pytest
    run), so 0 is the only correct answer here regardless of this
    machine's actual hardware -- a separate, explicit opt-in test below
    exercises the nonzero/VRAM-bounded path."""
    import marl.marl_agents as ma
    n = rct._max_safe_gpu_workers()
    assert isinstance(n, int) and n >= 0
    assert os.environ.get("DOCR_USE_GPU") != "1", (
        "this test assumes GPU opt-in is NOT set in the test environment")
    assert n == 0, "must be 0 without the explicit DOCR_USE_GPU=1 opt-in"


def _gpu_flag_with_rank_job(_):
    import marl.marl_agents as ma
    import multiprocessing as mp
    identity = mp.current_process()._identity
    rank = identity[0] if identity else 1
    return rank, ma._GPU_AVAILABLE


def test_adaptive_gpu_in_worker_sequential_path():
    """_adaptive_gpu_in_worker(max_gpu_workers) must force _GPU_AVAILABLE
    False once the (1-indexed) worker rank exceeds max_gpu_workers --
    verified via the sequential path (same-process monkeypatch, always
    safe to run). The sequential path's "worker" is rank 1 by convention
    (see _adaptive_gpu_in_worker's identity-less fallback), so
    max_gpu_workers=0 must disable it.

    Deliberately does NOT also assert the "stays enabled within the cap"
    case here: _adaptive_gpu_in_worker is a one-shot per-process
    initializer with no re-enable path by design (matching real usage --
    it only ever runs once per real process, at that process's startup),
    so calling it twice in the same process with different
    max_gpu_workers values would just observe this test's own first call's
    leftover monkeypatch, not a fresh natural state. That case is instead
    verified correctly in test_adaptive_gpu_in_worker_real_pool_caps_correctly,
    where each worker genuinely starts in its own fresh process."""
    results = list(rct._run_pool([1], _gpu_flag_job, 1,
                                  initializer=rct._adaptive_gpu_in_worker, initargs=(0,)))
    assert results[0][1] is False


@_skip_if_low_ram
def test_adaptive_gpu_in_worker_real_pool_caps_correctly():
    """Inside a real 3-worker pool, only the workers whose rank falls
    within max_gpu_workers may retain CUDA -- every worker beyond the cap
    must be deterministically forced to _GPU_AVAILABLE=False regardless
    of whether this machine actually has a GPU (that part is
    deterministic either way); workers within the cap retain the
    machine's natural state, which this test independently confirms
    against marl.marl_agents._GPU_AVAILABLE rather than assuming either
    value. Deliberately NOT torch.cuda.is_available(): since the
    2026-10-22 policy change, _GPU_AVAILABLE also requires the explicit
    DOCR_USE_GPU=1 opt-in (GPU auto-activation was removed after being
    measured as a net slowdown for the real backprop training path on
    this project's own hardware), so it can be False even when CUDA
    hardware is physically present and torch.cuda.is_available() is
    True -- that's the actual contract _adaptive_gpu_in_worker preserves
    or overrides, not raw hardware presence."""
    import marl.marl_agents as ma
    natural_gpu_state = ma._GPU_AVAILABLE

    results = list(rct._run_pool([1, 2, 3], _gpu_flag_with_rank_job, 3,
                                  initializer=rct._adaptive_gpu_in_worker, initargs=(1,)))
    by_rank = dict(results)
    assert set(by_rank.keys()) == {1, 2, 3}
    assert by_rank[1] is natural_gpu_state, "rank within the cap should keep the natural GPU state"
    assert by_rank[2] is False, "rank beyond the cap must be forced off"
    assert by_rank[3] is False, "rank beyond the cap must be forced off"


@_skip_if_low_ram
def test_core_k_season_sweep_parallel_matches_sequential():
    """Real end-to-end check of the exact pattern Cores G/H/K all share:
    train one real (fast-mode) base policy, then dispatch the same
    eval-sweep job list through _run_pool at n_workers=1 vs n_workers=2 --
    across a real process boundary this time, not the fake bridge above.
    Bypasses run_core_K()'s own cache.get()/put() wrapper (keyed only on
    seed/config, not n_workers) by rebuilding its job list directly, so
    the two dispatches can't just both return one cached result. Slow
    (trains a real base policy, ~2-4 min) in addition to being RAM-gated."""
    from core_eval_bridge import CoreEvalBridge

    bridge = CoreEvalBridge(seed=7, fast_mode=True)
    bridge._get_base_result()  # trains the shared base policy once
    seasons = ["spring", "summer", "autumn", "winter"]
    jobs = [(("season", s), "main", 58, dict(season=s)) for s in seasons]

    seq = dict(rct._run_pool(jobs, rct._core_bridge_eval_job, 1,
                              initializer=rct._init_core_bridge_workers,
                              initargs=({"main": bridge},)))
    par = dict(rct._run_pool(jobs, rct._core_bridge_eval_job, 2,
                              initializer=rct._init_core_bridge_workers,
                              initargs=({"main": bridge},)))

    assert seq.keys() == par.keys()
    for key in seq:
        assert seq[key] == par[key], f"mismatch at {key}: {seq[key]} vs {par[key]}"
