#!/usr/bin/env python3
"""
benchmark_gpu_rewrite.py
────────────────────────
Re-benchmark CPU vs GPU for the "move everything to torch" rewrite
(2026-10-26 -- 2026-11-02), on whatever machine this is actually run on.

Why this script exists
-----------------------
This repo's own history has reversed its GPU verdict twice already:
  - ES-era MAPPO actor update:        GPU 1.32x-1.58x FASTER  (2026-10-21)
  - real-backprop MAPPO actor update: GPU 0.45x-0.67x SLOWER  (2026-10-22)
  - GA-SQP's population fitness eval: GPU 0.07x-0.63x SLOWER, every tested
    scale (pop_size 60-100)           (2026-10-22, still true today)
Both reversals trace to the same cause: this project's per-call compute is
small relative to GPU's fixed kernel-launch/host-device-transfer overhead.
The 2026-10-26 "move everything to torch" rewrite (relay_physics.py,
ga_sqp.py, metaheuristics.py, safe_rl.py, marl_env.py) made GPU *available*
and *correct* everywhere -- it did NOT, and should not be assumed to, make
GPU *faster*. Only re-measuring on the actual target hardware answers that.

Usage
-----
    python scripts/benchmark_gpu_rewrite.py                  # all benchmarks
    python scripts/benchmark_gpu_rewrite.py --skip-metaheuristic  # faster
    python scripts/benchmark_gpu_rewrite.py --system 39bus

Requires a CUDA-capable GPU and torch built with CUDA support. If none is
available, every GPU column prints "N/A" and only the CPU numbers are
measured (still useful as a baseline for comparison against a GPU machine's
numbers later).

This script does NOT require DOCR_USE_GPU=1 to be set -- every class this
script benchmarks (SharedGaussianActor, MAPPOCentralCritic,
CentralizedGaussianActor, CentralizedCritic) and function
(objective_function_batch) accepts an explicit `device=` argument that
overrides the project's default opt-in policy, so CPU and GPU can both be
measured within one process, one invocation.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from network.network_data import get_8bus, get_14bus, get_39bus
from network.relay_physics import objective_function_batch, TMS_MIN, TMS_MAX

SYSTEMS = {"8bus": get_8bus, "14bus": get_14bus, "39bus": get_39bus}


def _fmt_speedup(cpu_s: float, gpu_s: float | None) -> str:
    if gpu_s is None:
        return "N/A"
    ratio = cpu_s / gpu_s
    tag = "faster" if ratio > 1 else "slower"
    return f"{ratio:.3f}x ({tag})"


def _cuda_available() -> bool:
    try:
        return torch.cuda.is_available()
    except Exception:
        return False


def bench_relay_physics_batch(system: str, pop_size: int = 100, n_calls: int = 50) -> dict:
    """objective_function_batch(): the function GA-SQP's fitness_batch()
    and (post-2026-10-26) HWOA/HHO/JAYA/EEO's per-iteration fitness
    evaluation route through. This is the single most GPU-relevant
    function in the whole rewrite -- batches a whole population's fitness
    evaluation into one call."""
    net = SYSTEMS[system]()
    n = net.n_relays
    rng = np.random.default_rng(0)
    ps_min, ps_max = net.ps_bounds()
    TMS_batch = rng.uniform(TMS_MIN, TMS_MAX, (pop_size, n))
    PS_batch = rng.uniform(ps_min, ps_max, (pop_size, n))

    t0 = time.perf_counter()
    for _ in range(n_calls):
        objective_function_batch(net, TMS_batch, PS_batch, device="cpu")
    cpu_s = (time.perf_counter() - t0) / n_calls

    gpu_s = None
    if _cuda_available():
        # Warm up (first CUDA call pays one-time context/kernel-compile cost)
        objective_function_batch(net, TMS_batch, PS_batch, device="cuda")
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_calls):
            objective_function_batch(net, TMS_batch, PS_batch, device="cuda")
        torch.cuda.synchronize()
        gpu_s = (time.perf_counter() - t0) / n_calls

    return {"cpu_s": cpu_s, "gpu_s": gpu_s, "pop_size": pop_size, "n_relays": n}


def bench_actor_critic_update(system: str, batch_t: int = 64) -> dict:
    """SharedGaussianActor.update()/MAPPOCentralCritic.update(): one real
    PPO-clip / MSE backprop step. This is what EXPERIMENT_GUIDE.md section
    2.2.4 already benchmarked as 0.45x-0.67x (slower) on this project's
    GTX 1050 -- re-run here on whatever GPU is actually present."""
    from marl.marl_agents import SharedGaussianActor, MAPPOCentralCritic

    net = SYSTEMS[system]()
    n = net.n_relays
    obs_dim, act_dim = 14, 2
    rng = np.random.default_rng(0)
    obs_batch = rng.normal(size=(batch_t, obs_dim)).astype(np.float32)
    act_batch = rng.uniform(-1, 1, (batch_t, act_dim)).astype(np.float32)
    adv_batch = rng.normal(size=batch_t).astype(np.float32)
    old_lp_batch = rng.normal(size=batch_t).astype(np.float32)

    n_calls = 20

    actor_cpu = SharedGaussianActor(obs_dim, act_dim, seed=0, device="cpu")
    t0 = time.perf_counter()
    for _ in range(n_calls):
        actor_cpu.update(obs_batch, act_batch, adv_batch, old_lp_batch)
    actor_cpu_s = (time.perf_counter() - t0) / n_calls

    actor_gpu_s = None
    if _cuda_available():
        actor_gpu = SharedGaussianActor(obs_dim, act_dim, seed=0, device="cuda")
        actor_gpu.update(obs_batch, act_batch, adv_batch, old_lp_batch)  # warm up
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_calls):
            actor_gpu.update(obs_batch, act_batch, adv_batch, old_lp_batch)
        torch.cuda.synchronize()
        actor_gpu_s = (time.perf_counter() - t0) / n_calls

    return {"actor_cpu_s": actor_cpu_s, "actor_gpu_s": actor_gpu_s, "n_relays": n}


def bench_metaheuristic(system: str, pop_size: int = 30, n_gen: int = 50) -> dict:
    """One full run_hwoa() call, CPU vs GPU -- the vectorized population
    loop (2026-10-26) plus its batched fitness evaluation. device is
    controlled by DOCR_USE_GPU, not an explicit kwarg here (run_hwoa() has
    no device= parameter -- it goes through _fitness_batch() ->
    objective_function_batch(), which resolves the device via the
    project's opt-in policy), so this benchmark launches two subprocesses
    instead of toggling in-process."""
    import subprocess

    net_fn = f"get_{system}" if system != "8bus" else "get_8bus"
    script = (
        "import time; from network.network_data import " + net_fn + "; "
        "from baselines.metaheuristics import run_hwoa; "
        f"net = {net_fn}(); "
        "t0 = time.perf_counter(); "
        f"run_hwoa(net, n_whales={pop_size}, max_iter={n_gen}, seed=0, use_sqp=False, verbose=False); "
        "print(time.perf_counter() - t0)"
    )
    cpu_out = subprocess.run(
        [sys.executable, "-c", script], cwd=str(Path(__file__).resolve().parent.parent),
        capture_output=True, text=True,
    )
    cpu_s = float(cpu_out.stdout.strip().splitlines()[-1]) if cpu_out.returncode == 0 else None

    gpu_s = None
    if _cuda_available():
        import os
        env = dict(**{**__import__("os").environ, "DOCR_USE_GPU": "1"})
        gpu_out = subprocess.run(
            [sys.executable, "-c", script], cwd=str(Path(__file__).resolve().parent.parent),
            capture_output=True, text=True, env=env,
        )
        gpu_s = float(gpu_out.stdout.strip().splitlines()[-1]) if gpu_out.returncode == 0 else None

    return {"cpu_s": cpu_s, "gpu_s": gpu_s, "pop_size": pop_size, "n_gen": n_gen}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--system", choices=list(SYSTEMS), default=None,
                         help="Run only this system (default: all three)")
    parser.add_argument("--skip-metaheuristic", action="store_true",
                         help="Skip the metaheuristic benchmark (slowest, spawns subprocesses)")
    args = parser.parse_args()

    systems = [args.system] if args.system else list(SYSTEMS)
    cuda_ok = _cuda_available()

    print("=" * 78)
    print("GPU RE-BENCHMARK -- 'move everything to torch' rewrite (2026-10-26)")
    print("=" * 78)
    if cuda_ok:
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        print("No CUDA device detected -- GPU columns will read N/A.")
    print()

    print("-" * 78)
    print("1. objective_function_batch()  (GA-SQP fitness, post-2026-10-26")
    print("   HWOA/HHO/JAYA/EEO fitness)  --  pop_size=100, averaged over 50 calls")
    print("-" * 78)
    print(f"{'System':<10} {'CPU (ms)':>12} {'GPU (ms)':>12} {'Verdict':>22}")
    for sysname in systems:
        r = bench_relay_physics_batch(sysname)
        gpu_ms = f"{r['gpu_s']*1000:.3f}" if r["gpu_s"] is not None else "N/A"
        print(f"{sysname:<10} {r['cpu_s']*1000:>12.3f} {gpu_ms:>12} "
              f"{_fmt_speedup(r['cpu_s'], r['gpu_s']):>22}")
    print()

    print("-" * 78)
    print("2. SharedGaussianActor.update()  --  one PPO-clip backprop step,")
    print("   batch_t=64, averaged over 20 calls (already measured 0.45x-0.67x")
    print("   on a GTX 1050 -- see EXPERIMENT_GUIDE.md section 2.2.4)")
    print("-" * 78)
    print(f"{'System':<10} {'CPU (ms)':>12} {'GPU (ms)':>12} {'Verdict':>22}")
    for sysname in systems:
        r = bench_actor_critic_update(sysname)
        gpu_ms = f"{r['actor_gpu_s']*1000:.3f}" if r["actor_gpu_s"] is not None else "N/A"
        print(f"{sysname:<10} {r['actor_cpu_s']*1000:>12.3f} {gpu_ms:>12} "
              f"{_fmt_speedup(r['actor_cpu_s'], r['actor_gpu_s']):>22}")
    print()

    if not args.skip_metaheuristic:
        print("-" * 78)
        print("3. run_hwoa()  --  one full vectorized metaheuristic run,")
        print(f"   n_whales=30, max_iter=50, use_sqp=False (subprocess-timed,")
        print("   DOCR_USE_GPU=1 vs unset -- slower to run than 1/2 above)")
        print("-" * 78)
        print(f"{'System':<10} {'CPU (s)':>12} {'GPU (s)':>12} {'Verdict':>22}")
        for sysname in systems:
            r = bench_metaheuristic(sysname)
            if r["cpu_s"] is None:
                print(f"{sysname:<10} {'ERROR':>12}")
                continue
            gpu_s_str = f"{r['gpu_s']:.3f}" if r["gpu_s"] is not None else "N/A"
            print(f"{sysname:<10} {r['cpu_s']:>12.3f} {gpu_s_str:>12} "
                  f"{_fmt_speedup(r['cpu_s'], r['gpu_s']):>22}")
        print()

    print("=" * 78)
    print("Read this table, don't assume it. If GPU loses here too, leave")
    print("DOCR_USE_GPU unset and keep scaling via --n-workers (CPU")
    print("multiprocessing) instead -- that's this project's one scaling")
    print("lever that's actually paid off, on every machine tested so far.")
    print("If GPU wins here, it's the first machine where that's been true --")
    print("worth confirming on a second run before trusting it for a full")
    print("campaign.")
    print("=" * 78)


if __name__ == "__main__":
    main()
