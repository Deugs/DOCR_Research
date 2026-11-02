#!/usr/bin/env python3
"""
benchmark_vectorized_env.py
────────────────────────────
Phase V10 of the GPU-vectorized-environment redesign (see
elegant-rolling-map.md for full context): the one benchmark this whole
redesign depends on. Nothing in Phases V1-V9 should be presented as a
performance win without this -- this project's own history
(`benchmark_gpu_rewrite.py`'s docstring) has reversed its GPU verdict twice
already on less ambitious changes than running `n_envs` parallel
environments as one batched torch computation.

This dev machine's torch build is CPU-only (confirmed earlier in this
project's history -- two CUDA wheel install attempts both failed on disk
space and were reverted); the first real CUDA numbers for this redesign can
only come from running this script on the actual target remote server.

Usage
-----
    python scripts/benchmark_vectorized_env.py                    # default sweep
    python scripts/benchmark_vectorized_env.py --system 39bus
    python scripts/benchmark_vectorized_env.py --n-envs 1 64 256 1024
    python scripts/benchmark_vectorized_env.py --skip-rollout      # env.step() only, faster

Measures two things, since they can have different crossover points:
  1. Raw `VectorizedDynamicDOCREnv.step()` throughput alone (the batched
     physics/reward/observation core built in Phases V1, V3, V6, V-PF).
  2. Full rollout-collection throughput (`VectorizedMAPPOTrainer.
     _collect_rollout`'s per-step loop: actor.sample_batch + critic.
     value_batch + env.step()) -- the more realistic end-to-end number,
     since actor/critic calls have their own, independently-measured
     GPU-vs-CPU verdict (see `benchmark_gpu_rewrite.py`'s point 2) that
     compounds with the environment's own.

Both are reported as wall-clock per 1,000,000 real (single-environment,
single-step) equivalents -- i.e. `elapsed / (n_envs * n_calls) * 1e6` -- so
numbers at different `n_envs` are directly comparable to each other and to
the single-env `DynamicDOCREnv`/`MAPPOMARLTrainer` baseline this redesign is
meant to outperform.
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

SYSTEMS = {"8bus": get_8bus, "14bus": get_14bus, "39bus": get_39bus}
DEFAULT_N_ENVS_SWEEP = [1, 64, 256, 1024, 4096]


def _cuda_available() -> bool:
    try:
        return torch.cuda.is_available()
    except Exception:
        return False


def _fmt_speedup(cpu_s: float, gpu_s: float | None) -> str:
    if gpu_s is None or gpu_s <= 0:
        return "N/A"
    ratio = cpu_s / gpu_s
    tag = "faster" if ratio > 1 else "slower"
    return f"{ratio:.3f}x ({tag})"


def bench_env_step(system: str, n_envs: int, device: str, n_steps: int = 20) -> float | None:
    """Per-1M-step-equivalents wall clock for VectorizedDynamicDOCREnv.step()
    alone, at this n_envs/device. Returns None if construction fails on this
    device (e.g. requesting cuda with no CUDA device present)."""
    from marl.vectorized_marl_env import VectorizedDynamicDOCREnv

    try:
        net = SYSTEMS[system]()
        env = VectorizedDynamicDOCREnv(net, n_envs=n_envs, max_steps=n_steps + 5,
                                        seed=0, device=torch.device(device))
    except Exception as e:
        print(f"    (skipped {device}, n_envs={n_envs}: {e})")
        return None

    rng = np.random.default_rng(0)
    actions = rng.uniform(-1, 1, (n_steps, n_envs, env.n_agents, env.ACT_DIM)).astype(np.float64)

    env.step(actions[0])  # warm up (first CUDA call pays context/compile cost)
    if device == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for t in range(n_steps):
        env.step(actions[t])
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    return elapsed / (n_envs * n_steps) * 1e6


def bench_rollout(system: str, n_envs: int, device: str, rollout_len: int = 16) -> float | None:
    """Per-1M-step-equivalents wall clock for a full rollout-collection step
    (actor.sample_batch + critic.value_batch + env.step()), the realistic
    end-to-end training throughput number."""
    from marl.vectorized_marl_env import VectorizedDynamicDOCREnv
    from marl.vectorized_marl_agents import VectorizedMAPPOTrainer

    try:
        net = SYSTEMS[system]()
        env = VectorizedDynamicDOCREnv(net, n_envs=n_envs, max_steps=rollout_len + 5,
                                        seed=0, device=torch.device(device))
        trainer = VectorizedMAPPOTrainer(env, hidden_actor=64, hidden_critic=128,
                                          rollout_len=rollout_len, seed=0, device=torch.device(device))
    except Exception as e:
        print(f"    (skipped {device}, n_envs={n_envs}: {e})")
        return None

    obs = env.reset()
    trainer._collect_rollout(obs)  # warm up
    if device == "cuda":
        torch.cuda.synchronize()

    obs = env.reset()
    t0 = time.perf_counter()
    trainer._collect_rollout(obs)
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    return elapsed / (n_envs * rollout_len) * 1e6


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--system", choices=list(SYSTEMS), default="8bus",
                         help="Network system to benchmark (default: 8bus; only 8/14/39-bus "
                              "are supported -- VectorizedDynamicDOCREnv requires real "
                              "per-branch impedance data for its power-flow solver)")
    parser.add_argument("--n-envs", type=int, nargs="+", default=DEFAULT_N_ENVS_SWEEP,
                         help=f"n_envs values to sweep (default: {DEFAULT_N_ENVS_SWEEP})")
    parser.add_argument("--skip-rollout", action="store_true",
                         help="Skip the full rollout-collection benchmark (env.step() only)")
    args = parser.parse_args()

    cuda_ok = _cuda_available()

    print("=" * 86)
    print("VECTORIZED-ENV GPU BENCHMARK -- Phase V10 of the GPU-native vectorized-env redesign")
    print("=" * 86)
    if cuda_ok:
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        print("No CUDA device detected -- GPU columns will read N/A. This dev machine is known")
        print("to be CPU-only torch as of this redesign; run this script on the target remote")
        print("server to get real numbers.")
    print(f"System: {args.system}")
    print()

    print("-" * 86)
    print("1. VectorizedDynamicDOCREnv.step() alone -- wall clock per 1M single-env-step")
    print("   equivalents (elapsed / (n_envs * n_steps) * 1e6)")
    print("-" * 86)
    print(f"{'n_envs':>10} {'CPU (us)':>14} {'GPU (us)':>14} {'Verdict':>22}")
    for n_envs in args.n_envs:
        cpu_t = bench_env_step(args.system, n_envs, "cpu")
        gpu_t = bench_env_step(args.system, n_envs, "cuda") if cuda_ok else None
        if cpu_t is None:
            continue
        gpu_str = f"{gpu_t:.3f}" if gpu_t is not None else "N/A"
        print(f"{n_envs:>10} {cpu_t:>14.3f} {gpu_str:>14} {_fmt_speedup(cpu_t, gpu_t):>22}")
    print()

    if not args.skip_rollout:
        print("-" * 86)
        print("2. Full rollout-collection step (actor.sample_batch + critic.value_batch +")
        print("   env.step()) -- the realistic end-to-end training throughput number")
        print("-" * 86)
        print(f"{'n_envs':>10} {'CPU (us)':>14} {'GPU (us)':>14} {'Verdict':>22}")
        for n_envs in args.n_envs:
            cpu_t = bench_rollout(args.system, n_envs, "cpu")
            gpu_t = bench_rollout(args.system, n_envs, "cuda") if cuda_ok else None
            if cpu_t is None:
                continue
            gpu_str = f"{gpu_t:.3f}" if gpu_t is not None else "N/A"
            print(f"{n_envs:>10} {cpu_t:>14.3f} {gpu_str:>14} {_fmt_speedup(cpu_t, gpu_t):>22}")
        print()

    print("=" * 86)
    print("Read this table, don't assume it. The whole premise of this redesign is that GPU")
    print("loses on small per-call compute (confirmed repeatedly elsewhere in this project) but")
    print("might win once n_envs is large enough to give it real work per dispatch -- these")
    print("numbers are the only way to know whether that premise actually held, and at what")
    print("n_envs the crossover (if any) happens. If GPU still loses even at n_envs=4096, the")
    print("CPU-multiprocessing (--n-workers) path remains this project's one proven scaling")
    print("lever and this redesign's vectorized-env path should not replace it.")
    print("=" * 86)


if __name__ == "__main__":
    main()
