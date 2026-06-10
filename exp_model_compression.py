"""
exp_model_compression.py
────────────────────────
Figure 53 — Model Compression & Quantization Impact.

Applies three compression techniques to a trained MAPPO actor and evaluates
performance retention vs. model size reduction:

  (a) INT8 quantisation    : round weights to 8-bit fixed point
  (b) Magnitude pruning    : zero out the smallest 50% of weights
  (c) Knowledge distillation: train a half-sized student MLP to mimic the
                              teacher's outputs on stored rollout data

Metrics per compression
-----------------------
  model_size_kb  : weight storage in kilobytes
  mean_OF        : objective function on 50 test episodes
  cti_violations : mean P/B pair CTI violations per episode
  inference_ms   : mean inference latency over 1000 calls

Outputs
-------
  outputs/fig53_compression.npz
  outputs/fig53_compression.png
"""

import numpy as np
import time
import os, sys, copy

sys.path.insert(0, os.path.dirname(__file__))
os.makedirs("outputs", exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from network.network_data import get_8bus
from marl.marl_env import DynamicDOCREnv
from marl.marl_agents import MAPPOMARLTrainer, SharedGaussianActor, MLP
from network.relay_physics import check_constraints


# ── Helpers ────────────────────────────────────────────────────────────────────

def _size_kb(actor: SharedGaussianActor) -> float:
    """Return approximate weight storage size in kilobytes."""
    return actor.get_all_params().nbytes / 1024.0


def _eval_actor(actor: SharedGaussianActor, net, n_episodes: int = 30,
                seed: int = 77) -> dict:
    """Evaluate an actor (not a full trainer) on net."""
    env = DynamicDOCREnv(net, seed=seed, use_pandapower=False, fault_type="3ph")
    rng = np.random.default_rng(seed)
    of_list, cti_list = [], []

    for ep in range(n_episodes):
        obs = env.reset(seed=seed + ep)
        done = False
        min_of = float("inf")
        while not done:
            actions = np.zeros((env.n_agents, env.ACT_DIM))
            for i in range(env.n_agents):
                a, _ = actor.sample(obs[i], rng)
                actions[i] = a
            obs, _, done, info = env.step(actions)
            min_of = min(min_of, info.get("OF", float("inf")))
        of_list.append(min_of)
        TMS, PS = env.get_state()
        c = check_constraints(net, TMS, PS)
        cti_list.append(c["cti_violations"])

    return {
        "mean_OF":   float(np.mean(of_list)),
        "std_OF":    float(np.std(of_list)),
        "cti_viols": float(np.mean(cti_list)),
    }


def _latency_ms(actor: SharedGaussianActor, obs_dim: int,
                n_calls: int = 1000, seed: int = 0) -> float:
    rng  = np.random.default_rng(seed)
    obs  = rng.standard_normal(obs_dim).astype(np.float32)
    # Warmup
    for _ in range(50):
        actor.mean(obs)
    t0 = time.perf_counter()
    for _ in range(n_calls):
        actor.mean(obs)
    return (time.perf_counter() - t0) / n_calls * 1000.0   # ms


# ── Compression functions ──────────────────────────────────────────────────────

def compress_int8(actor: SharedGaussianActor) -> SharedGaussianActor:
    """
    Symmetric per-tensor INT8 quantisation.
    Scale = max(|w|) / 127; round to nearest integer; rescale back.
    """
    compressed = copy.deepcopy(actor)
    params = compressed.get_all_params()
    scale  = np.max(np.abs(params)) / 127.0
    if scale < 1e-9:
        return compressed
    quantised = np.round(params / scale).astype(np.int8).astype(np.float64) * scale
    compressed.set_all_params(quantised)
    return compressed


def compress_prune(actor: SharedGaussianActor, sparsity: float = 0.50) -> SharedGaussianActor:
    """
    Unstructured magnitude pruning: zero out the smallest `sparsity` fraction
    of weights by absolute value.
    """
    compressed = copy.deepcopy(actor)
    params     = compressed.get_all_params()
    threshold  = np.percentile(np.abs(params), sparsity * 100)
    params[np.abs(params) < threshold] = 0.0
    compressed.set_all_params(params)
    return compressed


def compress_distil(teacher: SharedGaussianActor, obs_dim: int, act_dim: int,
                     hidden: int = 16, n_steps: int = 5000,
                     seed: int = 0) -> SharedGaussianActor:
    """
    Knowledge distillation: train a half-width student MLP to mimic the
    teacher's mean actions on random observations sampled from the unit sphere.

    Loss: MSE between teacher.mean(obs) and student.mean(obs).
    Optimised with SGD + momentum via ES gradient.
    """
    rng     = np.random.default_rng(seed)
    student = SharedGaussianActor(obs_dim, act_dim, hidden=hidden, seed=seed)

    lr    = 5e-3
    sigma = 0.02
    n_pert = 8
    batch  = 32

    # Momentum accumulator
    vel = np.zeros(len(student.get_all_params()))
    mu  = 0.9

    for _ in range(n_steps):
        # Sample batch of random obs (proxy for replay buffer)
        obs_batch = rng.standard_normal((batch, obs_dim)).astype(np.float64)
        teacher_out = np.stack([teacher.mean(o) for o in obs_batch])

        theta = student.get_all_params()
        grads, losses = [], []

        for _ in range(n_pert):
            eps = rng.standard_normal(len(theta)) * sigma
            for sign in (+1.0, -1.0):
                student.set_all_params(theta + sign * eps)
                student_out = np.stack([student.mean(o) for o in obs_batch])
                loss = -float(np.mean((student_out - teacher_out) ** 2))
                grads.append(sign * eps)
                losses.append(loss)

        student.set_all_params(theta)   # restore

        l_arr  = np.array(losses)
        l_norm = (l_arr - l_arr.mean()) / (l_arr.std() + 1e-8)
        grad   = sum(ln * g for ln, g in zip(l_norm, grads))
        grad  /= (2 * n_pert * sigma)

        vel = mu * vel + lr * grad
        student.set_all_params(theta + vel)

    return student


# ── Main ──────────────────────────────────────────────────────────────────────

def main(train_steps: int = 8000, n_eval: int = 30, seed: int = 42):
    net      = get_8bus()
    obs_dim  = 10   # DynamicDOCREnv.OBS_DIM
    act_dim  = 2    # DynamicDOCREnv.ACT_DIM

    print("Figure 53 — Model Compression Experiment")
    print(f"  System: {net.name}  ({net.n_relays} relays)")

    # ── Train baseline ────────────────────────────────────────────────────────
    print(f"\n  [1/5] Training baseline MAPPO ({train_steps} steps)...")
    env     = DynamicDOCREnv(net, seed=seed, use_pandapower=False)
    trainer = MAPPOMARLTrainer(env, use_comm=False,
                                hidden_actor=64, hidden_critic=64, seed=seed)
    trainer.train(total_steps=train_steps, verbose=False)
    base_actor = copy.deepcopy(trainer.actor)

    results = {}

    # ── Baseline ──────────────────────────────────────────────────────────────
    print("  [2/5] Evaluating baseline...")
    base_ev = _eval_actor(base_actor, net, n_eval, seed=seed+1)
    base_ev["size_kb"]   = _size_kb(base_actor)
    base_ev["latency_ms"] = _latency_ms(base_actor, obs_dim)
    results["Baseline (FP64)"] = base_ev
    print(f"        OF={base_ev['mean_OF']:.4f}  size={base_ev['size_kb']:.1f} KB  "
          f"latency={base_ev['latency_ms']:.3f} ms")

    # ── INT8 quantisation ─────────────────────────────────────────────────────
    print("  [3/5] INT8 quantisation...")
    int8_actor = compress_int8(base_actor)
    int8_ev    = _eval_actor(int8_actor, net, n_eval, seed=seed+2)
    int8_ev["size_kb"]    = _size_kb(int8_actor) * 0.25   # 8-bit = ¼ float64
    int8_ev["latency_ms"] = _latency_ms(int8_actor, obs_dim)
    results["INT8 Quantised"] = int8_ev
    print(f"        OF={int8_ev['mean_OF']:.4f}  size={int8_ev['size_kb']:.1f} KB  "
          f"latency={int8_ev['latency_ms']:.3f} ms")

    # ── Pruning 50% ───────────────────────────────────────────────────────────
    print("  [4/5] Magnitude pruning (50% sparsity)...")
    prun_actor = compress_prune(base_actor, sparsity=0.50)
    prun_ev    = _eval_actor(prun_actor, net, n_eval, seed=seed+3)
    # Storage benefit: sparse representation (rough estimate: 50% fewer nonzeros)
    prun_ev["size_kb"]    = _size_kb(prun_actor) * 0.50
    prun_ev["latency_ms"] = _latency_ms(prun_actor, obs_dim)
    results["Pruned (50%)"] = prun_ev
    print(f"        OF={prun_ev['mean_OF']:.4f}  size={prun_ev['size_kb']:.1f} KB  "
          f"latency={prun_ev['latency_ms']:.3f} ms")

    # ── Knowledge distillation ────────────────────────────────────────────────
    print("  [5/5] Knowledge distillation (half-width student)...")
    dist_actor = compress_distil(base_actor, obs_dim, act_dim,
                                  hidden=32, n_steps=3000, seed=seed+4)
    dist_ev    = _eval_actor(dist_actor, net, n_eval, seed=seed+5)
    dist_ev["size_kb"]    = _size_kb(dist_actor)
    dist_ev["latency_ms"] = _latency_ms(dist_actor, obs_dim)
    results["Distilled (½ width)"] = dist_ev
    print(f"        OF={dist_ev['mean_OF']:.4f}  size={dist_ev['size_kb']:.1f} KB  "
          f"latency={dist_ev['latency_ms']:.3f} ms")

    # ── Save ─────────────────────────────────────────────────────────────────
    methods  = list(results.keys())
    of_arr   = np.array([results[m]["mean_OF"]   for m in methods])
    cti_arr  = np.array([results[m]["cti_viols"] for m in methods])
    kb_arr   = np.array([results[m]["size_kb"]   for m in methods])
    lat_arr  = np.array([results[m]["latency_ms"] for m in methods])

    np.savez("outputs/fig53_compression.npz",
             methods=np.array(methods),
             mean_OF=of_arr, cti_violations=cti_arr,
             size_kb=kb_arr, latency_ms=lat_arr)
    print("\n  Saved: outputs/fig53_compression.npz")

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.suptitle("Fig 53 — Model Compression Impact", fontsize=11)

    colors  = ["#2563eb", "#39d353", "#f7c948", "#ff6b35"]
    x       = np.arange(len(methods))
    width   = 0.55

    for ax, (metric, ylabel, arr) in zip(
        axes,
        [("OF",       "Objective Function (s)",     of_arr),
         ("Size",     "Model Size (KB)",             kb_arr),
         ("Latency",  "Inference Latency (ms)",      lat_arr)]
    ):
        bars = ax.bar(x, arr, color=colors, edgecolor="white", width=width)
        for bar, v in zip(bars, arr):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(arr) * 0.02,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=7.5)
        ax.set_xticks(x)
        ax.set_xticklabels(methods, fontsize=7, rotation=15, ha="right")
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_title(metric, fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig("outputs/fig53_compression.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: outputs/fig53_compression.png")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--train_steps", type=int, default=8000)
    p.add_argument("--n_eval",      type=int, default=30)
    p.add_argument("--seed",        type=int, default=42)
    args = p.parse_args()
    main(args.train_steps, args.n_eval, args.seed)
