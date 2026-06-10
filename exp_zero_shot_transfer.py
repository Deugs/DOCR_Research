"""
exp_zero_shot_transfer.py
─────────────────────────
Figure 25 — Zero-Shot Transfer: 8-Bus Policy → 30-Bus System.

Protocol
--------
1. Train MAPPO (with communication) on IEEE 8-bus at 100% PV.
2. Zero-shot: directly evaluate the trained 8-bus policy on IEEE 30-bus
   without any weight updates.
3. Fine-tune: continue training the 8-bus policy on 30-bus for
   `finetune_steps` steps.
4. Scratch: train a fresh MAPPO on 30-bus from random initialisation.
5. Report: OF and CTI violations for all three conditions vs. training steps.

Transfer efficiency
-------------------
  TE = (zero_shot_OF - scratch_final_OF) / (source_OF - scratch_final_OF)
  TE ∈ [0,1]; higher = better transfer.

The observation space is the same (OBS_DIM = 10) for both systems.
The shared actor accepts any n_agents × 10 observation matrix, so
transferring to a larger system (30-bus, 42 agents) is architecturally
valid — the actor processes agents independently.

Outputs
-------
  outputs/fig25_zero_shot_transfer.npz
  outputs/fig25_zero_shot_transfer.png
"""

import numpy as np
import os, sys, copy

sys.path.insert(0, os.path.dirname(__file__))
os.makedirs("outputs", exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from network.network_data import get_8bus, get_30bus
from marl.marl_env import DynamicDOCREnv
from marl.marl_agents import MAPPOMARLTrainer
from network.relay_physics import check_constraints


# ── Training helpers ────────────────────────────────────────────────────────

def _make_trainer(net, seed: int, hidden: int = 32) -> MAPPOMARLTrainer:
    env = DynamicDOCREnv(net, seed=seed, use_pandapower=False,
                          fault_type="3ph", curriculum_stage=5)   # 100% PV
    return MAPPOMARLTrainer(env, use_comm=True,
                             hidden_actor=hidden, hidden_critic=64, seed=seed)


def _eval_policy(trainer: MAPPOMARLTrainer, net, n_episodes: int = 20,
                 seed: int = 99) -> dict:
    """Evaluate a trainer's current actor on `net` for n_episodes."""
    env = DynamicDOCREnv(net, seed=seed, use_pandapower=False,
                          fault_type="3ph", curriculum_stage=5)
    rng = np.random.default_rng(seed)

    of_list, cti_viol_list = [], []
    for ep in range(n_episodes):
        obs = env.reset(seed=seed + ep)
        done = False
        ep_of = []
        while not done:
            actions = np.zeros((env.n_agents, env.ACT_DIM))
            for i in range(env.n_agents):
                act, _ = trainer.actor.sample(obs[i], rng)
                actions[i] = act
            obs, _, done, info = env.step(actions)
            ep_of.append(info.get("OF", float("inf")))
        of_list.append(float(np.min(ep_of)))

        TMS, PS = env.get_state()
        c = check_constraints(net, TMS, PS)
        cti_viol_list.append(c["cti_violations"])

    return {
        "mean_OF":   float(np.mean(of_list)),
        "std_OF":    float(np.std(of_list)),
        "cti_viols": float(np.mean(cti_viol_list)),
    }


# ── Main experiment ─────────────────────────────────────────────────────────

def main(source_steps: int = 8000,
         finetune_steps: int = 4000,
         scratch_steps: int = 12000,
         n_eval_ep: int = 15,
         seed: int = 42):

    net8  = get_8bus()
    net30 = get_30bus()

    print("Figure 25 — Zero-Shot Transfer Experiment")
    print(f"  Source: {net8.name}  ({net8.n_relays} relays)")
    print(f"  Target: {net30.name} ({net30.n_relays} relays)")
    print()

    # ── 1. Train source policy on 8-bus ──────────────────────────────────────
    print(f"  [1/4] Training source policy on 8-bus ({source_steps} steps)...")
    src_trainer = _make_trainer(net8, seed=seed)
    src_result  = src_trainer.train(total_steps=source_steps, verbose=False)
    source_eval = _eval_policy(src_trainer, net8, n_eval_ep, seed=seed+1)
    print(f"        Source OF = {source_eval['mean_OF']:.4f} ± {source_eval['std_OF']:.4f}")

    # ── 2. Zero-shot evaluation on 30-bus ─────────────────────────────────────
    print("  [2/4] Zero-shot evaluation on 30-bus (no weight updates)...")
    zs_eval = _eval_policy(src_trainer, net30, n_eval_ep, seed=seed+2)
    print(f"        Zero-shot OF = {zs_eval['mean_OF']:.4f}  "
          f"CTI violations = {zs_eval['cti_viols']:.1f}")

    # ── 3. Fine-tune on 30-bus (warm-start from 8-bus weights) ───────────────
    print(f"  [3/4] Fine-tuning on 30-bus ({finetune_steps} steps, warm-start)...")
    # Build a 30-bus trainer and copy the shared actor weights from 8-bus
    ft_trainer = _make_trainer(net30, seed=seed)
    # Copy actor params (same OBS_DIM=10 and ACT_DIM=2 → weights are compatible)
    ft_trainer.actor.set_all_params(src_trainer.actor.get_all_params())

    ft_history = []
    # We train in chunks to record intermediate evaluations
    chunk = finetune_steps // 5
    for chunk_i in range(5):
        ft_trainer.train(total_steps=chunk, verbose=False)
        ev = _eval_policy(ft_trainer, net30, n_eval_ep // 2, seed=seed+10+chunk_i)
        ft_history.append(ev["mean_OF"])
    ft_eval = _eval_policy(ft_trainer, net30, n_eval_ep, seed=seed+20)
    print(f"        Fine-tune final OF = {ft_eval['mean_OF']:.4f}  "
          f"CTI violations = {ft_eval['cti_viols']:.1f}")

    # ── 4. Scratch training on 30-bus ─────────────────────────────────────────
    print(f"  [4/4] Training from scratch on 30-bus ({scratch_steps} steps)...")
    sc_trainer = _make_trainer(net30, seed=seed + 7)
    sc_history = []
    chunk_s = scratch_steps // 5
    for chunk_i in range(5):
        sc_trainer.train(total_steps=chunk_s, verbose=False)
        ev = _eval_policy(sc_trainer, net30, n_eval_ep // 2, seed=seed+30+chunk_i)
        sc_history.append(ev["mean_OF"])
    sc_eval = _eval_policy(sc_trainer, net30, n_eval_ep, seed=seed+40)
    print(f"        Scratch final OF    = {sc_eval['mean_OF']:.4f}  "
          f"CTI violations = {sc_eval['cti_viols']:.1f}")

    # ── Transfer efficiency ────────────────────────────────────────────────────
    denom = source_eval["mean_OF"] - sc_eval["mean_OF"]
    if abs(denom) > 1e-6:
        te = (zs_eval["mean_OF"] - sc_eval["mean_OF"]) / denom
    else:
        te = float("nan")
    print(f"\n  Transfer Efficiency = {te:.3f}")

    # ── Save results ──────────────────────────────────────────────────────────
    np.savez(
        "outputs/fig25_zero_shot_transfer.npz",
        source_of        = np.array([source_eval["mean_OF"]]),
        zeroshot_of      = np.array([zs_eval["mean_OF"]]),
        zeroshot_viols   = np.array([zs_eval["cti_viols"]]),
        finetune_history = np.array(ft_history),
        finetune_final   = np.array([ft_eval["mean_OF"]]),
        scratch_history  = np.array(sc_history),
        scratch_final    = np.array([sc_eval["mean_OF"]]),
        transfer_efficiency = np.array([te]),
        finetune_steps   = np.array([finetune_steps]),
        scratch_steps    = np.array([scratch_steps]),
    )
    print("  Saved: outputs/fig25_zero_shot_transfer.npz")

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    fig.suptitle("Fig 25 — Zero-Shot Transfer: 8-Bus → 30-Bus", fontsize=11)

    # Left: learning curve comparison
    ax = axes[0]
    ft_x = np.linspace(0, finetune_steps, len(ft_history))
    sc_x = np.linspace(0, scratch_steps,  len(sc_history))
    ax.axhline(zs_eval["mean_OF"], color="#f7c948", ls="--", lw=1.5,
               label=f"Zero-shot (no training): {zs_eval['mean_OF']:.3f}")
    ax.plot(ft_x, ft_history, color="#39d353", lw=1.8,
            label=f"Fine-tune (warm-start): {ft_eval['mean_OF']:.3f}")
    ax.plot(sc_x, sc_history, color="#2563eb", lw=1.8, ls="-.",
            label=f"Scratch: {sc_eval['mean_OF']:.3f}")
    ax.set_xlabel("Training Steps on 30-Bus", fontsize=8)
    ax.set_ylabel("Objective Function (s)", fontsize=8)
    ax.set_title("Objective vs. Steps on Target System", fontsize=9)
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # Right: bar chart of final values
    ax2 = axes[1]
    labels  = ["Source\n(8-bus)", "Zero-shot\n(30-bus)", "Fine-tune\n(30-bus)", "Scratch\n(30-bus)"]
    values  = [source_eval["mean_OF"], zs_eval["mean_OF"],
               ft_eval["mean_OF"],     sc_eval["mean_OF"]]
    colours = ["#a371f7", "#f7c948", "#39d353", "#2563eb"]
    bars = ax2.bar(labels, values, color=colours, edgecolor="white", width=0.5)
    for bar, val in zip(bars, values):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=8)
    ax2.set_ylabel("Final OF (s)", fontsize=8)
    ax2.set_title(f"Final Performance Comparison\n(Transfer Efficiency = {te:.2f})",
                  fontsize=9)
    ax2.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig("outputs/fig25_zero_shot_transfer.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: outputs/fig25_zero_shot_transfer.png")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--source_steps",   type=int, default=8000)
    p.add_argument("--finetune_steps", type=int, default=4000)
    p.add_argument("--scratch_steps",  type=int, default=12000)
    p.add_argument("--seed",           type=int, default=42)
    args = p.parse_args()
    main(args.source_steps, args.finetune_steps, args.scratch_steps, seed=args.seed)
