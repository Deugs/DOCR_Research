"""
exp_adversarial_robustness.py
─────────────────────────────
Figure 54 — Adversarial Robustness: Perturbed Observation Response.

Implements FGSM and PGD adversarial attacks on relay observations using
finite-difference gradient approximation (no PyTorch/autograd required).

Attack formulation
------------------
  FGSM:  obs_adv = obs + ε · sign( ∇_obs L(π, obs) )
  PGD:   repeated FGSM steps clipped to ε-ball, n_iter=10

  ∇_obs L approximated by finite differences:
    ∂L/∂obs_k ≈ [L(obs + δ·e_k) − L(obs − δ·e_k)] / (2δ)

  where L = −mean(actions·advantages) (policy loss surrogate).

Outputs
-------
  outputs/fig54_adversarial.npz
      epsilon_values   : (n_eps,)
      fgsm_of          : (n_eps,)  mean OF under FGSM
      pgd_of           : (n_eps,)  mean OF under PGD
      clean_of         : scalar
      fgsm_cti         : (n_eps,)
      pgd_cti          : (n_eps,)

  outputs/fig54_adversarial.png
"""

import numpy as np
import os, sys, copy

sys.path.insert(0, os.path.dirname(__file__))
os.makedirs("outputs", exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from network.network_data import get_8bus
from marl.marl_env import DynamicDOCREnv
from marl.marl_agents import MAPPOMARLTrainer
from network.relay_physics import check_constraints


# ── Finite-difference gradient of policy loss w.r.t. obs ─────────────────────

def _policy_loss(actor, obs_batch: np.ndarray) -> float:
    """
    Surrogate policy loss: negative mean action magnitude (proxy).
    Higher action magnitude under perturbation → more disruption.
    """
    total = 0.0
    for obs_i in obs_batch:
        a = actor.mean(obs_i)
        total += float(np.sum(a ** 2))
    return total / max(len(obs_batch), 1)


def _fd_gradient(actor, obs: np.ndarray, delta: float = 1e-3) -> np.ndarray:
    """
    Finite-difference gradient of policy loss w.r.t. a single observation.
    Returns ∇_obs L of shape (obs_dim,).
    """
    grad = np.zeros_like(obs)
    for k in range(len(obs)):
        obs_p = obs.copy(); obs_p[k] += delta
        obs_m = obs.copy(); obs_m[k] -= delta
        lp = _policy_loss(actor, [obs_p])
        lm = _policy_loss(actor, [obs_m])
        grad[k] = (lp - lm) / (2.0 * delta)
    return grad


# ── FGSM ─────────────────────────────────────────────────────────────────────

def fgsm_attack(actor, obs: np.ndarray, epsilon: float) -> np.ndarray:
    """
    Fast Gradient Sign Method.
    obs shape: (n_agents, obs_dim)
    """
    adv = obs.copy()
    for i in range(len(obs)):
        g = _fd_gradient(actor, obs[i])
        adv[i] = obs[i] + epsilon * np.sign(g)
    return adv


# ── PGD ──────────────────────────────────────────────────────────────────────

def pgd_attack(actor, obs: np.ndarray, epsilon: float,
               step_size: float = None, n_iter: int = 10) -> np.ndarray:
    """
    Projected Gradient Descent attack.
    Iteratively applies FGSM with step_size and projects back into ε-ball.
    """
    if step_size is None:
        step_size = epsilon / n_iter

    adv = obs.copy()
    for _ in range(n_iter):
        delta = np.zeros_like(obs)
        for i in range(len(obs)):
            g = _fd_gradient(actor, adv[i])
            delta[i] = step_size * np.sign(g)
        adv = adv + delta
        # Project back to ε-ball
        perturbation = np.clip(adv - obs, -epsilon, epsilon)
        adv = obs + perturbation
    return adv


# ── Evaluation under attack ───────────────────────────────────────────────────

def _eval_under_attack(actor, net, attack_fn, n_episodes: int = 30,
                        seed: int = 0) -> dict:
    env = DynamicDOCREnv(net, seed=seed, use_pandapower=False, fault_type="3ph")
    rng = np.random.default_rng(seed)
    of_list, cti_list = [], []

    for ep in range(n_episodes):
        obs = env.reset(seed=seed + ep)
        done = False
        min_of = float("inf")
        while not done:
            # Apply attack to observations
            adv_obs = attack_fn(obs)
            actions = np.zeros((env.n_agents, env.ACT_DIM))
            for i in range(env.n_agents):
                a, _ = actor.sample(adv_obs[i], rng)
                actions[i] = a
            obs, _, done, info = env.step(actions)
            min_of = min(min_of, info.get("OF", float("inf")))
        of_list.append(min_of)
        TMS, PS = env.get_state()
        c = check_constraints(net, TMS, PS)
        cti_list.append(c["cti_violations"])

    return {
        "mean_OF":   float(np.mean(of_list)),
        "cti_viols": float(np.mean(cti_list)),
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main(train_steps: int = 8000, n_eval: int = 25, seed: int = 42):
    net = get_8bus()

    print("Figure 54 — Adversarial Robustness Experiment")
    print(f"  System: {net.name}  ({net.n_relays} relays)")

    # Train base policy
    print(f"\n  Training base policy ({train_steps} steps)...")
    env     = DynamicDOCREnv(net, seed=seed, use_pandapower=False)
    trainer = MAPPOMARLTrainer(env, use_comm=False,
                                hidden_actor=64, hidden_critic=64, seed=seed)
    trainer.train(total_steps=train_steps, verbose=False)
    actor = trainer.actor

    # Clean (no attack) baseline
    def no_attack(obs): return obs
    print("  Clean evaluation...")
    clean = _eval_under_attack(actor, net, no_attack, n_eval, seed=seed+1)
    print(f"    Clean OF = {clean['mean_OF']:.4f}  CTI viols = {clean['cti_viols']:.1f}")

    # Epsilon sweep
    epsilons = [0.01, 0.03, 0.05, 0.10, 0.20]
    fgsm_of, fgsm_cti = [], []
    pgd_of,  pgd_cti  = [], []

    for eps in epsilons:
        print(f"  ε = {eps:.2f}  ", end="", flush=True)

        fgsm_fn = lambda obs, e=eps: fgsm_attack(actor, obs, e)
        fgsm_r  = _eval_under_attack(actor, net, fgsm_fn, n_eval, seed=seed+2)
        fgsm_of.append(fgsm_r["mean_OF"])
        fgsm_cti.append(fgsm_r["cti_viols"])
        print(f"FGSM OF={fgsm_r['mean_OF']:.4f}  ", end="", flush=True)

        pgd_fn = lambda obs, e=eps: pgd_attack(actor, obs, e, n_iter=5)
        pgd_r  = _eval_under_attack(actor, net, pgd_fn, n_eval, seed=seed+3)
        pgd_of.append(pgd_r["mean_OF"])
        pgd_cti.append(pgd_r["cti_viols"])
        print(f"PGD OF={pgd_r['mean_OF']:.4f}")

    # Save
    np.savez("outputs/fig54_adversarial.npz",
             epsilon_values = np.array(epsilons),
             clean_of       = np.array([clean["mean_OF"]]),
             clean_cti      = np.array([clean["cti_viols"]]),
             fgsm_of        = np.array(fgsm_of),
             fgsm_cti       = np.array(fgsm_cti),
             pgd_of         = np.array(pgd_of),
             pgd_cti        = np.array(pgd_cti))
    print("\n  Saved: outputs/fig54_adversarial.npz")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    fig.suptitle("Fig 54 — Adversarial Robustness under FGSM and PGD", fontsize=11)

    for ax, (title, fgsm_v, pgd_v, clean_v, ylabel) in zip(axes, [
        ("Objective Function",  fgsm_of,  pgd_of,  clean["mean_OF"],  "OF (s)"),
        ("CTI Violations",      fgsm_cti, pgd_cti, clean["cti_viols"],"Mean Violations"),
    ]):
        ax.axhline(clean_v, color="#8b949e", ls="--", lw=1.2, label="Clean (no attack)")
        ax.plot(epsilons, fgsm_v, "o-", color="#f7c948", lw=1.8, label="FGSM")
        ax.plot(epsilons, pgd_v,  "s-", color="#ff6b35", lw=1.8, label="PGD (5 iter)")
        ax.set_xlabel("Perturbation ε", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        ax.set_xticks(epsilons)

    fig.tight_layout()
    fig.savefig("outputs/fig54_adversarial.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: outputs/fig54_adversarial.png")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--train_steps", type=int, default=8000)
    p.add_argument("--n_eval",      type=int, default=25)
    p.add_argument("--seed",        type=int, default=42)
    args = p.parse_args()
    main(args.train_steps, args.n_eval, args.seed)
