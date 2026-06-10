"""
exp_byzantine_detection.py
──────────────────────────
Figure 20 — Byzantine Agent Detection via Trust Score Evolution.

Runs a trained MAPPO-Comm policy while injecting Byzantine behaviour into
one or two agents and tracks how ReputationManager trust scores evolve.

Attack types
------------
  random    : Agent sends random messages each step.
  collusion : Agent always sends maximum positive message (bias relay up).
  fdi       : Agent injects scaled false current reading into its observation.

Outputs
-------
  outputs/fig20_byzantine_trust_evolution.npz
      trust_scores    : (n_steps, n_agents, 3)  one column per attack type
      honest_mean     : (n_steps, 3)
      byzantine_score : (n_steps, 3)
      detection_step  : (3,)  first step each attack is flagged (trust < 0.5)
      false_positive  : (3,)  fraction of honest agents ever flagged

  outputs/fig20_byzantine_trust_evolution.png
"""

import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
os.makedirs("outputs", exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from network.network_data import get_8bus
from marl.marl_env import DynamicDOCREnv
from marl.marl_agents import MAPPOMARLTrainer
from deployment.byzantine_resilience import ReputationManager, ReputationConfig


# ── Helpers ────────────────────────────────────────────────────────────────────

def _train_base_policy(net, steps=8000, seed=0):
    env = DynamicDOCREnv(net, seed=seed, use_pandapower=False, fault_type="3ph")
    trainer = MAPPOMARLTrainer(env, use_comm=True, hidden_actor=32,
                               hidden_critic=64, seed=seed)
    result = trainer.train(total_steps=steps, verbose=False)
    return trainer, result


def _make_byzantine_obs(obs_i: np.ndarray, attack: str, rng: np.random.Generator,
                         step: int) -> np.ndarray:
    """Return corrupted observation for a Byzantine agent."""
    corrupted = obs_i.copy()
    if attack == "random":
        corrupted = rng.uniform(-1, 1, obs_i.shape)
    elif attack == "collusion":
        # Claim maximum fault current and zero CTI slack to alarm backups
        corrupted[2] = 1.0    # If_norm = max
        corrupted[3] = -1.0   # cti_slack = worst
        corrupted[4] = 1.0    # viol_flag = True
    elif attack == "fdi":
        # Scale fault current reading by ×3 (false data injection)
        corrupted[2] = min(1.0, obs_i[2] * 3.0)
    return corrupted


def run_byzantine_experiment(n_eval_steps: int = 300,
                              byzantine_agents: list = None,
                              attack: str = "random",
                              seed: int = 42) -> dict:
    """
    Run one Byzantine experiment.

    Parameters
    ----------
    n_eval_steps : int
        Steps to evaluate (post-training).
    byzantine_agents : list[int]
        Indices of agents to corrupt.  Default: [0].
    attack : str
        One of "random", "collusion", "fdi".
    seed : int

    Returns
    -------
    dict with trust_scores (n_steps, n_agents), detection_step, false_positive_rate
    """
    if byzantine_agents is None:
        byzantine_agents = [0]

    rng = np.random.default_rng(seed)
    net = get_8bus()
    n_agents = net.n_relays

    # Train a base policy (short run for experiment speed)
    trainer, _ = _train_base_policy(net, steps=5000, seed=seed)
    env = trainer.env

    rep_config = ReputationConfig(
        initial_trust=1.0,
        penalty_false_data=0.3,
        penalty_deviation=0.1,
        reward_consistent=0.05,
        min_trust=0.1,
        max_trust=2.0,
        window_size=10,
    )
    reputation = ReputationManager(n_agents=n_agents, config=rep_config)

    obs = env.reset(seed=seed)
    trust_log = []      # (n_steps, n_agents)
    detection_step = {byz: None for byz in byzantine_agents}

    for step in range(n_eval_steps):
        # Corrupt Byzantine agents' observations
        corrupted_obs = obs.copy()
        for byz in byzantine_agents:
            corrupted_obs[byz] = _make_byzantine_obs(obs[byz], attack, rng, step)

        # Get actions (policy uses corrupted obs for Byzantine agents)
        actions = np.zeros((n_agents, env.ACT_DIM))
        for i in range(n_agents):
            act, _ = trainer.actor.sample(corrupted_obs[i], rng)
            actions[i] = act

        obs_next, rewards, done, info = env.step(actions)

        # Update reputation based on reward signal (higher reward = consistent)
        for i in range(n_agents):
            neighbor_rewards = [float(rewards[j])
                                for j in range(n_agents) if j != i]
            reputation.update(
                agent_id=i,
                value=float(rewards[i]),
                neighbor_values=neighbor_rewards,
            )

        trust_now = reputation.trust_scores.copy()
        trust_log.append(trust_now)

        # Check detection
        for byz in byzantine_agents:
            if detection_step[byz] is None and trust_now[byz] < 0.5:
                detection_step[byz] = step

        if done:
            obs = env.reset()
        else:
            obs = obs_next

    trust_arr = np.array(trust_log)   # (n_steps, n_agents)

    honest_mask = np.ones(n_agents, dtype=bool)
    for byz in byzantine_agents:
        honest_mask[byz] = False

    false_positives = int(np.any(trust_arr[:, honest_mask] < 0.5))
    fp_rate = false_positives / max(1, honest_mask.sum())

    return {
        "trust_scores":     trust_arr,
        "honest_mean":      trust_arr[:, honest_mask].mean(axis=1),
        "byzantine_scores": trust_arr[:, byzantine_agents],
        "detection_steps":  detection_step,
        "fp_rate":          fp_rate,
        "attack":           attack,
        "n_agents":         n_agents,
        "byzantine_agents": byzantine_agents,
    }


# ── Main: run all three attack types ──────────────────────────────────────────

def main(n_eval_steps: int = 300, seed: int = 42):
    attacks   = ["random", "collusion", "fdi"]
    results   = {}

    print("Figure 20 — Byzantine Detection Experiment")
    print(f"  System : IEEE 8-bus ({get_8bus().n_relays} agents)")
    print(f"  Steps  : {n_eval_steps} eval steps per attack")
    print()

    for atk in attacks:
        print(f"  Attack: {atk} ...", end=" ", flush=True)
        res = run_byzantine_experiment(
            n_eval_steps=n_eval_steps,
            byzantine_agents=[0],
            attack=atk,
            seed=seed,
        )
        results[atk] = res
        det = list(res["detection_steps"].values())[0]
        print(f"detection_step={det}  fp_rate={res['fp_rate']:.3f}")

    # ── Save npz ───────────────────────────────────────────────────────────────
    npz_data = {}
    for atk in attacks:
        r = results[atk]
        npz_data[f"trust_{atk}"]       = r["trust_scores"]
        npz_data[f"honest_mean_{atk}"] = r["honest_mean"]
        npz_data[f"byz_score_{atk}"]   = r["byzantine_scores"][:, 0]
        npz_data[f"det_step_{atk}"]    = np.array(
            list(r["detection_steps"].values()), dtype=float
        )
        npz_data[f"fp_rate_{atk}"]     = np.array([r["fp_rate"]])

    np.savez("outputs/fig20_byzantine.npz", **npz_data)
    print("\n  Saved: outputs/fig20_byzantine.npz")

    # ── Plot ───────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    fig.suptitle("Fig 20 — Byzantine Agent Trust Score Evolution", fontsize=11)

    colors_honest = "#39d353"
    colors_byz    = "#ff6b35"
    thresh_color  = "#f7c948"

    for ax, atk in zip(axes, attacks):
        r = results[atk]
        steps = np.arange(len(r["honest_mean"]))

        ax.plot(steps, r["honest_mean"],
                color=colors_honest, lw=1.8, label="Honest agents (mean)")
        ax.plot(steps, r["byzantine_scores"][:, 0],
                color=colors_byz, lw=1.8, label="Byzantine agent")
        ax.axhline(0.5, color=thresh_color, ls="--", lw=1, label="Detection threshold")

        det = list(r["detection_steps"].values())[0]
        if det is not None:
            ax.axvline(det, color=colors_byz, ls=":", lw=1.2,
                       label=f"Detected @ step {det}")

        ax.set_title(f"Attack: {atk}", fontsize=9)
        ax.set_xlabel("Evaluation Step", fontsize=8)
        ax.set_ylim(0, 2.2)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)

    axes[0].set_ylabel("Trust Score", fontsize=8)
    fig.tight_layout()
    fig.savefig("outputs/fig20_byzantine_trust_evolution.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: outputs/fig20_byzantine_trust_evolution.png")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--seed",  type=int, default=42)
    args = parser.parse_args()
    main(n_eval_steps=args.steps, seed=args.seed)
