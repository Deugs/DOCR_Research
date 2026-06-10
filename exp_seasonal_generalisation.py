"""
exp_seasonal_generalisation.py
───────────────────────────────
Figure 58 — Seasonal / Weather Generalisation.

Evaluates a policy trained on a summer peak profile across four seasonal
irradiance profiles without retraining.

Seasonal profiles (simplified TMY3-representative values, 24-hour):
  Summer  : peak irradiance 1050 W/m², mean temperature 32°C
  Spring  : peak irradiance 850 W/m²,  mean temperature 20°C
  Autumn  : peak irradiance 650 W/m²,  mean temperature 15°C
  Winter  : peak irradiance 350 W/m²,  mean temperature 8°C

Each season's profile is injected into the ScenarioGenerator's
MarkovCloudModel by overriding the clear-sky peak and temperature offsets.

Metrics per season
------------------
  mean_OF         : objective function
  cti_viols       : mean P/B pair CTI violations
  pv_curtailment  : fraction of time PV output < 20% of summer peak

Outputs
-------
  outputs/fig58_seasonal.npz
  outputs/fig58_seasonal.png
"""

import numpy as np
import os, sys

sys.path.insert(0, os.path.dirname(__file__))
os.makedirs("outputs", exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from network.network_data import get_8bus
from marl.marl_env import DynamicDOCREnv
from marl.marl_agents import MAPPOMARLTrainer
from pv_model.scenario_generator import ScenarioGenerator, MarkovCloudModel
from network.relay_physics import check_constraints


# ── Seasonal profile factory ──────────────────────────────────────────────────

SEASONS = {
    "Summer": {"g_peak": 1050.0, "t_mean": 32.0, "cloud_bias": 0.0},
    "Spring": {"g_peak": 850.0,  "t_mean": 20.0, "cloud_bias": 0.05},
    "Autumn": {"g_peak": 650.0,  "t_mean": 15.0, "cloud_bias": 0.10},
    "Winter": {"g_peak": 350.0,  "t_mean": 8.0,  "cloud_bias": 0.20},
}


def _seasonal_irradiance(season_params: dict, n_steps: int = 96) -> np.ndarray:
    """
    Generate a 24-hour irradiance trace (n_steps points, 15-min resolution).
    Peak at noon, smooth cosine envelope, scaled to season peak.
    """
    hours      = np.linspace(0, 24, n_steps)
    angle      = 2 * np.pi * (hours - 12) / 24.0
    cos_factor = np.maximum(0, np.cos(angle))
    irr        = season_params["g_peak"] * cos_factor

    # Add stochastic cloud cover (more cloud in winter/autumn)
    rng   = np.random.default_rng(sum(ord(c) for c in str(season_params)))
    cloud = rng.random(n_steps) < season_params["cloud_bias"]
    irr   = np.where(cloud, irr * 0.3, irr)

    return irr


def _season_pv_levels(irr: np.ndarray, summer_peak: float = 1050.0) -> float:
    """Convert hourly irradiance to effective PV penetration level [0,1]."""
    return float(np.clip(irr.mean() / summer_peak, 0.0, 1.0))


# ── Evaluation helper ────────────────────────────────────────────────────────

def _eval_seasonal(actor, net, season_name: str, season_params: dict,
                   n_episodes: int = 30, seed: int = 0) -> dict:
    irr      = _seasonal_irradiance(season_params)
    pv_level = _season_pv_levels(irr)

    env = DynamicDOCREnv(net, seed=seed, use_pandapower=False, fault_type="3ph")
    rng = np.random.default_rng(seed)

    of_list, cti_list = [], []

    for ep in range(n_episodes):
        # Reset with the season's effective PV level
        obs = env.reset(seed=seed + ep, pv_level=pv_level)
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

    # PV curtailment: fraction of 15-min intervals where irr < 20% of summer
    curtailment = float(np.mean(irr < 0.20 * 1050.0))

    return {
        "season":         season_name,
        "pv_level":       pv_level,
        "mean_OF":        float(np.mean(of_list)),
        "std_OF":         float(np.std(of_list)),
        "cti_viols":      float(np.mean(cti_list)),
        "pv_curtailment": curtailment,
        "irradiance":     irr,
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main(train_steps: int = 8000, n_eval: int = 30, seed: int = 42):
    net = get_8bus()

    print("Figure 58 — Seasonal Generalisation Experiment")
    print(f"  System: {net.name}  ({net.n_relays} relays)")

    # Train on summer peak (full PV, curriculum_stage=5)
    print(f"\n  Training on Summer peak ({train_steps} steps, 100% PV)...")
    env     = DynamicDOCREnv(net, seed=seed, use_pandapower=False,
                              fault_type="3ph", curriculum_stage=5)
    trainer = MAPPOMARLTrainer(env, use_comm=False,
                                hidden_actor=64, hidden_critic=64, seed=seed)
    trainer.train(total_steps=train_steps, verbose=False)
    actor = trainer.actor

    # Evaluate on each season
    results = {}
    for season, params in SEASONS.items():
        print(f"  Evaluating: {season}  (pv_level ≈ {_season_pv_levels(_seasonal_irradiance(params)):.2f})")
        res = _eval_seasonal(actor, net, season, params, n_eval, seed=seed + hash(season) % 1000)
        results[season] = res
        print(f"    OF = {res['mean_OF']:.4f} ± {res['std_OF']:.4f}  "
              f"CTI = {res['cti_viols']:.1f}  curtailment = {res['pv_curtailment']:.2f}")

    # Degradation relative to summer
    summer_of = results["Summer"]["mean_OF"]
    print(f"\n  Seasonal degradation (relative to Summer OF={summer_of:.4f}):")
    for season, res in results.items():
        deg = (res["mean_OF"] - summer_of) / max(summer_of, 1e-6) * 100
        print(f"    {season:8s}: +{deg:.1f}%")

    # Save
    seasons_list = list(results.keys())
    np.savez(
        "outputs/fig58_seasonal.npz",
        seasons         = np.array(seasons_list),
        pv_levels       = np.array([results[s]["pv_level"]       for s in seasons_list]),
        mean_OF         = np.array([results[s]["mean_OF"]         for s in seasons_list]),
        std_OF          = np.array([results[s]["std_OF"]          for s in seasons_list]),
        cti_viols       = np.array([results[s]["cti_viols"]       for s in seasons_list]),
        pv_curtailment  = np.array([results[s]["pv_curtailment"]  for s in seasons_list]),
        irradiance_summer = results["Summer"]["irradiance"],
        irradiance_winter = results["Winter"]["irradiance"],
    )
    print("\n  Saved: outputs/fig58_seasonal.npz")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.suptitle("Fig 58 — Seasonal / Weather Generalisation (no retraining)", fontsize=11)

    colors  = ["#f7c948", "#39d353", "#ff6b35", "#2563eb"]   # summer, spring, autumn, winter
    x       = np.arange(len(seasons_list))
    w       = 0.55

    # Objective function
    of_vals = [results[s]["mean_OF"] for s in seasons_list]
    of_std  = [results[s]["std_OF"]  for s in seasons_list]
    bars0 = axes[0].bar(x, of_vals, color=colors, width=w, edgecolor="white")
    axes[0].errorbar(x, of_vals, yerr=of_std, fmt="none", color="white",
                     capsize=4, linewidth=1.2)
    for b, v in zip(bars0, of_vals):
        axes[0].text(b.get_x() + w/2, v + 0.005, f"{v:.3f}",
                     ha="center", fontsize=7.5)
    axes[0].set_title("Objective Function", fontsize=9)
    axes[0].set_xticks(x); axes[0].set_xticklabels(seasons_list, fontsize=8)
    axes[0].set_ylabel("OF (s)", fontsize=8); axes[0].grid(True, alpha=0.3, axis="y")

    # CTI violations
    cti_vals = [results[s]["cti_viols"] for s in seasons_list]
    axes[1].bar(x, cti_vals, color=colors, width=w, edgecolor="white")
    axes[1].set_title("CTI Violations", fontsize=9)
    axes[1].set_xticks(x); axes[1].set_xticklabels(seasons_list, fontsize=8)
    axes[1].set_ylabel("Mean Violations / Episode", fontsize=8)
    axes[1].grid(True, alpha=0.3, axis="y")

    # Irradiance profiles
    t = np.linspace(0, 24, 96)
    for s, c in zip(seasons_list, colors):
        axes[2].plot(t, results[s]["irradiance"], color=c, lw=1.5, label=s)
    axes[2].set_title("Irradiance Profiles", fontsize=9)
    axes[2].set_xlabel("Hour of Day", fontsize=8)
    axes[2].set_ylabel("Irradiance (W/m²)", fontsize=8)
    axes[2].legend(fontsize=7); axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig("outputs/fig58_seasonal.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: outputs/fig58_seasonal.png")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--train_steps", type=int, default=8000)
    p.add_argument("--n_eval",      type=int, default=30)
    p.add_argument("--seed",        type=int, default=42)
    args = p.parse_args()
    main(args.train_steps, args.n_eval, args.seed)
