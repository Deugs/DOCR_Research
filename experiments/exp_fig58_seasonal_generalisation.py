
"""
exp_seasonal_generalisation.py  —  Figure 58
─────────────────────────────────────────────
Seasonal / Weather Generalisation.

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
import os
import sys
import zlib
import logging
import argparse
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
os.makedirs("outputs", exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from network.network_data import get_8bus
from marl.marl_env import DynamicDOCREnv
from marl.marl_agents import MAPPOMARLTrainer
from network.relay_physics import check_constraints

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


# ── Seasonal profile factory ──────────────────────────────────────────────────

SEASONS = {
    "Summer": {"g_peak": 1050.0, "t_mean": 32.0, "cloud_bias": 0.0, "color": "#f7c948"},
    "Spring": {"g_peak": 850.0,  "t_mean": 20.0, "cloud_bias": 0.05, "color": "#39d353"},
    "Autumn": {"g_peak": 650.0,  "t_mean": 15.0, "cloud_bias": 0.10, "color": "#ff6b35"},
    "Winter": {"g_peak": 350.0,  "t_mean": 8.0,  "cloud_bias": 0.20, "color": "#2563eb"},
}


def _seasonal_irradiance(season_params: dict, n_steps: int = 96, seed: int = 0) -> np.ndarray:
    """
    Generate a 24-hour irradiance trace (n_steps points, 15-min resolution).
    Peak at noon, smooth cosine envelope, scaled to season peak.
    """
    rng = np.random.default_rng(seed)
    hours = np.linspace(0, 24, n_steps)
    angle = 2 * np.pi * (hours - 12) / 24.0
    cos_factor = np.maximum(0, np.cos(angle))
    irr = season_params["g_peak"] * cos_factor

    # Add stochastic cloud cover (more cloud in winter/autumn)
    cloud = rng.random(n_steps) < season_params["cloud_bias"]
    irr = np.where(cloud, irr * 0.3, irr)

    return irr


def _season_pv_levels(irr: np.ndarray, summer_peak: float = 1050.0) -> float:
    """Convert hourly irradiance to effective PV penetration level [0,1]."""
    return float(np.clip(irr.mean() / summer_peak, 0.0, 1.0))


# ── Evaluation helper ────────────────────────────────────────────────────────

def _eval_seasonal(actor, net, season_name: str, season_params: dict,
                   n_episodes: int = 30, seed: int = 0) -> dict:
    irr = _seasonal_irradiance(season_params, seed=seed)
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
        "season": season_name,
        "pv_level": pv_level,
        "mean_OF": float(np.mean(of_list)),
        "std_OF": float(np.std(of_list)),
        "cti_viols": float(np.mean(cti_list)),
        "pv_curtailment": curtailment,
        "irradiance": irr,
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main(train_steps: int = 8000, n_eval: int = 30, seed: int = 42,
         output_dir: str = "outputs"):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    net = get_8bus()

    logger.info("Figure 58 — Seasonal Generalisation Experiment")
    logger.info("  System: %s  (%d relays)", net.name, net.n_relays)

    # Train on summer peak (full PV, curriculum_stage=5)
    logger.info("Training on Summer peak (%d steps, 100%% PV)...", train_steps)
    env = DynamicDOCREnv(net, seed=seed, use_pandapower=False,
                          fault_type="3ph", curriculum_stage=5)
    trainer = MAPPOMARLTrainer(env, use_comm=False,
                                hidden_actor=64, hidden_critic=64, seed=seed)
    trainer.train(total_steps=train_steps, verbose=False)
    actor = trainer.actor

    # Evaluate on each season
    results = {}
    for season, params in SEASONS.items():
        logger.info("Evaluating: %s  (pv_level ≈ %.2f)",
                    season, _season_pv_levels(_seasonal_irradiance(params, seed=seed)))
        # zlib.crc32 instead of hash(): Python's hash() on strings is
        # salted per-process by default (PYTHONHASHSEED randomization),
        # so re-running this script with the identical --seed produced a
        # different per-season eval seed (and thus different results)
        # on every run, despite the repo's own stated convention that
        # every experiment script be reproducible from an explicit seed.
        res = _eval_seasonal(actor, net, season, params, n_eval,
                             seed=seed + zlib.crc32(season.encode()) % 1000)
        results[season] = res
        logger.info("    OF = %.4f ± %.4f  CTI = %.1f  curtailment = %.2f",
                    res["mean_OF"], res["std_OF"], res["cti_viols"], res["pv_curtailment"])

    # Degradation relative to summer
    summer_of = results["Summer"]["mean_OF"]
    logger.info("Seasonal degradation (relative to Summer OF=%.4f):", summer_of)
    for season, res in results.items():
        deg = (res["mean_OF"] - summer_of) / max(summer_of, 1e-6) * 100
        logger.info("    %8s: +%.1f%%", season, deg)

    # Save
    seasons_list = list(results.keys())
    np.savez(
        out_path / "fig58_seasonal.npz",
        seasons=np.array(seasons_list),
        pv_levels=np.array([results[s]["pv_level"] for s in seasons_list]),
        mean_OF=np.array([results[s]["mean_OF"] for s in seasons_list]),
        std_OF=np.array([results[s]["std_OF"] for s in seasons_list]),
        cti_viols=np.array([results[s]["cti_viols"] for s in seasons_list]),
        pv_curtailment=np.array([results[s]["pv_curtailment"] for s in seasons_list]),
        irradiance_summer=results["Summer"]["irradiance"],
        irradiance_winter=results["Winter"]["irradiance"],
    )
    logger.info("Saved: %s", out_path / "fig58_seasonal.npz")

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Figure 58 — Seasonal / Weather Generalisation (no retraining)", fontsize=12, y=1.02)

    colors = [SEASONS[s]["color"] for s in seasons_list]
    x = np.arange(len(seasons_list))
    w = 0.55

    # Objective function
    ax = axes[0, 0]
    of_vals = [results[s]["mean_OF"] for s in seasons_list]
    of_std = [results[s]["std_OF"] for s in seasons_list]
    bars0 = ax.bar(x, of_vals, color=colors, width=w, edgecolor="white")
    ax.errorbar(x, of_vals, yerr=of_std, fmt="none", color="white",
                capsize=4, linewidth=1.2)
    for b, v in zip(bars0, of_vals):
        ax.text(b.get_x() + w / 2, v + 0.005, f"{v:.3f}",
                ha="center", fontsize=7.5)
    ax.set_title("Objective Function", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(seasons_list, fontsize=8)
    ax.set_ylabel("OF (s)", fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    # CTI violations
    ax = axes[0, 1]
    cti_vals = [results[s]["cti_viols"] for s in seasons_list]
    ax.bar(x, cti_vals, color=colors, width=w, edgecolor="white")
    ax.set_title("CTI Violations", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(seasons_list, fontsize=8)
    ax.set_ylabel("Mean Violations / Episode", fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    # Irradiance profiles
    ax = axes[1, 0]
    t = np.linspace(0, 24, 96)
    for s in seasons_list:
        ax.plot(t, results[s]["irradiance"], color=SEASONS[s]["color"],
                lw=1.5, label=s)
    ax.set_title("Irradiance Profiles", fontsize=10)
    ax.set_xlabel("Hour of Day", fontsize=8)
    ax.set_ylabel("Irradiance (W/m²)", fontsize=8)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Degradation bar chart
    ax = axes[1, 1]
    degradations = []
    for s in seasons_list:
        deg = (results[s]["mean_OF"] - summer_of) / max(summer_of, 1e-6) * 100
        degradations.append(deg)
    bars = ax.bar(x, degradations, color=colors, width=w, edgecolor="white")
    for b, v in zip(bars, degradations):
        ax.text(b.get_x() + w / 2, v + 0.5, f"{v:.1f}%",
                ha="center", fontsize=7.5)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title("Performance Degradation vs. Summer", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(seasons_list, fontsize=8)
    ax.set_ylabel("Degradation (%)", fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(out_path / "fig58_seasonal.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", out_path / "fig58_seasonal.png")

    # Summary table
    print("\\n" + "=" * 60)
    print("SEASONAL GENERALISATION SUMMARY")
    print("=" * 60)
    print(f"{'Season':>10} {'PV Level':>10} {'OF (s)':>10} {'CTI':>8} {'Curtail':>10} {'Degrad%':>10}")
    print("-" * 60)
    for s in seasons_list:
        r = results[s]
        deg = (r["mean_OF"] - summer_of) / max(summer_of, 1e-6) * 100
        print(f"{s:>10} {r['pv_level']:>10.2f} {r['mean_OF']:>10.4f} {r['cti_viols']:>8.1f} "
              f"{r['pv_curtailment']:>10.2f} {deg:>9.1f}%")
    print("=" * 60)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Figure 58 — Seasonal Generalisation")
    p.add_argument("--train_steps", type=int, default=8000)
    p.add_argument("--n_eval", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", type=str, default="outputs")
    args = p.parse_args()
    main(args.train_steps, args.n_eval, args.seed, args.output_dir)
