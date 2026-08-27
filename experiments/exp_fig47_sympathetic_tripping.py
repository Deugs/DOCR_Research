
"""
exp_fig47_sympathetic_tripping.py — Figure 47
─────────────────────────────────────────────
Sympathetic Tripping Prevention Analysis.

Identifies fault scenarios where traditional fixed settings cause
backup relay to trip unnecessarily due to PV reverse flow.
Compares: (a) traditional fixed settings → backup trips,
          (b) MARL-adapted settings → backup restrained.

Outputs
-------
  outputs/fig47_sympathetic_tripping.npz
  outputs/fig47_sympathetic_tripping.png
"""

import numpy as np
import os
import sys
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
os.makedirs("outputs", exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from network.network_data import get_8bus
from marl.marl_env import DynamicDOCREnv
from marl.marl_agents import MAPPOMARLTrainer
from network.relay_physics import check_constraints

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

# Traditional fixed settings (typical conservative values). Sized to
# net.n_relays inside main() -- get_8bus() has 14 relays, not the 7 this
# used to be hardcoded to, which crashed with IndexError on every run.
TRADITIONAL_TDS_VALUE = 0.5
TRADITIONAL_PSM_VALUE = 2.0


def compute_trip_decision(net, TDS: np.ndarray, PSM: np.ndarray,
                          fault_location: int, fault_current: float,
                          I_pickup_base: float = 100.0) -> Dict:
    """
    Compute trip decisions for all relays given settings and fault.
    Returns dict with trip times and decisions for each relay.
    """
    K, alpha = 0.14, 0.02  # IEC NI
    decisions = {}

    for i in range(net.n_relays):
        I_pickup = I_pickup_base * PSM[i]
        ratio = fault_current / max(I_pickup, 1e-9)

        if ratio > 1.0:
            t_op = TDS[i] * K / (ratio ** alpha - 1.0)
        else:
            t_op = float("inf")

        decisions[i] = {
            "t_op": t_op,
            "trips": np.isfinite(t_op) and t_op < 10.0,  # trips if < 10s
            "I_seen": fault_current,
            "I_pickup": I_pickup,
            "ratio": ratio,
        }

    return decisions


def identify_sympathetic_cases(net, env, trainer, n_scenarios: int = 50,
                                 seed: int = 42) -> List[Dict]:
    """
    Identify scenarios where sympathetic tripping occurs with traditional
    settings but is prevented by MARL settings.
    """
    rng = np.random.default_rng(seed)
    cases = []

    TRADITIONAL_TDS = np.full(net.n_relays, TRADITIONAL_TDS_VALUE)
    TRADITIONAL_PSM = np.full(net.n_relays, TRADITIONAL_PSM_VALUE)

    for scenario in range(n_scenarios):
        obs = env.reset(seed=seed + scenario)
        done = False

        # Get MARL actions
        actions = np.zeros((env.n_agents, env.ACT_DIM))
        for i in range(env.n_agents):
            a, _ = trainer.actor.sample(obs[i], rng)
            actions[i] = a
        obs, _, done, info = env.step(actions)

        # Get MARL settings
        TMS_marl, PS_marl = env.get_state()

        # Simulate a fault with high PV reverse flow
        # (simplified: high fault current at a bus with significant PV)
        fault_bus = rng.integers(0, net.n_buses)
        fault_current = 8000 + rng.uniform(0, 4000)  # High fault current

        # Compute trip decisions
        trad_decisions = compute_trip_decision(net, TRADITIONAL_TDS, TRADITIONAL_PSM,
                                               fault_bus, fault_current)
        marl_decisions = compute_trip_decision(net, TMS_marl, PS_marl,
                                               fault_bus, fault_current)

        # Check for sympathetic tripping:
        # Backup trips when primary should handle it
        for pair in net.pairs:
            prim = pair.primary
            back = pair.backup

            trad_prim_trips = trad_decisions[prim]["trips"]
            trad_back_trips = trad_decisions[back]["trips"]
            marl_prim_trips = marl_decisions[prim]["trips"]
            marl_back_trips = marl_decisions[back]["trips"]

            # Sympathetic tripping: backup trips when primary also trips
            # (should be coordinated: backup waits for primary + CTI)
            trad_sympathetic = trad_prim_trips and trad_back_trips
            marl_sympathetic = marl_prim_trips and marl_back_trips

            if trad_sympathetic and not marl_sympathetic:
                cases.append({
                    "scenario": scenario,
                    "fault_bus": fault_bus,
                    "fault_current": fault_current,
                    "primary": prim,
                    "backup": back,
                    "pair": pair,
                    "trad_prim_t": trad_decisions[prim]["t_op"],
                    "trad_back_t": trad_decisions[back]["t_op"],
                    "marl_prim_t": marl_decisions[prim]["t_op"],
                    "marl_back_t": marl_decisions[back]["t_op"],
                    "trad_TDS": TRADITIONAL_TDS.copy(),
                    "trad_PSM": TRADITIONAL_PSM.copy(),
                    "marl_TDS": TMS_marl.copy(),
                    "marl_PSM": PS_marl.copy(),
                })

    return cases


def main(train_steps: int = 8000, n_scenarios: int = 50, seed: int = 42,
         output_dir: str = "outputs"):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    net = get_8bus()
    env = DynamicDOCREnv(net, use_pandapower=False, fault_type="3ph", curriculum_stage=5)

    logger.info("Figure 47 — Sympathetic Tripping Prevention Analysis")
    logger.info("  Training policy (%d steps)...", train_steps)

    trainer = MAPPOMARLTrainer(env, use_comm=False, seed=seed)
    trainer.train(total_steps=train_steps, verbose=False)

    # Identify sympathetic tripping cases
    logger.info("  Identifying sympathetic tripping cases (%d scenarios)...", n_scenarios)
    cases = identify_sympathetic_cases(net, env, trainer, n_scenarios, seed)
    logger.info("  Found %d sympathetic tripping cases prevented by MARL", len(cases))

    # ── Save ─────────────────────────────────────────────────────────────────
    if cases:
        save_dict = {
            "n_cases": len(cases),
            "scenario_ids": np.array([c["scenario"] for c in cases]),
            "fault_buses": np.array([c["fault_bus"] for c in cases]),
            "fault_currents": np.array([c["fault_current"] for c in cases]),
            "primaries": np.array([c["primary"] for c in cases]),
            "backups": np.array([c["backup"] for c in cases]),
            "trad_prim_t": np.array([c["trad_prim_t"] for c in cases]),
            "trad_back_t": np.array([c["trad_back_t"] for c in cases]),
            "marl_prim_t": np.array([c["marl_prim_t"] for c in cases]),
            "marl_back_t": np.array([c["marl_back_t"] for c in cases]),
        }
    else:
        save_dict = {"n_cases": 0}

    np.savez(out_path / "fig47_sympathetic_tripping.npz", **save_dict)
    logger.info("Saved: %s", out_path / "fig47_sympathetic_tripping.npz")

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Figure 47 — Sympathetic Tripping Prevention Analysis", fontsize=12)

    if cases:
        # Left: Timeline comparison for first 10 cases
        ax = axes[0]
        n_show = min(10, len(cases))
        y_positions = np.arange(n_show)

        for i, case in enumerate(cases[:n_show]):
            y = y_positions[i]
            # Traditional: both trip (sympathetic)
            ax.barh(y, case["trad_prim_t"], left=0, color="#ff6b35", height=0.3,
                    alpha=0.8, label="Primary (trad)" if i == 0 else "")
            ax.barh(y + 0.15, case["trad_back_t"], left=0, color="#ff6b6b", height=0.3,
                    alpha=0.8, label="Backup (trad)" if i == 0 else "")

            # MARL: primary trips, backup restrained
            ax.barh(y - 0.4, case["marl_prim_t"], left=0, color="#39d353", height=0.3,
                    alpha=0.8, label="Primary (MARL)" if i == 0 else "")
            if case["marl_back_t"] < 10:
                ax.barh(y - 0.25, case["marl_back_t"], left=0, color="#a371f7", height=0.3,
                        alpha=0.8, label="Backup (MARL)" if i == 0 else "")
            else:
                ax.text(0.1, y - 0.25, "NO TRIP", fontsize=7, color="#39d353", va="center")

        ax.set_yticks(y_positions - 0.1)
        ax.set_yticklabels([f"Case {c['scenario']}\\nP{c['primary']}-B{c['backup']}" 
                            for c in cases[:n_show]], fontsize=7)
        ax.set_xlabel("Operating Time (s)", fontsize=10)
        ax.set_title("Trip Time Comparison (Traditional vs. MARL)", fontsize=11)
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.3, axis="x")
        ax.set_xlim(0, max([c["trad_back_t"] for c in cases[:n_show]]) * 1.2)

        # Right: Settings comparison scatter
        ax = axes[1]
        for case in cases:
            ax.scatter(case["trad_TDS"][case["primary"]], case["trad_PSM"][case["primary"]],
                       c="#ff6b35", s=50, alpha=0.5, marker="o")
            ax.scatter(case["marl_TDS"][case["primary"]], case["marl_PSM"][case["primary"]],
                       c="#39d353", s=50, alpha=0.5, marker="s")

        ax.set_xlabel("TDS (s)", fontsize=10)
        ax.set_ylabel("PSM (×I_pickup)", fontsize=10)
        ax.set_title("Primary Relay Settings: Traditional vs. MARL", fontsize=11)
        ax.grid(True, alpha=0.3)
    else:
        for ax in axes:
            ax.text(0.5, 0.5, "No sympathetic tripping cases detected\\n(try increasing n_scenarios)",
                    ha="center", va="center", transform=ax.transAxes, fontsize=12)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)

    fig.tight_layout()
    fig.savefig(out_path / "fig47_sympathetic_tripping.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", out_path / "fig47_sympathetic_tripping.png")

    # Summary
    print("\\n" + "=" * 60)
    print("SYMPATHETIC TRIPPING ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"Scenarios tested:       {n_scenarios}")
    print(f"Sympathetic cases found: {len(cases)}")
    if cases:
        print(f"Mean traditional backup time: {np.mean([c['trad_back_t'] for c in cases]):.3f}s")
        print(f"Mean MARL backup time:        {np.mean([c['marl_back_t'] for c in cases]):.3f}s")
        print(f"Prevention rate:              {len(cases)/n_scenarios*100:.1f}%")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Figure 47 — Sympathetic Tripping")
    parser.add_argument("--train_steps", type=int, default=8000)
    parser.add_argument("--n_scenarios", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="outputs")
    args = parser.parse_args()
    main(args.train_steps, args.n_scenarios, args.seed, args.output_dir)
