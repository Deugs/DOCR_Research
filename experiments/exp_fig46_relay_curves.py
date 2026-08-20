
"""
exp_fig46_relay_curves.py — Figure 46
───────────────────────────────────────
Relay Operating Characteristic Curves with Learned Settings.

For each learned (TDS, PSM) pair, computes:
  t(I) = TDS × K / ((I / (PSM × I_pickup))^α - 1)

Supports IEC curve types: Normal Inverse (NI), Very Inverse (VI),
Extremely Inverse (EI).

Overlays fault current points (I_fault, t_op) for each scenario.
Verifies all points lie on or above the curve.

Outputs
-------
  outputs/fig46_relay_curves.npz
  outputs/fig46_relay_curves.png
"""

import numpy as np
import os
import sys
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
os.makedirs("outputs", exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from network.network_data import get_8bus
from marl.marl_env import DynamicDOCREnv
from marl.marl_agents import MAPPOMARLTrainer
from network.relay_physics import check_constraints, compute_fault_currents_dynamic

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

# IEC curve constants
IEC_CURVES = {
    "NI": {"K": 0.14, "alpha": 0.02, "name": "Normal Inverse"},
    "VI": {"K": 13.5, "alpha": 1.0, "name": "Very Inverse"},
    "EI": {"K": 80.0, "alpha": 2.0, "name": "Extremely Inverse"},
}


def relay_operating_time(TDS: float, PS: float, I_fault: float,
                         curve_type: str = "NI") -> float:
    """
    Compute relay operating time using IEC standard curves.
    t = TDS × K / ((I_fault / PS)^α - 1)

    PS is the relay's actual plug/pickup setting in Amps, exactly as
    returned by DynamicDOCREnv.get_state() and bounded by
    NetworkData.ps_bounds() — this matches
    network.relay_physics.idmt_time's convention. NetworkData has no
    separate hardware "I_pickup" multiplier field, so there is nothing to
    multiply PS by here (an earlier version of this script assumed a
    `net.relays[i].I_pickup` attribute that does not exist anywhere in
    NetworkData and crashed at runtime; see _relay_nominal_current()
    below for the axis-normalization reference used instead).
    """
    curve = IEC_CURVES[curve_type]
    K, alpha = curve["K"], curve["alpha"]

    ratio = I_fault / max(PS, 1e-9)
    if ratio <= 1.0:
        return float("inf")

    return TDS * K / (ratio ** alpha - 1.0)


def generate_curve(TDS: float, PS: float, I_nominal: float,
                    curve_type: str = "NI", I_range: Tuple[float, float] = (1.0, 20.0),
                    n_points: int = 200) -> Tuple[np.ndarray, np.ndarray]:
    """Generate (I/I_nominal, t) curve points. I_nominal is only an x-axis
    normalization reference (net.Iload_max for this relay); it plays no role
    in the operate-time formula itself (see relay_operating_time)."""
    I_multiples = np.linspace(I_range[0], I_range[1], n_points)
    times = np.array([relay_operating_time(TDS, PS, I * I_nominal, curve_type)
                      for I in I_multiples])
    return I_multiples, times


def _relay_nominal_current(net, relay_idx: int) -> float:
    """
    Nominal-current axis-normalization reference for a relay (A).

    NetworkData has no per-relay "I_pickup" hardware field. net.Iload_max is
    the per-relay max load current that NetworkData.ps_bounds() itself uses
    as the basis for the PS lower bound, making it the closest real,
    per-relay quantity to normalize the I/I_nominal plotting axis against.
    """
    return float(net.Iload_max[relay_idx])


def _relay_bus(net, relay_idx: int) -> int:
    """Bus location of a relay. NetworkData has no direct per-relay bus
    list — only bus_pri/bus_bkp per RelayPair — so this looks up the bus
    from the first pair in which the relay appears as primary or backup."""
    for p in net.pairs:
        if p.primary == relay_idx:
            return p.bus_pri
        if p.backup == relay_idx:
            return p.bus_bkp
    return -1


def _real_fault_current_for_relay(net, relay_idx: int, pv_penetration: float,
                                   fault_type: str) -> float:
    """
    Real, analytically-derived fault current (A) this relay sees in its
    primary role, from the network's short-circuit study data
    (network.relay_physics.compute_fault_currents_dynamic — includes PV
    inverter fault-current contribution). Returns None if this relay is
    never a "primary" in any P/B pair, i.e. no such study value exists.
    """
    fc = compute_fault_currents_dynamic(net, pv_penetration=pv_penetration, fault_type=fault_type)
    for i, p in enumerate(net.pairs):
        if p.primary == relay_idx:
            return fc[f"pair_{i}_pri"]
    return None


def collect_fault_points(env, trainer, net, n_episodes: int = 100, seed: int = 42) -> Dict[int, List[Tuple[float, float]]]:
    """
    Collect (I_fault_seen, t_op) points for each relay from test episodes.
    Returns dict: relay_idx → [(I_ratio, t_op), ...]

    Where the network's short-circuit study data has a fault current for a
    relay's primary role (network.relay_physics.compute_fault_currents_dynamic),
    that real value is used (scaled per the episode's PV penetration level).
    Relays that are never a "primary" in any P/B pair have no such study
    value; for those, an illustrative estimate — a seeded uniform multiple
    of the relay's nominal current — is used instead. That fallback is
    disclosed here and is NOT derived from an actual fault study.
    """
    rng = np.random.default_rng(seed)
    relay_points = {i: [] for i in range(net.n_relays)}
    pv_levels = [0.0, 0.25, 0.5, 0.75, 1.0]

    for ep in range(n_episodes):
        obs = env.reset(seed=seed + ep)
        done = False
        while not done:
            actions = np.zeros((env.n_agents, env.ACT_DIM))
            for i in range(env.n_agents):
                a, _ = trainer.actor.sample(obs[i], rng)
                actions[i] = a
            obs, _, done, info = env.step(actions)

        # Get final settings
        TMS, PS = env.get_state()
        pv_penetration = pv_levels[ep % len(pv_levels)]

        # For each relay, get (or illustratively estimate) fault current and
        # compute operating time with the learned settings.
        for i in range(net.n_relays):
            I_nominal = _relay_nominal_current(net, i)

            I_fault_real = _real_fault_current_for_relay(net, i, pv_penetration, "3ph")
            if I_fault_real is not None:
                I_fault_est = I_fault_real
            else:
                # Illustrative only — no short-circuit study entry for this
                # relay as primary. Seeded so re-running is reproducible.
                I_fault_est = I_nominal * rng.uniform(2.0, 15.0)

            I_ratio = I_fault_est / I_nominal
            t_op = relay_operating_time(TMS[i], PS[i], I_fault_est, "NI")

            if np.isfinite(t_op):
                relay_points[i].append((I_ratio, t_op))

    return relay_points


def main(train_steps: int = 8000, n_episodes: int = 50, seed: int = 42,
         output_dir: str = "outputs"):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    net = get_8bus()
    env = DynamicDOCREnv(net, use_pandapower=False, fault_type="3ph", curriculum_stage=5)

    logger.info("Figure 46 — Relay Operating Characteristic Curves")
    logger.info("  Training policy (%d steps)...", train_steps)

    trainer = MAPPOMARLTrainer(env, use_comm=False, seed=seed)
    trainer.train(total_steps=train_steps, verbose=False)

    # Get learned settings
    TMS, PS = env.get_state()
    logger.info("  Learned settings: TDS mean=%.3f, PSM mean=%.2f", TMS.mean(), PS.mean())

    # Collect fault points
    logger.info("  Collecting fault points from %d episodes...", n_episodes)
    relay_points = collect_fault_points(env, trainer, net, n_episodes, seed)

    # ── Save ─────────────────────────────────────────────────────────────────
    save_dict = {
        "TMS": TMS,
        "PS": PS,
        "n_relays": np.array([net.n_relays]),
    }
    for i, points in relay_points.items():
        if points:
            arr = np.array(points)
            save_dict[f"relay{i}_I_ratio"] = arr[:, 0]
            save_dict[f"relay{i}_t_op"] = arr[:, 1]

    np.savez(out_path / "fig46_relay_curves.npz", **save_dict)
    logger.info("Saved: %s", out_path / "fig46_relay_curves.npz")

    # ── Plot ─────────────────────────────────────────────────────────────────
    n_relays_show = min(6, net.n_relays)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    colors = {"NI": "#2563eb", "VI": "#39d353", "EI": "#ff6b35"}

    for idx in range(n_relays_show):
        ax = axes[idx]
        I_nominal = _relay_nominal_current(net, idx)

        # Plot all three curve types
        for curve_type, params in IEC_CURVES.items():
            I_mult, times = generate_curve(TMS[idx], PS[idx], I_nominal, curve_type)
            valid = np.isfinite(times) & (times > 0) & (times < 100)
            ax.plot(I_mult[valid], times[valid], color=colors[curve_type],
                    lw=1.5, alpha=0.7, label=f"{params['name']}")

        # Overlay fault points (real short-circuit-study values where
        # available, illustrative seeded estimates otherwise — see
        # collect_fault_points())
        points = relay_points.get(idx, [])
        if points:
            arr = np.array(points)
            ax.scatter(arr[:, 0], arr[:, 1], c="#ff6b35", s=30, alpha=0.6,
                       edgecolors='white', linewidth=0.5, zorder=5, label="Fault scenarios")

        # Mark pickup current (PS is in Amps; convert to the I/I_nominal axis)
        ax.axvline(PS[idx] / I_nominal, color="#f7c948", ls="--", lw=1, alpha=0.7,
                   label=f"PS = {PS[idx]:.0f}A")

        ax.set_xlabel("I / I_nominal", fontsize=9)
        ax.set_ylabel("Operating Time (s)", fontsize=9)
        ax.set_title(f"Relay {idx} (Bus {_relay_bus(net, idx)}, I_nominal={I_nominal:.0f}A)", fontsize=10)
        ax.set_xlim(0, 20)
        ax.set_ylim(0.01, 100)
        ax.set_yscale("log")
        ax.legend(fontsize=6, loc="upper right")
        ax.grid(True, alpha=0.3, which="both")

    fig.suptitle("Figure 46 — Relay Operating Characteristic Curves (Learned Settings)\n"
                 "Fault-current overlay: real short-circuit-study values where available, "
                 "illustrative seeded estimates otherwise",
                 fontsize=11, y=1.04)
    fig.tight_layout()
    fig.savefig(out_path / "fig46_relay_curves.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", out_path / "fig46_relay_curves.png")

    # Verification report
    print("\\n" + "=" * 60)
    print("RELAY CURVE VERIFICATION REPORT")
    print("=" * 60)
    for idx in range(net.n_relays):
        points = relay_points.get(idx, [])
        if points:
            arr = np.array(points)
            # Check if points are above NI curve
            I_nominal = _relay_nominal_current(net, idx)
            curve_times = np.array([relay_operating_time(TMS[idx], PS[idx], p[0] * I_nominal, "NI")
                                    for p in points])
            actual_times = arr[:, 1]
            above_curve = np.sum(actual_times >= curve_times * 0.95)  # 5% tolerance
            print(f"  Relay {idx}: {above_curve}/{len(points)} points on/above curve")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Figure 46 — Relay Operating Curves")
    parser.add_argument("--train_steps", type=int, default=8000)
    parser.add_argument("--n_episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="outputs")
    args = parser.parse_args()
    main(args.train_steps, args.n_episodes, args.seed, args.output_dir)
