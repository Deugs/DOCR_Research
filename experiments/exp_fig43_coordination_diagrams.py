"""
exp_coordination_diagrams.py  —  Figure 43
───────────────────────────────────────────
Relay Coordination Time-Distance Diagram.

For each line segment, places faults at 10%, 50%, 90% of line length.
Computes:
  t_primary  = TDS × (PSM^α - 1) / β   [IEC normal inverse]
  t_backup   = backup relay operating time
Ensures t_backup(x) - t_primary(x) ≥ 0.3 s (CTI) across all x.

Outputs
-------
  outputs/fig43_coordination_diagram.npz
  outputs/fig43_coordination_diagram.png
"""

import numpy as np
import os
import sys
import logging
import argparse
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from network.network_data import get_8bus
from marl.marl_env import DynamicDOCREnv
from marl.marl_agents import MAPPOMARLTrainer
from network.relay_physics import check_constraints

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

# IEC Normal Inverse constants
K_IEC = 0.14
ALPHA_IEC = 0.02

# NetworkData has no per-line length data anywhere (topology is implicit
# in RelayPair.bus_pri/bus_bkp; see NetworkData.n_lines's own docstring:
# "There is no separate `lines` table on NetworkData") -- this nominal
# medium-voltage feeder segment length is an illustrative placeholder for
# the x-axis distance scale, not a measured per-line value.
NOMINAL_LINE_LENGTH_KM = 10.0


def relay_operating_time(TDS: float, PSM: float, I_fault: float, I_pickup: float) -> float:
    """
    IEC normal inverse operating time.
    t = TDS × K / ((I / (PSM × I_pickup))^α - 1)
    """
    ratio = I_fault / max(PSM * I_pickup, 1e-9)
    if ratio <= 1.0:
        return float("inf")
    return TDS * K_IEC / (ratio ** ALPHA_IEC - 1.0)


def compute_coordination_curve(net, TMS: np.ndarray, PS: np.ndarray,
                                pair_idx: int, fault_positions: np.ndarray,
                                I_base: float = 5000.0) -> dict:
    """
    Compute primary and backup operating times along a line.

    Parameters
    ----------
    net : NetworkData
    TMS, PS : learned relay settings
    pair_idx : index into net.pairs -- NetworkData has no separate
        `.lines` table or RelayPair.line_idx field (topology is implicit
        in RelayPair.bus_pri/bus_bkp; see NetworkData.n_lines's own
        docstring), so each P/B pair is addressed directly by its own
        index rather than looked up via a nonexistent line table.
    fault_positions : array of fractional distances [0, 1] along line
    I_base : base fault current magnitude (A)

    Returns
    -------
    dict with distances, t_primary, t_backup, CTI_margin
    """
    pair = net.pairs[pair_idx]
    prim_idx = pair.primary
    back_idx = pair.backup
    # PS[i] is this codebase's real per-relay pickup setting (see
    # network/relay_physics.py's idmt_time: ratio = If / PS) -- there is
    # no separate `net.relays[i].I_pickup` field anywhere on NetworkData.
    I_pickup_prim = PS[prim_idx]
    I_pickup_back = PS[back_idx]

    # Fault current decreases with distance from source (simplified linear)
    distances = fault_positions * NOMINAL_LINE_LENGTH_KM
    I_fault_prim = I_base * (1.0 - 0.3 * fault_positions)   # primary sees higher current
    I_fault_back = I_base * (1.0 - 0.5 * fault_positions)   # backup sees lower current

    # PSM=1.0: this codebase's PS values already are the full pickup
    # current (see network/relay_physics.py's idmt_time: ratio = If/PS,
    # no separate multiplier) -- passing PS[idx] as BOTH the PSM and
    # I_pickup args below would square it, always saturating the ratio
    # below 1.0 and returning inf for every fault position regardless of
    # distance (confirmed: previously did exactly that for every curve).
    t_prim = np.array([relay_operating_time(TMS[prim_idx], 1.0,
                                               I_fault_prim[i], I_pickup_prim)
                       for i in range(len(fault_positions))])
    t_back = np.array([relay_operating_time(TMS[back_idx], 1.0,
                                               I_fault_back[i], I_pickup_back)
                       for i in range(len(fault_positions))])

    cti_margin = t_back - t_prim

    return {
        "distances_km": distances,
        "t_primary": t_prim,
        "t_backup": t_back,
        "cti_margin": cti_margin,
        "primary_relay": prim_idx,
        "backup_relay": back_idx,
        "line_name": f"Line {pair.bus_pri}-{pair.bus_bkp}",
    }


def main(train_steps: int = 8000, seed: int = 42, output_dir: str = "outputs"):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    net = get_8bus()
    env = DynamicDOCREnv(net, use_pandapower=False, fault_type="3ph", curriculum_stage=5)

    logger.info("Training policy (%d steps)...", train_steps)
    trainer = MAPPOMARLTrainer(env, use_comm=False, seed=seed)
    trainer.train(total_steps=train_steps, verbose=False)

    TMS, PS = env.get_state()

    # Select representative P/B pairs (3-4 as per guide). NetworkData has
    # no `.lines` table -- net.n_pairs is the real count of addressable
    # primary/backup relay pairs.
    representative_pairs = list(range(min(4, net.n_pairs)))
    fault_positions = np.array([0.1, 0.3, 0.5, 0.7, 0.9])

    all_curves = []
    for pair_idx in representative_pairs:
        curve = compute_coordination_curve(net, TMS, PS, pair_idx, fault_positions)
        if curve:
            all_curves.append(curve)
            logger.info("Line %s: min CTI margin = %.3f s", curve["line_name"],
                        np.min(curve["cti_margin"]))

    # Save
    save_dict = {"n_curves": np.array([len(all_curves)])}
    for i, c in enumerate(all_curves):
        for key in ["distances_km", "t_primary", "t_backup", "cti_margin"]:
            save_dict[f"curve{i}_{key}"] = c[key]
        save_dict[f"curve{i}_name"] = np.array([c["line_name"]])
        save_dict[f"curve{i}_prim"] = np.array([c["primary_relay"]])
        save_dict[f"curve{i}_back"] = np.array([c["backup_relay"]])

    np.savez(out_path / "fig43_coordination_diagram.npz", **save_dict)
    logger.info("Saved: %s", out_path / "fig43_coordination_diagram.npz")

    # Plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_curves = len(all_curves)
    fig, axes = plt.subplots(1, n_curves, figsize=(4 * n_curves, 4.5), sharey=True)
    if n_curves == 1:
        axes = [axes]

    colors = ["#2563eb", "#ff6b35"]
    for ax, curve in zip(axes, all_curves):
        d = curve["distances_km"]
        ax.plot(d, curve["t_primary"], "o-", color=colors[0], lw=2,
                label=f"Primary R{curve['primary_relay']}", markersize=6)
        ax.plot(d, curve["t_backup"], "s-", color=colors[1], lw=2,
                label=f"Backup R{curve['backup_relay']}", markersize=6)

        # CTI = 0.3s shaded region
        ax.fill_between(d, curve["t_primary"], curve["t_primary"] + 0.3,
                        color="#39d353", alpha=0.15, label="CTI = 0.3 s")
        ax.axhline(0.3, color="#39d353", ls="--", lw=1, alpha=0.5)

        # Mark violations
        viol_mask = curve["cti_margin"] < 0.3
        if np.any(viol_mask):
            ax.scatter(d[viol_mask], curve["t_backup"][viol_mask],
                       color="red", s=80, zorder=5, marker="x", linewidths=2,
                       label="CTI violation")

        ax.set_xlabel("Distance from Fault (km)", fontsize=9)
        ax.set_title(curve["line_name"], fontsize=10)
        ax.legend(fontsize=7, loc="upper left")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)

    axes[0].set_ylabel("Operating Time (s)", fontsize=9)
    fig.suptitle("Figure 43 — Relay Coordination Time-Distance Diagram", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path / "fig43_coordination_diagram.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", out_path / "fig43_coordination_diagram.png")

    # Compliance report
    print("\\n" + "=" * 60)
    print("COORDINATION COMPLIANCE REPORT")
    print("=" * 60)
    all_compliant = True
    for c in all_curves:
        min_cti = float(np.min(c["cti_margin"]))
        compliant = min_cti >= 0.3
        all_compliant = all_compliant and compliant
        status = "PASS ✓" if compliant else "FAIL ✗"
        print(f"  {c['line_name']:20s}  min CTI = {min_cti:.3f} s  {status}")
    print("=" * 60)
    print(f"OVERALL: {'ALL COMPLIANT ✓' if all_compliant else 'VIOLATIONS DETECTED ✗'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Figure 43 — Coordination Time-Distance Diagram")
    p.add_argument("--train_steps", type=int, default=8000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", type=str, default="outputs")
    args = p.parse_args()
    main(args.train_steps, args.seed, args.output_dir)
