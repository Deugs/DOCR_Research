
"""
exp_fig36_action_space_geometry.py — Figure 36
────────────────────────────────────────────────
Action Space Bounds & Safety Projection Geometry.

Grid search over TDS × PSM space. For each (TDS, PSM) pair,
simulates all faults and computes minimum CTI across all P/B pairs.
Marks feasible (CTI ≥ 0.3s) and infeasible regions.

Overlays: (a) raw policy actions (before projection),
          (b) projected actions (after safety layer).

Outputs
-------
  outputs/fig36_action_space.npz
  outputs/fig36_action_space.png
"""

import numpy as np
import os
import sys
import logging
import argparse
from pathlib import Path
from typing import Tuple, List

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
os.makedirs("outputs", exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from matplotlib.colors import LinearSegmentedColormap

from network.network_data import get_8bus
from marl.marl_env import DynamicDOCREnv
from marl.marl_agents import MAPPOMARLTrainer
from network.relay_physics import check_constraints

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def compute_cti_for_settings(net, TDS: np.ndarray, PSM: np.ndarray,
                               fault_type: str = "3ph", n_fault_locs: int = 20) -> float:
    """
    Compute minimum CTI margin across all P/B pairs for given relay settings.
    Returns the minimum CTI (s); negative means violation.
    """
    env = DynamicDOCREnv(net, use_pandapower=False, fault_type=fault_type, curriculum_stage=5)
    env.set_state(TDS, PSM)

    min_cti = float('inf')
    rng = np.random.default_rng(42)

    for _ in range(n_fault_locs):
        obs = env.reset(seed=rng.integers(0, 10000))
        done = False
        while not done:
            # Use random actions just to advance environment
            actions = rng.uniform(-0.1, 0.1, (env.n_agents, env.ACT_DIM))
            obs, _, done, info = env.step(actions)

        TMS, PS = env.get_state()
        c = check_constraints(net, TMS, PS)
        cti_min = c.get("cti_min", -1.0)
        if cti_min < min_cti:
            min_cti = cti_min

    return min_cti


def grid_search_feasibility(net, tds_grid: np.ndarray, psm_grid: np.ndarray,
                            n_fault_locs: int = 20) -> np.ndarray:
    """
    Grid search: for each (TDS, PSM) pair, compute min CTI.
    Returns 2D array of CTI margins.
    """
    n_tds = len(tds_grid)
    n_psm = len(psm_grid)
    cti_grid = np.zeros((n_tds, n_psm))

    logger.info("Grid search: %d × %d = %d points", n_tds, n_psm, n_tds * n_psm)

    for i, tds in enumerate(tds_grid):
        for j, psm in enumerate(psm_grid):
            TDS = np.full(net.n_relays, tds)
            PSM = np.full(net.n_relays, psm)
            cti = compute_cti_for_settings(net, TDS, PSM, n_fault_locs=n_fault_locs)
            cti_grid[i, j] = cti
            if (i * n_psm + j) % 50 == 0:
                logger.info("  Progress: %d/%d  (TDS=%.3f, PSM=%.2f → CTI=%.3f)",
                            i * n_psm + j, n_tds * n_psm, tds, psm, cti)

    return cti_grid


def collect_policy_actions(trainer, env, n_episodes: int = 50, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Collect raw (pre-projection) and projected actions from trained policy."""
    rng = np.random.default_rng(seed)
    raw_actions = []
    projected_actions = []

    for ep in range(n_episodes):
        obs = env.reset(seed=seed + ep)
        done = False
        while not done:
            # Get raw action from policy (before safety projection)
            actions_raw = np.zeros((env.n_agents, env.ACT_DIM))
            for i in range(env.n_agents):
                # Sample from actor (this is pre-projection)
                a_raw, _ = trainer.actor.sample(obs[i], rng)
                actions_raw[i] = a_raw

            # Step environment (applies safety projection internally)
            obs, _, done, info = env.step(actions_raw)

            # Get projected actions from environment state
            TMS, PS = env.get_state()
            # Convert back to normalized action space
            tds_norm = (TMS - 0.1) / 0.9  # denormalize from [0.1, 1.0]
            psm_norm = (PS - 1.0) / 9.0   # denormalize from [1.0, 10.0]
            actions_proj = np.column_stack([tds_norm, psm_norm])

            raw_actions.append(actions_raw.mean(axis=0))
            projected_actions.append(actions_proj.mean(axis=0))

    return np.array(raw_actions), np.array(projected_actions)


def main(grid_size: int = 40, n_fault_locs: int = 15, train_steps: int = 8000,
         seed: int = 42, output_dir: str = "outputs"):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    net = get_8bus()
    env = DynamicDOCREnv(net, use_pandapower=False, fault_type="3ph", curriculum_stage=5)

    logger.info("Figure 36 — Action Space Geometry & Safety Projection")
    logger.info("  Grid size: %d × %d", grid_size, grid_size)

    # Define grids
    tds_grid = np.linspace(0.1, 1.0, grid_size)
    psm_grid = np.linspace(1.0, 10.0, grid_size)

    # ── Grid search for feasibility ──────────────────────────────────────────
    logger.info("[1/3] Running feasibility grid search...")
    cti_grid = grid_search_feasibility(net, tds_grid, psm_grid, n_fault_locs)

    # ── Train policy and collect actions ────────────────────────────────────
    logger.info("[2/3] Training policy (%d steps)...", train_steps)
    trainer = MAPPOMARLTrainer(env, use_comm=False, seed=seed)
    trainer.train(total_steps=train_steps, verbose=False)

    logger.info("[3/3] Collecting policy actions...")
    raw_actions, proj_actions = collect_policy_actions(trainer, env, n_episodes=30, seed=seed)

    # ── Save ─────────────────────────────────────────────────────────────────
    np.savez(
        out_path / "fig36_action_space.npz",
        tds_grid=tds_grid,
        psm_grid=psm_grid,
        cti_grid=cti_grid,
        raw_actions=raw_actions,
        projected_actions=proj_actions,
        cti_threshold=0.3,
    )
    logger.info("Saved: %s", out_path / "fig36_action_space.npz")

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Figure 36 — Action Space Bounds & Safety Projection Geometry", fontsize=12)

    # Custom colormap: red (infeasible) → yellow (boundary) → green (feasible)
    colors = ["#ff6b6b", "#f7c948", "#39d353"]
    cmap = LinearSegmentedColormap.from_list("feasibility", colors, N=256)

    # Left: Feasibility heatmap
    ax = axes[0]
    T, P = np.meshgrid(tds_grid, psm_grid)
    im = ax.pcolormesh(T, P, cti_grid.T, cmap=cmap, shading='auto', vmin=-0.5, vmax=1.0)
    ax.contour(T, P, cti_grid.T, levels=[0.3], colors='white', linewidths=2, linestyles='--')
    ax.clabel(ax.contour(T, P, cti_grid.T, levels=[0.3], colors='white'),
              inline=True, fontsize=9, fmt='CTI = 0.3s')

    # Overlay policy actions
    ax.scatter(raw_actions[:, 0] * 0.9 + 0.1, raw_actions[:, 1] * 9.0 + 1.0,
               c='#ff6b35', s=30, alpha=0.6, edgecolors='white', linewidth=0.5,
               label='Raw policy actions', zorder=5)
    ax.scatter(proj_actions[:, 0] * 0.9 + 0.1, proj_actions[:, 1] * 9.0 + 1.0,
               c='#00d4ff', s=30, alpha=0.6, edgecolors='white', linewidth=0.5,
               label='Projected actions', zorder=5)

    # Draw projection arrows for subset
    n_arrows = min(15, len(raw_actions))
    idx = np.random.default_rng(seed).choice(len(raw_actions), n_arrows, replace=False)
    for i in idx:
        x1 = raw_actions[i, 0] * 0.9 + 0.1
        y1 = raw_actions[i, 1] * 9.0 + 1.0
        x2 = proj_actions[i, 0] * 0.9 + 0.1
        y2 = proj_actions[i, 1] * 9.0 + 1.0
        if abs(x2 - x1) > 0.01 or abs(y2 - y1) > 0.1:
            ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                        arrowprops=dict(arrowstyle="->", color='#ff6b35', lw=1.2, alpha=0.7))

    ax.set_xlabel("TDS (s)", fontsize=10)
    ax.set_ylabel("PSM (×I_pickup)", fontsize=10)
    ax.set_title("Feasible Region & Policy Trajectory", fontsize=11)
    ax.set_xlim(0.05, 1.05)
    ax.set_ylim(0.5, 10.5)
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3)
    plt.colorbar(im, ax=ax, shrink=0.8, label="Min CTI (s)")

    # Right: 3D surface view
    ax2 = axes[1]
    from mpl_toolkits.mplot3d import Axes3D
    fig2 = plt.figure(figsize=(8, 6))
    ax3d = fig2.add_subplot(111, projection='3d')
    surf = ax3d.plot_surface(T, P, cti_grid.T, cmap=cmap, alpha=0.9,
                             edgecolor='none', antialiased=True)
    ax3d.contour(T, P, cti_grid.T, levels=[0.3], colors='white', linewidths=2)
    ax3d.set_xlabel("TDS (s)", fontsize=9)
    ax3d.set_ylabel("PSM (×I_pickup)", fontsize=9)
    ax3d.set_zlabel("Min CTI (s)", fontsize=9)
    ax3d.set_title("CTI Margin Surface", fontsize=11)
    fig2.colorbar(surf, shrink=0.5, label="Min CTI (s)")
    fig2.savefig(out_path / "fig36_action_space_3d.png", dpi=200, bbox_inches='tight')
    plt.close(fig2)
    logger.info("Saved 3D view: %s", out_path / "fig36_action_space_3d.png")

    # Use right panel for projection magnitude histogram instead
    ax = axes[1]
    proj_magnitude = np.linalg.norm(proj_actions - raw_actions, axis=1)
    ax.hist(proj_magnitude, bins=30, color='#a371f7', edgecolor='white', alpha=0.85)
    ax.axvline(proj_magnitude.mean(), color='#ff6b35', ls='--', lw=2,
               label=f"Mean = {proj_magnitude.mean():.3f}")
    ax.set_xlabel("Projection Magnitude (L2 norm)", fontsize=10)
    ax.set_ylabel("Frequency", fontsize=10)
    ax.set_title("Safety Projection Magnitude Distribution", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path / "fig36_action_space.png", dpi=200, bbox_inches='tight')
    plt.close(fig)
    logger.info("Saved: %s", out_path / "fig36_action_space.png")

    # Summary
    feasible_mask = cti_grid >= 0.3
    n_feasible = feasible_mask.sum()
    n_total = cti_grid.size
    print("\\n" + "=" * 60)
    print("ACTION SPACE GEOMETRY SUMMARY")
    print("=" * 60)
    print(f"Grid points:           {n_total}")
    print(f"Feasible (CTI ≥ 0.3):  {n_feasible} ({n_feasible/n_total*100:.1f}%)")
    print(f"Infeasible:            {n_total - n_feasible} ({(n_total-n_feasible)/n_total*100:.1f}%)")
    print(f"Mean projection mag:   {proj_magnitude.mean():.4f}")
    print(f"Max projection mag:    {proj_magnitude.max():.4f}")
    print(f"Policy actions in feasible region: {np.sum(cti_grid[np.clip((raw_actions[:,0]*0.9+0.1 - 0.1)/0.9 * grid_size, 0, grid_size-1).astype(int), np.clip((raw_actions[:,1]*9.0+1.0 - 1.0)/9.0 * grid_size, 0, grid_size-1).astype(int)] >= 0.3)}/{len(raw_actions)}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Figure 36 — Action Space Geometry")
    parser.add_argument("--grid_size", type=int, default=40)
    parser.add_argument("--n_fault_locs", type=int, default=15)
    parser.add_argument("--train_steps", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="outputs")
    args = parser.parse_args()
    main(args.grid_size, args.n_fault_locs, args.train_steps, args.seed, args.output_dir)
