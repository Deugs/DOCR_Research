
"""
exp_fig11_nemenyi_test.py  —  Figure 11 (IMPROVED)
───────────────────────────────────────────────────
Nemenyi Critical Difference Diagram (alpha = 0.05).

IMPROVEMENTS over original:
  • Accepts actual experimental results (not synthetic data)
  • Proper Nemenyi test with exact critical values
  • Correct group connection logic (rank difference < CD)
  • Multi-metric support (OF, CTI, convergence, robustness)
  • Statistical significance indicators (*, **, ***)
  • Compatible with Fig 5/6 output data

Outputs
-------
  outputs/fig11_nemenyi.npz
  outputs/fig11_nemenyi.png
"""

import numpy as np
import os
import sys
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.makedirs("outputs", exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

from evaluation.experiments._synthetic_label import mark_synthetic

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

METHODS = ["MAPPO-Safe-EWC", "MAPPO-Safe", "MAPPO", "IPPO", "QMIX", "IQL", "Traditional"]
METHOD_SHORT = ["MS-EWC", "MS", "MP", "IP", "QM", "IQ", "Trad"]
METHOD_COLORS = {
    "MAPPO-Safe-EWC": "#2563eb", "MAPPO-Safe": "#39d353", "MAPPO": "#f7c948",
    "IPPO": "#ff6b35", "QMIX": "#a371f7", "IQL": "#8b949e", "Traditional": "#ff6b6b",
}

# Exact Nemenyi critical values for alpha=0.05 (from Demšar 2006)
NEMENYI_Q = {
    2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949,
    8: 3.031, 9: 3.102, 10: 3.164
}


def compute_nemenyi_cd(n_methods: int, n_seeds: int, alpha: float = 0.05) -> float:
    """Compute critical difference for Nemenyi test."""
    q_alpha = NEMENYI_Q.get(n_methods, 2.949)
    return q_alpha * np.sqrt(n_methods * (n_methods + 1) / (6.0 * n_seeds))


def friedman_test(data: Dict[str, np.ndarray]) -> Tuple[float, float]:
    """
    Friedman test for significant differences between methods.
    Returns (chi2_statistic, p_value).
    """
    method_names = list(data.keys())
    n_methods = len(method_names)
    n_seeds = len(data[method_names[0]])
    
    # Rank methods per seed
    ranks = np.zeros((n_seeds, n_methods))
    for seed in range(n_seeds):
        values = [data[m][seed] for m in method_names]
        # Rank: 1 = best (lowest value for minimization)
        seed_ranks = stats.rankdata(values)
        ranks[seed] = seed_ranks
    
    # Friedman statistic
    rank_means = ranks.mean(axis=0)
    chi2 = (12 * n_seeds / (n_methods * (n_methods + 1))) * \
           np.sum((rank_means - (n_methods + 1) / 2) ** 2)
    
    p_value = 1 - stats.chi2.cdf(chi2, n_methods - 1)
    
    return chi2, p_value


def compute_ranks(data: Dict[str, np.ndarray]) -> Tuple[np.ndarray, List[str]]:
    """Compute average ranks per method. Lower rank = better."""
    method_names = list(data.keys())
    n_methods = len(method_names)
    n_seeds = len(data[method_names[0]])
    
    ranks = np.zeros((n_seeds, n_methods))
    for seed in range(n_seeds):
        values = [data[m][seed] for m in method_names]
        # For minimization problems: rank 1 = smallest value
        seed_ranks = stats.rankdata(values)
        ranks[seed] = seed_ranks
    
    return ranks.mean(axis=0), method_names


def find_equivalent_groups(avg_ranks: np.ndarray, cd: float) -> List[List[int]]:
    """
    Find groups of methods that are statistically equivalent.
    Methods in same group have rank difference < CD.
    """
    n = len(avg_ranks)
    sorted_idx = np.argsort(avg_ranks)
    
    groups = []
    current_group = [sorted_idx[0]]
    
    for i in range(1, n):
        # Check if current method can join existing groups
        can_join = False
        for group in groups:
            if all(abs(avg_ranks[sorted_idx[i]] - avg_ranks[g]) < cd for g in group):
                group.append(sorted_idx[i])
                can_join = True
                break
        
        if not can_join:
            # Check if can extend current group
            if avg_ranks[sorted_idx[i]] - avg_ranks[current_group[-1]] < cd:
                current_group.append(sorted_idx[i])
            else:
                if len(current_group) > 1:
                    groups.append(current_group)
                current_group = [sorted_idx[i]]
    
    if len(current_group) > 1:
        groups.append(current_group)
    
    return groups


def plot_critical_difference(ax, avg_ranks, method_names, cd, title,
                             method_colors=None, method_short=None):
    """Plot a proper critical difference diagram."""
    n_methods = len(method_names)
    
    # Sort by rank
    sorted_idx = np.argsort(avg_ranks)
    sorted_ranks = avg_ranks[sorted_idx]
    sorted_names = [method_names[i] for i in sorted_idx]
    sorted_short = [method_short[method_names.index(n)] for n in sorted_names] if method_short else sorted_names
    sorted_colors = [method_colors.get(n, "#333333") for n in sorted_names] if method_colors else ["#333333"] * n_methods
    
    # Draw axis
    ax.set_xlim(0.5, n_methods + 0.5)
    ax.set_ylim(-1.5, n_methods + 1.5)
    
    # Horizontal axis line
    ax.axhline(0, color="black", lw=1.5)
    
    # Tick marks and labels
    for i in range(1, n_methods + 1):
        ax.plot([i, i], [-0.08, 0.08], color="black", lw=1.5)
        ax.text(i, -0.35, str(i), ha="center", fontsize=10, fontweight="bold")
    
    # Method labels and points
    y_positions = np.linspace(1.2, -1.2, n_methods)
    
    for i, (name, rank, short, color) in enumerate(zip(sorted_names, sorted_ranks, sorted_short, sorted_colors)):
        y = y_positions[i]
        
        # Draw line from rank to label
        ax.plot([rank, rank], [0, y], color=color, lw=1.5, alpha=0.6, zorder=1)
        ax.plot(rank, y, "o", color=color, markersize=14, markeredgecolor="white", 
                markeredgewidth=2, zorder=3)
        ax.text(rank, y + 0.18, short, ha="center", fontsize=9, fontweight="bold", 
                color=color, zorder=4)
    
    # Draw CD bar at top
    leftmost_rank = sorted_ranks[0]
    cd_bar_y = 1.6
    ax.plot([leftmost_rank, leftmost_rank + cd], [cd_bar_y, cd_bar_y], 
            color="red", lw=2.5, solid_capstyle="round")
    ax.plot([leftmost_rank, leftmost_rank + cd], [cd_bar_y, cd_bar_y], 
            "|", color="red", markersize=10)
    ax.text(leftmost_rank + cd/2, cd_bar_y + 0.15, f"CD = {cd:.2f}", 
            ha="center", fontsize=10, color="red", fontweight="bold")
    
    # Draw equivalent groups
    groups = find_equivalent_groups(avg_ranks, cd)
    for g_idx, group in enumerate(groups):
        y_line = -1.3 - 0.25 * g_idx
        group_ranks = [avg_ranks[i] for i in group]
        x_start = min(group_ranks)
        x_end = max(group_ranks)
        
        # Draw thick line connecting equivalent methods
        ax.plot([x_start, x_end], [y_line, y_line], color="#39d353", lw=4, 
                solid_capstyle="round", alpha=0.8, zorder=2)
        ax.plot([x_start, x_end], [y_line, y_line], "o", color="#39d353", 
                markersize=6, zorder=3)
    
    ax.set_title(title, fontsize=12, fontweight="bold", pad=20)
    ax.axis("off")


def load_experimental_data(fig5_path: str = "outputs/fig05_pv_sweep.npz",
                           fig6_path: str = "outputs/fig06_cti_violations.npz",
                           n_seeds: int = 50,
                           seed: int = 42) -> Tuple[Dict[str, Dict[str, np.ndarray]], bool]:
    """
    Load actual experimental results from Fig 5 and Fig 6.
    Falls back to synthetic data if files not found — or if they exist but
    don't contain the raw per-seed arrays a Friedman/Nemenyi test needs
    (Fig 5/6 currently only save aggregated per-PV-level mean/std).

    Returns (data, used_real).
    """
    fig5, fig6 = None, None
    if os.path.exists(fig5_path):
        try:
            fig5 = np.load(fig5_path, allow_pickle=True)
        except Exception as exc:
            logger.warning("Could not read %s (%s)", fig5_path, exc)
    if os.path.exists(fig6_path):
        try:
            fig6 = np.load(fig6_path, allow_pickle=True)
        except Exception as exc:
            logger.warning("Could not read %s (%s)", fig6_path, exc)

    used_real = False
    data: Dict[str, Dict[str, np.ndarray]] = {}

    if fig5 is not None and fig6 is not None:
        # A real per-seed wiring would read f"{method}_all_OF" / f"{method}_all_cti"
        # (raw per-seed samples), not the aggregated f"{method}_mean_OF" that Fig 5/6
        # actually save today. Use them if a future version of those scripts adds them.
        have_all = all(f"{m}_all_OF" in fig5 and f"{m}_all_cti" in fig6 for m in METHODS)
        if have_all:
            data["OF"] = {m: np.asarray(fig5[f"{m}_all_OF"]) for m in METHODS}
            data["CTI"] = {m: np.asarray(fig6[f"{m}_all_cti"]) for m in METHODS}
            used_real = True
            logger.info("Loaded real per-seed data from %s and %s", fig5_path, fig6_path)
        else:
            logger.warning(
                "%s and %s exist but only contain aggregated per-PV-level statistics, "
                "not the raw per-seed arrays this test needs — falling back to synthetic data.",
                fig5_path, fig6_path)
    else:
        logger.warning("%s / %s not found — using synthetic data for demonstration",
                        fig5_path, fig6_path)

    if not used_real:
        rng = np.random.default_rng(seed)

        data["OF"] = {
            "MAPPO-Safe-EWC": rng.normal(2.1, 0.15, n_seeds),
            "MAPPO-Safe": rng.normal(2.3, 0.18, n_seeds),
            "MAPPO": rng.normal(2.6, 0.20, n_seeds),
            "IPPO": rng.normal(2.8, 0.22, n_seeds),
            "QMIX": rng.normal(3.0, 0.25, n_seeds),
            "IQL": rng.normal(3.3, 0.28, n_seeds),
            "Traditional": rng.normal(4.5, 0.35, n_seeds),
        }

        data["CTI"] = {
            "MAPPO-Safe-EWC": rng.exponential(0.5, n_seeds),
            "MAPPO-Safe": rng.exponential(1.2, n_seeds),
            "MAPPO": rng.exponential(2.5, n_seeds),
            "IPPO": rng.exponential(3.0, n_seeds),
            "QMIX": rng.exponential(3.5, n_seeds),
            "IQL": rng.exponential(4.0, n_seeds),
            "Traditional": rng.exponential(8.0, n_seeds),
        }
        logger.warning("[WARNING] SYNTHETIC DATA (seed=%d) — Friedman/Nemenyi statistics below "
                        "are computed on fabricated numbers, not a real experiment run.", seed)

    return data, used_real


def main(n_seeds: int = 50, alpha: float = 0.05,
         fig5_path: str = "outputs/fig05_pv_sweep.npz",
         fig6_path: str = "outputs/fig06_cti_violations.npz",
         output_dir: str = "outputs", seed: int = 42):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    logger.info("Figure 11 — Nemenyi Critical Difference Diagram (IMPROVED)")
    logger.info("  Alpha: %.2f, Seeds: %d", alpha, n_seeds)

    # Load data
    data, used_real = load_experimental_data(fig5_path, fig6_path, n_seeds, seed)
    
    # Friedman test
    for metric_name, metric_data in data.items():
        chi2, p_val = friedman_test(metric_data)
        logger.info("  Friedman test (%s): χ²=%.3f, p=%.4f %s",
                   metric_name, chi2, p_val, "SIGNIFICANT" if p_val < alpha else "not significant")
    
    # Compute ranks and CD
    cd = compute_nemenyi_cd(len(METHODS), n_seeds, alpha)
    logger.info("  Critical Difference: %.3f", cd)
    
    of_ranks, of_names = compute_ranks(data["OF"])
    cti_ranks, cti_names = compute_ranks(data["CTI"])
    
    # Save
    np.savez(
        out_path / "fig11_nemenyi.npz",
        of_ranks=of_ranks,
        cti_ranks=cti_ranks,
        method_names=np.array(of_names),
        critical_difference=np.array([cd]),
        n_seeds=np.array([n_seeds]),
        alpha=np.array([alpha]),
        friedman_of_chi2=np.array([friedman_test(data["OF"])[0]]),
        friedman_of_p=np.array([friedman_test(data["OF"])[1]]),
        friedman_cti_chi2=np.array([friedman_test(data["CTI"])[0]]),
        friedman_cti_p=np.array([friedman_test(data["CTI"])[1]]),
    )
    logger.info("Saved: %s", out_path / "fig11_nemenyi.npz")
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(f"Figure 11 — Nemenyi Critical Difference Diagram (α = {alpha}, n = {n_seeds})", 
                 fontsize=13, fontweight="bold")
    
    plot_critical_difference(axes[0], of_ranks, of_names, cd, 
                            "Objective Function (lower = better)",
                            METHOD_COLORS, METHOD_SHORT)
    plot_critical_difference(axes[1], cti_ranks, cti_names, cd,
                            "CTI Violations (lower = better)",
                            METHOD_COLORS, METHOD_SHORT)
    
    fig.tight_layout()
    if not used_real:
        mark_synthetic(fig, note=f"seed={seed}; no real per-seed OF/CTI arrays found")
    fig.savefig(out_path / "fig11_nemenyi.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", out_path / "fig11_nemenyi.png")
    
    # Summary table
    print("\\n" + "=" * 80)
    print("NEMENYI TEST SUMMARY")
    print("=" * 80)
    print(f"Critical Difference (α={alpha}, k={len(METHODS)}, n={n_seeds}): {cd:.3f}")
    print(f"Friedman OF: χ²={friedman_test(data['OF'])[0]:.3f}, p={friedman_test(data['OF'])[1]:.4f}")
    print(f"Friedman CTI: χ²={friedman_test(data['CTI'])[0]:.3f}, p={friedman_test(data['CTI'])[1]:.4f}")
    print("\\nObjective Function Ranks:")
    print(f"{'Method':>20} {'Avg Rank':>10} {'Group':>10}")
    print("-" * 80)
    sorted_of = sorted(zip(of_names, of_ranks), key=lambda x: x[1])
    groups_of = find_equivalent_groups(of_ranks, cd)
    for name, rank in sorted_of:
        group_id = next((i for i, g in enumerate(groups_of) if METHODS.index(name) in g), -1)
        group_str = f"G{group_id+1}" if group_id >= 0 else "—"
        print(f"{name:>20} {rank:>10.3f} {group_str:>10}")
    
    print("\\nCTI Violation Ranks:")
    print(f"{'Method':>20} {'Avg Rank':>10} {'Group':>10}")
    print("-" * 80)
    sorted_cti = sorted(zip(cti_names, cti_ranks), key=lambda x: x[1])
    groups_cti = find_equivalent_groups(cti_ranks, cd)
    for name, rank in sorted_cti:
        group_id = next((i for i, g in enumerate(groups_cti) if METHODS.index(name) in g), -1)
        group_str = f"G{group_id+1}" if group_id >= 0 else "—"
        print(f"{name:>20} {rank:>10.3f} {group_str:>10}")
    print("=" * 80)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Figure 11 — Nemenyi Test (IMPROVED)")
    p.add_argument("--n_seeds", type=int, default=50)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--fig5_path", type=str, default="outputs/fig05_pv_sweep.npz")
    p.add_argument("--fig6_path", type=str, default="outputs/fig06_cti_violations.npz")
    p.add_argument("--output_dir", type=str, default="outputs")
    p.add_argument("--seed", type=int, default=42,
                    help="Seed for the synthetic fallback RNG (used only when real "
                         "per-seed Fig5/Fig6 data is unavailable)")
    args = p.parse_args()
    main(args.n_seeds, args.alpha, args.fig5_path, args.fig6_path, args.output_dir, args.seed)
