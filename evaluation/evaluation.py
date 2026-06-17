"""
evaluation.py
─────────────
Metrics, comparison tables, and matplotlib figures for
GA-SQP (control) vs MARL-IPPO (proposed).
Supports all IEEE test systems: 3, 8, 14, 30, 39, 57, 118 buses.

Phase 1 Enhancement: Multi-fault-type comparison, PV penetration analysis,
topology robustness plots, and curriculum learning progress visualization.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from network.relay_physics import check_constraints, CTI, FAULT_TYPES

STYLE = {
    "figure.facecolor": "#0d1117",
    "axes.facecolor":   "#161b22",
    "axes.edgecolor":   "#30363d",
    "axes.labelcolor":  "#8b949e",
    "axes.titlecolor":  "#e6edf3",
    "xtick.color":      "#8b949e",
    "ytick.color":      "#8b949e",
    "grid.color":       "#21262d",
    "grid.linewidth":   0.8,
    "text.color":       "#e6edf3",
    "legend.facecolor": "#161b22",
    "legend.edgecolor": "#30363d",
    "font.family":      "monospace",
}
plt.rcParams.update(STYLE)

COL_GA   = "#2563eb"
COL_MARL = "#39d353"
COL_PRI  = "#00d4ff"
COL_BKP  = "#f7c948"
COL_CTI  = "#ff6b35"
COL_PV   = "#a371f7"   # Phase 1: PV color
COL_TOP  = "#f778ba"   # Phase 1: Topology color


def _ax(ax, title, xlabel="", ylabel=""):
    ax.set_title(title, fontsize=9, pad=6)
    ax.set_xlabel(xlabel, fontsize=7.5)
    ax.set_ylabel(ylabel, fontsize=7.5)
    ax.grid(True, alpha=0.4)
    ax.tick_params(labelsize=7)


def plot_convergence(results_dict, out_path):
    """Plot convergence curves for GA-SQP vs MARL-IPPO."""
    nets = list(results_dict.keys())
    fig, axes = plt.subplots(1, len(nets), figsize=(14, 4))
    if len(nets) == 1: axes = [axes]
    fig.suptitle("Convergence: GA-SQP vs MARL-IPPO",
                 fontsize=11, y=1.01)
    for ax, key in zip(axes, nets):
        res = results_dict[key]
        gh  = res.get("GA-SQP", {}).get("ga_history", [])
        mh  = res.get("MARL",   {}).get("history",    [])
        if gh:
            ax.plot(gh, color=COL_GA,   lw=1.8,
                    label=f"GA-SQP  → {res['GA-SQP']['OF']:.3f}s")
        if mh:
            k = np.ones(8)/8
            ax.plot(np.convolve(mh, k, 'same') if len(mh)>20 else mh,
                    color=COL_MARL, lw=1.8,
                    label=f"MARL → {res['MARL']['OF']:.3f}s")
        _ax(ax, key.upper(), "Iteration/Episode", "OF (s)")
        ax.legend(fontsize=7); ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  ✓ {out_path}")


def plot_relay_times(res_ga, res_marl, net, out_path, fault_type="3ph"):
    """Plot primary/backup relay operating times."""
    c_ga   = check_constraints(net, res_ga["TMS"],   res_ga["PS"], fault_type=fault_type)
    c_marl = check_constraints(net, res_marl["TMS"], res_marl["PS"], fault_type=fault_type)
    x, w   = np.arange(net.n_pairs), 0.35
    fig, axes = plt.subplots(2, 1, figsize=(14, 7))
    fig.suptitle(f"Primary/Backup Times — {net.name} ({fault_type})", fontsize=11)
    for ax, r, c, title in [
        (axes[0], res_ga,   c_ga,   "GA-SQP (Control)"),
        (axes[1], res_marl, c_marl, "MARL-IPPO (Proposed)"),
    ]:
        ax.bar(x-w/2, c["t_pri"], w, label="Primary", color=COL_PRI, alpha=0.85)
        ax.bar(x+w/2, c["t_bkp"], w, label="Backup",  color=COL_BKP, alpha=0.85)
        viol = c["delta_cti"] < CTI - 1e-6
        if np.any(viol):
            ax.scatter(x[viol], c["t_bkp"][viol]+0.02, marker="v",
                       color=COL_CTI, s=40, zorder=5,
                       label=f"CTI viol ({viol.sum()})")
        ax.axhline(CTI, color=COL_CTI, lw=0.8, ls="--",
                   label=f"CTI={CTI}s")
        _ax(ax, f"{title}  |  OF={r['OF']:.3f}s  |  "
                f"CTI ok: {c['cti_satisfied']}",
            "Pair Index", "Time (s)")
        ax.legend(fontsize=7)
        step = max(1, net.n_pairs // 20)
        ax.set_xticks(x[::step])
        ax.set_xticklabels(x[::step], fontsize=6)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig); print(f"  ✓ {out_path}")


def plot_cti_distribution(res_ga, res_marl, net, out_path, fault_type="3ph"):
    """Plot CTI margin distribution histogram."""
    c_ga   = check_constraints(net, res_ga["TMS"],   res_ga["PS"], fault_type=fault_type)
    c_marl = check_constraints(net, res_marl["TMS"], res_marl["PS"], fault_type=fault_type)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"CTI Distribution — {net.name} ({fault_type})", fontsize=11)
    for ax, c, label, col in [
        (axes[0], c_ga,   "GA-SQP",    COL_GA),
        (axes[1], c_marl, "MARL-IPPO", COL_MARL),
    ]:
        d = c["delta_cti"]
        ax.hist(d, bins=20, color=col, alpha=0.8, edgecolor="#0d1117")
        ax.axvline(CTI, color=COL_CTI, lw=1.5, ls="--",
                   label=f"CTI min={CTI}s")
        ax.axvline(float(np.mean(d)), color="#fff", lw=1.0, ls=":",
                   label=f"mean={np.mean(d):.3f}s")
        _ax(ax, f"{label}  |  violations: "
                f"{int(np.sum(d < CTI))}/{len(d)}",
            "CTI (s)", "Count")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig); print(f"  ✓ {out_path}")


def plot_relay_settings(res_ga, res_marl, net, out_path):
    """Plot optimal relay settings (TMS vs PS)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle(f"Optimal Relay Settings — {net.name}", fontsize=11)
    ax = axes[0]
    ax.scatter(res_ga["TMS"],   res_ga["PS"],
               color=COL_GA,   alpha=0.7, s=30, label="GA-SQP")
    ax.scatter(res_marl["TMS"], res_marl["PS"],
               color=COL_MARL, alpha=0.7, s=30, label="MARL", marker="^")
    _ax(ax, "TMS vs PS", "TMS (s)", "PS (pu)")
    ax.legend(fontsize=7)
    ax2 = axes[1]
    ri = np.arange(net.n_relays)
    ax2.plot(ri, res_ga["TMS"],   "o-", color=COL_GA,
             lw=1.2, ms=4, alpha=0.85, label="TMS — GA-SQP")
    ax2.plot(ri, res_marl["TMS"], "^-", color=COL_MARL,
             lw=1.2, ms=4, alpha=0.85, label="TMS — MARL")
    _ax(ax2, "TMS per Relay", "Relay Index", "TMS (s)")
    ax2.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig); print(f"  ✓ {out_path}")


def plot_summary(results_dict, out_path):
    """Cross-system summary comparison."""
    nets    = list(results_dict.keys())
    of_ga   = [results_dict[n]["GA-SQP"]["OF"] for n in nets]
    of_marl = [results_dict[n]["MARL"]["OF"]   for n in nets]
    t_ga    = [results_dict[n]["GA-SQP"]["elapsed"] for n in nets]
    t_marl  = [results_dict[n]["MARL"]["elapsed"]   for n in nets]
    cv_ga   = [results_dict[n]["GA-SQP"]["constraints"]["cti_violations"]
               for n in nets]
    cv_marl = [results_dict[n]["MARL"]["constraints"]["cti_violations"]
               for n in nets]
    x, w = np.arange(len(nets)), 0.3
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.suptitle("GA-SQP vs MARL-IPPO — Summary", fontsize=11)
    for ax, ga_v, ml_v, title, ylabel in [
        (axes[0], of_ga,   of_marl, "Total Operating Time", "OF (s)"),
        (axes[1], cv_ga,   cv_marl, "CTI Violations",       "# violations"),
        (axes[2], t_ga,    t_marl,  "Computation Time",     "Seconds"),
    ]:
        ax.bar(x-w/2, ga_v, w, label="GA-SQP",   color=COL_GA,   alpha=0.85)
        ax.bar(x+w/2, ml_v, w, label="MARL-IPPO", color=COL_MARL, alpha=0.85)
        _ax(ax, title, "Test System", ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels([n.upper() for n in nets])
        ax.legend(fontsize=7)
    for i, (g, m) in enumerate(zip(of_ga, of_marl)):
        imp = (g - m) / g * 100
        axes[0].text(i, max(g, m)*1.02, f"{imp:+.1f}%",
                     ha="center", fontsize=7,
                     color=COL_MARL if imp > 0 else COL_CTI)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig); print(f"  ✓ {out_path}")


# ── Phase 1 New Plots ─────────────────────────────────────────────────────────

def plot_pv_penetration_analysis(results_by_pv, net_name, out_path):
    """
    Phase 1: Plot objective function vs PV penetration level.

    Parameters:
        results_by_pv: Dict mapping pv_level -> {method -> result_dict}
        net_name: Network name for title
        out_path: Output file path
    """
    pv_levels = sorted(results_by_pv.keys())

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle(f"PV Penetration Analysis — {net_name}", fontsize=11)

    # OF vs PV level
    ax = axes[0]
    of_ga = [results_by_pv[pv]["GA-SQP"]["OF"] for pv in pv_levels]
    of_marl = [results_by_pv[pv]["MARL"]["OF"] for pv in pv_levels]

    ax.plot(pv_levels, of_ga, "o-", color=COL_GA, lw=2, label="GA-SQP", markersize=6)
    ax.plot(pv_levels, of_marl, "^-", color=COL_MARL, lw=2, label="MARL-IPPO", markersize=6)
    ax.fill_between(pv_levels, of_ga, of_marl, alpha=0.1, color=COL_PV)
    _ax(ax, "Objective Function vs PV Penetration", "PV Penetration (%)", "OF (s)")
    ax.set_xticks(pv_levels)
    ax.set_xticklabels([f"{int(p*100)}%" for p in pv_levels])
    ax.legend(fontsize=7)

    # CTI violations vs PV level
    ax = axes[1]
    cv_ga = [results_by_pv[pv]["GA-SQP"]["constraints"]["cti_violations"] for pv in pv_levels]
    cv_marl = [results_by_pv[pv]["MARL"]["constraints"]["cti_violations"] for pv in pv_levels]

    ax.plot(pv_levels, cv_ga, "o-", color=COL_GA, lw=2, label="GA-SQP", markersize=6)
    ax.plot(pv_levels, cv_marl, "^-", color=COL_MARL, lw=2, label="MARL-IPPO", markersize=6)
    ax.axhline(0, color=COL_CTI, lw=1, ls="--", label="Target: 0 violations")
    _ax(ax, "CTI Violations vs PV Penetration", "PV Penetration (%)", "# Violations")
    ax.set_xticks(pv_levels)
    ax.set_xticklabels([f"{int(p*100)}%" for p in pv_levels])
    ax.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  ✓ {out_path}")


def plot_fault_type_comparison(results_by_fault, net_name, out_path):
    """
    Phase 1: Plot performance across different fault types.

    Parameters:
        results_by_fault: Dict mapping fault_type -> {method -> result_dict}
        net_name: Network name for title
    """
    fault_types = list(results_by_fault.keys())

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(fault_types))
    w = 0.35

    of_ga = [results_by_fault[ft]["GA-SQP"]["OF"] for ft in fault_types]
    of_marl = [results_by_fault[ft]["MARL"]["OF"] for ft in fault_types]

    ax.bar(x - w/2, of_ga, w, label="GA-SQP", color=COL_GA, alpha=0.85)
    ax.bar(x + w/2, of_marl, w, label="MARL-IPPO", color=COL_MARL, alpha=0.85)

    _ax(ax, f"Fault Type Generalization — {net_name}", "Fault Type", "OF (s)")
    ax.set_xticks(x)
    ax.set_xticklabels(fault_types)
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  ✓ {out_path}")


def plot_curriculum_progress(stage_history, out_path):
    """
    Phase 1: Plot curriculum learning progress across stages.

    Parameters:
        stage_history: Dict mapping stage_idx -> list of OF values
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    stages = sorted(stage_history.keys())
    for stage in stages:
        history = stage_history[stage]
        if history:
            # Moving average
            window = min(50, len(history) // 4 + 1)
            if len(history) > window:
                smoothed = np.convolve(history, np.ones(window)/window, mode='valid')
            else:
                smoothed = history
            ax.plot(smoothed, lw=1.5, label=f"Stage {stage} (PV {int(stage*20)}%)", alpha=0.8)

    _ax(ax, "Curriculum Learning Progress", "Episode", "Objective Function (s)")
    ax.legend(fontsize=7)
    ax.set_yscale("log")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  ✓ {out_path}")


def plot_topology_robustness(results_by_topology, net_name, out_path):
    """
    Phase 1: Plot performance under different topology changes (N-1).

    Parameters:
        results_by_topology: Dict mapping topology_desc -> {method -> result_dict}
    """
    topologies = list(results_by_topology.keys())

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(topologies))
    w = 0.35

    of_ga = [results_by_topology[t]["GA-SQP"]["OF"] for t in topologies]
    of_marl = [results_by_topology[t]["MARL"]["OF"] for t in topologies]

    ax.bar(x - w/2, of_ga, w, label="GA-SQP", color=COL_GA, alpha=0.85)
    ax.bar(x + w/2, of_marl, w, label="MARL-IPPO", color=COL_MARL, alpha=0.85)

    _ax(ax, f"Topology Robustness (N-1) — {net_name}", "Topology Scenario", "OF (s)")
    ax.set_xticks(x)
    ax.set_xticklabels(topologies, rotation=45, ha="right", fontsize=6)
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  ✓ {out_path}")


def print_comparison_table(results_dict):
    """Print formatted comparison table."""
    LINE = "─" * 100
    print(f"\n{LINE}")
    print(f"  {'Network':<12} {'Method':<14} {'OF (s)':>9} "
          f"{'CTI viol':>9} {'CTI min':>10} {'Time (s)':>10} "
          f"{'Fault':>8} {'PV%':>6}")
    print(LINE)
    for net_key, res in results_dict.items():
        for meth in ("GA-SQP", "MARL"):
            r = res[meth]
            c = r["constraints"]
            ft = r.get("fault_type", "3ph")
            pv = r.get("pv_level", 0.0) * 100
            print(f"  {net_key:<12} {meth:<14} "
                  f"{r['OF']:>9.4f} {c['cti_violations']:>9d} "
                  f"{c['cti_min']:>10.4f} {r['elapsed']:>10.2f} "
                  f"{ft:>8} {pv:>5.0f}%")
        imp = (res["GA-SQP"]["OF"] - res["MARL"]["OF"]) \
              / res["GA-SQP"]["OF"] * 100
        print(f"  {'':12} {'Improvement':<14} {imp:>+8.2f}%\n")
    print(LINE)


def print_pv_analysis_table(results_by_pv, net_name):
    """Phase 1: Print PV penetration analysis table."""
    LINE = "─" * 90
    print(f"\n{LINE}")
    print(f"  PV Penetration Analysis — {net_name}")
    print(LINE)
    print(f"  {'PV Level':>8} {'Method':>12} {'OF (s)':>10} {'CTI Viol':>10} "
          f"{'CTI Min':>10} {'Time (s)':>10}")
    print(LINE)
    for pv in sorted(results_by_pv.keys()):
        for meth in ("GA-SQP", "MARL"):
            r = results_by_pv[pv][meth]
            c = r["constraints"]
            print(f"  {pv*100:>7.0f}% {meth:>12} "
                  f"{r['OF']:>10.4f} {c['cti_violations']:>10d} "
                  f"{c['cti_min']:>10.4f} {r['elapsed']:>10.2f}")
        print()
    print(LINE)