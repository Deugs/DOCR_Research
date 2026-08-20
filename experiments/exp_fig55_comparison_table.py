
"""
exp_fig55_comparison_table.py — Figure 55
───────────────────────────────────────────
Comprehensive Method Comparison Table.

Aggregates all experimental results into a single LaTeX/markdown table.
Columns: Algorithm, Objective (±std), CTI Violation %, Convergence Episodes,
Inference Latency (ms), Memory (MB), Robustness Score, N-1 Recovery Time.

Requires prior experimental results from Figures 5, 6, 12, 13, 14, 18.

Outputs
-------
  outputs/fig55_comparison_table.tex
  outputs/fig55_comparison_table.md
  outputs/fig55_comparison_table.png
"""

import numpy as np
import os
import sys
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.makedirs("outputs", exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from evaluation.experiments._synthetic_label import mark_synthetic

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

METHODS = ["Traditional", "GA-SQP", "IPPO", "MAPPO", "MAPPO-Comm",
           "MAPPO-Safe", "MAPPO-Safe-EWC", "HWOA", "HHO", "JAYA", "EEO"]

METRICS = [
    "objective_mean", "objective_std", "cti_violations", "cti_violations_std",
    "convergence_episodes", "inference_latency_ms", "memory_mb",
    "robustness_score", "n1_recovery_time", "sample_efficiency_auc"
]


def _read_npz(path: Path) -> Optional[dict]:
    """Best-effort load of a result .npz written by another figure script.
    Returns None (and logs why) instead of raising, so a missing/corrupt
    upstream file just means those metrics fall back to the synthetic path."""
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=True) as d:
            return dict(d)
    except Exception as exc:
        logger.warning("Could not read %s (%s); skipping", path, exc)
        return None


def load_or_synthesize_results(output_dir: str = "outputs",
                                seed: int = 42) -> Tuple[Dict[str, Dict[str, float]], bool]:
    """Load results from previous experiments where available; synthesize the rest.

    Returns (results, any_synthetic) — any_synthetic is True iff at least one
    method/metric had to be fabricated because no real upstream result file
    (or the expected key inside it) was found, so callers can honestly label
    the output when that happens.
    """
    out_path = Path(output_dir)
    results = {method: {} for method in METHODS}

    # These are written by their own generator scripts into the same outputs/
    # directory this script writes to (see each script's own --output_dir).
    fig05 = _read_npz(out_path / "fig05_pv_sweep.npz")
    fig06 = _read_npz(out_path / "fig06_cti_violations.npz")
    fig12 = _read_npz(out_path / "fig12_convergence.npz")
    fig41 = _read_npz(out_path / "fig41_sample_efficiency.npz")

    for method in METHODS:
        if fig05 is not None and f"{method}_mean_OF" in fig05 and f"{method}_std_OF" in fig05:
            # fig05 stores one value per PV level (PV_LEVELS); last entry = 100% PV,
            # matching this table's "IEEE 8-Bus, 100% PV" caption.
            results[method]["objective_mean"] = float(np.asarray(fig05[f"{method}_mean_OF"])[-1])
            results[method]["objective_std"] = float(np.asarray(fig05[f"{method}_std_OF"])[-1])

        if fig06 is not None and f"{method}_violation_rate" in fig06:
            results[method]["cti_violations"] = float(np.asarray(fig06[f"{method}_violation_rate"])[-1])
            std_key = f"{method}_violation_rate_std"
            if std_key in fig06:
                results[method]["cti_violations_std"] = float(np.asarray(fig06[std_key])[-1])

        if fig12 is not None and f"{method}_convergence_times" in fig12:
            ct = np.asarray(fig12[f"{method}_convergence_times"])
            if ct.size and np.any(ct):
                results[method]["convergence_episodes"] = float(np.mean(ct))

        if fig41 is not None and f"{method}_auc" in fig41:
            results[method]["sample_efficiency_auc"] = float(np.asarray(fig41[f"{method}_auc"]).reshape(-1)[0])

    if any(f is not None for f in (fig05, fig06, fig12, fig41)):
        loaded_methods = sorted({m for m in METHODS if results[m]})
        logger.info("Loaded real values for %d/%d methods from prior figure outputs: %s",
                    len(loaded_methods), len(METHODS), loaded_methods)
    else:
        logger.warning("No prior figure output .npz files found under %s; "
                        "all metrics will be synthetic", out_path)

    # Fallback characteristics used ONLY for whatever wasn't actually loaded above.
    base_of = {"Traditional": 12.5, "GA-SQP": 10.2, "IPPO": 9.8, "MAPPO": 8.5,
               "MAPPO-Comm": 7.8, "MAPPO-Safe": 7.2, "MAPPO-Safe-EWC": 6.8,
               "HWOA": 9.5, "HHO": 9.3, "JAYA": 9.7, "EEO": 9.4}

    base_cti = {"Traditional": 8.5, "GA-SQP": 3.2, "IPPO": 4.5, "MAPPO": 2.8,
                "MAPPO-Comm": 2.1, "MAPPO-Safe": 0.5, "MAPPO-Safe-EWC": 0.3,
                "HWOA": 3.8, "HHO": 3.5, "JAYA": 4.0, "EEO": 3.7}

    rng = np.random.default_rng(seed)
    any_synthetic = False
    synthetic_metrics = set()

    for method in METHODS:
        r = results[method]
        if "objective_mean" not in r:
            r["objective_mean"] = base_of[method]
            r["objective_std"] = base_of[method] * 0.05
            any_synthetic = True
            synthetic_metrics.add("objective_mean/std")
        elif "objective_std" not in r:
            r["objective_std"] = r["objective_mean"] * 0.05

        if "cti_violations" not in r:
            r["cti_violations"] = base_cti[method]
            r["cti_violations_std"] = base_cti[method] * 0.2
            any_synthetic = True
            synthetic_metrics.add("cti_violations")
        elif "cti_violations_std" not in r:
            r["cti_violations_std"] = r["cti_violations"] * 0.2

        for metric, lo, hi in [
            ("convergence_episodes", 2000, 8000),
            ("inference_latency_ms", 5, 50),
            ("memory_mb", 50, 500),
            ("robustness_score", 0.6, 0.95),
            ("n1_recovery_time", 1, 10),
            ("sample_efficiency_auc", 1000, 5000),
        ]:
            if metric not in r:
                r[metric] = float(rng.uniform(lo, hi))
                any_synthetic = True
                synthetic_metrics.add(metric)

    if any_synthetic:
        logger.warning("[WARNING] SYNTHETIC DATA used for: %s (seed=%d)",
                        ", ".join(sorted(synthetic_metrics)), seed)

    return results, any_synthetic


def generate_latex_table(results: Dict, output_path: Path, is_synthetic: bool = False):
    """Generate publication-ready LaTeX table."""
    synthetic_note = (
        r"\% SYNTHETIC DATA WARNING: one or more columns below were not found in a real"
        "\n" r"\% experiment output and were fabricated for illustration only." "\n"
        if is_synthetic else ""
    )
    latex = synthetic_note + r"""\begin{table*}[t]
\centering
\caption{Comprehensive Method Comparison (IEEE 8-Bus, 100\% PV)}
\label{tab:comparison}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lccccccccc}
\toprule
\textbf{Method} & \textbf{OF (s)} & \textbf{CTI Viol.} & \textbf{Conv. Ep.} & \textbf{Lat. (ms)} & \textbf{Mem. (MB)} & \textbf{Robust.} & \textbf{N-1 Rec.} & \textbf{AUC} \\
\midrule
"""

    for method in METHODS:
        r = results[method]
        of_str = f"${r['objective_mean']:.3f} \\pm {r['objective_std']:.3f}$"
        cti_str = f"${r['cti_violations']:.1f} \\pm {r['cti_violations_std']:.2f}$"
        conv_str = f"{r['convergence_episodes']:.0f}"
        lat_str = f"{r['inference_latency_ms']:.1f}"
        mem_str = f"{r['memory_mb']:.0f}"
        rob_str = f"{r['robustness_score']:.3f}"
        n1_str = f"{r['n1_recovery_time']:.1f}"
        auc_str = f"{r['sample_efficiency_auc']:.0f}"

        latex += f"{method:20s} & {of_str:25s} & {cti_str:20s} & {conv_str:10s} & {lat_str:10s} & {mem_str:10s} & {rob_str:10s} & {n1_str:10s} & {auc_str:10s} \\\\\n"

    latex += r"""\bottomrule
\end{tabular}%
}
\end{table*}
"""

    with open(output_path, "w") as f:
        f.write(latex)
    logger.info("Saved LaTeX table: %s", output_path)


def generate_markdown_table(results: Dict, output_path: Path, is_synthetic: bool = False):
    """Generate markdown table for README/documentation."""
    lines = ["# Comprehensive Method Comparison", ""]
    if is_synthetic:
        lines += [
            "**[WARNING] SYNTHETIC DATA** — one or more columns below were not found "
            "in a real experiment output (`outputs/fig05_pv_sweep.npz`, "
            "`fig06_cti_violations.npz`, `fig12_convergence.npz`, `fig41_sample_efficiency.npz`) "
            "and were fabricated for illustration only. Do not cite as experimental evidence.",
            "",
        ]
    lines += [
        "| Method | OF (s) | CTI Viol. | Conv. Ep. | Lat. (ms) | Mem. (MB) | Robust. | N-1 Rec. | AUC |",
        "|--------|--------|-----------|-----------|-----------|-----------|---------|----------|-----|",
    ]

    for method in METHODS:
        r = results[method]
        line = f"| {method:<18s} | {r['objective_mean']:>6.3f}±{r['objective_std']:.3f} | " \
               f"{r['cti_violations']:>5.1f}±{r['cti_violations_std']:.2f} | " \
               f"{r['convergence_episodes']:>7.0f} | {r['inference_latency_ms']:>7.1f} | " \
               f"{r['memory_mb']:>7.0f} | {r['robustness_score']:>5.3f} | " \
               f"{r['n1_recovery_time']:>6.1f} | {r['sample_efficiency_auc']:>5.0f} |"
        lines.append(line)

    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    logger.info("Saved Markdown table: %s", output_path)


def generate_heatmap(results: Dict, output_path: Path, is_synthetic: bool = False):
    """Generate heatmap visualization of the comparison table."""
    fig, ax = plt.subplots(figsize=(14, 8))

    # Build data matrix
    data = np.zeros((len(METHODS), len(METRICS)))
    for i, method in enumerate(METHODS):
        for j, metric in enumerate(METRICS):
            data[i, j] = results[method][metric]

    # Normalize each column to [0, 1] for visualization
    data_norm = np.zeros_like(data)
    for j in range(data.shape[1]):
        col = data[:, j]
        if np.std(col) > 0:
            # For metrics where lower is better, invert
            if METRICS[j] in ["objective_mean", "cti_violations", "convergence_episodes",
                               "inference_latency_ms", "memory_mb", "n1_recovery_time"]:
                data_norm[:, j] = 1.0 - (col - col.min()) / (col.max() - col.min())
            else:
                data_norm[:, j] = (col - col.min()) / (col.max() - col.min())

    im = ax.imshow(data_norm, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(METRICS)))
    ax.set_xticklabels([m.replace("_", " ").title() for m in METRICS], rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(METHODS)))
    ax.set_yticklabels(METHODS, fontsize=9)

    # Add text annotations
    for i in range(len(METHODS)):
        for j in range(len(METRICS)):
            val = data[i, j]
            text_color = "white" if data_norm[i, j] < 0.5 else "black"
            if METRICS[j] in ["objective_mean", "cti_violations"]:
                text = f"{val:.2f}"
            else:
                text = f"{val:.1f}"
            ax.text(j, i, text, ha="center", va="center", fontsize=6, color=text_color)

    ax.set_title("Figure 55 — Comprehensive Method Comparison Heatmap", fontsize=12, pad=20)
    plt.colorbar(im, ax=ax, shrink=0.6, label="Normalized Score (green = better)")

    fig.tight_layout()
    if is_synthetic:
        mark_synthetic(fig, note="one or more columns fabricated; no real result file found")
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved heatmap: %s", output_path)


def main(output_dir: str = "outputs", seed: int = 42):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    logger.info("Figure 55 — Comprehensive Comparison Table")

    results, any_synthetic = load_or_synthesize_results(output_dir, seed)

    # Generate outputs
    generate_latex_table(results, out_path / "fig55_comparison_table.tex", any_synthetic)
    generate_markdown_table(results, out_path / "fig55_comparison_table.md", any_synthetic)
    generate_heatmap(results, out_path / "fig55_comparison_table.png", any_synthetic)

    # Also save as NPZ
    save_dict = {}
    for method in METHODS:
        for metric in METRICS:
            save_dict[f"{method}_{metric}"] = np.array([results[method][metric]])
    np.savez(out_path / "fig55_comparison_table.npz", **save_dict)
    logger.info("Saved: %s", out_path / "fig55_comparison_table.npz")

    print("\\n" + "=" * 80)
    print("COMPARISON TABLE SUMMARY")
    print("=" * 80)
    print(f"{'Method':<20} {'OF':>8} {'CTI':>6} {'Conv':>6} {'Lat':>6} {'Mem':>6} {'Rob':>6}")
    print("-" * 80)
    for method in METHODS:
        r = results[method]
        print(f"{method:<20} {r['objective_mean']:>8.3f} {r['cti_violations']:>6.1f} "
              f"{r['convergence_episodes']:>6.0f} {r['inference_latency_ms']:>6.1f} "
              f"{r['memory_mb']:>6.0f} {r['robustness_score']:>6.3f}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Figure 55 — Comparison Table")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--seed", type=int, default=42,
                         help="Seed for the synthetic fallback RNG (used only for metrics "
                              "not found in a real prior figure output)")
    args = parser.parse_args()
    main(args.output_dir, args.seed)
