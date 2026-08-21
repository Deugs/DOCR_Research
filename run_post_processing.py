#!/usr/bin/env python3
"""
run_post_processing.py — Complete Figure Generation from Cached Core Training Data
─────────────────────────────────────────────────────────────────────────────────
Generates ALL 58 thesis figures by post-processing cached core training results (A-K).
NO training is performed here — pure plotting and analysis.

REFACTORED: Added config file support via docl_config.py, progress tracking, 
ETA estimation, batch processing, and rich output.

Configuration (precedence: CLI > config file > preset > env vars > defaults):
  • --config FILE       Load YAML/JSON config file
  • --preset NAME       Use built-in preset (8bus_full, 8bus_fast, etc.)
  • --system NAME       Override system (3bus, 8bus, 14bus, ...)
  • --fast              Enable fast mode (reduced iterations)
  • All CLI args from docl_config.py supported

DATA PROVENANCE WARNING (2026-08-07): every fig_NN_* function below that
follows the pattern `data = load_cache(name); if data is None: data =
{<hardcoded literals>}` now sets `is_synthetic = data is None`, logs a
`logger.warning(...)` naming the missing cache, and watermarks the saved
figure via `mark_synthetic()` whenever that fallback actually triggers (41
functions total, including fig_18_inference_latency and
fig_53_model_compression — see tests/test_phase6.py / test_phase7.py for
regression coverage). Only 3 of the 11 core caches referenced across this file (A/B/C:
"core_A_mappo_safe_ewc_8bus_100pv", "core_B_curriculum", "core_C_baselines")
currently exist in results/core_cache/ — see cache_index.json. Caches D
through K (ablation, hparam, scalability, robustness, transfer, deployment,
protection, seasonal) are all missing, so most figures generated from a
fresh checkout will still be synthetic; they are now just honestly labeled
as such rather than silently presented as real. Run the relevant Core
training (see EXPERIMENT_GUIDE.md) to replace them with real data.
"""

import argparse
import os
import sys
import json
import time
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
import logging
import threading

# ──────────────────────────────────────────────────────────────────
# Import Configuration System
# ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.absolute()
sys.path.insert(0, str(BASE_DIR))

try:
    from docl_config import (
        DOCRConfig, SystemConfig, TrainingConfig, ExperimentConfig, PostProcessingConfig,
        build_config_parser, config_from_args, print_config_summary,
        generate_example_config, list_presets, get_preset
    )
    CONFIG_SYSTEM_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import docl_config.py: {e}")
    print("Falling back to minimal config implementation.")
    CONFIG_SYSTEM_AVAILABLE = False

# ──────────────────────────────────────────────────────────────────
# Configuration & Paths (config-driven or fallback)
# ──────────────────────────────────────────────────────────────────
# Global config object - populated from CLI/config file
_config: Optional[DOCRConfig] = None

def get_config() -> DOCRConfig:
    """Get the current configuration. Raises if not initialized."""
    if _config is None:
        raise RuntimeError("Configuration not initialized. Call initialize_config() first.")
    return _config

def initialize_config(args: argparse.Namespace) -> DOCRConfig:
    """Initialize configuration from CLI args using docl_config system."""
    global _config
    if CONFIG_SYSTEM_AVAILABLE:
        _config = config_from_args(args)
    else:
        # Fallback minimal config
        _config = DOCRConfig()
        if hasattr(args, 'fast') and args.fast:
            _config.experiment.fast_mode = True
            _config.apply_fast_mode()
    return _config

# Directory paths derived from config
def get_figures_dir() -> Path:
    """Get figures directory from config or default."""
    if _config is not None:
        return Path(_config.postprocessing.figures_dir)
    return BASE_DIR / "results" / "figures"

def get_cache_dir() -> Path:
    """Get cache directory from config or default."""
    if _config is not None:
        return Path(_config.experiment.cache_dir)
    return BASE_DIR / "results" / "core_cache"

def get_reports_dir() -> Path:
    """Get reports directory from config or default."""
    if _config is not None:
        return Path(_config.experiment.output_dir) / "reports"
    return BASE_DIR / "results" / "reports"

# Lazy-evaluated paths (resolved at runtime after config is loaded)
#@property
def CACHE_DIR() -> Path:
    return get_cache_dir()

#@property
def FIGURES_DIR() -> Path:
    return get_figures_dir()

#@property
def REPORT_DIR() -> Path:
    return get_reports_dir()

# ──────────────────────────────────────────────────────────────────
# Directory Setup (config-driven)
# ──────────────────────────────────────────────────────────────────
def setup_directories(config: DOCRConfig):
    """Create all output directories based on configuration."""
    fig_dir = Path(config.postprocessing.figures_dir)

    for subdir in ["figs_01_10", "figs_11_20", "figs_21_30", "figs_31_40",
                   "figs_41_50", "figs_51_58"]:
        (fig_dir / subdir).mkdir(parents=True, exist_ok=True)

    reports_dir = Path(config.experiment.output_dir) / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    return fig_dir, reports_dir

# ──────────────────────────────────────────────────────────────────
# Logging Setup
# ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────
# Progress Tracking
# ──────────────────────────────────────────────────────────────────
class ProgressTracker:
    """Thread-safe progress tracker with ETA estimation."""

    def __init__(self, total: int, desc: str = "", unit: str = "fig", 
                 width: int = 40, enable: bool = True):
        self.total = max(total, 1)
        self.desc = desc
        self.unit = unit
        self.width = width
        self.enable = enable and sys.stdout.isatty()
        self.current = 0
        self._lock = threading.Lock()
        self._start_time = time.time()
        self._last_update = self._start_time
        self._times = []
        self._printed = False
        self._postfix = {}

    def update(self, n: int = 1):
        with self._lock:
            now = time.time()
            if n > 0:
                self._times.append((n, now - self._last_update))
                if len(self._times) > 20:
                    self._times.pop(0)
            self.current += n
            self._last_update = now
            if self.enable:
                self._draw()

    def set_postfix(self, **kwargs):
        self._postfix = kwargs
        if self.enable:
            self._draw()

    def _draw(self):
        if not self.enable:
            return
        frac = min(self.current / self.total, 1.0)
        filled = int(self.width * frac)
        bar = "█" * filled + "░" * (self.width - filled)

        eta_str = "calculating..."
        if self._times:
            total_n = sum(t[0] for t in self._times)
            total_time = sum(t[1] for t in self._times)
            if total_n > 0 and total_time > 0:
                rate = total_n / total_time
                remaining = (self.total - self.current) / rate
                eta_str = str(timedelta(seconds=int(remaining)))

        elapsed = timedelta(seconds=int(time.time() - self._start_time))
        postfix = ""
        if self._postfix:
            postfix = " | " + ", ".join(f"{k}={v}" for k, v in self._postfix.items())

        line = f"\r{self.desc} [{bar}] {self.current}/{self.total} ({frac*100:.1f}%) | Elapsed: {elapsed} | ETA: {eta_str}{postfix}"
        sys.stdout.write(line)
        sys.stdout.flush()
        self._printed = True

    def close(self):
        with self._lock:
            if self._printed and self.enable:
                elapsed = timedelta(seconds=int(time.time() - self._start_time))
                sys.stdout.write(f"\r{self.desc} [{'█' * self.width}] {self.total}/{self.total} (100.0%) | Total: {elapsed} — DONE\n")
                sys.stdout.flush()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class FigureProgress:
    """Tracks figure generation progress across multiple figures."""

    def __init__(self, total_figures: int):
        self.total_figures = total_figures
        self.completed = []
        self.failed = []
        self._start_time = time.time()
        self._lock = threading.Lock()

    def add_completed(self, fig_num: int, duration: float):
        with self._lock:
            self.completed.append({"fig": fig_num, "duration": duration})

    def add_failed(self, fig_num: int, error: str):
        with self._lock:
            self.failed.append({"fig": fig_num, "error": error})

    def print_summary(self):
        total_elapsed = time.time() - self._start_time
        logger.info(f"\n{'='*70}")
        logger.info("FIGURE GENERATION SUMMARY")
        logger.info(f"{'='*70}")
        logger.info(f"Total time: {timedelta(seconds=int(total_elapsed))}")
        logger.info(f"Completed: {len(self.completed)}/{self.total_figures}")
        logger.info(f"Failed: {len(self.failed)}/{self.total_figures}")

        if self.completed:
            avg_dur = sum(f["duration"] for f in self.completed) / len(self.completed)
            logger.info(f"Average time per figure: {avg_dur:.2f}s")

        if self.failed:
            logger.info(f"\nFailed figures:")
            for f in self.failed:
                logger.info(f"  Fig {f['fig']}: {f['error']}")

        logger.info(f"{'='*70}")

    def save_report(self, path: Path):
        report = {
            "timestamp": datetime.now().isoformat(),
            "total_figures": self.total_figures,
            "completed": len(self.completed),
            "failed": len(self.failed),
            "total_elapsed_seconds": time.time() - self._start_time,
            "completed_figures": self.completed,
            "failed_figures": self.failed,
        }
        with open(path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        logger.info(f"Figure report saved to: {path}")


# ──────────────────────────────────────────────────────────────────
# Cache Loading (config-driven paths)
# ──────────────────────────────────────────────────────────────────
def load_cache1(name: str):
    """Load a cached core result by name."""
    cache_file = get_cache_dir() / f"{name}.pkl"
    if not cache_file.exists():
        index_file = get_cache_dir() / "cache_index.json"
        if index_file.exists():
            with open(index_file) as f:
                index = json.load(f)
            for key, info in index.items():
                if info.get("name") == name:
                    cache_file = get_cache_dir() / f"{key}.pkl"
                    break
    if cache_file.exists():
        import pickle
        with open(cache_file, "rb") as f:
            return pickle.load(f)
    logger.warning(f"Cache not found: {name}. Using synthetic fallback.")
    return None

def load_cache(name: str):
    """Load a cached core result by name, checking both old and new cache formats."""
    # Try old format first
    cache_file = get_cache_dir() / f"{name}.pkl"
    if not cache_file.exists():
        index_file = get_cache_dir() / "cache_index.json"
        if index_file.exists():
            with open(index_file) as f:
                index = json.load(f)
            # A Core's cache config gets a new content-hash key every
            # time it's re-run under a different config (a bumped
            # version marker after a bug fix, a different --fast/n_runs
            # setting, ...) — the OLD key/entry is never removed from
            # the index. Picking the first name-match in dict iteration
            # order (== insertion/chronological order) silently kept
            # returning whichever run happened to be cached FIRST under
            # that name forever, regardless of which one is actually the
            # intended, current result. Prefer the most recently created
            # entry — the standard "last write wins" cache semantic, and
            # the correct one as long as the cache directory doesn't
            # accumulate leftover smoke-test artifacts alongside real
            # runs (clean those up rather than relying on some other
            # heuristic, e.g. size, to guess around them — see git
            # history around 2026-08-20 for a case where exactly that
            # had happened and briefly caused this function to prefer a
            # larger but pre-bugfix cached result over a smaller, fresh,
            # correct one).
            matches = [(key, info) for key, info in index.items() if info.get("name") == name]
            if matches:
                key, _ = max(matches, key=lambda kv: kv[1].get("created", ""))
                cache_file = get_cache_dir() / f"{key}.pkl"

    if cache_file.exists():
        import pickle
        with open(cache_file, "rb") as f:
            return pickle.load(f)
    
    # Try new bridge cache format
    # Map figure names to bridge cache files
    bridge_map = {
        "core_G_robustness": ["core_eval_fig13.json", "core_eval_fig20.json", 
                              "core_eval_fig22.json", "core_eval_fig24.json", "core_eval_fig54.json"],
        "core_H_transfer": ["core_eval_fig23.json", "core_eval_fig25.json"],
        "core_I_deployment": ["core_eval_fig53.json"],  # fig18 is latency, handled separately
        "core_J_protection": ["core_eval_fig44.json", "core_eval_fig45.json", 
                             "core_eval_fig46.json", "core_eval_fig47.json"],
        "core_K_seasonal": ["core_eval_fig58.json"],
    }
    
    bridge_files = bridge_map.get(name, [])
    for bf in bridge_files:
        bridge_path = get_cache_dir() / bf
        if bridge_path.exists():
            with open(bridge_path, 'r') as f:
                data = json.load(f)
            logger.info(f"Loaded bridge cache: {bridge_path}")
            return data.get("data", {})
    
    logger.warning(f"Cache not found: {name}. Using synthetic fallback.")
    return None


def load_bridge_cache(fig_num: int) -> Optional[Dict]:
    """Load the real evaluation result CoreEvalBridge cached for THIS
    exact figure number (results/core_cache/core_eval_fig{NN}.json).

    Unlike load_cache()'s bridge_map, which maps a Core NAME to a list of
    files and returns the first one that happens to exist (silently
    handing figure N's caller a different figure's cache data if N's own
    file is missing but an earlier one in the list isn't — see fig 44/47
    both living under "core_J_protection"), this looks up the one file
    that can only ever contain fig_num's own data.
    """
    cache_file = get_cache_dir() / f"core_eval_fig{fig_num:02d}.json"
    if not cache_file.exists():
        return None
    with open(cache_file) as f:
        payload = json.load(f)
    logger.info(f"Loaded bridge cache: {cache_file}")
    return payload.get("data")


def _annotate_panel_synthetic(ax, note: str) -> None:
    """Lightweight per-axes counterpart to mark_synthetic() for figures
    where only ONE panel is fabricated while another panel in the same
    figure is real — a full-figure diagonal watermark would misrepresent
    the real panel as fake too."""
    ax.text(0.5, 0.5, f"ILLUSTRATIVE\n(not measured — {note})",
            transform=ax.transAxes, fontsize=10, color="red", alpha=0.45,
            ha="center", va="center", rotation=25, zorder=1000)
    logger.warning(f"[panel-synthetic] {note}")


def save_fig(fig, path: Path, dpi: int = None):
    """Save figure and close. Uses config DPI if available."""
    if dpi is None and _config is not None:
        dpi = _config.postprocessing.dpi
    elif dpi is None:
        dpi = 200
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved: {path}")


def get_fig_path(fig_num: int, subdir: str) -> Path:
    """Get figure save path using config format."""
    fmt = "pdf"
    if _config is not None:
        fmt = _config.postprocessing.fig_format

    fig_dir = get_figures_dir()
    return fig_dir / subdir / f"fig_{fig_num:02d}_{get_fig_name(fig_num)}.{fmt}"


def get_fig_name(fig_num: int) -> str:
    """Get figure name from mapping."""
    names = {
        2: "data_flow", 3: "8bus_topology", 4: "30bus_topology",
        5: "objective_vs_pv", 6: "cti_vs_pv", 7: "curriculum",
        8: "attention_weights", 9: "cti_violations", 10: "lagrange_multipliers",
        11: "nemenyi", 12: "convergence_bands", 13: "communication_stress",
        14: "n1_contingency", 15: "scalability", 16: "ewc_effectiveness",
        18: "inference_latency", 19: "pv_ride_through", 20: "byzantine",
        21: "action_projection", 22: "bandwidth_tradeoff", 23: "unseen_faults",
        24: "sensor_robustness", 25: "zero_shot_transfer", 29: "ewc_continual",
        31: "multi_metric_ranking", 32: "effect_size", 33: "radar_chart",
        34: "nn_architecture", 35: "observation_space", 36: "action_space_geometry",
        37: "per_agent_curves", 38: "attention_evolution", 39: "ablation",
        40: "hyperparameter", 41: "sample_efficiency", 42: "warm_start",
        43: "coordination_diagram", 44: "fault_contribution", 45: "daily_profile",
        46: "relay_curves", 47: "sympathetic_tripping", 48: "shap",
        49: "decision_boundary", 50: "critical_pairs", 52: "digital_twin_pipeline",
        53: "model_compression", 54: "adversarial", 55: "comparison_table",
        56: "pareto_front", 57: "bootstrap_ci", 58: "seasonal",
    }
    return names.get(fig_num, "figure")


def timed_figure(fig_num: int, fig_progress: FigureProgress = None):
    """Decorator for timing figure generation."""
    def decorator(func: Callable):
        def wrapper(*args, **kwargs):
            t0 = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - t0
                if fig_progress:
                    fig_progress.add_completed(fig_num, duration)
                logger.info(f"[Fig {fig_num:02d}] ✓ Generated in {duration:.2f}s")
                return result
            except Exception as e:
                duration = time.time() - t0
                if fig_progress:
                    fig_progress.add_failed(fig_num, str(e))
                logger.error(f"[Fig {fig_num:02d}] ✗ Failed after {duration:.2f}s: {e}")
                raise
        return wrapper
    return decorator

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

# ──────────────────────────────────────────────────────────────────
# Cache Loading
# ──────────────────────────────────────────────────────────────────
@timed_figure(2)
def fig_02_data_flow():
    """Fig 2: Data Flow Architecture — schematic diagram."""
    logger.info("\\n[Fig 02] Data Flow Architecture")
    fig, ax = plt.subplots(figsize=(14, 8))
    stages = ["OpenDSS\\nSimulation", "State\\nExtraction", "GNN\\nEncoder",
              "Actor-Critic", "Safety\\nProjection", "Action\\nExecution", "Reward\\nComputation"]
    x_pos = np.linspace(0.05, 0.95, len(stages))
    y_pos = [0.5] * len(stages)
    colors = ["#2563eb", "#39d353", "#f7c948", "#a371f7", "#ff6b35", "#39d353", "#2563eb"]
    
    for i, (x, y, stage, c) in enumerate(zip(x_pos, y_pos, stages, colors)):
        ax.add_patch(plt.Rectangle((x-0.06, y-0.08), 0.12, 0.16, 
                                    facecolor=c, edgecolor="white", linewidth=2, alpha=0.85))
        ax.text(x, y, stage, ha="center", va="center", fontsize=9, fontweight="bold", color="white")
        if i < len(stages) - 1:
            ax.annotate("", xy=(x_pos[i+1]-0.06, y), xytext=(x+0.06, y),
                       arrowprops=dict(arrowstyle="->", color="#ff6b35", lw=2))
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0.2, 0.8)
    ax.set_title("Figure 2 — Data Flow Architecture (Single Training Step)", fontsize=13)
    ax.axis("off")
    save_fig(fig, get_fig_path(2, "figs_01_10"))


# ==================================================================
# FIGURE 3: IEEE 8-Bus Topology
# ==================================================================
@timed_figure(3)
def fig_03_8bus_topology():
    """Fig 3: 8-Bus Test System Topology with DOCR Placement."""
    logger.info("\\n[Fig 03] IEEE 8-Bus Topology")
    fig, ax = plt.subplots(figsize=(10, 8))
    buses = {1: (0.2, 0.8), 2: (0.5, 0.9), 3: (0.8, 0.8), 4: (0.3, 0.5),
             5: (0.7, 0.5), 6: (0.2, 0.2), 7: (0.5, 0.1), 8: (0.8, 0.2)}
    lines = [(1,2), (2,3), (1,4), (2,4), (2,5), (3,5), (4,5), (4,6), (5,6), (6,7), (7,8), (5,8)]
    
    for (i, j) in lines:
        x1, y1 = buses[i]
        x2, y2 = buses[j]
        ax.plot([x1, x2], [y1, y2], 'k-', lw=2, alpha=0.6)
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.plot(mx, my, 'ro', markersize=8, markerfacecolor='#ff6b35', markeredgecolor='white', markeredgewidth=1.5)
    
    for bus, (x, y) in buses.items():
        ax.add_patch(plt.Circle((x, y), 0.03, facecolor='#2563eb', edgecolor='white', linewidth=2))
        ax.text(x, y, str(bus), ha='center', va='center', fontsize=10, fontweight='bold', color='white')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.set_title("Figure 3 — IEEE 8-Bus Test System with DOCR Placement (0% PV)", fontsize=13)
    ax.axis('off')
    save_fig(fig, get_fig_path(3, "figs_01_10"))


# ==================================================================
# FIGURE 4: IEEE 30-Bus Topology
# ==================================================================
@timed_figure(4)
def fig_04_30bus_topology():
    """Fig 4: 30-Bus Test System Topology."""
    logger.info("\\n[Fig 04] IEEE 30-Bus Topology")
    fig, ax = plt.subplots(figsize=(12, 10))
    np.random.seed(42)
    n_buses = 30
    angles = np.linspace(0, 2*np.pi, n_buses, endpoint=False)
    radius = 0.4
    buses = {i+1: (0.5 + radius * np.cos(a), 0.5 + radius * np.sin(a)) for i, a in enumerate(angles)}
    
    for i in range(n_buses):
        j = (i + 1) % n_buses
        x1, y1 = buses[i+1]
        x2, y2 = buses[j+1]
        ax.plot([x1, x2], [y1, y2], 'k-', lw=1, alpha=0.4)
    for i in [0, 5, 10, 15, 20, 25]:
        x1, y1 = buses[i+1]
        ax.plot([0.5, x1], [0.5, y1], 'k-', lw=1, alpha=0.3)
    
    for bus, (x, y) in buses.items():
        color = '#2563eb' if bus <= 10 else '#39d353' if bus <= 20 else '#f7c948'
        ax.add_patch(plt.Circle((x, y), 0.02, facecolor=color, edgecolor='white', linewidth=1.5))
        ax.text(x, y-0.05, str(bus), ha='center', va='top', fontsize=7)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.set_title("Figure 4 — IEEE 30-Bus Test System with DOCR Placement", fontsize=13)
    ax.axis('off')
    save_fig(fig, get_fig_path(4, "figs_01_10"))


# ==================================================================
# FIGURE 5: Objective Function vs. PV Penetration
# ==================================================================
@timed_figure(5)
def fig_05_objective_vs_pv():
    """Fig 5: Objective vs PV — from Core B curriculum evaluations."""
    logger.info("\\n[Fig 05] Objective Function vs. PV Penetration")
    data = load_cache("core_B_curriculum")
    is_synthetic = data is None
    if data is None:
        logger.warning("Fig 05: 'core_B_curriculum' cache not found — using synthetic/hardcoded fallback data, NOT a real measurement.")
        pv_levels = [0.0, 0.20, 0.40, 0.60]
        data = {"task_evaluations": [{pv: {"objective": 1.0 + pv * 1.5 + np.random.rand() * 0.3} for pv in pv_levels} for _ in range(10)]}

    # Core B's active sweep trains on [0.0, 0.20, 0.40, 0.60] PV (see
    # run_core_training.py's TASKS) — was [0, .25, .5, .75, 1.0] before
    # the sweep was rescoped to 0-60% PV, which no longer matches the
    # real task_evaluations keys and raised a KeyError on real data.
    pv_levels = [0.0, 0.20, 0.40, 0.60]
    agg = {pv: [] for pv in pv_levels}
    for eval_run in data.get("task_evaluations", []):
        for pv, vals in eval_run.items():
            if isinstance(pv, str): pv = float(pv)
            agg[pv].append(vals.get("objective", 0))
    
    means = [np.mean(agg[pv]) for pv in pv_levels]
    stds = [np.std(agg[pv]) for pv in pv_levels]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.errorbar(pv_levels, means, yerr=stds, fmt="o-", color="#2563eb", lw=2, markersize=10, capsize=5, label="MAPPO-Safe-EWC")
    ax.set_xlabel("PV Penetration", fontsize=12)
    ax.set_ylabel("Objective Function (s)", fontsize=12)
    ax.set_title("Figure 5 — Objective Function vs. PV Penetration (8-Bus)", fontsize=13)
    ax.set_xticks(pv_levels)
    ax.set_xticklabels([f"{int(p*100)}%" for p in pv_levels])
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    if is_synthetic:
        from evaluation.experiments._synthetic_label import mark_synthetic
        mark_synthetic(fig, note="core_B_curriculum cache missing")
    save_fig(fig, get_fig_path(5, "figs_01_10"))


# ==================================================================
# FIGURE 6: CTI Violation Rate vs. PV Penetration
# ==================================================================
@timed_figure(6)
def fig_06_cti_vs_pv():
    """Fig 6: CTI vs PV — from Core B curriculum evaluations."""
    logger.info("\\n[Fig 06] CTI Violation Rate vs. PV Penetration")
    data = load_cache("core_B_curriculum")
    is_synthetic = data is None
    if data is None:
        logger.warning("Fig 06: 'core_B_curriculum' cache not found — using synthetic/hardcoded fallback data, NOT a real measurement.")
        pv_levels = [0.0, 0.20, 0.40, 0.60]
        data = {"task_evaluations": [{pv: {"cti_rate": max(0, 0.5 - pv * 0.4 + np.random.rand() * 0.1)} for pv in pv_levels} for _ in range(10)]}

    pv_levels = [0.0, 0.20, 0.40, 0.60]
    agg = {pv: [] for pv in pv_levels}
    for eval_run in data.get("task_evaluations", []):
        for pv, vals in eval_run.items():
            if isinstance(pv, str): pv = float(pv)
            agg[pv].append(vals.get("cti_rate", 0))
    
    means = [np.mean(agg[pv]) for pv in pv_levels]
    stds = [np.std(agg[pv]) for pv in pv_levels]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.errorbar(pv_levels, means, yerr=stds, fmt="s-", color="#ff6b35", lw=2, markersize=10, capsize=5)
    ax.set_xlabel("PV Penetration", fontsize=12)
    ax.set_ylabel("CTI Violation Rate (%)", fontsize=12)
    ax.set_title("Figure 6 — CTI Violation Rate vs. PV Penetration (8-Bus)", fontsize=13)
    ax.set_xticks(pv_levels)
    ax.set_xticklabels([f"{int(p*100)}%" for p in pv_levels])
    ax.grid(True, alpha=0.3)
    if is_synthetic:
        from evaluation.experiments._synthetic_label import mark_synthetic
        mark_synthetic(fig, note="core_B_curriculum cache missing")
    save_fig(fig, get_fig_path(6, "figs_01_10"))


# ==================================================================
# FIGURE 7: Curriculum Learning Progression
# ==================================================================
@timed_figure(7)
def fig_07_curriculum():
    """Fig 7: Curriculum progression — from Core B."""
    logger.info("\\n[Fig 07] Curriculum Learning Progression")
    data = load_cache("core_B_curriculum")
    is_synthetic = data is None
    if data is None:
        logger.warning("Fig 07: 'core_B_curriculum' cache not found — using synthetic/hardcoded fallback data, NOT a real measurement.")
        data = {"with_ewc": [{"task_performances": np.array([[0.8,0,0,0],[0.78,0.82,0,0],[0.75,0.80,0.85,0],[0.73,0.78,0.83,0.87]])}]}

    ewc_mats = [r["task_performances"] for r in data.get("with_ewc", [])]
    avg_mat = np.mean(ewc_mats, axis=0) if ewc_mats else np.eye(4) * 0.8
    n_tasks = avg_mat.shape[0]

    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.imshow(avg_mat, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    # Core B's active sweep is 4 stages: 0/20/40/60% PV (see run_core_training.py's TASKS)
    tasks = ["0%", "20%", "40%", "60%"][:n_tasks]
    ax.set_xticks(range(n_tasks))
    ax.set_yticks(range(n_tasks))
    ax.set_xticklabels([f"Eval {t}" for t in tasks], fontsize=10)
    ax.set_yticklabels([f"Train {t}" for t in tasks], fontsize=10)
    ax.set_xlabel("Evaluation Task (PV Level)", fontsize=12)
    ax.set_ylabel("Training Task (PV Level)", fontsize=12)
    ax.set_title("Figure 7 — Curriculum Learning Progression (MAPPO-Safe-EWC)", fontsize=13)
    for i in range(n_tasks):
        for j in range(n_tasks):
            if avg_mat[i, j] > 0:
                ax.text(j, i, f"{avg_mat[i, j]:.2f}", ha="center", va="center", fontsize=10, 
                       color="white" if avg_mat[i, j] < 0.5 else "black")
    plt.colorbar(im, ax=ax, shrink=0.8, label="Performance")
    if is_synthetic:
        from evaluation.experiments._synthetic_label import mark_synthetic
        mark_synthetic(fig, note="core_B_curriculum cache missing")
    save_fig(fig, get_fig_path(7, "figs_01_10"))


# ==================================================================
# FIGURE 8: Learned GNN Attention Weights
# ==================================================================
@timed_figure(8)
def fig_08_attention_weights():
    """Fig 8: Attention weights heatmap — from Core A final snapshot."""
    logger.info("\\n[Fig 08] Learned GNN Attention Weights")
    data = load_cache("core_A_mappo_safe_ewc_8bus_100pv")
    is_synthetic = data is None
    if data is None:
        logger.warning("Fig 08: 'core_A_mappo_safe_ewc_8bus_100pv' cache not found — using synthetic/hardcoded fallback data, NOT a real measurement.")
        n_agents = 7
        data = {f"attention_snap_{n_agents}_mean": np.random.dirichlet(np.ones(n_agents) * 3, n_agents)}
    
    snap_keys = [k for k in data if "attention_snap_" in k and "_mean" in k]
    if snap_keys:
        attn = data[snap_keys[-1]]
    else:
        attn = np.random.dirichlet(np.ones(7) * 3, 7)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Figure 8 — Learned GNN Attention Weights (8-Bus, 100% PV)", fontsize=13)
    
    ax = axes[0]
    im = ax.imshow(attn, cmap="viridis", aspect="auto")
    ax.set_title("Attention Matrix", fontsize=12)
    ax.set_xlabel("Agent (from)", fontsize=11)
    ax.set_ylabel("Agent (to)", fontsize=11)
    plt.colorbar(im, ax=ax, shrink=0.7)
    
    ax = axes[1]
    off_diag = attn[~np.eye(attn.shape[0], dtype=bool)].flatten()
    ax.hist(off_diag, bins=20, color="#2563eb", edgecolor="white", alpha=0.8)
    ax.set_xlabel("Attention Weight (off-diagonal)", fontsize=11)
    ax.set_ylabel("Frequency", fontsize=11)
    ax.set_title("Distribution of Attention Weights", fontsize=12)
    ax.grid(True, alpha=0.3)
    
    if is_synthetic:
        from evaluation.experiments._synthetic_label import mark_synthetic
        mark_synthetic(fig, note="core_A_mappo_safe_ewc_8bus_100pv cache missing")
    save_fig(fig, get_fig_path(8, "figs_01_10"))


# ==================================================================
# FIGURE 9: CTI Violation Dynamics
# ==================================================================
@timed_figure(9)
def fig_09_cti_violations():
    """Fig 9: CTI dynamics — from Core A."""
    logger.info("\\n[Fig 09] CTI Violation Dynamics During Training")
    data = load_cache("core_A_mappo_safe_ewc_8bus_100pv")
    is_synthetic = data is None
    if data is None:
        logger.warning("Fig 09: 'core_A_mappo_safe_ewc_8bus_100pv' cache not found — using synthetic/hardcoded fallback data, NOT a real measurement.")
        n_eps = 2000
        data = {"episodes": np.arange(n_eps), "cti_violation_rates_mean": np.maximum(0, 80 * np.exp(-np.arange(n_eps) / 400) + np.random.randn(n_eps) * 2),
                "cumulative_violations_mean": np.cumsum(np.random.poisson(2, n_eps)), "global_returns_mean": -5 + 8 * (1 - np.exp(-np.arange(n_eps) / 300))}
    
    eps = data.get("episodes", np.arange(len(data.get("cti_violation_rates_mean", []))))
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("Figure 9 — CTI Violation Dynamics During Training (8-Bus, 100% PV)", fontsize=13)
    
    ax = axes[0, 0]
    cti_mean = data.get("cti_violation_rates_mean", np.zeros_like(eps))
    cti_std = data.get("cti_violation_rates_std", np.zeros_like(eps) * 0.1)
    ax.plot(eps, cti_mean, color="#2563eb", lw=2, label="MAPPO-Safe-EWC")
    ax.fill_between(eps, cti_mean - cti_std, cti_mean + cti_std, alpha=0.2, color="#2563eb")
    ax.set_ylabel("CTI Violation Rate (%)", fontsize=10)
    ax.set_title("Per-Episode Violation Rate", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    ax = axes[0, 1]
    lag_mean = data.get("lagrange_norms_mean", np.zeros_like(eps))
    ax.plot(eps, lag_mean, color="#39d353", lw=2)
    ax.set_ylabel("||lambda|| (L2 norm)", fontsize=10)
    ax.set_title("Lagrange Multiplier Evolution", fontsize=11)
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 0]
    cum_mean = data.get("cumulative_violations_mean", np.cumsum(np.zeros_like(eps)))
    ax.plot(eps, cum_mean, color="#2563eb", lw=2)
    ax.set_ylabel("Cumulative CTI Violations", fontsize=10)
    ax.set_xlabel("Episode", fontsize=10)
    ax.set_title("Cumulative Violations", fontsize=11)
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 1]
    safe_mean = data.get("safety_activations_mean", np.zeros_like(eps))
    ax.plot(eps, safe_mean, color="#a371f7", lw=2)
    ax.set_ylabel("Safety Activations", fontsize=10)
    ax.set_xlabel("Episode", fontsize=10)
    ax.set_title("Safety Projection Frequency", fontsize=11)
    ax.grid(True, alpha=0.3)
    
    if is_synthetic:
        from evaluation.experiments._synthetic_label import mark_synthetic
        mark_synthetic(fig, note="core_A_mappo_safe_ewc_8bus_100pv cache missing")
    save_fig(fig, get_fig_path(9, "figs_01_10"))


# ==================================================================
# FIGURE 10: Lagrange Multiplier Evolution
# ==================================================================
@timed_figure(10)
def fig_10_lagrange_multipliers():
    """Fig 10: Lagrange evolution — from Core A."""
    logger.info("\\n[Fig 10] Lagrange Multiplier Evolution")
    data = load_cache("core_A_mappo_safe_ewc_8bus_100pv")
    is_synthetic = data is None
    if data is None:
        logger.warning("Fig 10: 'core_A_mappo_safe_ewc_8bus_100pv' cache not found — using synthetic/hardcoded fallback data, NOT a real measurement.")
        n_eps = 2000
        data = {"episodes": np.arange(n_eps), "lagrange_norms_mean": 0.5 + 8 * (1 - np.exp(-np.arange(n_eps) / 300)) + np.random.randn(n_eps) * 0.2,
                "lagrange_norms_std": np.ones(n_eps) * 0.5}
    
    eps = data.get("episodes", np.arange(len(data.get("lagrange_norms_mean", []))))
    lag_mean = data.get("lagrange_norms_mean", np.zeros_like(eps))
    lag_std = data.get("lagrange_norms_std", np.zeros_like(eps))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(eps, lag_mean, color="#2563eb", lw=2, label="Mean ||lambda||")
    ax.fill_between(eps, lag_mean - lag_std, lag_mean + lag_std, alpha=0.2, color="#2563eb")
    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Lagrange Multiplier Norm ||lambda||", fontsize=12)
    ax.set_title("Figure 10 — Lagrange Multiplier Evolution per P/B Pair", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    if is_synthetic:
        from evaluation.experiments._synthetic_label import mark_synthetic
        mark_synthetic(fig, note="core_A_mappo_safe_ewc_8bus_100pv cache missing")
    save_fig(fig, get_fig_path(10, "figs_01_10"))


# ==================================================================
# FIGURE 11: Nemenyi Critical Difference
# ==================================================================
@timed_figure(11)
def fig_11_nemenyi():
    """Fig 11: Nemenyi test — from Core C baseline comparison."""
    logger.info("\\n[Fig 11] Nemenyi Critical Difference Diagram")
    data = load_cache("core_C_baselines_8bus")
    is_synthetic = data is None
    if data is None:
        logger.warning("Fig 11: 'core_C_baselines_8bus' cache not found — using synthetic/hardcoded fallback data, NOT a real measurement.")
        methods = ["MAPPO-Safe-EWC", "MAPPO-Safe", "MAPPO", "IPPO", "Traditional"]
        data = {m: np.random.normal(3.0 - i * 0.3, 0.5, 50) for i, m in enumerate(methods)}
    
    methods = list(data.keys())
    n_methods = len(methods)
    final_perf = {m: np.mean(v[-100:] if hasattr(v, '__len__') and len(v) > 100 else v) for m, v in data.items()}
    sorted_methods = sorted(methods, key=lambda m: final_perf[m], reverse=True)
    ranks = {m: i + 1 for i, m in enumerate(sorted_methods)}
    
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.set_xlim(0.5, n_methods + 0.5)
    ax.set_ylim(-0.5, 1.5)
    ax.axhline(0.5, color="black", lw=1)
    for i in range(1, n_methods + 1):
        ax.axvline(i, color="gray", alpha=0.3, ls="--")
        ax.text(i, 0.3, str(i), ha="center", fontsize=12, fontweight="bold")
    
    y_pos = 1.0
    for m in sorted_methods:
        ax.text(ranks[m], y_pos, m, ha="center", fontsize=10,
               bbox=dict(boxstyle="round,pad=0.3", facecolor="#2563eb" if "EWC" in m else "#f7c948", alpha=0.7))
        y_pos = 0.0 if y_pos == 1.0 else 1.0
    
    ax.set_title("Figure 11 — Nemenyi Critical Difference Diagram (alpha = 0.05)", fontsize=13)
    ax.axis("off")
    if is_synthetic:
        from evaluation.experiments._synthetic_label import mark_synthetic
        mark_synthetic(fig, note="core_C_baselines_8bus cache missing")
    save_fig(fig, get_fig_path(11, "figs_11_20"))


# ==================================================================
# FIGURE 12: Convergence Curves with 95% CI
# ==================================================================
@timed_figure(12)
def fig_12_convergence_bands():
    """Fig 12: Convergence — from Core A + Core C combined."""
    logger.info("\\n[Fig 12] Convergence Curves with 95% Confidence Bands")
    core_A = load_cache("core_A_mappo_safe_ewc_8bus_100pv")
    core_C = load_cache("core_C_baselines_8bus")
    is_synthetic = core_A is None or core_C is None

    if is_synthetic:
        logger.warning("Fig 12: 'core_A_mappo_safe_ewc_8bus_100pv' and/or 'core_C_baselines_8bus' cache not found — using synthetic/hardcoded fallback data, NOT a real measurement.")
        n_eps = 2000
        methods = {
            "MAPPO-Safe-EWC": {"mean": -5 + 8.5 * (1 - np.exp(-np.arange(n_eps) / 350)), "std": np.ones(n_eps) * 0.8},
            "MAPPO-Safe": {"mean": -5 + 7.8 * (1 - np.exp(-np.arange(n_eps) / 450)), "std": np.ones(n_eps) * 0.9},
            "MAPPO": {"mean": -5 + 7.0 * (1 - np.exp(-np.arange(n_eps) / 550)), "std": np.ones(n_eps) * 1.0},
            "IPPO": {"mean": -5 + 6.5 * (1 - np.exp(-np.arange(n_eps) / 650)), "std": np.ones(n_eps) * 1.1},
            "Traditional": {"mean": np.ones(n_eps) * 2.0, "std": np.ones(n_eps) * 0.2},
        }
    else:
        methods = {}
        methods["MAPPO-Safe-EWC"] = {"mean": core_A.get("global_returns_mean", np.zeros(2000)), "std": core_A.get("global_returns_std", np.zeros(2000))}
        for method, returns in core_C.items():
            methods[method] = {"mean": returns.mean(axis=0) if hasattr(returns, 'mean') else np.zeros(2000),
                              "std": returns.std(axis=0) if hasattr(returns, 'std') else np.zeros(2000)}
    
    colors = {"MAPPO-Safe-EWC": "#2563eb", "MAPPO-Safe": "#39d353", "MAPPO": "#f7c948", "IPPO": "#ff6b35", "Traditional": "#ff6b6b"}
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Figure 12 — Convergence Curves with 95% Confidence Bands (8-Bus, 100% PV)", fontsize=13)
    
    ax = axes[0]
    episodes = np.arange(len(list(methods.values())[0]["mean"]))
    for method, data in methods.items():
        mean = data["mean"][:len(episodes)]
        std = data["std"][:len(episodes)]
        se = std / np.sqrt(50)
        ax.plot(episodes, mean, color=colors.get(method, "#333"), lw=2, label=method)
        ax.fill_between(episodes, mean - 1.96*se, mean + 1.96*se, alpha=0.1, color=colors.get(method, "#333"))
    ax.set_xlabel("Episode", fontsize=11)
    ax.set_ylabel("Episode Return", fontsize=11)
    ax.set_title("Learning Curves", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    ax = axes[1]
    # Only mean/std per method are cached (not raw per-episode returns),
    # so this panel always resamples a distribution around them rather
    # than showing genuine per-episode data — true even when the curves
    # in the left panel are real. Seeded so re-running is deterministic.
    rng_box = np.random.default_rng(42)
    final_returns = []
    labels = []
    for method, data in methods.items():
        if method == "Traditional": continue
        final_mean = data["mean"][-100:].mean() if len(data["mean"]) > 100 else data["mean"][-1]
        final_std = data["std"][-100:].mean() if len(data["std"]) > 100 else data["std"][-1]
        final_returns.append(rng_box.normal(final_mean, final_std, 50))
        labels.append(method)

    bp = ax.boxplot(final_returns, tick_labels=labels, patch_artist=True)
    for patch, label in zip(bp["boxes"], labels):
        patch.set_facecolor(colors.get(label, "#333"))
        patch.set_alpha(0.7)
    ax.set_ylabel("Final Return (last 100 episodes)", fontsize=11)
    ax.set_title("Final Performance Distribution", fontsize=12)
    ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    _annotate_panel_synthetic(ax, "resampled from cached mean/std, not raw per-episode returns")

    if is_synthetic:
        from evaluation.experiments._synthetic_label import mark_synthetic
        mark_synthetic(fig, note="core_A_mappo_safe_ewc_8bus_100pv/core_C_baselines_8bus cache missing")
    save_fig(fig, get_fig_path(12, "figs_11_20"))

# ==================================================================
# FIGURE 13: Communication Stress
# ==================================================================
@timed_figure(13)
def fig_13_communication_stress():
    """Fig 13: Performance under communication stress — from Core G."""
    logger.info("\\n[Fig 13] Communication Stress Robustness")
    data = load_cache("core_G_robustness")
    is_synthetic = data is None
    if data is None:
        logger.warning("Fig 13: 'core_G_robustness' cache not found — using synthetic/hardcoded fallback data, NOT a real measurement.")
        data = {"communication": {"packet_loss": {pl: max(0, 0.85 - pl * 0.8) for pl in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]},
                "latency": {lat: max(0, 0.85 - lat / 1000 * 0.3) for lat in [0, 100, 200, 300, 400, 500]}}}
    
    comm = data.get("communication", {})
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Figure 13 — Performance Degradation under Communication Stress", fontsize=13)
    
    ax = axes[0]
    pl_data = comm.get("packet_loss", {})
    x = sorted([float(k) for k in pl_data.keys()])
    y = [pl_data[k] for k in x]
    ax.plot(x, y, 'o-', color="#2563eb", lw=2, markersize=8)
    ax.set_xlabel("Packet Loss Rate", fontsize=11)
    ax.set_ylabel("Relative Performance", fontsize=11)
    ax.set_title("Packet Loss Impact", fontsize=12)
    ax.grid(True, alpha=0.3)
    
    ax = axes[1]
    lat_data = comm.get("latency", {})
    x = sorted([int(k) for k in lat_data.keys()])
    y = [lat_data[k] for k in x]
    ax.plot(x, y, 's-', color="#ff6b35", lw=2, markersize=8)
    ax.set_xlabel("Latency (ms)", fontsize=11)
    ax.set_ylabel("Relative Performance", fontsize=11)
    ax.set_title("Latency Impact", fontsize=12)
    ax.grid(True, alpha=0.3)
    
    if is_synthetic:
        from evaluation.experiments._synthetic_label import mark_synthetic
        mark_synthetic(fig, note="core_G_robustness cache missing")
    save_fig(fig, get_fig_path(13, "figs_11_20"))


# ==================================================================
# FIGURE 14: N-1 Contingency
# ==================================================================
@timed_figure(14)
def fig_14_n1_contingency():
    """Fig 14: N-1 contingency impact — from experiments/exp_fig14_n1_contingency.py's npz, if run."""
    logger.info("\\n[Fig 14] N-1 Contingency Impact and Recovery")
    # experiments/exp_fig14_n1_contingency.py is a real, standalone
    # per-line N-1 evaluation (trains a policy, outages each line via
    # NetworkData.apply_n1_outage(), evaluates pre-trained vs.
    # fine-tuned) that saves outputs/fig14_n1_contingency.npz. No Core
    # run currently produces the 12-line matrix this figure needs
    # (Core G's real bridge only covers figs 13/20/22/24/54 — see
    # run_core_training.py:run_core_G), and this module is pure
    # post-processing (no training here), so we load that npz if
    # someone has run the real script, else honestly disclose.
    npz_path = BASE_DIR / "outputs" / "fig14_n1_contingency.npz"
    real = None
    if npz_path.exists():
        try:
            with np.load(npz_path, allow_pickle=True) as npz:
                real = {k: npz[k] for k in npz.files}
        except Exception as e:
            logger.warning(f"Fig 14: failed to load {npz_path}: {e}")
    is_synthetic = real is None

    if is_synthetic:
        logger.warning(f"Fig 14: '{npz_path}' not found — using synthetic/hardcoded fallback data, NOT a real measurement. Run experiments/exp_fig14_n1_contingency.py to produce real data.")
        rng = np.random.default_rng(42)
        line_outages = [f"L{i}" for i in range(1, 13)]
        impact = [rng.uniform(0.1, 0.4) for _ in line_outages]
        recovery = [rng.uniform(0.7, 0.95) for _ in line_outages]
        ylabel = "Relative Performance"
        label_impact, label_recovery = "Immediate Impact", "After Recovery"
    else:
        line_outages = [str(n) for n in real["line_names"]]
        baseline_of = float(real["baseline_of"][0])
        # Normalize objective values to the same "relative performance"
        # scale the figure's axis promises (1.0 = matches baseline;
        # lower objective is better, so express as baseline/observed).
        impact = [baseline_of / max(v, 1e-9) for v in real["pretrain_of"]]
        recovery = [baseline_of / max(v, 1e-9) for v in real["finetuned_of"]]
        ylabel = "Relative Performance (baseline OF / observed OF)"
        label_impact, label_recovery = "Pre-trained (no adapt)", "After Fine-tuning"

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(line_outages))
    width = 0.35
    ax.bar(x - width/2, impact, width, label=label_impact, color="#ff6b35", alpha=0.8, edgecolor="white")
    ax.bar(x + width/2, recovery, width, label=label_recovery, color="#39d353", alpha=0.8, edgecolor="white")
    ax.set_xlabel("Line Outage", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title("Figure 14 — N-1 Contingency Impact and Recovery (30-Bus)", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(line_outages, rotation=45, ha="right")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")
    if is_synthetic:
        from evaluation.experiments._synthetic_label import mark_synthetic
        mark_synthetic(fig, note=f"{npz_path.name} missing")
    save_fig(fig, get_fig_path(14, "figs_11_20"))


# ==================================================================
# FIGURE 15: Scalability Analysis
# ==================================================================
@timed_figure(15)
def fig_15_scalability():
    """Fig 15: Scalability — from Core F."""
    logger.info("\\n[Fig 15] Scalability Analysis")
    data = load_cache("core_F_scalability")
    is_synthetic = data is None
    if data is None:
        logger.warning("Fig 15: 'core_F_scalability' cache not found — using synthetic/hardcoded fallback data, NOT a real measurement.")
        data = {
            "3bus": {"n_relays": 6, "train_time": 45, "final_performance": 0.92},
            "8bus": {"n_relays": 14, "train_time": 120, "final_performance": 0.85},
            "14bus": {"n_relays": 20, "train_time": 210, "final_performance": 0.83},
            "30bus": {"n_relays": 42, "train_time": 350, "final_performance": 0.80},
            "39bus": {"n_relays": 46, "train_time": 480, "final_performance": 0.78},
            "57bus": {"n_relays": 80, "train_time": 900, "final_performance": 0.75},
            "118bus": {"n_relays": 186, "train_time": 2400, "final_performance": 0.68},
        }
    
    systems = list(data.keys())
    n_relays = [data[s]["n_relays"] for s in systems]
    times = [data[s]["train_time"] for s in systems]
    perfs = [data[s]["final_performance"] for s in systems]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Figure 15 — Scalability Analysis Across Test Systems", fontsize=13)
    
    ax = axes[0]
    ax.scatter(n_relays, times, s=150, color="#2563eb", edgecolors="white", linewidth=2, zorder=5)
    for s, nr, t in zip(systems, n_relays, times):
        ax.annotate(s, (nr, t), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=9)
    z = np.polyfit(n_relays, times, 2)
    p = np.poly1d(z)
    x_line = np.linspace(min(n_relays), max(n_relays), 100)
    ax.plot(x_line, p(x_line), "--", color="#ff6b35", lw=2, alpha=0.7, label="Quadratic fit")
    ax.set_xlabel("Number of Relays", fontsize=11)
    ax.set_ylabel("Training Time (s)", fontsize=11)
    ax.set_title("Training Time Scaling", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    ax = axes[1]
    ax.scatter(n_relays, perfs, s=150, color="#39d353", edgecolors="white", linewidth=2, zorder=5)
    for s, nr, p in zip(systems, n_relays, perfs):
        ax.annotate(s, (nr, p), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=9)
    ax.set_xlabel("Number of Relays", fontsize=11)
    ax.set_ylabel("Final Performance", fontsize=11)
    ax.set_title("Performance vs. System Size", fontsize=12)
    ax.set_ylim(0.5, 1.0)
    ax.grid(True, alpha=0.3)
    
    if is_synthetic:
        from evaluation.experiments._synthetic_label import mark_synthetic
        mark_synthetic(fig, note="core_F_scalability cache missing")
    save_fig(fig, get_fig_path(15, "figs_11_20"))

# ==================================================================
# FIGURE 16: EWC Effectiveness
# ==================================================================
@timed_figure(16)
def fig_16_ewc_effectiveness():
    """Fig 16: EWC effectiveness — from Core B."""
    logger.info("\\n[Fig 16] EWC Effectiveness")
    data = load_cache("core_B_curriculum")
    is_synthetic = data is None
    if data is None:
        logger.warning("Fig 16: 'core_B_curriculum' cache not found — using synthetic/hardcoded fallback data, NOT a real measurement.")
        data = {"with_ewc": [{"task_performances": np.array([[0.85,0,0,0],[0.83,0.84,0,0],[0.82,0.83,0.84,0],[0.81,0.82,0.83,0.84]])}],
                "without_ewc": [{"task_performances": np.array([[0.85,0,0,0],[0.70,0.84,0,0],[0.55,0.68,0.83,0],[0.45,0.52,0.65,0.82]])}]}

    ewc_mats = [r["task_performances"] for r in data.get("with_ewc", [])]
    noewc_mats = [r["task_performances"] for r in data.get("without_ewc", [])]
    ewc_avg = np.mean(ewc_mats, axis=0) if ewc_mats else np.eye(4) * 0.8
    noewc_avg = np.mean(noewc_mats, axis=0) if noewc_mats else np.eye(4) * 0.5
    n_tasks = ewc_avg.shape[0]
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Figure 16 — EWC Effectiveness: Performance Drop After Task Transition", fontsize=13)
    
    ax = axes[0]
    im = ax.imshow(ewc_avg, cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_title("With EWC", fontsize=12)
    ax.set_xlabel("Evaluated on Task", fontsize=10)
    ax.set_ylabel("Trained up to Task", fontsize=10)
    plt.colorbar(im, ax=ax, shrink=0.7)
    
    ax = axes[1]
    im = ax.imshow(noewc_avg, cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_title("Without EWC", fontsize=12)
    ax.set_xlabel("Evaluated on Task", fontsize=10)
    ax.set_ylabel("Trained up to Task", fontsize=10)
    plt.colorbar(im, ax=ax, shrink=0.7)
    
    ax = axes[2]
    x = np.arange(n_tasks)
    ax.plot(x, np.diag(ewc_avg), "o-", color="#2563eb", lw=2, markersize=8, label="With EWC")
    ax.plot(x, np.diag(noewc_avg), "s-", color="#ff6b35", lw=2, markersize=8, label="Without EWC")
    ax.set_xticks(x)
    ax.set_xticklabels([f"T{i}" for i in range(n_tasks)], fontsize=10)
    ax.set_xlabel("Task", fontsize=11)
    ax.set_ylabel("Performance", fontsize=11)
    ax.set_title("Current Task Performance (Forgetting)", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    if is_synthetic:
        from evaluation.experiments._synthetic_label import mark_synthetic
        mark_synthetic(fig, note="core_B_curriculum cache missing")
    save_fig(fig, get_fig_path(16, "figs_11_20"))


# ==================================================================
# FIGURE 18: Inference Latency
# ==================================================================
@timed_figure(18)
def fig_18_inference_latency():
    """Fig 18: Inference latency distribution — from Core I."""
    logger.info("\\n[Fig 18] Inference Latency Distribution")
    data = load_cache("core_I_deployment")
    is_synthetic = data is None
    if is_synthetic:
        logger.warning("Fig 18: 'core_I_deployment' cache not found — using synthetic "
                        "lognormal latencies, NOT a real measurement.")
        rng = np.random.default_rng(42)
        latencies = rng.lognormal(-3, 0.5, 1000) * 1000
        data = {"latency": {"latencies_ms": latencies, "mean_ms": np.mean(latencies), "p50_ms": np.percentile(latencies, 50), "p99_ms": np.percentile(latencies, 99)}}

    lat = data.get("latency", {})
    latencies = lat.get("latencies_ms", np.random.lognormal(-3, 0.5, 1000) * 1000)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Figure 18 — Inference Latency Distribution (Intel NUC i5, 1000 runs)", fontsize=13)
    
    ax = axes[0]
    ax.hist(latencies, bins=50, color="#2563eb", edgecolor="white", alpha=0.8)
    ax.axvline(np.mean(latencies), color="red", ls="--", lw=2, label=f"Mean: {np.mean(latencies):.2f} ms")
    ax.axvline(np.percentile(latencies, 99), color="orange", ls="--", lw=2, label=f"P99: {np.percentile(latencies, 99):.2f} ms")
    ax.set_xlabel("Latency (ms)", fontsize=11)
    ax.set_ylabel("Frequency", fontsize=11)
    ax.set_title("Latency Histogram", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    ax = axes[1]
    sorted_lat = np.sort(latencies)
    cdf = np.arange(1, len(sorted_lat) + 1) / len(sorted_lat)
    ax.plot(sorted_lat, cdf, color="#39d353", lw=2)
    ax.axhline(0.99, color="orange", ls="--", lw=1, alpha=0.7)
    ax.axvline(np.percentile(latencies, 99), color="orange", ls="--", lw=1, alpha=0.7)
    ax.set_xlabel("Latency (ms)", fontsize=11)
    ax.set_ylabel("CDF", fontsize=11)
    ax.set_title("Cumulative Distribution", fontsize=12)
    ax.grid(True, alpha=0.3)

    if is_synthetic:
        from evaluation.experiments._synthetic_label import mark_synthetic
        mark_synthetic(fig, note="core_I_deployment cache missing")
    save_fig(fig, get_fig_path(18, "figs_11_20"))


# ==================================================================
# FIGURE 19: PV Ride-Through Curves
# ==================================================================
# Not defined here — Fig 19 is generated by phase1_scripts/exp_fig19.py
# (registered in run_thesis_experiments.py's MANUAL_FIGURE_SCRIPTS and
# actually invoked by run_phase1() in the real pipeline). This module
# previously ALSO defined and registered its own fig_19_pv_ride_through()
# with a completely different curve (different IEC/IEEE thresholds, axis
# scale, and framing) under the same figure number and output path —
# whichever of the two happened to run last would silently overwrite the
# other's file with no indication of the conflict. Since
# phase1_scripts/exp_fig19.py is the one the orchestrated pipeline
# actually uses, the duplicate here was removed rather than kept
# alongside it.


# ==================================================================
# FIGURE 20: Byzantine Agent Detection
# ==================================================================
@timed_figure(20)
def fig_20_byzantine():
    """Fig 20: Byzantine resilience — from Core G."""
    logger.info("\\n[Fig 20] Byzantine Agent Detection")
    data = load_cache("core_G_robustness")
    is_synthetic = data is None
    if data is None:
        logger.warning("Fig 20: 'core_G_robustness' cache not found — using synthetic/hardcoded fallback data, NOT a real measurement.")
        data = {"byzantine": {"random": 0.72, "collusion": 0.62, "fdi": 0.67}}
    
    byz = data.get("byzantine", {})
    attacks = list(byz.keys())
    values = list(byz.values())
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#2563eb", "#ff6b35", "#a371f7"]
    bars = ax.bar(attacks, values, color=colors, edgecolor="white", linewidth=2, alpha=0.85)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f"{val:.2f}", 
                ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_ylabel("Relative Performance", fontsize=12)
    ax.set_title("Figure 20 — Byzantine Agent Detection via Trust Score Evolution", fontsize=13)
    if is_synthetic:
        ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis="y")
    if is_synthetic:
        from evaluation.experiments._synthetic_label import mark_synthetic
        mark_synthetic(fig, note="core_G_robustness cache missing")
    save_fig(fig, get_fig_path(20, "figs_11_20"))


# ==================================================================
# FIGURE 21: Action Projection Efficacy
# ==================================================================
@timed_figure(21)
def fig_21_action_projection():
    """Fig 21: Action projection — from Core A safety activation data."""
    logger.info("\\n[Fig 21] Action Projection Efficacy")
    data = load_cache("core_A_mappo_safe_ewc_8bus_100pv")
    is_synthetic = data is None
    if data is None:
        logger.warning("Fig 21: 'core_A_mappo_safe_ewc_8bus_100pv' cache not found — using synthetic/hardcoded fallback data, NOT a real measurement.")
        n_eps = 2000
        data = {"episodes": np.arange(n_eps), "safety_activations_mean": 50 * np.exp(-np.arange(n_eps) / 300) + np.random.randn(n_eps) * 2,
                "safety_activations_std": np.ones(n_eps) * 3}
    
    eps = data.get("episodes", np.arange(len(data.get("safety_activations_mean", []))))
    safe_mean = data.get("safety_activations_mean", np.zeros_like(eps))
    safe_std = data.get("safety_activations_std", np.zeros_like(eps))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(eps, safe_mean, color="#a371f7", lw=2, label="Safety Activations")
    ax.fill_between(eps, safe_mean - safe_std, safe_mean + safe_std, alpha=0.2, color="#a371f7")
    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Safety Projections per Episode", fontsize=12)
    ax.set_title("Figure 21 — Action Projection Efficacy During Training", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    if is_synthetic:
        from evaluation.experiments._synthetic_label import mark_synthetic
        mark_synthetic(fig, note="core_A_mappo_safe_ewc_8bus_100pv cache missing")
    save_fig(fig, get_fig_path(21, "figs_21_30"))


# ==================================================================
# FIGURE 22: Bandwidth vs Performance Tradeoff
# ==================================================================
@timed_figure(22)
def fig_22_bandwidth_tradeoff():
    """Fig 22: Bandwidth tradeoff — from Core G."""
    logger.info("\\n[Fig 22] Bandwidth vs. Performance Trade-off")
    data = load_cache("core_G_robustness")
    is_synthetic = data is None
    if data is None:
        logger.warning("Fig 22: 'core_G_robustness' cache not found — using synthetic/hardcoded fallback data, NOT a real measurement.")
        data = {"bandwidth": {1: {"performance": 0.80, "bandwidth_mb": 12}, 2: {"performance": 0.85, "bandwidth_mb": 24},
                              3: {"performance": 0.84, "bandwidth_mb": 36}, 4: {"performance": 0.82, "bandwidth_mb": 48}}}
    
    bw = data.get("bandwidth", {})
    depths = sorted([int(k) for k in bw.keys()])
    perfs = [bw[d]["performance"] for d in depths]
    # Real values are in kbps (the actual quantity _eval_bandwidth
    # computes: message size x agent count x compression, converted to
    # kilobits/sec) — a different unit than the synthetic fallback's
    # invented "MB/episode" scale. Read whichever key is present and
    # label the axis honestly rather than forcing both onto one
    # (mis-scaled) unit.
    if depths and "bandwidth_kbps" in bw[depths[0]]:
        bws = [bw[d]["bandwidth_kbps"] for d in depths]
        bw_label = "Bandwidth (kbps)"
    else:
        bws = [bw[d]["bandwidth_mb"] for d in depths]
        bw_label = "Bandwidth (MB/episode)"

    fig, ax1 = plt.subplots(figsize=(10, 6))
    color1 = "#2563eb"
    ax1.set_xlabel("GNN Depth (Communication Hops)", fontsize=12)
    ax1.set_ylabel("Performance", color=color1, fontsize=12)
    ax1.plot(depths, perfs, 'o-', color=color1, lw=2, markersize=10, label="Performance")
    ax1.tick_params(axis='y', labelcolor=color1)
    if is_synthetic:
        ax1.set_ylim(0.7, 0.9)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    color2 = "#ff6b35"
    ax2.set_ylabel(bw_label, color=color2, fontsize=12)
    ax2.bar(depths, bws, alpha=0.3, color=color2, width=0.3, label="Bandwidth")
    ax2.tick_params(axis='y', labelcolor=color2)
    
    fig.suptitle("Figure 22 — Bandwidth vs. Performance Trade-off", fontsize=13)
    fig.tight_layout()
    if is_synthetic:
        from evaluation.experiments._synthetic_label import mark_synthetic
        mark_synthetic(fig, note="core_G_robustness cache missing")
    save_fig(fig, get_fig_path(22, "figs_21_30"))


# ==================================================================
# FIGURE 23: Unseen Fault Types
# ==================================================================
@timed_figure(23)
def fig_23_unseen_faults():
    """Fig 23: Generalization to unseen fault types — from Core H."""
    logger.info("\\n[Fig 23] Generalization to Unseen Fault Types")
    data = load_cache("core_H_transfer")
    is_synthetic = data is None
    if data is None:
        logger.warning("Fig 23: 'core_H_transfer' cache not found — using synthetic/hardcoded fallback data, NOT a real measurement.")
        data = {"unseen_faults": {"bolted": 0.85, "high-impedance": 0.78, "evolving": 0.72, "simultaneous": 0.68}}
    
    # Real shape (see run_core_H) is {real_fault_type_name: relative_perf},
    # keyed by whichever IEEE fault types weren't used during training —
    # same {name: float} shape as the synthetic fallback, so no branching
    # needed here, just an honest axis label/limit (relative performance
    # can exceed 1.0, unlike the fallback's fixed illustrative scores).
    faults = data.get("unseen_faults", {})
    names = list(faults.keys())
    values = list(faults.values())

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#2563eb", "#39d353", "#f7c948", "#ff6b35"]
    bars = ax.barh(names, values, color=colors, edgecolor="white", linewidth=2, alpha=0.85)
    for bar, val in zip(bars, values):
        ax.text(val + 0.01, bar.get_y() + bar.get_height()/2, f"{val:.2f}",
                ha="left", va="center", fontsize=11, fontweight="bold")
    ax.set_xlabel("Performance" if is_synthetic else "Relative Performance (base_of / unseen_of)", fontsize=12)
    ax.set_title("Figure 23 — Generalization to Unseen Fault Types", fontsize=13)
    if is_synthetic:
        ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.3, axis="x")
    if is_synthetic:
        from evaluation.experiments._synthetic_label import mark_synthetic
        mark_synthetic(fig, note="core_H_transfer cache missing")
    save_fig(fig, get_fig_path(23, "figs_21_30"))


# ==================================================================
# FIGURE 24: Sensor Robustness
# ==================================================================
@timed_figure(24)
def fig_24_sensor_robustness():
    """Fig 24: Sensor robustness — from Core G."""
    logger.info("\\n[Fig 24] Sensor Robustness")
    data = load_cache("core_G_robustness")
    is_synthetic = data is None
    if data is None:
        logger.warning("Fig 24: 'core_G_robustness' cache not found — using synthetic/hardcoded fallback data, NOT a real measurement.")
        data = {"sensor": {"noise": {0.0: 0.85, 0.05: 0.75, 0.1: 0.65, 0.15: 0.55, 0.2: 0.45},
                           "bias": {0.0: 0.85, 0.05: 0.78, 0.1: 0.70},
                           "missing": {0.0: 0.85, 0.1: 0.80, 0.2: 0.72}}}
    
    sensor = data.get("sensor", {})
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Figure 24 — Sensor Robustness: Performance Under Measurement Degradation", fontsize=13)
    
    ax = axes[0]
    noise = sensor.get("noise", {})
    x = sorted([float(k) for k in noise.keys()])
    y = [noise[k] for k in x]
    ax.plot(x, y, 'o-', color="#2563eb", lw=2, markersize=8)
    ax.set_xlabel("Noise Level (sigma)", fontsize=11)
    ax.set_ylabel("Relative Performance", fontsize=11)
    ax.set_title("Gaussian Noise", fontsize=12)
    ax.grid(True, alpha=0.3)
    
    ax = axes[1]
    bias = sensor.get("bias", {})
    x = sorted([float(k) for k in bias.keys()])
    y = [bias[k] for k in x]
    ax.plot(x, y, 's-', color="#ff6b35", lw=2, markersize=8)
    ax.set_xlabel("Bias Level", fontsize=11)
    ax.set_ylabel("Relative Performance", fontsize=11)
    ax.set_title("Systematic Bias", fontsize=12)
    ax.grid(True, alpha=0.3)
    
    ax = axes[2]
    missing = sensor.get("missing", {})
    x = sorted([float(k) for k in missing.keys()])
    y = [missing[k] for k in x]
    ax.plot(x, y, '^-', color="#39d353", lw=2, markersize=8)
    ax.set_xlabel("Missing Rate", fontsize=11)
    ax.set_ylabel("Relative Performance", fontsize=11)
    ax.set_title("Missing Measurements", fontsize=12)
    ax.grid(True, alpha=0.3)
    
    if is_synthetic:
        from evaluation.experiments._synthetic_label import mark_synthetic
        mark_synthetic(fig, note="core_G_robustness cache missing")
    save_fig(fig, get_fig_path(24, "figs_21_30"))


# ==================================================================
# FIGURE 25: Zero-Shot Transfer
# ==================================================================
@timed_figure(25)
def fig_25_zero_shot_transfer():
    """Fig 25: Zero-shot transfer — from Core H."""
    logger.info("\\n[Fig 25] Zero-Shot Transfer 8->30 Bus")
    data = load_cache("core_H_transfer")
    is_synthetic = data is None
    if data is None:
        logger.warning("Fig 25: 'core_H_transfer' cache not found — using synthetic/hardcoded fallback data, NOT a real measurement.")
        data = {"zero_shot": {"warm_start": {"performance": 0.75, "convergence_ep": 500}, "cold_start": {"performance": 0.20, "convergence_ep": 2000}}}
    
    zs = data.get("zero_shot", {})
    # _eval_zero_shot_transfer's real output (see core_eval_bridge.
    # _extract_fig_data, fig 25) is keyed "transfer_of"/"cti_ok"/"system"
    # (+ "base_of", added in run_core_H) — there is no "warm_start"/
    # "cold_start" split anywhere: no code path here trains this target
    # system's own policy from scratch to provide a genuine cold-start
    # baseline. Reading zs.get("warm_start"/"cold_start") against a REAL
    # cache hit always missed and fell back to the identical fabricated
    # 0.75/0.20 placeholders, undisclosed. Plot only what's actually real:
    # one relative-performance bar for the zero-shot transfer itself.
    if not is_synthetic and "transfer_of" in zs:
        base_of = zs.get("base_of")
        transfer_of = zs["transfer_of"]
        rel_perf = float(base_of / max(transfer_of, 1e-9)) if base_of else None
        cti_ok = zs.get("cti_ok")

        fig, ax = plt.subplots(figsize=(8, 6))
        if rel_perf is not None:
            ax.bar(["Zero-Shot Transfer\\n(learned -> " + str(zs.get("system", "target")) + ")"],
                   [rel_perf], color="#2563eb", alpha=0.85, edgecolor="white", width=0.5)
            ax.text(0, rel_perf + 0.02, f"{rel_perf:.2f}", ha="center", fontsize=12, fontweight="bold")
        ax.set_ylabel("Relative Performance (base_of / transfer_of)", fontsize=12)
        ax.set_title(f"Figure 25 — Zero-Shot Transfer: 8-Bus -> {zs.get('system', 'target')}", fontsize=13)
        ax.grid(True, alpha=0.3, axis="y")
        cti_text = "CTI satisfied" if cti_ok else "CTI violated"
        _annotate_panel_synthetic(
            ax, f"cold-start (train-from-scratch) baseline not measured — {cti_text} on transfer"
        )
    else:
        warm = zs.get("warm_start", {})
        cold = zs.get("cold_start", {})

        fig, ax = plt.subplots(figsize=(10, 6))
        categories = ["Warm Start\\n(Transfer)", "Cold Start\\n(Scratch)"]
        perfs = [warm.get("performance", 0.75), cold.get("performance", 0.20)]
        convs = [warm.get("convergence_ep", 500), cold.get("convergence_ep", 2000)]

        x = np.arange(len(categories))
        width = 0.35
        ax.bar(x - width/2, perfs, width, label="Final Performance", color="#2563eb", alpha=0.8, edgecolor="white")
        ax2 = ax.twinx()
        ax2.bar(x + width/2, convs, width, label="Convergence Episodes", color="#ff6b35", alpha=0.8, edgecolor="white")

        ax.set_ylabel("Final Performance", fontsize=12, color="#2563eb")
        ax2.set_ylabel("Convergence Episodes", fontsize=12, color="#ff6b35")
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.set_title("Figure 25 — Zero-Shot Transfer: 8-Bus -> 14-Bus (60% PV)", fontsize=13)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3, axis="y")

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    if is_synthetic:
        from evaluation.experiments._synthetic_label import mark_synthetic
        mark_synthetic(fig, note="core_H_transfer cache missing")
    save_fig(fig, get_fig_path(25, "figs_21_30"))

# ==================================================================
# FIGURE 29: EWC Continual Adaptation
# ==================================================================
@timed_figure(29)
def fig_29_ewc_continual():
    """Fig 29: EWC visualization — from Core B."""
    logger.info("\\n[Fig 29] EWC Continual Adaptation")
    data = load_cache("core_B_curriculum")
    is_synthetic = data is None
    if data is None:
        logger.warning("Fig 29: 'core_B_curriculum' cache not found — using synthetic/hardcoded fallback data, NOT a real measurement.")
        data = {"with_ewc": [{"task_performances": np.random.rand(5, 5) * 0.3 + 0.6}],
                "without_ewc": [{"task_performances": np.random.rand(5, 5) * 0.3 + 0.4}]}
    
    ewc_mats = [r["task_performances"] for r in data.get("with_ewc", [])]
    noewc_mats = [r["task_performances"] for r in data.get("without_ewc", [])]
    ewc_avg = np.mean(ewc_mats, axis=0) if ewc_mats else np.eye(5) * 0.7
    noewc_avg = np.mean(noewc_mats, axis=0) if noewc_mats else np.eye(5) * 0.5
    # Number of curriculum tasks actually trained (run_core_B's real
    # task_performances matrices are n_tasks x n_tasks, e.g. 4x4 in fast
    # mode) — the right panel below previously hardcoded 5 regardless,
    # crashing with a shape mismatch whenever the real matrix wasn't 5x5
    # (np.diag(ewc_avg) has len == ewc_avg.shape[0]).
    n_tasks = ewc_avg.shape[0]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Figure 29 — Elastic Weight Consolidation (EWC) for Continual Adaptation", fontsize=13)
    
    ax = axes[0]
    # core_B_curriculum never stores a per-parameter Fisher Information
    # diagonal (FisherInformationEstimator's real output — see
    # marl/continual_trainer.py) — only aggregate task_performances
    # matrices, used in the right panel below. This left panel was always
    # illustrative regardless of is_synthetic, but previously used an
    # unseeded np.random.exponential() and was never watermarked, so it
    # looked identical to a real per-parameter Fisher plot even when the
    # rest of the figure was real. Explicitly seeded + annotated instead.
    fisher = np.random.default_rng(29).exponential(1.0, 100)
    fisher = np.sort(fisher)[::-1]
    ax.bar(range(100), fisher, color="#2563eb", edgecolor="white", alpha=0.8)
    ax.set_xlabel("Parameter Index (sorted)", fontsize=11)
    ax.set_ylabel("Fisher Information F_i", fontsize=11)
    ax.set_title("Parameter Importance", fontsize=12)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, axis="y")
    _annotate_panel_synthetic(ax, "no per-parameter Fisher diagonal in core_B_curriculum cache")
    
    ax = axes[1]
    x = np.arange(n_tasks)
    ax.plot(x, np.diag(ewc_avg), "o-", color="#2563eb", lw=2, markersize=8, label="With EWC")
    ax.plot(x, np.diag(noewc_avg), "s-", color="#ff6b35", lw=2, markersize=8, label="Without EWC")
    ax.set_xticks(x)
    ax.set_xticklabels([f"T{i}" for i in range(n_tasks)], fontsize=10)
    ax.set_xlabel("Task", fontsize=11)
    ax.set_ylabel("Performance on Task 0", fontsize=11)
    ax.set_title("Catastrophic Forgetting Prevention", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    if is_synthetic:
        from evaluation.experiments._synthetic_label import mark_synthetic
        mark_synthetic(fig, note="core_B_curriculum cache missing")
    save_fig(fig, get_fig_path(29, "figs_21_30"))


# ==================================================================
# FIGURE 31: Multi-Metric Ranking
# ==================================================================
@timed_figure(31)
def fig_31_multi_metric_ranking():
    """Fig 31: Overall performance ranking — from Core C."""
    logger.info("\\n[Fig 31] Multi-Metric Ranking")
    data = load_cache("core_C_baselines_8bus")
    is_synthetic = data is None
    if data is None:
        logger.warning("Fig 31: 'core_C_baselines_8bus' cache not found — using synthetic/hardcoded fallback data, NOT a real measurement.")
        methods = ["MAPPO-Safe-EWC", "MAPPO-Safe", "MAPPO", "IPPO", "Traditional"]
        data = {m: np.random.normal(3.0 - i * 0.3, 0.5, 50) for i, m in enumerate(methods)}
    
    methods = list(data.keys())
    n_methods = len(methods)
    final_perf = {m: np.mean(v[-100:] if hasattr(v, '__len__') and len(v) > 100 else v) for m, v in data.items()}
    sorted_methods = sorted(methods, key=lambda m: final_perf[m], reverse=True)
    ranks = {m: i + 1 for i, m in enumerate(sorted_methods)}
    
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.set_xlim(0.5, n_methods + 0.5)
    ax.set_ylim(-0.5, 1.5)
    ax.axhline(0.5, color="black", lw=1)
    for i in range(1, n_methods + 1):
        ax.axvline(i, color="gray", alpha=0.3, ls="--")
        ax.text(i, 0.3, str(i), ha="center", fontsize=12, fontweight="bold")
    
    y_pos = 1.0
    colors = ["#2563eb" if "EWC" in m else "#f7c948" for m in sorted_methods]
    for m, c in zip(sorted_methods, colors):
        ax.text(ranks[m], y_pos, m, ha="center", fontsize=10,
               bbox=dict(boxstyle="round,pad=0.3", facecolor=c, alpha=0.7))
        y_pos = 0.0 if y_pos == 1.0 else 1.0
    
    ax.set_title("Figure 31 — Overall Performance Ranking with Critical Difference", fontsize=13)
    ax.axis("off")
    if is_synthetic:
        from evaluation.experiments._synthetic_label import mark_synthetic
        mark_synthetic(fig, note="core_C_baselines_8bus cache missing")
    save_fig(fig, get_fig_path(31, "figs_31_40"))


# ==================================================================
# FIGURE 32: Effect Size Analysis
# ==================================================================
@timed_figure(32)
def fig_32_effect_size():
    """Fig 32: Effect size — MAPPO-Safe-EWC vs baselines from Core C."""
    logger.info("\\n[Fig 32] Effect Size Analysis")
    data = load_cache("core_C_baselines_8bus")
    is_synthetic = data is None
    if data is None:
        logger.warning("Fig 32: 'core_C_baselines_8bus' cache not found — using synthetic/hardcoded fallback data, NOT a real measurement.")
        methods = ["MAPPO-Safe-EWC", "MAPPO-Safe", "MAPPO", "IPPO", "Traditional"]
        data = {m: np.random.normal(3.0 - i * 0.3, 0.5, 50) for i, m in enumerate(methods)}
    
    baseline = "MAPPO-Safe-EWC"
    baseline_vals = data.get(baseline, np.random.normal(3.0, 0.5, 50))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    methods = [m for m in data.keys() if m != baseline]
    cohens_d = []
    for m in methods:
        vals = data[m] if hasattr(data[m], '__len__') else np.random.normal(2.5, 0.5, 50)
        if hasattr(vals, '__len__') and len(vals) > 1:
            d = (np.mean(vals) - np.mean(baseline_vals)) / np.sqrt((np.std(vals)**2 + np.std(baseline_vals)**2) / 2)
        else:
            d = np.random.uniform(-0.5, 0.5)
        cohens_d.append(d)
    
    colors = ["#39d353" if d > 0 else "#ff6b35" for d in cohens_d]
    bars = ax.barh(methods, cohens_d, color=colors, edgecolor="white", alpha=0.85)
    ax.axvline(0, color="black", lw=1)
    ax.axvline(0.2, color="gray", ls="--", alpha=0.5, label="Small effect")
    ax.axvline(0.5, color="gray", ls="--", alpha=0.7, label="Medium effect")
    ax.axvline(0.8, color="gray", ls="--", alpha=0.9, label="Large effect")
    ax.set_xlabel("Cohen's d (vs. MAPPO-Safe-EWC)", fontsize=12)
    ax.set_title("Figure 32 — Effect Size Analysis: MAPPO-Safe-EWC vs. Baselines", fontsize=13)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.3, axis="x")
    if is_synthetic:
        from evaluation.experiments._synthetic_label import mark_synthetic
        mark_synthetic(fig, note="core_C_baselines_8bus cache missing")
    save_fig(fig, get_fig_path(32, "figs_31_40"))


# ==================================================================
# FIGURE 33: Radar Chart Comparison
# ==================================================================
@timed_figure(33)
def fig_33_radar_chart():
    """Fig 33: Multi-dimensional method comparison — from Core C."""
    logger.info("\\n[Fig 33] Multi-Dimensional Method Comparison")
    data = load_cache("core_C_baselines_8bus")
    is_synthetic = data is None
    if data is None:
        logger.warning("Fig 33: 'core_C_baselines_8bus' cache not found — using synthetic/hardcoded fallback data, NOT a real measurement.")
        methods = ["MAPPO-Safe-EWC", "MAPPO-Safe", "MAPPO", "IPPO", "Traditional"]
        data = {m: np.random.normal(3.0 - i * 0.3, 0.5, 50) for i, m in enumerate(methods)}
    
    categories = ["Objective\\nFunction", "CTI\\nSatisfaction", "Convergence\\nSpeed", "Sample\\nEfficiency", "Robustness"]
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    colors = {"MAPPO-Safe-EWC": "#2563eb", "MAPPO-Safe": "#39d353", "MAPPO": "#f7c948", "IPPO": "#ff6b35", "Traditional": "#ff6b6b"}

    # core_C_baselines only stores each method's per-episode return array —
    # a single real scalar per method (its mean), never a genuine breakdown
    # across "CTI Satisfaction"/"Convergence Speed"/"Sample Efficiency"/
    # "Robustness". This function fabricated all 5 radar axes as that ONE
    # real scalar plus independent random jitter per axis, unconditionally
    # (even when is_synthetic was False, i.e. even for a real cache hit) —
    # a fake multi-dimensional shape built from one real number. Seeded so
    # the illustrative jitter is at least reproducible, and always
    # watermarked regardless of is_synthetic since there is no code path
    # (real or synthetic) that computes genuine values for 4 of these 5 axes.
    rng = np.random.default_rng(33)
    for method in ["MAPPO-Safe-EWC", "MAPPO-Safe", "MAPPO", "IPPO", "Traditional"]:
        if method in data:
            vals = data[method]
            base = np.mean(vals[-100:] if hasattr(vals, '__len__') and len(vals) > 100 else vals)
            values = [base / 5.0 + rng.uniform(-0.1, 0.1) for _ in range(N)]
            values += values[:1]
            ax.plot(angles, values, 'o-', color=colors.get(method, "#333"), lw=2, label=method, markersize=6)
            ax.fill(angles, values, alpha=0.1, color=colors.get(method, "#333"))

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_title("Figure 33 — Multi-Dimensional Method Comparison (8-Bus, 100% PV)", fontsize=13, y=1.08)
    ax.legend(fontsize=9, loc="upper right", bbox_to_anchor=(1.3, 1.1))
    ax.grid(True)
    from evaluation.experiments._synthetic_label import mark_synthetic
    note = ("core_C_baselines cache missing" if is_synthetic else
            "core_C_baselines has no per-dimension breakdown — 4 of 5 axes are "
            "illustrative jitter around the one real scalar (mean objective)")
    mark_synthetic(fig, note=note)
    save_fig(fig, get_fig_path(33, "figs_31_40"))


# ==================================================================
# FIGURE 34: Neural Network Architecture
# ==================================================================
@timed_figure(34)
def fig_34_nn_architecture():
    """Fig 34: NN architecture detail — schematic diagram."""
    logger.info("\\n[Fig 34] Neural Network Architecture Detail")
    fig, ax = plt.subplots(figsize=(14, 10))
    
    layers = [
        ("Observation\\n(64-dim)", 0.05, 0.5, 0.08, 0.4, "#2563eb"),
        ("GNN Encoder\\n(2-layer GAT)", 0.25, 0.5, 0.08, 0.35, "#39d353"),
        ("Actor MLP\\n(128->64->32)", 0.45, 0.7, 0.08, 0.25, "#f7c948"),
        ("Critic MLP\\n(128->64->32)", 0.45, 0.3, 0.08, 0.25, "#f7c948"),
        ("Safety Layer\\n(Projection)", 0.65, 0.7, 0.08, 0.2, "#ff6b35"),
        ("Value Output\\n(1-dim)", 0.65, 0.3, 0.08, 0.2, "#a371f7"),
        ("Action\\n(2-dim: TMS, PS)", 0.85, 0.7, 0.08, 0.15, "#2563eb"),
        ("Global State\\nValue", 0.85, 0.3, 0.08, 0.15, "#2563eb"),
    ]
    
    for name, x, y, w, h, color in layers:
        ax.add_patch(plt.Rectangle((x-w/2, y-h/2), w, h, facecolor=color, edgecolor="white", linewidth=2, alpha=0.85))
        ax.text(x, y, name, ha="center", va="center", fontsize=9, fontweight="bold", color="white")
    
    ax.annotate("", xy=(0.17, 0.5), xytext=(0.13, 0.5), arrowprops=dict(arrowstyle="->", color="gray", lw=1.5))
    ax.annotate("", xy=(0.37, 0.7), xytext=(0.33, 0.5), arrowprops=dict(arrowstyle="->", color="gray", lw=1.5))
    ax.annotate("", xy=(0.37, 0.3), xytext=(0.33, 0.5), arrowprops=dict(arrowstyle="->", color="gray", lw=1.5))
    ax.annotate("", xy=(0.57, 0.7), xytext=(0.53, 0.7), arrowprops=dict(arrowstyle="->", color="gray", lw=1.5))
    ax.annotate("", xy=(0.57, 0.3), xytext=(0.53, 0.3), arrowprops=dict(arrowstyle="->", color="gray", lw=1.5))
    ax.annotate("", xy=(0.77, 0.7), xytext=(0.73, 0.7), arrowprops=dict(arrowstyle="->", color="gray", lw=1.5))
    ax.annotate("", xy=(0.77, 0.3), xytext=(0.73, 0.3), arrowprops=dict(arrowstyle="->", color="gray", lw=1.5))
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Figure 34 — MAPPO-Safe-EWC Neural Network Architecture Detail", fontsize=13)
    ax.axis("off")
    save_fig(fig, get_fig_path(34, "figs_31_40"))


# ==================================================================
# FIGURE 35: Observation Space Feature Breakdown
# ==================================================================
@timed_figure(35)
def fig_35_observation_space():
    """Fig 35: Observation space breakdown — schematic."""
    logger.info("\\n[Fig 35] Observation Space Feature Breakdown")
    fig, ax = plt.subplots(figsize=(12, 8))
    
    features = [
        ("Local Voltage (3-dim)", 0.15, 0.8, 0.25, 0.1, "#2563eb"),
        ("Current Flow (3-dim)", 0.15, 0.65, 0.25, 0.1, "#2563eb"),
        ("Relay Status (2-dim)", 0.15, 0.5, 0.25, 0.1, "#39d353"),
        ("PV Generation (1-dim)", 0.15, 0.35, 0.25, 0.1, "#f7c948"),
        ("Neighbor States (14-dim)", 0.15, 0.2, 0.25, 0.1, "#a371f7"),
        ("Temporal Context (4-dim)", 0.15, 0.05, 0.25, 0.1, "#ff6b35"),
    ]
    
    for name, x, y, w, h, color in features:
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=color, edgecolor="white", linewidth=2, alpha=0.85))
        ax.text(x + w/2, y + h/2, name, ha="center", va="center", fontsize=10, fontweight="bold", color="white")
    
    ax.add_patch(plt.Rectangle((0.55, 0.2), 0.3, 0.7, facecolor="#1a1a2e", edgecolor="white", linewidth=2, alpha=0.9))
    ax.text(0.7, 0.55, "Total: 27-dim\\nper agent\\n\\n(8-bus system)\\n\\nConcatenated: 189-dim", 
            ha="center", va="center", fontsize=11, fontweight="bold", color="white")
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Figure 35 — Observation Space Feature Breakdown & Encoding", fontsize=13)
    ax.axis("off")
    save_fig(fig, get_fig_path(35, "figs_31_40"))


# ==================================================================
# FIGURE 36: Action Space Geometry
# ==================================================================
@timed_figure(36)
def fig_36_action_space_geometry():
    """Fig 36: Action space bounds and safety projection geometry."""
    logger.info("\\n[Fig 36] Action Space Bounds & Safety Projection Geometry")
    fig, ax = plt.subplots(figsize=(10, 10))
    
    tms_range = np.linspace(0.1, 2.0, 100)
    ps_range = np.linspace(1.0, 10.0, 100)
    TMS, PS = np.meshgrid(tms_range, ps_range)
    constraint = TMS * 0.5 + PS * 0.05 >= 0.3
    
    ax.contourf(TMS, PS, constraint.astype(float), levels=[0, 0.5, 1], colors=["#ff6b35", "#39d353"], alpha=0.3)
    ax.contour(TMS, PS, constraint.astype(float), levels=[0.5], colors="white", linewidths=2)

    # No Core currently calls bridge.run_evaluation(36) (the real
    # _eval_action_space_grid/action_space_grid_data path exists in
    # core_eval_bridge.py but nothing populates a cache from it), and the
    # constraint boundary above (TMS*0.5 + PS*0.05 >= 0.3) is an
    # illustrative formula, not the real CTI constraint — so this whole
    # figure is unconditionally synthetic. Seeded explicitly (was a bare
    # np.random.seed(42) global reseed) and always watermarked.
    rng = np.random.default_rng(36)
    raw_actions = rng.random((50, 2)) * [1.9, 9.0] + [0.1, 1.0]
    ax.scatter(raw_actions[:, 0], raw_actions[:, 1], c="#ff6b35", s=50, alpha=0.6, label="Raw Actions", edgecolors="white")

    proj_actions = raw_actions.copy()
    proj_actions[:, 0] = np.maximum(proj_actions[:, 0], 0.3)
    ax.scatter(proj_actions[:, 0], proj_actions[:, 1], c="#39d353", s=50, alpha=0.6, label="Projected (Safe)", edgecolors="white")

    for i in range(0, 50, 5):
        ax.annotate("", xy=(proj_actions[i, 0], proj_actions[i, 1]),
                   xytext=(raw_actions[i, 0], raw_actions[i, 1]),
                   arrowprops=dict(arrowstyle="->", color="gray", lw=1, alpha=0.5))

    ax.set_xlabel("Time Multiplier Setting (TMS)", fontsize=12)
    ax.set_ylabel("Plug Setting (PS)", fontsize=12)
    ax.set_title("Figure 36 — Action Space Bounds & Safety Projection Geometry", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    from evaluation.experiments._synthetic_label import mark_synthetic
    mark_synthetic(fig, note="no Core evaluation feeds this figure — illustrative constraint geometry + random action sample")
    save_fig(fig, get_fig_path(36, "figs_31_40"))


# ==================================================================
# FIGURE 37: Per-Agent Learning Curves
# ==================================================================
@timed_figure(37)
def fig_37_per_agent_curves():
    """Fig 37: Per-agent curves — from Core A per-agent reward data."""
    logger.info("\\n[Fig 37] Per-Agent Learning Curves")
    data = load_cache("core_A_mappo_safe_ewc_8bus_100pv")
    is_synthetic = data is None
    if data is None:
        logger.warning("Fig 37: 'core_A_mappo_safe_ewc_8bus_100pv' cache not found — using synthetic/hardcoded fallback data, NOT a real measurement.")
        n_eps, n_agents = 2000, 7
        data = {"episodes": np.arange(n_eps), "per_agent_returns_mean": np.random.randn(n_eps, n_agents).cumsum(axis=0) * 0.01 + np.linspace(-1, 1, n_eps)[:, None]}
    
    eps = data.get("episodes", np.arange(len(data.get("per_agent_returns_mean", []))))
    per_agent = data.get("per_agent_returns_mean", np.zeros((len(eps), 7)))
    n_agents = per_agent.shape[1] if len(per_agent.shape) > 1 else 7
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Figure 37 — Per-Agent Learning Curves (Individual Convergence)", fontsize=13)
    
    ax = axes[0]
    colors = plt.cm.tab10(np.linspace(0, 1, n_agents))
    window = 50
    for i in range(n_agents):
        rewards = per_agent[:, i] if len(per_agent.shape) > 1 else per_agent
        if len(rewards) >= window:
            smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
            ax.plot(eps[window-1:], smoothed, color=colors[i], lw=1.5, label=f"Agent {i}", alpha=0.8)
    ax.set_xlabel("Episode", fontsize=11)
    ax.set_ylabel("Smoothed Reward", fontsize=11)
    ax.set_title("All Agents (Overlaid)", fontsize=12)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    
    ax = axes[1]
    convergence_eps = []
    for i in range(n_agents):
        rewards = per_agent[:, i] if len(per_agent.shape) > 1 else per_agent
        final = rewards[-50:].mean() if len(rewards) > 50 else rewards.mean()
        target = 0.9 * final
        conv_ep = len(rewards)
        for ep in range(50, len(rewards)):
            if np.mean(rewards[ep-50:ep]) >= target:
                conv_ep = ep
                break
        convergence_eps.append(conv_ep)
    
    sorted_idx = np.argsort(convergence_eps)
    bars = ax.barh(range(n_agents), [convergence_eps[i] for i in sorted_idx],
                   color=[colors[i] for i in sorted_idx], edgecolor="white")
    ax.set_yticks(range(n_agents))
    ax.set_yticklabels([f"Agent {i}" for i in sorted_idx], fontsize=9)
    ax.set_xlabel("Convergence Episode", fontsize=11)
    ax.set_title("Convergence Speed Ranking", fontsize=12)
    ax.grid(True, alpha=0.3, axis="x")
    ax.invert_yaxis()
    
    if is_synthetic:
        from evaluation.experiments._synthetic_label import mark_synthetic
        mark_synthetic(fig, note="core_A_mappo_safe_ewc_8bus_100pv cache missing")
    save_fig(fig, get_fig_path(37, "figs_31_40"))


# ==================================================================
# FIGURE 38: Attention Weight Evolution
# ==================================================================
@timed_figure(38)
def fig_38_attention_evolution():
    """Fig 38: Attention evolution — from Core A attention snapshots."""
    logger.info("\\n[Fig 38] Attention Weight Evolution")
    data = load_cache("core_A_mappo_safe_ewc_8bus_100pv")
    is_synthetic = data is None
    if data is None:
        logger.warning("Fig 38: 'core_A_mappo_safe_ewc_8bus_100pv' cache not found — using synthetic/hardcoded fallback data, NOT a real measurement.")
        n_agents = 14
        data = {f"attention_snap_0": np.ones((n_agents, n_agents)) / n_agents,
                f"attention_snap_100": np.random.dirichlet(np.ones(n_agents), n_agents),
                f"attention_snap_500": np.random.dirichlet(np.ones(n_agents) * 2, n_agents),
                f"attention_snap_1000": np.random.dirichlet(np.ones(n_agents) * 3, n_agents),
                f"attention_snap_5000": np.random.dirichlet(np.ones(n_agents) * 4, n_agents),
                "attention_entropy_mean": 1.5 * np.exp(-np.arange(5000) / 1000) + 0.5}
    
    # Real aggregation (run_core_training.py's _aggregate_core_A_results)
    # stores keys as "attention_snap_{ep}_mean" (averaged across runs);
    # only the hardcoded synthetic fallback above uses the bare
    # "attention_snap_{ep}" form. Accept both so real data isn't
    # silently filtered down to an empty dict.
    snapshots = {}
    for key in data:
        if not key.startswith("attention_snap_"):
            continue
        ep_str = key[len("attention_snap_"):]
        if ep_str.endswith("_mean"):
            ep_str = ep_str[: -len("_mean")]
        if not ep_str.isdigit():
            continue
        snapshots[int(ep_str)] = data[key]

    if not snapshots:
        # Real Core A training uses the batched-ES trainer, whose loop
        # never calls the per-step attention-snapshot capture in
        # marl_agents.py (that capture only fires along the rollout-based
        # training path) — so a real cache can legitimately carry zero
        # attention_snap_* keys. Fall back to the labeled placeholder
        # rather than crashing on an empty gridspec.
        logger.warning("Fig 38: no 'attention_snap_*' entries in real data — "
                        "using synthetic/hardcoded fallback data, NOT a real measurement.")
        n_agents = 14
        data = {"attention_snap_0": np.ones((n_agents, n_agents)) / n_agents,
                "attention_snap_100": np.random.dirichlet(np.ones(n_agents), n_agents),
                "attention_snap_500": np.random.dirichlet(np.ones(n_agents) * 2, n_agents),
                "attention_snap_1000": np.random.dirichlet(np.ones(n_agents) * 3, n_agents),
                "attention_snap_5000": np.random.dirichlet(np.ones(n_agents) * 4, n_agents),
                "attention_entropy_mean": 1.5 * np.exp(-np.arange(5000) / 1000) + 0.5}
        snapshots = {0: data["attention_snap_0"], 100: data["attention_snap_100"],
                     500: data["attention_snap_500"], 1000: data["attention_snap_1000"],
                     5000: data["attention_snap_5000"]}
        is_synthetic = True

    entropy = data.get("attention_entropy_mean", np.array([]))

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, len(snapshots), hspace=0.4, wspace=0.3)
    fig.suptitle("Figure 38 — Attention Weight Evolution During Training", fontsize=14)
    
    for idx, ep in enumerate(sorted(snapshots.keys())):
        ax = fig.add_subplot(gs[0, idx])
        attn = snapshots[ep]
        im = ax.imshow(attn, cmap="viridis", aspect="auto", vmin=0, vmax=attn.max())
        ax.set_title(f"Episode {ep}", fontsize=10)
        plt.colorbar(im, ax=ax, shrink=0.7)
    
    ax = fig.add_subplot(gs[1, :])
    if len(entropy) > 0:
        ax.plot(entropy, "o-", color="#2563eb", lw=1.5, markersize=3, alpha=0.7)
        for ep in sorted(snapshots.keys()):
            if ep < len(entropy):
                ax.axvline(ep, color="red", ls="--", alpha=0.5, lw=1)
    ax.set_xlabel("Episode", fontsize=11)
    ax.set_ylabel("Attention Entropy", fontsize=11)
    ax.set_title("Entropy Decay Over Training (lower = more focused)", fontsize=12)
    ax.grid(True, alpha=0.3)
    
    ax = fig.add_subplot(gs[2, :])
    for ep in sorted(snapshots.keys()):
        attn = snapshots[ep]
        off_diag = attn[~np.eye(attn.shape[0], dtype=bool)].flatten()
        ax.hist(off_diag, bins=20, alpha=0.4, label=f"Ep {ep}", density=True)
    ax.set_xlabel("Attention Weight (off-diagonal)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title("Attention Distribution Evolution", fontsize=12)
    ax.legend(fontsize=8, ncol=len(snapshots))
    ax.grid(True, alpha=0.3)
    
    if is_synthetic:
        from evaluation.experiments._synthetic_label import mark_synthetic
        mark_synthetic(fig, note="core_A_mappo_safe_ewc_8bus_100pv cache missing")
    save_fig(fig, get_fig_path(38, "figs_31_40"))


# ==================================================================
# FIGURE 39: Ablation Study
# ==================================================================
@timed_figure(39)
def fig_39_ablation():
    """Fig 39: Ablation — from Core D."""
    logger.info("\\n[Fig 39] Systematic Ablation Study")
    data = load_cache("core_D_ablation")
    is_synthetic = data is None
    if data is None:
        logger.warning("Fig 39: 'core_D_ablation' cache not found — using synthetic/hardcoded fallback data, NOT a real measurement.")
        configs = ["MAPPO", "MAPPO+GNN", "MAPPO+GNN+Safe", "MAPPO+GNN+Safe+EWC", "MAPPO+GNN+Safe+EWC+Curr"]
        data = {cfg: [{"OF": 5.0 - i * 0.5 + np.random.rand() * 0.2, "CTI": 5.0 - i * 0.8 + np.random.rand() * 0.5} for _ in range(10)] for i, cfg in enumerate(configs)}
    
    configs = list(data.keys())
    of_means = [np.mean([r["OF"] for r in data[cfg]]) for cfg in configs]
    of_stds = [np.std([r["OF"] for r in data[cfg]]) for cfg in configs]
    cti_means = [np.mean([r["CTI"] for r in data[cfg]]) for cfg in configs]
    cti_stds = [np.std([r["CTI"] for r in data[cfg]]) for cfg in configs]
    
    colors = ["#2563eb", "#39d353", "#f7c948", "#ff6b35", "#a371f7"]
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Figure 39 — Systematic Ablation Study", fontsize=14)
    
    ax = axes[0]
    x = np.arange(len(configs))
    bars = ax.bar(x, of_means, yerr=of_stds, color=colors, edgecolor="white", capsize=5, alpha=0.9)
    for bar, val in zip(bars, of_means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(of_means)*0.02, f"{val:.3f}", 
                ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("+", "+\\n") for c in configs], fontsize=8)
    ax.set_ylabel("Objective Function (s)", fontsize=11)
    ax.set_title("Final Performance", fontsize=12)
    ax.grid(True, alpha=0.3, axis="y")
    
    ax = axes[1]
    bars = ax.bar(x, cti_means, yerr=cti_stds, color=colors, edgecolor="white", capsize=5, alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("+", "+\\n") for c in configs], fontsize=8)
    ax.set_ylabel("CTI Violations", fontsize=11)
    ax.set_title("Constraint Satisfaction", fontsize=12)
    ax.grid(True, alpha=0.3, axis="y")
    
    ax = axes[2]
    gains = [of_means[i-1] - of_means[i] if i > 0 else 0 for i in range(len(configs))]
    gain_colors = ["#39d353" if g > 0 else "#ff6b35" for g in gains]
    bars = ax.barh(range(len(configs)), gains, color=gain_colors, edgecolor="white")
    ax.set_yticks(range(len(configs)))
    ax.set_yticklabels([c.replace("MAPPO+", "") for c in configs], fontsize=9)
    ax.set_xlabel("Marginal Improvement (s)", fontsize=11)
    ax.set_title("Component Contribution", fontsize=12)
    ax.axvline(0, color="white", lw=1)
    ax.grid(True, alpha=0.3, axis="x")
    
    if is_synthetic:
        from evaluation.experiments._synthetic_label import mark_synthetic
        mark_synthetic(fig, note="core_D_ablation cache missing")
    save_fig(fig, get_fig_path(39, "figs_31_40"))


# ==================================================================
# FIGURE 40: Hyperparameter Sensitivity
# ==================================================================
@timed_figure(40)
def fig_40_hyperparameter():
    """Fig 40: Hyperparameter grid — from Core E."""
    logger.info("\\n[Fig 40] Hyperparameter Sensitivity Heatmap")
    data = load_cache("core_E_hparam")
    is_synthetic = data is None
    if data is None:
        logger.warning("Fig 40: 'core_E_hparam' cache not found — using synthetic/hardcoded fallback data, NOT a real measurement.")
        data = []
        for lamb in [1e2, 1e3, 1e4, 1e5, 1e6]:
            for alpha in [1e-5, 1e-4, 1e-3, 1e-2]:
                for hid in [16, 32, 64, 128]:
                    of_val = 2.0 + np.exp(-lamb / 1e5) + alpha * 100 + 1.0 / hid + np.random.rand() * 0.1
                    data.append({"lambda_ewc": lamb, "alpha_lambda": alpha, "hidden_dim": hid, "OF": of_val})
    
    lambdas = sorted(set(d["lambda_ewc"] for d in data))
    alphas = sorted(set(d["alpha_lambda"] for d in data))
    hiddens = sorted(set(d["hidden_dim"] for d in data))
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle("Figure 40 — Hyperparameter Sensitivity Heatmap", fontsize=14)
    
    for h_idx, hidden in enumerate(hiddens[:6]):
        ax = axes[h_idx // 3, h_idx % 3]
        of_matrix = np.zeros((len(lambdas), len(alphas)))
        for i, lamb in enumerate(lambdas):
            for j, alpha in enumerate(alphas):
                matches = [d for d in data if d["lambda_ewc"] == lamb and d["alpha_lambda"] == alpha and d["hidden_dim"] == hidden]
                if matches:
                    of_matrix[i, j] = np.mean([m["OF"] for m in matches])
        
        im = ax.imshow(of_matrix, cmap="RdYlGn_r", aspect="auto")
        ax.set_xticks(range(len(alphas)))
        ax.set_yticks(range(len(lambdas)))
        ax.set_xticklabels([f"{a:.0e}" for a in alphas], fontsize=8, rotation=45)
        ax.set_yticklabels([f"{l:.0e}" for l in lambdas], fontsize=8)
        ax.set_xlabel("alpha_lambda", fontsize=9)
        ax.set_ylabel("lambda_ewc", fontsize=9)
        ax.set_title(f"d_h = {hidden}", fontsize=10)
        plt.colorbar(im, ax=ax, shrink=0.7, label="OF (s)")
    
    if is_synthetic:
        from evaluation.experiments._synthetic_label import mark_synthetic
        mark_synthetic(fig, note="core_E_hparam cache missing")
    save_fig(fig, get_fig_path(40, "figs_31_40"))


# ==================================================================
# FIGURE 41: Sample Efficiency
# ==================================================================
@timed_figure(41)
def fig_41_sample_efficiency():
    """Fig 41: Sample efficiency — from Core A convergence data."""
    logger.info("\\n[Fig 41] Sample Efficiency Comparison")
    data = load_cache("core_A_mappo_safe_ewc_8bus_100pv")
    is_synthetic = data is None
    if data is None:
        logger.warning("Fig 41: 'core_A_mappo_safe_ewc_8bus_100pv' cache not found — using synthetic/hardcoded fallback data, NOT a real measurement.")
        n_eps = 2000
        data = {"global_returns_mean": -5 + 8 * (1 - np.exp(-np.arange(n_eps) / 300))}
    
    returns = data.get("global_returns_mean", np.zeros(2000))
    cum_steps = np.cumsum(np.ones(len(returns)) * 50)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(cum_steps, returns, color="#2563eb", lw=2, label="MAPPO-Safe-EWC")
    milestones = [10000, 50000, 100000]
    for ms in milestones:
        if ms < len(cum_steps):
            idx = np.searchsorted(cum_steps, ms)
            ax.axvline(ms, color="gray", ls="--", alpha=0.5)
            ax.scatter([ms], [returns[idx]], color="red", s=50, zorder=5)
    
    ax.set_xlabel("Cumulative Environment Steps", fontsize=12)
    ax.set_ylabel("Episode Return", fontsize=12)
    ax.set_title("Figure 41 — Sample Efficiency Comparison", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    if is_synthetic:
        from evaluation.experiments._synthetic_label import mark_synthetic
        mark_synthetic(fig, note="core_A_mappo_safe_ewc_8bus_100pv cache missing")
    save_fig(fig, get_fig_path(41, "figs_31_40"))


# ==================================================================
# FIGURE 42: Warm-Start vs Cold-Start
# ==================================================================
@timed_figure(42)
def fig_42_warm_start():
    """Fig 42: Warm-start vs cold-start — from Core B extension."""
    logger.info("\\n[Fig 42] Warm-Start vs. Cold-Start Initialization")
    data = load_cache("core_B_curriculum")
    is_synthetic = data is None
    if data is None:
        logger.warning("Fig 42: 'core_B_curriculum' cache not found — using synthetic/hardcoded fallback data, NOT a real measurement.")
        data = {"with_ewc": [{"task_performances": np.array([[0.85,0,0,0],[0.83,0.84,0,0],[0.82,0.83,0.84,0],[0.81,0.82,0.83,0.84]])}]}

    ewc_mats = [r["task_performances"] for r in data.get("with_ewc", [])]
    avg_mat = np.mean(ewc_mats, axis=0) if ewc_mats else np.eye(4) * 0.8
    n_tasks = avg_mat.shape[0]

    warm_start = np.diag(avg_mat)
    cold_start = [0.3, 0.35, 0.4, 0.45][:n_tasks]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(n_tasks)
    width = 0.35
    ax.bar(x - width/2, warm_start, width, label="Warm-Start (Curriculum)", color="#2563eb", alpha=0.8, edgecolor="white")
    ax.bar(x + width/2, cold_start, width, label="Cold-Start (Scratch)", color="#ff6b35", alpha=0.8, edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(["0%", "20%", "40%", "60%"][:n_tasks])
    ax.set_xlabel("PV Penetration Level", fontsize=12)
    ax.set_ylabel("Performance", fontsize=12)
    ax.set_title("Figure 42 — Warm-Start vs. Cold-Start Initialization", fontsize=13)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis="y")
    if is_synthetic:
        from evaluation.experiments._synthetic_label import mark_synthetic
        mark_synthetic(fig, note="core_B_curriculum cache missing")
    save_fig(fig, get_fig_path(42, "figs_31_40"))

# ==================================================================
# FIGURE 43: Relay Coordination Time-Distance Diagram
# ==================================================================
@timed_figure(43)
def fig_43_coordination_diagram():
    """Fig 43: Coordination diagram — from Core J."""
    logger.info("\\n[Fig 43] Relay Coordination Time-Distance Diagram")
    data = load_cache("core_J_protection")
    is_synthetic = data is None
    if data is None:
        logger.warning("Fig 43: 'core_J_protection' cache not found — using synthetic/hardcoded fallback data, NOT a real measurement.")
        n_pairs = 21
        distances = np.linspace(0, 10, n_pairs)
        op_times_primary = 0.1 + 0.05 * distances + np.random.normal(0, 0.01, n_pairs)
        op_times_backup = 0.3 + 0.08 * distances + np.random.normal(0, 0.01, n_pairs)
        data = {"coordination": {"distances": distances, "primary": op_times_primary, "backup": op_times_backup}}
    
    coord = data.get("coordination", {})
    distances = coord.get("distances", np.linspace(0, 10, 21))
    primary = coord.get("primary", 0.1 + 0.05 * distances)
    backup = coord.get("backup", 0.3 + 0.08 * distances)
    
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(distances, primary, 'o-', color="#2563eb", lw=2, markersize=6, label="Primary Relay")
    ax.plot(distances, backup, 's-', color="#ff6b35", lw=2, markersize=6, label="Backup Relay")
    ax.fill_between(distances, primary, backup, alpha=0.1, color="#39d353", label="CTI Margin")
    ax.axhspan(0.2, 0.4, alpha=0.1, color="yellow", label="Target CTI Zone")
    
    ax.set_xlabel("Distance from Source (km)", fontsize=12)
    ax.set_ylabel("Operating Time (s)", fontsize=12)
    ax.set_title("Figure 43 — Relay Coordination Time-Distance Diagram (8-Bus, 100% PV)", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    if is_synthetic:
        from evaluation.experiments._synthetic_label import mark_synthetic
        mark_synthetic(fig, note="core_J_protection cache missing")
    save_fig(fig, get_fig_path(43, "figs_41_50"))


# ==================================================================
# FIGURE 44: Fault Current Contribution
# ==================================================================
@timed_figure(44)
def fig_44_fault_contribution():
    """Fig 44: Fault current contribution pie chart — from Core J."""
    logger.info("\\n[Fig 44] Fault Current Contribution Pie Chart")
    data = load_cache("core_J_protection")
    is_synthetic = data is None
    if data is None:
        logger.warning("Fig 44: 'core_J_protection' cache not found — using synthetic/hardcoded fallback data, NOT a real measurement.")
        data = {"fault_contribution": {"PV": 35, "Grid": 45, "Synchronous": 20}}
    
    contrib = data.get("fault_contribution", {"PV": 35, "Grid": 45, "Synchronous": 20})
    # The real evaluator (training_pipeline.py's _eval_fault_contributions)
    # legitimately can't compute this without pandapower installed, and
    # records {"error": "..."} rather than crashing — but that sentinel
    # isn't plottable as pie-slice sizes, so treat it the same as a
    # missing cache: fall back to the honest placeholder.
    if not contrib or "error" in contrib or not all(isinstance(v, (int, float)) for v in contrib.values()):
        logger.warning(f"Fig 44: 'fault_contribution' data unusable ({contrib}) — using synthetic/hardcoded fallback data, NOT a real measurement.")
        contrib = {"PV": 35, "Grid": 45, "Synchronous": 20}
        is_synthetic = True
    labels = list(contrib.keys())
    sizes = list(contrib.values())
    palette = ["#f7c948", "#2563eb", "#39d353", "#ff6b35", "#8b5cf6", "#14b8a6"]
    colors = [palette[i % len(palette)] for i in range(len(sizes))]
    explode = [0.05] + [0] * (len(sizes) - 1)

    fig, ax = plt.subplots(figsize=(10, 8))
    wedges, texts, autotexts = ax.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
                                       shadow=False, startangle=90, textprops={'fontsize': 12, 'fontweight': 'bold'})
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    ax.set_title("Figure 44 — Fault Current Contribution Pie Chart (100% PV)", fontsize=13)
    if is_synthetic:
        from evaluation.experiments._synthetic_label import mark_synthetic
        mark_synthetic(fig, note="core_J_protection cache missing")
    save_fig(fig, get_fig_path(44, "figs_41_50"))


# ==================================================================
# FIGURE 45: Daily PV Profile & Adaptive Relay Response
# ==================================================================
@timed_figure(45)
def fig_45_daily_profile():
    """Fig 45: Daily PV generation profile — from Core J."""
    logger.info("\\n[Fig 45] Daily PV Profile & Adaptive Relay Response")
    data = load_cache("core_J_protection")
    is_synthetic = data is None
    if data is None:
        logger.warning("Fig 45: 'core_J_protection' cache not found — using synthetic/hardcoded fallback data, NOT a real measurement.")
        hours = np.arange(24)
        pv_gen = np.maximum(0, np.sin(np.pi * hours / 12) * 100)
        relay_response = 0.2 + 0.1 * (1 - pv_gen / 100)
        data = {"daily_profile": {"hours": hours, "pv_generation": pv_gen, "relay_response": relay_response}}
    
    profile = data.get("daily_profile", {})
    # _eval_daily_pv's real output (see core_eval_bridge._extract_fig_data,
    # fig 45) is keyed "tms"/"ps"/"of"/"irradiance" — completely different
    # keys from the synthetic fallback's "hours"/"pv_generation"/
    # "relay_response". Reading the fallback's key names against a REAL
    # cache hit silently missed every .get() and fell back to the exact
    # same fabricated sinusoidal formulas — undisclosed, since is_synthetic
    # stayed False. Branch on which shape is actually present instead.
    if not is_synthetic and "irradiance" in profile:
        irradiance = np.asarray(profile["irradiance"], dtype=float)
        hours = np.arange(len(irradiance))
        pv_gen = 100.0 * irradiance / max(float(np.max(irradiance)), 1e-9)
        relay_response = np.asarray(profile.get("of", []), dtype=float)
        response_label = "Objective Function (learned TMS/PS that hour)"
        response_ylim = None
    else:
        hours = profile.get("hours", np.arange(24))
        pv_gen = profile.get("pv_generation", np.maximum(0, np.sin(np.pi * hours / 12) * 100))
        relay_response = profile.get("relay_response", 0.2 + 0.1 * (1 - pv_gen / 100))
        response_label = "Relay Response Time (s)"
        response_ylim = (0.1, 0.4)

    fig, ax1 = plt.subplots(figsize=(12, 6))
    color1 = "#f7c948"
    ax1.fill_between(hours, pv_gen, alpha=0.3, color=color1)
    ax1.plot(hours, pv_gen, color=color1, lw=2, label="PV Generation (%)")
    ax1.set_xlabel("Hour of Day", fontsize=12)
    ax1.set_ylabel("PV Generation (%)", color=color1, fontsize=12)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_xlim(0, 23)
    ax1.set_xticks(range(0, 24, 2))

    ax2 = ax1.twinx()
    color2 = "#2563eb"
    ax2.plot(hours, relay_response, color=color2, lw=2, label=response_label)
    ax2.set_ylabel(response_label, color=color2, fontsize=12)
    ax2.tick_params(axis='y', labelcolor=color2)
    if response_ylim is not None:
        ax2.set_ylim(*response_ylim)

    fig.suptitle("Figure 45 — Daily PV Generation Profile & Adaptive Relay Response", fontsize=13)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    ax1.grid(True, alpha=0.3)
    if is_synthetic:
        from evaluation.experiments._synthetic_label import mark_synthetic
        mark_synthetic(fig, note="core_J_protection cache missing")
    save_fig(fig, get_fig_path(45, "figs_41_50"))


# ==================================================================
# FIGURE 46: Relay Operating Characteristic Curves
# ==================================================================
@timed_figure(46)
def fig_46_relay_curves():
    """Fig 46: Relay operating curves — from Core J."""
    logger.info("\\n[Fig 46] Relay Operating Characteristic Curves")
    data = load_cache("core_J_protection")
    is_synthetic = data is None
    if data is None:
        logger.warning("Fig 46: 'core_J_protection' cache not found — using synthetic/hardcoded fallback data, NOT a real measurement.")
        current_multiples = np.linspace(1, 20, 100)
        tds, psm = 0.5, 2.0
        op_time = tds * 0.14 / ((current_multiples / psm) ** 0.02 - 1)
        data = {"relay_curves": {"current_multiples": current_multiples, "operating_time": op_time, "TDS": tds, "PSM": psm}}
    
    curves = data.get("relay_curves", {})
    # _eval_relay_curves's real output (see core_eval_bridge._extract_fig_data,
    # fig 46) is keyed "relay_0".."relay_{n-1}", each holding "If_range_A"
    # (Amps, not current multiples) / "t_op_s" / "TDS" / "PSM" — none of
    # curves.get("current_multiples"/"operating_time"/"TDS"/"PSM") exist on
    # a real cache hit, so every .get() silently missed and fell back to
    # the identical fabricated IDMT formula even when real per-relay curves
    # were available. Plot relay 0's real curve (converting Amps to current
    # multiples via I/PSM) when present.
    real_relay_keys = sorted(k for k in curves.keys() if str(k).startswith("relay_"))
    if not is_synthetic and real_relay_keys:
        relay0 = curves[real_relay_keys[0]]
        tds = relay0["TDS"]
        psm = relay0["PSM"]
        if_range_a = np.asarray(relay0["If_range_A"], dtype=float)
        current_multiples = if_range_a / psm
        # t_op_s is +inf for If <= PS (relay does not trip) — replace with
        # NaN so matplotlib leaves a real gap instead of plotting garbage.
        op_time = np.asarray(relay0["t_op_s"], dtype=float)
        op_time[~np.isfinite(op_time)] = np.nan
        relay_note = f" (relay 0 of {len(real_relay_keys)})"
    else:
        current_multiples = curves.get("current_multiples", np.linspace(1, 20, 100))
        op_time = curves.get("operating_time", 0.5 * 0.14 / ((current_multiples / 2.0) ** 0.02 - 1))
        tds = curves.get("TDS", 0.5)
        psm = curves.get("PSM", 2.0)
        relay_note = ""

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(current_multiples, op_time, color="#2563eb", lw=2,
            label=f"Learned: TDS={tds:.2f}, PS={psm:.2f}{relay_note}")

    with np.errstate(divide="ignore", invalid="ignore"):
        traditional = 0.3 * 0.14 / ((current_multiples / 1.5) ** 0.02 - 1)
    # Only defined (and only means "does not trip") for I/PS > 1.5 — below
    # that the formula produces a negative/non-physical "operating time".
    traditional = np.where(current_multiples > 1.5, traditional, np.nan)
    ax.plot(current_multiples, traditional, '--', color="#ff6b35", lw=2, label="Traditional: TDS=0.30, PS=1.50")

    ax.set_xlabel("Current Multiple (I / I_pickup)", fontsize=12)
    ax.set_ylabel("Operating Time (s)", fontsize=12)
    ax.set_title("Figure 46 — Relay Operating Characteristic Curves with Learned Settings", fontsize=13)
    if is_synthetic:
        # Fixed range/scale the synthetic formula above was authored for.
        ax.set_xlim(1, 20)
        ax.set_ylim(0, 5)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    if is_synthetic:
        from evaluation.experiments._synthetic_label import mark_synthetic
        mark_synthetic(fig, note="core_J_protection cache missing")
    save_fig(fig, get_fig_path(46, "figs_41_50"))


# ==================================================================
# FIGURE 47: Sympathetic Tripping Prevention
# ==================================================================
@timed_figure(47)
def fig_47_sympathetic_tripping():
    """Fig 47: Sympathetic tripping prevention — from Core J / core_eval_fig47."""
    logger.info("\\n[Fig 47] Sympathetic Tripping Prevention Analysis")
    # core_eval_fig47.json holds REAL data from
    # training_pipeline._eval_sympathetic_tripping: {"sympathetic": <CTI
    # violation count under the trained policy>, "traditional": <count
    # under fixed, non-adaptive settings>} — a count comparison, not the
    # "cases_prevented/cases_total out of a fixed scenario set" shape
    # this figure originally assumed. load_cache("core_J_protection")
    # is NOT used here: its bridge_map returns whichever of
    # fig44/45/46/47's cache files happens to exist first, which can
    # silently hand this function a different figure's data.
    real = load_bridge_cache(47)
    is_synthetic = real is None or "sympathetic" not in real or "traditional" not in real
    if is_synthetic:
        logger.warning("Fig 47: 'core_eval_fig47.json' cache not found/incomplete — using synthetic/hardcoded fallback data, NOT a real measurement.")
        marl_viol, trad_viol = 5, 20
    else:
        marl_viol, trad_viol = int(real["sympathetic"]), int(real["traditional"])

    # Honest reframing: the real signal is a CTI-violation COUNT (lower
    # is better) for MARL vs. traditional settings, not a discrete
    # prevented/not-prevented tally out of a fixed case set. "Prevented"
    # is the reduction in violations MARL achieves relative to traditional.
    total = max(trad_viol, marl_viol, 1)
    prevented = max(total - marl_viol, 0)
    not_prevented = total - prevented

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Figure 47 — Sympathetic Tripping Prevention Analysis", fontsize=13)

    ax = axes[0]
    labels = ["Prevented\n(vs. traditional)", "Remaining\nviolations"]
    sizes = [prevented, not_prevented]
    colors = ["#39d353", "#ff6b35"]
    wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, autopct='%1.0f%%',
                                       startangle=90, textprops={'fontsize': 11, 'fontweight': 'bold'})
    for autotext in autotexts:
        autotext.set_color('white')
    ax.set_title(f"MARL: {marl_viol} violations vs. Traditional: {trad_viol}", fontsize=12)

    # The trained policy is only ever evaluated against ONE fault type
    # per run (config.fault_type) — there is no real per-fault-type
    # breakdown anywhere in training_pipeline._eval_sympathetic_tripping,
    # so this panel is illustrative regardless of whether the pie chart
    # above used real data.
    ax = axes[1]
    scenarios = ["Line-to-Ground", "Line-to-Line", "3-Phase", "High-Z"]
    prevented_by_type = [4, 5, 3, 3]
    not_prevented_by_type = [1, 0, 1, 1]
    x = np.arange(len(scenarios))
    width = 0.35
    ax.bar(x - width/2, prevented_by_type, width, label="Prevented", color="#39d353", alpha=0.8, edgecolor="white")
    ax.bar(x + width/2, not_prevented_by_type, width, label="Not Prevented", color="#ff6b35", alpha=0.8, edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=15, ha="right")
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("By Fault Type", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    _annotate_panel_synthetic(ax, "no per-fault-type breakdown in _eval_sympathetic_tripping")

    if is_synthetic:
        from evaluation.experiments._synthetic_label import mark_synthetic
        mark_synthetic(fig, note="core_eval_fig47.json cache missing — entire figure fabricated")
    save_fig(fig, get_fig_path(47, "figs_41_50"))


# ==================================================================
# FIGURE 48: SHAP Feature Importance
# ==================================================================
@timed_figure(48)
def fig_48_shap():
    """Fig 48: SHAP feature importance — from core_eval_fig48 (real KernelExplainer output)."""
    logger.info("\\n[Fig 48] SHAP Feature Importance")
    # Real obs schema (marl/marl_env.py DynamicDOCREnv._get_obs, OBS_DIM=10):
    # note ps_n appears twice by construction (a pre-existing duplicate
    # feature, not a labeling error introduced here).
    features = ["TMS (norm.)", "PS (norm.)", "Fault current (norm.)", "CTI slack (norm.)",
                "CTI violation flag", "Own trip time (norm.)", "PS (norm.) [dup]",
                "PV generation level", "Topology change flag", "Sensor quality"]

    real = load_bridge_cache(48)
    shap_arr = None
    if real is not None and real.get("shap") is not None:
        shap_arr = np.asarray(real["shap"])
    is_synthetic = shap_arr is None or shap_arr.size == 0

    if is_synthetic:
        logger.warning("Fig 48: 'core_eval_fig48.json' cache not found/empty — using synthetic/hardcoded fallback data, NOT a real measurement.")
        rng = np.random.default_rng(42)
        shap_values = np.sort(rng.exponential(0.1, len(features)))[::-1]
        order = np.arange(len(features))
    else:
        # shap_arr shape: (n_actions, n_explain, n_features) from
        # shap.KernelExplainer on the actor's multi-output mean action.
        mean_abs = np.abs(shap_arr).mean(axis=tuple(range(shap_arr.ndim - 1)))
        if mean_abs.shape[0] != len(features):
            logger.warning(f"Fig 48: cached SHAP feature count ({mean_abs.shape[0]}) != known obs schema ({len(features)}); truncating/padding to match.")
            mean_abs = np.resize(mean_abs, len(features))
        order = np.argsort(mean_abs)[::-1]
        shap_values = mean_abs[order]

    ordered_features = [features[i] for i in order]

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(ordered_features)))[::-1]
    bars = ax.barh(range(len(ordered_features)), shap_values, color=colors, edgecolor="white")
    ax.set_yticks(range(len(ordered_features)))
    ax.set_yticklabels(ordered_features, fontsize=10)
    ax.set_xlabel("Mean |SHAP Value|", fontsize=12)
    ax.set_title("Figure 48 — SHAP Feature Importance for Relay Decisions", fontsize=13)
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis="x")
    if is_synthetic:
        from evaluation.experiments._synthetic_label import mark_synthetic
        mark_synthetic(fig, note="core_eval_fig48.json cache missing")
    save_fig(fig, get_fig_path(48, "figs_41_50"))


# ==================================================================
# FIGURE 49: Decision Boundary Visualization
# ==================================================================
@timed_figure(49)
def fig_49_decision_boundary():
    """Fig 49: Decision boundary — 2D t-SNE projection from core_eval_fig49 (real _eval_tsne output)."""
    logger.info("\\n[Fig 49] Decision Boundary Visualization")
    real = load_bridge_cache(49)
    embeddings = actions = None
    if real is not None:
        emb = real.get("embeddings")
        act = real.get("actions")
        if emb is not None and act is not None and len(emb) > 0:
            embeddings, actions = np.asarray(emb), np.asarray(act)
    is_synthetic = embeddings is None

    if is_synthetic:
        logger.warning("Fig 49: 'core_eval_fig49.json' cache not found/empty — using synthetic/hardcoded fallback data, NOT a real measurement.")
        rng = np.random.default_rng(42)
        n_points = 500
        x = rng.standard_normal(n_points)
        y = rng.standard_normal(n_points)
        labels = (x + y > 0).astype(int)
    else:
        x, y = embeddings[:, 0], embeddings[:, 1]
        # Real per-point "action class": whether the trained policy's
        # aggregate action pushes settings up or down at that sampled
        # observation — the sign of the summed action vector.
        labels = (actions.sum(axis=-1) > 0).astype(int)

    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(x, y, c=labels, cmap="RdYlGn", alpha=0.6, s=30, edgecolors="white", linewidth=0.5)

    if is_synthetic:
        x_line = np.linspace(-3, 3, 100)
        ax.plot(x_line, -x_line, 'k--', lw=2, label="Decision Boundary")
        ax.contourf(np.linspace(-3, 3, 50), np.linspace(-3, 3, 50),
                    np.zeros((50, 50)), levels=[0, 0.5, 1], colors=["#ff6b35", "#39d353"], alpha=0.1)
        ax.legend(fontsize=11)

    ax.set_xlabel("t-SNE Dimension 1", fontsize=12)
    ax.set_ylabel("t-SNE Dimension 2", fontsize=12)
    ax.set_title("Figure 49 — Decision Boundary Visualization (2D Projection)", fontsize=13)
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label="Action Class")
    if is_synthetic:
        from evaluation.experiments._synthetic_label import mark_synthetic
        mark_synthetic(fig, note="core_eval_fig49.json cache missing")
    save_fig(fig, get_fig_path(49, "figs_41_50"))


# ==================================================================
# FIGURE 50: Critical P/B Pair Identification
# ==================================================================
@timed_figure(50)
def fig_50_critical_pairs():
    """Fig 50: Critical pair identification — from core_eval_fig50 (real _eval_critical_pairs output)."""
    logger.info("\\n[Fig 50] Critical P/B Pair Identification")
    real = load_bridge_cache(50)
    real_pairs = None
    if isinstance(real, list) and len(real) > 0:
        real_pairs = real
    is_synthetic = real_pairs is None

    if is_synthetic:
        logger.warning("Fig 50: 'core_eval_fig50.json' cache not found/empty — using synthetic/hardcoded fallback data, NOT a real measurement.")
        rng = np.random.default_rng(42)
        pairs = [f"P{i}-B{j}" for i in range(1, 8) for j in range(i + 1, 8)]
        metric = np.sort(rng.poisson(5, len(pairs)))[::-1]
        xlabel = "CTI Violation Count"
    else:
        # Real signal is a per-pair CTI margin (distance_to_violation:
        # t_bkp - t_pri - CTI; negative = violating), not a violation
        # COUNT — there's no repeated-trial tally in
        # _eval_critical_pairs, just one margin per pair at the
        # evaluated operating point. Sort most-critical (smallest
        # margin) first, matching the figure's "priority ranking" intent.
        ordered = sorted(real_pairs, key=lambda p: p.get("distance_to_violation", 0.0))
        pairs = [f"P{p['primary']}-B{p['backup']}" for p in ordered]
        metric = np.array([p.get("distance_to_violation", 0.0) for p in ordered])
        xlabel = "CTI Margin (s) — negative = violating"

    fig, ax = plt.subplots(figsize=(12, 8))
    colors = plt.cm.Reds(np.linspace(0.3, 1.0, len(pairs)))[::-1]
    bars = ax.barh(range(len(pairs)), metric, color=colors, edgecolor="white")
    ax.set_yticks(range(len(pairs)))
    ax.set_yticklabels(pairs, fontsize=8)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_title("Figure 50 — Critical P/B Pair Identification & Priority Ranking", fontsize=13)
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis="x")
    if is_synthetic:
        from evaluation.experiments._synthetic_label import mark_synthetic
        mark_synthetic(fig, note="core_eval_fig50.json cache missing")
    save_fig(fig, get_fig_path(50, "figs_41_50"))


# ==================================================================
# FIGURE 52: Digital Twin Pipeline (Conceptual)
# ==================================================================
@timed_figure(52)
def fig_52_digital_twin_pipeline():
    """Fig 52: Digital twin training-to-deployment pipeline — schematic."""
    logger.info("\\n[Fig 52] Digital Twin Training-to-Deployment Pipeline")
    fig, ax = plt.subplots(figsize=(14, 6))
    
    stages = ["Data\\nCollection", "Digital Twin\\nTraining", "Validation &\\nTesting", 
              "Model\\nCompression", "Edge\\nDeployment", "Real-Time\\nMonitoring"]
    x_pos = np.linspace(0.08, 0.92, len(stages))
    
    for i, (x, stage) in enumerate(zip(x_pos, stages)):
        color = ["#2563eb", "#39d353", "#f7c948", "#ff6b35", "#a371f7", "#2563eb"][i]
        ax.add_patch(plt.Rectangle((x-0.06, 0.35), 0.12, 0.3, facecolor=color, edgecolor="white", linewidth=2, alpha=0.85))
        ax.text(x, 0.5, stage, ha="center", va="center", fontsize=9, fontweight="bold", color="white")
        if i < len(stages) - 1:
            ax.annotate("", xy=(x_pos[i+1]-0.06, 0.5), xytext=(x+0.06, 0.5),
                       arrowprops=dict(arrowstyle="->", color="gray", lw=2))
    
    ax.annotate("", xy=(0.08, 0.3), xytext=(0.92, 0.3),
               arrowprops=dict(arrowstyle="->", color="red", lw=1.5, ls="--", connectionstyle="arc3,rad=-0.3"))
    ax.text(0.5, 0.15, "Feedback Loop: Performance Monitoring -> Retraining", ha="center", fontsize=10, 
            style="italic", color="red")
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Figure 52 — Digital Twin Training-to-Deployment Pipeline", fontsize=13)
    ax.axis("off")
    save_fig(fig, get_fig_path(52, "figs_41_50"))


# ==================================================================
# FIGURE 53: Model Compression & Quantization
# ==================================================================
@timed_figure(53)
def fig_53_model_compression():
    """Fig 53: Model compression impact — from Core I."""
    logger.info("\\n[Fig 53] Model Compression & Quantization Impact")
    data = load_cache("core_I_deployment")
    is_synthetic = data is None
    if is_synthetic:
        logger.warning("Fig 53: 'core_I_deployment' cache not found — using hardcoded "
                        "placeholder compression numbers, NOT a real measurement.")
        data = {"compression": {"original": {"size_mb": 45, "performance": 0.85}, "int8": {"size_mb": 12, "performance": 0.84},
                              "pruning": {"size_mb": 22, "performance": 0.83}, "distillation": {"size_mb": 18, "performance": 0.82}}}

    comp = data.get("compression", {})
    # The real evaluator (core_eval_bridge.py's fig-53 extractor) only
    # ever measures ONE compression config per run — {"of", "size_kb",
    # "drop"} — not a per-method {"size_mb", "performance"} comparison
    # across original/int8/pruning/distillation. Rather than inventing a
    # method breakdown that was never actually measured, fall back to
    # the labeled placeholder when the shape doesn't match.
    if not comp or not all(isinstance(v, dict) and "size_mb" in v and "performance" in v for v in comp.values()):
        logger.warning(f"Fig 53: 'compression' data not in per-method {{size_mb, performance}} form ({comp}) — "
                        "using hardcoded placeholder compression numbers, NOT a real measurement.")
        comp = {"original": {"size_mb": 45, "performance": 0.85}, "int8": {"size_mb": 12, "performance": 0.84},
                "pruning": {"size_mb": 22, "performance": 0.83}, "distillation": {"size_mb": 18, "performance": 0.82}}
        is_synthetic = True
    methods = list(comp.keys())
    sizes = [comp[m]["size_mb"] for m in methods]
    perfs = [comp[m]["performance"] for m in methods]
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    x = np.arange(len(methods))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, sizes, width, label="Model Size (MB)", color="#2563eb", alpha=0.8, edgecolor="white")
    ax1.set_ylabel("Model Size (MB)", color="#2563eb", fontsize=12)
    ax1.tick_params(axis='y', labelcolor="#2563eb")
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, fontsize=10)
    
    ax2 = ax1.twinx()
    bars2 = ax2.bar(x + width/2, perfs, width, label="Performance", color="#39d353", alpha=0.8, edgecolor="white")
    ax2.set_ylabel("Performance", color="#39d353", fontsize=12)
    ax2.tick_params(axis='y', labelcolor="#39d353")
    ax2.set_ylim(0.75, 0.9)
    
    ax1.set_title("Figure 53 — Model Compression & Quantization Impact", fontsize=13)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    ax1.grid(True, alpha=0.3, axis="y")
    if is_synthetic:
        from evaluation.experiments._synthetic_label import mark_synthetic
        mark_synthetic(fig, note="core_I_deployment cache missing")
    save_fig(fig, get_fig_path(53, "figs_51_58"))


# ==================================================================
# FIGURE 54: Adversarial Robustness
# ==================================================================
@timed_figure(54)
def fig_54_adversarial():
    """Fig 54: Adversarial robustness — from Core G."""
    logger.info("\\n[Fig 54] Adversarial Robustness")
    data = load_cache("core_G_robustness")
    is_synthetic = data is None
    if data is None:
        logger.warning("Fig 54: 'core_G_robustness' cache not found — using synthetic/hardcoded fallback data, NOT a real measurement.")
        data = {"adversarial": {"fgsm": {0.01: 0.82, 0.05: 0.70, 0.1: 0.55, 0.2: 0.35}, "pgd": {0.01: 0.80, 0.05: 0.65, 0.1: 0.45, 0.2: 0.20}}}
    
    adv = data.get("adversarial", {})
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for attack, color in [("fgsm", "#2563eb"), ("pgd", "#ff6b35")]:
        if attack in adv:
            epsilons = sorted([float(k) for k in adv[attack].keys()])
            values = [adv[attack][k] for k in epsilons]
            ax.plot(epsilons, values, 'o-', color=color, lw=2, markersize=8, label=attack.upper())
    
    ax.set_xlabel("Perturbation Magnitude (epsilon)", fontsize=12)
    ax.set_ylabel("Relative Performance", fontsize=12)
    ax.set_title("Figure 54 — Adversarial Robustness: Perturbed Observation Response", fontsize=13)
    ax.legend(fontsize=11)
    if is_synthetic:
        ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    if is_synthetic:
        from evaluation.experiments._synthetic_label import mark_synthetic
        mark_synthetic(fig, note="core_G_robustness cache missing")
    save_fig(fig, get_fig_path(54, "figs_51_58"))


# ==================================================================
# FIGURE 55: Comparison Table
# ==================================================================
@timed_figure(55)
def fig_55_comparison_table():
    """Fig 55: Comprehensive method comparison table — from Core C."""
    logger.info("\\n[Fig 55] Comprehensive Method Comparison Table")
    data = load_cache("core_C_baselines_8bus")
    is_synthetic = data is None
    if data is None:
        logger.warning("Fig 55: 'core_C_baselines_8bus' cache not found — using synthetic/hardcoded fallback data, NOT a real measurement.")
        methods = ["MAPPO-Safe-EWC", "MAPPO-Safe", "MAPPO", "IPPO", "Traditional"]
        data = {m: np.random.normal(3.0 - i * 0.3, 0.5, 50) for i, m in enumerate(methods)}
    
    methods = list(data.keys())
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis('off')

    # core_C_baselines only stores each method's per-episode POST-TRAINING
    # evaluation returns (see run_core_training._core_C_job) — no CTI-
    # satisfaction tracking and no training curve (so no real "episode
    # training converged at" value can be derived even from real data).
    # The last two columns were always np.random.uniform/randint, drawn
    # unconditionally (even on a real cache hit) and never disclosed since
    # is_synthetic only reflects whether the CACHE itself was found.
    # Seeded so the illustrative values are at least reproducible, and the
    # table always footnotes them regardless of is_synthetic.
    rng = np.random.default_rng(55)
    table_data = []
    for method in methods:
        vals = data[method]
        if hasattr(vals, '__len__') and len(vals) > 0:
            mean_val = np.mean(vals[-100:] if len(vals) > 100 else vals)
            std_val = np.std(vals[-100:] if len(vals) > 100 else vals)
            table_data.append([method, f"{mean_val:.3f}", f"{std_val:.3f}",
                             f"{rng.uniform(0.7, 0.95):.2f}*", f"{rng.integers(500, 5000)}*"])
        else:
            table_data.append([method, "N/A", "N/A", "N/A", "N/A"])

    columns = ["Method", "Final OF (s)", "Std Dev", "CTI Satisfied (%)", "Conv. Episodes"]
    table = ax.table(cellText=table_data, colLabels=columns, cellLoc='center', loc='center',
                     colWidths=[0.25, 0.15, 0.15, 0.2, 0.15])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    for i in range(len(columns)):
        table[(0, i)].set_facecolor("#2563eb")
        table[(0, i)].set_text_props(weight='bold', color='white')

    ax.text(0.5, 0.02, "* illustrative — core_C_baselines_8bus tracks neither CTI "
            "satisfaction nor a training convergence curve", transform=ax.transAxes,
            ha="center", fontsize=8, style="italic", color="#555")
    ax.set_title("Figure 55 — Comprehensive Method Comparison Table (8-Bus, 100% PV)", fontsize=13, pad=20)
    if is_synthetic:
        from evaluation.experiments._synthetic_label import mark_synthetic
        mark_synthetic(fig, note="core_C_baselines_8bus cache missing")
    save_fig(fig, get_fig_path(55, "figs_51_58"))


# ==================================================================
# FIGURE 56: Pareto Front
# ==================================================================
@timed_figure(56)
def fig_56_pareto_front():
    """Fig 56: Multi-objective Pareto front — from Core C."""
    logger.info("\\n[Fig 56] Multi-Objective Pareto Front")
    data = load_cache("core_C_baselines_8bus")
    is_synthetic = data is None
    if data is None:
        logger.warning("Fig 56: 'core_C_baselines_8bus' cache not found — using synthetic/hardcoded fallback data, NOT a real measurement.")
        methods = ["MAPPO-Safe-EWC", "MAPPO-Safe", "MAPPO", "IPPO", "Traditional"]
        data = {m: np.random.normal(3.0 - i * 0.3, 0.5, 50) for i, m in enumerate(methods)}
    
    # core_C_baselines stores only each method's per-episode returns — no
    # per-relay "operating speed" vs "selectivity" trade-off is measured
    # anywhere in Core C, so `data` (real or synthetic fallback) was never
    # actually read below; this whole scatter + the 5 named method_points
    # were fabricated unconditionally. Because is_synthetic only reflects
    # whether the cache FILE was found (not whether its contents were
    # used), a real cache hit still produced this identical fabricated
    # plot with no disclosure. Seeded explicitly and always watermarked —
    # there is no code path, real or synthetic, that measures this
    # trade-off.
    rng = np.random.default_rng(56)
    n_points = 100
    speed = rng.uniform(0.1, 0.9, n_points)
    selectivity = 1 - speed + rng.normal(0, 0.05, n_points)
    selectivity = np.clip(selectivity, 0, 1)

    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(speed, selectivity, c=speed + selectivity, cmap="RdYlGn", s=100, alpha=0.6, edgecolors="white")
    
    pareto_x = np.linspace(0.2, 0.8, 50)
    pareto_y = 1 - pareto_x * 0.8
    ax.plot(pareto_x, pareto_y, 'r--', lw=2, label="Pareto Front")
    
    method_points = {"MAPPO-Safe-EWC": (0.75, 0.85), "MAPPO-Safe": (0.65, 0.75), 
                     "MAPPO": (0.55, 0.65), "IPPO": (0.45, 0.55), "Traditional": (0.3, 0.4)}
    for method, (x, y) in method_points.items():
        ax.scatter([x], [y], s=200, marker='*', edgecolors='black', linewidths=1.5, zorder=5)
        ax.annotate(method, (x, y), textcoords="offset points", xytext=(10, 10), fontsize=9, fontweight='bold')
    
    ax.set_xlabel("Operating Speed (normalized)", fontsize=12)
    ax.set_ylabel("Selectivity (normalized)", fontsize=12)
    ax.set_title("Figure 56 — Multi-Objective Pareto Front (Speed vs. Selectivity)", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label="Combined Score")
    from evaluation.experiments._synthetic_label import mark_synthetic
    note = ("core_C_baselines_8bus cache missing" if is_synthetic else
            "core_C_baselines_8bus has no speed/selectivity trade-off data — "
            "entire plot is illustrative")
    mark_synthetic(fig, note=note)
    save_fig(fig, get_fig_path(56, "figs_51_58"))


# ==================================================================
# FIGURE 57: Bootstrap Confidence Interval Convergence
# ==================================================================
@timed_figure(57)
def fig_57_bootstrap_ci():
    """Fig 57: Bootstrap CI convergence — from Core A."""
    logger.info("\\n[Fig 57] Bootstrap Confidence Interval Convergence")
    data = load_cache("core_A_mappo_safe_ewc_8bus_100pv")
    is_synthetic = data is None
    if data is None:
        logger.warning("Fig 57: 'core_A_mappo_safe_ewc_8bus_100pv' cache not found — using synthetic/hardcoded fallback data, NOT a real measurement.")
        n_eps = 2000
        data = {"global_returns_mean": -5 + 8 * (1 - np.exp(-np.arange(n_eps) / 300)),
                "global_returns_std": np.ones(n_eps) * 0.8}
    
    returns = data.get("global_returns_mean", np.zeros(2000))
    std = data.get("global_returns_std", np.ones(len(returns)) * 0.5)
    
    np.random.seed(42)
    sample_sizes = np.logspace(1, np.log10(len(returns)), 50).astype(int)
    ci_widths = []
    for n in sample_sizes:
        samples = np.random.choice(returns[:n], size=(1000, n), replace=True)
        means = samples.mean(axis=1)
        ci_low, ci_high = np.percentile(means, [2.5, 97.5])
        ci_widths.append(ci_high - ci_low)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(sample_sizes, ci_widths, 'o-', color="#2563eb", lw=2, markersize=6)
    ax.axhline(0.5, color="red", ls="--", lw=1.5, label="Target CI Width")
    ax.set_xlabel("Number of Episodes", fontsize=12)
    ax.set_ylabel("95% CI Width", fontsize=12)
    ax.set_title("Figure 57 — Bootstrap Confidence Interval Convergence", fontsize=13)
    ax.set_xscale("log")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    if is_synthetic:
        from evaluation.experiments._synthetic_label import mark_synthetic
        mark_synthetic(fig, note="core_A_mappo_safe_ewc_8bus_100pv cache missing")
    save_fig(fig, get_fig_path(57, "figs_51_58"))


# ==================================================================
# FIGURE 58: Seasonal/Weather Generalization
# ==================================================================
@timed_figure(58)
def fig_58_seasonal():
    """Fig 58: Seasonal generalization — from Core K."""
    logger.info("\\n[Fig 58] Seasonal/Weather Generalization")
    data = load_cache("core_K_seasonal")
    is_synthetic = data is None
    if data is None:
        logger.warning("Fig 58: 'core_K_seasonal' cache not found — using synthetic/hardcoded fallback data, NOT a real measurement.")
        data = {"spring": {"performance": 0.83, "pv_variability": 0.3}, "summer": {"performance": 0.80, "pv_variability": 0.5},
                "autumn": {"performance": 0.84, "pv_variability": 0.25}, "winter": {"performance": 0.82, "pv_variability": 0.15}}

    # The real evaluator (core_eval_bridge.py's fig-58 extractor) only
    # measures {"of", "cti_ok"} per season — it has no notion of a 0-1
    # "performance" score or a "pv_variability" constant (that's
    # descriptive per-season metadata, not something the evaluation
    # measures). Rather than inventing a performance conversion that was
    # never actually validated, fall back to the labeled placeholder.
    if not data or not all(isinstance(v, dict) and "performance" in v and "pv_variability" in v for v in data.values()):
        logger.warning(f"Fig 58: season data missing 'performance'/'pv_variability' ({data}) — "
                        "using synthetic/hardcoded fallback data, NOT a real measurement.")
        data = {"spring": {"performance": 0.83, "pv_variability": 0.3}, "summer": {"performance": 0.80, "pv_variability": 0.5},
                "autumn": {"performance": 0.84, "pv_variability": 0.25}, "winter": {"performance": 0.82, "pv_variability": 0.15}}
        is_synthetic = True

    seasons = list(data.keys())
    perfs = [data[s]["performance"] for s in seasons]
    variabilities = [data[s]["pv_variability"] for s in seasons]
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    x = np.arange(len(seasons))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, perfs, width, label="Performance", color="#2563eb", alpha=0.8, edgecolor="white")
    ax1.set_ylabel("Performance", color="#2563eb", fontsize=12)
    ax1.tick_params(axis='y', labelcolor="#2563eb")
    ax1.set_xticks(x)
    ax1.set_xticklabels([s.capitalize() for s in seasons])
    ax1.set_ylim(0.7, 0.9)
    
    ax2 = ax1.twinx()
    bars2 = ax2.bar(x + width/2, variabilities, width, label="PV Variability", color="#f7c948", alpha=0.8, edgecolor="white")
    ax2.set_ylabel("PV Variability (sigma)", color="#f7c948", fontsize=12)
    ax2.tick_params(axis='y', labelcolor="#f7c948")
    ax2.set_ylim(0, 0.6)
    
    ax1.set_title("Figure 58 — Seasonal/Weather Generalization (Extended PV Profiles)", fontsize=13)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    ax1.grid(True, alpha=0.3, axis="y")
    if is_synthetic:
        from evaluation.experiments._synthetic_label import mark_synthetic
        mark_synthetic(fig, note="core_K_seasonal cache missing")
    save_fig(fig, get_fig_path(58, "figs_51_58"))

# ═════════════════════════════════════════════════════════════════
# FIGURE MAPPING (Complete — all scripted figures)
# ═════════════════════════════════════════════════════════════════
FIGURE_FUNCTIONS = {
    # Phase 1: System Architecture & Power System Diagrams
    2: fig_02_data_flow,
    3: fig_03_8bus_topology,
    4: fig_04_30bus_topology,
    5: fig_05_objective_vs_pv,
    6: fig_06_cti_vs_pv,
    7: fig_07_curriculum,
    8: fig_08_attention_weights,
    9: fig_09_cti_violations,
    10: fig_10_lagrange_multipliers,
    # 19 intentionally absent — see the comment above fig_20_byzantine's
    # module section header (previously fig_19_pv_ride_through here
    # duplicated/diverged from phase1_scripts/exp_fig19.py, the one the
    # real pipeline actually uses).

    # Phase 2: Methodology, Algorithms & Formulations
    29: fig_29_ewc_continual,
    34: fig_34_nn_architecture,
    35: fig_35_observation_space,
    36: fig_36_action_space_geometry,
    
    # Phase 3: Training Dynamics & Learning Curves
    12: fig_12_convergence_bands,
    16: fig_16_ewc_effectiveness,
    21: fig_21_action_projection,
    37: fig_37_per_agent_curves,
    38: fig_38_attention_evolution,
    
    # Phase 4: Performance Evaluation
    11: fig_11_nemenyi,
    
    # Phase 5: Statistical Analysis & Ranking
    31: fig_31_multi_metric_ranking,
    32: fig_32_effect_size,
    33: fig_33_radar_chart,
    55: fig_55_comparison_table,
    56: fig_56_pareto_front,
    57: fig_57_bootstrap_ci,
    
    # Phase 6: Robustness & Resilience
    13: fig_13_communication_stress,
    14: fig_14_n1_contingency,
    20: fig_20_byzantine,
    22: fig_22_bandwidth_tradeoff,
    24: fig_24_sensor_robustness,
    54: fig_54_adversarial,
    
    # Phase 7: Scalability & Transfer Learning
    15: fig_15_scalability,
    23: fig_23_unseen_faults,
    25: fig_25_zero_shot_transfer,
    
    # Phase 8: Deployment & Real-Time Implementation
    18: fig_18_inference_latency,
    53: fig_53_model_compression,
    
    # Phase 9: Neural Architecture & Interpretability
    48: fig_48_shap,
    49: fig_49_decision_boundary,
    50: fig_50_critical_pairs,
    
    # Phase 10: Ablation & Sensitivity Analysis
    39: fig_39_ablation,
    40: fig_40_hyperparameter,
    41: fig_41_sample_efficiency,
    42: fig_42_warm_start,
    
    # Phase 11: Supplementary Analysis
    43: fig_43_coordination_diagram,
    44: fig_44_fault_contribution,
    45: fig_45_daily_profile,
    46: fig_46_relay_curves,
    47: fig_47_sympathetic_tripping,
    52: fig_52_digital_twin_pipeline,
    58: fig_58_seasonal,
}

# Phase mapping for --phase CLI
PHASE_FIGURES = {
    1: [2, 3, 4, 5, 6, 7, 8, 9, 10, 19],
    2: [29, 34, 35, 36],
    3: [12, 16, 21, 37, 38],
    4: [11],
    5: [31, 32, 33, 55, 56, 57],
    6: [13, 14, 20, 22, 24, 54],
    7: [15, 23, 25],
    8: [18, 53],
    9: [48, 49, 50],
    10: [39, 40, 41, 42],
    11: [43, 44, 45, 46, 47, 52, 58],
}


# ═════════════════════════════════════════════════════════════════
# MAIN ORCHESTRATOR
# ═════════════════════════════════════════════════════════════════
def run_all_figures():
    """Generate all figures from cached data with progress tracking."""
    logger.info("\\n" + "=" * 70)
    logger.info("GENERATING ALL FIGURES FROM CACHED CORE TRAINING DATA")
    logger.info("=" * 70)
    
    fig_progress = FigureProgress(total_figures=len(FIGURE_FUNCTIONS))
    
    with ProgressTracker(total=len(FIGURE_FUNCTIONS), desc="[Figures] Generating", unit="fig") as pbar:
        for fig_num in sorted(FIGURE_FUNCTIONS.keys()):
            try:
                t0 = time.time()
                FIGURE_FUNCTIONS[fig_num]()
                duration = time.time() - t0
                fig_progress.add_completed(fig_num, duration)
                pbar.update(1)
                pbar.set_postfix(fig=fig_num, time=f"{duration:.1f}s")
            except Exception as e:
                duration = time.time() - t0
                fig_progress.add_failed(fig_num, str(e))
                logger.error(f"[Fig {fig_num}] FAILED: {e}")
                pbar.update(1)
                pbar.set_postfix(fig=fig_num, status="FAILED")
    
    # Print summary and save report
    fig_progress.print_summary()
    report_path = get_reports_dir() / f"figure_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    fig_progress.save_report(report_path)
    
    logger.info("\\n" + "=" * 70)
    logger.info("ALL FIGURES GENERATED")
    logger.info(f"Output directory: {get_figures_dir()}")
    logger.info(f"Report: {report_path}")
    logger.info("=" * 70)



def main():
    parser = argparse.ArgumentParser(
        description="Generate thesis figures from cached core training data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage:
  python run_post_processing.py --all              # All figures with progress
  python run_post_processing.py --fig 9 10 12        # Selected figures
  python run_post_processing.py --phase 2            # Phase 2 figures
  python run_post_processing.py --list              # List available figures
  python run_post_processing.py --report            # Show last figure generation report
        """
    )
    parser.add_argument("--all", action="store_true", help="Generate all figures")
    parser.add_argument("--fig", type=int, nargs="+", help="Specific figure numbers")
    parser.add_argument("--phase", type=int, help="Generate figures for a phase (1-11)")
    parser.add_argument("--list", action="store_true", help="List available figures")
    parser.add_argument("--report", action="store_true", help="Show last figure generation report")
    parser.add_argument("--fast", action="store_true", help="Enable fast mode")
    parser.add_argument("--config", "-c", type=str, help="Config file path")
    parser.add_argument("--preset", "-p", type=str, help="Preset name")

    args = parser.parse_args()

    # Initialize config and CREATE DIRECTORIES before anything else
    try:
        config = initialize_config(args)
    except Exception as e:
        logger.warning(f"Config initialization warning: {e}")
        global _config
        if _config is None:
            _config = DOCRConfig() if CONFIG_SYSTEM_AVAILABLE else None
        config = _config

    # ALWAYS setup directories before figure generation
    if config is not None:
        setup_directories(config)
    else:
        # Fallback: create default directories
        fig_dir = BASE_DIR / "results" / "figures"
        for subdir in ["figs_01_10", "figs_11_20", "figs_21_30", "figs_31_40",
                       "figs_41_50", "figs_51_58"]:
            (fig_dir / subdir).mkdir(parents=True, exist_ok=True)
        (BASE_DIR / "results" / "reports").mkdir(parents=True, exist_ok=True)

    if args.report:
        reports_dir = get_reports_dir()
        reports = sorted(reports_dir.glob("*.json"), reverse=True)
        if reports:
            with open(reports[0]) as f:
                report = json.load(f)
            print(f"\nLast figure report: {reports[0].name}")
            print(f"Timestamp: {report.get('timestamp', 'N/A')}")
            print(f"Completed: {report.get('completed', 0)}/{report.get('total_figures', 0)}")
            print(f"Failed: {report.get('failed', 0)}")
            print(f"Total time: {timedelta(seconds=int(report.get('total_elapsed_seconds', 0)))}")
            if report.get('failed_figures'):
                print(f"\nFailed figures:")
                for f in report['failed_figures']:
                    print(f"  Fig {f['fig']}: {f['error']}")
        else:
            print("No figure reports found.")
        return

    if args.list:
        print("\nAvailable figures (generated from cached core training data):")
        print("=" * 60)
        for fig_num in sorted(FIGURE_FUNCTIONS.keys()):
            func = FIGURE_FUNCTIONS[fig_num]
            doc = func.__doc__.split(chr(10))[0] if func.__doc__ else "No description"
            print(f"  Fig {fig_num:<3} — {doc}")
        print(f"\nTotal: {len(FIGURE_FUNCTIONS)} figures")
        cache_dir = get_cache_dir()
        print(f"\nCache directory: {cache_dir}")
        cache_index = cache_dir / "cache_index.json"
        if cache_index.exists():
            with open(cache_index) as f:
                index = json.load(f)
            print(f"Cached core runs: {len(index)}")
            for key, info in index.items():
                print(f"  {key}: {info.get('name', 'unknown')}")
        return

    if args.all:
        run_all_figures()
    elif args.fig:
        fig_progress = FigureProgress(total_figures=len(args.fig))
        with ProgressTracker(total=len(args.fig), desc="[Figures] Selected", unit="fig") as pbar:
            for fig_num in args.fig:
                if fig_num in FIGURE_FUNCTIONS:
                    try:
                        t0 = time.time()
                        FIGURE_FUNCTIONS[fig_num]()
                        duration = time.time() - t0
                        fig_progress.add_completed(fig_num, duration)
                        pbar.update(1)
                        pbar.set_postfix(fig=fig_num, time=f"{duration:.1f}s")
                    except Exception as e:
                        duration = time.time() - t0
                        fig_progress.add_failed(fig_num, str(e))
                        logger.error(f"Figure {fig_num} failed: {e}")
                        pbar.update(1)
                else:
                    logger.error(f"Figure {fig_num} not available")
                    pbar.update(1)
        fig_progress.print_summary()
    elif args.phase:
        if args.phase in PHASE_FIGURES:
            figs = PHASE_FIGURES[args.phase]
            fig_progress = FigureProgress(total_figures=len(figs))
            with ProgressTracker(total=len(figs), desc=f"[Figures] Phase {args.phase}", unit="fig") as pbar:
                for fig_num in figs:
                    if fig_num in FIGURE_FUNCTIONS:
                        try:
                            t0 = time.time()
                            FIGURE_FUNCTIONS[fig_num]()
                            duration = time.time() - t0
                            fig_progress.add_completed(fig_num, duration)
                            pbar.update(1)
                            pbar.set_postfix(fig=fig_num, time=f"{duration:.1f}s")
                        except Exception as e:
                            duration = time.time() - t0
                            fig_progress.add_failed(fig_num, str(e))
                            logger.error(f"Figure {fig_num} failed: {e}")
                            pbar.update(1)
            fig_progress.print_summary()
        else:
            logger.error(f"Phase {args.phase} not defined")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()