#!/usr/bin/env python3
"""
run_thesis_experiments.py — Master Experiment Orchestrator for DOCR-MARL Thesis
══════════════════════════════════════════════════════════════════════════════════
Refactored with comprehensive progress tracking, ETA estimation, intelligent
caching, and unified config file support via docl_config.py.

Architecture:
  • Phase 1: Manual diagrams (no code execution)
  • Phase 2: Core training — delegates to run_core_training.py (once, not per-fig)
  • Phases 3-8: Post-processing — direct Python calls (no subprocess overhead)
  • Phase 9: Final compilation & report generation

Configuration (precedence: CLI > config file > preset > env vars > defaults):
  • --config FILE       Load YAML/JSON config file
  • --preset NAME       Use built-in preset (8bus_full, 8bus_fast, etc.)
  • --system NAME       Override system (3bus, 8bus, 14bus, ...)
  • --fast              Enable fast mode (reduced iterations)
  • All CLI args from docl_config.py supported

Usage:
    python run_thesis_experiments.py --all                       # full suite
    python run_thesis_experiments.py --all --config my_config.yaml  # custom config
    python run_thesis_experiments.py --train-only                  # just training
    python run_thesis_experiments.py --figures-only                # just figures
    python run_thesis_experiments.py --phase 2                     # specific phase
    python run_thesis_experiments.py --fig 9 10 12                 # selected figures
    python run_thesis_experiments.py --fast                        # smoke test
    python run_thesis_experiments.py --clear-cache                 # reset cache
    python run_thesis_experiments.py --progress                    # show last report
    python run_thesis_experiments.py --list                        # list all figures
"""

import argparse
import os
import sys
import io
import time
import json
import pickle
import subprocess
import threading
import importlib.util
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Set, Tuple, Optional, Callable, Any, Union
from dataclasses import dataclass, field
from contextlib import contextmanager
import logging

# ──────────────────────────────────────────────────────────────────
# Import Configuration System
# ──────────────────────────────────────────────────────────────────
# Add parent directory to path for config import
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
# Windows Unicode Fix
# ──────────────────────────────────────────────────────────────────
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

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
OUTPUT_DIR: Path = BASE_DIR / "results"
FIGURES_DIR: Path = OUTPUT_DIR / "figures"
REPORT_DIR: Path = OUTPUT_DIR / "reports"
CACHE_DIR: Path = OUTPUT_DIR / "core_cache"
PROGRESS_DIR: Path = OUTPUT_DIR / "progress"
LOG_DIR: Path = OUTPUT_DIR / "logs"

FAST_MODE = False

# ──────────────────────────────────────────────────────────────────
# Directory Setup
# ──────────────────────────────────────────────────────────────────
def setup_directories(config: DOCRConfig):
    """Create all output directories based on configuration."""
    output = Path(config.experiment.output_dir)
    fig_dir = output / "figures"

    subdirs = ["figs_01_10", "figs_11_20", "figs_21_30", "figs_31_40",
               "figs_41_50", "figs_51_58", "logs", "checkpoints", "cache", "reports", "progress"]
    for subdir in subdirs:
        (output / subdir).mkdir(parents=True, exist_ok=True)

    return output, fig_dir

# ──────────────────────────────────────────────────────────────────
# Logging Setup
# ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

LOG_FILE = LOG_DIR / f"experiment_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
REPORT_FILE = REPORT_DIR / f"experiment_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

# File handler for persistent logging
file_handler = logging.FileHandler(LOG_FILE, encoding='utf-8')
file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s"))
logger.addHandler(file_handler)

# ──────────────────────────────────────────────────────────────────
# Color / Terminal Utilities
# ──────────────────────────────────────────────────────────────────
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    END = '\033[0m'


def colorize(text: str, color_code: str) -> str:
    """Apply color if terminal supports it."""
    if hasattr(sys.stdout, 'isatty') and sys.stdout.isatty():
        return f"{color_code}{text}{Colors.END}"
    return text


def log(msg: str, level: str = "info"):
    """Log to console and file with safe encoding."""
    safe_msg = msg.replace('→', '->').replace('←', '<-').replace('—', '-')
    if level == "error":
        logger.error(safe_msg)
    elif level == "warning":
        logger.warning(safe_msg)
    elif level == "debug":
        logger.debug(safe_msg)
    else:
        logger.info(safe_msg)


# ──────────────────────────────────────────────────────────────────
# Progress Tracking Infrastructure
# ──────────────────────────────────────────────────────────────────
@dataclass
class PhaseResult:
    """Result of a single phase execution."""
    phase_num: int
    phase_name: str
    status: str = "not_started"
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    figures_generated: List[int] = field(default_factory=list)
    figures_failed: List[Dict] = field(default_factory=list)
    details: Dict = field(default_factory=dict)

    @property
    def duration(self) -> Optional[float]:
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None

    @property
    def duration_str(self) -> str:
        d = self.duration
        if d is None:
            return "N/A"
        return str(timedelta(seconds=int(d)))


@dataclass
class FigureResult:
    """Result of a single figure generation."""
    fig_num: int
    phase: int
    status: str = "not_started"
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    error: Optional[str] = None

    @property
    def duration(self) -> Optional[float]:
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None


class ExperimentProgress:
    """
    Thread-safe progress tracker for the entire experiment suite.
    Tracks phases, figures, ETA, and produces summary reports.
    """

    def __init__(self, total_phases: int = 9, total_figures: int = 58):
        self.total_phases = total_phases
        self.total_figures = total_figures
        self.phases: Dict[int, PhaseResult] = {}
        self.figures: Dict[int, FigureResult] = {}
        self._lock = threading.Lock()
        self._start_time = time.time()
        self._phase_start_times: Dict[int, float] = {}

    def start_phase(self, phase_num: int, phase_name: str):
        """Mark a phase as started."""
        with self._lock:
            now = time.time()
            self.phases[phase_num] = PhaseResult(
                phase_num=phase_num,
                phase_name=phase_name,
                status="running",
                start_time=now
            )
            self._phase_start_times[phase_num] = now

        log("")
        log("=" * 70)
        log(colorize(f"PHASE {phase_num}: {phase_name}", Colors.BOLD + Colors.CYAN))
        log("=" * 70)

    def finish_phase(self, phase_num: int, success: bool = True, details: Dict = None):
        """Mark a phase as complete or failed."""
        with self._lock:
            if phase_num in self.phases:
                self.phases[phase_num].end_time = time.time()
                self.phases[phase_num].status = "complete" if success else "failed"
                if details:
                    self.phases[phase_num].details.update(details)

        status_icon = "✓" if success else "✗"
        status_color = Colors.GREEN if success else Colors.RED
        duration = self.phases[phase_num].duration
        dur_str = f" ({timedelta(seconds=int(duration))})" if duration else ""
        log(colorize(f"{status_icon} Phase {phase_num} {self.phases[phase_num].phase_name}: {self.phases[phase_num].status.upper()}{dur_str}", status_color))

    def skip_phase(self, phase_num: int, reason: str = ""):
        """Mark a phase as skipped."""
        with self._lock:
            self.phases[phase_num] = PhaseResult(
                phase_num=phase_num,
                phase_name=self.phases.get(phase_num, PhaseResult(phase_num, "?")).phase_name,
                status="skipped",
                details={"skip_reason": reason}
            )
        log(colorize(f"○ Phase {phase_num}: SKIPPED ({reason})", Colors.YELLOW))

    def start_figure(self, fig_num: int, phase: int):
        """Mark a figure as being generated."""
        with self._lock:
            self.figures[fig_num] = FigureResult(
                fig_num=fig_num,
                phase=phase,
                status="running",
                start_time=time.time()
            )

    def finish_figure(self, fig_num: int, success: bool = True, error: str = None):
        """Mark a figure as complete or failed."""
        with self._lock:
            if fig_num in self.figures:
                self.figures[fig_num].end_time = time.time()
                self.figures[fig_num].status = "complete" if success else "failed"
                self.figures[fig_num].error = error

                # Update phase tracking
                phase_num = self.figures[fig_num].phase
                if phase_num in self.phases:
                    if success:
                        self.phases[phase_num].figures_generated.append(fig_num)
                    else:
                        self.phases[phase_num].figures_failed.append({"fig": fig_num, "error": error})

        if success:
            dur = self.figures[fig_num].duration
            dur_str = f" in {dur:.2f}s" if dur else ""
            log(colorize(f"  [Fig {fig_num:02d}] ✓ Generated{dur_str}", Colors.GREEN))
        else:
            log(colorize(f"  [Fig {fig_num:02d}] ✗ Failed: {error}", Colors.RED), level="error")

    def get_eta(self) -> str:
        """Estimate remaining time based on completed work."""
        with self._lock:
            completed_phases = sum(1 for p in self.phases.values() if p.status == "complete")
            completed_figs = sum(1 for f in self.figures.values() if f.status == "complete")

            if completed_phases == 0 and completed_figs == 0:
                return "calculating..."

            elapsed = time.time() - self._start_time

            # Weight phases and figures
            total_work = self.total_phases * 10 + self.total_figures
            completed_work = completed_phases * 10 + completed_figs

            if completed_work == 0:
                return "calculating..."

            rate = completed_work / elapsed
            remaining_work = total_work - completed_work
            eta_seconds = remaining_work / rate

            return str(timedelta(seconds=int(eta_seconds)))

    def print_summary(self):
        """Print comprehensive experiment summary."""
        total_elapsed = time.time() - self._start_time

        log("")
        log("=" * 70)
        log(colorize("EXPERIMENT SUITE SUMMARY", Colors.BOLD + Colors.HEADER))
        log("=" * 70)
        log(f"Total elapsed: {timedelta(seconds=int(total_elapsed))}")
        log(f"Mode: {'FAST' if FAST_MODE else 'FULL'}")
        log(f"ETA: {self.get_eta()}")
        log("-" * 70)

        # Phase summary
        for phase_num in sorted(self.phases.keys()):
            p = self.phases[phase_num]
            icon = {"complete": "✓", "failed": "✗", "running": "▶",
                    "not_started": "○", "skipped": "⊘"}.get(p.status, "?")
            color = {"complete": Colors.GREEN, "failed": Colors.RED, "running": Colors.CYAN,
                     "not_started": Colors.DIM, "skipped": Colors.YELLOW}.get(p.status, Colors.END)
            fig_count = len(p.figures_generated)
            fail_count = len(p.figures_failed)
            fail_str = f" [{fail_count} failed]" if fail_count > 0 else ""
            log(colorize(f"  {icon} Phase {phase_num}: {p.phase_name:<30} {p.status.upper():<10} {p.duration_str}{fail_str}", color))

        log("-" * 70)

        # Figure summary
        total_completed = sum(1 for f in self.figures.values() if f.status == "complete")
        total_failed = sum(1 for f in self.figures.values() if f.status == "failed")
        log(f"Figures: {total_completed} generated, {total_failed} failed, {self.total_figures - total_completed - total_failed} pending")

        if total_failed > 0:
            log("Failed figures:")
            for fig_num, fig_result in self.figures.items():
                if fig_result.status == "failed":
                    log(f"  Fig {fig_num:02d}: {fig_result.error}", level="error")

        log("=" * 70)

    def save_report(self, path: Optional[Path] = None):
        """Save comprehensive JSON report."""
        if path is None:
            path = REPORT_FILE

        report = {
            "timestamp": datetime.now().isoformat(),
            "fast_mode": FAST_MODE,
            "total_elapsed_seconds": time.time() - self._start_time,
            "eta": self.get_eta(),
            "phases": {
                str(p.phase_num): {
                    "name": p.phase_name,
                    "status": p.status,
                    "duration_seconds": p.duration,
                    "figures_generated": p.figures_generated,
                    "figures_failed": p.figures_failed,
                    "details": p.details
                }
                for p in self.phases.values()
            },
            "figures": {
                str(f.fig_num): {
                    "phase": f.phase,
                    "status": f.status,
                    "duration_seconds": f.duration,
                    "error": f.error
                }
                for f in self.figures.values()
            },
            "summary": {
                "total_phases": self.total_phases,
                "completed_phases": sum(1 for p in self.phases.values() if p.status == "complete"),
                "failed_phases": sum(1 for p in self.phases.values() if p.status == "failed"),
                "total_figures": self.total_figures,
                "completed_figures": sum(1 for f in self.figures.values() if f.status == "complete"),
                "failed_figures": sum(1 for f in self.figures.values() if f.status == "failed"),
            }
        }

        with open(path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        log(f"Report saved to: {path}")
        return path


# ──────────────────────────────────────────────────────────────────
# Progress Bar (Console UI)
# ──────────────────────────────────────────────────────────────────
class ProgressBar:
    """Simple thread-safe progress bar for console output."""

    def __init__(self, total: int, desc: str = "", width: int = 40):
        self.total = max(total, 1)
        self.desc = desc
        self.width = width
        self.current = 0
        self._lock = threading.Lock()
        self._start = time.time()
        self._enabled = sys.stdout.isatty()

    def update(self, n: int = 1, postfix: Dict = None):
        with self._lock:
            self.current = min(self.current + n, self.total)
            if self._enabled:
                self._draw(postfix)

    def _draw(self, postfix: Dict = None):
        frac = self.current / self.total
        filled = int(self.width * frac)
        bar = "█" * filled + "░" * (self.width - filled)
        elapsed = timedelta(seconds=int(time.time() - self._start))

        postfix_str = ""
        if postfix:
            postfix_str = " | " + ", ".join(f"{k}={v}" for k, v in postfix.items())

        line = f"\r{self.desc} [{bar}] {self.current}/{self.total} ({frac*100:.1f}%) | Elapsed: {elapsed}{postfix_str}"
        sys.stdout.write(line)
        sys.stdout.flush()

    def close(self, message: str = "DONE"):
        with self._lock:
            if self._enabled:
                elapsed = timedelta(seconds=int(time.time() - self._start))
                sys.stdout.write(f"\r{self.desc} [{'█' * self.width}] {self.total}/{self.total} (100.0%) | Total: {elapsed} — {message}\n")
                sys.stdout.flush()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# ──────────────────────────────────────────────────────────────────
# Context Managers for Timing
# ──────────────────────────────────────────────────────────────────
@contextmanager
def timed_phase(phase_num: int, phase_name: str, progress: ExperimentProgress):
    """Context manager for timing a phase and updating progress."""
    progress.start_phase(phase_num, phase_name)
    try:
        yield
        progress.finish_phase(phase_num, success=True)
    except Exception as e:
        progress.finish_phase(phase_num, success=False, details={"error": str(e)})
        log(colorize(f"Phase {phase_num} failed: {e}", Colors.RED), level="error")
        raise


@contextmanager
def timed_figure(fig_num: int, phase: int, progress: ExperimentProgress):
    """Context manager for timing a single figure generation."""
    progress.start_figure(fig_num, phase)
    t0 = time.time()
    try:
        yield
        progress.finish_figure(fig_num, success=True)
    except Exception as e:
        progress.finish_figure(fig_num, success=False, error=str(e))
        # Don't re-raise — allow other figures to continue
        log(colorize(f"Fig {fig_num} error: {e}", Colors.RED), level="error")


# ──────────────────────────────────────────────────────────────────
# Subprocess Runner (with live output)
# ──────────────────────────────────────────────────────────────────
def run_subprocess(cmd: str, description: str, check: bool = True,
                   env_vars: Dict = None, timeout: Optional[int] = None) -> int:
    """
    Run a subprocess command with live output streaming and progress tracking.
    Returns exit code. Raises on failure if check=True.
    """
    sep = "─" * 70
    log("")
    log(colorize(sep, Colors.CYAN))
    log(colorize(f"▶ {description}", Colors.BOLD + Colors.BLUE))
    log(f"  CMD: {cmd}")
    t0 = time.time()

    env = os.environ.copy()
    if env_vars:
        env.update(env_vars)
    if FAST_MODE:
        env["DOCR_FAST"] = "1"

    try:
        process = subprocess.Popen(
            cmd, shell=True, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1
        )

        output_lines = []
        for line in process.stdout:
            line = line.rstrip()
            output_lines.append(line)
            sys.stdout.write(f"  {colorize('│', Colors.BLUE)} {line}\n")
            sys.stdout.flush()

        process.wait(timeout=timeout)
        elapsed = time.time() - t0

        if process.returncode == 0:
            log(colorize(f"  ✓ SUCCESS in {elapsed:.1f}s", Colors.GREEN))
        else:
            log(colorize(f"  ✗ FAILED after {elapsed:.1f}s (exit {process.returncode})", Colors.RED), level="error")
            if check:
                output_text = '\n'.join(output_lines)
                if any(x in output_text for x in ["Placeholder", "No such file", "cannot find",
                                                   "not found", "ModuleNotFoundError", "ImportError"]):
                    log(colorize("  ⚠ Continuing despite error (placeholder/missing module)", Colors.YELLOW), level="warning")
                    return process.returncode
                raise subprocess.CalledProcessError(process.returncode, cmd)

        return process.returncode

    except subprocess.TimeoutExpired:
        process.kill()
        log(colorize(f"  ✗ TIMEOUT after {timeout}s", Colors.RED), level="error")
        raise


# ──────────────────────────────────────────────────────────────────
# Direct Post-Processing (avoid subprocess overhead)
# ──────────────────────────────────────────────────────────────────
def load_post_processing_module() -> Optional[Any]:
    """
    Dynamically import run_post_processing.py to call figure functions directly.
    Avoids subprocess spawning overhead for figure generation.
    """
    pp_path = BASE_DIR / "run_post_processing.py"
    if not pp_path.exists():
        log(colorize(f"run_post_processing.py not found at {pp_path}", Colors.RED), level="warning")
        return None

    spec = importlib.util.spec_from_file_location("post_processing", pp_path)
    if spec is None or spec.loader is None:
        return None

    module = importlib.util.module_from_spec(spec)
    sys.modules["post_processing"] = module
    spec.loader.exec_module(module)
    return module


def generate_figure_direct(fig_num: int, pp_module: Any, progress: ExperimentProgress,
                           phase: int) -> bool:
    """
    Generate a single figure by calling the post-processing module directly.
    Returns True on success, False on failure.
    """
    if pp_module is None:
        return False

    fig_map = getattr(pp_module, 'FIGURE_FUNCTIONS', {})
    if fig_num not in fig_map:
        log(colorize(f"Fig {fig_num} not found in post-processing module", Colors.YELLOW), level="warning")
        return False

    try:
        with timed_figure(fig_num, phase, progress):
            fig_map[fig_num]()
        return True
    except Exception as e:
        # Error already logged by timed_figure context manager
        return False


def generate_figures_batch(fig_nums: List[int], phase: int, progress: ExperimentProgress,
                         use_subprocess: bool = False) -> Tuple[int, int]:
    """
    Generate multiple figures with progress tracking.
    Returns (success_count, fail_count).
    """
    if not fig_nums:
        return 0, 0

    pp_module = None if use_subprocess else load_post_processing_module()
    success_count = 0
    fail_count = 0

    with ProgressBar(total=len(fig_nums), desc=f"[Phase {phase}] Figures") as pbar:
        for fig_num in fig_nums:
            if use_subprocess or pp_module is None:
                # Fallback to subprocess
                try:
                    run_subprocess(
                        f"python run_post_processing.py --fig {fig_num}",
                        f"Fig {fig_num}: Post-processing",
                        check=False
                    )
                    progress.finish_figure(fig_num, success=True)
                    success_count += 1
                except Exception as e:
                    progress.finish_figure(fig_num, success=False, error=str(e))
                    fail_count += 1
            else:
                # Direct call (much faster)
                if generate_figure_direct(fig_num, pp_module, progress, phase):
                    success_count += 1
                else:
                    fail_count += 1

            pbar.update(1, postfix={"fig": fig_num, "ok": success_count, "fail": fail_count})

    pbar.close(f"{success_count}/{len(fig_nums)} generated")
    return success_count, fail_count


# ──────────────────────────────────────────────────────────────────
# Cache Utilities
# ──────────────────────────────────────────────────────────────────
def check_cache_status() -> Dict:
    """Check status of core training cache."""
    cache_index = CACHE_DIR / "cache_index.json"
    status = {"exists": False, "entries": 0, "cores_present": set()}

    if cache_index.exists():
        status["exists"] = True
        try:
            with open(cache_index) as f:
                index = json.load(f)
            status["entries"] = len(index)
            for key, info in index.items():
                name = info.get("name", "")
                if name.startswith("core_"):
                    core_letter = name[5].upper() if len(name) > 5 else "?"
                    status["cores_present"].add(core_letter)
        except Exception as e:
            log(f"Error reading cache index: {e}", level="warning")

    return status


def clear_all_caches():
    """Clear all cached results and progress reports."""
    log(colorize("Clearing all caches...", Colors.YELLOW))

    # Clear core cache
    try:
        run_subprocess("python run_core_training.py --clear-cache", "Clear core training cache", check=False)
    except Exception:
        pass

    # Clear figure outputs
    for subdir in ["figs_01_10", "figs_11_20", "figs_21_30", "figs_31_40", "figs_41_50", "figs_51_58"]:
        d = FIGURES_DIR / subdir
        if d.exists():
            for f in d.glob("*.pdf"):
                f.unlink()

    # Clear reports
    for f in REPORT_DIR.glob("*.json"):
        f.unlink()
    for f in PROGRESS_DIR.glob("*.json"):
        f.unlink()

    log(colorize("✓ All caches cleared", Colors.GREEN))


# ──────────────────────────────────────────────────────────────────
# Manual Figure Script Registry
# ──────────────────────────────────────────────────────────────────

# Mapping from figure number to standalone script path.
MANUAL_FIGURE_SCRIPTS: Dict[Union[int, str], Path] = {
    1:   BASE_DIR / "phase1_scripts" / "exp_fig01.py",
    2:   BASE_DIR / "phase1_scripts" / "exp_fig02.py",
    3:   BASE_DIR / "phase1_scripts" / "exp_fig03.py",
    4:   BASE_DIR / "phase1_scripts" / "exp_fig04.py",
    17:  BASE_DIR / "phase1_scripts" / "exp_fig17.py",
    19:  BASE_DIR / "phase1_scripts" / "exp_fig19.py",
    26:  BASE_DIR / "phase1_scripts" / "exp_fig26.py",
    27:  BASE_DIR / "phase1_scripts" / "exp_fig27.py",
    28:  BASE_DIR / "phase1_scripts" / "exp_fig28.py",
    29:  BASE_DIR / "phase1_scripts" / "exp_fig29.py",
    30:  BASE_DIR / "phase1_scripts" / "exp_fig30.py",
    34:  BASE_DIR / "phase1_scripts" / "exp_fig34.py",
    35:  BASE_DIR / "phase1_scripts" / "exp_fig35.py",
    51:  BASE_DIR / "phase1_scripts" / "exp_fig51.py",
    52:  BASE_DIR / "phase1_scripts" / "exp_fig52.py",
}

TRULY_MANUAL_FIGURES: Set[Union[int, str]] = {"26a", 36}


def generate_manual_figure(fig_num: Union[int, str], output_dir: Path,
                           progress: ExperimentProgress) -> bool:
    """Generate a single manual/conceptual figure from standalone script."""
    script_path = MANUAL_FIGURE_SCRIPTS.get(fig_num)

    if script_path is None:
        if fig_num in TRULY_MANUAL_FIGURES:
            log(f"  [Fig {fig_num}] ○ Truly manual — no script available (create in LaTeX/draw.io)",
                level="warning")
        else:
            log(f"  [Fig {fig_num}] ✗ No script mapped for this figure", level="warning")
        return False

    if not script_path.exists():
        log(f"  [Fig {fig_num}] ✗ Script not found: {script_path}", level="warning")
        return False

    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        with timed_figure(fig_num if isinstance(fig_num, int) else 0, 1, progress):
            module_name = f"manual_fig_{fig_num}"
            spec = importlib.util.spec_from_file_location(module_name, script_path)
            if spec is None or spec.loader is None:
                raise ImportError(f"Cannot load spec for {script_path}")

            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)

            if not hasattr(module, 'generate_figure'):
                raise AttributeError(f"{script_path} does not define generate_figure()")

            module.generate_figure(str(output_dir))

        return True
    except Exception as e:
        return False


def run_phase1(progress: ExperimentProgress):
    """Phase 1: Manual diagrams & Architecture (Week 1)."""
    with timed_phase(1, "Manual Diagrams & Architecture", progress):
        phase1_output = FIGURES_DIR / "figs_01_10"

        all_phase1_figs = sorted(MANUAL_FIGURE_SCRIPTS.keys()) + sorted(TRULY_MANUAL_FIGURES,
                                                                          key=lambda x: (isinstance(x, str), x))

        log(f"Phase 1: {len(MANUAL_FIGURE_SCRIPTS)} auto-generatable + {len(TRULY_MANUAL_FIGURES)} truly manual figures")
        log(f"Output directory: {phase1_output}")

        success_count = 0
        fail_count = 0
        manual_count = 0

        with ProgressBar(total=len(all_phase1_figs), desc="[Phase 1] Manual Figures") as pbar:
            for fig_num in all_phase1_figs:
                if fig_num in TRULY_MANUAL_FIGURES:
                    log(f"  [Fig {fig_num}] ○ Truly manual — create in docs/diagrams/ or LaTeX")
                    manual_count += 1
                    pbar.update(1)
                    continue

                if generate_manual_figure(fig_num, phase1_output, progress):
                    success_count += 1
                else:
                    fail_count += 1
                pbar.update(1)

        pbar.close(f"{success_count} generated, {fail_count} failed, {manual_count} manual")
        log(f"Phase 1 complete: {success_count} generated, {fail_count} failed, {manual_count} manual")


# ──────────────────────────────────────────────────────────────────
# Phase Implementations
# ──────────────────────────────────────────────────────────────────

FIGURE_CORE_MAP = {
    8: "A", 9: "A", 10: "A", 12: "A", 21: "A", 37: "A", 38: "A", 41: "A",
    48: "A", 49: "A", 50: "A", 55: "A", 56: "A", 57: "A",
    5: "B", 6: "B", 7: "B", 16: "B", 29: "B", 39: "B", 42: "B",
    11: "C", 31: "C", 32: "C", 33: "C",
    39: "D", 40: "E", 15: "F",
    13: "G", 14: "G", 20: "G", 22: "G", 24: "G", 54: "G",
    23: "H", 25: "H", 18: "I", 53: "I",
    43: "J", 44: "J", 45: "J", 46: "J", 47: "J",
    58: "K",
}

PHASE_FIGURES = {
    1: [],
    2: [],
    # Figs 5, 6, 7, 8, 12 added: these have real generation code in
    # run_post_processing.py (FIGURE_FUNCTIONS) and are fed by Core
    # A/B training data (run_core_training.py explicitly lists
    # "Core B feeds Figs 5,6,7,16,29,39"), but were absent from every
    # dispatch table in this file (PHASE_FIGURES, MANUAL_FIGURE_SCRIPTS,
    # TRULY_MANUAL_FIGURES) — meaning a full pipeline run computed the
    # underlying training data but never actually rendered these plots.
    3: [5, 6, 7, 8, 9, 10, 12, 16, 21, 37, 38, 41],
    4: [13, 14, 20, 22, 24, 54],
    5: [15, 23, 25],
    6: [18, 53],
    # Fig 58 added: Core K output (seasonal generalisation), same
    # supplementary-analysis grouping as 43-47.
    7: [43, 44, 45, 46, 47, 48, 49, 50, 58],
    # Figs 39, 40, 42 added to Phase 8 (the last phase that actually
    # calls generate_figures_batch): ablation/hparam/warm-start figures
    # (Core B/D/E), previously unreachable from any phase. Fig 39 in
    # particular is the EWC ablation figure — the one figure that
    # actually demonstrates the EWC continual-learning fix made
    # elsewhere in this codebase, so its absence from the pipeline
    # would have silently meant that fix was never visualised or
    # checked end-to-end as part of a real run.
    # NOTE: these were NOT placed under key 9 — run_phase9 ("Final
    # Compilation") never calls generate_figures_batch at all, it only
    # counts already-generated PDFs and writes a summary report, so any
    # figure numbers listed there would silently never be produced.
    8: [11, 31, 32, 33, 55, 56, 57, 39, 40, 42],
    9: [],
}

MANUAL_FIGURES = {1, 2, 3, 4, 17, 19, 26, "26a", 27, 28, 29, 30, 34, 35, 36, 51, 52}

def run_phase2(progress: ExperimentProgress):
    """Phase 2: Core Training Runs with real evaluations."""
    with timed_phase(2, "Core Training Runs (A-K with Real Evals)", progress):
        config = get_config()

        log("Running unified core training via run_core_training.py...")
        log(f"  System: {config.system.name.upper()} | PV: {config.system.pv_penetration:.0%} | Fault: {config.system.fault_type}")
        log(f"  Runs: {config.training.n_runs} | Episodes: {config.training.n_episodes}")
        log(f"  Features: GNN={'✓' if config.training.use_comm else '✗'} | Safety={'✓' if config.training.use_safety else '✗'} | EWC={'✓' if config.training.use_ewc else '✗'}")

        # Build command with config parameters
        fast_flag = " --fast" if config.experiment.fast_mode else ""
        system_flag = f" --system {config.system.name}"
        pv_flag = f" --pv {config.system.pv_penetration}"
        runs_flag = f" --n-runs {config.training.n_runs}"
        episodes_flag = f" --episodes {config.training.n_episodes}"

        cmd = f"python run_core_training.py --all{fast_flag}{system_flag}{pv_flag}{runs_flag}{episodes_flag}"

        run_subprocess(
            cmd,
            "Phase 2: Unified Core Training (all 11 cores A-K)",
            check=False,
            timeout=7200 if config.experiment.fast_mode else 86400
        )

        # ── Objective 4: Multi-system, multi-method baseline comparison ──────
        # benchmark_suite.py is the ONLY code that runs the full Obj4
        # comparison required by the thesis: GA-SQP, HWOA, HHO, JAYA, EEO
        # vs MARL variants across IEEE 8/30/39-bus at all PV levels.
        # It was previously an orphan standalone script never called from
        # this pipeline, meaning the named metaheuristic baselines from
        # Objective 4 were never part of any automated experiment run.
        log("\n[Phase 2] Running Objective 4 benchmark comparison (GA-SQP/HWOA/HHO/JAYA/EEO vs MARL)...")
        try:
            from evaluation.benchmark_suite import BenchmarkSuite, BenchmarkConfig

            obj4_systems = ["8bus", "14bus", "39bus"]
            obj4_pv_levels = [0.0, 0.2, 0.4, 0.6]
            obj4_methods = [
                "GA-SQP", "HWOA", "HHO", "JAYA", "EEO",
                "MAPPO-Safe-EWC", "MAPPO-Safe", "MAPPO",
            ]

            n_runs = 3 if config.experiment.fast_mode else config.training.n_runs
            marl_steps = 5000 if config.experiment.fast_mode else 30000
            meta_iter = 100 if config.experiment.fast_mode else 500

            all_pv_results = {}
            for pv_level in obj4_pv_levels:
                log(f"  PV level: {int(pv_level*100)}%")
                bc = BenchmarkConfig(
                    systems=obj4_systems,
                    methods=obj4_methods,
                    n_runs=n_runs,
                    pv_level=pv_level,
                    fault_type=config.system.fault_type,
                    marl_steps=marl_steps,
                    max_iter=meta_iter,
                    pop_size=20 if config.experiment.fast_mode else 30,
                    seed_base=config.experiment.seed,
                    verbose=False,
                )
                suite = BenchmarkSuite(config=bc)
                suite.run_all()
                suite.print_comparison_table()
                all_pv_results[pv_level] = suite.results

                # Save results per PV level
                results_path = (
                    Path(config.experiment.output_dir)
                    / f"benchmark_pv{int(pv_level*100):03d}.pkl"
                )
                try:
                    import pickle
                    with open(results_path, "wb") as f:
                        pickle.dump(suite.results, f)
                    log(f"    Saved: {results_path}")
                except Exception as save_err:
                    log(f"    Warning: could not save results: {save_err}")

                # Statistical significance tests across methods, for this PV level
                try:
                    from evaluation.stat_analysis import run_benchmark_analysis

                    # BenchmarkSuite.results is {system: [run_dict, ...]}
                    # where each run_dict is {method: result_dict}.
                    # run_benchmark_analysis expects
                    # {system: {method: [value, value, ...]}} — transform
                    # before calling, rather than guessing a nonexistent
                    # compare_methods() API that was never real.
                    transformed = {}
                    for sys_name, run_list in suite.results.items():
                        transformed[sys_name] = {}
                        for run_dict in run_list:
                            for method, result in run_dict.items():
                                of_val = result.get("OF") if isinstance(result, dict) else result
                                if of_val is None:
                                    continue
                                transformed[sys_name].setdefault(method, []).append(of_val)

                    stat_report = run_benchmark_analysis(transformed, metric="OF", verbose=False)
                    stat_path = (
                        Path(config.experiment.output_dir)
                        / f"stat_comparison_pv{int(pv_level*100):03d}.txt"
                    )
                    with open(stat_path, "w") as f:
                        f.write(str(stat_report))
                    log(f"    Statistical report saved: {stat_path}")
                except Exception as stat_err:
                    log(f"    Warning: statistical tests skipped for PV{int(pv_level*100)}%: {stat_err}")

            log("[Phase 2] Objective 4 benchmark comparison complete.")

        except ImportError as e:
            log(f"  Warning: benchmark_suite not available ({e}). Objective 4 comparison skipped.")
        except Exception as e:
            log(f"  Warning: Objective 4 benchmark failed ({e}). Core training results still valid.")


def run_phase3(progress: ExperimentProgress):
    """Phase 3: Training Dynamics (Week 4)."""
    with timed_phase(3, "Training Dynamics", progress):
        figs = PHASE_FIGURES[3]
        log(f"Generating {len(figs)} figures from Core A/B cached data: {figs}")
        success, fail = generate_figures_batch(figs, 3, progress)
        log(f"Phase 3 complete: {success} generated, {fail} failed")


def run_phase4(progress: ExperimentProgress):
    """Phase 4: Robustness & Resilience (Week 5)."""
    with timed_phase(4, "Robustness & Resilience", progress):
        figs = PHASE_FIGURES[4]
        log(f"Generating {len(figs)} figures from Core G cached data: {figs}")
        success, fail = generate_figures_batch(figs, 4, progress)
        log(f"Phase 4 complete: {success} generated, {fail} failed")


def run_phase5(progress: ExperimentProgress):
    """Phase 5: Scalability & Transfer Learning (Week 6)."""
    with timed_phase(5, "Scalability & Transfer Learning", progress):
        figs = PHASE_FIGURES[5]
        log(f"Generating {len(figs)} figures from Core F/H cached data: {figs}")
        success, fail = generate_figures_batch(figs, 5, progress)
        log(f"Phase 5 complete: {success} generated, {fail} failed")


def run_phase6(progress: ExperimentProgress):
    """Phase 6: Deployment & Hardware (Week 7)."""
    with timed_phase(6, "Deployment & Hardware", progress):
        figs = PHASE_FIGURES[6]
        log(f"Generating {len(figs)} figures from Core I cached data: {figs}")
        success, fail = generate_figures_batch(figs, 6, progress)
        log(f"Phase 6 complete: {success} generated, {fail} failed")


def run_phase7(progress: ExperimentProgress):
    """Phase 7: Protection Engineering & Interpretability (Week 8)."""
    with timed_phase(7, "Protection Engineering & Interpretability", progress):
        figs = PHASE_FIGURES[7]
        log(f"Generating {len(figs)} figures from Core J/A cached data: {figs}")
        success, fail = generate_figures_batch(figs, 7, progress)
        log(f"Phase 7 complete: {success} generated, {fail} failed")


def run_phase8(progress: ExperimentProgress):
    """Phase 8: Statistical Aggregation (Week 9)."""
    with timed_phase(8, "Statistical Aggregation", progress):
        figs = PHASE_FIGURES[8]
        log(f"Generating {len(figs)} figures from Core C cached data: {figs}")
        success, fail = generate_figures_batch(figs, 8, progress)
        log(f"Phase 8 complete: {success} generated, {fail} failed")


def run_phase9(progress: ExperimentProgress):
    """Phase 9: Final Compilation (Week 10)."""
    with timed_phase(9, "Final Compilation", progress):
        config = get_config()
        fig_count = sum(1 for _ in FIGURES_DIR.rglob("*.pdf")) if FIGURES_DIR.exists() else 0
        cache_status = check_cache_status()

        report = {
            "timestamp": datetime.now().isoformat(),
            "fast_mode": config.experiment.fast_mode,
            "output_dir": str(OUTPUT_DIR),
            "figures_expected": 58,
            "figures_generated": fig_count,
            "figures_manual": len(MANUAL_FIGURES),
            "training_strategy": "11 core runs (A-K) + post-processing",
            "estimated_gpu_days": "8-10 (full)" if not config.experiment.fast_mode else "~1-2 (fast)",
            "cache_status": {
                "exists": cache_status["exists"],
                "entries": cache_status["entries"],
                "cores_present": sorted(cache_status["cores_present"])
            },
            "phases": list(range(1, 10)),
            "core_runs": ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K"],
            "config": {
                "system": config.system.name,
                "pv_penetration": config.system.pv_penetration,
                "fault_type": config.system.fault_type,
                "n_runs": config.training.n_runs,
                "n_episodes": config.training.n_episodes,
                "use_comm": config.training.use_comm,
                "use_safety": config.training.use_safety,
                "use_ewc": config.training.use_ewc,
            }
        }

        report_path = REPORT_DIR / "compilation_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        log(f"  -> Compilation report saved to {report_path}")
        log(f"  -> Figures generated: {fig_count}/58 ({fig_count - len(MANUAL_FIGURES)} auto + {len(MANUAL_FIGURES)} manual)")
        log(f"  -> Cache entries: {cache_status['entries']}")
        log(f"  -> Cached cores: {sorted(cache_status['cores_present'])}")


# ──────────────────────────────────────────────────────────────────
# Master Orchestrator
# ──────────────────────────────────────────────────────────────────
def run_all(progress: Optional[ExperimentProgress] = None) -> ExperimentProgress:
    """Run complete experiment suite with full progress tracking."""
    if progress is None:
        progress = ExperimentProgress(total_phases=9, total_figures=58)

    config = get_config()

    log("")
    log("=" * 70)
    log(colorize("DOCR-MARL THESIS: COMPLETE EXPERIMENT SUITE", Colors.BOLD + Colors.HEADER))
    log(f"Strategy: 11 core training runs + unified post-processing")
    log(f"System: {config.system.name.upper()} | PV: {config.system.pv_penetration:.0%} | Fault: {config.system.fault_type}")
    log(f"Runs: {config.training.n_runs} | Episodes: {config.training.n_episodes}")
    log(f"Estimated: ~8-10 GPU-days (full) / ~1-2 hours (fast)")
    log("=" * 70)

    t0 = time.time()

    try:
        run_phase1(progress)
        run_phase2(progress)
        run_phase3(progress)
        run_phase4(progress)
        run_phase5(progress)
        run_phase6(progress)
        run_phase7(progress)
        run_phase8(progress)
        run_phase9(progress)
    except KeyboardInterrupt:
        log(colorize("\n\nInterrupted by user", Colors.YELLOW))
        progress.print_summary()
        progress.save_report()
        raise
    except Exception as e:
        log(colorize(f"\n\nSuite failed: {e}", Colors.RED), level="error")
        progress.print_summary()
        progress.save_report()
        raise

    progress.print_summary()
    report_path = progress.save_report()

    log("")
    log("=" * 70)
    log(colorize("ALL PHASES COMPLETE", Colors.BOLD + Colors.GREEN))
    log(f"Total time: {timedelta(seconds=int(time.time() - t0))}")
    log(f"Report: {report_path}")
    log("=" * 70)

    return progress


def run_train_only(progress: Optional[ExperimentProgress] = None) -> ExperimentProgress:
    """Run only core training (Phase 2)."""
    if progress is None:
        progress = ExperimentProgress(total_phases=1, total_figures=0)
    run_phase2(progress)
    progress.print_summary()
    progress.save_report()
    return progress


def run_figures_only(progress: Optional[ExperimentProgress] = None) -> ExperimentProgress:
    """Generate all figures from existing cache (no training)."""
    if progress is None:
        progress = ExperimentProgress(total_phases=8, total_figures=58)

    log("")
    log("=" * 70)
    log(colorize("FIGURES ONLY MODE (no training)", Colors.BOLD + Colors.YELLOW))
    log("=" * 70)

    cache_status = check_cache_status()
    if not cache_status["exists"]:
        log(colorize("WARNING: No cache found. Run training first or use --all.", Colors.RED), level="warning")
        return progress

    log(f"Cache entries: {cache_status['entries']}")
    log(f"Cached cores: {sorted(cache_status['cores_present'])}")

    for phase_num in [3, 4, 5, 6, 7, 8]:
        phase_map = {
            3: run_phase3, 4: run_phase4, 5: run_phase5,
            6: run_phase6, 7: run_phase7, 8: run_phase8
        }
        phase_map[phase_num](progress)

    run_phase9(progress)
    progress.print_summary()
    progress.save_report()
    return progress


def run_single_phase(phase_num: int, progress: Optional[ExperimentProgress] = None) -> ExperimentProgress:
    """Run a specific phase."""
    if progress is None:
        progress = ExperimentProgress(total_phases=1, total_figures=len(PHASE_FIGURES.get(phase_num, [])))

    phase_map = {
        1: run_phase1, 2: run_phase2, 3: run_phase3, 4: run_phase4,
        5: run_phase5, 6: run_phase6, 7: run_phase7, 8: run_phase8, 9: run_phase9
    }

    if phase_num not in phase_map:
        log(colorize(f"Phase {phase_num} not defined", Colors.RED), level="error")
        return progress

    phase_map[phase_num](progress)
    progress.print_summary()
    progress.save_report()
    return progress


def run_selected_figures(fig_nums: List[int], progress: Optional[ExperimentProgress] = None) -> ExperimentProgress:
    """Generate selected figures with progress tracking."""
    if progress is None:
        progress = ExperimentProgress(total_phases=1, total_figures=len(fig_nums))

    log("")
    log("=" * 70)
    log(colorize(f"SELECTED FIGURES: {fig_nums}", Colors.BOLD + Colors.CYAN))
    log("=" * 70)

    figs_by_phase: Dict[int, List[int]] = {}
    for fig_num in fig_nums:
        phase = 0
        for p, figs in PHASE_FIGURES.items():
            if fig_num in figs:
                phase = p
                break
        if phase not in figs_by_phase:
            figs_by_phase[phase] = []
        figs_by_phase[phase].append(fig_num)

    for phase, figs in sorted(figs_by_phase.items()):
        log(f"\nPhase {phase} figures: {figs}")
        success, fail = generate_figures_batch(figs, phase, progress)

    progress.print_summary()
    progress.save_report()
    return progress


def show_progress_report():
    """Display the last experiment progress report."""
    reports = sorted(REPORT_DIR.glob("experiment_report_*.json"), reverse=True)
    if not reports:
        reports = sorted(PROGRESS_DIR.glob("*.json"), reverse=True)

    if reports:
        with open(reports[0]) as f:
            report = json.load(f)
        print(f"\nLast experiment report: {reports[0].name}")
        print(f"Timestamp: {report.get('timestamp', 'N/A')}")
        print(f"Mode: {'FAST' if report.get('fast_mode') else 'FULL'}")
        print(f"Total elapsed: {timedelta(seconds=int(report.get('total_elapsed_seconds', 0)))}")
        print(f"ETA: {report.get('eta', 'N/A')}")
        print(f"\nPhases:")
        for phase_num, info in sorted(report.get('phases', {}).items(), key=lambda x: int(x[0])):
            status = info.get('status', 'unknown')
            dur = info.get('duration_seconds', 0)
            dur_str = f" [{timedelta(seconds=int(dur))}]" if dur else ""
            figs = len(info.get('figures_generated', []))
            fails = len(info.get('figures_failed', []))
            fig_str = f" ({figs} figs" + (f", {fails} failed)" if fails > 0 else ")") if figs > 0 else ""
            icon = {"complete": "✓", "failed": "✗", "running": "▶", "not_started": "○", "skipped": "⊘"}.get(status, "?")
            print(f"  {icon} Phase {phase_num}: {info.get('name', '?'):<30} {status.upper():<10}{dur_str}{fig_str}")

        summary = report.get('summary', {})
        print(f"\nSummary: {summary.get('completed_figures', 0)}/{summary.get('total_figures', 0)} figures, " +
              f"{summary.get('completed_phases', 0)}/{summary.get('total_phases', 0)} phases")
    else:
        print("No experiment reports found. Run experiments first.")


def list_all_figures():
    """List all figures and their data sources."""
    print("\nDOCR-MARL Thesis Figures:")
    print("=" * 70)
    print("\nPOST-PROCESSED from Core Training Cache:")

    all_post_figs = sorted(set().union(*PHASE_FIGURES.values()))
    for fig in all_post_figs:
        core = FIGURE_CORE_MAP.get(fig, "?")
        print(f"  Fig {fig:<3} → Core {core} cache + run_post_processing.py")

    print("\nMANUAL/CONCEPTUAL (no automated generation):")
    for fig in sorted(MANUAL_FIGURES, key=lambda x: (isinstance(x, str), x)):
        print(f"  Fig {fig:<4} → Manual diagram / LaTeX / draw.io")

    print(f"\nTotal: {len(all_post_figs)} auto-generated + {len(MANUAL_FIGURES)} manual = 58 figures")
    print(f"\nCache directory: {CACHE_DIR}")

    cache_status = check_cache_status()
    if cache_status["exists"]:
        print(f"Cache entries: {cache_status['entries']}")
        print(f"Cached cores: {sorted(cache_status['cores_present'])}")
    else:
        print("Cache: Not initialized (run training first)")


# ──────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    """Build argument parser combining docl_config and experiment-specific args."""

    if CONFIG_SYSTEM_AVAILABLE:
        config_parser = build_config_parser()

        # Add experiment-specific arguments ONLY
        exp_group = config_parser.add_argument_group("Experiment Orchestration")
        exp_group.add_argument("--all", action="store_true", help="Run all 9 phases with progress tracking")
        exp_group.add_argument("--phase", type=int, choices=range(1, 10), help="Run specific phase (1-9)")
        exp_group.add_argument("--fig", type=int, nargs="+", help="Generate specific figure numbers")
        exp_group.add_argument("--list", action="store_true", help="List all figures and sources")
        exp_group.add_argument("--train-only", action="store_true", help="Only run core training (Phase 2)")
        exp_group.add_argument("--figures-only", action="store_true", help="Only generate figures from cache")
        exp_group.add_argument("--progress", action="store_true", help="Show last experiment progress report")
        exp_group.add_argument("--use-subprocess", action="store_true", help="Use subprocess for figures (slower, fallback)")

        return config_parser
    else:
        # Fallback minimal parser
        # Fallback minimal parser (includes all args since no docl_config)
        parser = argparse.ArgumentParser(
            description="DOCR-MARL Thesis Experiment Orchestrator",
            formatter_class=argparse.RawDescriptionHelpFormatter
        )
        parser.add_argument("--all", action="store_true", help="Run all 9 phases")
        parser.add_argument("--phase", type=int, choices=range(1, 10), help="Run specific phase (1-9)")
        parser.add_argument("--fig", type=int, nargs="+", help="Generate specific figure numbers")
        parser.add_argument("--fast", action="store_true", help="Reduced iterations for testing")
        parser.add_argument("--list", action="store_true", help="List all figures and sources")
        parser.add_argument("--train-only", action="store_true", help="Only run core training (Phase 2)")
        parser.add_argument("--figures-only", action="store_true", help="Only generate figures from cache")
        parser.add_argument("--clear-cache", action="store_true", help="Clear all caches")
        parser.add_argument("--progress", action="store_true", help="Show last progress report")
        parser.add_argument("--use-subprocess", action="store_true", help="Use subprocess for figures")
        parser.add_argument("--config", "-c", type=str, help="Config file path")
        parser.add_argument("--preset", "-p", type=str, help="Preset configuration")
        parser.add_argument("--system", "-s", type=str, help="System name")
        parser.add_argument("--n-runs", type=int, help="Number of runs")
        parser.add_argument("--episodes", "-e", type=int, help="Episodes per run")
        return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    # Initialize configuration from args
    try:
        config = initialize_config(args)
    except Exception as e:
        print(f"Error initializing configuration: {e}")
        sys.exit(1)

    # Setup directories based on config
    global OUTPUT_DIR, FIGURES_DIR, REPORT_DIR, CACHE_DIR, PROGRESS_DIR, LOG_DIR
    OUTPUT_DIR, FIGURES_DIR = setup_directories(config)
    REPORT_DIR = OUTPUT_DIR / "reports"
    CACHE_DIR = OUTPUT_DIR / "core_cache"
    PROGRESS_DIR = OUTPUT_DIR / "progress"
    LOG_DIR = OUTPUT_DIR / "logs"

    # Print config summary
    if CONFIG_SYSTEM_AVAILABLE:
        print_config_summary(config)

    global FAST_MODE
    FAST_MODE = config.experiment.fast_mode
    if FAST_MODE:
        os.environ["DOCR_FAST"] = "1"
        log(colorize("[FAST MODE] Using reduced iterations for testing.", Colors.YELLOW))

    if args.clear_cache:
        clear_all_caches()
        return

    if args.progress:
        show_progress_report()
        return

        # Handle figure listing
    if hasattr(args, 'list') and args.list:
        list_all_figures()
        return

    # Main execution paths
    try:
        if args.train_only:
            run_train_only()
        elif args.figures_only:
            run_figures_only()
        elif args.all:
            run_all()
        elif args.phase:
            run_single_phase(args.phase)
        elif args.fig:
            run_selected_figures(args.fig)
        else:
            parser.print_help()
    except Exception as e:
        log(colorize(f"\nExecution failed: {e}", Colors.RED), level="error")
        sys.exit(1)


if __name__ == "__main__":
    main()