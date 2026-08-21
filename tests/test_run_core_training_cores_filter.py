"""
tests/test_run_core_training_cores_filter.py
──────────────────────────────────────────────
Regression test for run_core_training.py's --cores / config.experiment.cores
filtering.

config.experiment.cores used to be parsed into the config object and then
never read again anywhere -- run_all_cores() unconditionally ran and
manifested all 11 cores as "complete" regardless of what a preset (e.g.
scalability_sweep's cores=["F"]) or --cores actually requested, silently
burning the full training budget (including Core A/C's expensive
5000-episode x 50-run sweeps) on every run.

Tests resolve_requested_cores() directly -- the pure function
run_all_cores() was refactored to delegate to -- rather than calling
run_all_cores() itself: even "fast_mode + use_synthetic", a real
(uncached) Core actually trains, which is far too slow for a test that
runs as part of routine `pytest tests/`. This still exercises exactly the
logic that was broken (which cores get selected from a config), without
the cost of real training or of writing into the repo's real
results/core_cache/results/progress.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from run_core_training import resolve_requested_cores, ALL_CORE_LETTERS
from docl_config import DOCRConfig


def test_explicit_cores_restricts_selection():
    cfg = DOCRConfig()
    cfg.experiment.cores = ["F"]
    assert resolve_requested_cores(cfg, None) == ["F"]


def test_multiple_explicit_cores_preserve_canonical_order():
    cfg = DOCRConfig()
    cfg.experiment.cores = ["D", "B"]  # deliberately out of order
    assert resolve_requested_cores(cfg, None) == ["B", "D"]


def test_default_cores_config_runs_everything():
    cfg = DOCRConfig()  # experiment.cores defaults to all 11 letters
    assert resolve_requested_cores(cfg, None) == ALL_CORE_LETTERS


def test_falls_back_to_global_config_when_no_explicit_config_given():
    global_cfg = DOCRConfig()
    global_cfg.experiment.cores = ["K"]
    assert resolve_requested_cores(None, global_cfg) == ["K"]


def test_no_config_at_all_runs_everything():
    assert resolve_requested_cores(None, None) == ALL_CORE_LETTERS


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
