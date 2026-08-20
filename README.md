# DOCR-MARL Thesis Research

Multi-Agent Reinforcement Learning for Directional Overcurrent Relay (DOCR)
coordination on distribution networks with PV penetration.

This file is the quick-start command reference. For narrative detail —
what each phase actually does, per-figure data provenance, Docker/cloud
deployment, troubleshooting, and the reproducibility checklist — see
**[EXPERIMENT_GUIDE.md](EXPERIMENT_GUIDE.md)**. Repository layout and
active scope are documented in **[CLAUDE.md](CLAUDE.md)**.

---

## 1. Setup

```bash
python3 --version   # 3.9+ required, 3.11 recommended (matches Dockerfile)
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

`pandapower` and `torch` in `requirements.txt` are optional: `pandapower`
is only used for Fig 44 (falls back to an analytical fault-current model
without it); `torch` only activates a GPU backend for the ES update when
a CUDA device is present, and is a no-op otherwise. Everything else runs
on plain NumPy/SciPy.

Verify the install:

```bash
python3 -m pytest tests/ -q
```

---

## 2. Clearing the cache (start a fresh experiment)

Every Core training run (A–K) is cached under `results/core_cache/` —
keyed by a hash of its config, **not** overwritten automatically when you
change code or parameters. Before a new experimental run whose results
you don't want silently merged with (or shadowed by) a prior run's cache,
clear it first:

```bash
# Full reset: Core training cache + all generated figures + reports/progress
python3 run_thesis_experiments.py --clear-cache

# Core training cache only (equivalent underlying call)
python3 run_core_training.py --clear-cache

# Check what's currently cached before deciding whether to clear
python3 run_core_training.py --cache-status
```

`--clear-cache` deletes every `.pkl` in `results/core_cache/` and resets
`cache_index.json` — the next training run starts completely from
scratch. There is no in-between "clear just one Core" CLI flag; if you
need that, delete the matching entries from
`results/core_cache/cache_index.json` and their `.pkl` files by hand (see
`CacheManager.invalidate(pattern=...)` in `run_core_training.py`).

To also wipe the tracked figure/log/report artifacts already checked into
the repo (not just the cache), follow the cache clear with:

```bash
rm -rf results/figures/*/*.pdf results/reports/*.json results/logs/*.txt
```

---

## 3. Quick validation (run this first, always)

Fast preset — 2 runs × 500 steps per Core, ~1–3 hours wall-clock —
verifies the whole pipeline end-to-end before committing to a full run:

```bash
python3 run_thesis_experiments.py --all --preset thesis_sweep_fast
```

Narrower checks:

```bash
# Just Core training (Phase 2), fast mode
python3 run_thesis_experiments.py --train-only --preset thesis_sweep_fast

# Just one Core, directly (bypasses the phase orchestrator)
python3 run_core_training.py --core-A --fast --seed 42

# Just figure generation from whatever is already cached
python3 run_thesis_experiments.py --figures-only

# A single figure
python3 run_thesis_experiments.py --fig 5
```

---

## 4. Full experiment run

```bash
python3 run_thesis_experiments.py --all --preset thesis_sweep --output-dir results
```

Runs all 9 phases (Core training A–K, baseline comparison, then every
figure/table). This is multi-day on a single CPU core — see
`EXPERIMENT_GUIDE.md` §2.2.4 for real measured timings and
`--n-workers N` parallelization. To run unattended on a server:

```bash
nohup python3 run_thesis_experiments.py --all --preset thesis_sweep \
    > results/logs/full_run.log 2>&1 &
echo $!   # save the PID
```

Parallelize Core training across CPU cores (bit-identical results,
verified — see `EXPERIMENT_GUIDE.md` §2.2.4):

```bash
python3 run_thesis_experiments.py --all --preset thesis_sweep --n-workers 8
```

Other useful presets (`docl_config.py`): `8bus_full`, `14bus_full`,
`39bus_full`, `3bus_toy` (quick smoke test), `scalability_sweep`,
`curriculum_learning`, `ablation_study`. List them all:

```bash
python3 run_thesis_experiments.py --list-presets
```

### Re-running a single phase

Phases 3–9 read from the Phase-2 cache and can be re-run independently
without retraining:

```bash
python3 run_thesis_experiments.py --phase 4 --preset thesis_sweep   # e.g. robustness figures only
```

### Re-running a single Core

```bash
python3 run_core_training.py --core-G --seed 42
python3 run_core_training.py --all --n-workers 8   # all 11 cores, parallelized
```

---

## 5. Generating figures without retraining

```bash
python3 run_thesis_experiments.py --fig 5           # one figure
python3 run_thesis_experiments.py --fig 5 6 7 8 12   # several
python3 run_thesis_experiments.py --figures-only     # every figure from cache
python3 run_thesis_experiments.py --list             # list all figures + their data sources

# Or drive the figure dispatcher directly:
python3 run_post_processing.py --all
python3 run_post_processing.py --fig 45 46
```

Any figure whose backing Core cache is missing/empty falls back to a
clearly-watermarked synthetic placeholder rather than failing silently —
see `EXPERIMENT_GUIDE.md` §2.2.1 for the full data-provenance table.

---

## 6. Tests

```bash
python3 -m pytest tests/ -q          # full suite (151 tests as of this writing)
python3 -m pytest tests/test_phase4.py -q   # one phase's tests
```

---

## 7. Docker

```bash
docker build -t docr-marl .
docker run --rm -v "$(pwd)/results:/app/results" -v "$(pwd)/outputs:/app/outputs" \
    docr-marl python3 run_thesis_experiments.py --all --preset thesis_sweep_fast
```

See `EXPERIMENT_GUIDE.md` §11 for the GPU build variant and cloud/remote
deployment notes.
