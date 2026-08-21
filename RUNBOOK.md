# RUNBOOK — Running the Full DOCR-MARL Thesis Experiment

Ordered, start-to-finish instructions for producing every result the thesis
needs, under the **active scope**: **8-bus / 14-bus / 39-bus** systems,
**0–60% PV penetration** (0%, 20%, 40%, 60%).

This is a step-by-step walkthrough, not a reference — for the full command
catalogue see `README.md`; for narrative detail on what each phase does,
per-figure data provenance, and Docker/cloud deployment, see
`EXPERIMENT_GUIDE.md`. Repository layout is documented in `CLAUDE.md`.

---

## Step 0 — Understand what "full experiment" actually means here

The pipeline has two parts that run separately, because only one of them is
currently multi-system-aware:

1. **The single-system pipeline** (`run_thesis_experiments.py`) — Core
   training runs A–K plus almost all 58 figures. This runs once, against
   one system (8-bus by default). Cores D–K and most figures are not yet
   parametrized by system — they represent the 8-bus condition only.
2. **The multi-system sweep** — Figures 5 and 6 (OF and CTI-violation-rate
   vs. PV penetration) and Core C (MARL-vs-baselines comparison) *are*
   parametrized by system, and are the results that actually need
   producing three times, once per system in scope.

Run them in the order below. Steps 1–4 are one-time setup; Step 5 is the
single-system pipeline; Step 6 drives the multi-system sweep via
`run_full_experiment.sh`, a script that wraps what used to be six separate
manual steps (per-system Fig 5, Fig 6, Core C, figure regeneration,
results consolidation, and final verification) with skip-if-already-done
logic, so re-running it after an interruption only redoes what's missing.

---

## Step 1 — Set up the environment

```bash
python3 --version   # 3.9+ required, 3.11 recommended
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

`pandapower` and `torch` are optional (see `README.md` §1 for what falls
back if they're absent). Everything else needs plain NumPy/SciPy only.

## Step 2 — Verify the install

```bash
python3 -m pytest tests/ -q
```

Expect all tests to pass (165 as of this writing) before doing anything
else. If this fails, stop and fix it here — every step below assumes a
clean baseline.

## Step 3 — Decide: fresh run or continuing an existing one?

Every Core training run (A–K) is cached under `results/core_cache/`,
keyed by a hash of its full config — it will **not** be silently
overwritten by a re-run with different parameters, it will just look like
a cache hit against the old config and get skipped. Two situations call
for clearing first:

- You're starting the thesis experiment from scratch (recommended the
  first time, so you know exactly what's real).
- You changed code that affects a Core's output and want it retrained.

```bash
# Full reset: Core cache + generated figures/reports/logs
python3 run_thesis_experiments.py --clear-cache
rm -rf results/figures/*/*.pdf results/reports/*.json results/logs/*.txt
rm -rf outputs/*/*.npz outputs/*/*.png outputs/*/*.txt   # multi-system sweep outputs

# Otherwise, just check what's already cached
python3 run_core_training.py --cache-status
```

If you're continuing a partially-done run (e.g. you already have 8-bus
results and only need 14-bus/39-bus), skip the reset and go straight to
the step that's missing.

## Step 4 — Quick validation (always do this before committing to a full run)

```bash
python3 run_thesis_experiments.py --all --preset thesis_sweep_fast
```

Fast preset (2 runs × 500 steps per Core), roughly 1–3 hours wall-clock.
This exercises the entire pipeline end-to-end with placeholder-scale
parameters, so you catch a broken environment or a code error before
spending days on the real thing. Confirm it completes with no errors in
`results/reports/experiment_report_*.json` before continuing.

---

## Step 5 — Run the single-system pipeline (Cores A–K + all figures, 8-bus)

```bash
python3 run_thesis_experiments.py --all --preset thesis_sweep --output-dir results
```

This is the multi-day, full-scale run: all 11 Core training runs (A–K) at
full episode counts, followed by all 58 figures. To run it unattended on a
server:

```bash
nohup python3 run_thesis_experiments.py --all --preset thesis_sweep \
    > results/logs/full_run.log 2>&1 &
echo $!   # save the PID so you can check on it / kill it later
```

To parallelize Core training across CPU cores (verified bit-identical
regardless of worker count — see `EXPERIMENT_GUIDE.md` §2.2.4):

```bash
python3 run_thesis_experiments.py --all --preset thesis_sweep --n-workers 8
```

Check progress at any time with:

```bash
python3 run_thesis_experiments.py --progress
```

**What this step does NOT give you**: real 14-bus/39-bus data for Cores
D–K or the figures they feed, and only an 8-bus baseline comparison for
Core C (Core C gets re-run per-system in Step 7, which supersedes the
8-bus-only comparison this step produces). That's expected — those Cores
aren't parametrized by system yet (see `CLAUDE.md`'s "adequate as-is"
note); this step is still required because it's what produces the other
~45 figures.

## Step 6 — Run the multi-system sweep

```bash
./run_full_experiment.sh
```

This drives, for each system in scope (8bus, 14bus, 39bus), in order: Fig
5 (OF vs. PV), Fig 6 (CTI violation rate vs. PV), and Core C (MARL vs.
baselines) — skipping any system/stage whose output already exists —
then regenerates the figures that read Core C's cache (Figs 11, 12,
31–33, 55–56), consolidates everything into
`results/reports/results_registry.{json,csv}` via
`experiments/build_results_registry.py`, and finishes with
`pytest tests/ -q` plus a cache-status check. Each stage's log lands in
`results/logs/<stage>_<system>.txt`; a failure or interrupt in one
system/stage does not abort the others, and a summary at the end lists
what was skipped and what failed.

**Safe to interrupt and re-run** — just run the same command again.
Completed stages (an existing `outputs/{system}/fig05_pv_sweep.npz`, an
existing `outputs/{system}/fig06_cti_violations.npz`, or an existing
`core_C_baselines_<system>` cache entry) are skipped, not redone, so
resuming after a container restart or a killed terminal only re-attempts
what's actually missing.

Useful variations:

```bash
# Resume only the outstanding pieces (see "Known gaps" below) -- same as
# running with no arguments; nothing already done gets re-run
./run_full_experiment.sh

# Just one stage, one system
./run_full_experiment.sh --stages fig6 --systems 14bus

# Faster, noisier pass (real data, less statistical power)
./run_full_experiment.sh --n-seeds 3 --train-steps 2000

# Force a redo of something already completed
./run_full_experiment.sh --force --stages fig6 --systems 14bus

# All options
./run_full_experiment.sh --help
```

**What the script deliberately does NOT do**: decide whether to clear the
cache first (Step 3) or run the single-system main pipeline (Step 5).
Those still need a human call and a separate invocation — see Step 0 for
why everything isn't just one script.

**A caution carried over unchanged from the original per-stage
instructions**: the Fig 6 sweep this script runs per system has no
`--n_workers` and no partial-save — it runs fully sequentially (7 methods
× 4 PV levels × `n_seeds`) and only writes output once, at the very end.
At reduced settings (`n_seeds=2, train_steps=2000`) an 8-bus pass took
roughly 2.5 hours in a prior run; a 14-bus pass at the same settings
projected past 14 hours before completing. Run the script under `nohup`
(or `screen`/`tmux`) for this reason:

```bash
nohup ./run_full_experiment.sh > results/logs/full_experiment.txt 2>&1 &
echo $!   # save the PID
```

If Fig 6 for a given system is interrupted, its stage has nothing to
resume from — re-running the script will simply redo that system's Fig 6
sweep from scratch (correctly, since no output file exists yet to skip
against), not silently pick up partway through.

### Manual equivalent — running one piece by hand

The script is exactly the commands below, looped over systems with a
skip-if-done check in front of each. Use these directly if you want to
run a single piece without the script, or to understand what's happening
under the hood.

```bash
# Fig 5, one system
python3 baselines/experiments/unified_exp_fig05.py \
    --system 8bus --n_seeds 10 --train_steps 5000 --n_workers 4 \
    --output_dir outputs

# Fig 6, one system
python3 evaluation/utils/exp_fig06_cti_violation_rate.py \
    --system 8bus --n_seeds 10 --train_steps 5000 --output_dir outputs

# Core C, one system (cached under core_C_baselines_<system>; a repeat
# call with the same config is a fast cache hit, not a full re-run)
python3 run_core_training.py --core-C --system 8bus --seed 42

# Force a specific system's Core C cache entry to retrain
python3 -c "
from run_core_training import CacheManager
from pathlib import Path
CacheManager(Path('results/core_cache')).invalidate(pattern='core_C_baselines_8bus')
"

# Regenerate the figures that read Core C / Fig 5 / Fig 6 data
python3 run_thesis_experiments.py --fig 11 12 31 32 33 55 56

# Consolidate into one results registry (full rebuild each time -- always
# safe to re-run)
python3 experiments/build_results_registry.py

# Final verification
python3 -m pytest tests/ -q
python3 run_thesis_experiments.py --list          # confirm every figure's data source
python3 run_core_training.py --cache-status       # confirm all 11 Cores + per-system Core C are cached
```

Cross-check `results/reports/results_registry.json` has rows for all
three systems (8bus, 14bus, 39bus) at all four PV levels before treating
the multi-system results as complete. Any figure still falling back to
synthetic data prints a `[SYNTHETIC]` watermark and logs a warning naming
the missing cache entry — search `results/logs/` for `synthetic` to find
what's still outstanding.

---

## Optional — Docker

```bash
docker build -t docr-marl .
docker run --rm -v "$(pwd)/results:/app/results" -v "$(pwd)/outputs:/app/outputs" \
    docr-marl python3 run_thesis_experiments.py --all --preset thesis_sweep_fast
```

Same steps above apply inside the container — substitute the `docker run`
invocation for the bare `python3` calls, keeping `-v` mounts for
`results/` and `outputs/` so output persists on the host. See
`EXPERIMENT_GUIDE.md` §11 for the GPU build variant and remote/cloud
deployment.

---

## Known gaps as of this writing

- Fig 5 and Fig 6 (Step 6) have not yet been run to completion for every
  system in this environment — 39-bus Fig 5 and 14-bus/39-bus Fig 6 are
  outstanding (interrupted by environment restarts during a prior
  session; Fig 6's lack of partial-save means those need a full re-run,
  not a resume). Running `./run_full_experiment.sh` will skip the systems
  already done (8-bus Fig 5+6, 14-bus Fig 5) and only attempt what's
  actually missing.
- Cores D–K and the figures they feed represent the 8-bus condition only;
  extending them to 14-bus/39-bus is out of scope for the current
  codebase (see `CLAUDE.md`) and would need the same kind of
  per-system parametrization already done for Core C.
