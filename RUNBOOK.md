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

Run them in the order below. Steps 1–4 are one-time setup; Steps 5–9 are
the actual experiment.

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

## Step 6 — PV-penetration sweep, Figure 5 (OF vs. PV), for each system

Run once per system in scope. This is real MARL + metaheuristic training,
not cached — expect real wall-clock time (see the note on `--n_seeds` /
`--train_steps` below).

```bash
for SYS in 8bus 14bus 39bus; do
    python3 baselines/experiments/unified_exp_fig05.py \
        --system "$SYS" --n_seeds 10 --train_steps 5000 --n_workers 4 \
        --output_dir outputs
done
```

- `--n_seeds` / `--train_steps` default to 10 / 5000 (thesis-scale). If
  you need a faster pass first, reduce both (e.g. `--n_seeds 3
  --train_steps 2000`) — the output is still real, just noisier and with
  less statistical power in the significance tests it also writes to
  `outputs/{system}/fig05_pv_sweep_stats.txt`.
- `--n_workers` parallelizes the (method, PV level, seed) job grid within
  one system. Verified bit-identical regardless of worker count (BLAS
  threads are pinned specifically to guarantee this — see the comment in
  `unified_exp_fig05.py`'s `main()`).
- Output lands in `outputs/{system}/fig05_pv_sweep.{npz,png}` plus the
  stats file, all system-namespaced so a subsequent system's run can't
  overwrite a previous one's.

## Step 7 — CTI violation-rate sweep, Figure 6, for each system

```bash
for SYS in 8bus 14bus 39bus; do
    python3 evaluation/utils/exp_fig06_cti_violation_rate.py \
        --system "$SYS" --n_seeds 10 --train_steps 5000 \
        --output_dir outputs
done
```

**This script has no `--n_workers` and no partial-save** — it runs fully
sequentially (7 methods × 4 PV levels × `n_seeds`) and only writes output
once, at the very end. In practice this is the slowest step in the whole
runbook: at reduced settings (`n_seeds=2, train_steps=2000`) an 8-bus pass
took roughly 2.5 hours in prior runs; a 14-bus pass at the same settings
projected past 14 hours before completing. Plan for this to run overnight
or longer per system at thesis-scale settings, and run it under `nohup`
(or `screen`/`tmux`) so an interrupted terminal session doesn't lose the
whole run:

```bash
nohup python3 evaluation/utils/exp_fig06_cti_violation_rate.py \
    --system 14bus --n_seeds 10 --train_steps 5000 --output_dir outputs \
    > results/logs/fig06_14bus.log 2>&1 &
```

If it's interrupted, there is nothing to resume — it must be re-run from
the start.

## Step 8 — Core C (MARL vs. baselines), for each system

```bash
for SYS in 8bus 14bus 39bus; do
    python3 run_core_training.py --core-C --system "$SYS" --seed 42
done
```

Drop `--seed 42` to use the config default, or set your own — just be
consistent across systems if you want results comparable seed-for-seed.
Cached per system under the name `core_C_baselines_<system>` (see
`results/core_cache/cache_index.json`), so re-running a system you've
already done is a fast no-op unless you change the config or clear the
cache for it specifically:

```bash
python3 -c "
from run_core_training import CacheManager
from pathlib import Path
CacheManager(Path('results/core_cache')).invalidate(pattern='core_C_baselines_8bus')
"
```

## Step 9 — Regenerate the figures that read Core C / Fig 5 / Fig 6 data

Steps 6–8 write their own `.png`s directly, but the figures inside the
main pipeline that also read Core C's cache (Figs 11, 12, 31–33, 55–56)
need re-generating once Core C has real per-system data:

```bash
python3 run_thesis_experiments.py --fig 11 12 31 32 33 55 56
```

## Step 10 — Consolidate everything into one results registry

```bash
python3 experiments/build_results_registry.py
```

Pulls every system's Fig 5/6 output and Core C cache entry into
`results/reports/results_registry.{json,csv}` — one row per
(method, system, PV level), ready for further statistical analysis,
a spreadsheet, or a notebook. Re-run this any time after adding a new
system's data; it's a full rebuild, not an incremental append, so it's
always safe to re-run.

## Step 11 — Final verification

```bash
python3 -m pytest tests/ -q
python3 run_thesis_experiments.py --list          # confirm every figure's data source
python3 run_core_training.py --cache-status       # confirm all 11 Cores + per-system Core C are cached
```

Cross-check `results/reports/results_registry.json` has rows for all
three systems (8bus, 14bus, 39bus) at all four PV levels before treating
the multi-system results as complete. Any figure still falling back to
synthetic data prints a `[SYNTHETIC]` watermark and logs a warning
naming the missing cache entry — search `results/logs/` for `synthetic`
to find what's still outstanding.

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

- Fig 5 (Step 6) and Fig 6 (Step 7) have not yet been run to completion
  for every system in this environment — 39-bus Fig 5 and 14-bus/39-bus
  Fig 6 are outstanding (interrupted by environment restarts during a
  prior session; Fig 6's lack of partial-save means those need a full
  re-run, not a resume).
- Cores D–K and the figures they feed represent the 8-bus condition only;
  extending them to 14-bus/39-bus is out of scope for the current
  codebase (see `CLAUDE.md`) and would need the same kind of
  per-system parametrization already done for Core C.
