# DOCR-MARL Thesis Experiment Guide

Complete instructions for reproducing all results — training, evaluation,
figures, and tables — for the MARL-based Directional Overcurrent Relay
coordination thesis.

**Active scope:** IEEE 8-bus, 14-bus, and 39-bus networks at PV penetrations
of 0 %, 20 %, 40 %, and 60 %.

---

## 0. Prerequisites

### 0.1 Python environment

```bash
python --version   # must be 3.9 or later
pip install numpy scipy pyyaml scikit-learn matplotlib shap
pip install pandapower   # optional — analytical fallback active without it
```

`pandapower` is needed only for Fig 44 (fault contribution analysis).
Everything else works with the built-in analytical fault-current model.

### 0.2 File layout

Place every file delivered in this session in one directory.
The directory must contain at minimum:

```
docl_config.py               ← config system (now fixed)
run_thesis_experiments.py    ← main orchestrator
run_core_training.py         ← 11-core MARL training engine
run_post_processing.py       ← figure generation dispatch
benchmark_suite.py           ← GA-SQP / HWOA / HHO / JAYA / EEO vs MARL
stat_analysis.py             ← move into evaluation/ sub-package (see §0.3)
network/
  network_data.py            ← 8/14/39-bus definitions + N-1 outage physics
  relay_physics.py
  topology_manager.py
  pandapower_network.py
marl/
  marl_agents.py
  marl_env.py
  safe_rl.py
  continual_trainer.py
  comm_graph.py
baselines/
  ga_sqp.py
  metaheuristics.py
evaluation/
  __init__.py                ← empty file
  stat_analysis.py           ← copy of stat_analysis.py (required import path)
pv_model/
  pv_inverter.py
  scenario_generator.py
environment/
  dynamic_network.py
  topology_manager.py        ← symlink/copy from network/
  relay_physics.py           ← symlink/copy from network/
```

### 0.3 Create the evaluation package (one-time)

`benchmark_suite.py` and `thesis_validation.py` both import from
`evaluation.stat_analysis`. Create the package and copy the file in:

```bash
mkdir -p evaluation
touch evaluation/__init__.py
cp stat_analysis.py evaluation/stat_analysis.py
```

### 0.4 Resolve duplicate-namespace copies

Two modules are imported under two different package prefixes by different
parts of the codebase (a pre-existing structural defect, not introduced by
the fixes in this session):

```bash
cp network/topology_manager.py environment/topology_manager.py
cp network/relay_physics.py    environment/relay_physics.py
cp network/pandapower_network.py network/pandapower_network.py   # already there
```

### 0.5 Verify the setup

```bash
python -c "
from network.network_data import NETWORKS
from marl.marl_env import DynamicDOCREnv
from marl.marl_agents import MAPPOMARLTrainer
from benchmark_suite import BenchmarkSuite
print('Setup OK — all imports resolved')
for s in ['8bus','14bus','39bus']:
    n = NETWORKS[s]()
    print(f'  {s}: {n.n_relays} relays, {len(n.pairs)} pairs, {n.n_lines} lines')
"
```

Expected output:
```
Setup OK — all imports resolved
  8bus:  14 relays, 20 pairs, 9 lines
  14bus: 40 relays, 52 pairs, 20 lines
  39bus: 92 relays, 86 pairs, 46 lines
```

**Update (2026-09-09):** 39bus's relay/N-1 topology was rebuilt from a
real 46-branch/39-bus graph (verified against MATPOWER case39.m) instead
of an abbreviated 23-branch/22-bus graph that left 17 buses — including
all 10 real generator buses — with zero relay or N-1 coverage. `n_lines`
now reports real branch counts (9 for 8bus: 7 lines + 2 transformers; 20
for 14bus; 46 for 39bus) instead of a relay-pair-derived tuple count. The
old 39bus graph is preserved as `NETWORKS["39bus_reduced"]` (46 relays,
42 pairs, 22-of-39 bus coverage) so previously-published results using
it stay reproducible. See §9's new "Topology provenance" subsection for
the full history, the retraining this requires, and the known remaining
limitation (27 of 92 relays end up with no backup pair, mostly an
artifact of MATPOWER's arbitrary branch from/to direction convention,
same class of limitation 14/57/118bus already have).

---

## 1. Quick validation (run first, always)

Before committing to a full run, verify the entire pipeline executes
without error using the fast preset (2 runs × 500 steps):

```bash
python run_thesis_experiments.py --all --preset thesis_sweep_fast
```

Expected wall-clock: **1–3 hours** on a modern CPU.
Produces real (small-budget) figures in `results/figures/` and logs in
`results/logs/`. Check that Phase 2 completes with a non-empty benchmark
comparison table printed to stdout.

To verify individual pieces independently:

```bash
# Just Phase 2 training + Obj4 comparison (fastest meaningful check)
python run_thesis_experiments.py --train-only --preset thesis_sweep_fast

# Just figure generation from existing cache
python run_thesis_experiments.py --figures-only

# A single figure
python run_thesis_experiments.py --fig 5
```

---

## 2. Full experiment run

### 2.1 Single command (recommended)

```bash
python run_thesis_experiments.py --all \
    --preset thesis_sweep \
    --output-dir results
```

This runs all 9 phases in sequence. Estimated wall-clock: **3–5 days** on a
single machine (the code says "8–10 GPU-days"; on a decent laptop with no
GPU, ES training is the bottleneck).

To run on a server and keep running after disconnect:

```bash
nohup python run_thesis_experiments.py --all --preset thesis_sweep \
    > results/logs/full_run.log 2>&1 &
echo $!   # save PID
```

### 2.2 What each phase does

| Phase | What runs | Key outputs |
|-------|-----------|-------------|
| 1 | Manual-script figures (architecture diagrams, relay curves, network maps) | `results/figures/figs_01_10/` |
| 2 | **All 11 training cores (A–K)** + **Obj4 baseline comparison (GA-SQP, HWOA, HHO, JAYA, EEO vs MARL, all three systems, all four PV levels)** + statistical significance tests | `results/core_cache/`, `results/benchmark_pv*.pkl`, `results/stat_comparison_pv*.txt` |
| 3 | Training-dynamics figures (PV sweep Fig 5/6/7/8, convergence Fig 12, Lagrange Fig 10, safe-RL Fig 16/21, per-agent Fig 37/38/41) | Figures 5–12, 16, 21, 37, 38, 41 |
| 4 | Robustness & resilience (comm stress Fig 13, N-1 contingency Fig 14, Byzantine Fig 20, sensor robustness Fig 22/24, SHAP Fig 54) | Figures 13, 14, 20, 22, 24, 54 |
| 5 | Scalability & transfer (Fig 15, unseen faults Fig 23, zero-shot Fig 25) | Figures 15, 23, 25 |
| 6 | Deployment (compression Fig 18, model size Fig 53) | Figures 18, 53 |
| 7 | Protection engineering & interpretability (sympathetic tripping Fig 43, fault contribution Fig 44, daily PV Fig 45, relay curves Fig 46/47, t-SNE Fig 48/49, critical pairs Fig 50, seasonal Fig 58) | Figures 43–50, 58 |
| 8 | Statistical aggregation + ablation/hparam (Figs 11, 31–33, 55–57, 39, 40, 42) | Figures 11, 31–33, 39, 40, 42, 55–57 |
| 9 | Final compilation report | `results/reports/final_report.json` |

### 2.2.1 Data provenance warning — synthetic vs. real figures

**As of 2026-08-07, the following figures are generated from hand-typed literal
data arrays in `evaluation/experiments/*.py`, not from a trained model or a
real experiment run** (see `experiments/GAP_ANALYSIS.md` for how this was
found). Each affected script now prints a `[WARNING] SYNTHETIC DATA` line and
watermarks its own output — do not cite these as experimental evidence until
they are replaced with a real data pipeline:

| Figure | Script | Status |
|--------|--------|--------|
| 13 (comm stress) | `evaluation/experiments/communication_stress.py` | synthetic |
| 14 (N-1 contingency) | `evaluation/experiments/n1_contingency.py` | synthetic |
| 15 (scalability) | `evaluation/experiments/scalability_sweep.py` | synthetic |
| 18 (inference latency) | `deployment/benchmark_latency.py` | synthetic (lognormal noise, no model loaded) |
| 20 (Byzantine resilience) | `evaluation/experiments/byzantine_resilience.py` | synthetic |
| 22 (bandwidth tradeoff) | `evaluation/experiments/bandwidth_tradeoff.py` | synthetic |
| 23 (unseen faults) | `environment/experiments/unseen_faults.py` | synthetic |
| 24 (sensor robustness) | `environment/experiments/sensor_degradation.py` | synthetic |
| 25 (zero-shot transfer) | `evaluation/experiments/zero_shot_transfer.py` | synthetic |
| 29 (EWC continual learning) | `marl/experiments/ewc_visualization.py` | synthetic (also had a duplicate `--system` argparse registration that crashed every invocation — fixed) |
| 36 (action space geometry) | `environment/experiments/action_space_geometry.py` | synthetic |
| 41 (sample efficiency) | `marl/utils/sample_efficiency.py` | synthetic |
| 42 (warm start) | `marl/experiments/warm_start.py` | synthetic (same duplicate-argparse crash as Fig 29 — fixed) |
| 43 (coordination diagram) | `network/utils/coordination_diagram.py` | synthetic |
| 44 (fault contribution) | `network/utils/fault_contribution.py` | synthetic |
| 45 (daily PV profile) | `data/experiments/daily_adaptive_response.py` | synthetic |
| 46 (relay curves) | `network/utils/relay_curves.py` | synthetic |
| 47 (sympathetic tripping) | `network/utils/sympathetic_trip_cases.py` | synthetic |
| 48 (SHAP) | `evaluation/experiments/shap_analysis.py` | synthetic |
| 49 (decision boundary) | `evaluation/experiments/decision_boundary.py` | synthetic |
| 50 (critical pairs) | `network/utils/critical_pairs.py` | synthetic |
| 53 (model compression) | `evaluation/experiments/model_compression.py` | synthetic |
| 54 (adversarial robustness) | `environment/experiments/adversarial_robustness.py` | synthetic |
| 58 (seasonal generalization) | `evaluation/experiments/seasonal_generalization.py` | synthetic |

That is 23 of the ~58 thesis figures. All 23 share the same "CLI Adapter"
docstring pattern and were almost certainly generated in one batch; each now
imports `evaluation.experiments._synthetic_label.mark_synthetic` and
watermarks its own output.

**A second, separate implementation of most of these same figures lives in
`run_post_processing.py`** (`fig_NN_*` functions, dispatched via
`FIGURE_FUNCTIONS` — this is the code path `run_thesis_experiments.py`
actually calls, not the CLI adapters above). 41 of its functions follow the
pattern `data = load_cache(name); if data is None: data = {hardcoded}` —
i.e. they silently fall back to fabricated numbers whenever the named
`results/core_cache/` entry is missing. As of 2026-08-07 every one of these
41 functions now sets `is_synthetic = data is None`, logs a warning naming
the missing cache, and watermarks the figure when the fallback actually
fires. (2026-08-08: `results/core_cache/` was cleared once to verify this
disclosure behavior end-to-end; Cores A–K were re-run immediately after and
the cache was repopulated with real results.)

**Update (2026-08-29): Cores A–K re-run fresh** (`run_core_training.py --all
--fast`, ~40 commits' worth of production-path fixes after the prior cache),
and all 58 figures regenerated (`run_post_processing.py --all`, 50/50
auto-generated, 0 failures). As of that run, only 8 of the 58 figures are
synthetic — each for a pre-existing, documented architectural reason (not a
missing cache): Fig 14 (a separate standalone script's own missing .npz),
Fig 33/56 (Core C doesn't measure the specific per-dimension/speed-vs-
selectivity breakdown those figures plot), Fig 36 (no Core evaluation feeds
it by design), Fig 38 (Core A's batched-ES trainer never captures per-step
attention snapshots), Fig 53/18 (Core I's Phase 2 GNN latency has no real
measurement path — see `experiments/exp_fig18_inference_latency.py`'s own
disclosure), and Fig 58 (Core K measures `{of, cti_ok}` per season, not the
`performance`/`pv_variability` fields the figure plots). See each figure
function's own code comment in `run_post_processing.py` for the full
rationale. Also fixed in this pass: `shap==0.41.0`'s prebuilt C extension is
ABI-incompatible with numpy>=2.0, which was silently degrading Fig 48 (SHAP)
to its null fallback with no visible error — `requirements.txt` now pins
`shap>=0.51.0`.
See the module docstring at the top of `run_post_processing.py` for the full
list of figures and their cache dependencies, and `tests/test_phase6.py` /
`tests/test_phase7.py` for regression coverage locking the disclosure
behavior in.

Figures produced by `evaluation/utils/exp_fig06_cti_violation_rate.py` and the
repaired `baselines/experiments/unified_exp_fig05.py` (Fig 5, now including
GA-SQP/HWOA/HHO/JAYA/EEO with real Wilcoxon/Mann-Whitney + Cliff's delta
testing) are genuine — they train/evaluate real models when their Core cache
is present.

**Update (2026-09-05): 4 of the 8 previously-listed synthetic figures fixed
to use real data**, each by capturing/wiring data the pipeline already
computed but discarded, or a small new eval reusing an already-trained
policy (no new training runs): Fig 44 (a real bug — the pandapower
external grid was missing `s_sc_max_mva`, now set); Fig 12b, 29a, and 38
(Core A/C's real per-episode returns, per-parameter Fisher diagonal, and
per-step attention snapshots were already computed and then discarded
before reaching the cache); Fig 58 (Core K now also reports a real
`performance`/`pv_variability` pair per season); Fig 47b (per-fault-type
sympathetic-tripping breakdown, looped over the 3 real fault types this
codebase models — there is no "High-Z" fault-impedance model anywhere,
so that category is dropped rather than fabricated); Fig 14 (Core G now
sweeps the already-implemented, previously-unused
`_eval_n1_contingency` over every line); Fig 36 (a new
`action_projection_samples` capture in Core A's eval loop, using the
same `ActionProjector` the trained policy is actually subject to); and
Fig 53 (Core I now runs its already-implemented int8/pruning compression
once per method instead of once total — "distillation" is dropped, since
no student-network training exists anywhere in this codebase). Figs 33
and 56 gained 3 of their axes for real (CTI satisfaction rate,
convergence episode, and — for 56 — real per-relay operating time/CTI
margin) via new per-run metrics on Core C, but keep a smaller illustrative
"Robustness" component pending a dedicated eval. Figs 18, 33/56 (partial),
38 (unconditionally-illustrative sub-panel), 53, and 58 remain
architecturally bounded per the reasons above; see each figure function's
code comment in `run_post_processing.py` for specifics.

**Update (2026-09-09): observation space expanded 10→12 (OBS_DIM), two new
features added.** `marl/marl_env.py::DynamicDOCREnv._get_obs()` now appends
per-agent voltage (`v_norm`) and power-flow direction (`p_dir`). Read the
actual disclosure in that function's own comments before citing these as
new real signal: `v_norm` is a genuine AC-solve quantity only under
`use_pandapower=True`, which the code's own comment says is NOT what "the
vast majority of training" actually uses — under the (default) analytical
path it's a deterministic formula of two features already in the
observation vector. `p_dir` is **never** a real per-step measurement in
either mode; it's always `2*pv_level*is_dg_bus - 1` — for non-DG-bus
agents a constant, for DG-bus agents a value perfectly linearly collinear
with the already-present `pv_level` feature. Neither is fabricated (both
are disclosed in-code), but "add real V and P_dir observation features"
overstates what these actually deliver for the training regime this
pipeline's own comments say matters most.

Found and fixed while checking this change didn't miss any stale
`OBS_DIM=10` references: `experiments/exp_fig53_model_compression.py` had
`obs_dim = 10` hardcoded (would crash feeding a 10-wide synthetic batch
into the real 12-wide actor) — now derived from `env.OBS_DIM`/`env.ACT_DIM`
directly. `run_post_processing.py::fig_48_shap()`'s feature-name list still
had 10 entries (now 12, matching `marl_env.py`'s own documented feature
order) — and, separately, its SHAP-array axis-averaging logic assumed a
fixed `(n_actions, n_explain, n_features)` shape that shap==0.51.0 (this
session's numpy-2.x-compatible upgrade from the broken 0.41.0 pin) doesn't
actually use; verified the real installed version returns `(n_explain,
n_features, n_actions)` instead, so every SHAP figure generated since that
upgrade had silently averaged over the wrong axes. Now locates the feature
axis by its actual size rather than assuming a fixed position.

### 2.2.2 Training efficiency (2026-08-07)

Three changes reduce how long real training actually takes, none of which
change results:

- `marl/marl_agents.py`'s ES actor/critic updates now batch their forward
  passes across a whole rollout in one vectorized call instead of looping
  per-timestep — ~25x faster actor updates, ~5x faster critic updates,
  verified bit-identical to the old per-timestep loop.
- `MAPPOMARLTrainer.train()` accepts `early_stop_patience` (default `None`,
  opt-in) to stop once the objective hasn't improved in that many
  consecutive rollouts, instead of always running the full fixed
  `total_steps` budget. Gated on reaching the final curriculum stage when
  curriculum learning is enabled.
- The same batching approach was applied to `marl/single_agent_ppo.py`
  (14.7x on `log_prob`) and `marl/comm_graph_gradient.py`'s critic (4.1x),
  both verified bit-identical to their per-timestep originals.
  **Not** applied to `marl/comm_graph.py`'s GNN attention layer — measured
  *slower* when batched (0.5x–0.9x at n=14–46 agents). The O(n²) pairwise
  tensor construction (`repeat`+`concatenate`) allocates a large temporary
  array per call; batching over the rollout multiplies that allocation by
  the rollout length, which costs more than the Python loop it removes.
  Benchmark before assuming a vectorization helps — it does for plain
  matmul-based layers, not for this attention mechanism.
- `--n_workers N` parallelizes independent (method/config, seed) job grids
  across N processes in `baselines/experiments/unified_exp_fig05.py`,
  `baselines/experiments/exp_fig05_pv_sweep_evaluation.py`,
  `experiments/exp_fig12_convergence_curves.py`,
  `marl/experiments/exp_fig15_scalability.py`,
  `marl/experiments/exp_fig16_ewc_effectiveness.py`,
  `marl/experiments/exp_fig39_ablation.py`, and
  `marl/experiments/exp_fig40_hparam_grid.py` — each job is independently
  seeded and self-contained, so results are bit-identical regardless of N
  (verified). Each worker is pinned to 1 BLAS thread via `threadpoolctl`;
  without this, N worker processes each trying to use all available cores
  for NumPy/BLAS oversubscribes the machine and makes `--n_workers 4`
  *slower* than sequential, not faster — measured directly during
  development (1m4s sequential vs 1m11s naively-parallel, vs 18s once
  thread-pinned, on `unified_exp_fig05.py`). Default is `--n_workers 1`
  (sequential); pass a worker count matching available cores to use this.
  Results are pre-allocated by seed/config index (never appended in
  completion order) so paired statistical tests across methods stay
  correctly aligned regardless of which worker finishes first, and seed
  offsets use `zlib.crc32` instead of `hash()` (Python randomizes `hash()`
  per process by default, which silently breaks reproducibility across
  worker processes).
- While instrumenting the above scripts for parallelization, three
  pre-existing correctness bugs were found and fixed (unrelated to
  parallelization itself, but each blocked the script from completing a
  run at all):
  - `marl/experiments/exp_fig39_ablation.py`'s Fisher-information estimator
    (`_estimate_fisher_for_task`) sampled actions from raw environment
    observations even when `use_comm=True`, crashing with a matmul
    dimension mismatch the moment EWC ran on a comm-enabled config (2 of 5
    ablation arms). Fixed to apply the same comm-graph enhancement already
    used correctly elsewhere in the file.
  - `marl/experiments/exp_fig16_ewc_effectiveness.py` and
    `marl/experiments/exp_fig40_hparam_grid.py` each called `main()` with
    more positional arguments than the function accepted (an undefined
    `system` in the latter's case) — both scripts raised immediately on
    invocation. Fixed by adding the missing `system`/`n_workers` parameters
    and threading `system` through to the network-loading calls that had
    been silently ignoring it.
  - `baselines/experiments/exp_fig05_pv_sweep_evaluation.py` only added its
    own directory to `sys.path`, not the repo root, so running it directly
    (`python3 baselines/experiments/exp_fig05_pv_sweep_evaluation.py`)
    failed with `ModuleNotFoundError: No module named 'network'`. Fixed to
    match the repo-root-relative pattern used elsewhere.

### 2.2.3 Standalone-invocation bugs fixed repo-wide (2026-08-07)

A full sweep (`python3 <script> --help` from repo root, for every script
with a `sys.path.insert`/`__main__` block) found and fixed every remaining
instance of the `sys.path` bug noted above, plus a few unrelated crash
bugs turned up by the same sweep:

- ~25 files across `experiments/`, `mains/`, `marl/` (top level),
  `marl/utils/`, and `evaluation/utils/` only added their own directory to
  `sys.path`, not the repo root, so `from network...`/`from marl...`
  failed immediately with `ModuleNotFoundError` whenever run directly
  (they worked only when something else had already fixed up `sys.path`
  first, e.g. via `run_thesis_experiments.py`'s direct-call path). Fixed
  to the same repo-root-relative pattern used elsewhere.
- `marl/experiments/attention_evolution.py` and
  `marl/experiments/curriculum_trainer.py` each registered `--system`
  twice with `argparse`, which raises `ArgumentError` unconditionally —
  neither script could run at all, regardless of arguments.
- `mains/main1.py` imported `MARLTrainer`, a class that no longer exists
  in `marl/marl_agents.py` (renamed to `MAPPOMARLTrainer` — every other
  `mains/main*.py` already used the new name). Fixed the import and the
  two call sites.
- `phase2_scripts/*.py` (5 of 7 files) assumed the `exp_fig*.py` file they
  wrap lives alongside them — true only if this legacy adapter package
  (see `README_PHASE2_ADAPTERS.md`) had been copied over the repo root as
  its own instructions describe, which never happened. In this repo the
  real files live in `marl/experiments/` or `marl/utils/`. Pointed each
  adapter's `sys.path` entry at the real location instead of assuming
  co-location. (Confirmed unreferenced by any live pipeline code — these
  adapters are superseded by same-named, more capable versions already in
  `marl/experiments/`/`marl/utils/` — but fixed rather than removed since
  removal wasn't asked for.)

Verified: every script in this sweep now reaches its own `--help` output
(or, for scripts without argparse, runs past the import section) instead
of crashing on import.

### 2.3 Running phases individually

If a phase fails (e.g. power outage mid-run), resume from that phase:

```bash
# Re-run just Phase 4 (robustness figures) after fixing an issue
python run_thesis_experiments.py --phase 4 --preset thesis_sweep

# Re-run Phase 2 training only (if baselines need to be re-computed)
python run_thesis_experiments.py --phase 2 --preset thesis_sweep
```

Phases 3–9 all read from the cache written by Phase 2, so they can be
re-run independently without re-training.

### 2.4 Per-system runs (if splitting across machines)

To run each network on a separate machine and merge results:

```bash
# Machine 1
python run_thesis_experiments.py --all --system 8bus --pv 0.6 \
    --output-dir results_8bus

# Machine 2
python run_thesis_experiments.py --all --system 14bus --pv 0.6 \
    --output-dir results_14bus

# Machine 3
python run_thesis_experiments.py --all --system 39bus --pv 0.6 \
    --output-dir results_39bus
```

Then merge the benchmark pickle files manually before running Phase 8:

```bash
python -c "
import pickle, glob
all_results = {}
for f in glob.glob('results_*/benchmark_pv060.pkl'):
    with open(f,'rb') as fh:
        r = pickle.load(fh)
    all_results.update(r)
with open('results/benchmark_pv060_merged.pkl','wb') as fh:
    pickle.dump(all_results, fh)
"
```

### 2.2.4 Parallelized Core training, and an optional GPU backend (2026-08-08)

**Core A–F parallelization.** These are the six Cores that do real,
independent per-item training with no shared state — A/B/C/D each run
`n_runs` (or `n_runs × methods`/`n_runs × configs`) trainings, E runs an
80-config hyperparameter grid × `n_seeds`, F runs one training per network
size — so they all parallelize the same way the standalone scripts in
§2.2.2 do:

```bash
python run_core_training.py --all --n-workers 8       # match your machine's core count
python run_thesis_experiments.py --all --n-workers 8  # threads the same flag into Phase 2
```

Verified bit-identical output between `--n-workers 1` and `--n-workers 4`
on real (non-synthetic) training for all of A, E, and F: Core A 3.8x
speedup (49.6s → 13.1s), Core E 4.18x (74.0s → 17.7s, small grid), Core F
3.76x (dispatch/aggregation verified against a fast stub — its own
per-system training cost, hardcoded at 30,000 steps per system, is the
same code path already verified via Core A). As with every other
`--n-workers` flag in this repo: BLAS threads are pinned to 1 per worker
via `threadpoolctl` to avoid oversubscription, results are pre-allocated
by run/method/config/system index rather than appended in completion
order, and the `hash()`-based seed offsets Cores C and D used internally
(non-deterministic across processes — see §2.2.2's `zlib.crc32` note) were
fixed the same way. Default is `--n-workers 1` (sequential, matching prior
behavior exactly).

**Update: Cores G/H/I/K are now parallelized too** (`run_core_G`/`run_core_H`/
`run_core_I`/`run_core_K` all accept `n_workers`, same pinned-BLAS-thread
`mp.Pool` pattern as A-F). Each still only makes a handful (1-5) of
sequential-by-design calls into `core_eval_bridge.py` per config, so the
speedup is smaller than A-F's, but this is no longer accurate as an
"unparallelized, left as-is" description. Fast-mode `n_runs` for A-D now
also auto-scales to the worker pool size via `_n_runs_for_wave()` (fills
one full pool "wave" rather than leaving workers idle) — this means the
same `--fast` run's sample size (and therefore any confidence interval)
genuinely differs depending on `--n-workers`/pool size, which is NOT
currently logged/disclosed anywhere in the run's own output. If comparing
fast-mode results across two runs, check `n_runs` in
`results/core_cache/cache_index.json`'s config for that entry rather than
assuming it matches.

From-scratch time estimate for Phase 2 at full scale (`thesis_sweep`
preset), measured directly on one CPU core rather than trusting this
codebase's own stale "~8-10 GPU-days" comment (there's no GPU dependency
anywhere in the *default* code path to make that unit meaningful) or
EXPERIMENT_GUIDE's previously-stated "1-3 hours" for the fast preset (that
undercounts too — measured fast-preset cost for Cores A-D alone is closer
to 7-8 hours single-core): **Cores A-D full-scale is roughly 40 days
single-core sequential**, dominated by Core C's baseline comparison (4
trainable methods × 50 runs × 5000 episodes); Core E adds a further chunk
(80 configs × `n_seeds`) and Core F adds ~30,000 steps per network size.
`--n-workers N` divides this close to linearly (per the measured
3.7x-4.2x-on-4-workers figures above) — on a 16-32 core cloud VM (see
§11), this becomes on the order of days rather than weeks.

**GPU support (2026-10-22 rewrite — supersedes everything this
subsection said before that date).** The ES-based actor/critic update
this subsection used to describe (`_update_actor_es`/`_update_critic_es`
and their GPU-only population-batched variants) no longer exists.
`marl/marl_agents.py`'s `SharedGaussianActor`/`MAPPOCentralCritic` (and
`marl/comm_graph.py`'s GNN, and `marl/single_agent_ppo.py`'s centralized
baseline) are now real `torch.nn.Module`s trained via backprop + Adam —
see `marl/marl_agents.py`'s module docstring for the full history. This
is a much bigger change than swapping an optimizer: real backprop does
one lean forward+backward pass per update call, versus ES's old
`2×n_pert` forward passes, which matters a lot for the GPU verdict below.

**GPU no longer auto-activates, and is opt-in for a specific reason.**
Earlier (2026-10-21), the ES-era GPU path was benchmarked for the first
time on real hardware (an NVIDIA GTX 1050, 4GB) and measured as a real
1.32x–1.58x speedup on the actor update. After the backprop rewrite, the
same benchmark was repeated against the new code and the verdict
**reversed**:

| System | Agents | CPU actor sweep | GPU actor sweep | Speedup |
|---|---|---|---|---|
| 8-bus | 14 | 0.253s | 0.563s | 0.449x (slower) |
| 39-bus | 92 | 2.215s | 3.298s | 0.672x (slower) |

Mechanistically this makes sense even though it's the opposite of the
ES-era result: real backprop's per-call compute is already so much
cheaper than ES's old `2×n_pert`-forward-pass pattern that GPU's fixed
per-call overhead (kernel launch, synchronization, host-device transfer)
now dominates even more heavily than before, on a *smaller* total compute
budget. The critic showed a small, inconsistent edge at 39-bus (1.164x)
but the critic update is tiny in absolute terms either way (~0.03s) and
not a load-bearing result. Both this and the original ES-era benchmark
agree on one thing: on this exact hardware, GPU is not a default win —
only the *direction* of the verdict changed when the algorithm did.

Because of this, GPU use requires an explicit environment variable,
`DOCR_USE_GPU=1`, checked once at `marl/marl_agents.py` import time
(`_GPU_AVAILABLE = torch.cuda.is_available() and
os.environ.get("DOCR_USE_GPU") == "1"`). Without it, every actor/critic/
GNN module defaults to `device="cpu"` regardless of what hardware is
physically present — auto-enabling GPU whenever it happened to exist was
the previous (now corrected) behavior, and would silently make training
*slower* by default on any machine with a GPU, which is the opposite of
what this subsection is about. **If you're on different hardware** (a
much more powerful GPU, larger batch/population sizes, or both) **where
you have real reason to expect this verdict flips, set
`DOCR_USE_GPU=1` and re-measure with the same methodology before trusting
it** — this project's own experience (twice now, in opposite directions)
is the reason this guide keeps saying not to assume a GPU/vectorization
port helps without checking.

Also re-confirmed under the new algorithm (unchanged from the ES-era
finding): switching tensors from `float64` to `float32` on the theory
that consumer GPUs cripple double-precision throughput measured
**slower** (0.72x–0.76x) — the RNG's native `float64` numpy arrays cost
more to cast than any float32 throughput gain recovers, at tensor sizes
this small. float64 stays.

**Real risk found while benchmarking this, not present in any actual
call site in this codebase, but worth knowing:** importing `torch`
before `pandapower` in the same process segfaults immediately (zero
Python traceback, zero diagnostic output) on at least one real
development machine used on this project — a native OpenMP/MKL runtime
conflict between torch's and pandapower's (via numba) bundled runtimes.
Confirmed reproducible, and order-dependent: `pandapower` → `torch` is
safe, `torch` → `pandapower` crashes. Every real entry point in this
codebase (`run_core_training.py` et al.) imports `network`/`marl.marl_env`
— which transitively imports `pandapower` via
`environment/dynamic_network.py` — before `marl.marl_agents` — which
imports `torch` — so this is **not currently triggered** by any real
code path. It would be triggered by any new script, notebook, or
interactive session that imports `torch` first for an unrelated reason
before importing anything from this project. If you ever see Python exit
with no traceback at all while this project's modules are involved, check
import order before assuming it's a hardware fault.

**This project's real speed lever for the full campaign is still CPU
multiprocessing** (§2.2.4 above, 3.7x–4.2x on just 4 workers, scaling
further with more cores) — not GPU, now more clearly than ever given the
benchmark above. `run_core_training.py`'s multi-worker pools no longer
blanket-disable GPU (`_disable_gpu_in_worker()` has been superseded by
`_adaptive_gpu_in_worker()` + `_max_safe_gpu_workers()`, which cap how
many concurrent workers may each open their own CUDA context based on
real free VRAM — preventing the `torch.AcceleratorError: CUDA error: out
of memory` crash already hit once on this project's 4GB card from
multiple workers contending for one small GPU). That cap only matters at
all once `DOCR_USE_GPU=1` is set; by default every worker stays on CPU,
which — per the benchmark above — is also currently the *faster* choice
on top of being the always-safe one.

Numerical correctness of both the batched-population reformulation and its
GPU/torch port were verified by comparing against the original
per-perturbation-loop implementation with identical starting parameters and
identical RNG state (`copy.deepcopy`'d bit-generator state, restored before
each variant runs) — actor and critic gradients matched to ~1e-16 (float64
machine epsilon), for both `use_comm=True` and `use_comm=False`. **What
could not be verified: actual speedup on real GPU hardware.** No GPU was
available in the environment this was written in, so the "GPU-only, not
CPU" backend-selection choice is an informed bet — GPUs are generally
better suited than CPU/BLAS to many-small-matmul batched workloads like
this one (kernel-launch overhead amortizes across the batch instead of
being paid per perturbation) — not a benchmarked result. If you have access
to a CUDA machine, `python -m pytest tests/ -q` after installing torch with
CUDA support is a reasonable place to start re-verifying this, followed by
timing `MAPPOMARLTrainer.train()` before/after forcing `torch.cuda.is_available()`
to confirm it actually helps at your problem sizes before trusting it for a
real run — the same "benchmark, don't assume" standard this finding itself
was held to.

---

## 3. Results structure

After a complete run, your `results/` directory will contain:

```
results/
├── core_cache/              ← TrainingResult objects (pickle) for all 11 cores
│   ├── core_A_results.pkl
│   ├── core_B_results.pkl
│   └── ...
├── figures/
│   ├── figs_01_10/         ← Architecture, system diagrams
│   ├── figs_11_20/         ← Training curves, convergence, safety
│   ├── figs_21_30/         ← PV sweep, comm, robustness
│   ├── figs_31_40/         ← Ablation, hparam, scalability
│   ├── figs_41_50/         ← Protection engineering, interpretability
│   └── figs_51_58/         ← Transfer, compression, seasonal
├── benchmark_pv000.pkl     ← GA-SQP/HWOA/HHO/JAYA/EEO vs MARL at 0% PV
├── benchmark_pv020.pkl     ← ... at 20% PV
├── benchmark_pv040.pkl     ← ... at 40% PV
├── benchmark_pv060.pkl     ← ... at 60% PV
├── stat_comparison_pv000.txt  ← Friedman/Wilcoxon/Cliff's-delta at 0% PV
├── stat_comparison_pv020.txt
├── stat_comparison_pv040.txt
├── stat_comparison_pv060.txt
├── logs/
│   └── run_YYYYMMDD_HHMMSS.log
└── reports/
    └── final_report.json
```

---

## 4. Generating individual figures after training

All figures can be regenerated from cached training data without retraining:

```bash
# Regenerate a single figure
python run_thesis_experiments.py --fig 5

# Regenerate a range
python run_thesis_experiments.py --fig 5 6 7 8 12

# Regenerate all figures from cache
python run_thesis_experiments.py --figures-only

# List all figures and their sources
python run_thesis_experiments.py --list
```

---

## 5. Generating numerical tables

### 5.1 Objective 4 comparison table (main results table)

The primary numerical table — MARL vs. GA-SQP, HWOA, HHO, JAYA, EEO across
all three networks and four PV levels — is produced automatically during
Phase 2 and printed to stdout. To regenerate it from saved results:

```bash
python - << 'EOF'
import pickle
from benchmark_suite import BenchmarkSuite

for pv in [0, 20, 40, 60]:
    with open(f'results/benchmark_pv{pv:03d}.pkl', 'rb') as f:
        results = pickle.load(f)
    suite = BenchmarkSuite.__new__(BenchmarkSuite)
    suite.results = results
    print(f"\n=== PV Penetration: {pv}% ===")
    suite.print_comparison_table()
EOF
```

### 5.2 LaTeX table from TrainingPipeline results

```bash
python - << 'EOF'
import json
from results_aggregator import ResultsAggregator

with open('results/core_cache/phase5_results.json') as f:
    data = json.load(f)

agg = ResultsAggregator(data)
agg.compute_all_statistics()
agg.generate_latex_table('results/tables/main_results.tex')
agg.print_summary_table()
EOF
```

### 5.3 Statistical significance table

```bash
python - << 'EOF'
from evaluation.stat_analysis import run_benchmark_analysis
import pickle

# Load and merge all PV levels
transformed = {}
for pv in [0, 20, 40, 60]:
    with open(f'results/benchmark_pv{pv:03d}.pkl','rb') as f:
        suite_results = pickle.load(f)
    for sys_name, run_list in suite_results.items():
        key = f"{sys_name}_PV{pv:03d}"
        transformed[key] = {}
        for run_dict in run_list:
            for method, result in run_dict.items():
                of_val = result.get('OF') if isinstance(result, dict) else result
                if of_val is not None:
                    transformed[key].setdefault(method, []).append(of_val)

report = run_benchmark_analysis(transformed, metric='OF', verbose=True)
with open('results/tables/stat_significance.txt','w') as f:
    f.write(str(report))
print("Saved: results/tables/stat_significance.txt")
EOF
```

### 5.4 Formal novelty-claim validation (C1–C5)

`evaluation/thesis_validation.py`'s `ThesisValidator` is the formal
statistical validator for the thesis's 5 novelty claims (C1: MAPPO-Comm
beats baselines, C2: Safe RL achieves zero CTI violations, C3: EWC
prevents forgetting, C4: curriculum reduces sample complexity, C5:
scales to 39-bus with sub-100ms inference). It reads a flat JSON list of
per-run results from `outputs/phase5/phase5_results.json` — historically
only ever written by the separate, legacy `mains/main5.py --phase5`
entry point (never actually run in this project), **not** by
`run_thesis_experiments.py`'s Core A–K pipeline described in §2 above.

`experiments/core_to_phase5_bridge.py` bridges the two: it reads the
real results Core A–K + the Objective-4 benchmark comparison already
produce (`results/benchmark_pv*.pkl`, `results/core_cache/*.pkl`) and
writes them into `outputs/phase5/phase5_results.json` in the exact
shape `ThesisValidator` expects, so the claims can be validated from the
same run that produced the 58 figures — no separate, redundant
large-scale training pass required.

```bash
python -m experiments.core_to_phase5_bridge
# Writes outputs/phase5/phase5_results.json, then immediately runs
# ThesisValidator against it and prints which of C1-C5 pass/fail with
# the bridged data.
```

**This bridge does not fabricate data to fill coverage gaps** — the
Reproducibility Checklist's data-provenance standard (§2.2.1 above)
applies here too. Known structural gaps, left honestly absent rather
than invented:

- **Single-PPO** (needed by C1): not trained anywhere in the active
  pipeline.
- **C5's 39-bus inference latency / bandwidth**: Core I's latency
  measurement and Core G's bandwidth sweep are both 8-bus only; no Core
  measures either quantity on 39-bus.
- **C4's sample-efficiency ratio**: needs a per-checkpoint `history`
  array with a comparable logging cadence between a curriculum-trained
  and non-curriculum run; Core B (the only `curriculum=True` path) logs
  per-PV-stage, not per-episode like the other Cores/benchmark suite do,
  so this comparison is left unset rather than forced across mismatched
  granularities.

See the module's own docstring for the full method-name disambiguation
note (the same method name, e.g. "MAPPO-Safe-EWC", trains WITH GNN
communication in Core C but WITHOUT it in `benchmark_pv*.pkl` — the
bridge treats these as genuinely different configurations, not
interchangeable samples of "the same" method).

---

## 6. Standalone experiment scripts

These scripts can be run independently to regenerate specific figures
without going through the full pipeline. All accept `--system` and are
parameterised for the active 0/20/40/60% PV sweep.

```bash
# Fig 5 — PV sweep (OF vs PV level, all methods, one system)
python exp_fig05_pv_sweep_evaluation.py --system 8bus --n_seeds 10
python exp_fig05_pv_sweep_evaluation.py --system 14bus --n_seeds 10
python exp_fig05_pv_sweep_evaluation.py --system 39bus --n_seeds 10

# Fig 6 — CTI violation rate vs PV level
python exp_fig06_cti_violation_rate.py --system 8bus
python exp_fig06_cti_violation_rate.py --system 14bus
python exp_fig06_cti_violation_rate.py --system 39bus

# Fig 12 — Convergence curves (all methods, one system, at 60% PV)
python exp_fig12_convergence_curves.py --system 8bus  --n_runs 50
python exp_fig12_convergence_curves.py --system 14bus --n_runs 50
python exp_fig12_convergence_curves.py --system 39bus --n_runs 30

# Fig 39 — EWC ablation study (8-bus, 0→20→40→60% curriculum)
python exp_fig39_ablation.py --n_seeds 10
```

For a fast sanity check of any script, reduce seed count:

```bash
python exp_fig05_pv_sweep_evaluation.py --system 8bus --n_seeds 2 --train_steps 500
```

---

## 7. Manual figures (require human action)

Two figures have no generation script and must be created manually:

| Figure | Description | Recommended tool |
|--------|-------------|-----------------|
| Fig 26a | System architecture block diagram | draw.io / LaTeX TikZ |

Place the output PDF in `results/figures/figs_21_30/fig_26a.pdf` before
Phase 9 compiles the final figure count. (Fig 36 — action-space geometry
— used to require a hand-drawn illustration here too, but as of the
data-provenance update below it now reads real raw-vs-projected action
samples from Core A's cache and no longer needs manual placement.)

Phase 1 also runs scripts for Figs 1–4, 17, 19, 26–30, 34, 35, 51, 52
from `phase1_scripts/`. These are standalone diagram-generation scripts
that should be present in a `phase1_scripts/` subdirectory.

---

## 8. Missing file: `docl_config.py`

`run_thesis_experiments.py` requires `docl_config.py` to be present in the
same directory. The updated version (with the correct 60% PV max and
`experiment.seed` field) is provided in this session's output files.

Without it, the orchestrator prints a warning and uses a minimal fallback
config — training will still run but preset selection, YAML config files,
and CLI overrides (`--system`, `--n-runs`, etc.) will not work.

---

## 9. Troubleshooting

### "ModuleNotFoundError: No module named 'evaluation'"
```bash
mkdir -p evaluation && touch evaluation/__init__.py
cp stat_analysis.py evaluation/stat_analysis.py
```

### "ModuleNotFoundError: No module named 'environment.topology_manager'"
```bash
cp network/topology_manager.py environment/topology_manager.py
cp network/relay_physics.py    environment/relay_physics.py
```

### Phase 2 completes but benchmark_pv*.pkl files are empty or missing
The `BenchmarkSuite` import failed silently. Check the Phase 2 log for
"benchmark_suite not available" and resolve any remaining import errors
before re-running `--phase 2`.

### A figure fails to generate ("cached data not found")
The figure depends on training data from Phase 2. Either:
- Re-run Phase 2: `python run_thesis_experiments.py --phase 2`
- Or run the figure's standalone script directly (see §6)

### Results differ between runs
The PPO trainer (real PyTorch autograd + Adam as of 2026-10-22, see
§2.2.4 — previously OpenAI-ES style, numpy-only) is stochastic.
Run with `--n-runs 30` or higher for 8/14-bus, 10+ for 39-bus, and
report mean ± std across seeds. The `stat_analysis.py` Wilcoxon tests
and Cliff's delta already account for variance across seeds.

### 39-bus training is very slow
39-bus has 92 agents (rebuilt from the real topology — see "Topology
provenance" below; was 46 before 2026-09-09); each PPO update does one
forward+backward pass across all 92 agents' rollout data. Measured
directly: an env-step benchmark at 92 agents runs ~2.05x the per-step
cost of the old 46-agent graph. Reduce `--n-runs` to 10 for exploratory
runs.

### Topology provenance: 39bus's relay graph, and what changed on 2026-09-09

`get_39bus()` used to build its relay/pair/N-1 graph from an abbreviated
23-branch, 22-of-39-bus literal that was disconnected from the real
46-branch/39-bus topology (verified against MATPOWER case39.m: 34 lines
+ 12 transformers, 10 real generators) that `PandapowerNetwork` used for
actual AC power-flow solves. 17 of 39 buses — including all 10 real
generator buses (30-39) — had zero relay coverage in the abbreviated
graph, so Core G's "N-1 contingency on the 39-bus New England system"
sweep (Figs 13/14/20/22/24/54) never outaged a single branch connected
to a generator.

`get_39bus()` is now built from the real 46-branch topology directly:
92 relays, 86 pairs, full 39-of-39 bus coverage. The exact same
pair-construction algorithm already used at 57/118-bus scale is reused
unchanged — only its input branch list changed. `PandapowerNetwork`'s
power-flow behavior is bit-identical before/after (it was always reading
the real topology; only the relay/N-1 side was disconnected from it).

The old 23-branch graph is preserved as a separate factory,
`get_39bus_reduced()` (registered as `NETWORKS["39bus_reduced"]`, 46
relays / 42 pairs / 22-of-39 buses), so any already-published result
computed against it stays reproducible and citable as a named,
out-of-active-scope provenance variant — it is not deleted, just no
longer the default `"39bus"`.

**Known limitation, not fixed by this change:** 27 of the new 92 relays
end up with no backup pair (mostly the 9 radial generator step-up
terminals, which are legitimately backup-less, plus 18 relays orphaned
by an artifact of MATPOWER's arbitrary branch from/to direction
convention — the same class of limitation 14/57/118-bus already have,
just at a different ratio). This is pinned by a non-regression test
rather than silently allowed to drift.

**Relay indices are not comparable pre/post this change.** Six of the
23 previously-abbreviated branches had incorrect MVA limits relative to
the real case39 data, and the branch ordering (hence relay numbering)
differs entirely. Any per-relay or per-pair comparison against
pre-2026-09-09 39-bus results is meaningless; only system-level
aggregates (objective function, CTI violation rate, latency) are
comparable, and even those only loosely given the ~2x agent-count
change.

**Cache invalidation:** `NetworkData.topology_fingerprint()` (a stable
hash over bus/relay/pair/branch/DG-placement structure, including
`base_voltage`) is now folded into every Core's cache key for each
system it touches, so a stale pre-change cached policy can no longer
silently read back as a valid cache hit — it naturally misses instead.
`--cache-status` prints each entry's recorded `topology_fp` for audit.
This does **not** retroactively invalidate already-cached entries (that
requires either re-running with the same config, which will now miss
and retrain, or an explicit `CacheManager.invalidate(...)` call before a
retraining pass) — see the reproducibility checklist below.

Because the 8-bus fix in this same round (next subsection) also changed
`base_voltage` (11kV → 150kV), and `topology_fingerprint()` deliberately
includes `base_voltage`, **every Core that trains on 8-bus is also
affected** — not just the 39/14-bus-specific ones. In practice that's
9 of the 11 Cores (A, D, E, J, K are 8bus-only; F, G, H, I are
multi-system). A full retraining pass across all affected Cores is
tracked as a separate follow-up (GPU-heavy; not done in this round) —
this round's changes are data-layer-only, verified via direct
pandapower/topology smoke tests and the full test suite, not via
retraining.

### 8-bus reverse-power-flow anomaly: root cause and fix (2026-09-09)

`experiments/exp_reverse_power_flow.py` (Objective 1) previously reported
865-886% max line loading for 8-bus at all PV levels — traced to two
independent, compounding defects, both scoped to the fallback code path
`PandapowerNetwork._build_network()` took because `get_8bus()` had no
real `line_data`:
1. The fallback load-construction loop created one `pp.create_load` row
   **per relay** instead of per bus — any bus with multiple incident
   relays (buses 5/7/8 on 8bus) had its load summed 2-3x by pandapower.
   Fixed to aggregate by `max()` per bus (each relay's `Iload_max` rates
   that relay's own branch, not an additive independent load); this
   fallback path is shared by 3/8/30/57/118-bus, so all of them benefit.
2. `get_8bus()` had no real line impedance/rating data, so it fell back
   to a generic low-voltage cable type (0.27 kA) mismatched to its
   actual 150kV subtransmission currents. Real `line_data` (7 lines + 2
   step-up transformers, impedances/loads sourced from Rajput & Pandya,
   "On 8-Bus Test System for Solving Challenges in Relay Coordination,"
   IEEE 2016, Tables II-V — cross-referenced against this repo's
   existing `get_8bus()` relay/fault-current data, which already
   matches that source's Case-2 variant with a 400 MVA external grid at
   bus 4) is now used instead. `base_voltage` corrected 11.0 → 150.0 kV.
   The existing 14-relay/20-pair relay-coordination graph is left
   untouched (same established pattern as 39-bus's relay-graph/
   `line_data` split) — only the real power-flow topology changed.

   **Line thermal ratings (mva_limit) are not published in the cited
   sources** (they cover relay coordination, not thermal loading), so a
   disclosed engineering estimate (~156 MVA/line, from an assumed 600A
   150kV overhead-line ampacity) is used and flagged via
   `line_data["data_provenance"]` (`lines`/`loads`/`transformers` marked
   `cited_verified`, `mva_limit` marked `engineering_estimate`,
   `loading_percent_calibrated: False`). `exp_reverse_power_flow.py`'s
   JSON output now surfaces this via a per-system `model_fidelity`
   block, so a reader doesn't need to find a code comment to know 8-bus's
   loading percentages carry this caveat.

Result: 8-bus max line loading dropped from 885.7% to 68.96% (plausible
for a 150kV subtransmission system), with 4 real load rows (was 14
duplicated rows summing to 94.65 MW; now totals the real 240 MW system
load), converged power flow, min voltage 0.982 pu.

**The "zero reversals at any PV level, on any system" finding is real,
not a bug**, and is now a quantified, disclosed result rather than a
bare zero: all three systems size distributed PV at only ~10% of local
bus load, and the sweep only reaches 60% penetration, so net export past
any monitored line is mathematically out of reach at current sizing —
confirmed by the new `pv_to_load_ratio`/`per_dg_bus_pv_to_local_load`
fields in the JSON output (e.g. 39-bus's PV buses sit at 5.6-6.1% of
local load even at 60% penetration). The closest any line comes to
reversing is captured by the new `flow_ratio_vs_baseline`/
`min_flow_ratio`/`closest_to_reversal_line` fields (e.g. 39-bus line 18
falls to ~21% of its baseline flow at 60% PV). `PV_LEVELS` and DG
capacities were deliberately left unchanged to get this finding — do not
change them to try to force a reversal to appear; if demonstrating
reversal at higher penetration becomes a thesis requirement, that's a
separate, explicit scope decision, reported as an additional labelled
sensitivity arm alongside (never replacing) this calibrated result.

Separately, 14-bus's distributed PV was relocated off its default slack
bus (bus 1) onto real load buses — previously ~23 of its 27 MW of
"distributed PV" sat on the slack bus, invisibly absorbed by `ext_grid`,
making its PV sweep nearly inert (its most-affected line moved only
1.1% across the whole 0-60% sweep). This follows the same load-only-bus
pattern already used for 39-bus's PV placement. Like the 39/8-bus
changes above, this changes what 14-bus's cached MARL policies actually
observed and is covered by `topology_fingerprint()`'s DG-placement term.

`results/reverse_power_flow_analysis.json` has **not** been regenerated
as part of this round — see the reproducibility checklist below for
when/how to do that once the model changes are fully verified together.

Also newly documented: Phase E (`run_full_campaign.py`), which runs this
reverse-power-flow analysis once per full campaign, degrading gracefully
if pandapower is unavailable. Skip it with `--skip-reverse-power-flow`.

---

## 10. Reproducibility checklist

Before submitting results, verify:

- [ ] `python run_thesis_experiments.py --list` shows all 58 figures
- [ ] `results/benchmark_pv*.pkl` exists for all four PV levels
- [ ] `results/stat_comparison_pv*.txt` contains non-trivial Friedman p-values
- [ ] `results/figures/` contains PDFs for all auto-generated figures
- [ ] Fig 26a placed manually in the correct subdirectory (Fig 36 is now
      auto-generated from real data — see §7)
- [ ] Seed is fixed to 42 (default) for all runs, or varied and averaged
- [ ] System scope is exactly 8bus, 14bus, 39bus (not 30bus)
- [ ] PV sweep is exactly 0 %, 20 %, 40 %, 60 % (not 0–100 %)
- [ ] `exp_fig39_ablation.py`'s EWC ablation was run with curriculum=True
      to verify catastrophic-forgetting prevention claim
- [ ] After the 2026-09-09 topology fix (39bus rebuild, 8bus real
      line_data, 14bus PV relocation — see §9): confirm `--cache-status`
      shows a `topology_fp` recorded on every Core entry you rely on
      before trusting it as current; a pre-2026-09-09 cached entry with
      no `topology_fp` recorded is stale and should be retrained, not
      reused. Any Core touching 8bus (A, D, E, J, K, F, G, H, I — 9 of
      11) or 14bus/39bus (B, C, F, G, H, I) needs this check.
- [ ] `results/reverse_power_flow_analysis.json` should be regenerated
      (via `python -m experiments.exp_reverse_power_flow`) and diffed
      against its currently-committed version once the full model chain
      is verified together — 39bus's numbers should be unchanged to
      ~1e-9 (nothing in the 2026-09-09 round touches its power-flow
      model), 8bus/14bus should move to the plausible ranges documented
      in §9.

---

## 11. Running in Docker / on the cloud (2026-08-08)

The codebase runs entirely on CPU by default. As of 2026-10-22, training
is real PyTorch autograd + Adam (previously hand-rolled NumPy/SciPy MLPs
trained via evolution strategies — see §2.2.4), so `torch` is a required
dependency and the default image build below always installs its
CPU-only wheel. GPU is an explicit opt-in (`DOCR_USE_GPU=1`, §2.2.4,
§11.1a) — it does not auto-activate, because re-benchmarking the real
backprop update showed it is actually slower than CPU on this project's
test hardware. `Dockerfile`, `docker-compose.yml`, `.dockerignore`, and
`docker/entrypoint.sh` at the repo root package the whole pipeline.

### 11.1 Build

```bash
docker build -t docr-marl .
# or: docker compose build
```

Multi-stage build: a `builder` stage with `build-essential`/`gfortran`
installs `requirements.txt` into a venv; the final `runtime` stage is a
plain `python:3.11-slim` with just that venv and the app code copied in
— no compilers in the shipped image. Runs as a non-root user (`docr`).

### 11.1a GPU build (optional)

`torch` is **required** as of the 2026-10-22 ES-to-backprop rewrite (§2.2.4)
— training now runs real autograd + Adam, not hand-rolled NumPy ES — so the
default build above always installs it, via the CPU-only wheel
(`--index-url https://download.pytorch.org/whl/cpu`) to keep the image
small. If you're deploying to a machine with a GPU and want the
CUDA-bundled wheel instead:

```bash
docker build --build-arg INSTALL_TORCH_CUDA=true -t docr-marl-gpu .
docker run --gpus all --rm \
    -e DOCR_USE_GPU=1 \
    -v "$(pwd)/results:/app/results" -v "$(pwd)/outputs:/app/outputs" \
    docr-marl-gpu python3 run_thesis_experiments.py --all --preset thesis_sweep_fast
```

`--gpus all` requires the NVIDIA Container Toolkit installed on the host
(most GPU cloud VM images — AWS Deep Learning AMIs, GCP's GPU-optimized
images, etc. — already have it). The `DOCR_USE_GPU=1` env var is also
required at runtime — GPU does **not** auto-activate just because CUDA is
present (see §2.2.4): re-benchmarking the real-backprop update showed it is
actually *slower* than CPU on this project's test hardware, so the default
everywhere, container or not, is CPU. Without `DOCR_USE_GPU=1`, a GPU build
still runs correctly on CPU, same as the default build. Treat GPU timing as
untested on your own hardware until you've re-measured it per §2.2.4's
methodology — the slowdown was specific to a GTX 1050 and may not hold on a
more capable card.

### 11.2 Run

Every experiment script writes to `results/` and/or `outputs/` — mount
both as volumes so output survives past the container's life instead of
being lost when it's removed:

```bash
docker run --rm \
    -v "$(pwd)/results:/app/results" \
    -v "$(pwd)/outputs:/app/outputs" \
    docr-marl python3 run_thesis_experiments.py --all --preset thesis_sweep_fast

# Equivalently:
docker compose run --rm docr python3 run_thesis_experiments.py --all --preset thesis_sweep_fast

# Any of the individual scripts throughout this guide work the same way, e.g.:
docker compose run --rm docr python3 marl/experiments/exp_fig39_ablation.py --n_seeds 10 --n_workers 4
docker compose run --rm docr python3 -m pytest tests/ -q
```

`--n_workers N` (see §2.2.2) should match the *container's* visible CPU
count, not the host's, if you've capped the container with `--cpus`/
`deploy.resources.limits` — set N accordingly, or leave it unset and let
each script default to sequential.

**First run seeds the cache automatically, if there's anything to seed.**
`results/core_cache/` is where Core A–K results get cached (§2.2 Phase 2)
so later phases and figure generation don't retrain from scratch. It was
cleared on 2026-08-08 (see §2.2.1) and shipped empty for a while — Cores
have since been re-run, and as of this writing it holds 50 real cached
entries (~3.6MB). Bind-mounting an empty host `results/` directory over
`/app/results` would otherwise shadow whatever was baked into the image
at build time and silently force a full retrain. `docker/entrypoint.sh`
prevents that: on container start, if the mounted `results/core_cache/`
is empty, it's populated from whatever the image had baked in before your
command runs — confirmed working end-to-end (all 50 entries correctly
seeded into a fresh volume). The first time you actually run Core
training against a mounted volume, its output persists there and every
subsequent container start sees it — this is a no-op once your host's
`results/` directory has real content.

### 11.3 On a remote server / cloud VM

Nothing here is cloud-specific — this is a plain CPU container, so the
same `docker build`/`docker run` commands work unchanged over SSH on any
VM (AWS EC2, GCP Compute Engine, Azure VM, a bare-metal box, etc.):

```bash
# copy the repo to the remote host, then:
ssh user@remote-host 'cd DOCR_Research && docker compose build'
ssh user@remote-host 'cd DOCR_Research && docker compose run --rm -d docr \
    python3 run_thesis_experiments.py --all --preset thesis_sweep'
```

For managed batch/serverless runners (AWS ECS/Fargate or Batch, GCP Cloud
Run Jobs, Azure Container Instances), push the built image to that
provider's registry and point the job at it with a command override —
no GPU instance type is needed anywhere in this pipeline. Whichever
runner you use, mount or sync a persistent volume to `/app/results` (an
EBS volume, GCS/S3 fuse mount, etc.) so the Core cache and figures
survive past a single job's lifetime — a fresh, unmounted container
filesystem disappears with the job.

### 11.4 What's excluded from the image

`.dockerignore` keeps the build context small by excluding `.git/` and
all *generated* figure/report output under `results/` and `outputs/` —
everything there is regenerated at runtime into the mounted volumes.
`results/core_cache/` is the deliberate exception (see §11.2 above). The
`.dockerignore` entry excluding `papers/` (the unrelated USS v2.0
paper-drafts programme — see `CLAUDE.md`) is now moot: that directory and
its contents were deleted from the repo on 2026-08-08, so there's nothing
left for it to exclude. Left in place in case Programme 2 content ever
returns to this path.

### 11.5 Verified end-to-end (2026-10-15)

This Dockerfile was originally authored and its pieces individually
verified (a from-scratch `pip install -r requirements.txt` into a clean
venv, the full `pytest tests/` suite passing against it, the entrypoint's
seeding logic exercised standalone, and `docker compose config` validating
the compose file) in a sandbox whose network policy blocks pulls from
Docker Hub's CDN, so the actual `docker build` could not be executed
end-to-end at the time.

It has since been run for real -- `docker build`, then `docker run` against
a completely fresh named volume -- and three real bugs surfaced that no
amount of "individually verified pieces" caught, because each only shows
up in a genuinely fresh environment:

1. **`.dockerignore` didn't exclude `.venv/`** (or `venv/`/`env/`/`ENV/`),
   so a local virtual environment (1.6GB in the environment that found
   this) got included in every build context, massively slowing `docker
   build`'s context-transfer step for no benefit. Fixed by adding those
   patterns to `.dockerignore`.
2. **`docker/entrypoint.sh` had CRLF line endings** after a Windows
   checkout with `core.autocrlf=true` (no `.gitattributes` existed to
   override it for shell scripts), which breaks its `#!/bin/sh` shebang
   at container runtime with a misleading `exec: no such file or
   directory` error. Fixed by normalizing the file to LF and adding
   `.gitattributes` (`*.sh text eol=lf`) so this can't regress.
3. **`run_thesis_experiments.py` crashed on any completely fresh
   `results/` directory** -- its module-level `logging.FileHandler` opens
   a log file before `setup_directories()` (which creates `logs/`) ever
   runs, so a fresh clone, fresh container, or freshly bind-mounted empty
   volume all crashed with `FileNotFoundError` before printing even
   `--help`. This only stayed hidden because every local dev machine
   already had a populated `results/logs/` from prior runs. Fixed with an
   explicit `LOG_DIR.mkdir(parents=True, exist_ok=True)` before the
   handler is constructed.

With all three fixed, a full `docker build -t docr-marl .` followed by
`docker run --rm -v <fresh-volume>:/app/results -v <fresh-volume>:/app/outputs
docr-marl` succeeds cleanly: non-root `docr` user confirmed, all core
dependencies (`numpy`, `scipy`, `pandapower`, `shap`, `scikit-learn`, etc.)
import correctly, and the entrypoint correctly seeds all 50 of the baked-in
`core_cache` entries into the fresh volume.
