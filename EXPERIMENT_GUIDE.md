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
  8bus:  14 relays, 20 pairs, 15 lines
  14bus: 20 relays, 26 pairs, 15 lines
  39bus: 46 relays, 42 pairs, 32 lines
```

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
fires. **Update (2026-08-08): `results/core_cache/` was cleared** (previously
3 of the 11 core caches existed — `core_A_mappo_safe_ewc_8bus_100pv`,
`core_B_curriculum`, `core_C_baselines`) and is now empty, so *every*
`run_post_processing.py` figure generated from the current checkout is
synthetic until Cores A–K are (re-)run. They are still honestly labeled as
such rather than silently passed off as real.
See the module docstring at the top of `run_post_processing.py` for the full
list of missing caches, and `tests/test_phase6.py` / `tests/test_phase7.py`
for regression coverage locking this behavior in.

Figures produced by `evaluation/utils/exp_fig06_cti_violation_rate.py` and the
repaired `baselines/experiments/unified_exp_fig05.py` (Fig 5, now including
GA-SQP/HWOA/HHO/JAYA/EEO with real Wilcoxon/Mann-Whitney + Cliff's delta
testing) are genuine — they train/evaluate real models when their Core cache
is present.

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

**Core A–D parallelization.** These are the four Cores that dominate Phase 2's
wall-clock cost (see the from-scratch time estimate below) — each runs
`n_runs` (or `n_runs × methods`/`n_runs × configs`) independent trainings
with no shared state between them, so they parallelize the same way the
standalone scripts in §2.2.2 do:

```bash
python run_core_training.py --all --n-workers 8       # match your machine's core count
python run_thesis_experiments.py --all --n-workers 8  # threads the same flag into Phase 2
```

Verified bit-identical output between `--n-workers 1` and `--n-workers 4` on
real (non-synthetic) Core A training, with a genuine 3.8x wall-clock
speedup on 4 workers (49.6s → 13.1s at small scale). As with every other
`--n-workers` flag in this repo: BLAS threads are pinned to 1 per worker via
`threadpoolctl` to avoid oversubscription, results are pre-allocated by
run/method/config index rather than appended in completion order, and the
`hash()`-based seed offsets Cores C and D used internally (non-deterministic
across processes — see §2.2.2's `zlib.crc32` note) were fixed the same way.
Default is `--n-workers 1` (sequential, matching prior behavior exactly).

From-scratch time estimate for Phase 2 at full scale (`thesis_sweep`
preset), measured directly on one CPU core rather than trusting this
codebase's own stale "~8-10 GPU-days" comment (there's no GPU dependency
anywhere in the *default* code path to make that unit meaningful) or
EXPERIMENT_GUIDE's previously-stated "1-3 hours" for the fast preset (that
undercounts too — measured fast-preset cost for Cores A-D alone is closer
to 7-8 hours single-core): **Cores A-D full-scale is roughly 40 days
single-core sequential**, dominated by Core C's baseline comparison (4
trainable methods × 50 runs × 5000 episodes). `--n-workers N` divides this
close to linearly (per the measured 3.8x-on-4-workers figure above) — on a
16-32 core cloud VM (see §11), this becomes on the order of days rather
than weeks.

**Optional GPU backend.** `marl/marl_agents.py`'s ES actor/critic update
(`_update_actor_es`/`_update_critic_es` — the same hot path vectorized in
§2.2.2) has a second, GPU-only implementation that additionally batches
across the 2×n_pert perturbation *population*, not just across the T-step
rollout, via `torch.einsum`. It activates automatically — nothing to
configure — when both `torch` is importable and `torch.cuda.is_available()`
is true; otherwise (including "torch is installed but no GPU," the common
case) every existing code path is exactly what it was before.

This is *not* also used as the CPU implementation, on purpose: benchmarked
directly, the same batched-population reformulation measured **slower**
than the existing per-perturbation loop on CPU (0.51x–0.68x — building the
`(T, 2×n_pert, hidden)` intermediate tensors costs more than the loop's
small per-perturbation BLAS calls save). This is the same category of
result as `marl/comm_graph.py`'s negative vectorization finding in
§2.2.2 — a batching strategy that looks like a strict improvement on paper
can lose on CPU once you account for allocation cost, so it isn't assumed
to help without measuring it.

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
| Fig 36 | Action-space geometry (hand-drawn illustration) | draw.io / Inkscape |

Place the output PDFs in `results/figures/figs_21_30/fig_26a.pdf` and
`results/figures/figs_31_40/fig_36.pdf` respectively before Phase 9
compiles the final figure count.

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
The ES-based trainer (OpenAI-ES style, numpy-only) is stochastic.
Run with `--n-runs 30` or higher for 8/14-bus, 10+ for 39-bus, and
report mean ± std across seeds. The `stat_analysis.py` Wilcoxon tests
and Cliff's delta already account for variance across seeds.

### 39-bus training is very slow
39-bus has 46 agents; each ES perturbation does 10×46 = 460 forward passes.
At ~24s/500 steps for 8-bus (14 agents), expect 39-bus to be roughly 3×
slower per step. Reduce `--n-runs` to 10 for exploratory runs.

---

## 10. Reproducibility checklist

Before submitting results, verify:

- [ ] `python run_thesis_experiments.py --list` shows all 58 figures
- [ ] `results/benchmark_pv*.pkl` exists for all four PV levels
- [ ] `results/stat_comparison_pv*.txt` contains non-trivial Friedman p-values
- [ ] `results/figures/` contains PDFs for all auto-generated figures
- [ ] Figs 26a and 36 placed manually in the correct subdirectories
- [ ] Seed is fixed to 42 (default) for all runs, or varied and averaged
- [ ] System scope is exactly 8bus, 14bus, 39bus (not 30bus)
- [ ] PV sweep is exactly 0 %, 20 %, 40 %, 60 % (not 0–100 %)
- [ ] `exp_fig39_ablation.py`'s EWC ablation was run with curriculum=True
      to verify catastrophic-forgetting prevention claim

---

## 11. Running in Docker / on the cloud (2026-08-08)

The codebase runs entirely on CPU by default (hand-rolled NumPy/SciPy MLPs
trained via evolution strategies) — no PyTorch/TensorFlow required — so any
machine that can run Docker can run these experiments. As of 2026-08-08
there's also an optional, auto-detected GPU backend (§2.2.4); the default
image build below deliberately does not include it (see §11.1a). `Dockerfile`,
`docker-compose.yml`, `.dockerignore`, and `docker/entrypoint.sh` at the
repo root package the whole pipeline.

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

The default build above skips `torch` even though it's listed in
`requirements.txt` — its CUDA-bundled wheel is large, and per §2.2.4 the
plain CPU path already benchmarks faster on this workload's problem sizes,
so most deployments gain nothing from paying that download cost. If you're
deploying to a machine that actually has a GPU and want
`marl/marl_agents.py`'s GPU backend to activate:

```bash
docker build --build-arg INSTALL_TORCH=true -t docr-marl-gpu .
docker run --gpus all --rm \
    -v "$(pwd)/results:/app/results" -v "$(pwd)/outputs:/app/outputs" \
    docr-marl-gpu python3 run_thesis_experiments.py --all --preset thesis_sweep_fast
```

`--gpus all` requires the NVIDIA Container Toolkit installed on the host
(most GPU cloud VM images — AWS Deep Learning AMIs, GCP's GPU-optimized
images, etc. — already have it). Without `--gpus all` at `docker run` time,
a GPU-build image still runs correctly — `torch.cuda.is_available()` just
comes back `False` inside the container and everything uses the CPU path,
same as the default build. As documented in §2.2.4, actual GPU speedup for
this workload was not benchmarked (no GPU was available while writing this)
— treat this as untested until you've measured it on your own hardware.

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
cleared on 2026-08-08 (see §2.2.1) and ships empty today — but the
mechanism stays in place for whenever it's populated again: bind-mounting
an empty host `results/` directory over `/app/results` would otherwise
shadow whatever was baked into the image at build time and silently force
a full retrain. `docker/entrypoint.sh` prevents that: on container start,
if the mounted `results/core_cache/` is empty, it's populated from
whatever the image had baked in (nothing, right now) before your command
runs. The first time you actually run Core training against a mounted
volume, its output persists there and every subsequent container start
sees it — this is a no-op once your host's `results/` directory has real
content.

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

### 11.5 Verifying this in your own environment

This Dockerfile was authored and its pieces individually verified
(a from-scratch `pip install -r requirements.txt` into a clean venv, the
full `pytest tests/` suite passing against it, the entrypoint's seeding
logic exercised standalone, and `docker compose config` validating the
compose file) in a sandbox whose network policy blocks pulls from Docker
Hub's CDN, so the actual `docker build` could not be executed end-to-end
here. Run `docker build -t docr-marl .` once in your own environment
before relying on it for a real cloud run.
