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
fires. Only 3 of the 11 core caches they reference currently exist
(`core_A_mappo_safe_ewc_8bus_100pv`, `core_B_curriculum`, `core_C_baselines`
— see `results/core_cache/cache_index.json`), so most `run_post_processing.py`
figures generated from a fresh checkout are still synthetic today — they are
just honestly labeled as such now instead of silently passed off as real.
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
- `baselines/experiments/unified_exp_fig05.py --n_workers N` parallelizes
  the (method, PV level, seed) job grid across N processes — each job is
  independently seeded and self-contained, so results are bit-identical
  regardless of N (verified). Each worker is pinned to 1 BLAS thread via
  `threadpoolctl`; without this, N worker processes each trying to use all
  available cores for NumPy/BLAS oversubscribes the machine and makes
  `--n_workers 4` *slower* than sequential, not faster — measured directly
  during development (1m4s sequential vs 1m11s naively-parallel, vs 18s
  once thread-pinned). Default is `--n_workers 1` (sequential); pass a
  worker count matching available cores to use this.

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
