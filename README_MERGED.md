# DOCR-MARL Merged Experiment Pipeline

## Overview

This is a **merged, cache-efficient** version of the DOCR-MARL thesis experiment suite that reduces redundant training runs by ~75%.

### Original Problem
- **58 figures** requiring training
- **~30+ redundant training sessions** (same configs re-trained for each figure)
- Figures 9, 10, 21, 37, 38 all trained MAPPO-Safe-EWC separately on 8-bus, 100% PV

### Merged Solution
- **11 core training runs (A–K)** that cache all results — Cores A–F do
  real independent training and parallelize with `--n-workers`; Cores G–K
  are smaller and make a handful of evaluation calls each (see
  `EXPERIMENT_GUIDE.md` §2.2 for the full breakdown)
- **Post-processing scripts** generate figures from cached data (no re-training)
- All figures that share training data are generated from a single cached run

**On compute time:** this codebase is CPU-only by default (hand-rolled
NumPy MLPs trained via evolution strategies — no GPU dependency in the
default path; see `EXPERIMENT_GUIDE.md` §2.2.4 for the optional,
GPU-only-when-available backend and why it isn't used on CPU). Measured
directly rather than estimated: full-scale (`thesis_sweep` preset) Cores
A–D alone run **~40 CPU-days sequential** on one core; the fast preset
(`thesis_sweep_fast`) is **~7–8 hours sequential**. `--n-workers N`
divides this close to linearly — see the two new sections below for how
to actually run either scale.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         run_thesis_experiments.py                        │
│                    (Master orchestrator — updated)                       │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
        ┌─────────────────────┐         ┌─────────────────────────┐
        │  run_core_training  │         │  run_post_processing    │
        │  (11 training runs) │         │  (figure generation)    │
        └─────────────────────┘         └─────────────────────────┘
                    │                               ▲
                    │    ┌──────────────────────┐   │
                    └───►│  results/core_cache/ │◄──┘
                         │  (content-addressable) │
                         └──────────────────────┘
```

### Core Training Runs

Cores A–F do real, independent per-item training (parallelizable with
`--n-workers`); Cores G–K are smaller, each making 1–5 sequential
evaluation calls via `core_eval_bridge.py`. Full-scale (`thesis_sweep`)
run counts shown; the `thesis_sweep_fast` preset reduces each by roughly
5–25x (see the "Full thesis-scale run" section below for measured
wall-clock times, both sequential and parallelized):

| Core Run | Configuration | Feeds Figures |
|----------|--------------|---------------|
| **A** | MAPPO-Safe-EWC, 8-bus, 60% PV, 50 runs | 9, 10, 12, 21, 37, 38, 41, 48-50, 55-57 |
| **B** | Curriculum 0%→60% PV, 10 runs (with/without EWC) | 5, 6, 7, 16, 29, 39 |
| **C** | Baselines (5 methods incl. Traditional), 50 runs | 11, 12, 31-33, 55-56 |
| **D** | Ablation (5 configs), 10 runs | 39 |
| **E** | Hyperparameter grid, 80 configs × 3 seeds | 40 |
| **F** | Scalability (up to 7 systems) | 15 |
| **G–K** | Robustness, transfer, deployment, protection, seasonal — smaller, sequential | 13, 20, 22-25, 44-47, 53, 54 |

### Post-Processing Figures (No Training)

| Figure | Source Core Run | Generation Time |
|--------|----------------|-----------------|
| Fig 5 (Objective vs PV) | Core B | ~1 min |
| Fig 6 (CTI vs PV) | Core B | ~1 min |
| Fig 7 (Curriculum) | Core B | ~1 min |
| Fig 9 (CTI Dynamics) | Core A | ~1 min |
| Fig 10 (Lagrange) | Core A | ~1 min |
| Fig 11 (Nemenyi) | Core C | ~1 min |
| Fig 12 (Convergence) | Core A + C | ~1 min |
| Fig 15 (Scalability) | Core F | ~1 min |
| Fig 16 (EWC Effectiveness) | Core B | ~1 min |
| Fig 21 (Action Projection) | Core A | ~1 min |
| Fig 29 (EWC Continual) | Core B | ~1 min |
| Fig 37 (Per-Agent Curves) | Core A | ~1 min |
| Fig 38 (Attention Evolution) | Core A | ~1 min |
| Fig 39 (Ablation) | Core D | ~1 min |
| Fig 40 (Hyperparameter) | Core E | ~1 min |
| Fig 41 (Sample Efficiency) | Core A | ~1 min |

---

## Usage

### Full Suite
```bash
python run_thesis_experiments.py --all
```

### Fast Smoke Test
```bash
python run_thesis_experiments.py --all --fast
```

### Training Only (Cache Results)
```bash
python run_core_training.py --all
```

### Figures Only (From Cache)
```bash
python run_post_processing.py --all
```

### Individual Core Runs
```bash
python run_core_training.py --core-A          # MAPPO-Safe-EWC main run
python run_core_training.py --core-B          # Curriculum
python run_core_training.py --core-C          # Baselines
python run_core_training.py --core-D          # Ablation
python run_core_training.py --core-E          # Hyperparameter grid
python run_core_training.py --core-F          # Scalability
```

### Individual Figures
```bash
python run_post_processing.py --fig 9 10 12     # Selected figures
python run_post_processing.py --phase 2         # Phase 2 figures
python run_post_processing.py --list            # List available
```

### Cache Management
```bash
python run_core_training.py --cache-status      # Show cache entries
python run_core_training.py --clear-cache       # Clear all caches
```

---

## Running the Full Thesis-Scale Experiments on a PC

The active thesis scope (`thesis_sweep` preset: 8/14/39-bus, 0/20/40/60%
PV) is a genuinely large run — see Time Comparison below before starting.
Full step-by-step detail, troubleshooting, and per-phase reruns are in
`EXPERIMENT_GUIDE.md`; this is the condensed path.

**1. Install dependencies**
```bash
python --version   # 3.9+
pip install -r requirements.txt
# pandapower (already in requirements.txt) is optional — an analytical
# fallback is used automatically if it's absent, except for Fig 44.
```

**2. Verify the setup**
```bash
python -c "
from network.network_data import NETWORKS
from marl.marl_env import DynamicDOCREnv
from marl.marl_agents import MAPPOMARLTrainer
print('Setup OK — all imports resolved')
"
```

**3. Run a fast smoke test first (always do this before committing to full scale)**
```bash
python run_thesis_experiments.py --all --preset thesis_sweep_fast --n-workers N
```
Replace `N` with your machine's core count (check with `nproc` on
Linux/macOS or the Task Manager on Windows). This produces real
(non-synthetic) results at reduced scale in a few hours — see the Time
Comparison table above for what to expect.

**4. Run the full thesis-scale suite**
```bash
python run_thesis_experiments.py --all --preset thesis_sweep --n-workers N
```
This is the run that actually reproduces every figure at full statistical
power. At full scale, `--n-workers` is not optional in practice — without
it this is a multi-week single-core job (see Time Comparison). Even
parallelized across a typical desktop's cores, expect this to run for
multiple days; plan to leave it running unattended (a workstation you can
leave on, or see the Docker/cloud section below for something you don't
have to keep your own machine on for).

**5. If it gets interrupted**
Every Core's result is cached by a content-addressable key (see Cache
System below) as soon as that Core finishes — re-running the same command
picks up from the last completed Core rather than starting over. Use
`python run_core_training.py --cache-status` to see what's already done.

**6. Generate figures once training is cached**
```bash
python run_post_processing.py --all
```
This step alone is fast (~15 min) — it only reads from `results/core_cache/`.

---

## Running on the Cloud / Remote Server via Docker

The codebase runs entirely on CPU by default (see the Overview above), so
any cloud VM or remote server that can run Docker can run this pipeline —
nothing here is provider-specific. `Dockerfile`, `docker-compose.yml`, and
`docker/entrypoint.sh` at the repo root package the whole pipeline; full
detail (including the seed-cache mechanism and what's excluded from the
image) is in `EXPERIMENT_GUIDE.md` §11.

**1. Get the repo onto the remote host**
```bash
# from your machine:
scp -r . user@remote-host:~/DOCR_Research
# or, on the remote host directly:
git clone <your-fork-url> DOCR_Research && cd DOCR_Research
```

**2. Build the image on the remote host**
```bash
docker build -t docr-marl .
# GPU-capable build (see below for when this is worth it):
# docker build --build-arg INSTALL_TORCH=true -t docr-marl-gpu .
```

**3. Run it, with `results/`/`outputs/` mounted so output survives past the container's life**
```bash
docker run --rm \
    -v "$(pwd)/results:/app/results" \
    -v "$(pwd)/outputs:/app/outputs" \
    docr-marl python3 run_thesis_experiments.py --all --preset thesis_sweep --n-workers N
```
Replace `N` with the *container's* visible CPU count (matters if you've
capped it with `--cpus`). Since this is a multi-day job, run it detached
so it survives your SSH session ending:
```bash
docker run -d --rm --name docr-run \
    -v "$(pwd)/results:/app/results" -v "$(pwd)/outputs:/app/outputs" \
    docr-marl python3 run_thesis_experiments.py --all --preset thesis_sweep --n-workers N
docker logs -f docr-run   # reattach to watch progress; Ctrl-C to detach without stopping it
```
Or with `docker-compose.yml`: `docker compose run -d --rm docr python3 run_thesis_experiments.py --all --preset thesis_sweep --n-workers N`.

**4. GPU (optional)**
`marl/marl_agents.py` has an auto-detected GPU backend for its ES
actor/critic update (see `EXPERIMENT_GUIDE.md` §2.2.4). It only helps if
you built with `--build-arg INSTALL_TORCH=true` **and** run with
`--gpus all` on a host with NVIDIA drivers + the NVIDIA Container Toolkit
installed (most cloud GPU VM images already have this). Read §2.2.4
before reaching for this — the same batched approach measured *slower*
than the default CPU path when benchmarked, and actual GPU speedup for
this specific workload hasn't been verified on real hardware.

**5. Picking a cloud VM**
Since the workload is CPU-bound and each Core's `n_runs` are independent,
prioritize **core count** over any other spec — a 16-32 vCPU instance with
`--n-workers` matching that count is the most direct way to compress the
~40 CPU-day full-scale estimate down to a few days. No GPU instance type
is needed unless you're specifically testing the optional GPU backend
above.

---

## Cache System

The cache uses **content-addressable storage**:

```python
# Cache key = SHA256(config_dict)
config = {
    "name": "core_A_mappo_safe_ewc_8bus_100pv",
    "method": "MAPPO-Safe-EWC",
    "system": "8bus",
    "pv": 1.0,
    "n_runs": 50,
    "n_episodes": 5000,
    "seed": 42,
}
# key = "a3f7e2d9..." (first 16 chars)
# Stored at: results/core_cache/a3f7e2d9.pkl
```

Changing any parameter (even `seed`) creates a new cache entry. This ensures reproducibility while avoiding redundant computation.

---

## Data Flow

### Figure 9 Example
```
Core A Training → UnifiedTrainingLogger
    ├── cti_violation_rates[]  ──► Fig 9 (Panel 1)
    ├── lagrange_norms[]       ──► Fig 9 (Panel 2) + Fig 10
    ├── cumulative_violations[] ──► Fig 9 (Panel 3)
    ├── safety_activations[]   ──► Fig 9 (Panel 4) + Fig 21
    ├── per_agent_returns[]    ──► Fig 37
    └── attention_snapshots{}  ──► Fig 38
```

### Figure 5 Example
```
Core B Curriculum Training
    ├── task_evaluations[0.0]  ──┐
    ├── task_evaluations[0.25] ──┼──► Fig 5 (objective vs PV)
    ├── task_evaluations[0.5]  ──┤    Fig 6 (CTI vs PV)
    ├── task_evaluations[0.75] ──┤
    └── task_evaluations[1.0]  ──┘
```

---

## File Structure

```
project/
├── run_thesis_experiments.py      # Master orchestrator (updated)
├── run_core_training.py           # NEW: Unified training launcher
├── run_post_processing.py         # NEW: Figure generator
│
├── results/
│   ├── core_cache/                # Cached training results
│   │   ├── cache_index.json       # Cache metadata
│   │   ├── a3f7e2d9.pkl          # Core A results
│   │   ├── b8e1c4f2.pkl          # Core B results
│   │   └── ...
│   ├── figures/                   # Generated figures
│   │   ├── figs_01_10/
│   │   ├── figs_11_20/
│   │   └── ...
│   └── logs/                      # Training logs
│
├── marl/experiments/              # Original experiment scripts
│   ├── exp_fig09_cti_violation_dynamics.py
│   ├── exp_fig12_convergence_curves.py
│   └── ...
│
└── network/                       # Power system models
    └── ...
```

---

## Time Comparison

Measured directly (see `EXPERIMENT_GUIDE.md` §2.2.4), single CPU core,
sequential — no GPU dependency in the default path:

| Scenario | Cores A-D alone | With `--n-workers 4` |
|----------|-----------------|----------------------|
| **Full scale** (`thesis_sweep`) | ~40 CPU-days | ~10+ days |
| **Fast mode** (`thesis_sweep_fast`) | ~7-8 hours | ~2-4 hours |

---

## Migration Guide

### From Original Scripts

If you have existing experiment scripts (`exp_fig*.py`), they remain compatible. The merged system:

1. **Imports** your existing `exp_fig*.py` modules when available
2. **Falls back** to synthetic data when imports fail (for testing)
3. **Wraps** your training loops with `UnifiedTrainingLogger` to capture all metrics

### Adding a New Figure

1. Add metric capture to `UnifiedTrainingLogger` in `run_core_training.py`
2. Add figure generation function to `run_post_processing.py`
3. Map figure number in `FIGURE_FUNCTIONS` dict

```python
# In run_post_processing.py
def fig_99_new_figure():
    data = load_cache("core_A_mappo_safe_ewc_8bus_100pv")
    # ... generate figure from cached data ...
    save_fig(fig, FIGURES_DIR / "figs_51_58" / "fig_99_new_figure.pdf")

FIGURE_FUNCTIONS[99] = fig_99_new_figure
```

---

## Troubleshooting

### Cache Misses
If cache isn't being hit, check:
```bash
python run_core_training.py --cache-status
```
Ensure config dicts are identical (including parameter order — use `sort_keys=True`).

### Missing Imports
The system gracefully falls back to synthetic data when `network.network_data` or `marl.marl_agents` aren't available. For production, ensure your project structure is on `PYTHONPATH`.

### Memory
Core A with 50 runs × 5000 episodes stores ~500MB of cached data. Adjust `n_runs` if memory-constrained.

---

## Citation

If using this merged pipeline, reference the original thesis:

```bibtex
@thesis{docrl_marl_2026,
  title={Multi-Agent Reinforcement Learning for Adaptive Protection Coordination in PV-Rich Distribution Networks},
  author={[Your Name]},
  year={2026},
  institution={[Your Institution]},
  note={Merged experiment pipeline — 75\% compute reduction via unified core training + post-processing}
}
```
