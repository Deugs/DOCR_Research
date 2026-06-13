# DOCR-MARL Merged Experiment Pipeline

## Overview

This is a **merged, cache-efficient** version of the DOCR-MARL thesis experiment suite that reduces redundant training runs by ~75%.

### Original Problem
- **58 figures** requiring training
- **~30+ redundant training sessions** (same configs re-trained for each figure)
- **~40 GPU-days** estimated compute time
- Figures 9, 10, 21, 37, 38 all trained MAPPO-Safe-EWC separately on 8-bus, 100% PV

### Merged Solution
- **6 core training runs** that cache all results
- **Post-processing scripts** generate figures from cached data (no re-training)
- **~8-10 GPU-days** estimated compute time
- All figures that share training data are generated from a single cached run

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
        │  (6 training runs)  │         │  (figure generation)    │
        └─────────────────────┘         └─────────────────────────┘
                    │                               ▲
                    │    ┌──────────────────────┐   │
                    └───►│  results/core_cache/ │◄──┘
                         │  (content-addressable) │
                         └──────────────────────┘
```

### Core Training Runs

| Core Run | Configuration | Feeds Figures | Est. Time |
|----------|--------------|---------------|-----------|
| **A** | MAPPO-Safe-EWC, 8-bus, 100% PV, 50 seeds | 9, 10, 12, 21, 37, 38, 41, 48-50, 55-57 | ~3-4 days |
| **B** | Curriculum 0%→100% PV, 10 seeds | 5, 6, 7, 16, 29, 39 | ~1-2 days |
| **C** | Baselines (4 methods), 50 seeds | 11, 12, 31-33, 55-56 | ~2-3 days |
| **D** | Ablation (5 configs), 10 seeds | 39 | ~0.5-1 day |
| **E** | Hyperparameter grid, 3 seeds × 80 configs | 40 | ~0.5-1 day |
| **F** | Scalability (5 systems) | 15 | ~0.5 day |

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

| Scenario | Training Runs | Post-Process | Total GPU-Days |
|----------|--------------|--------------|----------------|
| **Original** (naïve) | ~30 separate | Minimal | ~40 |
| **Merged** (this) | 6 core runs | ~15 min | **~8-10** |
| **Fast mode** | 6 core runs (reduced) | ~15 min | **~1-2 hours** |

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
