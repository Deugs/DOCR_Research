# DOCR-MARL Phase 2 CLI Adapters

## Overview

This package contains **46 CLI adapter scripts** that bridge `run_thesis_experiments.py` 
(Your master experiment launcher) with your actual experiment files (`exp_fig*.py`).

## Problem Solved

`run_thesis_experiments.py` calls 40+ scripts that **don't exist** in your codebase:
- `marl/curriculum_trainer.py` → calls `exp_fig09_cti_violation_dynamics.py`
- `marl/experiments/single_task_training.py` → calls `exp_fig09_cti_violation_dynamics.py`
- `marl/utils/lagrange_analysis.py` → calls `exp_fig10_lagrange_evolution.py`
- `marl/experiments/ewc_ablation.py` → calls `exp_fig16_ewc_effectiveness.py`
- `marl/utils/safety_projection_analysis.py` → calls `exp_fig21_action_projection.py`
- `marl/experiments/per_agent_curves.py` → calls `exp_fig37_per_agent_curves.py` (with use_comm fix)
- `marl/experiments/attention_evolution.py` → stub (no actual attention tracking)

Plus 33 additional stub adapters for all other missing scripts across Phases 1-9.

## Installation

### Option 1: Copy to Project Root (Recommended)

From your project root (`DOCR_Research/`):

```bash
# Copy all adapter directories to your project
xcopy /E /I "path\to\phase2_scripts_package\*" .
```

Or on Linux/Mac:
```bash
cp -r path/to/phase2_scripts_package/* /path/to/your/DOCR_Research/
```

### Option 2: Use as Standalone Package

Place the entire `phase2_scripts/` directory next to your `run_thesis_experiments.py` 
and modify the `sys.path` in `run_thesis_experiments.py` to include it.

## File Structure

```
phase2_scripts/              # Core Phase 3 adapters (7 files)
├── curriculum_trainer.py         → Fig 7 (maps to exp_fig09)
├── single_task_training.py       → Fig 9 (maps to exp_fig09)
├── lagrange_analysis.py          → Fig 10 (maps to exp_fig10)
├── ewc_ablation.py               → Fig 16 (maps to exp_fig16)
├── safety_projection_analysis.py → Fig 21 (maps to exp_fig21)
├── per_agent_curves.py           → Fig 37 (maps to exp_fig37 + use_comm fix)
├── attention_evolution.py        → Fig 38 (stub)
├── experiments/               # Symlinks/copies for run_thesis_experiments.py paths
│   ├── single_task_training.py
│   ├── ewc_ablation.py
│   ├── per_agent_curves.py
│   └── attention_evolution.py
└── utils/                     # Symlinks/copies for run_thesis_experiments.py paths
    ├── lagrange_analysis.py
    └── safety_projection_analysis.py

# Additional adapters for all other phases:
environment/experiments/     # Fig 2, 24, 36, 54
environment/utils/           # Fig 35
network/utils/               # Fig 3, 4, 43, 44, 46, 47, 50
data/standards/              # Fig 19
data/experiments/            # Fig 45
marl/experiments/            # Fig 29, 39, 40, 42
marl/utils/                  # Fig 34, 41
evaluation/experiments/      # Fig 13, 14, 15, 20, 22, 23, 25, 53, 54, 58
evaluation/utils/            # Fig 6, 11, 31, 32, 33, 55, 56, 57
baselines/experiments/       # Fig 5
deployment/                  # Fig 18
```

## Key Fix: use_comm=True Dimension Mismatch

The `per_agent_curves.py` adapter includes a critical patch for the `actor.sample()` 
dimension mismatch when `use_comm=True`:

```python
# When use_comm=True, sample() may return 3 elements: (action, log_prob, comm_msg)
# When use_comm=False, sample() returns 2 elements: (action, log_prob)
sample_result = trainer.actor.sample(obs[i], rng)
if isinstance(sample_result, (tuple, list)):
    if len(sample_result) >= 2:
        a = sample_result[0]
        log_prob = sample_result[1]
    else:
        a = sample_result[0]
        log_prob = None
else:
    a = sample_result
    log_prob = None
```

## Adapter Types

| Type | Count | Description |
|------|-------|-------------|
| **Real mapping** | 6 | Calls actual `exp_fig*.py` experiments |
| **Stub (synthetic)** | 33 | Generates placeholder visualizations |
| **Manual/conceptual** | 7 | Already handled by `run_thesis_experiments.py` logging |

## Usage

After installation, run your thesis experiments as normal:

```bash
# Test Phase 3 (training dynamics)
python run_thesis_experiments.py --phase3 --fast

# Test single figure
python run_thesis_experiments.py --figure 9 --fast

# Full suite (warning: very long)
python run_thesis_experiments.py --all --fast
```

## Adapter Categories

### Phase 1: System Architecture (Week 1)
- `environment/experiments/data_flow_logging.py` → Fig 2
- `network/utils/plot_topology.py` → Fig 3, 4
- `data/standards/frt_curves.py` → Fig 19

### Phase 2: Methodology (Week 1 cont.)
- `marl/experiments/ewc_visualization.py` → Fig 29
- `marl/utils/extract_architecture.py` → Fig 34
- `environment/utils/observation_breakdown.py` → Fig 35
- `environment/experiments/action_space_geometry.py` → Fig 36

### Phase 3: Training Dynamics (Week 4) ← CORE
- `phase2_scripts/curriculum_trainer.py` → Fig 7
- `phase2_scripts/single_task_training.py` → Fig 9
- `phase2_scripts/lagrange_analysis.py` → Fig 10
- `phase2_scripts/ewc_ablation.py` → Fig 16
- `phase2_scripts/safety_projection_analysis.py` → Fig 21
- `phase2_scripts/per_agent_curves.py` → Fig 37 (with use_comm fix)
- `phase2_scripts/attention_evolution.py` → Fig 38

### Phase 4: Performance Evaluation (Week 5)
- `baselines/experiments/pv_sweep_evaluation.py` → Fig 5
- `evaluation/utils/cti_analysis.py` → Fig 6

### Phase 5: Statistical Analysis (Week 9)
- `evaluation/utils/nemenyi_test.py` → Fig 11
- `evaluation/utils/multi_metric_ranking.py` → Fig 31
- `evaluation/utils/effect_size.py` → Fig 32
- `evaluation/utils/radar_chart.py` → Fig 33
- `evaluation/utils/comparison_table.py` → Fig 55
- `evaluation/utils/pareto_front.py` → Fig 56
- `evaluation/utils/bootstrap_convergence.py` → Fig 57

### Phase 6: Robustness & Resilience (Week 5)
- `evaluation/experiments/communication_stress.py` → Fig 13
- `evaluation/experiments/n1_contingency.py` → Fig 14
- `evaluation/experiments/byzantine_resilience.py` → Fig 20
- `evaluation/experiments/bandwidth_tradeoff.py` → Fig 22
- `environment/experiments/sensor_degradation.py` → Fig 24
- `environment/experiments/adversarial_robustness.py` → Fig 54

### Phase 7: Scalability & Transfer (Week 6)
- `evaluation/experiments/scalability_sweep.py` → Fig 15
- `environment/experiments/unseen_faults.py` → Fig 23
- `evaluation/experiments/zero_shot_transfer.py` → Fig 25

### Phase 8: Deployment (Week 7)
- `deployment/benchmark_latency.py` → Fig 18
- `evaluation/experiments/model_compression.py` → Fig 53

### Phase 9: Interpretability (Week 8)
- `evaluation/experiments/shap_analysis.py` → Fig 48
- `evaluation/experiments/decision_boundary.py` → Fig 49
- `network/utils/critical_pairs.py` → Fig 50
- `network/utils/coordination_diagram.py` → Fig 43
- `network/utils/fault_contribution.py` → Fig 44
- `data/experiments/daily_adaptive_response.py` → Fig 45
- `network/utils/relay_curves.py` → Fig 46
- `network/utils/sympathetic_trip_cases.py` → Fig 47

### Phase 10: Ablation & Sensitivity (Week 8 cont.)
- `marl/experiments/ablation_study.py` → Fig 39
- `marl/experiments/hyperparameter_grid.py` → Fig 40
- `marl/utils/sample_efficiency.py` → Fig 41
- `marl/experiments/warm_start.py` → Fig 42
- `evaluation/experiments/seasonal_generalization.py` → Fig 58

## Notes

- **Real adapters** (6 files) import and call your actual experiment scripts
- **Stub adapters** (33 files) generate synthetic but realistic-looking visualizations
- All adapters accept the same CLI arguments as `run_thesis_experiments.py` expects
- Stub adapters are marked with `[Adapter] STUB: Generating placeholder output...`
- You should replace stub adapters with real implementations as your codebase grows
