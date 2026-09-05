# DOCR-MARL Training Pipeline

## Overview

The `training_pipeline.py` module orchestrates large-scale experiments for **Directional Overcurrent Relay Coordination using Multi-Agent Reinforcement Learning (DOCR-MARL)**. It supports systematic evaluation across multiple test systems, PV penetration levels, fault types, and algorithms — with full data collection for all 17 thesis figures.

---

## Table of Contents

1. [Architecture](#architecture)
2. [Core Classes](#core-classes)
3. [Figure-Specific Evaluations](#figure-specific-evaluations)
4. [Experiment Matrices](#experiment-matrices)
5. [Result Analysis](#result-analysis)
6. [Configuration Presets](#configuration-presets)
7. [Usage Examples](#usage-examples)
8. [Output Structure](#output-structure)
9. [Checkpoint & Recovery](#checkpoint--recovery)
10. [API Reference](#api-reference)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    TrainingPipeline                             │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐   │
│  │  Part 1-2   │  │   Part 3-4  │  │     Part 5-10       │   │
│  │  Core Infra │→ │ Figure Eval │→ │  Analysis & Utils   │   │
│  │             │  │             │  │                     │   │
│  │ • Config    │  │ • Comm      │  │ • Matrix Builders │   │
│  │ • Result    │  │ • N-1       │  │ • Aggregation     │   │
│  │ • Pipeline  │  │ • Byzantine │  │ • Presets         │   │
│  │ • Dispatch  │  │ • ...       │  │ • Logging         │   │
│  └─────────────┘  └─────────────┘  └─────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Core Classes

### `TrainingConfig`

Extended configuration dataclass supporting all 17 figure-specific parameters.

```python
from experiments.training_pipeline import TrainingConfig

config = TrainingConfig(
    method="MAPPO-Comm",      # Algorithm
    system="14bus",           # Test system
    pv_level=0.5,             # PV penetration (0.0–1.0)
    fault_type="3ph",         # Fault type
    n_runs=10,                # Independent runs
    seed_offset=0,            # Random seed base

    # Figure-specific (opt-in)
    comm_stress_level=0.3,   # Fig 13: stress level
    n1_line_outage=2,        # Fig 14: line to trip
    byzantine_agents=[1, 3], # Fig 20: malicious agents
    # ... (40+ fields total)
)
```

**Core Fields:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `method` | `str` | — | Algorithm: `GA-SQP`, `MAPPO`, `MAPPO-Comm`, `MAPPO-Safe`, `MAPPO-Safe-EWC`, `HWOA`, `HHO`, `JAYA`, `EEO`, `Single-PPO` |
| `system` | `str` | — | Test system: `8bus`, `14bus`, `39bus` |
| `pv_level` | `float` | — | PV penetration: `0.0`, `0.2`, `0.4`, `0.6` |
| `fault_type` | `str` | — | Fault: `3ph`, `LL`, `1ph`, `2ph-g` |
| `n_runs` | `int` | `10` | Independent runs for statistical significance |
| `ga_pop` | `int` | `100` | GA population size |
| `ga_gen` | `int` | `200` | GA generations |
| `meta_pop` | `int` | `30` | Metaheuristic population |
| `meta_iter` | `int` | `500` | Metaheuristic iterations |
| `marl_steps` | `int` | `30000` | MARL training steps |
| `use_curriculum` | `bool` | `False` | Enable curriculum learning |
| `use_safe_rl` | `bool` | `False` | Enable Safe RL constraints |
| `use_ewc` | `bool` | `False` | Enable Elastic Weight Consolidation |
| `use_comm` | `bool` | `False` | Enable GNN communication |

**Figure-Specific Fields:**

| Figure | Field | Type | Description |
|--------|-------|------|-------------|
| **13** | `comm_stress_level` | `float` | Stress severity (0.0–1.0) |
| **13** | `packet_loss_rate` | `float` | Message drop probability |
| **13** | `latency_ms` | `float` | Artificial delay (ms) |
| **13** | `quantization_bits` | `int` | Message precision (32/16/8/4) |
| **14** | `n1_line_outage` | `Optional[int]` | Line index to trip |
| **14** | `n1_recovery_eval` | `bool` | Evaluate post-recovery |
| **20** | `byzantine_agents` | `List[int]` | Malicious agent IDs |
| **20** | `byzantine_strategy` | `str` | `random`/`reverse`/`colluding` |
| **22** | `gnn_depth` | `int` | GNN layers (1–4) |
| **22** | `compression_ratio` | `float` | Parameter compression |
| **22** | `message_dim` | `int` | Communication dimension |
| **23** | `fault_impedance_ohm` | `float` | High-Z fault impedance |
| **23** | `evolving_fault` | `bool` | Time-varying impedance |
| **24** | `sensor_noise_std` | `float` | Observation noise σ |
| **24** | `missing_sensor_rate` | `float` | Sensor dropout probability |
| **25** | `eval_system` | `Optional[str]` | Target system for transfer |
| **36** | `action_space_grid` | `bool` | Enable TDS/PSM grid search |
| **36** | `grid_tds_range` | `Tuple[float,float]` | TDS search bounds |
| **36** | `grid_psm_range` | `Tuple[float,float]` | PSM search bounds |
| **36** | `grid_resolution` | `int` | Grid points per axis |
| **44** | `run_shortcircuit_study` | `bool` | Run pandapower SC |
| **45** | `daily_profile` | `bool` | 24-hour irradiance sim |
| **46** | `relay_curve_eval` | `bool` | Compute t_op vs I_f |
| **47** | `sympathetic_tripping_eval` | `bool` | Check miscoordination |
| **48** | `shap_eval` | `bool` | Run SHAP analysis |
| **49** | `tsne_eval` | `bool` | Collect obs/action pairs |
| **50** | `critical_pair_eval` | `bool` | Extract λ multipliers |
| **53** | `quantize_bits` | `Optional[int]` | Quantization precision |
| **53** | `prune_ratio` | `float` | Weight pruning fraction |
| **54** | `adversarial_attack` | `Optional[str]` | `FGSM`/`PGD`/`SPSA` |
| **54** | `attack_epsilon` | `float` | Perturbation magnitude |
| **58** | `season` | `Optional[str]` | `spring`/`summer`/`autumn`/`winter` |

---

### `TrainingResult`

Extended result dataclass with figure-specific outputs.

```python
result = TrainingResult(
    config=config,
    run_id=0,
    best_of=12.34,           # Best objective function value
    best_tms=np.array([...]), # Best TMS settings
    best_ps=np.array([...]),  # Best PS settings
    cti_satisfied=True,       # CTI constraints met
    cti_violations=0,       # Number of violations
    cti_min=0.35,            # Minimum CTI margin
    elapsed_seconds=45.2,
    history=[15.2, 14.1, ...],

    # Figure-specific (populated by eval methods)
    comm_stress_of=13.5,    # Fig 13
    n1_of=14.2,             # Fig 14
    # ... (30+ fields)
)
```

---

### `TrainingPipeline`

Main execution engine.

```python
from experiments.training_pipeline import TrainingPipeline

pipeline = TrainingPipeline(
    configs=[config1, config2, ...],
    n_workers=4,                    # Parallel workers (-1 = auto)
    checkpoint_dir="outputs/checkpoints",
    results_dir="outputs/results",
    verbose=True,
)

# Run all experiments
results = pipeline.run_all()

# Save results
pipeline.save_results("outputs/results/my_experiment.json")
pipeline.save_results_by_figure()  # Auto-generates fig*.json files
```

**Key Methods:**

| Method | Description |
|--------|-------------|
| `run_all()` | Execute all configurations with checkpoint/resume |
| `save_results(filepath)` | Save all results to single JSON |
| `save_results_by_figure()` | Save organized per-figure JSON files |

---

## Figure-Specific Evaluations

Each figure has a dedicated `_eval_*` method that runs **only when the corresponding config field is non-default** (opt-in design).

| Figure | Method | Trigger Condition | Output Fields |
|--------|--------|-------------------|---------------|
| **13** | `_eval_comm_stress` | `comm_stress_level > 0` or `packet_loss_rate > 0` | `comm_stress_of`, `comm_stress_cti_ok`, `comm_stress_latency` |
| **14** | `_eval_n1_contingency` | `n1_line_outage is not None` | `n1_of`, `n1_cti_ok`, `n1_recovery_of`, `n1_line_index` |
| **20** | `_eval_byzantine` | `len(byzantine_agents) > 0` | `byzantine_of`, `byzantine_detected`, `byzantine_trust_scores` |
| **22** | `_eval_bandwidth` | `compression_eval=True` or `gnn_depth != 2` or `compression_ratio != 1.0` | `bandwidth_kbps`, `bandwidth_of`, `compression_params` |
| **23** | `_eval_unseen_faults` | `unseen_fault_eval=True` or `fault_impedance_ohm > 0` | `unseen_fault_of`, `unseen_fault_cti_ok`, `fault_impedance` |
| **24** | `_eval_sensor_robustness` | `sensor_noise_std > 0` or `missing_sensor_rate > 0` | `noisy_of`, `noisy_cti_ok`, `sensor_snr` |
| **25** | `_eval_zero_shot_transfer` | `eval_system != system` | `transfer_of`, `transfer_cti_ok`, `transfer_system` |
| **36** | `_eval_action_space_grid` | `action_space_grid=True` | `action_space_grid_data` (dict with tds/psm grids) |
| **44** | `_eval_fault_contributions` | `run_shortcircuit_study=True` | `fault_contributions` (dict: bus → %) |
| **45** | `_eval_daily_pv` | `daily_profile=True` | `daily_tms_trajectory`, `daily_ps_trajectory`, `daily_of_trajectory`, `daily_irradiance` |
| **46** | `_eval_relay_curves` | `relay_curve_eval=True` | `relay_curves` (dict per relay: If_range, t_op) |
| **47** | `_eval_sympathetic_tripping` | `sympathetic_tripping_eval=True` | `sympathetic_trips`, `traditional_trips` |
| **48** | `_eval_shap` | `shap_eval=True` | `shap_values`, `shap_base_value` |
| **49** | `_eval_tsne` | `tsne_eval=True` | `tsne_observations`, `tsne_actions`, `tsne_embeddings` |
| **50** | `_eval_critical_pairs` | `critical_pair_eval=True` | `critical_pairs` (list of dicts with λ, violations) |
| **53** | `_eval_compression` | `compression_eval=True` or `quantize_bits is not None` or `prune_ratio > 0` | `compressed_of`, `compressed_size_kb`, `compression_accuracy_drop` |
| **54** | `_eval_adversarial` | `adversarial_eval=True` or `adversarial_attack is not None` | `adversarial_of`, `adversarial_success_rate`, `attack_perturbation_norm` |
| **58** | `_eval_seasonal` | `season is not None` | `seasonal_of`, `seasonal_cti_ok` |

---

## Experiment Matrices

### Base Matrix Builder

```python
from experiments.training_pipeline import create_experiment_matrix

configs = create_experiment_matrix(
    systems=["8bus", "14bus"],
    methods=["GA-SQP", "MAPPO", "MAPPO-Comm", "MAPPO-Safe"],
    pv_levels=[0.0, 0.2, 0.4, 0.6],
    fault_types=["3ph"],
    n_runs=10,
    fast=False,  # True = reduced params for quick testing
)
```

### Figure-Specific Matrix Builders

```python
from experiments.training_pipeline import (
    create_fig13_comm_stress_matrix,
    create_fig14_n1_contingency_matrix,
    # ... etc
)

# Fig 13: Communication stress sweep
fig13_configs = create_fig13_comm_stress_matrix(
    base_configs=base_configs,
    stress_levels=[0.0, 0.3, 0.6, 0.9],
    packet_loss_rates=[0.0, 0.1, 0.3, 0.5],
    quantization_bits=[32, 16, 8, 4],
)

# Fig 14: N-1 contingency sweep
fig14_configs = create_fig14_n1_contingency_matrix(
    base_configs=base_configs,
    line_indices=[0, 1, 2, 3, 4, 5],
    include_recovery=True,
)

# Fig 25: Zero-shot transfer
fig25_configs = create_fig25_zero_shot_matrix(
    base_configs=base_configs,
    train_systems=["8bus"],
    eval_systems=["14bus", "39bus"],
)
```

### Master Suite Builder

```python
from experiments.training_pipeline import create_full_experiment_suite

suite = create_full_experiment_suite(
    systems=["8bus", "14bus"],
    methods=["MAPPO", "MAPPO-Comm", "MAPPO-Safe"],
    pv_levels=[0.0, 0.3, 0.6],
    fault_types=["3ph"],
    n_runs=10,
    fast=False,
    figures=["base", "13", "14", "22", "24", "25", "58"],  # Select subset
)

# Flatten for execution
all_configs = []
for fig_name, configs in suite.items():
    all_configs.extend(configs)
```

---

## Result Analysis

### `ResultAnalyzer`

```python
from experiments.training_pipeline import ResultAnalyzer

analyzer = ResultAnalyzer(results)

# Best result overall
best = analyzer.get_best_result()
print(f"Best OF: {best.best_of:.4f}")

# Best by method/system
best_mappo = analyzer.get_best_result(method="MAPPO", system="14bus")

# Pareto front (multi-objective)
pareto = analyzer.get_pareto_front(
    objectives=[("best_of", -1), ("cti_min", 1)]
)

# Statistical comparison (Wilcoxon rank-sum)
comparison = analyzer.compare_methods(metric="best_of")
for method, stats in comparison.items():
    print(f"{method}: {stats['mean']:.4f} ± {stats['std']:.4f}")

# Figure summary
summary = analyzer.figure_summary("fig13")
for stress_level, stats in summary.items():
    print(f"Stress {stress_level}: OF={stats['mean']:.4f} ± {stats['std']:.4f}")

# Export LaTeX table
analyzer.export_latex_table("outputs/table.tex")
```

### Aggregation Functions

```python
from experiments.training_pipeline import (
    aggregate_by_method,
    aggregate_by_pv_level,
    aggregate_by_figure,
)

# By method
method_stats = aggregate_by_method(results)

# By PV level
pv_stats = aggregate_by_pv_level(results)

# By any figure field
stress_stats = aggregate_by_figure(results, "comm_stress_level")
```

---

## Configuration Presets

### `ExperimentPresets`

One-line setup for common experimental scenarios.

```python
from experiments.training_pipeline import ExperimentPresets

# Thesis Chapter 5: Core validation
configs = ExperimentPresets.thesis_chapter5_validation(fast=False)

# Thesis Chapter 6: Robustness (returns dict)
suite = ExperimentPresets.thesis_chapter6_robustness(fast=False)
# suite["base"], suite["fig14_n1"], suite["fig13_comm"], etc.

# Thesis Chapter 7: Transfer learning
suite = ExperimentPresets.thesis_chapter7_transfer(fast=False)

# IEEE paper submission
suite = ExperimentPresets.ieee_paper_experiments(fast=False)

# Quick demo (1-2 minutes)
configs = ExperimentPresets.quick_demo()
```

---

## Usage Examples

### Example 1: Basic Usage

```python
from experiments.training_pipeline import (
    create_experiment_matrix, TrainingPipeline
)

configs = create_experiment_matrix(
    systems=["8bus"],
    methods=["GA-SQP", "JAYA"],
    pv_levels=[0.0, 0.6],
    n_runs=2,
    fast=True,
)

pipeline = TrainingPipeline(configs, n_workers=1)
results = pipeline.run_all()
pipeline.save_results("outputs/results/demo.json")
```

### Example 2: Full Suite with Figures

```python
from experiments.training_pipeline import (
    create_full_experiment_suite, TrainingPipeline
)

suite = create_full_experiment_suite(
    systems=["8bus", "14bus"],
    methods=["MAPPO", "MAPPO-Comm", "MAPPO-Safe"],
    pv_levels=[0.0, 0.3, 0.6],
    figures=["base", "13", "14", "22", "24", "25"],
    fast=True,
)

all_configs = [c for cfg_list in suite.values() for c in cfg_list]
pipeline = TrainingPipeline(all_configs, n_workers=4)
results = pipeline.run_all()
pipeline.save_results_by_figure()  # → fig13_comm_stress.json, etc.
```

### Example 3: Thesis Chapter with Analysis

```python
from experiments.training_pipeline import (
    ExperimentPresets, TrainingPipeline, ResultAnalyzer
)

suite = ExperimentPresets.thesis_chapter6_robustness(fast=False)
all_configs = [c for cfg_list in suite.values() for c in cfg_list]

pipeline = TrainingPipeline(all_configs, n_workers=8)
results = pipeline.run_all()

analyzer = ResultAnalyzer(results)
analyzer.export_latex_table("outputs/chapter6_table.tex")

for fig in ["fig13", "fig14", "fig24"]:
    summary = analyzer.figure_summary(fig)
    print(f"{fig}: {len(summary)} parameter groups")
```

### Example 4: Resume Interrupted Experiment

```python
from experiments.training_pipeline import (
    ExperimentRecovery, ExperimentPresets, TrainingPipeline
)

# Check for incomplete runs
interrupted = ExperimentRecovery.find_incomplete_runs()
print(f"Interrupted: {len(interrupted)}")

# Re-run — checkpoint automatically skips completed configs
configs = ExperimentPresets.quick_demo()
pipeline = TrainingPipeline(configs, n_workers=1)
results = pipeline.run_all()  # Resumes automatically

# Merge all result files
merged = ExperimentRecovery.merge_result_files()
```

### Example 5: Custom Parameter Sweep

```python
from experiments.training_pipeline import (
    create_experiment_matrix, create_fig22_bandwidth_matrix,
    TrainingPipeline, ResultAnalyzer
)

base = create_experiment_matrix(
    systems=["8bus"],
    methods=["MAPPO-Comm"],
    pv_levels=[0.5],
    n_runs=3,
    fast=True,
)

configs = create_fig22_bandwidth_matrix(
    base,
    gnn_depths=[1, 2, 3],
    compression_ratios=[1.0, 0.5, 0.1],
    message_dims=[2, 4, 8, 16],
)

pipeline = TrainingPipeline(configs, n_workers=2)
results = pipeline.run_all()

analyzer = ResultAnalyzer(results)
summary = analyzer.figure_summary("fig22")
for ratio, stats in sorted(summary.items()):
    print(f"Compression {ratio}: OF={stats['mean']:.4f} ± {stats['std']:.4f}")
```

---

## Output Structure

```
outputs/
├── checkpoints/
│   └── pipeline_checkpoint.json          # Resume state
├── results/
│   ├── phase5_results.json              # Master results file
│   ├── base_quick_test.json             # Custom named output
│   ├── fig13_comm_stress.json           # Fig 13 data
│   ├── fig14_n1_contingency.json        # Fig 14 data
│   ├── fig20_byzantine.json             # Fig 20 data
│   ├── fig22_bandwidth.json             # Fig 22 data
│   ├── fig23_unseen_faults.json         # Fig 23 data
│   ├── fig24_sensor_robustness.json     # Fig 24 data
│   ├── fig25_zero_shot.json             # Fig 25 data
│   ├── fig36_action_space.json          # Fig 36 data
│   ├── fig44_fault_contrib.json         # Fig 44 data
│   ├── fig45_daily_pv.json              # Fig 45 data
│   ├── fig46_relay_curves.json          # Fig 46 data
│   ├── fig47_sympathetic.json           # Fig 47 data
│   ├── fig48_shap.json                  # Fig 48 data
│   ├── fig49_tsne.json                  # Fig 49 data
│   ├── fig50_critical_pairs.json        # Fig 50 data
│   ├── fig53_compression.json           # Fig 53 data
│   ├── fig54_adversarial.json           # Fig 54 data
│   ├── fig58_seasonal.json              # Fig 58 data
│   ├── chapter6_table.tex               # LaTeX table export
│   └── merged_results.json              # Deduplicated merge
└── logs/
    └── experiment.log                   # Structured event log
```

---

## Checkpoint & Recovery

### Automatic Checkpointing

The pipeline auto-saves checkpoint after **every completed configuration**:

```json
// outputs/checkpoints/pipeline_checkpoint.json
{
  "completed_configs": [
    "MAPPO_8bus_PV0.0_3ph_r0",
    "MAPPO_8bus_PV0.0_3ph_r1000",
    // ...
  ],
  "n_results": 42
}
```

### Resume Behavior

```python
# On re-run, automatically skips completed configs
pipeline = TrainingPipeline(configs, n_workers=4)
results = pipeline.run_all()  # Resumes from checkpoint
```

### Recovery Utilities

```python
from experiments.training_pipeline import ExperimentRecovery

# Find interrupted runs (checkpointed but no result)
interrupted = ExperimentRecovery.find_incomplete_runs()

# Merge all figure JSONs into one master file
merged = ExperimentRecovery.merge_result_files()
```

---

## API Reference

### `TrainingConfig`

| Method | Returns | Description |
|--------|---------|-------------|
| `to_dict()` | `Dict` | Serialize to dictionary |
| `__post_init__()` | — | Validate configuration |

### `TrainingResult`

| Method | Returns | Description |
|--------|---------|-------------|
| `to_dict()` | `Dict` | Serialize to dictionary (numpy → list) |

### `TrainingPipeline`

| Method | Returns | Description |
|--------|---------|-------------|
| `run_all()` | `List[TrainingResult]` | Execute all configs |
| `save_results(filepath)` | `None` | Save to JSON |
| `save_results_by_figure()` | `None` | Save per-figure JSONs |

### `ResultAnalyzer`

| Method | Returns | Description |
|--------|---------|-------------|
| `get_best_result(method, system)` | `TrainingResult` | Best result by criteria |
| `get_pareto_front(objectives)` | `List[TrainingResult]` | Pareto-optimal results |
| `compare_methods(metric)` | `Dict` | Statistical comparison |
| `figure_summary(figure_name)` | `Dict` | Per-parameter-group stats |
| `export_latex_table(filepath)` | `None` | LaTeX table export |

### `ExperimentPresets`

| Method | Returns | Description |
|--------|---------|-------------|
| `thesis_chapter5_validation(fast)` | `List[TrainingConfig]` | Core validation |
| `thesis_chapter6_robustness(fast)` | `Dict[str, List]` | Robustness suite |
| `thesis_chapter7_transfer(fast)` | `Dict[str, List]` | Transfer learning |
| `ieee_paper_experiments(fast)` | `Dict[str, List]` | Paper submission |
| `quick_demo()` | `List[TrainingConfig]` | Minimal demo |

### `ExperimentLogger`

| Method | Returns | Description |
|--------|---------|-------------|
| `log_event(type, config, message)` | `None` | Log event |
| `log_result(result)` | `None` | Log completed result |
| `log_error(config, error)` | `None` | Log error |
| `get_summary()` | `Dict` | Experiment summary |

### `ExperimentRecovery`

| Method | Returns | Description |
|--------|---------|-------------|
| `find_incomplete_runs()` | `List[str]` | Interrupted config keys |
| `merge_result_files()` | `List[Dict]` | Deduplicated merge |

---

## Command-Line Usage

```bash
# Run specific example
python training_pipeline.py 1   # Basic demo
python training_pipeline.py 2   # Full suite
python training_pipeline.py 3   # Thesis Chapter 6
python training_pipeline.py 4   # Resume interrupted
python training_pipeline.py 5   # Custom bandwidth sweep

# Default (Example 1)
python training_pipeline.py
```

---

## Dependencies

| Package | Required For | Install |
|---------|-------------|---------|
| `numpy` | Core | `pip install numpy` |
| `pandas` | `ResultAnalyzer._to_dataframe()` | `pip install pandas` |
| `scipy` | Statistical tests in `compare_methods()` | `pip install scipy` |
| `shap` | Fig 48 SHAP analysis | `pip install shap` |
| `scikit-learn` | Fig 49 t-SNE | `pip install scikit-learn` |
| `pandapower` | Fig 44 short-circuit | `pip install pandapower` |

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | Original | Base pipeline with 5 methods, 3 systems, checkpointing |
| 2.0 | Extended | +17 figure evaluations, matrix builders, analysis tools |

---

## Citation

If using this pipeline in your thesis or publication:

```bibtex
@software{docl_marl_pipeline,
  title={DOCR-MARL Training Pipeline},
  author={[Your Name]},
  year={2026},
  note={Extended version with 17 figure-specific evaluations}
}
```
