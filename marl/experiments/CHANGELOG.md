# DOCR-MARL Codebase Fixes — Complete Changelog

## Files Modified (13 files)

### 1. exp_fig39_ablation.py
**Issues fixed:**
- `trainer.train(episodes=...)` → `trainer.train(total_steps=...)` (×2)
- Added `system` parameter to `run_config()`, `main()`, and argparse
- Network selection: `get_8bus()` / `get_14bus()` / `get_30bus()` / `get_39bus()`

### 2. exp_fig12_convergence_curves.py
**Issues fixed:**
- Added 5 metaheuristic baselines: GA-SQP, HWOA, HHO, JAYA, EEO
- Added full metaheuristic optimizer implementations (GA, WOA, HHO, JAYA, EFO)
- Added `system` parameter throughout
- Added statistical test suite: Wilcoxon signed-rank, Mann-Whitney U, Welch's t-test, Cohen's d
- Added `evaluate_metaheuristic()` for fixed-setting evaluation

### 3. exp_fig40_hparam_grid.py
**Issues fixed:**
- Added `system` parameter
- Fixed `MAPPOMARLTrainer` instantiation with `try/except` for `hidden_actor`/`hidden_critic`
- Network selection based on system parameter

### 4. exp_fig09_cti_violation_dynamics.py
**Issues fixed:**
- Added `system` parameter to `main()` and argparse
- Network selection: 8/14/30/39-bus

### 5. exp_fig16_ewc_effectiveness.py
**Issues fixed:**
- Added `system` parameter to `main()`, `run_curriculum()`, and argparse
- Network selection based on system parameter

### 6. exp_fig37_per_agent_curves.py
**Issues fixed:**
- Added `system` parameter to `main()` and argparse
- Network selection based on system parameter

### 7. exp_fig38_attention_extraction_v2.py
**Issues fixed:**
- Removed `Mock8BusNetwork` fallback (now raises `ValueError` on unknown system)
- Replaced `get_8bus()` with `get_network(system)`
- Added `system` parameter to `run_figure8()`, `run_figure38()`, `main()`, and argparse

### 8. batch_runs.py (CLI wrapper)
**Issues fixed:**
- Passes `args.system` to `fig12_main()`

### 9. ablation_study.py (CLI wrapper)
**Issues fixed:**
- Passes `args.system` to `fig39_main()`

### 10. hyperparameter_grid.py (CLI wrapper)
**Issues fixed:**
- Passes `args.system` to `fig40_main()`

### 11. per_agent_curves.py (CLI wrapper)
**Issues fixed:**
- Passes `args.system` to `fig37_main()`

### 12. ewc_ablation.py (CLI wrapper)
**Issues fixed:**
- Passes `args.system` to `fig16_main()`

### 13. single_task_training.py (CLI wrapper)
**Issues fixed:**
- Passes `args.system` to `fig09_main()`

## Files Created (1 new file)

### 14. exp_fig15_scalability.py (NEW)
**Purpose:** Scalability analysis across 8/14/30/39-bus systems
**Features:**
- Compares MAPPO-Safe-EWC, MAPPO-Safe, MAPPO vs GA-SQP, HWOA, HHO, JAYA, EEO
- Uses `mealpy` library for metaheuristic implementations
- NO synthetic fallback — raises `RuntimeError` if `mealpy` not installed
- Raises `ValueError` if network data missing
- Outputs: `fig15_scalability.npz`, `fig15_scalability.png`

## Files with Minor Consistency Updates (4 files)

### 15. attention_evolution.py
- Added `--system` argparse argument (stub still generates synthetic data)

### 16. curriculum_trainer.py
- Added `--system` argparse argument
- Passes system to `exp_fig09.main()`

### 17. ewc_visualization.py
- Added `--system` argparse argument (stub still generates synthetic data)

### 18. warm_start.py
- Added `--system` argparse argument (stub still generates synthetic data)

## Remaining Issues (Not Fixed)

1. **attention_evolution.py, ewc_visualization.py, warm_start.py** — Still generate synthetic data.
   These need actual experiment implementations or should be removed from thesis claims.

2. **mealpy dependency** — The metaheuristic implementations require `pip install mealpy`.
   If unavailable, `exp_fig15_scalability.py` and `exp_fig12_convergence_curves.py` will crash
   with a clear error message instead of silently generating fake data.

3. **Network data availability** — `get_14bus()`, `get_30bus()`, `get_39bus()` must exist in
   `network.network_data`. If they don't exist, scripts will fail with `ImportError`.

4. **Fig 38 attention** — `exp_fig38_attention_extraction_v2.py` still uses synthetic attention
   matrices because it cannot extract real attention without a trained GNN model.
   The `Mock8BusNetwork` fallback was removed, but the attention generation logic
   remains synthetic when `network.network_data` is unavailable.

## Validation Checklist

After applying these fixes, run:

```bash
# Test 8-bus (should work if network data exists)
python exp_fig39_ablation.py --system 8bus --n_seeds 2
python exp_fig12_convergence_curves.py --system 8bus --n_runs 3 --n_episodes 100

# Test 30/39-bus (requires network data)
python exp_fig15_scalability.py --systems 30bus 39bus --n_runs 3

# Test metaheuristic comparison
python exp_fig12_convergence_curves.py --system 8bus --n_runs 5 --n_episodes 200

# Test statistical tests output
python exp_fig12_convergence_curves.py --system 8bus --n_runs 10 --n_episodes 500
```

## Dependencies Added

```
mealpy>=2.0.0   # For metaheuristic baselines (GA-SQP, HWOA, HHO, JAYA, EEO)
```
