# Migration Guide: From Broken Scripts to Fixed Scripts

## What to Replace

### 1. Replace `exp_fig05_pv_sweep_evaluation.py`
**Old:** `baselines/experiments/exp_fig05_pv_sweep_evaluation.py`
**New:** `baselines/experiments/unified_exp_fig05.py`

**Key changes:**
- Added `system` parameter (`8bus` | `30bus` | `39bus`)
- Added `METAS = ["GA-SQP", "HWOA", "HHO", "JAYA", "EEO"]` to `ALL_METHODS`
- Added `evaluate_metaheuristic()` function calling `run_ga_sqp()` and `run_metaheuristic()`
- Added `run_statistical_tests()` with Wilcoxon signed-rank, Mann-Whitney U, Cohen's d, rank-biserial r
- Added `SeedResult` and `MethodResults` dataclasses for structured data
- Output now goes to `outputs/{system}/` instead of flat `outputs/`
- Saves `.npz`, `.png`, AND `.txt` stats report

### 2. Replace `pv_sweep_evaluation.py`
**Old:** `baselines/experiments/pv_sweep_evaluation.py`
**New:** `baselines/experiments/pv_sweep_evaluation_fixed.py`

**Key changes:**
- `fig05_main()` now receives `system=args.system`
- Removed monkey-patch hack (fix root cause in MARL agent instead)
- Actually passes `--methods` and `--pv-levels` (or uses defaults from unified script)
- Prints confirmation of which system actually ran

### 3. Add `run_all_systems.py`
**New file:** `baselines/experiments/run_all_systems.py`

**Purpose:**
- Orchestrates sequential runs: 8bus → 30bus → 39bus
- Generates `combined_scalability.png` (replaces broken Fig 15)
- No silent synthetic fallback — fails loudly if data missing

### 4. Add `unified_exp_fig06_cti_violations.py`
**New file:** `baselines/experiments/unified_exp_fig06_cti_violations.py`

**Purpose:**
- Consumes Fig 5 NPZ outputs (no re-training needed)
- Produces CTI violation plots for ANY system
- Saves both `.png` and `.txt` table

## Quick Start

```bash
# Run all three systems (will take hours)
python baselines/experiments/run_all_systems.py --n_seeds 10 --output_dir outputs

# Or run individual systems
python baselines/experiments/unified_exp_fig05.py --system 8bus --n_seeds 10
python baselines/experiments/unified_exp_fig05.py --system 30bus --n_seeds 10
python baselines/experiments/unified_exp_fig05.py --system 39bus --n_seeds 10

# Generate CTI violation plots from existing data
python baselines/experiments/unified_exp_fig06_cti_violations.py --system 8bus
python baselines/experiments/unified_exp_fig06_cti_violations.py --system 30bus
python baselines/experiments/unified_exp_fig06_cti_violations.py --system 39bus
```

## Verification Checklist

After running, verify:
- [ ] `outputs/8bus/fig05_pv_sweep.npz` exists
- [ ] `outputs/30bus/fig05_pv_sweep.npz` exists
- [ ] `outputs/39bus/fig05_pv_sweep.npz` exists
- [ ] Each NPZ contains keys: `GA-SQP_mean_OF`, `HWOA_mean_OF`, `HHO_mean_OF`, `JAYA_mean_OF`, `EEO_mean_OF`
- [ ] `outputs/{system}/fig05_pv_sweep_stats.txt` contains Wilcoxon p-values
- [ ] `outputs/combined_scalability.png` shows all 3 systems side-by-side
- [ ] No file contains "synthetic" or "fallback" in its output
