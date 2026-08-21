#!/usr/bin/env bash
#
# run_full_experiment.sh
# ───────────────────────
# Drives the multi-system sweep described in RUNBOOK.md Steps 6-11: the
# per-system Fig 5 / Fig 6 / Core C loop, regenerating the Core-C-fed
# figures, consolidating into the results registry, and final verification.
#
# Deliberately does NOT wrap RUNBOOK.md Steps 1-5 (environment setup, the
# cache-clear decision, the fast validation run, or the multi-day single-
# system Core A-K + figures pipeline) -- those involve judgment calls
# (fresh vs. continue, which preset) that belong in a human's hands, not a
# script's, and this environment has repeatedly seen container restarts
# kill multi-hour unattended jobs mid-run with no way to recover the lost
# work, so blindly chaining everything into one script would make that
# worse, not better.
#
# Safe to interrupt and re-run: each stage is skipped (not re-run) if its
# output already exists, unless --force is passed. A failure or interrupt
# in one system's stage does not abort the others -- failures are
# collected and reported in a summary at the end, and the script exits
# non-zero if any occurred.
#
# Usage:
#   ./run_full_experiment.sh                      # everything, all 3 systems, thesis-scale
#   ./run_full_experiment.sh --systems "8bus 14bus"
#   ./run_full_experiment.sh --stages "fig5 fig6"  # resume just the outstanding sweeps
#   ./run_full_experiment.sh --n-seeds 3 --train-steps 2000   # faster, noisier pass
#   ./run_full_experiment.sh --force --stages fig6 --systems 14bus   # redo one system/stage
#
set -uo pipefail

SYSTEMS="8bus 14bus 39bus"
STAGES="fig5 fig6 corec figures registry verify"
N_SEEDS=10
TRAIN_STEPS=5000
N_WORKERS=4
CORE_C_SEED=42
FORCE=0
OUTPUTS_DIR="outputs"
CACHE_DIR="results/core_cache"
LOG_DIR="results/logs"

usage() {
    sed -n '2,33p' "$0" | sed 's/^# \{0,1\}//'
    exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --systems) SYSTEMS="$2"; shift 2 ;;
        --stages) STAGES="$2"; shift 2 ;;
        --n-seeds) N_SEEDS="$2"; shift 2 ;;
        --train-steps) TRAIN_STEPS="$2"; shift 2 ;;
        --n-workers) N_WORKERS="$2"; shift 2 ;;
        --core-c-seed) CORE_C_SEED="$2"; shift 2 ;;
        --outputs-dir) OUTPUTS_DIR="$2"; shift 2 ;;
        --force) FORCE=1; shift ;;
        -h|--help) usage 0 ;;
        *) echo "Unknown argument: $1" >&2; usage 1 ;;
    esac
done

mkdir -p "$LOG_DIR"
FAILURES=()
SKIPPED=()

stage_enabled() {
    [[ " $STAGES " == *" $1 "* ]]
}

log() {
    echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] $*"
}

run_logged() {
    # run_logged <log-file> <description> -- <command...>
    local logfile="$1" desc="$2"
    shift 2
    [[ "$1" == "--" ]] && shift
    log "START: $desc  (log: $logfile)"
    if "$@" > "$logfile" 2>&1; then
        log "OK:    $desc"
        return 0
    else
        local rc=$?
        log "FAIL:  $desc (exit $rc) -- see $logfile"
        FAILURES+=("$desc")
        return "$rc"
    fi
}

# ── Fig 5: OF vs. PV penetration, per system ────────────────────────────
if stage_enabled fig5; then
    for sys in $SYSTEMS; do
        out="$OUTPUTS_DIR/$sys/fig05_pv_sweep.npz"
        if [[ -f "$out" && "$FORCE" -ne 1 ]]; then
            log "SKIP:  Fig 5 ($sys) -- $out already exists (use --force to redo)"
            SKIPPED+=("fig5:$sys")
            continue
        fi
        run_logged "$LOG_DIR/fig05_${sys}.txt" "Fig 5 ($sys)" -- \
            python3 baselines/experiments/unified_exp_fig05.py \
                --system "$sys" --n_seeds "$N_SEEDS" --train_steps "$TRAIN_STEPS" \
                --n_workers "$N_WORKERS" --output_dir "$OUTPUTS_DIR"
    done
fi

# ── Fig 6: CTI violation rate vs. PV penetration, per system ────────────
if stage_enabled fig6; then
    for sys in $SYSTEMS; do
        out="$OUTPUTS_DIR/$sys/fig06_cti_violations.npz"
        if [[ -f "$out" && "$FORCE" -ne 1 ]]; then
            log "SKIP:  Fig 6 ($sys) -- $out already exists (use --force to redo)"
            SKIPPED+=("fig6:$sys")
            continue
        fi
        log "NOTE:  Fig 6 has no --n-workers and no partial-save -- this can take" \
            "many hours per system (see RUNBOOK.md Step 7); an interrupt here loses" \
            "the whole run, not just the remainder."
        run_logged "$LOG_DIR/fig06_${sys}.txt" "Fig 6 ($sys)" -- \
            python3 evaluation/utils/exp_fig06_cti_violation_rate.py \
                --system "$sys" --n_seeds "$N_SEEDS" --train_steps "$TRAIN_STEPS" \
                --output_dir "$OUTPUTS_DIR"
    done
fi

# ── Core C: MARL vs. baselines, per system ──────────────────────────────
if stage_enabled corec; then
    for sys in $SYSTEMS; do
        already_cached=$(python3 -c "
import json, sys
try:
    idx = json.load(open('$CACHE_DIR/cache_index.json'))
except FileNotFoundError:
    idx = {}
print(any(v.get('name') == 'core_C_baselines_$sys' for v in idx.values()))
")
        if [[ "$already_cached" == "True" && "$FORCE" -ne 1 ]]; then
            log "SKIP:  Core C ($sys) -- already cached as core_C_baselines_$sys (use --force to redo)"
            SKIPPED+=("corec:$sys")
            continue
        fi
        if [[ "$FORCE" -eq 1 && "$already_cached" == "True" ]]; then
            python3 -c "
from run_core_training import CacheManager
from pathlib import Path
CacheManager(Path('$CACHE_DIR')).invalidate(pattern='core_C_baselines_$sys')
"
        fi
        run_logged "$LOG_DIR/core_C_${sys}.txt" "Core C ($sys)" -- \
            python3 run_core_training.py --core-C --system "$sys" --seed "$CORE_C_SEED"
    done
fi

# ── Regenerate the figures that read Core C's cache ─────────────────────
if stage_enabled figures; then
    run_logged "$LOG_DIR/figures_core_c.txt" "Regenerate Figs 11,12,31-33,55-56" -- \
        python3 run_thesis_experiments.py --fig 11 12 31 32 33 55 56
fi

# ── Consolidate into the results registry ───────────────────────────────
if stage_enabled registry; then
    run_logged "$LOG_DIR/build_results_registry.txt" "Build results registry" -- \
        python3 experiments/build_results_registry.py
fi

# ── Final verification ───────────────────────────────────────────────────
if stage_enabled verify; then
    run_logged "$LOG_DIR/verify_tests.txt" "pytest tests/" -- \
        python3 -m pytest tests/ -q
    python3 run_core_training.py --cache-status
fi

# ── Summary ──────────────────────────────────────────────────────────────
echo
log "Systems: $SYSTEMS | Stages: $STAGES"
log "Skipped (${#SKIPPED[@]}): ${SKIPPED[*]:-none}"
if [[ "${#FAILURES[@]}" -eq 0 ]]; then
    log "All requested stages completed (or were already done)."
    exit 0
else
    log "FAILED (${#FAILURES[@]}): ${FAILURES[*]}"
    log "Re-run this same command to retry only what's missing -- completed" \
        "stages will be skipped automatically."
    exit 1
fi
