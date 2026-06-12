#!/usr/bin/env python3
"""
Master experiment launcher for DOCR-MARL thesis.
Runs all experiments defined in the Figure Compendium (Figures 1-58) and generates outputs.

Aligned with: DOCR_MARL_Figure_1_to_58_Experimental_Guide.docx

Usage:
    python run_thesis_experiments.py --all                       # full suite (~30-40 GPU-days)
    python run_thesis_experiments.py --phase1                   # manual diagrams + architecture
    python run_thesis_experiments.py --phase2                   # core training runs (START FIRST)
    python run_thesis_experiments.py --phase3                   # training dynamics
    python run_thesis_experiments.py --phase4                   # robustness & resilience
    python run_thesis_experiments.py --phase5                   # scalability & transfer
    python run_thesis_experiments.py --phase6                   # deployment & hardware
    python run_thesis_experiments.py --phase7                   # protection engineering
    python run_thesis_experiments.py --phase8                   # statistical aggregation
    python run_thesis_experiments.py --phase9                   # final compilation
    python run_thesis_experiments.py --figure <num>             # run single figure
    python run_thesis_experiments.py --fast                     # smoke test (reduced runs)
"""

import argparse
import os
import subprocess
import sys
import io
import time
import json
from pathlib import Path
from datetime import datetime

# ---------------------------------------------------------------------
# Windows Unicode Fix
# ---------------------------------------------------------------------
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
BASE_DIR = Path(__file__).parent.absolute()
OUTPUT_DIR = BASE_DIR / "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Pre-create all figure subdirectories so they're available everywhere
# (including when figure functions are called before main() runs)
for subdir in ["figs_01_10", "figs_11_20", "figs_21_30", "figs_31_40",
               "figs_41_50", "figs_51_58", "logs", "checkpoints", "cache"]:
    os.makedirs(OUTPUT_DIR / subdir, exist_ok=True)

# Logging
LOG_FILE = OUTPUT_DIR / f"experiment_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

# Fast mode: reduced iterations for quick testing
FAST_MODE = "--fast" in sys.argv
N_RUNS = "2" if FAST_MODE else "50"
N_RUNS_BASELINE = "2" if FAST_MODE else "10"
N_RUNS_HPARAM = "2" if FAST_MODE else "3"
SEED = 42
N_WORKERS = "4"

def log(msg):
    """Print and log message with encoding safety."""
    safe_msg = msg.replace('→', '->').replace('←', '<-').replace('—', '-')
    print(safe_msg)
    with open(LOG_FILE, "a", encoding='utf-8') as f:
        f.write(f"[{datetime.now().strftime('%H:%M:%S')}] {safe_msg}\\n")

def run_cmd(cmd, description, check=True, env_vars=None):
    """Run a shell command, print status, and log output."""
    log(f"\\n[RUN] {description}")
    log(f"CMD: {cmd}")
    t0 = time.time()

    env = os.environ.copy()
    if env_vars:
        env.update(env_vars)
    if FAST_MODE:
        env["DOCR_FAST"] = "1"

    ret = subprocess.run(cmd, shell=True, env=env, capture_output=True, text=True)
    elapsed = time.time() - t0

    if ret.stdout:
        log(ret.stdout)
    if ret.stderr:
        log(f"[STDERR] {ret.stderr}")

    if ret.returncode != 0 and check:
        log(f"ERROR: command failed after {elapsed:.1f}s (exit code {ret.returncode})")
        # Don't sys.exit(1) for Phase 1 placeholders — just log and continue
        if "Placeholder" in ret.stderr or "No such file" in ret.stderr:
            log("WARNING: Placeholder module or missing dependency. Continuing...")
            return ret.returncode
        sys.exit(1)

    log(f"Finished in {elapsed:.1f}s")
    return ret.returncode

# ==============================================================================
# SECTION 1: SYSTEM ARCHITECTURE & POWER SYSTEM DIAGRAMS (Figs 1-4, 17, 19, 27, 43-47)
# ==============================================================================

def fig_01_system_architecture():
    """Figure 1: High-Level System Architecture
    Status: MANUAL DIAGRAM — No experiment needed.
    """
    log("[FIG 1] Manual diagram — create in docs/diagrams/fig_01_system_arch.drawio")
    log("  -> Reference modules: env/opendss_interface.py, agents/mappo_safe_ewc.py, comm/gnn_layer.py, deploy/iec61850_bridge.py")

def fig_02_data_flow():
    """Figure 2: Data Flow Architecture (Single Training Step)"""
    run_cmd(
        f"python environment/experiments/data_flow_logging.py --system 8bus --episodes 1 --debug",
        "Fig 2: Data Flow Architecture (single episode DEBUG logging)"
    )

def fig_03_8bus_topology():
    """Figure 3: IEEE 8-Bus Test System with DOCR Placement (0% PV)"""
    # FIX: Use Path operator / instead of f-string with literal quotes
    out = OUTPUT_DIR / "figs_01_10" / "fig_03_8bus_topology.png"
    run_cmd(
        f'python network/utils/plot_topology.py --system 8bus --pv 0.0 --output "{out}"',
        "Fig 3: IEEE 8-Bus Topology with DOCR Placement"
    )

def fig_04_30bus_topology():
    """Figure 4: IEEE 30-Bus Test System with DOCR Placement"""
    # FIX: Use Path operator / instead of f-string with literal quotes
    out = OUTPUT_DIR / "figs_01_10" / "fig_04_30bus_topology.png"
    run_cmd(
        f'python network/utils/plot_topology.py --system 30bus --pv 0.0 --output "{out}"',
        "Fig 4: IEEE 30-Bus Topology with DOCR Placement"
    )

def fig_17_iec61850_architecture():
    """Figure 17: IEC 61850 Deployment Architecture
    Status: MANUAL DIAGRAM — No experiment needed.
    """
    log("[FIG 17] Manual diagram — create in docs/diagrams/fig_17_iec61850_arch.drawio")
    log("  -> Reference: deploy/iec61850_bridge.py Logical Node mapping")

def fig_19_pv_ride_through():
    """Figure 19: PV Inverter Fault Ride-Through Curve (IEC 61427-1 / IEEE 1547-2018)"""
    # FIX: Quote the path in the shell command to handle spaces
    out = OUTPUT_DIR / "figs_01_10" / "fig_19_frt_curves.pdf"
    run_cmd(
        f'python data/standards/frt_curves.py --standards iec61427 ieee1547 --output "{out}"',
        "Fig 19: PV Inverter FRT Curves (IEC 61427-1 / IEEE 1547-2018)"
    )

def fig_27_decpomdp():
    """Figure 27: Dec-POMDP Formulation for DOCR Coordination
    Status: CONCEPTUAL DIAGRAM — No experiment needed.
    """
    log("[FIG 27] Conceptual diagram — create in docs/diagrams/fig_27_decpomdp.tikz")
    log("  -> Reference dimensions from env/config.yaml")

def fig_43_coordination_diagram():
    """Figure 43: Relay Coordination Time-Distance Diagram"""
    out = OUTPUT_DIR / "figs_41_50" / "fig_43_coordination_diagram.pdf"
    run_cmd(
        f'python network/utils/coordination_diagram.py --system 8bus --pv 1.0 --output "{out}"',
        "Fig 43: Relay Coordination Time-Distance Diagram"
    )

def fig_44_fault_contribution():
    """Figure 44: Fault Current Contribution Pie Chart (100% PV)"""
    out = OUTPUT_DIR / "figs_41_50" / "fig_44_fault_contribution.pdf"
    run_cmd(
        f'python network/utils/fault_contribution.py --system 8bus --pv 1.0 --output "{out}"',
        "Fig 44: Fault Current Contribution Pie Chart (100% PV)"
    )

def fig_45_daily_profile():
    """Figure 45: Daily PV Generation Profile & Adaptive Relay Response"""
    out = OUTPUT_DIR / "figs_41_50" / "fig_45_daily_profile.pdf"
    run_cmd(
        f'python data/experiments/daily_adaptive_response.py --system 8bus --tmy3 data/tmy3/phoenix_az_tmy3.csv --output "{out}"',
        "Fig 45: Daily PV Profile & Adaptive Relay Response"
    )

def fig_46_relay_curves():
    """Figure 46: Relay Operating Characteristic Curves with Learned Settings"""
    out = OUTPUT_DIR / "figs_41_50" / "fig_46_relay_curves.pdf"
    run_cmd(
        f'python network/utils/relay_curves.py --system 8bus --pv 1.0 --method MAPPO-Safe-EWC --output "{out}"',
        "Fig 46: Relay Operating Characteristic Curves"
    )

def fig_47_sympathetic_tripping():
    """Figure 47: Sympathetic Tripping Prevention Analysis"""
    out = OUTPUT_DIR / "figs_41_50" / "fig_47_sympathetic_tripping.pdf"
    run_cmd(
        f'python network/utils/sympathetic_trip_cases.py --system 8bus --pv 1.0 --output "{out}"',
        "Fig 47: Sympathetic Tripping Prevention Analysis"
    )

# ==============================================================================
# SECTION 2: METHODOLOGY, ALGORITHMS & FORMULATIONS (Figs 26, 26a, 28-30, 34-36)
# ==============================================================================

def fig_26_mappo_flowchart():
    """Figure 26: MAPPO-Safe-EWC Training Algorithm Flowchart
    Status: MANUAL DIAGRAM — Create from Algorithm 1 pseudocode.
    """
    log("[FIG 26] Manual flowchart — create in docs/diagrams/fig_26_mappo_flowchart.tikz")
    log("  -> Reference: train.py main loop structure")

def fig_26a_ctde_flowchart():
    """Figure 26a: MAPPO-Safe-EWC CTDE Alternative View
    Status: MANUAL DIAGRAM
    """
    log("[FIG 26a] Manual CTDE diagram — create in docs/diagrams/fig_26a_ctde_flowchart.tikz")
    log("  -> Reference: agents/ctde_mapper.py")

def fig_28_safe_rl_dual():
    """Figure 28: Safe RL Dual Approach
    Status: CONCEPTUAL DIAGRAM — No experiment needed.
    """
    log("[FIG 28] Conceptual diagram — create in docs/diagrams/fig_28_safe_rl_dual.tikz")
    log("  -> Reference: agents/safety_layer.py lambda update equation")

def fig_29_ewc_continual():
    """Figure 29: Elastic Weight Consolidation (EWC) for Continual Adaptation"""
    out = OUTPUT_DIR / "figs_21_30" / "fig_29_ewc_continual.pdf"
    run_cmd(
        f'python marl/experiments/ewc_visualization.py --system 8bus --tasks 0.0 0.25 0.5 0.75 1.0 --output "{out}"',
        "Fig 29: EWC Continual Adaptation Visualization"
    )

def fig_30_pseudocode():
    """Figure 30: Algorithm 1: MAPPO-Safe-EWC Training Procedure (Pseudocode)
    Status: TYPESET FROM CODE — No experiment needed.
    """
    log("[FIG 30] Typeset pseudocode — create in docs/diagrams/fig_30_pseudocode.tex")
    log("  -> Reference: train.py lines 1-32")

def fig_34_nn_architecture():
    """Figure 34: MAPPO-Safe-EWC Neural Network Architecture Detail"""
    out = OUTPUT_DIR / "figs_31_40" / "fig_34_nn_architecture.pdf"
    run_cmd(
        f'python marl/utils/extract_architecture.py --config marl/configs/network.yaml --output "{out}"',
        "Fig 34: Neural Network Architecture Detail"
    )

def fig_35_observation_space():
    """Figure 35: Observation Space Feature Breakdown & Encoding"""
    out = OUTPUT_DIR / "figs_31_40" / "fig_35_observation_space.pdf"
    run_cmd(
        f'python environment/utils/observation_breakdown.py --system 8bus --output "{out}"',
        "Fig 35: Observation Space Feature Breakdown"
    )

def fig_36_action_space_geometry():
    """Figure 36: Action Space Bounds & Safety Projection Geometry"""
    out = OUTPUT_DIR / "figs_31_40" / "fig_36_action_space_geometry.pdf"
    run_cmd(
        f'python environment/experiments/action_space_geometry.py --system 8bus --grid 100 --output "{out}"',
        "Fig 36: Action Space Bounds & Safety Projection Geometry (~2h)"
    )

# ==============================================================================
# SECTION 3: TRAINING DYNAMICS & LEARNING CURVES (Figs 7-10, 12, 16, 21, 37-38)
# ==============================================================================

def fig_07_curriculum():
    """Figure 7: Curriculum Learning Progression (MAPPO-Safe-EWC, 8-Bus)"""
    out = OUTPUT_DIR / "figs_01_10" / "fig_07_curriculum.pdf"
    run_cmd(
        f'python marl/curriculum_trainer.py --system 8bus --tasks 0.0 0.25 0.5 0.75 1.0 --episodes 5000 --n_runs {N_RUNS} --output "{out}"',
        "Fig 7: Curriculum Learning Progression"
    )

def fig_08_attention_weights():
    """Figure 8: Learned GNN Attention Weights (8-Bus, 100% PV)"""
    out = OUTPUT_DIR / "figs_01_10" / "fig_08_attention_weights.pdf"
    run_cmd(
        f'python marl/experiments/extract_attention.py --system 8bus --pv 1.0 --n_scenarios 100 --seed {SEED} --output "{out}"',
        "Fig 8: Learned GNN Attention Weights"
    )

def fig_09_cti_violations():
    """Figure 9: CTI Violation Dynamics During Training (8-Bus, 100% PV)"""
    out = OUTPUT_DIR / "figs_01_10" / "fig_09_cti_violations.pdf"
    run_cmd(
        f'python marl/experiments/single_task_training.py --system 8bus --pv 1.0 --method MAPPO-Safe --episodes 5000 --seed {SEED} --output "{out}"',
        "Fig 9: CTI Violation Dynamics During Training (~4h)"
    )

def fig_10_lagrange_multipliers():
    """Figure 10: Lagrange Multiplier Evolution per P/B Pair"""
    log_file = OUTPUT_DIR / "logs" / "single_task_8bus_100pv.json"
    out = OUTPUT_DIR / "figs_01_10" / "fig_10_lagrange_multipliers.pdf"
    run_cmd(
        f'python marl/utils/lagrange_analysis.py --log "{log_file}" --output "{out}"',
        "Fig 10: Lagrange Multiplier Evolution (post-process from Fig 9 logs)"
    )

def fig_12_convergence_bands():
    """Figure 12: Convergence Curves with 95% Confidence Bands (8-Bus, 100% PV)"""
    n_runs = "5" if FAST_MODE else "50"
    out = OUTPUT_DIR / "figs_11_20" / "fig_12_convergence_bands.pdf"
    run_cmd(
        f'python marl/experiments/batch_runs.py --system 8bus --pv 1.0 --method MAPPO-Safe-EWC --n_runs {n_runs} --episodes 5000 --output "{out}"',
        f"Fig 12: Convergence Curves with 95% CI ({n_runs} runs, ~2-3 days GPU)"
    )

def fig_16_ewc_effectiveness():
    """Figure 16: EWC Effectiveness — Performance Drop After Task Transition"""
    out = OUTPUT_DIR / "figs_11_20" / "fig_16_ewc_effectiveness.pdf"
    run_cmd(
        f'python marl/experiments/ewc_ablation.py --system 8bus --tasks 0.0 0.25 0.5 0.75 1.0 --n_runs {N_RUNS_BASELINE} --output "{out}"',
        "Fig 16: EWC Effectiveness Ablation"
    )

def fig_21_action_projection():
    """Figure 21: Action Projection Efficacy During Training"""
    log_file = OUTPUT_DIR / "logs" / "single_task_8bus_100pv.json"
    out = OUTPUT_DIR / "figs_21_30" / "fig_21_action_projection.pdf"
    run_cmd(
        f'python marl/utils/safety_projection_analysis.py --log "{log_file}" --output "{out}"',
        "Fig 21: Action Projection Efficacy (post-process from Fig 9 logs)"
    )

def fig_37_per_agent_curves():
    """Figure 37: Per-Agent Learning Curves (Individual Convergence)"""
    out = OUTPUT_DIR / "figs_31_40" / "fig_37_per_agent_curves.pdf"
    run_cmd(
        f'python marl/experiments/per_agent_curves.py --system 8bus --pv 1.0 --episodes 30000 --seed {SEED} --output "{out}"',
        "Fig 37: Per-Agent Learning Curves"
    )

def fig_38_attention_evolution():
    """Figure 38: Attention Weight Evolution During Training"""
    out = OUTPUT_DIR / "figs_31_40" / "fig_38_attention_evolution.pdf"
    run_cmd(
        f'python marl/experiments/attention_evolution.py --system 8bus --pv 1.0 --snapshots 0 100 500 1000 5000 --seed {SEED} --output "{out}"',
        "Fig 38: Attention Weight Evolution During Training"
    )

# ==============================================================================
# SECTION 4: PERFORMANCE EVALUATION (Figs 5-6, 43-46) — see Section 1 for 43-46
# ==============================================================================

def fig_05_objective_vs_pv():
    """Figure 5: Objective Function vs. PV Penetration (8-Bus System)"""
    n_runs = "2" if FAST_MODE else "50"
    out = OUTPUT_DIR / "figs_01_10" / "fig_05_objective_vs_pv.pdf"
    run_cmd(
        f'python baselines/experiments/pv_sweep_evaluation.py --system 8bus --methods MAPPO-Safe-EWC MAPPO-Safe MAPPO IPPO QMIX IQL Traditional --pv-levels 0.0 0.25 0.5 0.75 1.0 --n_runs {n_runs} --output "{out}"',
        f"Fig 5: Objective vs. PV Penetration ({n_runs} seeds, 1750 runs full)"
    )

def fig_06_cti_vs_pv():
    """Figure 6: CTI Violation Rate vs. PV Penetration (8-Bus System)"""
    out = OUTPUT_DIR / "figs_01_10" / "fig_06_cti_vs_pv.pdf"
    run_cmd(
        f'python evaluation/utils/cti_analysis.py --data "{OUTPUT_DIR}/pv_sweep_results.pkl" --output "{out}"',
        "Fig 6: CTI Violation Rate vs. PV (post-process from Fig 5 data)"
    )

# ==============================================================================
# SECTION 5: STATISTICAL ANALYSIS & RANKING (Figs 11, 31-33, 55-57)
# ==============================================================================

def fig_11_nemenyi():
    """Figure 11: Nemenyi Critical Difference Diagram (alpha = 0.05)"""
    out = OUTPUT_DIR / "figs_11_20" / "fig_11_nemenyi.pdf"
    run_cmd(
        f'python evaluation/utils/nemenyi_test.py --data "{OUTPUT_DIR}/pv_sweep_results.pkl" --alpha 0.05 --output "{out}"',
        "Fig 11: Nemenyi Critical Difference Diagram"
    )

def fig_31_multi_metric_ranking():
    """Figure 31: Overall Performance Ranking with Critical Difference"""
    out = OUTPUT_DIR / "figs_31_40" / "fig_31_multi_metric_ranking.pdf"
    run_cmd(
        f'python evaluation/utils/multi_metric_ranking.py --data "{OUTPUT_DIR}/benchmark_results.pkl" --weights 0.3 0.3 0.2 0.2 --output "{out}"',
        "Fig 31: Overall Performance Ranking with CD"
    )

def fig_32_effect_size():
    """Figure 32: Effect Size Analysis — MAPPO-Safe-EWC vs. Baselines"""
    out = OUTPUT_DIR / "figs_31_40" / "fig_32_effect_size.pdf"
    run_cmd(
        f'python evaluation/utils/effect_size.py --data "{OUTPUT_DIR}/pv_sweep_results.pkl" --baseline MAPPO-Safe-EWC --output "{out}"',
        "Fig 32: Effect Size Analysis (Cohen's d, Cliff's Delta)"
    )

def fig_33_radar_chart():
    """Figure 33: Multi-Dimensional Method Comparison (8-Bus, 100% PV)"""
    out = OUTPUT_DIR / "figs_31_40" / "fig_33_radar_chart.pdf"
    run_cmd(
        f'python evaluation/utils/radar_chart.py --data "{OUTPUT_DIR}/benchmark_results.pkl" --system 8bus --pv 1.0 --output "{out}"',
        "Fig 33: Multi-Dimensional Method Comparison"
    )

def fig_55_comparison_table():
    """Figure 55: Comprehensive Method Comparison Table"""
    out = OUTPUT_DIR / "figs_51_58" / "fig_55_comparison_table.tex"
    run_cmd(
        f'python evaluation/utils/comparison_table.py --data "{OUTPUT_DIR}/benchmark_results.pkl" --output "{out}"',
        "Fig 55: Comprehensive Method Comparison Table (LaTeX)"
    )

def fig_56_pareto_front():
    """Figure 56: Multi-Objective Pareto Front (Speed vs. Selectivity)"""
    out = OUTPUT_DIR / "figs_51_58" / "fig_56_pareto_front.pdf"
    run_cmd(
        f'python evaluation/utils/pareto_front.py --data "{OUTPUT_DIR}/pv_sweep_results.pkl" --output "{out}"',
        "Fig 56: Multi-Objective Pareto Front"
    )

def fig_57_bootstrap_ci():
    """Figure 57: Bootstrap Confidence Interval Convergence"""
    out = OUTPUT_DIR / "figs_51_58" / "fig_57_bootstrap_ci.pdf"
    run_cmd(
        f'python evaluation/utils/bootstrap_convergence.py --data "{OUTPUT_DIR}/logs/batch_runs_8bus_100pv.json" --output "{out}"',
        "Fig 57: Bootstrap CI Convergence"
    )

# ==============================================================================
# SECTION 6: ROBUSTNESS & RESILIENCE (Figs 13-14, 20, 22, 24, 47, 54)
# ==============================================================================

def fig_13_communication_stress():
    """Figure 13: Performance Degradation under Communication Stress"""
    out = OUTPUT_DIR / "figs_11_20" / "fig_13_communication_stress.pdf"
    run_cmd(
        f'python evaluation/experiments/communication_stress.py --system 8bus --pv 1.0 --method MAPPO-Safe-EWC --packet-loss 0.0 0.1 0.2 0.3 0.4 0.5 --latency 0 100 200 300 400 500 --output "{out}"',
        "Fig 13: Communication Stress Robustness"
    )

def fig_14_n1_contingency():
    """Figure 14: N-1 Contingency Impact and Recovery (30-Bus)"""
    out = OUTPUT_DIR / "figs_11_20" / "fig_14_n1_contingency.pdf"
    run_cmd(
        f'python evaluation/experiments/n1_contingency.py --system 30bus --method MAPPO-Safe-EWC --finetune-episodes 50 --output "{out}"',
        "Fig 14: N-1 Contingency Impact & Recovery (~6h)"
    )

def fig_20_byzantine():
    """Figure 20: Byzantine Agent Detection via Trust Score Evolution"""
    out = OUTPUT_DIR / "figs_11_20" / "fig_20_byzantine.pdf"
    run_cmd(
        f'python evaluation/experiments/byzantine_resilience.py --system 8bus --pv 1.0 --n_byzantine 2 --attack-types random collusion fdi --episodes 100 --seed {SEED} --output "{out}"',
        "Fig 20: Byzantine Agent Detection"
    )

def fig_22_bandwidth_tradeoff():
    """Figure 22: Bandwidth vs. Performance Trade-off"""
    out = OUTPUT_DIR / "figs_21_30" / "fig_22_bandwidth_tradeoff.pdf"
    run_cmd(
        f'python evaluation/experiments/bandwidth_tradeoff.py --system 8bus --pv 1.0 --gnn-depths 1 2 3 4 --compressions none int8 topk50 --output "{out}"',
        "Fig 22: Bandwidth vs. Performance Trade-off"
    )

def fig_24_sensor_robustness():
    """Figure 24: Sensor Robustness — Performance Under Measurement Degradation"""
    out = OUTPUT_DIR / "figs_21_30" / "fig_24_sensor_robustness.pdf"
    run_cmd(
        f'python environment/experiments/sensor_degradation.py --system 8bus --pv 1.0 --noise-levels 0.0 0.05 0.1 0.15 0.2 --bias-levels 0.0 0.05 0.1 --missing-rates 0.0 0.1 0.2 --output "{out}"',
        "Fig 24: Sensor Robustness"
    )

def fig_54_adversarial():
    """Figure 54: Adversarial Robustness — Perturbed Observation Response"""
    out = OUTPUT_DIR / "figs_51_58" / "fig_54_adversarial.pdf"
    run_cmd(
        f'python environment/experiments/adversarial_robustness.py --system 8bus --pv 1.0 --attack-types fgsm pgd --epsilons 0.01 0.05 0.1 0.2 --n_eval 100 --output "{out}"',
        "Fig 54: Adversarial Robustness"
    )

# ==============================================================================
# SECTION 7: SCALABILITY & TRANSFER LEARNING (Figs 15, 23, 25, 52)
# ==============================================================================

def fig_15_scalability():
    """Figure 15: Scalability Analysis Across Test Systems"""
    systems = "8bus 14bus 30bus" if FAST_MODE else "8bus 14bus 30bus 57bus 118bus"
    out = OUTPUT_DIR / "figs_11_20" / "fig_15_scalability.pdf"
    run_cmd(
        f'python evaluation/experiments/scalability_sweep.py --systems {systems} --method MAPPO-Safe-EWC --output "{out}"',
        f"Fig 15: Scalability Analysis ({systems})"
    )

def fig_23_unseen_faults():
    """Figure 23: Generalization to Unseen Fault Types"""
    out = OUTPUT_DIR / "figs_21_30" / "fig_23_unseen_faults.pdf"
    run_cmd(
        f'python environment/experiments/unseen_faults.py --system 8bus --pv 1.0 --fault-types bolted high-impedance evolving simultaneous --n_eval 100 --output "{out}"',
        "Fig 23: Generalization to Unseen Fault Types"
    )

def fig_25_zero_shot_transfer():
    """Figure 25: Zero-Shot Transfer — 8-Bus → 30-Bus (100% PV)"""
    out = OUTPUT_DIR / "figs_21_30" / "fig_25_zero_shot_transfer.pdf"
    run_cmd(
        f'python evaluation/experiments/zero_shot_transfer.py --source-system 8bus --target-system 30bus --pv 1.0 --finetune-steps 500 --scratch-steps 5000 --output "{out}"',
        "Fig 25: Zero-Shot Transfer 8→30 Bus"
    )

def fig_52_digital_twin_pipeline():
    """Figure 52: Digital Twin Training-to-Deployment Pipeline
    Status: CONCEPTUAL DIAGRAM — No experiment needed.
    """
    log("[FIG 52] Conceptual pipeline diagram — create in docs/diagrams/fig_52_digital_twin_pipeline.drawio")
    log("  -> Reference: deploy/iec61850_bridge.py, data collection → training → validation → deployment → monitoring")

# ==============================================================================
# SECTION 8: DEPLOYMENT & REAL-TIME IMPLEMENTATION (Figs 17-18, 51-53)
# ==============================================================================
# Fig 17: See Section 1

def fig_18_inference_latency():
    """Figure 18: Inference Latency Distribution (Intel NUC i5, 1000 runs)"""
    out = OUTPUT_DIR / "figs_11_20" / "fig_18_inference_latency.pdf"
    run_cmd(
        f'python deployment/benchmark_latency.py --model results/checkpoints/best_model_8bus_100pv.pt --n_calls 1000 --hardware nuc_i5 --output "{out}"',
        "Fig 18: Inference Latency Distribution (Intel NUC i5)"
    )

def fig_51_protocol_stack():
    """Figure 51: Communication Protocol Stack (OSI Layer Mapping)
    Status: CONCEPTUAL DIAGRAM — No experiment needed.
    """
    log("[FIG 51] Conceptual protocol diagram — create in docs/diagrams/fig_51_protocol_stack.drawio")
    log("  -> Reference: deploy/protocol_stack.py (OSI 7-layer mapping for GNN messages)")

def fig_53_model_compression():
    """Figure 53: Model Compression & Quantization Impact"""
    out = OUTPUT_DIR / "figs_51_58" / "fig_53_model_compression.pdf"
    run_cmd(
        f'python evaluation/experiments/model_compression.py --model results/checkpoints/best_model_8bus_100pv.pt --methods int8 pruning distillation --n_eval 100 --output "{out}"',
        "Fig 53: Model Compression & Quantization Impact"
    )

# ==============================================================================
# SECTION 9: NEURAL ARCHITECTURE & INTERPRETABILITY (Figs 8, 34-36, 48-50)
# ==============================================================================
# Figs 8, 34-36: See Sections 3 and 2

def fig_48_shap():
    """Figure 48: SHAP Feature Importance for Relay Decisions"""
    out = OUTPUT_DIR / "figs_41_50" / "fig_48_shap.pdf"
    run_cmd(
        f'python evaluation/experiments/shap_analysis.py --system 8bus --pv 1.0 --n_background 100 --agents all --output "{out}"',
        "Fig 48: SHAP Feature Importance (~10min/agent on CPU)"
    )

def fig_49_decision_boundary():
    """Figure 49: Decision Boundary Visualization (2D Projection)"""
    out = OUTPUT_DIR / "figs_41_50" / "fig_49_decision_boundary.pdf"
    run_cmd(
        f'python evaluation/experiments/decision_boundary.py --system 8bus --pv 1.0 --n_obs 5000 --method tsne --output "{out}"',
        "Fig 49: Decision Boundary Visualization (t-SNE/UMAP)"
    )

def fig_50_critical_pairs():
    """Figure 50: Critical P/B Pair Identification & Priority Ranking"""
    out = OUTPUT_DIR / "figs_41_50" / "fig_50_critical_pairs.pdf"
    run_cmd(
        f'python network/utils/critical_pairs.py --system 8bus --pv 1.0 --lambda-data "{OUTPUT_DIR}/logs/lagrange_8bus_100pv.json" --violation-data "{OUTPUT_DIR}/pv_sweep_results.pkl" --output "{out}"',
        "Fig 50: Critical P/B Pair Identification"
    )

# ==============================================================================
# SECTION 10: ABLATION & SENSITIVITY ANALYSIS (Figs 39-42, 55-56)
# ==============================================================================
# Figs 55-56: See Section 5

def fig_39_ablation():
    """Figure 39: Systematic Ablation Study — Component Contribution"""
    out = OUTPUT_DIR / "figs_31_40" / "fig_39_ablation.pdf"
    run_cmd(
        f'python marl/experiments/ablation_study.py --system 8bus --pv 1.0 --configs MAPPO MAPPO-GNN MAPPO-GNN-Safety MAPPO-GNN-Safety-EWC MAPPO-Safe-EWC-Curriculum --n_runs {N_RUNS_BASELINE} --output "{out}"',
        "Fig 39: Systematic Ablation Study"
    )

def fig_40_hyperparameter():
    """Figure 40: Hyperparameter Sensitivity Heatmap"""
    n_runs = "1" if FAST_MODE else "3"
    episodes = "500" if FAST_MODE else "2000"
    out = OUTPUT_DIR / "figs_31_40" / "fig_40_hyperparameter.pdf"
    run_cmd(
        f'python marl/experiments/hyperparameter_grid.py --system 8bus --pv 1.0 --lambda-ewc 1e2 1e3 1e4 1e5 1e6 --alpha-lambda 1e-5 1e-4 1e-3 1e-2 --hidden-dims 16 32 64 128 --n_runs {n_runs} --episodes {episodes} --output "{out}"',
        f"Fig 40: Hyperparameter Sensitivity Heatmap ({n_runs} seeds, {episodes} eps)"
    )

def fig_41_sample_efficiency():
    """Figure 41: Sample Efficiency Comparison"""
    out = OUTPUT_DIR / "figs_31_40" / "fig_41_sample_efficiency.pdf"
    run_cmd(
        f'python marl/utils/sample_efficiency.py --data "{OUTPUT_DIR}/logs/batch_runs_8bus_100pv.json" --output "{out}"',
        "Fig 41: Sample Efficiency Comparison (post-process from Fig 12)"
    )

def fig_42_warm_start():
    """Figure 42: Warm-Start vs. Cold-Start Initialization"""
    out = OUTPUT_DIR / "figs_31_40" / "fig_42_warm_start.pdf"
    run_cmd(
        f'python marl/experiments/warm_start.py --system 8bus --pv 1.0 --n_runs {N_RUNS_BASELINE} --episodes 5000 --output "{out}"',
        "Fig 42: Warm-Start vs. Cold-Start Initialization"
    )

# ==============================================================================
# SECTION 11: SUPPLEMENTARY ANALYSIS (Figs 49, 52, 54, 56-58)
# ==============================================================================
# Figs 49, 52, 54, 56: See respective sections

def fig_58_seasonal():
    """Figure 58: Seasonal/Weather Generalization (Extended PV Profiles)"""
    out = OUTPUT_DIR / "figs_51_58" / "fig_58_seasonal.pdf"
    run_cmd(
        f'python evaluation/experiments/seasonal_generalization.py --system 8bus --seasons spring summer autumn winter --n_eval 100 --output "{out}"',
        "Fig 58: Seasonal/Weather Generalization"
    )

# ==============================================================================
# FIGURE-TO-FUNCTION MAPPING (Complete, No Duplicates)
# ==============================================================================

FIGURE_FUNCTIONS = {
    # Section 1: System Architecture & Power System Diagrams
    1: fig_01_system_architecture,
    2: fig_02_data_flow,
    3: fig_03_8bus_topology,
    4: fig_04_30bus_topology,
    17: fig_17_iec61850_architecture,
    19: fig_19_pv_ride_through,
    27: fig_27_decpomdp,
    43: fig_43_coordination_diagram,
    44: fig_44_fault_contribution,
    45: fig_45_daily_profile,
    46: fig_46_relay_curves,
    47: fig_47_sympathetic_tripping,

    # Section 2: Methodology, Algorithms & Formulations
    26: fig_26_mappo_flowchart,
    "26a": fig_26a_ctde_flowchart,
    28: fig_28_safe_rl_dual,
    29: fig_29_ewc_continual,
    30: fig_30_pseudocode,
    34: fig_34_nn_architecture,
    35: fig_35_observation_space,
    36: fig_36_action_space_geometry,

    # Section 3: Training Dynamics & Learning Curves
    7: fig_07_curriculum,
    8: fig_08_attention_weights,
    9: fig_09_cti_violations,
    10: fig_10_lagrange_multipliers,
    12: fig_12_convergence_bands,
    16: fig_16_ewc_effectiveness,
    21: fig_21_action_projection,
    37: fig_37_per_agent_curves,
    38: fig_38_attention_evolution,

    # Section 4: Performance Evaluation
    5: fig_05_objective_vs_pv,
    6: fig_06_cti_vs_pv,

    # Section 5: Statistical Analysis & Ranking
    11: fig_11_nemenyi,
    31: fig_31_multi_metric_ranking,
    32: fig_32_effect_size,
    33: fig_33_radar_chart,
    55: fig_55_comparison_table,
    56: fig_56_pareto_front,
    57: fig_57_bootstrap_ci,

    # Section 6: Robustness & Resilience
    13: fig_13_communication_stress,
    14: fig_14_n1_contingency,
    20: fig_20_byzantine,
    22: fig_22_bandwidth_tradeoff,
    24: fig_24_sensor_robustness,
    54: fig_54_adversarial,

    # Section 7: Scalability & Transfer Learning
    15: fig_15_scalability,
    23: fig_23_unseen_faults,
    25: fig_25_zero_shot_transfer,
    52: fig_52_digital_twin_pipeline,

    # Section 8: Deployment & Real-Time Implementation
    18: fig_18_inference_latency,
    51: fig_51_protocol_stack,
    53: fig_53_model_compression,

    # Section 9: Neural Architecture & Interpretability
    48: fig_48_shap,
    49: fig_49_decision_boundary,
    50: fig_50_critical_pairs,

    # Section 10: Ablation & Sensitivity Analysis
    39: fig_39_ablation,
    40: fig_40_hyperparameter,
    41: fig_41_sample_efficiency,
    42: fig_42_warm_start,

    # Section 11: Supplementary Analysis
    58: fig_58_seasonal,
}

# Manual/conceptual figures that require no experiment
MANUAL_FIGURES = {1, 17, 19, 26, "26a", 27, 28, 30, 34, 35, 51, 52}

# ==============================================================================
# PHASE DEFINITIONS (Aligned with Figure Guide Document)
# ==============================================================================

def run_phase1():
    """Phase 1 (Week 1): Manual diagrams + Architecture

    Figures: 1, 3, 4, 17, 19, 26, 26a, 27, 28, 29, 30, 34, 35, 51, 52
    Est. Time: ~1 week (mostly manual work)
    """
    log("\\n" + "="*70)
    log("PHASE 1: Manual Diagrams & Architecture (Week 1)")
    log("="*70)
    fig_01_system_architecture()
    fig_02_data_flow()
    fig_03_8bus_topology()
    fig_04_30bus_topology()
    fig_17_iec61850_architecture()
    fig_19_pv_ride_through()
    fig_26_mappo_flowchart()
    fig_26a_ctde_flowchart()
    fig_27_decpomdp()
    fig_28_safe_rl_dual()
    fig_29_ewc_continual()
    fig_30_pseudocode()
    fig_34_nn_architecture()
    fig_35_observation_space()
    fig_51_protocol_stack()
    fig_52_digital_twin_pipeline()

def run_phase2():
    """Phase 2 (Week 2-3): Core Training Runs — START FIRST (longest)"""
    log("\\n" + "="*70)
    log("PHASE 2: Core Training Runs — START FIRST (Weeks 2-3)")
    log("="*70)
    fig_12_convergence_bands()
    fig_05_objective_vs_pv()
    fig_06_cti_vs_pv()
    fig_39_ablation()
    fig_40_hyperparameter()

def run_phase3():
    """Phase 3 (Week 4): Training Dynamics"""
    log("\\n" + "="*70)
    log("PHASE 3: Training Dynamics (Week 4)")
    log("="*70)
    fig_07_curriculum()
    fig_09_cti_violations()
    fig_10_lagrange_multipliers()
    fig_16_ewc_effectiveness()
    fig_21_action_projection()
    fig_37_per_agent_curves()
    fig_38_attention_evolution()

def run_phase4():
    """Phase 4 (Week 5): Robustness & Resilience"""
    log("\\n" + "="*70)
    log("PHASE 4: Robustness & Resilience (Week 5)")
    log("="*70)
    fig_13_communication_stress()
    fig_14_n1_contingency()
    fig_20_byzantine()
    fig_22_bandwidth_tradeoff()
    fig_24_sensor_robustness()
    fig_54_adversarial()

def run_phase5():
    """Phase 5 (Week 6): Scalability & Transfer Learning"""
    log("\\n" + "="*70)
    log("PHASE 5: Scalability & Transfer Learning (Week 6)")
    log("="*70)
    fig_15_scalability()
    fig_23_unseen_faults()
    fig_25_zero_shot_transfer()

def run_phase6():
    """Phase 6 (Week 7): Deployment & Real-Time Implementation"""
    log("\\n" + "="*70)
    log("PHASE 6: Deployment & Hardware (Week 7)")
    log("="*70)
    fig_18_inference_latency()
    fig_53_model_compression()

def run_phase7():
    """Phase 7 (Week 8): Protection Engineering & Interpretability"""
    log("\\n" + "="*70)
    log("PHASE 7: Protection Engineering & Interpretability (Week 8)")
    log("="*70)
    fig_43_coordination_diagram()
    fig_44_fault_contribution()
    fig_45_daily_profile()
    fig_46_relay_curves()
    fig_47_sympathetic_tripping()
    fig_48_shap()
    fig_49_decision_boundary()
    fig_50_critical_pairs()

def run_phase8():
    """Phase 8 (Week 9): Statistical Aggregation"""
    log("\\n" + "="*70)
    log("PHASE 8: Statistical Aggregation (Week 9)")
    log("="*70)
    fig_11_nemenyi()
    fig_31_multi_metric_ranking()
    fig_32_effect_size()
    fig_33_radar_chart()
    fig_55_comparison_table()
    fig_56_pareto_front()
    fig_57_bootstrap_ci()
    fig_58_seasonal()

def run_phase9():
    """Phase 9 (Week 10): Final Compilation"""
    log("\\n" + "="*70)
    log("PHASE 9: Final Compilation (Week 10)")
    log("="*70)
    log("  -> Compile all figures from results/figs_*/")
    log("  -> Cross-reference with thesis document")
    log("  -> Generate final LaTeX figure inclusion commands")
    log("  -> Run figure consistency checks")

    report = {
        "timestamp": datetime.now().isoformat(),
        "fast_mode": FAST_MODE,
        "output_dir": str(OUTPUT_DIR),
        "figures_expected": 58,
        "manual_figures": list(MANUAL_FIGURES),
        "phases": list(range(1, 10)),
        "total_compute_estimate": "30-40 GPU-days (full)" if not FAST_MODE else "~2-4 hours (fast)"
    }

    report_path = OUTPUT_DIR / "compilation_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    log(f"  -> Compilation report saved to {report_path}")

def run_all():
    """Run the complete experiment suite (all 9 phases)."""
    log("\\n" + "="*70)
    log("DOCR-MARL THESIS: FULL EXPERIMENT SUITE")
    log("WARNING: This will take ~30-40 GPU-days in full mode!")
    log("="*70)

    run_phase1()
    run_phase2()
    run_phase3()
    run_phase4()
    run_phase5()
    run_phase6()
    run_phase7()
    run_phase8()
    run_phase9()

    log("\\n" + "="*70)
    log("ALL PHASES COMPLETE")
    log("="*70)

def run_single_figure(fig_num):
    """Run a single figure by number."""
    if fig_num in FIGURE_FUNCTIONS:
        func = FIGURE_FUNCTIONS[fig_num]
        if fig_num in MANUAL_FIGURES:
            log(f"\\n[FIG {fig_num}] MANUAL/CONCEPTUAL — No experiment script.")
            log("  -> See instructions in the function docstring.")
        func()
    else:
        log(f"ERROR: Figure {fig_num} not found in mapping.")
        log(f"Valid figures: {sorted([k for k in FIGURE_FUNCTIONS.keys() if isinstance(k, int)])}")
        log(f"Valid string figures: {[k for k in FIGURE_FUNCTIONS.keys() if isinstance(k, str)]}")
        sys.exit(1)

# ==============================================================================
# CLI
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="DOCR-MARL Thesis Experiment Runner (Figures 1-58)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Execution Order (from Figure Guide):
  Phase 1 (Week 1):  Manual diagrams + architecture
  Phase 2 (Week 2-3): Core training runs — START FIRST (longest)
  Phase 3 (Week 4):   Training dynamics
  Phase 4 (Week 5):   Robustness & resilience
  Phase 5 (Week 6):   Scalability & transfer
  Phase 6 (Week 7):   Deployment & hardware
  Phase 7 (Week 8):   Protection engineering
  Phase 8 (Week 9):   Statistical aggregation
  Phase 9 (Week 10):  Final compilation

Examples:
  python run_thesis_experiments.py --phase2 --fast        # Quick smoke test
  python run_thesis_experiments.py --figure 5             # Single figure
  python run_thesis_experiments.py --all                  # Full suite
        """
    )

    parser.add_argument("--all", action="store_true", help="Run all 9 phases (~30-40 GPU-days)")
    parser.add_argument("--phase1", action="store_true", help="Week 1: Manual diagrams + architecture")
    parser.add_argument("--phase2", action="store_true", help="Weeks 2-3: Core training (START FIRST)")
    parser.add_argument("--phase3", action="store_true", help="Week 4: Training dynamics")
    parser.add_argument("--phase4", action="store_true", help="Week 5: Robustness & resilience")
    parser.add_argument("--phase5", action="store_true", help="Week 6: Scalability & transfer")
    parser.add_argument("--phase6", action="store_true", help="Week 7: Deployment & hardware")
    parser.add_argument("--phase7", action="store_true", help="Week 8: Protection engineering")
    parser.add_argument("--phase8", action="store_true", help="Week 9: Statistical aggregation")
    parser.add_argument("--phase9", action="store_true", help="Week 10: Final compilation")
    parser.add_argument("--figure", type=str, help="Run single figure (e.g., 5, 26a, 58)")
    parser.add_argument("--fast", action="store_true", help="Reduced iterations for testing")
    parser.add_argument("--list", action="store_true", help="List all figures and their status")

    args = parser.parse_args()

    # Handle --list
    if args.list:
        print("\\nDOCR-MARL Thesis Figures (1-58):")
        print("="*70)
        for fig_num in sorted(FIGURE_FUNCTIONS.keys(), key=lambda x: (isinstance(x, str), x)):
            func = FIGURE_FUNCTIONS[fig_num]
            status = "MANUAL" if fig_num in MANUAL_FIGURES else "SCRIPT"
            doc = func.__doc__.split("\\n")[0] if func.__doc__ else "No description"
            print(f"  Fig {fig_num:<4} [{status:<7}] {doc}")
        print(f"\\nTotal: {len(FIGURE_FUNCTIONS)} figures ({len(MANUAL_FIGURES)} manual, {len(FIGURE_FUNCTIONS) - len(MANUAL_FIGURES)} scripted)")
        return

    # Handle --fast
    global FAST_MODE, N_RUNS, N_RUNS_BASELINE, N_RUNS_HPARAM
    if args.fast:
        FAST_MODE = True
        N_RUNS = "2"
        N_RUNS_BASELINE = "2"
        N_RUNS_HPARAM = "1"
        os.environ["DOCR_FAST"] = "1"
        log("\\n[FAST MODE] Using reduced iterations for testing.\\n")

    # Output subdirectories already created at module load time

    # Route to appropriate runner
    if args.all:
        run_all()
    elif args.phase1:
        run_phase1()
    elif args.phase2:
        run_phase2()
    elif args.phase3:
        run_phase3()
    elif args.phase4:
        run_phase4()
    elif args.phase5:
        run_phase5()
    elif args.phase6:
        run_phase6()
    elif args.phase7:
        run_phase7()
    elif args.phase8:
        run_phase8()
    elif args.phase9:
        run_phase9()
    elif args.figure:
        run_single_figure(args.figure)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()