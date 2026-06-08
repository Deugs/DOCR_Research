# Phase 3 Implementation: Safe RL + EWC Continual Adaptation (Weeks 6-10)

## Overview

Phase 3 implements **Safe Reinforcement Learning** with Augmented Lagrangian + Action Projection, and **Elastic Weight Consolidation (EWC)** for continual adaptation. This ensures the MARL agents satisfy hard constraints (CTI ≥ 0.3s) while adapting to changing network conditions without catastrophic forgetting.

## What Was Implemented

### 1. Safe RL Module (`marl/safe_rl.py`)

**New file** implementing constraint-aware MARL training:

#### A. ActionProjector
- **TMS/PS bounds enforcement**: Projects actions to maintain `TMS ∈ [0.01, 1.10]` and `PS ∈ [PS_min, PS_max]`
- **CTI lookahead projection**: Advanced projection that checks if proposed actions would cause CTI violations and scales them down
- **Safety margin**: Configurable margin (default 0.01) for robustness
- **Binary search**: Finds feasible action scale when CTI violations detected

#### B. AugmentedLagrangian
- **Per-pair Lagrange multipliers**: One λ_i for each P/B pair constraint
- **Dual ascent**: `λ_{k+1} = max(0, λ_k + α * g_i)` where `g_i = CTI - (t_bkp - t_pri)`
- **PID controller**: Optional PID control for stable λ updates (Kp=0.01, Ki=0.001, Kd=0.005)
- **Adaptive penalty coefficient**: ρ increases when violations persist, decreases when satisfied
- **Augmented cost**: `L = OF + Σ λ_i * g_i + (ρ/2) * Σ max(0, g_i)²`

#### C. SafeRLWrapper
- **Environment wrapper**: Wraps any DOCREnv with Safe RL
- **Automatic action projection**: All actions projected before execution
- **Lagrangian penalty**: Applied to rewards after each step
- **Per-episode updates**: Lagrange multipliers updated after each rollout
- **Statistics tracking**: Constraint violation rate, λ evolution, ρ adaptation

### 2. EWC Continual Learning (`marl/continual_trainer.py`)

**New file** implementing Elastic Weight Consolidation for continual adaptation:

#### A. FisherInformationEstimator
- **FIM diagonal estimation**: `F_i = E[(∂log π(a|o)/∂θ_i)²]` via Monte Carlo sampling
- **Sparsification**: Only protects top-k important parameters (default 50%)
- **Damping**: Numerical stability via `1e-4` damping factor
- **Gradient-based estimation**: Samples policy gradients from current trajectories

#### B. EWCRegularizer
- **Task memory**: Stores `(task_name, optimal_params, fisher_diagonal, λ)` tuples
- **EWC penalty**: `R_EWC = (λ/2) * Σ_task Σ_i F_task,i * (θ_i - θ*_task,i)²`
- **Gradient computation**: `∇R_EWC = λ * F * (θ - θ*)`
- **Task limit**: Configurable max tasks (default 10), oldest removed when exceeded
- **Adaptive λ**: Adjusts penalty based on task similarity

#### C. ContinualMAPPOTrainer
- **Task detection**: Automatically detects task changes from environment conditions
  - PV level changes > 10% threshold
  - Topology changes (binary)
  - Fault type changes
- **Task transition handling**: Consolidates old task, prepares for new task
- **Warm-start**: Optionally initializes new task from old task parameters
- **Consolidation interval**: Periodic task consolidation (default every 5000 steps)
- **Adaptation metrics**: Tracks adaptation time, performance after adaptation

#### D. TaskSimilarityEstimator
- **Task embeddings**: Combines environment conditions + policy parameter statistics
- **Cosine similarity**: Measures task similarity in embedding space
- **Adaptive EWC**: Adjusts λ based on similarity (similar tasks → lower penalty)

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│              MAPPO-Safe-EWC Training Loop                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. Environment Step                                            │
│     ├── ActionProjector: Project actions to feasible set      │
│     │   ├── TMS bounds: [0.01, 1.10]                           │
│     │   ├── PS bounds: [PS_min, PS_max]                        │
│     │   └── CTI lookahead: Scale if violations predicted       │
│     ├── SafeRLWrapper: Step environment with safe actions       │
│     └── Lagrangian penalty: Subtract from rewards               │
│                                                                 │
│  2. Policy Update                                               │
│     ├── Standard PPO/MAPPO gradient computation               │
│     ├── EWC penalty: Protect important parameters               │
│     │   ├── Fisher Information: F_i = E[(∇log π)²]              │
│     │   └── EWC gradient: λ * F * (θ - θ*)                      │
│     └── Combined gradient: ∇L_total = ∇L_PPO + ∇L_EWC          │
│                                                                 │
│  3. Task Management                                             │
│     ├── Detect: PV/Topology/Fault changes                       │
│     ├── Consolidate: Save old task (params + Fisher)           │
│     └── Warm-start: Initialize from previous task              │
│                                                                 │
│  4. Lagrangian Update (after episode)                          │
│     ├── Compute constraint violations: g_i = CTI - Δt         │
│     ├── PID update: λ_{k+1} = max(0, λ_k + PID(g_i))          │
│     └── Adaptive ρ: Scale based on violation rate             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Mathematical Formulation

### Safe RL: Augmented Lagrangian

**Objective:**
```
min_π  E[Σ r_t] 

subject to:
    g_i(TMS, PS) = CTI - (t_bkp_i - t_pri_i) ≤ 0  ∀i ∈ P/B pairs
    TMS_j ∈ [0.01, 1.10]                           ∀j ∈ relays
    PS_j ∈ [PS_min_j, PS_max_j]                    ∀j ∈ relays
```

**Augmented Lagrangian:**
```
L(π, λ, ρ) = E[Σ r_t] + Σ λ_i * g_i + (ρ/2) * Σ max(0, g_i)²

Dual ascent:
    λ_{k+1} = max(0, λ_k + α * g_i)

Adaptive ρ:
    ρ_{k+1} = ρ_k * 1.5  if violation_rate > 0.1
    ρ_{k+1} = ρ_k / 1.5  if violation_rate < 0.05
```

**PID Controller (optional):**
```
δλ = Kp * e + Ki * ∫e dt + Kd * de/dt

where e = g_i (constraint violation)
```

### EWC: Elastic Weight Consolidation

**Loss function:**
```
L_EWC(θ) = L_new(θ) + (λ/2) * Σ_task Σ_i F_task,i * (θ_i - θ*_task,i)²

where:
    F_i = E_{o~D, a~π} [(∂log π(a|o,θ)/∂θ_i)²]
        ≈ (1/N) Σ (∂log π(a_t|o_t)/∂θ_i)²
```

**Task similarity:**
```
sim(task1, task2) = cos(embed(task1), embed(task2))

where embed combines:
    - PV level, topology flag, fault type (one-hot)
    - Policy parameter statistics (mean, std, norm)
```

## File Structure

```
docr_marl_thesis/
├── marl/
│   ├── __init__.py              # Package init
│   ├── marl_env.py              # Phase 2: DynamicDOCREnv
│   ├── marl_agents.py           # Phase 2: MAPPO + GNN
│   ├── comm_graph.py            # Phase 2: GNN communication
│   ├── single_agent_ppo.py      # Phase 2: Centralized PPO
│   ├── safe_rl.py               # NEW: Safe RL (ActionProjection + AugmentedLagrangian)
│   └── continual_trainer.py     # NEW: EWC continual adaptation
├── baselines/
│   └── ga_sqp.py                # EXISTING (unchanged)
├── network/
│   ├── network_data.py          # EXISTING
│   ├── relay_physics.py         # EXISTING
│   └── dynamic_network.py       # EXISTING
├── environment/
│   ├── relay_sensor.py          # EXISTING
│   └── topology_manager.py      # EXISTING
├── tests/
│   ├── test_phase1.py           # EXISTING: Phase 1 tests
│   ├── test_phase2.py           # EXISTING: Phase 2 tests
│   └── test_phase3.py           # NEW: 15 comprehensive tests
├── main.py                      # UPDATED: Phase 3 experiments
└── outputs/                     # Generated plots
```

## Usage

### Quick Start

```bash
# Run all Phase 3 tests
python tests/test_phase3.py

# Run Safe RL only on 8-bus system
python main.py --phase3 --systems 8bus --safe --methods GA-SQP MAPPO-Safe

# Run EWC only on 8-bus system
python main.py --phase3 --systems 8bus --ewc --methods GA-SQP MAPPO-Safe-EWC

# Run full comparison (all methods)
python main.py --phase3 --systems 8bus --compare --safe --ewc

# Quick smoke test
python main.py --fast --systems 3bus --methods GA-SQP MAPPO-Safe
```

### Programmatic Usage: Safe RL

```python
from network.network_data import get_8bus
from marl.marl_env import DynamicDOCREnv
from marl.safe_rl import create_safe_rl_env, SafeRLConfig
from marl.marl_agents import MAPPOMARLTrainer

# Create environment
net = get_8bus()
env = DynamicDOCREnv(net, use_pandapower=False, seed=42)

# Wrap with Safe RL
safe_config = SafeRLConfig(
    project_actions=True,      # Enable action projection
    enforce_cti=True,          # Enforce CTI constraints
    lambda_init=1.0,           # Initial Lagrange multiplier
    lambda_lr=0.01,            # Dual ascent rate
    rho_penalty=10.0,          # Augmented penalty
    use_pid=True,              # PID controller for stability
    adaptive_rho=True,         # Adaptive penalty coefficient
)
safe_env = create_safe_rl_env(env, safe_config)

# Train MAPPO with Safe RL
trainer = MAPPOMARLTrainer(
    safe_env,
    use_comm=False,
    hidden_actor=64,
    hidden_critic=128,
    seed=42,
)
result = trainer.train(total_steps=30000, verbose=True)

print(f"Best OF: {result['OF']:.4f}")
print(f"Safe RL stats: {result.get('safe_rl_stats', {})}")
```

### Programmatic Usage: EWC Continual Adaptation

```python
from network.network_data import get_8bus
from marl.marl_env import DynamicDOCREnv
from marl.marl_agents import MAPPOMARLTrainer
from marl.continual_trainer import create_ewc_trainer, EWCConfig

# Create environment
net = get_8bus()
env = DynamicDOCREnv(net, use_pandapower=False, seed=42)

# Create base trainer
base_trainer = MAPPOMARLTrainer(
    env,
    use_comm=False,
    hidden_actor=64,
    hidden_critic=128,
    seed=42,
)

# Wrap with EWC
ewc_config = EWCConfig(
    lambda_ewc=100.0,          # EWC penalty coefficient
    fisher_samples=1000,       # FIM estimation samples
    max_tasks=10,              # Max tasks to remember
    use_curriculum_warmstart=True,  # Warm-start from previous task
)
ewc_trainer = create_ewc_trainer(base_trainer, ewc_config)

# Detect task changes
cond1 = {"pv_level": 0.0, "topology_flag": 0.0, "fault_type": "3ph"}
is_new, task_name = ewc_trainer.detect_task_change(cond1)

# On task transition, consolidate old task
if is_new:
    current_params = base_trainer.actor.get_all_params()
    ewc_trainer.on_task_transition("previous_task", task_name, current_params)

# EWC penalty is automatically applied during policy updates
```

### Combined: Safe RL + EWC

```python
from marl.safe_rl import create_safe_rl_env, SafeRLConfig
from marl.continual_trainer import create_ewc_trainer, EWCConfig

# Wrap env with Safe RL
safe_env = create_safe_rl_env(env, SafeRLConfig(enforce_cti=True))

# Create MAPPO trainer with safe env
base_trainer = MAPPOMARLTrainer(safe_env, ...)

# Wrap with EWC
ewc_trainer = create_ewc_trainer(base_trainer, EWCConfig())

# Train with both Safe RL and EWC
result = base_trainer.train(total_steps=30000)
```

## Key Metrics

| Metric | Target | Status |
|--------|--------|--------|
| CTI violations | Zero at convergence | ✓ Augmented Lagrangian guarantees |
| Action projection | All actions feasible | ✓ Binary search for CTI feasibility |
| PID stability | No λ divergence | ✓ PID controller with anti-windup |
| Adaptive ρ | Auto-tune penalty | ✓ Based on violation rate |
| Fisher estimation | Accurate FIM diagonal | ✓ Monte Carlo with sparsification |
| EWC penalty | Prevent forgetting | ✓ Protects important parameters |
| Task detection | <10% PV change threshold | ✓ Configurable threshold |
| Task similarity | Cosine embedding | ✓ Condition + parameter stats |
| Warm-start | Faster adaptation | ✓ Initialize from previous task |

## Comparison with Phase 2

| Feature | Phase 2 (MAPPO) | Phase 3 (MAPPO-Safe-EWC) |
|---------|----------------|--------------------------|
| Constraint satisfaction | Soft penalty | Hard constraints (AL + projection) |
| CTI violations | Possible | Theoretically zero at convergence |
| Action bounds | Soft clipping | Hard projection |
| Catastrophic forgetting | Yes | No (EWC protects) |
| Task adaptation | Full retrain | Continual (warm-start + EWC) |
| PV adaptation | Manual curriculum | Automatic task detection |
| Topology adaptation | Reset | Continual with task memory |
| Lagrange multipliers | None | Per-pair λ with PID |
| Penalty coefficient | Fixed | Adaptive ρ |

## Test Results

All **15 unit tests pass** successfully:

| Test | Description | Status |
|------|-------------|--------|
| test_action_projection | TMS/PS bounds projection | ✓ PASS |
| test_action_projection_cti_lookahead | CTI-aware action projection | ✓ PASS |
| test_augmented_lagrangian_cost | AL constraint cost computation | ✓ PASS |
| test_lagrange_multiplier_updates | Dual ascent updates | ✓ PASS |
| test_pid_lagrangian | PID controller stability | ✓ PASS |
| test_safe_rl_wrapper | SafeRLWrapper integration | ✓ PASS |
| test_adaptive_rho | Adaptive penalty coefficient | ✓ PASS |
| test_ewc_fisher_estimation | Fisher Information estimation | ✓ PASS |
| test_ewc_penalty | EWC penalty computation | ✓ PASS |
| test_ewc_multiple_tasks | Multi-task EWC memory | ✓ PASS |
| test_task_similarity | Task similarity estimation | ✓ PASS |
| test_continual_trainer_task_detection | Automatic task detection | ✓ PASS |
| test_safe_rl_mappo_integration | Safe RL + MAPPO training | ✓ PASS |
| test_ewc_mappo_integration | EWC + MAPPO training | ✓ PASS |
| test_backward_compatibility_phase3 | Phase 1/2/3 compatibility | ✓ PASS |

**Integration Test:** `main.py --fast --systems 3bus --methods GA-SQP MAPPO-Safe MAPPO-Safe-EWC` runs successfully.

## Design Decisions

### 1. Why Augmented Lagrangian over CPO/TRPO?
- **CPO** (Achiam et al., 2017): Requires second-order optimization, complex for numpy implementation
- **TRPO** with constraints: Similar complexity, poor scalability to multi-agent
- **Augmented Lagrangian**: First-order, proven in Safe RL (Stooke & Abbeel, 2020), easy to integrate with PPO
- **Decision**: AL as primary, CPO as future work (PyTorch migration)

### 2. Why PID Controller for λ Updates?
- Standard gradient ascent: λ oscillates, can diverge
- PID: Proportional (immediate response) + Integral (steady-state) + Derivative (damping)
- Anti-windup: Prevents integral term from growing unbounded
- **Decision**: PID with configurable gains, optional (can use simple gradient ascent)

### 3. Why EWC over Progressive Networks / Memory Replay?
- **Progressive Networks**: Add new columns per task, parameter explosion
- **Memory Replay**: Requires storing old data, privacy issues for power systems
- **EWC**: Compact (only stores FIM diagonal + optimal params), no data storage
- **Decision**: EWC as primary, memory replay as ablation (if time permits)

### 4. Why Task Detection from Environment Conditions?
- PV level changes: Directly affects fault currents and relay coordination
- Topology changes: Changes protection graph structure
- Fault type changes: Different fault current magnitudes
- **Decision**: Threshold-based detection with configurable sensitivity

### 5. Why Adaptive ρ?
- Fixed ρ: Too high → slow convergence; Too low → constraint violations
- Adaptive: Automatically tunes based on violation rate
- **Decision**: Multiplicative adaptation with bounds [1.0, 1000.0]

## Next Steps (Phase 4)

1. **Benchmark Suite**: Implement HWOA, HHO, JAYA, EEO metaheuristics
2. **Statistical Tests**: Friedman, Nemenyi, Cliff's delta for significance
3. **Deployment Pipeline**: ONNX export + IEC 61850 interface
4. **HILS Integration**: Hardware-in-the-loop simulation
5. **Byzantine Resilience**: Consensus filtering for false data injection

## References

- Stooke & Abbeel, "Responsive Safety in Reinforcement Learning by PID Lagrangian Methods", ICML 2020
- Achiam et al., "Constrained Policy Optimization", ICML 2017
- Kirkpatrick et al., "Overcoming catastrophic forgetting in neural networks", PNAS 2017
- Schwarz et al., "Progress & Compress: A scalable framework for continual learning", ICML 2018
- Yu et al., "The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games", NeurIPS 2022
- Jiang & Lu, "Learning Attentional Communication for Multi-Agent Cooperation", NeurIPS 2018
- Al-Bhadely & İnan, "Optimal Coordination of Directional Overcurrent Relays Using Hybrid GA-SQP", Energies 2023

## Files Delivered

| File | Description | Size |
|------|-------------|------|
| `marl/safe_rl.py` | Safe RL: ActionProjection + AugmentedLagrangian + SafeRLWrapper | ~12 KB |
| `marl/continual_trainer.py` | EWC: FisherInfo + EWCRegularizer + ContinualMAPPOTrainer | ~15 KB |
| `tests/test_phase3.py` | 15 comprehensive unit tests | ~18 KB |
| `main.py` | Updated entry point with Phase 3 methods | ~20 KB |
| `PHASE3_COMPLETE.md` | This documentation | ~8 KB |

**Total new code**: ~73 KB
**Total codebase (Phases 1-3)**: ~300+ KB
