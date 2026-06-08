# Phase 2 Implementation: Decentralized Agent Architecture (Weeks 4-5)

## Overview

Phase 2 implements the decentralized agent architecture with GNN-based learned communication, following the unified methodology roadmap. This phase delivers the core multi-agent coordination framework that distinguishes the thesis contribution.

## What Was Implemented

### 1. GNN Communication Module (`marl/comm_graph.py`)

**New file** implementing learned communication between relay agents:

- **GNNMessageEncoder**: Compresses local observations into 8-float messages (~32 bytes)
- **GNNAttention**: Learns attention weights over relay adjacency graph
- **GNNMessageDecoder**: Combines own observation with aggregated neighbor messages
- **CommGraph**: End-to-end communication module with bandwidth tracking
- **build_relay_adjacency()**: Constructs communication graph from protection pairs

**Key Design Decisions:**
- Message dimension: 8 floats (32 bytes) — bandwidth-efficient
- Attention mechanism: GAT-style learned weights
- Bandwidth limit: 1 kbps per agent (practical for RF mesh/AMI)
- Graph topology: Derived from relay primary/backup pairs + same-bus connections

### 2. Single-Agent PPO Baseline (`marl/single_agent_ppo.py`)

**New file** — centralized PPO controlling all relays simultaneously:

- **CentralizedGaussianActor**: Single policy outputs [TMS_1..N, PS_1..N]
- **CentralizedCritic**: Value function on global state
- **CentralizedPPOEnvironment**: Global observation wrapper
- **SinglePPOTrainer**: PPO with ES-based policy gradient

**Purpose:** Proves that decentralization (MARL) beats centralization even with RL.

### 3. MAPPO with GNN Communication (`marl/marl_agents.py`)

**Major refactor** of Phase 1 IPPO to MAPPO:

- **SharedGaussianActor**: Parameter sharing across all agents (reduces sample complexity)
- **MAPPOCentralCritic**: Centralized value function on global state + neighbor stats
- **MAPPOBuffer**: Rollout buffer with communication tracking
- **MAPPOMARLTrainer**: Full MAPPO training loop with optional GNN communication

**Key Improvements over Phase 1 IPPO:**
- Centralized critic reduces variance (CTDE paradigm)
- Parameter sharing enables scaling to larger networks
- Optional GNN communication for coordination
- Global state includes neighbor statistics

### 4. Enhanced Environment (`marl/marl_env.py`)

**Refactored** with Phase 2 features:

- **Neighbor observations**: `get_neighbor_observations()` returns neighbor features per agent
- **Neighbor statistics**: TMS/PS/violation/slack from connected relays
- **Communication interface**: Ready for GNN message passing
- **Backward compatible**: All Phase 1 functionality preserved

### 5. Phase 2 Entry Point (`main.py`)

**Rewritten** to compare all methods:

| Method | Description | Purpose |
|--------|-------------|---------|
| GA-SQP | Classical optimization | Control baseline |
| Single-PPO | Centralized RL | Prove decentralization value |
| IPPO | Decentralized, no comm | Phase 1 baseline |
| MAPPO | Centralized critic, no comm | Core MARL algorithm |
| MAPPO-Comm | MAPPO + GNN communication | Full thesis contribution |

### 6. Comprehensive Tests (`tests/test_phase2.py`)

**13 unit tests** covering:
1. GNN encoder encoding/decoding
2. GNN attention mechanism
3. GNN decoder
4. CommGraph end-to-end
5. Relay adjacency for all test systems
6. Neighbor observations
7. Shared Gaussian actor (parameter sharing)
8. MAPPO centralized critic
9. MAPPO buffer with communication
10. Single-PPO integration
11. MAPPO integration with environment
12. Bandwidth tracking
13. Backward compatibility with Phase 1

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    MAPPO with GNN Communication              │
├─────────────────────────────────────────────────────────────┤
│  Agent 1    Agent 2    ...    Agent N                       │
│    │          │                │                            │
│    ▼          ▼                ▼                            │
│  ┌────┐    ┌────┐          ┌────┐                         │
│  │Obs │    │Obs │          │Obs │  (10-dim local obs)      │
│  └──┬─┘    └──┬─┘          └──┬─┘                         │
│     │         │               │                            │
│     ▼         ▼               ▼                            │
│  ┌────────┐ ┌────────┐   ┌────────┐                       │
│  │Encoder │ │Encoder │   │Encoder │  (obs -> 8-float msg)  │
│  └───┬────┘ └───┬────┘   └───┬────┘                       │
│      │          │             │                            │
│      └──────────┼─────────────┘                            │
│                 ▼                                           │
│          ┌────────────┐                                     │
│          │  GNN Graph  │  (attention-weighted aggregation)   │
│          │  (learned)  │                                     │
│          └─────┬──────┘                                     │
│                │                                              │
│      ┌────────┼────────┐                                    │
│      ▼        ▼        ▼                                    │
│  ┌────────┐ ┌────────┐ ┌────────┐                            │
│  │Decoder │ │Decoder │ │Decoder │  ([obs||msg] -> enhanced)│
│  └───┬────┘ └───┬────┘ └───┬────┘                            │
│      │          │           │                                │
│      ▼          ▼           ▼                                │
│  ┌────────┐ ┌────────┐ ┌────────┐                          │
│  │ Policy │ │ Policy │ │ Policy │  (shared Gaussian actor)   │
│  │(shared)│ │(shared)│ │(shared)│                            │
│  └───┬────┘ └───┬────┘ └───┬────┘                            │
│      │          │           │                                │
│      ▼          ▼           ▼                                │
│   [ΔTMS,ΔPS] [ΔTMS,ΔPS] [ΔTMS,ΔPS]                         │
│                                                              │
│  Centralized Critic (global state + neighbor stats)         │
│  └── Value estimate for variance reduction                  │
└─────────────────────────────────────────────────────────────┘
```

## File Structure

```
docr_marl_thesis/
├── marl/
│   ├── __init__.py              # Package init
│   ├── marl_env.py              # REFACTORED: +neighbor observations
│   ├── marl_agents.py           # REFACTORED: MAPPO + GNN communication
│   ├── comm_graph.py            # NEW: GNN encoder/decoder/attention
│   └── single_agent_ppo.py      # NEW: Centralized PPO baseline
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
│   └── test_phase2.py           # NEW: 13 comprehensive tests
├── main.py                      # REWRITTEN: Phase 2 experiments
└── outputs/                     # Generated plots
```

## Usage

### Quick Start

```bash
# Run all Phase 2 tests
python tests/test_phase2.py

# Run comparison on 8-bus system
python main.py --phase2 --systems 8bus --compare

# Run with GNN communication
python main.py --phase2 --systems 8bus --comm --compare

# Quick smoke test
python main.py --fast --systems 3bus --methods GA-SQP IPPO MAPPO

# Full comparison across all methods
python main.py --phase2 --systems 8bus 30bus --methods GA-SQP Single-PPO IPPO MAPPO MAPPO-Comm
```

### Programmatic Usage

```python
from network.network_data import get_8bus
from marl.marl_env import DynamicDOCREnv
from marl.marl_agents import MAPPOMARLTrainer

# Create environment
net = get_8bus()
env = DynamicDOCREnv(net, use_pandapower=False, seed=42)

# Train MAPPO with GNN communication
trainer = MAPPOMARLTrainer(
    env,
    use_comm=True,          # Enable GNN communication
    hidden_actor=64,
    hidden_critic=128,
    seed=42,
)
result = trainer.train(total_steps=30000, verbose=True)

print(f"Best OF: {result['OF']:.4f}")
print(f"Communication bandwidth: {result['comm_bandwidth']:.2f} kbps")
```

### GNN Communication Only

```python
from marl.comm_graph import CommGraph, CommConfig, build_relay_adjacency
from network.network_data import get_8bus

net = get_8bus()
adj = build_relay_adjacency(net)

config = CommConfig(message_dim=8, use_attention=True)
comm = CommGraph(
    n_agents=net.n_relays,
    obs_dim=10,
    enhanced_dim=18,
    adjacency=adj,
    config=config,
)

observations = np.random.randn(net.n_relays, 10)
enhanced_obs, messages, info = comm.forward(observations)

print(f"Messages shape: {messages.shape}")  # (14, 8)
print(f"Bandwidth: {comm.get_bandwidth_usage()['kbps']:.2f} kbps")
```

## Key Metrics

| Metric | Target | Status |
|--------|--------|--------|
| Message dimension | 8 floats (32 bytes) | ✓ Implemented |
| Bandwidth per agent | < 1 kbps | ✓ Verified |
| Parameter sharing | All agents share actor | ✓ Implemented |
| Centralized critic | Global state value | ✓ Implemented |
| Backward compatibility | Phase 1 tests pass | ✓ Verified |

## Comparison with Phase 1

| Feature | Phase 1 (IPPO) | Phase 2 (MAPPO) |
|---------|---------------|-----------------|
| Critic | Per-agent | Centralized (global state) |
| Actor | Per-agent | Shared (parameter sharing) |
| Communication | None | GNN (optional) |
| Baselines | GA-SQP | + Single-PPO, IPPO, MAPPO |
| Observation | 10-dim local | 10-dim + neighbor stats |
| Scalability | Limited | Better (parameter sharing) |

## Next Steps (Phase 3)

1. **Safe RL**: Augmented Lagrangian + action projection for CTI constraints
2. **EWC**: Continual adaptation with Fisher information masking
3. **Benchmark Suite**: HWOA, HHO, JAYA, EEO metaheuristics
4. **Statistical Tests**: Friedman, Nemenyi, Cliff's delta

## References

- Yu et al., "The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games", NeurIPS 2022
- Jiang & Lu, "Learning Attentional Communication for Multi-Agent Cooperation", NeurIPS 2018
- Schulman et al., "Proximal Policy Optimization Algorithms", arXiv 2017


## Test Results

All **13 unit tests pass** successfully:

| Test | Description | Status |
|------|-------------|--------|
| test_gnn_encoder | GNN message encoder (obs -> 8-float msg) | ✓ PASS |
| test_gnn_attention | GAT-style attention mechanism | ✓ PASS |
| test_gnn_decoder | Message decoder (enhanced obs) | ✓ PASS |
| test_comm_graph | End-to-end communication graph | ✓ PASS |
| test_relay_adjacency | Adjacency matrix for all test systems | ✓ PASS |
| test_neighbor_observations | Neighbor feature extraction | ✓ PASS |
| test_shared_gaussian_actor | Parameter sharing across agents | ✓ PASS |
| test_mappo_critic | Centralized critic with global state | ✓ PASS |
| test_mappo_buffer | Rollout buffer with comm tracking | ✓ PASS |
| test_single_ppo | Centralized PPO baseline | ✓ PASS |
| test_mappo_integration | MAPPO + environment integration | ✓ PASS |
| test_bandwidth_tracking | Per-agent bandwidth < 1 kbps | ✓ PASS |
| test_backward_compatibility | Phase 1 compatibility | ✓ PASS |

**Integration Test:** `main.py --fast --systems 3bus --methods GA-SQP IPPO MAPPO` runs successfully.

**Key Metrics Verified:**
- Message dimension: 8 floats (32 bytes) ✓
- Bandwidth per agent: 0.256 kbps (< 1 kbps limit) ✓
- Parameter sharing: All agents share actor weights ✓
- Centralized critic: Global state value estimation ✓
