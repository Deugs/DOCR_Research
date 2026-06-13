"""
DOCR-MARL: Directional Over-Current Relay Coordination via Multi-Agent RL
═════════════════════════════════════════════════════════════════════════

Package spanning all IEEE test systems: 3, 8, 14, 30, 39, 57, and 118 buses.

Modules
-------
network.network_data    — Test system data with load_network() universal loader
marl.marl_env           — Multi-agent environment (network-agnostic)
marl.marl_agents        — MAPPO with GNN communication (network-agnostic)
marl.comm_graph         — GNN message passing (network-agnostic)
marl.safe_rl            — Safe RL with Lagrangian constraints (network-agnostic)
marl.continual_trainer  — EWC continual learning (network-agnostic)
marl.single_agent_ppo   — Centralized PPO baseline (network-agnostic)

Quick Start
-----------
>>> from network.network_data import load_network
>>> net = load_network("8bus")      # or "3bus", "14bus", "30bus", "39bus", "57bus", "118bus"
>>> from marl.marl_env import DynamicDOCREnv
>>> env = DynamicDOCREnv(net)
"""

from network.network_data import (
    load_network,
    get_network_metadata,
    NETWORKS,
    NetworkData,
    RelayPair,
    DGConfig,
    get_3bus,
    get_8bus,
    get_14bus,
    get_30bus,
    get_39bus,
    get_57bus,
    get_118bus,
)

__all__ = [
    "load_network",
    "get_network_metadata",
    "NETWORKS",
    "NetworkData",
    "RelayPair",
    "DGConfig",
    "get_3bus",
    "get_8bus",
    "get_14bus",
    "get_30bus",
    "get_39bus",
    "get_57bus",
    "get_118bus",
]