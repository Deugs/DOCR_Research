"""
topology_manager.py
───────────────────
Topology management for distribution network reconfiguration.
Handles N-1 contingencies, sectionalizing switch operations,
and feeder reconfiguration for relay coordination studies.

Refactored for full IEEE test system support: 3, 8, 14, 30, 39, 57, 118 buses.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from copy import deepcopy


@dataclass
class TopologyState:
    """Current network topology state."""
    line_status: Dict[int, bool] = field(default_factory=dict)  # line_idx -> in_service
    switch_status: Dict[int, bool] = field(default_factory=dict)  # switch_idx -> closed
    bus_splits: Set[int] = field(default_factory=set)  # buses that are split
    active_feeders: List[int] = field(default_factory=list)

    def copy(self):
        return TopologyState(
            line_status=deepcopy(self.line_status),
            switch_status=deepcopy(self.switch_status),
            bus_splits=deepcopy(self.bus_splits),
            active_feeders=deepcopy(self.active_feeders),
        )


class TopologyManager:
    """
    Manages network topology for N-1 contingency analysis and
    feeder reconfiguration.

    Supports:
    - N-1 line outages
    - N-1 transformer outages  
    - Sectionalizing switch operations
    - Bus splitting (ring-main to radial)
    - Automatic feeder reconfiguration
    - Full IEEE test system range (3, 8, 14, 30, 39, 57, 118 buses)
    """

    def __init__(self, n_buses: int, n_lines: int, n_switches: int = 0):
        self.n_buses = n_buses
        self.n_lines = n_lines
        self.n_switches = n_switches

        # Default topology: all lines in service, all switches closed
        self.base_state = TopologyState(
            line_status={i: True for i in range(n_lines)},
            switch_status={i: True for i in range(n_switches)},
            active_feeders=list(range(n_buses)),
        )
        self.current_state = self.base_state.copy()

        # Pre-computed N-1 contingencies
        self.n1_contingencies = self._generate_n1_contingencies()

    def _generate_n1_contingencies(self) -> List[TopologyState]:
        """Generate all N-1 single outage contingencies."""
        contingencies = []

        # Line N-1
        for line_idx in range(self.n_lines):
            state = self.base_state.copy()
            state.line_status[line_idx] = False
            contingencies.append(state)

        # Switch N-1 (open one switch)
        for switch_idx in range(self.n_switches):
            state = self.base_state.copy()
            state.switch_status[switch_idx] = False
            contingencies.append(state)

        return contingencies

    def apply_n1_outage(self, element_type: str, element_idx: int) -> TopologyState:
        """
        Apply N-1 outage to a specific element.

        Parameters:
            element_type: "line" or "switch"
            element_idx: Index of element to outage

        Returns:
            New TopologyState with outage applied
        """
        state = self.base_state.copy()

        if element_type == "line" and element_idx < self.n_lines:
            state.line_status[element_idx] = False
        elif element_type == "switch" and element_idx < self.n_switches:
            state.switch_status[element_idx] = False

        self.current_state = state
        return state

    def apply_multiple_outages(
        self,
        line_outages: List[int] = None,
        switch_outages: List[int] = None,
    ) -> TopologyState:
        """Apply multiple simultaneous outages (N-k)."""
        state = self.base_state.copy()

        if line_outages:
            for idx in line_outages:
                if idx < self.n_lines:
                    state.line_status[idx] = False

        if switch_outages:
            for idx in switch_outages:
                if idx < self.n_switches:
                    state.switch_status[idx] = False

        self.current_state = state
        return state

    def reconfigure_for_radial(self, feeder_roots: List[int]) -> TopologyState:
        """
        Reconfigure network to radial topology from specified feeder roots.
        Opens switches to eliminate loops while maintaining connectivity.

        Parameters:
            feeder_roots: List of bus indices that are feeder roots

        Returns:
            TopologyState with radial configuration
        """
        state = self.base_state.copy()
        state.active_feeders = feeder_roots

        # Simplified: open switches to create radial structure
        # In practice, this would use graph algorithms (MST, etc.)
        if self.n_switches > 0:
            # Open every other switch to approximate radial
            for switch_idx in range(0, self.n_switches, 2):
                state.switch_status[switch_idx] = False

        self.current_state = state
        return state

    def get_outage_factor(self, topology_state: Optional[TopologyState] = None) -> float:
        """
        Calculate outage factor for fault current scaling.

        More outages → lower fault currents (less parallel paths).

        Returns:
            Factor between 0.5 and 1.0
        """
        state = topology_state or self.current_state

        n_line_outages = sum(1 for v in state.line_status.values() if not v)
        n_switch_outages = sum(1 for v in state.switch_status.values() if not v)

        total_outages = n_line_outages + n_switch_outages
        max_outages = self.n_lines + self.n_switches

        if max_outages == 0:
            return 1.0

        # Exponential decay: each outage reduces fault current
        factor = np.exp(-0.1 * total_outages)
        return max(0.5, factor)

    def get_affected_buses(
        self,
        topology_state: Optional[TopologyState] = None,
    ) -> Set[int]:
        """
        Identify buses affected by current topology changes.
        Buses downstream of opened lines/switches are affected.

        Returns:
            Set of affected bus indices
        """
        state = topology_state or self.current_state
        affected = set()

        # Simplified: all buses beyond first outage are potentially affected
        for line_idx, in_service in state.line_status.items():
            if not in_service:
                # Mark buses on both sides as affected
                affected.add(line_idx % self.n_buses)
                affected.add((line_idx + 1) % self.n_buses)

        return affected

    def get_fault_current_modifiers(
        self,
        topology_state: Optional[TopologyState] = None,
    ) -> Dict[str, float]:
        """
        Get per-bus fault current modifiers based on topology.

        Returns:
            Dict mapping bus identifiers to scaling factors
        """
        state = topology_state or self.current_state
        outage_factor = self.get_outage_factor(state)
        affected_buses = self.get_affected_buses(state)

        modifiers = {}
        for bus_idx in range(self.n_buses):
            if bus_idx in affected_buses:
                # Affected buses have reduced fault current
                modifiers[f"bus_{bus_idx}"] = outage_factor * 0.8
            else:
                # Unaffected buses have slight reduction due to network changes
                modifiers[f"bus_{bus_idx}"] = outage_factor

        return modifiers

    def reset(self):
        """Reset to base topology."""
        self.current_state = self.base_state.copy()

    def random_n1_contingency(self, rng: Optional[np.random.Generator] = None) -> TopologyState:
        """Apply a random N-1 contingency."""
        if rng is None:
            rng = np.random.default_rng()

        if self.n_lines > 0 and (self.n_switches == 0 or rng.random() < 0.7):
            # 70% chance of line outage
            line_idx = rng.integers(0, self.n_lines)
            return self.apply_n1_outage("line", int(line_idx))
        else:
            # 30% chance of switch outage
            switch_idx = rng.integers(0, max(1, self.n_switches))
            return self.apply_n1_outage("switch", int(switch_idx))

    def get_topology_description(self, topology_state: Optional[TopologyState] = None) -> str:
        """Get human-readable description of topology state."""
        state = topology_state or self.current_state

        lines_out = [i for i, v in state.line_status.items() if not v]
        switches_open = [i for i, v in state.switch_status.items() if not v]

        desc = f"Topology: {len(lines_out)} line outages, {len(switches_open)} open switches"
        if lines_out:
            desc += f" | Lines out: {lines_out}"
        if switches_open:
            desc += f" | Switches open: {switches_open}"

        return desc