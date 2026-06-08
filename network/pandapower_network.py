"""
pandapower_network.py
─────────────────────
Pandapower integration for dynamic power flow and short-circuit analysis.
Builds pandapower networks from NetworkData, runs power flow and fault studies,
and extracts fault currents for relay coordination.

Phase 1: Baseline Network Modeling with pandapower.
"""

import numpy as np
from typing import Dict, Tuple, Optional, List
import warnings
warnings.filterwarnings("ignore")

# Try to import pandapower; if not available, use fallback
try:
    import pandapower as pp
    import pandapower.shortcircuit as sc
    PANDAPOWER_AVAILABLE = True
except ImportError:
    PANDAPOWER_AVAILABLE = False
    print("Warning: pandapower not installed. Using analytical fallback for fault currents.")

from network.network_data import NetworkData, DGConfig
from network.relay_physics import compute_fault_currents_dynamic


class PandapowerNetwork:
    """
    Wrapper around pandapower network for dynamic analysis.
    Supports: power flow, short-circuit, topology changes, PV integration.
    """

    def __init__(self, net_data: NetworkData):
        self.net_data = net_data
        self.pp_net = None
        self._build_network()

    def _build_network(self):
        """Build pandapower network from NetworkData."""
        if not PANDAPOWER_AVAILABLE:
            self.pp_net = None
            return

        net = pp.create_empty_network(
            name=self.net_data.name,
            f_hz=50.0,
            sn_mva=self.net_data.base_mva
        )

        # Create buses
        bus_map = {}
        for i in range(self.net_data.n_buses):
            bus_map[i] = pp.create_bus(
                net, 
                vn_kv=self.net_data.base_voltage,
                name=f"Bus {i+1}"
            )

        # Add external grid (slack bus) at bus 0
        pp.create_ext_grid(net, bus=bus_map[0], vm_pu=1.0, va_degree=0.0)

        # Add lines (simplified radial/ring topology)
        # For 8-bus: typical distribution feeder
        if self.net_data.n_buses == 8:
            line_pairs = [(0,1), (1,2), (2,3), (1,4), (4,5), (5,6), (6,7), (3,7)]
        elif self.net_data.n_buses == 30:
            # 30-bus simplified topology
            line_pairs = [(i, i+1) for i in range(29)] + [(0, 29), (5, 10), (10, 15), (15, 20)]
        elif self.net_data.n_buses == 39:
            # 39-bus simplified (New England)
            line_pairs = [(i, i+1) for i in range(38)] + [(0, 10), (10, 20), (20, 30)]
        else:
            line_pairs = [(i, (i+1) % self.net_data.n_buses) for i in range(self.net_data.n_buses)]

        for from_bus, to_bus in line_pairs:
            if from_bus < self.net_data.n_buses and to_bus < self.net_data.n_buses:
                pp.create_line(
                    net,
                    from_bus=bus_map[from_bus],
                    to_bus=bus_map[to_bus],
                    length_km=1.0,
                    std_type="NAYY 4x150 SE" if self.net_data.base_voltage <= 20 else "NA2XS2Y 1x185 RM/25 12/20 kV"
                )

        # Add loads based on Iload_max (convert A to MW)
        for i in range(self.net_data.n_relays):
            bus_idx = min(i, self.net_data.n_buses - 1)
            p_mw = self.net_data.Iload_max[i] * self.net_data.base_voltage * 1.732 / 1e6  # 3-phase
            pp.create_load(
                net, 
                bus=bus_map[bus_idx], 
                p_mw=p_mw * 0.8,  # 0.8 pf
                q_mvar=p_mw * 0.6,
                name=f"Load_R{i}"
            )

        # Add DG/PV generators
        for dg in self.net_data.dg_buses:
            if dg.bus_idx < self.net_data.n_buses:
                pp.create_sgen(
                    net,
                    bus=bus_map[dg.bus_idx],
                    p_mw=dg.p_mw * dg.pv_penetration,
                    q_mvar=dg.q_mvar * dg.pv_penetration,
                    name=f"PV_Bus{dg.bus_idx}",
                    type="PV"
                )

        self.pp_net = net
        self.bus_map = bus_map

    def run_power_flow(self) -> bool:
        """Run AC power flow. Returns success status."""
        if not PANDAPOWER_AVAILABLE or self.pp_net is None:
            return False
        try:
            pp.runpp(self.pp_net, algorithm="nr", init="auto")
            return self.pp_net.converged
        except Exception as e:
            print(f"Power flow failed: {e}")
            return False

    def run_short_circuit(
        self, 
        fault_type: str = "3ph",
        topology_changes: Optional[Dict] = None
    ) -> Dict[str, float]:
        """
        Run short-circuit analysis and return fault currents at all buses.

        Parameters:
            fault_type: "3ph", "2ph", "1ph", "2ph-g"
            topology_changes: Dict with line outages, etc.

        Returns:
            Dict of fault currents (kA) at each bus
        """
        if not PANDAPOWER_AVAILABLE or self.pp_net is None:
            # Fallback to analytical calculation
            return self._analytical_fault_currents(fault_type, topology_changes)

        # Apply topology changes if specified
        if topology_changes:
            self._apply_topology_changes(topology_changes)

        try:
            # Map fault types to pandapower short-circuit types
            sc_type_map = {
                "3ph": "3ph",
                "LL": "2ph", 
                "1ph": "1ph",
                "2ph-g": "2ph"
            }
            sc_type = sc_type_map.get(fault_type, "3ph")

            sc.calc_sc(self.pp_net, fault=sc_type, case="max")

            # Extract fault currents (convert kA to A)
            fault_currents = {}
            for bus_idx in range(self.net_data.n_buses):
                if bus_idx in self.bus_map:
                    pp_bus = self.bus_map[bus_idx]
                    ik = self.pp_net.res_bus_sc.at[pp_bus, "ikss_ka"] * 1000  # kA to A
                    fault_currents[f"bus_{bus_idx}"] = float(ik)

            return fault_currents

        except Exception as e:
            print(f"Short-circuit calculation failed: {e}")
            return self._analytical_fault_currents(fault_type, topology_changes)

    def _analytical_fault_currents(
        self, 
        fault_type: str = "3ph",
        topology_changes: Optional[Dict] = None
    ) -> Dict[str, float]:
        """Analytical fallback when pandapower is unavailable."""
        fc = {}
        ft_scale = {"3ph": 1.0, "LL": 0.87, "1ph": 0.58, "2ph-g": 0.75}.get(fault_type, 1.0)

        # Use If_min as base fault current, scaled by fault type
        for i in range(self.net_data.n_buses):
            base_ik = self.net_data.If_min[min(i, len(self.net_data.If_min)-1)] if len(self.net_data.If_min) > 0 else 5000.0
            fc[f"bus_{i}"] = base_ik * ft_scale

        return fc

    def _apply_topology_changes(self, changes: Dict):
        """Apply N-1 contingencies, line outages, etc."""
        if not PANDAPOWER_AVAILABLE or self.pp_net is None:
            return

        # Line outages
        if "line_outages" in changes:
            for line_idx in changes["line_outages"]:
                if line_idx < len(self.pp_net.line):
                    self.pp_net.line.at[line_idx, "in_service"] = False

        # Bus splits (open switches)
        if "bus_splits" in changes:
            for bus_idx in changes["bus_splits"]:
                if bus_idx in self.bus_map:
                    # Disconnect all lines connected to this bus
                    for line_idx in range(len(self.pp_net.line)):
                        if (self.pp_net.line.at[line_idx, "from_bus"] == self.bus_map[bus_idx] or
                            self.pp_net.line.at[line_idx, "to_bus"] == self.bus_map[bus_idx]):
                            self.pp_net.line.at[line_idx, "in_service"] = False

    def update_pv_penetration(self, pv_level: float):
        """Update PV generation scaling for all DG buses."""
        if not PANDAPOWER_AVAILABLE or self.pp_net is None:
            return

        for dg in self.net_data.dg_buses:
            # Find sgen corresponding to this DG bus
            for sgen_idx in range(len(self.pp_net.sgen)):
                if f"PV_Bus{dg.bus_idx}" in str(self.pp_net.sgen.at[sgen_idx, "name"]):
                    self.pp_net.sgen.at[sgen_idx, "p_mw"] = dg.p_mw * pv_level
                    self.pp_net.sgen.at[sgen_idx, "q_mvar"] = dg.q_mvar * pv_level

    def get_bus_voltages(self) -> Dict[int, float]:
        """Return bus voltage magnitudes (pu) after power flow."""
        if not PANDAPOWER_AVAILABLE or self.pp_net is None:
            return {i: 1.0 for i in range(self.net_data.n_buses)}

        voltages = {}
        for bus_idx, pp_bus in self.bus_map.items():
            voltages[bus_idx] = float(self.pp_net.res_bus.at[pp_bus, "vm_pu"])
        return voltages

    def get_line_loadings(self) -> Dict[int, float]:
        """Return line loading percentages."""
        if not PANDAPOWER_AVAILABLE or self.pp_net is None:
            return {}

        loadings = {}
        for line_idx in range(len(self.pp_net.line)):
            if self.pp_net.line.at[line_idx, "in_service"]:
                loadings[line_idx] = float(self.pp_net.res_line.at[line_idx, "loading_percent"])
        return loadings


def build_pandapower_network(net_data: NetworkData) -> PandapowerNetwork:
    """Factory function to create PandapowerNetwork from NetworkData."""
    return PandapowerNetwork(net_data)
