"""
dynamic_network.py
──────────────────
Dynamic Network wrapper that combines:
- Pandapower network for power flow and short-circuit analysis
- PV inverter models with fault ride-through
- Topology manager for N-1 contingencies
- Scenario generator for training/validation data
- Sensor array for realistic measurements

Phase 1: Unified DistributionNetwork class for dynamic DOCR studies.
"""

import numpy as np
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass

from network.network_data import NetworkData, DGConfig, get_8bus, get_30bus, get_39bus
from network.relay_physics import (
    compute_fault_currents_dynamic, FAULT_TYPES, 
    compute_operating_times, objective_function, check_constraints
)
from network.pandapower_network import PandapowerNetwork, build_pandapower_network
from pv_model.pv_inverter import PVPlant, PVInverter
from pv_model.scenario_generator import ScenarioGenerator, DailyProfile
from environment.topology_manager import TopologyManager, TopologyState
from environment.relay_sensor import SensorArray, RelaySensor


@dataclass
class DynamicState:
    """Complete dynamic state of the distribution network."""
    # Relay settings
    TMS: np.ndarray
    PS: np.ndarray

    # Network conditions
    pv_penetration: float
    fault_type: str
    topology_state: TopologyState

    # Measurements
    bus_voltages: Dict[int, float]
    line_loadings: Dict[int, float]
    fault_currents: Dict[str, float]

    # Time
    time_of_day: float  # Hours (0-24)
    irradiance: float   # W/m²
    temperature: float  # °C
    load_pu: float      # Per-unit load


class DistributionNetwork:
    """
    Unified distribution network model for dynamic DOCR coordination.

    Integrates:
    1. Pandapower network (power flow, short-circuit)
    2. PV plants with IEC 61427-1 ride-through
    3. Topology manager (N-1, reconfiguration)
    4. Scenario generator (profiles, fault scenarios)
    5. Sensor array (CT/PT with noise)

    Usage:
        net = DistributionNetwork(get_8bus())
        state = net.reset(pv_level=0.5, fault_type="3ph")
        state = net.step(topology_changes={"line_outages": [2]})
    """

    def __init__(
        self,
        net_data: NetworkData,
        use_pandapower: bool = True,
        sensor_seed: int = 42,
    ):
        self.net_data = net_data
        self.name = net_data.name

        # Pandapower network (if available)
        self.use_pandapower = use_pandapower
        if use_pandapower:
            try:
                self.pp_net = build_pandapower_network(net_data)
                self.pp_available = self.pp_net.pp_net is not None
            except Exception:
                self.pp_available = False
                self.pp_net = None
        else:
            self.pp_available = False
            self.pp_net = None

        # PV plants (one per DG bus)
        self.pv_plants = {}
        for dg in net_data.dg_buses:
            self.pv_plants[dg.bus_idx] = PVPlant(
                bus_idx=dg.bus_idx,
                total_power_mw=dg.p_mw,
                rated_voltage_kv=net_data.base_voltage,
            )

        # Topology manager
        n_lines = max(1, net_data.n_buses - 1)
        self.topology = TopologyManager(
            n_buses=net_data.n_buses,
            n_lines=n_lines,
            n_switches=max(0, n_lines - 3),
        )

        # Scenario generator
        self.scenario_gen = ScenarioGenerator(seed=42)

        # Sensor array
        self.sensors = SensorArray(net_data.n_relays, seed=sensor_seed)

        # Current state
        self.current_state = None
        self.time = 0.0

        # Relay settings (initialized to defaults)
        self.TMS = np.full(net_data.n_relays, 0.5)
        self.PS = (net_data.ps_bounds()[0] + net_data.ps_bounds()[1]) / 2

    def reset(
        self,
        pv_level: float = 0.0,
        fault_type: str = "3ph",
        topology_changes: Optional[Dict] = None,
        time_of_day: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> DynamicState:
        """
        Reset network to initial state with specified conditions.

        Parameters:
            pv_level: PV penetration (0.0 to 1.0)
            fault_type: Fault type ("3ph", "LL", "1ph", "2ph-g")
            topology_changes: Dict with line_outages, bus_splits, etc.
            time_of_day: Hour of day (0-24), random if None
            seed: Random seed for reproducibility

        Returns:
            DynamicState with full network state
        """
        if seed is not None:
            np.random.seed(seed)

        # Reset topology
        self.topology.reset()
        if topology_changes:
            self.topology.apply_multiple_outages(
                line_outages=topology_changes.get("line_outages", []),
                switch_outages=topology_changes.get("switch_outages", []),
            )

        # Reset sensors
        self.sensors.reset_all()

        # Generate daily profile
        profile = self.scenario_gen.cloud_model.generate_daily_profile()

        if time_of_day is None:
            time_of_day = np.random.uniform(0, 24)

        # Interpolate profile values at time_of_day
        idx = int(time_of_day * 4) % len(profile.hours)  # 15-min resolution
        irradiance = profile.irradiance[idx]
        temperature = profile.temperature[idx]
        load_pu = profile.load_pu[idx]

        # Update PV plant outputs
        self._update_pv_output(pv_level, irradiance, temperature)

        # Update pandapower if available
        if self.pp_available:
            self.pp_net.update_pv_penetration(pv_level)
            self.pp_net.run_power_flow()
            bus_voltages = self.pp_net.get_bus_voltages()
            line_loadings = self.pp_net.get_line_loadings()
        else:
            bus_voltages = {i: 1.0 for i in range(self.net_data.n_buses)}
            line_loadings = {}

        # Compute fault currents
        fault_currents = self._compute_fault_currents(
            pv_level, fault_type, self.topology.current_state
        )

        # Initialize relay settings if not set
        if self.TMS is None:
            self.TMS = np.full(self.net_data.n_relays, 0.5)
        if self.PS is None:
            self.PS = (self.net_data.ps_bounds()[0] + self.net_data.ps_bounds()[1]) / 2

        self.current_state = DynamicState(
            TMS=self.TMS.copy(),
            PS=self.PS.copy(),
            pv_penetration=pv_level,
            fault_type=fault_type,
            topology_state=self.topology.current_state.copy(),
            bus_voltages=bus_voltages,
            line_loadings=line_loadings,
            fault_currents=fault_currents,
            time_of_day=time_of_day,
            irradiance=irradiance,
            temperature=temperature,
            load_pu=load_pu,
        )

        self.time = 0.0
        return self.current_state

    def step(
        self,
        topology_changes: Optional[Dict] = None,
        time_delta: float = 0.25,  # 15 minutes
    ) -> DynamicState:
        """
        Step network forward in time.

        Parameters:
            topology_changes: New topology changes to apply
            time_delta: Time step in hours

        Returns:
            Updated DynamicState
        """
        if self.current_state is None:
            raise RuntimeError("Network not initialized. Call reset() first.")

        # Update time
        self.time += time_delta
        self.current_state.time_of_day = (self.current_state.time_of_day + time_delta) % 24

        # Update topology if changes provided
        if topology_changes:
            self.topology.apply_multiple_outages(
                line_outages=topology_changes.get("line_outages", []),
                switch_outages=topology_changes.get("switch_outages", []),
            )
            self.current_state.topology_state = self.topology.current_state.copy()

        # Update PV output (simplified: assume irradiance changes slowly)
        self._update_pv_output(
            self.current_state.pv_penetration,
            self.current_state.irradiance,
            self.current_state.temperature,
        )

        # Recompute fault currents
        self.current_state.fault_currents = self._compute_fault_currents(
            self.current_state.pv_penetration,
            self.current_state.fault_type,
            self.current_state.topology_state,
        )

        # Update pandapower if available
        if self.pp_available:
            self.pp_net.run_power_flow()
            self.current_state.bus_voltages = self.pp_net.get_bus_voltages()
            self.current_state.line_loadings = self.pp_net.get_line_loadings()

        return self.current_state

    def set_relay_settings(self, TMS: np.ndarray, PS: np.ndarray):
        """Update relay settings (TMS and PS)."""
        self.TMS = TMS.copy()
        self.PS = PS.copy()
        if self.current_state is not None:
            self.current_state.TMS = TMS.copy()
            self.current_state.PS = PS.copy()

    def evaluate_relay_settings(
        self,
        TMS: Optional[np.ndarray] = None,
        PS: Optional[np.ndarray] = None,
    ) -> Dict:
        """
        Evaluate relay settings against current network state.

        Returns:
            Dict with OF, constraints, operating times, etc.
        """
        tms = TMS if TMS is not None else self.TMS
        ps = PS if PS is not None else self.PS

        topology_state = {
            "fault_currents": self.current_state.fault_currents if self.current_state else None
        } if self.current_state else None

        OF, sum_top, penalty = objective_function(
            self.net_data, tms, ps,
            fault_type=self.current_state.fault_type if self.current_state else "3ph",
            topology_state=topology_state,
        )

        constraints = check_constraints(
            self.net_data, tms, ps,
            fault_type=self.current_state.fault_type if self.current_state else "3ph",
            topology_state=topology_state,
        )

        return {
            "OF": OF,
            "sum_top": sum_top,
            "penalty": penalty,
            "constraints": constraints,
            "TMS": tms,
            "PS": ps,
        }

    def get_sensor_measurements(
        self,
        fault_condition: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get realistic sensor measurements for all relays.

        Returns:
            (measured_currents, measured_voltages)
        """
        if self.current_state is None:
            raise RuntimeError("Network not initialized. Call reset() first.")

        # Extract true values from current state
        true_currents = np.zeros(self.net_data.n_relays)
        true_voltages = np.zeros(self.net_data.n_relays)

        for i in range(self.net_data.n_relays):
            bus_idx = min(i, self.net_data.n_buses - 1)
            true_voltages[i] = self.current_state.bus_voltages.get(bus_idx, 1.0)

            # Approximate current from load and PV
            load_current = self.net_data.Iload_max[i] * self.current_state.load_pu
            pv_current = 0.0
            for dg in self.net_data.dg_buses:
                if dg.bus_idx == bus_idx:
                    pv_current = dg.p_mw * self.current_state.pv_penetration * 1e6 / (
                        np.sqrt(3) * self.net_data.base_voltage * 1e3
                    )
            true_currents[i] = load_current + pv_current

        return self.sensors.measure_all(true_currents, true_voltages, fault_condition)

    def _update_pv_output(self, pv_level: float, irradiance: float, temperature: float):
        """Update all PV plant outputs."""
        for bus_idx, plant in self.pv_plants.items():
            # Get bus voltage (approximate)
            v_pu = 1.0
            if self.current_state and bus_idx in self.current_state.bus_voltages:
                v_pu = self.current_state.bus_voltages[bus_idx]

            p, q, i = plant.get_total_output(
                irradiance=irradiance,
                temperature=temperature,
                v_terminal_pu=v_pu,
            )
            # Note: In full implementation, this would update pandapower sgen values

    def _compute_fault_currents(
        self,
        pv_level: float,
        fault_type: str,
        topology_state: TopologyState,
    ) -> Dict[str, float]:
        """Compute fault currents considering PV and topology."""
        # Get outage factor from topology
        outage_factor = self.topology.get_outage_factor(topology_state)

        # Compute dynamic fault currents
        fc = compute_fault_currents_dynamic(
            self.net_data,
            pv_penetration=pv_level,
            fault_type=fault_type,
            topology_changes={"outage_factor": outage_factor},
        )

        # Add PV contribution
        for bus_idx, plant in self.pv_plants.items():
            v_pu = 1.0  # Pre-fault voltage
            if self.current_state and bus_idx in self.current_state.bus_voltages:
                v_pu = self.current_state.bus_voltages[bus_idx]

            pv_i_contribution = plant.get_fault_current_contribution(v_pu)

            # Add to fault currents for pairs near this bus
            for i, pair in enumerate(self.net_data.pairs):
                if pair.bus_pri == bus_idx or pair.bus_bkp == bus_idx:
                    fc[f"pair_{i}_pri"] += pv_i_contribution * 1000  # kA to A
                    fc[f"pair_{i}_bkp"] += pv_i_contribution * 1000

        return fc


    def evaluate_current_settings(
        self,
        TMS: Optional[np.ndarray] = None,
        PS: Optional[np.ndarray] = None,
    ) -> Dict:
        """
        Evaluate relay settings against current network state.

        Returns:
            Dict with OF, constraints, operating times, etc.
        """
        tms = TMS if TMS is not None else self.TMS
        ps = PS if PS is not None else self.PS

        topology_state = {"fault_currents": self.current_state.fault_currents} if self.current_state else None

        OF, sum_top, penalty = objective_function(
            self.net_data, tms, ps,
            fault_type=self.current_state.fault_type if self.current_state else "3ph",
            topology_state=topology_state,
        )

        constraints = check_constraints(
            self.net_data, tms, ps,
            fault_type=self.current_state.fault_type if self.current_state else "3ph",
            topology_state=topology_state,
        )

        return {
            "OF": OF,
            "sum_top": sum_top,
            "penalty": penalty,
            "constraints": constraints,
            "TMS": tms,
            "PS": ps,
        }

    def get_scenario_batch(
        self,
        n_scenarios: int = 10,
        split: str = "train",
    ) -> List[DynamicState]:
        """Generate a batch of scenarios for training/validation."""
        scenarios = []
        for _ in range(n_scenarios):
            scenario = self.scenario_gen._generate_single_scenario(split)
            state = self.reset(
                pv_level=scenario["pv_level"],
                fault_type=scenario["fault_type"],
                topology_changes=scenario["topology"],
            )
            scenarios.append(state)
        return scenarios

    def get_curriculum_batch(
        self,
        stage: int = 0,
        n_scenarios: int = 100,
    ) -> List[DynamicState]:
        """Generate curriculum learning scenarios for a specific stage."""
        pv_levels = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        pv_level = pv_levels[min(stage, len(pv_levels) - 1)]

        scenarios = []
        for _ in range(n_scenarios):
            topology = {"line_outages": [], "switch_outages": []}
            if stage >= 4 and np.random.random() < 0.3:
                topology["line_outages"] = [int(np.random.randint(0, max(1, self.net_data.n_buses - 1)))]

            state = self.reset(
                pv_level=pv_level,
                fault_type="3ph",
                topology_changes=topology,
            )
            scenarios.append(state)
        return scenarios


def create_network(
    system_name: str = "8bus",
    use_pandapower: bool = True,
) -> DistributionNetwork:
    """Factory function to create DistributionNetwork by name."""
    from network.network_data import NETWORKS

    if system_name not in NETWORKS:
        raise ValueError(f"Unknown system: {system_name}. Available: {list(NETWORKS.keys())}")

    net_data = NETWORKS[system_name]()
    return DistributionNetwork(net_data, use_pandapower=use_pandapower)
