"""
relay_sensor.py
───────────────
Current Transformer (CT) and Potential Transformer (PT) simulation
with realistic noise models, saturation effects, and measurement errors.

Refactored for full IEEE test system support: 3, 8, 14, 30, 39, 57, 118 buses.
"""

import numpy as np
from typing import Tuple, Dict, Optional, List
from dataclasses import dataclass


@dataclass
class CTParameters:
    """Current Transformer parameters."""
    ratio: float = 1000.0      # CT ratio (e.g., 1000:5)
    accuracy_class: str = "5P20"  # IEC 60044-1 accuracy class
    burden_va: float = 15.0    # Rated burden (VA)
    knee_point_v: float = 100.0  # Knee point voltage (V)
    saturation_factor: float = 20.0  # Saturation limit (x rated current)

    # Error characteristics
    ratio_error: float = 0.01   # 1% ratio error
    phase_error: float = 0.5    # 0.5 degree phase error

    # Noise parameters
    noise_std: float = 0.005   # 0.5% standard deviation
    harmonic_distortion: float = 0.03  # 3% THD


@dataclass
class PTParameters:
    """Potential Transformer parameters."""
    ratio: float = 11000.0     # PT ratio (e.g., 11kV:110V)
    accuracy_class: str = "0.5"  # IEC 60044-2 accuracy class
    burden_va: float = 50.0    # Rated burden (VA)

    # Error characteristics
    ratio_error: float = 0.005  # 0.5% ratio error
    phase_error: float = 0.3    # 0.3 degree phase error

    # Noise parameters
    noise_std: float = 0.003   # 0.3% standard deviation
    ferroresonance_risk: bool = False


class RelaySensor:
    """
    Simulates CT and PT measurements for relay protection.

    Models:
    - Ideal measurement (no errors)
    - Realistic measurement (ratio/phase errors, noise, saturation)
    - Fault conditions (DC offset, CT saturation during faults)
    - Communication delays and packet loss

    Parameters:
        ct_params: CT parameters
        pt_params: PT parameters
        sampling_rate: Samples per cycle (e.g., 16, 32, 64)
        seed: Random seed for reproducibility
    """

    def __init__(
        self,
        ct_params: Optional[CTParameters] = None,
        pt_params: Optional[PTParameters] = None,
        sampling_rate: int = 32,
        seed: int = 42,
    ):
        self.ct = ct_params or CTParameters()
        self.pt = pt_params or PTParameters()
        self.sampling_rate = sampling_rate
        self.rng = np.random.default_rng(seed)

        # Time step (50Hz system, 20ms per cycle)
        self.dt = 1.0 / (50.0 * sampling_rate)

        # CT saturation state
        self.ct_saturated = False
        self.saturation_recovery_time = 0.0

        # Measurement history for filtering
        self.current_history = []
        self.voltage_history = []
        self.history_maxlen = 10

    def measure_current(
        self,
        true_current_a: float,  # True primary current (A)
        fault_condition: bool = False,
        dc_offset: bool = False,
    ) -> float:
        """
        Measure current through CT with realistic errors.

        Parameters:
            true_current_a: Actual primary current
            fault_condition: True if during fault (higher current, possible saturation)
            dc_offset: True if DC offset present (asymmetric fault current)

        Returns:
            Measured secondary current (A) scaled to primary equivalent
        """
        # Check for CT saturation
        rated_primary = self.ct.ratio * 5.0  # Assuming 5A secondary
        if fault_condition and abs(true_current_a) > rated_primary * self.ct.saturation_factor:
            self.ct_saturated = True
            self.saturation_recovery_time = 0.1  # 100ms recovery

        if self.ct_saturated:
            # Saturated CT: clipped output with decay
            self.saturation_recovery_time -= self.dt * self.sampling_rate
            if self.saturation_recovery_time <= 0:
                self.ct_saturated = False

            # Saturated output: limited to knee point
            saturated_value = np.sign(true_current_a) * rated_primary * self.ct.saturation_factor
            measured = saturated_value * (0.5 + 0.5 * np.exp(-self.saturation_recovery_time / 0.05))
        else:
            measured = true_current_a

        # Add ratio error
        measured *= (1 + self.rng.normal(0, self.ct.ratio_error))

        # Add phase error (converted to amplitude error for simplicity)
        phase_rad = np.radians(self.rng.normal(0, self.ct.phase_error))
        measured *= np.cos(phase_rad)

        # Add DC offset if present
        if dc_offset and fault_condition:
            dc_component = true_current_a * 0.6 * np.exp(-self.dt * 10)  # Decaying DC
            measured += dc_component

        # Add measurement noise
        noise = self.rng.normal(0, abs(measured) * self.ct.noise_std + 0.1)
        measured += noise

        # Add harmonic distortion
        if self.ct.harmonic_distortion > 0:
            for harmonic in [3, 5, 7]:
                harmonic_amp = abs(measured) * self.ct.harmonic_distortion / harmonic
                measured += harmonic_amp * np.sin(2 * np.pi * harmonic * 50 * self.dt)

        # Update history
        self.current_history.append(measured)
        if len(self.current_history) > self.history_maxlen:
            self.current_history.pop(0)

        return measured

    def measure_voltage(
        self,
        true_voltage_kv: float,  # True primary voltage (kV)
        fault_condition: bool = False,
    ) -> float:
        """
        Measure voltage through PT with realistic errors.

        Parameters:
            true_voltage_kv: Actual primary voltage (kV)
            fault_condition: True if during fault (voltage dip)

        Returns:
            Measured secondary voltage (kV) scaled to primary equivalent
        """
        measured = true_voltage_kv

        # Add ratio error
        measured *= (1 + self.rng.normal(0, self.pt.ratio_error))

        # Add phase error
        phase_rad = np.radians(self.rng.normal(0, self.pt.phase_error))
        measured *= np.cos(phase_rad)

        # Add measurement noise
        noise = self.rng.normal(0, abs(measured) * self.pt.noise_std + 0.05)
        measured += noise

        # Voltage dip during fault (more severe measurement error)
        if fault_condition and true_voltage_kv < 0.7:  # Below 70% voltage
            # Increased error during voltage dip
            dip_error = self.rng.normal(0, 0.02 * true_voltage_kv)
            measured += dip_error

        # Update history
        self.voltage_history.append(measured)
        if len(self.voltage_history) > self.history_maxlen:
            self.voltage_history.pop(0)

        return measured

    def measure_fault_current(
        self,
        true_fault_current_a: float,
        fault_type: str = "3ph",
    ) -> Dict[str, float]:
        """
        Measure fault current with all realistic effects.

        Returns dict with multiple measurements:
        - 'rms': RMS current
        - 'peak': Peak current
        - 'fundamental': Fundamental frequency component
        - 'dc_component': DC offset magnitude
        """
        # Generate one cycle of samples
        t = np.arange(self.sampling_rate) * self.dt

        # Fundamental frequency component
        fundamental = true_fault_current_a * np.sin(2 * np.pi * 50 * t)

        # DC offset (for asymmetric faults)
        dc_component = 0.0
        if fault_type in ["1ph", "2ph-g"]:
            dc_component = true_fault_current_a * 0.5 * np.exp(-t * 10)

        # Add noise
        noise = self.rng.normal(0, abs(true_fault_current_a) * self.ct.noise_std, self.sampling_rate)

        samples = fundamental + dc_component + noise

        # Apply CT saturation if current too high
        rated_primary = self.ct.ratio * 5.0
        if abs(true_fault_current_a) > rated_primary * self.ct.saturation_factor:
            # Clip samples
            max_val = rated_primary * self.ct.saturation_factor
            samples = np.clip(samples, -max_val, max_val)

        # Measurements
        rms = np.sqrt(np.mean(samples**2))
        peak = np.max(np.abs(samples))

        # Extract fundamental using DFT (simplified)
        fundamental_component = (2 / self.sampling_rate) * np.sum(
            samples * np.sin(2 * np.pi * 50 * t)
        )

        return {
            "rms": rms,
            "peak": peak,
            "fundamental": fundamental_component,
            "dc_component": np.mean(dc_component) if np.any(dc_component) else 0.0,
        }

    def apply_median_filter(self, window_size: int = 3) -> Tuple[float, float]:
        """
        Apply median filtering to reduce noise/outliers.

        Returns:
            (filtered_current, filtered_voltage)
        """
        if len(self.current_history) < window_size or len(self.voltage_history) < window_size:
            return (
                self.current_history[-1] if self.current_history else 0.0,
                self.voltage_history[-1] if self.voltage_history else 0.0,
            )

        filt_current = np.median(self.current_history[-window_size:])
        filt_voltage = np.median(self.voltage_history[-window_size:])

        return filt_current, filt_voltage

    def get_measurement_quality(self) -> Dict[str, float]:
        """
        Assess current measurement quality.

        Returns:
            Dict with SNR, THD estimate, saturation flag
        """
        if len(self.current_history) < 5:
            return {"snr_db": 0.0, "thd": 0.0, "saturated": self.ct_saturated}

        recent = np.array(self.current_history[-5:])
        signal = np.mean(np.abs(recent))
        noise = np.std(np.diff(recent))  # Rough noise estimate

        snr_db = 20 * np.log10(signal / (noise + 1e-6)) if noise > 0 else 60.0
        thd = self.ct.harmonic_distortion * 100  # Percentage

        return {
            "snr_db": snr_db,
            "thd": thd,
            "saturated": self.ct_saturated,
        }

    def reset(self):
        """Reset sensor state."""
        self.ct_saturated = False
        self.saturation_recovery_time = 0.0
        self.current_history = []
        self.voltage_history = []


class SensorArray:
    """Array of sensors for all relays in the network."""

    def __init__(self, n_relays: int, seed: int = 42):
        self.n_relays = n_relays
        self.sensors = [
            RelaySensor(seed=seed + i)
            for i in range(n_relays)
        ]

    def measure_all(
        self,
        true_currents: np.ndarray,
        true_voltages: np.ndarray,
        fault_condition: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Measure all relay currents and voltages."""
        measured_i = np.zeros(self.n_relays)
        measured_v = np.zeros(self.n_relays)

        for i in range(self.n_relays):
            measured_i[i] = self.sensors[i].measure_current(
                true_currents[i], fault_condition
            )
            measured_v[i] = self.sensors[i].measure_voltage(
                true_voltages[i], fault_condition
            )

        return measured_i, measured_v

    def measure_fault_currents(
        self,
        true_fault_currents: np.ndarray,
        fault_type: str = "3ph",
    ) -> List[Dict[str, float]]:
        """Measure fault currents for all relays."""
        measurements = []
        for i in range(self.n_relays):
            meas = self.sensors[i].measure_fault_current(
                true_fault_currents[i], fault_type
            )
            measurements.append(meas)
        return measurements

    def reset_all(self):
        """Reset all sensors."""
        for sensor in self.sensors:
            sensor.reset()