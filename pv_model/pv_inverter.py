"""
pv_inverter.py
────────────
PV Inverter Fault Ride-Through (FRT) model.

This module uses a simplified, linear approximation of PV inverter fault
ride-through and fault-current-injection behavior. It is loosely informed by
the general shape of grid-code fault ride-through requirements (see IEEE 1547
and IEC 61727 for the standards that actually govern grid-connected PV fault
behavior) but is not a literal, certified implementation of any single
standard's requirements.

Models:
- Normal operation: Maximum Power Point Tracking (MPPT)
- Fault ride-through: Current injection with reactive power support
- Post-fault recovery: Gradual return to normal operation

Phase 1: PV Inverter Model for Dynamic Fault Current Calculation.
"""

import numpy as np
from typing import Tuple, Dict, Optional
from dataclasses import dataclass

from network.relay_physics import FAULT_TYPE_WEIGHTS


@dataclass
class PVRideThroughCurve:
    """Fault ride-through voltage-time curve parameters."""
    v_low:  float = 0.0    # pu voltage where inverter trips
    v_high: float = 1.1   # pu voltage where inverter trips (overvoltage)
    t_ride: float = 0.15  # seconds of required ride-through
    t_rec:  float = 0.5   # seconds to recover after fault clearance
    k_i:    float = 1.2   # current injection multiplier during fault
    k_q:    float = 0.5   # reactive current injection factor


class PVInverter:
    """
    PV Inverter model with fault ride-through capability.

    Implements a simplified, linear approximation of fault ride-through and
    fault-current injection behavior, loosely informed by the general shape
    of grid-code requirements. IEEE 1547-2018 (Standard for Interconnection
    and Interoperability) and IEC 61727 are the standards that actually
    govern grid-connected PV fault behavior, referenced here for context —
    this model does not claim to reproduce their formulas precisely.

    Parameters:
        rated_power_mw: Rated active power (MW)
        rated_voltage_kv: Rated voltage (kV)
        frt_curve: Ride-through curve parameters
        mppt_efficiency: MPPT efficiency (0.95-0.99)
    """

    def __init__(
        self,
        rated_power_mw: float = 1.0,
        rated_voltage_kv: float = 0.4,
        frt_curve: Optional[PVRideThroughCurve] = None,
        mppt_efficiency: float = 0.98,
    ):
        self.rated_power_mw = rated_power_mw
        self.rated_voltage_kv = rated_voltage_kv
        self.frt = frt_curve or PVRideThroughCurve()
        self.mppt_efficiency = mppt_efficiency

        # Base values
        self.base_mva = rated_power_mw
        self.base_current_ka = rated_power_mw / (np.sqrt(3) * rated_voltage_kv)

        # State
        self.fault_active = False
        self.fault_start_time = 0.0
        self.post_fault_recovery = False
        self.recovery_start_time = 0.0
        # Latched trip state: once the ride-through window is exceeded and
        # the inverter trips, it must STAY tripped (output pinned to zero)
        # for as long as the underlying fault condition persists — it must
        # not re-arm on the very next call just because fault_triggered is
        # still True. It only clears via the post_fault_recovery path once
        # fault_triggered genuinely goes False (see calculate_output()).
        self.tripped = False

    def calculate_output(
        self,
        irradiance: float,       # W/m² (0-1200)
        temperature: float,      # °C
        v_terminal_pu: float,    # Terminal voltage (pu)
        time: float = 0.0,       # Current time (s)
        fault_triggered: bool = False,
    ) -> Tuple[float, float, float]:
        """
        Calculate PV inverter output (P, Q, I) given conditions.

        Returns:
            (p_mw, q_mvar, i_ka) — active power, reactive power, current
        """
        # Normal power output from irradiance and temperature
        p_available = self._mppt_power(irradiance, temperature)

        # If already tripped, stay tripped (output pinned to zero) for as
        # long as the underlying fault condition persists. Do NOT re-arm
        # fault_active / restart the ride-through clock just because
        # fault_triggered is still True on this call — that was the bug:
        # it caused the inverter to re-inject fault current every call
        # instead of remaining latched off. The tripped state only clears
        # once the fault condition genuinely ends (fault_triggered=False),
        # at which point normal post-fault recovery begins.
        if self.tripped:
            if fault_triggered:
                return 0.0, 0.0, 0.0
            self.tripped = False
            self.post_fault_recovery = True
            self.recovery_start_time = time

        # Check fault condition
        if fault_triggered and not self.fault_active:
            self.fault_active = True
            self.fault_start_time = time

        if self.fault_active:
            # Check if ride-through time exceeded
            if time - self.fault_start_time > self.frt.t_ride:
                if v_terminal_pu < self.frt.v_low or v_terminal_pu > self.frt.v_high:
                    # Trip — latch tripped state and zero output. Stays
                    # latched (see the `self.tripped` check above) until
                    # fault_triggered genuinely goes False.
                    self.fault_active = False
                    self.tripped = True
                    return 0.0, 0.0, 0.0

            # During fault: inject current with reactive support
            i_fault = self.base_current_ka * self.frt.k_i
            # Reactive current proportional to voltage dip
            q_factor = max(0, 1.0 - v_terminal_pu) * self.frt.k_q
            p_fault = p_available * 0.1  # Reduce active power during fault
            q_fault = self.rated_power_mw * q_factor

            return p_fault, q_fault, i_fault

        if self.post_fault_recovery:
            # Gradual recovery
            recovery_progress = min(1.0, (time - self.recovery_start_time) / self.frt.t_rec)
            if recovery_progress >= 1.0:
                self.post_fault_recovery = False

            p_out = p_available * recovery_progress
            q_out = 0.0
            i_out = self.base_current_ka * recovery_progress
            return p_out, q_out, i_out

        # Normal operation
        if v_terminal_pu < 0.5:  # Too low voltage
            return 0.0, 0.0, 0.0

        p_out = p_available * min(1.0, v_terminal_pu / 0.9)  # Voltage derating
        q_out = 0.0  # Unity power factor in normal operation (can be modified for volt-var)
        i_out = p_out / (np.sqrt(3) * self.rated_voltage_kv * v_terminal_pu) if v_terminal_pu > 0 else 0

        return p_out, q_out, i_out

    def _mppt_power(self, irradiance: float, temperature: float) -> float:
        """Calculate maximum power from irradiance and temperature."""
        # Simplified single-diode model
        g_ref = 1000.0  # W/m² reference
        t_ref = 25.0    # °C reference
        beta_p = -0.0045  # Temperature coefficient (%/°C)

        # Irradiance scaling
        p_irr = self.rated_power_mw * (irradiance / g_ref)

        # Temperature derating
        p_temp = p_irr * (1 + beta_p * (temperature - t_ref))

        return max(0.0, p_temp * self.mppt_efficiency)

    def get_fault_current_contribution(
        self,
        v_terminal_pu: float,
        fault_type: str = "3ph",
    ) -> float:
        """
        Calculate fault current contribution during grid fault.

        Simplified linear approximation: inverters are modeled as contributing
        roughly 1.2-1.5x rated current during a fault (via the k_i multiplier),
        loosely informed by the general shape of grid-code low-voltage
        ride-through requirements. IEEE 1547-2018 mandates reactive current
        injection for voltage support; this model applies that qualitatively
        without claiming exact standards compliance.

        `fault_type` scales the contribution by FAULT_TYPE_WEIGHTS (the same
        3ph/LL/1ph/2ph-g severity weighting relay_physics.py's own fault
        current calculation uses) -- previously accepted but never applied,
        so this returned the identical contribution regardless of fault
        type. Unbalanced faults (1ph, 2ph-g) engage less of the inverter's
        symmetrical-component current capacity than a balanced 3ph fault.

        Returns:
            Fault current contribution in kA
        """
        if v_terminal_pu > 0.9:
            return 0.0  # No fault

        # Current injection during fault
        i_contribution = self.base_current_ka * self.frt.k_i

        # Scale by voltage (deeper dip = more current)
        v_factor = max(0.0, 1.0 - v_terminal_pu) / 0.5  # 0 to 1
        i_contribution *= (0.5 + 0.5 * v_factor)

        i_contribution *= FAULT_TYPE_WEIGHTS.get(fault_type, 1.0)

        return i_contribution

    def reset(self):
        """Reset inverter state."""
        self.fault_active = False
        self.fault_start_time = 0.0
        self.post_fault_recovery = False
        self.recovery_start_time = 0.0
        self.tripped = False


class PVPlant:
    """Aggregation of multiple PV inverters at a bus."""

    def __init__(
        self,
        bus_idx: int,
        total_power_mw: float,
        num_inverters: int = 10,
        rated_voltage_kv: float = 0.4,
    ):
        self.bus_idx = bus_idx
        self.total_power_mw = total_power_mw
        self.num_inverters = num_inverters
        self.rated_voltage_kv = rated_voltage_kv

        # Create individual inverters
        inv_power = total_power_mw / num_inverters
        self.inverters = [
            PVInverter(
                rated_power_mw=inv_power,
                rated_voltage_kv=rated_voltage_kv,
            )
            for _ in range(num_inverters)
        ]

    def get_total_output(
        self,
        irradiance: float,
        temperature: float,
        v_terminal_pu: float,
        time: float = 0.0,
        fault_triggered: bool = False,
    ) -> Tuple[float, float, float]:
        """Aggregate output from all inverters."""
        p_total = 0.0
        q_total = 0.0
        i_total = 0.0

        for inv in self.inverters:
            p, q, i = inv.calculate_output(
                irradiance, temperature, v_terminal_pu, time, fault_triggered
            )
            p_total += p
            q_total += q
            i_total += i

        return p_total, q_total, i_total

    def get_fault_current_contribution(self, v_terminal_pu: float, fault_type: str = "3ph") -> float:
        """Total fault current from all inverters."""
        return sum(
            inv.get_fault_current_contribution(v_terminal_pu, fault_type)
            for inv in self.inverters
        )
