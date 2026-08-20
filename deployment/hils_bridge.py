"""
deployment/hils_bridge.py
──────────────────────────
Phase 7: IEC 61850 GOOSE/SV interface and HILS bridge.

Provides:
1. IEC 61850 GOOSE publisher/subscriber for relay settings (SIMULATED)
2. Sampled Values (SV) interface for current/voltage measurements (SIMULATED)
3. HILS (Hardware-in-the-Loop Simulation) bridge
4. Real-time synchronization
5. MMS (Manufacturing Message Specification) client (SIMULATED)

Usage:
    from deployment.hils_bridge import IEC61850Bridge, HILSConfig

    bridge = IEC61850Bridge()
    bridge.connect()

    # Publish relay settings via GOOSE (simulated)
    bridge.publish_settings(relay_id=0, TMS=0.5, PS=1.2)

    # Subscribe to measurements (simulated)
    measurements = bridge.subscribe_measurements()

Warning
-------
This module provides a **simulation/placeholder** implementation.
No actual IEC 61850 communication occurs. Production deployment
requires ``libiec61850`` or an equivalent C library with Python
bindings, which must be integrated into this bridge.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Tuple, Optional

import numpy as np

__all__ = [
    "IEC61850DataType",
    "HILSConfig",
    "RelaySetting",
    "MeasurementSample",
    "IEC61850Bridge",
    "HILSBridge",
    "RealTimeSynchronizer",
]


# ── Enumerations ────────────────────────────────────────────────────────────

class IEC61850DataType(Enum):
    """IEC 61850 data types (reference only — not used in simulation)."""

    FLOAT32 = "FLOAT32"
    INT32 = "INT32"
    BOOLEAN = "BOOLEAN"
    TIMESTAMP = "Timestamp"


# ── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class HILSConfig:
    """Hardware-in-the-Loop Simulation configuration."""

    host: str = "localhost"
    port: int = 102  # IEC 61850 default MMS port
    goose_interface: str = "eth0"
    sv_interface: str = "eth1"
    simulation_time_step_ms: float = 4.0  # 250 Hz
    real_time_factor: float = 1.0  # 1.0 = real-time
    timeout_ms: float = 100.0
    max_latency_ms: float = 10.0  # GOOSE max latency


@dataclass
class RelaySetting:
    """IEC 61850 relay setting data object (simulated)."""

    relay_id: int
    tms: float
    ps: float
    timestamp: float
    quality: str = "good"  # good | questionable | invalid


@dataclass
class MeasurementSample:
    """IEC 61850 sampled value (simulated)."""

    relay_id: int
    ia: float  # Phase A current (A)
    ib: float  # Phase B current (A)
    ic: float  # Phase C current (A)
    va: float  # Phase A voltage (V)
    vb: float  # Phase B voltage (V)
    vc: float  # Phase C voltage (V)
    timestamp: float
    sample_count: int


# ── IEC 61850 Bridge (SIMULATED) ─────────────────────────────────────────────

class IEC61850Bridge:
    """IEC 61850 communication bridge for DOCR-MARL deployment (SIMULATED).

    This is a **placeholder implementation**. It simulates:
    - GOOSE (Generic Object Oriented Substation Events) for fast relay settings.
    - SV (Sampled Values) for current/voltage measurements.
    - MMS (Manufacturing Message Specification) for configuration.

    No actual IEC 61850 packets are transmitted. To make this real:
    1. Install ``libiec61850`` (C library by MZ Automation)
    2. Build Python bindings (e.g., via ``ctypes`` or ``cffi``)
    3. Replace the ``connect()``, ``publish_settings()``, and
       ``subscribe_measurements()`` methods with libiec61850 calls.
    """

    def __init__(self, config: HILSConfig | None = None, seed: int = 42) -> None:
        self.config: HILSConfig = config or HILSConfig()
        self.connected: bool = False
        self._settings_log: list[RelaySetting] = []
        self._measurement_log: list[MeasurementSample] = []
        self._latency_history: list[float] = []
        # Seeded once at construction (rather than reseeding from wall-clock
        # time on every subscribe_measurements() call) so simulated SV
        # measurements are reproducible, matching the project convention of
        # explicit np.random.default_rng(seed) everywhere.
        self.rng: np.random.Generator = np.random.default_rng(seed)

    # ── Connection management ───────────────────────────────────────────────

    def connect(self) -> bool:
        """Simulate connection to IEC 61850 server.

        Returns
        -------
        bool
            Always True in simulation. Real implementation would return
            connection status from libiec61850.
        """
        self.connected = True
        print(f"  [IEC61850] Connected to {self.config.host}:{self.config.port}")
        return True

    def disconnect(self) -> None:
        """Simulate disconnection from IEC 61850 server."""
        self.connected = False
        print("  [IEC61850] Disconnected")

    # ── GOOSE: Settings publication ───────────────────────────────────────────

    def publish_settings(self, relay_id: int, tms: float, ps: float) -> bool:
        """Simulate publishing relay settings via GOOSE.

        GOOSE characteristics (reference):
        - Max latency: 10 ms (T0 = 2 s, T1-T4 = 2-8 ms)
        - Retransmission: immediate, then exponential backoff
        - No acknowledgment (publisher-subscriber model)

        Parameters
        ----------
        relay_id : int
            Target relay identifier.
        tms : float
            Time multiplier setting.
        ps : float
            Plug setting (pickup current).

        Returns
        -------
        bool
            True if the setting was queued for publication.
            Returns False if not connected.
        """
        if not self.connected:
            return False

        t0 = time.perf_counter()

        setting = RelaySetting(
            relay_id=relay_id,
            tms=tms,
            ps=ps,
            timestamp=time.time(),
        )
        self._settings_log.append(setting)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self._latency_history.append(elapsed_ms)

        return True

    # ── SV: Measurement subscription ──────────────────────────────────────────

    def subscribe_measurements(self, n_samples: int = 1) -> list[MeasurementSample]:
        """Simulate subscribing to SV (Sampled Values) measurements.

        SV characteristics (reference):
        - 80 samples/cycle (4 kHz at 50 Hz) or 256 samples/cycle
        - Fixed payload size for deterministic timing
        - No TCP/IP overhead (Ethernet multicast)

        Parameters
        ----------
        n_samples : int
            Number of samples to generate.

        Returns
        -------
        list[MeasurementSample]
            Simulated measurement samples.
            Returns empty list if not connected.
        """
        if not self.connected:
            return []

        samples: list[MeasurementSample] = []
        for _ in range(n_samples):
            sample = MeasurementSample(
                relay_id=0,
                ia=float(self.rng.normal(1000.0, 50.0)),
                ib=float(self.rng.normal(1000.0, 50.0)),
                ic=float(self.rng.normal(1000.0, 50.0)),
                va=float(self.rng.normal(11000.0, 100.0)),
                vb=float(self.rng.normal(11000.0, 100.0)),
                vc=float(self.rng.normal(11000.0, 100.0)),
                timestamp=time.time(),
                sample_count=len(self._measurement_log),
            )
            self._measurement_log.append(sample)
            samples.append(sample)

        return samples

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def get_latency_statistics(self) -> dict[str, float | bool | int]:
        """Return GOOSE publication latency statistics."""
        if not self._latency_history:
            return {}

        arr = np.array(self._latency_history)
        return {
            "mean_ms": float(np.mean(arr)),
            "std_ms": float(np.std(arr)),
            "max_ms": float(np.max(arr)),
            "p99_ms": float(np.percentile(arr, 99)),
            "meets_target": bool(np.max(arr) < self.config.max_latency_ms),
            "target_ms": self.config.max_latency_ms,
            "n_samples": len(arr),
        }

    def verify_goose_timing(self) -> dict[str, dict | bool]:
        """Verify GOOSE timing meets IEC 61850 requirements (simulated check)."""
        stats = self.get_latency_statistics()

        iec_requirements = {
            "T0_max_ms": 2000,   # Steady-state retransmission
            "T1_max_ms": 2,      # First retransmission after event
            "T2_max_ms": 4,      # Second retransmission
            "T3_max_ms": 8,      # Third retransmission
            "TAL_max_ms": 10,    # Time Allowed to Live
        }

        max_ms = stats.get("max_ms", float("inf"))
        return {
            "requirements": iec_requirements,
            "actual": stats,
            "compliant": max_ms < iec_requirements["TAL_max_ms"],
        }


# ── HILS Bridge ─────────────────────────────────────────────────────────────

class HILSBridge:
    """Hardware-in-the-Loop Simulation bridge.

    Connects the DOCR-MARL simulation to real protection relays
    via IEC 61850 for closed-loop testing.

    Architecture::

        ┌─────────────┐    IEC 61850    ┌─────────────┐
        │  Simulation │  <──────────>   │  Real Relay │
        │  (Python)   │    GOOSE/SV     │  (Hardware) │
        └─────────────┘                 └─────────────┘

    Note
    ----
    The IEC 61850 layer is currently simulated. See
    :class:`IEC61850Bridge` for integration notes.
    """

    def __init__(self, config: HILSConfig | None = None, seed: int = 42) -> None:
        self.config: HILSConfig = config or HILSConfig()
        self.iec_bridge: IEC61850Bridge = IEC61850Bridge(config, seed=seed)
        self.running: bool = False
        self.cycle_count: int = 0
        self.error_count: int = 0

    def start(self) -> bool:
        """Start HILS loop."""
        if not self.iec_bridge.connect():
            return False

        self.running = True
        print(
            f"  [HILS] Started at {self.config.simulation_time_step_ms} ms timestep"
        )
        return True

    def stop(self) -> None:
        """Stop HILS loop."""
        self.running = False
        self.iec_bridge.disconnect()
        print("  [HILS] Stopped")

    def run_step(
        self,
        relay_settings: dict[int, tuple[float, float]],
        measurements: dict | None = None,
    ) -> dict[str, object]:
        """Run one HILS step.

        Parameters
        ----------
        relay_settings : dict[int, tuple[float, float]]
            Mapping {relay_id: (TMS, PS)}.
        measurements : dict | None
            Optional real measurements from relays (bypasses simulation).

        Returns
        -------
        dict
            Step result with timing, errors, and measurements.
        """
        if not self.running:
            return {"error": "HILS not running"}

        t0 = time.perf_counter()

        # Publish settings to real relays (simulated)
        publish_ok = True
        for relay_id, (tms, ps) in relay_settings.items():
            if not self.iec_bridge.publish_settings(relay_id, tms, ps):
                publish_ok = False
                self.error_count += 1

        # Get measurements (simulated or real)
        if measurements is None:
            measurements = self.iec_bridge.subscribe_measurements(n_samples=1)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self.cycle_count += 1

        return {
            "cycle": self.cycle_count,
            "elapsed_ms": elapsed_ms,
            "publish_ok": publish_ok,
            "measurements": measurements,
            "error_count": self.error_count,
        }

    def get_statistics(self) -> dict[str, object]:
        """Return HILS session statistics."""
        goose_stats = self.iec_bridge.get_latency_statistics()

        return {
            "cycles": self.cycle_count,
            "errors": self.error_count,
            "error_rate": self.error_count / max(self.cycle_count, 1),
            "goose_latency": goose_stats,
            "real_time_factor": self.config.real_time_factor,
            "time_step_ms": self.config.simulation_time_step_ms,
        }

    def generate_report(self) -> str:
        """Generate HILS test report."""
        stats = self.get_statistics()
        goose = stats["goose_latency"]

        lines = [
            "=" * 70,
            "HILS TEST REPORT",
            "=" * 70,
            f"  Total cycles:       {stats['cycles']}",
            f"  Errors:             {stats['errors']} ({stats['error_rate']*100:.2f}%)",
            f"  Time step:          {stats['time_step_ms']:.1f} ms",
            f"  Real-time factor:   {stats['real_time_factor']:.1f}x",
            "",
            "GOOSE Latency:",
            f"  Mean:               {goose.get('mean_ms', 0):.2f} ms",
            f"  Max:                {goose.get('max_ms', 0):.2f} ms",
            f"  P99:                {goose.get('p99_ms', 0):.2f} ms",
            (
                "  Meets target:       YES ✓"
                if goose.get("meets_target", False)
                else "  Meets target:       NO ✗"
            ),
            "=" * 70,
        ]

        return "\n".join(lines)


# ── Real-Time Synchronizer ──────────────────────────────────────────────────

class RealTimeSynchronizer:
    """Real-time synchronization for HILS.

    Ensures simulation runs at real-time pace with nanosecond
    precision using busy-wait for sub-millisecond accuracy.
    """

    def __init__(self, target_dt_ms: float = 4.0) -> None:
        self.target_dt: float = target_dt_ms / 1000.0
        self._last_time: float = 0.0
        self._drift_history: list[float] = []

    def tick(self) -> None:
        """Wait until next timestep."""
        now = time.perf_counter()

        if self._last_time > 0.0:
            elapsed = now - self._last_time
            sleep_time = self.target_dt - elapsed

            if sleep_time > 0.001:
                time.sleep(sleep_time - 0.001)

            # Busy-wait for remaining time
            while time.perf_counter() - self._last_time < self.target_dt:
                pass

        actual_dt = (
            time.perf_counter() - self._last_time
            if self._last_time > 0.0
            else self.target_dt
        )
        self._drift_history.append(actual_dt - self.target_dt)
        self._last_time = time.perf_counter()

    def get_drift_statistics(self) -> dict[str, float | int]:
        """Return timing drift statistics."""
        if not self._drift_history:
            return {}

        arr = np.array(self._drift_history)
        return {
            "mean_drift_ms": float(np.mean(arr) * 1000.0),
            "std_drift_ms": float(np.std(arr) * 1000.0),
            "max_drift_ms": float(np.max(arr) * 1000.0),
            "n_samples": len(arr),
        }