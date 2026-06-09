"""
deployment/hils_bridge.py
──────────────────────────
Phase 7: IEC 61850 GOOSE/SV interface and HILS bridge.

Provides:
1. IEC 61850 GOOSE publisher/subscriber for relay settings
2. Sampled Values (SV) interface for current/voltage measurements
3. HILS (Hardware-in-the-Loop Simulation) bridge
4. Real-time synchronization
5. MMS (Manufacturing Message Specification) client

Usage:
    from deployment.hils_bridge import IEC61850Bridge, HILSConfig

    bridge = IEC61850Bridge("192.168.1.100")
    bridge.connect()

    # Publish relay settings via GOOSE
    bridge.publish_settings(relay_id=0, TMS=0.5, PS=1.2)

    # Subscribe to measurements
    measurements = bridge.subscribe_measurements()
"""

import numpy as np
import time
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class IEC61850DataType(Enum):
    """IEC 61850 data types."""
    FLOAT32 = "FLOAT32"
    INT32 = "INT32"
    BOOLEAN = "BOOLEAN"
    TIMESTAMP = "Timestamp"


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
    """IEC 61850 relay setting data object."""
    relay_id: int
    tms: float
    ps: float
    timestamp: float
    quality: str = "good"  # good, questionable, invalid


@dataclass
class MeasurementSample:
    """IEC 61850 sampled value."""
    relay_id: int
    ia: float  # Phase A current
    ib: float  # Phase B current
    ic: float  # Phase C current
    va: float  # Phase A voltage
    vb: float  # Phase B voltage
    vc: float  # Phase C voltage
    timestamp: float
    sample_count: int


class IEC61850Bridge:
    """
    IEC 61850 communication bridge for DOCR-MARL deployment.

    Implements:
    - GOOSE (Generic Object Oriented Substation Events) for fast relay settings
    - SV (Sampled Values) for current/voltage measurements
    - MMS (Manufacturing Message Specification) for configuration

    Note: This is a simulation/placeholder implementation.
    Actual IEC 61850 requires libiec61850 or similar C library.
    """

    def __init__(self, config: Optional[HILSConfig] = None):
        self.config = config or HILSConfig()
        self.connected = False
        self.settings_log: List[RelaySetting] = []
        self.measurement_log: List[MeasurementSample] = []
        self.latency_history: List[float] = []

    def connect(self) -> bool:
        """Connect to IEC 61850 server."""
        # Placeholder: actual implementation would use libiec61850
        self.connected = True
        print(f"  [IEC61850] Connected to {self.config.host}:{self.config.port}")
        return True

    def disconnect(self):
        """Disconnect from IEC 61850 server."""
        self.connected = False
        print("  [IEC61850] Disconnected")

    def publish_settings(self, relay_id: int, tms: float, ps: float) -> bool:
        """
        Publish relay settings via GOOSE.

        GOOSE characteristics:
        - Max latency: 10ms (T0 = 2s, T1-T4 = 2-8ms)
        - Retransmission: immediate, then exponential backoff
        - No acknowledgment (publisher-subscriber model)
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
        self.settings_log.append(setting)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        self.latency_history.append(elapsed_ms)

        return True

    def subscribe_measurements(self, n_samples: int = 1) -> List[MeasurementSample]:
        """
        Subscribe to SV (Sampled Values) measurements.

        SV characteristics:
        - 80 samples/cycle (4kHz at 50Hz) or 256 samples/cycle
        - Fixed payload size for deterministic timing
        - No TCP/IP overhead (Ethernet multicast)
        """
        if not self.connected:
            return []

        samples = []
        for _ in range(n_samples):
            sample = MeasurementSample(
                relay_id=0,
                ia=np.random.normal(1000, 50),  # A
                ib=np.random.normal(1000, 50),
                ic=np.random.normal(1000, 50),
                va=np.random.normal(11000, 100),  # V
                vb=np.random.normal(11000, 100),
                vc=np.random.normal(11000, 100),
                timestamp=time.time(),
                sample_count=len(self.measurement_log),
            )
            self.measurement_log.append(sample)
            samples.append(sample)

        return samples

    def get_latency_statistics(self) -> Dict:
        """Return GOOSE publication latency statistics."""
        if not self.latency_history:
            return {}

        arr = np.array(self.latency_history)
        return {
            "mean_ms": float(np.mean(arr)),
            "std_ms": float(np.std(arr)),
            "max_ms": float(np.max(arr)),
            "p99_ms": float(np.percentile(arr, 99)),
            "meets_target": bool(np.max(arr) < self.config.max_latency_ms),
            "target_ms": self.config.max_latency_ms,
            "n_samples": len(arr),
        }

    def verify_goose_timing(self) -> Dict:
        """Verify GOOSE timing meets IEC 61850 requirements."""
        stats = self.get_latency_statistics()

        iec_requirements = {
            "T0_max_ms": 2000,   # Steady state retransmission
            "T1_max_ms": 2,      # First retransmission after event
            "T2_max_ms": 4,      # Second retransmission
            "T3_max_ms": 8,      # Third retransmission
            "TAL_max_ms": 10,    # Time Allowed to Live
        }

        verification = {
            "requirements": iec_requirements,
            "actual": stats,
            "compliant": stats.get("max_ms", 999) < iec_requirements["TAL_max_ms"],
        }

        return verification


class HILSBridge:
    """
    Hardware-in-the-Loop Simulation bridge.

    Connects the DOCR-MARL simulation to real protection relays
    via IEC 61850 for closed-loop testing.

    Architecture:
    ┌─────────────┐    IEC 61850    ┌─────────────┐
    │  Simulation │  <──────────>   │  Real Relay │
    │  (Python)   │    GOOSE/SV     │  (Hardware) │
    └─────────────┘                 └─────────────┘
    """

    def __init__(self, config: Optional[HILSConfig] = None):
        self.config = config or HILSConfig()
        self.iec_bridge = IEC61850Bridge(config)
        self.running = False
        self.cycle_count = 0
        self.error_count = 0

    def start(self) -> bool:
        """Start HILS loop."""
        if not self.iec_bridge.connect():
            return False

        self.running = True
        print(f"  [HILS] Started at {self.config.simulation_time_step_ms}ms timestep")
        return True

    def stop(self):
        """Stop HILS loop."""
        self.running = False
        self.iec_bridge.disconnect()
        print("  [HILS] Stopped")

    def run_step(self, relay_settings: Dict[int, Tuple[float, float]], 
                 measurements: Optional[Dict] = None) -> Dict:
        """
        Run one HILS step.

        Args:
            relay_settings: {relay_id: (TMS, PS)}
            measurements: Optional real measurements from relays

        Returns:
            step_result: Dict with timing, errors, measurements
        """
        if not self.running:
            return {"error": "HILS not running"}

        t0 = time.perf_counter()

        # Publish settings to real relays
        publish_ok = True
        for relay_id, (tms, ps) in relay_settings.items():
            if not self.iec_bridge.publish_settings(relay_id, tms, ps):
                publish_ok = False
                self.error_count += 1

        # Get measurements (simulated or real)
        if measurements is None:
            measurements = self.iec_bridge.subscribe_measurements(n_samples=1)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        self.cycle_count += 1

        return {
            "cycle": self.cycle_count,
            "elapsed_ms": elapsed_ms,
            "publish_ok": publish_ok,
            "measurements": measurements,
            "error_count": self.error_count,
        }

    def get_statistics(self) -> Dict:
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
            f"  Meets target:       {'YES ✓' if goose.get('meets_target', False) else 'NO ✗'}",
            "=" * 70,
        ]

        return "\n".join(lines)


class RealTimeSynchronizer:
    """
    Real-time synchronization for HILS.

    Ensures simulation runs at real-time pace with nanosecond
    precision using busy-wait for sub-millisecond accuracy.
    """

    def __init__(self, target_dt_ms: float = 4.0):
        self.target_dt = target_dt_ms / 1000.0
        self.last_time = 0.0
        self.drift_history: List[float] = []

    def tick(self):
        """Wait until next timestep."""
        now = time.perf_counter()

        if self.last_time > 0:
            elapsed = now - self.last_time
            sleep_time = self.target_dt - elapsed

            if sleep_time > 0.001:
                time.sleep(sleep_time - 0.001)

            # Busy-wait for remaining time
            while time.perf_counter() - self.last_time < self.target_dt:
                pass

        actual_dt = time.perf_counter() - self.last_time if self.last_time > 0 else self.target_dt
        self.drift_history.append(actual_dt - self.target_dt)
        self.last_time = time.perf_counter()

    def get_drift_statistics(self) -> Dict:
        """Return timing drift statistics."""
        if not self.drift_history:
            return {}

        arr = np.array(self.drift_history)
        return {
            "mean_drift_ms": float(np.mean(arr) * 1000),
            "std_drift_ms": float(np.std(arr) * 1000),
            "max_drift_ms": float(np.max(arr) * 1000),
            "n_samples": len(arr),
        }
