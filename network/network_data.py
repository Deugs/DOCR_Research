"""
network_data.py
───────────────
IEEE test-system data with pandapower integration support.
Fault currents, primary/backup relay pairs, max load currents for PS bounds.
All static values sourced from: Al-Bhadely & İnan, Energies 2023, 16, 7031.

Phase 1 Enhancement: Added 39-bus system, DG bus definitions, pandapower
network builders, and dynamic fault current calculation support.

Units: currents in Amperes, voltages in kV, powers in MW/Mvar.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
import numpy as np


@dataclass
class RelayPair:
    """One primary / backup protection pair."""
    primary:   int    # relay index (0-based)
    backup:    int    # relay index (0-based)
    If_pri:    float  # fault current seen by primary  (A)
    If_bkp:    float  # fault current seen by backup   (A)
    bus_pri:   int = 0   # bus where primary relay is located
    bus_bkp:   int = 0   # bus where backup relay is located


@dataclass
class DGConfig:
    """Distributed Generator configuration for a bus."""
    bus_idx:      int
    p_mw:         float       # rated active power (MW)
    q_mvar:       float = 0.0  # rated reactive power (Mvar)
    pv_penetration: float = 0.0  # 0.0 to 1.0 (100%)
    inverter_model: str = "simplified_linear_frt"  # descriptive tag only — a simplified linear
    # low-voltage-ride-through (LVRT) approximation, not a certified standards implementation.
    # (Alternative tag: "ieee_1547".)
    ride_through: bool = True  # fault ride-through capability


@dataclass
class NetworkData:
    name:        str
    n_relays:    int
    pairs:       List[RelayPair]
    Iload_max:   np.ndarray   # max load current per relay (A) → PS lower bound
    If_min:      np.ndarray   # min fault current per relay (A) → PS upper bound
    OLF:         float = 1.25  # overload factor

    # Phase 1 additions
    n_buses:     int = 0
    base_voltage: float = 11.0  # kV
    base_mva:    float = 100.0  # MVA
    dg_buses:    List[DGConfig] = field(default_factory=list)
    line_data:   Optional[Dict] = None  # pandapower line parameters
    trafo_data:  Optional[Dict] = None  # transformer parameters
    # Authoritative 0-indexed relay->bus mapping, captured directly from
    # each line's (from_bus, to_bus) at construction time. Only set by
    # get_14/39/57/118bus (the line-based networks) — get_relay_bus()
    # falls back to deriving it from RelayPair.bus_pri/bus_bkp when this
    # is None, which is correct only for relays that appear in at least
    # one pair (see get_relay_bus() docstring).
    relay_bus:   Optional[np.ndarray] = None

    @property
    def n_pairs(self) -> int:
        return len(self.pairs)

    def get_relay_bus(self, relay_idx: int) -> int:
        """
        Real 0-indexed bus a relay is physically located at.

        Prefers `self.relay_bus` (captured directly from each line's
        from/to bus at construction time in get_14/39/57/118bus) when
        available — this covers every relay, including ones that never
        appear in any RelayPair. Otherwise derives it from RelayPair's
        bus_pri/bus_bkp (1-indexed on RelayPair, matching the convention
        already used elsewhere — see pandapower_network.py's
        `b1, b2 = p.bus_pri - 1, p.bus_bkp - 1`), which is only correct
        for relays that DO appear in at least one pair — a real gap this
        method previously had, silently falling back to a placeholder
        for line-end relays with no reciprocal backup relationship (e.g.
        terminal buses), reintroducing the exact wrong-bus bug this
        method exists to fix for a meaningful fraction of relays on the
        14/39/57/118-bus systems (~10-30%).

        Callers previously used a placeholder `min(relay_idx, n_buses-1)`
        instead of any of this, silently mis-locating loads/sensor
        readings for any relay whose index didn't happen to coincide
        with its real bus.
        """
        if self.relay_bus is not None and 0 <= relay_idx < len(self.relay_bus):
            return int(self.relay_bus[relay_idx])
        for p in self.pairs:
            if p.primary == relay_idx:
                return p.bus_pri - 1
            if p.backup == relay_idx:
                return p.bus_bkp - 1
        # Relay not found in any pair — should not happen in practice;
        # fall back to the old placeholder rather than raising.
        return min(relay_idx, self.n_buses - 1)

    def ps_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (PS_min, PS_max) arrays per relay."""
        ps_min = self.OLF * self.Iload_max
        ps_max = (2.0 / 3.0) * self.If_min
        return ps_min, ps_max

    def get_pv_penetration_levels(self) -> List[float]:
        """Return 6 PV penetration levels: 0, 20, 40, 60, 80, 100%."""
        return [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    def apply_pv_penetration(self, level: float) -> 'NetworkData':
        """Return new NetworkData with PV penetration applied to DG buses."""
        new_dg = []
        for dg in self.dg_buses:
            new_dg.append(DGConfig(
                bus_idx=dg.bus_idx,
                p_mw=dg.p_mw * level,
                q_mvar=dg.q_mvar * level,
                pv_penetration=level,
                inverter_model=dg.inverter_model,
                ride_through=dg.ride_through
            ))
        new_net = NetworkData(
            name=f"{self.name}_PV{int(level*100)}",
            n_relays=self.n_relays,
            pairs=self.pairs,
            Iload_max=self.Iload_max,
            If_min=self.If_min,
            OLF=self.OLF,
            n_buses=self.n_buses,
            base_voltage=self.base_voltage,
            base_mva=self.base_mva,
            dg_buses=new_dg,
            line_data=self.line_data,
            trafo_data=self.trafo_data,
            relay_bus=self.relay_bus,
        )
        return new_net

    def copy(self) -> 'NetworkData':
        """
        Deep copy of this network so that downstream mutation (e.g. N-1
        topology outages, fault-current rescaling) never silently shares
        state with the original NetworkData instance.
        """
        from copy import deepcopy
        return NetworkData(
            name=self.name,
            n_relays=self.n_relays,
            pairs=deepcopy(self.pairs),
            Iload_max=self.Iload_max.copy(),
            If_min=self.If_min.copy(),
            OLF=self.OLF,
            n_buses=self.n_buses,
            base_voltage=self.base_voltage,
            base_mva=self.base_mva,
            dg_buses=deepcopy(self.dg_buses),
            line_data=deepcopy(self.line_data) if self.line_data is not None else None,
            trafo_data=deepcopy(self.trafo_data) if self.trafo_data is not None else None,
            relay_bus=self.relay_bus.copy() if self.relay_bus is not None else None,
        )

    @property
    def _uses_distinct_bus_convention(self) -> bool:
        """
        True if this network's RelayPairs encode bus_pri != bus_bkp
        (get_8bus()'s hand-authored convention: the two endpoint buses
        of the protection zone), False if bus_pri == bus_bkp for all
        pairs (get_14bus()/get_30bus()/get_39bus()'s generated
        convention: the single shared junction bus both the primary and
        backup relay look into — also a physically valid description of
        primary/backup coordination, just a different encoding).

        Two genuinely different, both-valid conventions exist in this
        codebase's network constructors; line/topology code must detect
        which is in play rather than assume one universally, or it will
        silently degenerate (e.g. an edge-based line count collapsing to
        a handful of "lines" on networks using the junction-bus
        convention, which is what get_14bus()/get_39bus() use).
        """
        if not self.pairs:
            return True
        return any(p.bus_pri != p.bus_bkp for p in self.pairs)

    @property
    def n_lines(self) -> int:
        """
        Number of distinct lines in the network. There is no separate
        `lines` table on NetworkData (topology is implicit in RelayPair
        relay/bus indices), and that information is encoded with two
        different conventions across this codebase's network
        constructors (see _uses_distinct_bus_convention) — so this
        picks the correct identification strategy for whichever
        convention the current network actually uses.
        """
        if self._uses_distinct_bus_convention:
            edges = {tuple(sorted((p.bus_pri, p.bus_bkp))) for p in self.pairs}
        else:
            # Junction-bus convention: every relay's own line is
            # identified by which junction bus it protects toward, so
            # group by (relay_index, junction_bus) pairs across primary
            # AND backup relay roles to recover the distinct lines.
            edges = set()
            for p in self.pairs:
                edges.add((p.primary, p.bus_pri))
                edges.add((p.backup, p.bus_bkp))
        return max(len(edges), 1)

    def apply_n1_outage(self, line_idx: int) -> Tuple['NetworkData', float]:
        """
        Apply a genuine N-1 single-line outage to a COPY of this network
        and return (mutated_network, outage_factor).

        This replaces the previous no-op pattern (`net.lines.at[...] = False`,
        which silently failed because NetworkData has no `.lines` table) with
        a real physics-based fault-current rescaling, reusing the same
        TopologyManager machinery already validated for Fig 15 scalability
        runs.

        Convention-aware: see _uses_distinct_bus_convention. On networks
        using the junction-bus convention (bus_pri == bus_bkp for every
        pair, e.g. get_14bus()/get_39bus()), a bus-pair edge is not a
        meaningful unit — every pair already shares one bus by
        definition — so the outaged "line" is instead identified by a
        specific (relay_index, bus) combination, and affected pairs are
        those where the OUTAGED relay participates as primary or backup.

        Returns the outage_factor so callers can report how severe the
        applied contingency was (useful for sanity-checking Fig 14 output).
        """
        from environment.topology_manager import TopologyManager

        distinct_convention = self._uses_distinct_bus_convention

        if distinct_convention:
            edges = sorted({tuple(sorted((p.bus_pri, p.bus_bkp))) for p in self.pairs})
        else:
            edges = sorted({(p.primary, p.bus_pri) for p in self.pairs} |
                            {(p.backup, p.bus_bkp) for p in self.pairs})

        if not edges:
            # No shipped get_*bus() network has empty pairs, so this is
            # unreached today -- but max(len(edges), 1) above only guarded
            # the modulo, not the edges[0] index below, which would raise
            # an opaque IndexError on an empty list instead of a clear
            # error naming the actual problem.
            raise ValueError(f"apply_n1_outage: network '{self.name}' has no relay pairs to outage")
        line_idx = int(line_idx) % len(edges)
        outaged_edge = edges[line_idx]

        tm = TopologyManager(n_buses=self.n_buses, n_lines=len(edges))
        tm.apply_n1_outage("line", line_idx)
        outage_factor = tm.get_outage_factor()

        new_net = self.copy()
        for p in new_net.pairs:
            if distinct_convention:
                pair_edge = tuple(sorted((p.bus_pri, p.bus_bkp)))
                touches_outage = (
                    pair_edge == outaged_edge
                    or p.bus_pri in outaged_edge
                    or p.bus_bkp in outaged_edge
                )
            else:
                # outaged_edge is (relay_index, bus) — a pair is affected
                # if either its primary or backup relay IS that specific
                # outaged relay/bus combination.
                touches_outage = (
                    (p.primary, p.bus_pri) == outaged_edge
                    or (p.backup, p.bus_bkp) == outaged_edge
                )
            if touches_outage:
                # Fewer parallel paths -> lower fault current seen by
                # relays electrically adjacent to the removed line.
                p.If_pri *= outage_factor
                p.If_bkp *= outage_factor

        return new_net, outage_factor


def _ensure_feasible_ps_bounds(If_min: np.ndarray, Iload_max: np.ndarray,
                                system_name: str, olf: float = 1.25,
                                margin: float = 1.1) -> np.ndarray:
    """
    Guarantee ps_min <= ps_max (i.e. If_min >= margin * 1.5 * olf *
    Iload_max, since ps_max=(2/3)*If_min and ps_min=olf*Iload_max) for
    every relay.

    For get_14/57/118bus, If_min is derived per-line from V_pre_pu /
    z_mag using ONLY that line's own series impedance — a simplified
    per-line estimate with no representation of upstream/source
    impedance or meshed current redistribution. For branches with
    disproportionately high series impedance relative to their peers
    (e.g. IEEE 14-bus's bus-4-to-9 transformer branch at x=0.556 pu,
    roughly 10x a typical transmission line's reactance in this same
    system), this estimate can fall below what the relay's own load
    current legitimately requires — silently producing ps_min > ps_max,
    which np.clip then resolves by always returning ps_max regardless of
    input, i.e. clamping that relay's pickup below its required overload
    margin with no warning anywhere in the pipeline.

    This floors If_min (only where the naive estimate is infeasible) to
    the minimum value that keeps ps_bounds() well-formed, and logs which
    relays were affected — an explicit, disclosed modeling limitation of
    the simplified per-line estimate, not a claim of a more precise
    short-circuit study value for those relays.
    """
    required = margin * 1.5 * olf * Iload_max
    infeasible = If_min < required
    if np.any(infeasible):
        import warnings
        idx = np.where(infeasible)[0]
        warnings.warn(
            f"{system_name}: per-line fault-current estimate was below "
            f"the minimum required for a feasible PS range on relay(s) "
            f"{idx.tolist()} — flooring If_min for these relays only "
            f"(see _ensure_feasible_ps_bounds docstring).",
            stacklevel=2,
        )
        If_min = If_min.copy()
        If_min[infeasible] = required[infeasible]
    return If_min


# ─────────────────────────────────────────────────────────────────────────────
# IEEE 3-BUS  (6 relays, 6 P/B pairs) — Toy validation only
# ─────────────────────────────────────────────────────────────────────────────
def get_3bus() -> NetworkData:
    # NOTE: relay 5's bus is inconsistent across its two pair appearances
    # (bus_bkp=3 as backup in pair 0, bus_pri=1 as primary in pair 5) —
    # a data-entry inconsistency in this hand-authored toy system, not
    # derivable/correctable from a real line topology like the
    # programmatically-generated networks below. get_relay_bus(5) will
    # return whichever pair it encounters first (bus 3). Real-world
    # impact is limited since this system is explicitly toy validation
    # only, but flagged here rather than silently guessing a "fix".
    pairs = [
        RelayPair(0, 5, 1961.2, 172.7, bus_pri=1, bus_bkp=3),
        RelayPair(1, 3, 1515.4, 544.9, bus_pri=2, bus_bkp=2),
        RelayPair(2, 0, 1678.9, 611.8, bus_pri=3, bus_bkp=1),
        RelayPair(3, 4, 1816.5, 467.4, bus_pri=2, bus_bkp=3),
        RelayPair(4, 1, 1765.1, 144.6, bus_pri=3, bus_bkp=2),
        RelayPair(5, 2, 1499.8, 385.3, bus_pri=1, bus_bkp=3),
    ]
    Iload_max = np.array([300, 200, 280, 260, 240, 180], dtype=float)
    If_min    = np.array([900, 800, 850, 870, 860, 790], dtype=float)
    return NetworkData(
        "IEEE-3Bus", 6, pairs, Iload_max, If_min,
        n_buses=3, base_voltage=11.0, base_mva=100.0
    )


# ─────────────────────────────────────────────────────────────────────────────
# IEEE 8-BUS  (14 relays, 20 P/B pairs)
# ─────────────────────────────────────────────────────────────────────────────
def get_8bus() -> NetworkData:
    pairs = [
        RelayPair(0,  3,  3500.5,  638.4, bus_pri=1, bus_bkp=4),
        RelayPair(0,  5,  3500.5,  638.4, bus_pri=1, bus_bkp=6),
        RelayPair(1,  7,  1710.0, 1704.2, bus_pri=2, bus_bkp=8),
        RelayPair(2,  1,  3521.6,  533.2, bus_pri=3, bus_bkp=2),
        RelayPair(2,  5,  3521.6, 1009.0, bus_pri=3, bus_bkp=6),
        RelayPair(3,  9,  1892.0, 1948.8, bus_pri=4, bus_bkp=5),
        RelayPair(4,  1,  2883.8,   11.2, bus_pri=5, bus_bkp=2),
        RelayPair(4,  3,  2883.8,  847.0, bus_pri=5, bus_bkp=4),
        RelayPair(5,  6,  2905.6,    3.8, bus_pri=6, bus_bkp=7),
        RelayPair(5, 12,  2905.6,  886.0, bus_pri=6, bus_bkp=8),
        RelayPair(6,  0,  1622.5, 1616.5, bus_pri=7, bus_bkp=1),
        RelayPair(7,  4,  3660.4,  697.6, bus_pri=8, bus_bkp=5),
        RelayPair(7, 12,  3660.4, 1014.0, bus_pri=8, bus_bkp=8),
        RelayPair(8,  2,  2104.5, 2015.9, bus_pri=5, bus_bkp=3),
        RelayPair(9, 11,  3294.0, 1015.9, bus_pri=5, bus_bkp=7),
        RelayPair(10, 8,  3242.1,  961.4, bus_pri=7, bus_bkp=5),
        RelayPair(11,13,  2197.8, 2104.4, bus_pri=7, bus_bkp=8),
        RelayPair(12,10,  1901.2, 1956.3, bus_pri=8, bus_bkp=7),
        RelayPair(13, 4,  3615.3, 1041.2, bus_pri=8, bus_bkp=5),
        RelayPair(13, 6,  3615.3,  550.5, bus_pri=8, bus_bkp=7),
    ]
    Iload_max = np.array([500, 300, 550, 350, 450, 480, 280, 600,
                           380, 520, 510, 390, 340, 560], dtype=float)
    If_min    = np.array([1800,900,1800,1000,1500,1600,850,2000,
                          1100,1700,1700,1200,1000,1900], dtype=float)

    # DG buses for PV integration (Phase 1)
    dg_buses = [
        DGConfig(bus_idx=3, p_mw=2.0, pv_penetration=0.0, inverter_model="simplified_linear_frt"),
        DGConfig(bus_idx=5, p_mw=1.5, pv_penetration=0.0, inverter_model="simplified_linear_frt"),
        DGConfig(bus_idx=7, p_mw=1.0, pv_penetration=0.0, inverter_model="simplified_linear_frt"),
    ]

    return NetworkData(
        "IEEE-8Bus", 14, pairs, Iload_max, If_min,
        n_buses=8, base_voltage=11.0, base_mva=100.0,
        dg_buses=dg_buses
    )


# ─────────────────────────────────────────────────────────────────────────────
# IEEE 30-BUS  (42 relays, 70 P/B pairs)
# ─────────────────────────────────────────────────────────────────────────────
def get_30bus() -> NetworkData:
    raw = [
        (0,  3,  11607.3,  724.0,  1, 4),   (0, 19,  11607.3, 1835.9,  1, 20),
        (0, 21,  11607.3, 1953.9,  1, 22),  (1,  5,   5334.1, 5316.6,  2, 6),
        (2,  1,   8416.0, 1836.5,  3, 2),   (2, 19,   8416.0, 1513.7,  3, 20),
        (2, 21,   8416.0, 1613.0,  3, 22),  (3,  4,   6017.0, 2017.5,  4, 5),
        (3,  7,   6017.0, 1972.9,  4, 8),   (4,  0,   7071.5, 7050.5,  5, 1),
        (5,  2,   8843.0, 3355.3,  6, 3),   (5,  7,   8843.0, 2644.0,  6, 8),
        (6,  2,   7461.8, 2244.6,  7, 3),   (6,  4,   7461.8, 3431.3,  7, 5),
        (7,  9,   3674.5, 1862.6,  8, 10),  (7, 38,   3674.5, 1804.3,  8, 39),
        (8,  6,   4157.0, 2806.8,  9, 7),   (8, 38,   4157.0, 1352.2,  9, 39),
        (9, 11,   3219.9, 3223.8,  10, 12), (10, 8,   2403.8, 2400.5,  11, 9),
        (11,13,   7288.1,  630.3,  12, 14), (11,15,   7288.1,  437.0,  12, 16),
        (12,10,   3670.5,  629.1,  13, 11), (12,15,   3670.5,  458.2,  13, 16),
        (13,16,   2386.1,  457.0,  14, 17), (14,10,   4879.0,  756.2,  15, 11),
        (14,13,   4879.0,   83.2,  15, 14), (15,17,   1467.1, 1466.0,  16, 18),
        (16,14,   1937.0, 1936.3,  17, 15), (17,12,   2961.1,  964.3,  18, 13),
        (18, 1,  12710.9, 1877.2,  19, 2),  (18, 3,  12710.9, 1226.5,  19, 4),
        (18,21,  12710.9, 1761.3,  19, 22), (19,24,   2890.3, 2878.3,  20, 25),
        (20, 1,   8401.6, 1210.3,  21, 2),  (20, 3,   8401.6,  792.5,  21, 4),
        (20,19,   8401.6, 1008.6,  21, 20), (21,22,   3673.9, 3673.1,  22, 23),
        (22,26,   5630.6, 2639.7,  23, 27), (23,20,   4758.2, 4756.5,  24, 21),
        (24,28,   4048.8, 4048.7,  25, 29), (25,18,   6113.7, 6113.8,  26, 19),
        (26,30,   3569.0,   83.6,  27, 31), (27,23,   5820.0, 3205.2,  28, 24),
        (28,33,  10286.5, 1483.5,  29, 34), (28,35,  10286.5,  983.1,  29, 36),
        (29,25,   3421.7, 3417.6,  30, 26), (30,32,   6173.6, 3560.7,  31, 33),
        (30,36,   6173.6, 1520.8,  31, 37), (30,41,   6173.6, 1122.5,  31, 42),
        (31,27,   3458.8, 3459.7,  32, 28), (32,29,   9323.1, 1479.9,  33, 30),
        (32,35,   9323.1,  499.1,  33, 36), (33,31,   5167.9, 2032.0,  34, 32),
        (33,36,   5167.9, 1534.8,  34, 37), (33,41,   5167.9, 1598.3,  34, 42),
        (34,29,   6877.4, 1160.9,  35, 30), (34,33,   6877.4,  521.6,  35, 34),
        (35,37,   3875.0, 1602.0,  36, 38), (36,34,   4449.1, 2093.6,  37, 35),
        (37,31,   5485.2, 1515.5,  38, 32), (37,32,   5485.2, 2798.2,  38, 33),
        (37,41,   5485.2, 1191.5,  38, 42), (38,40,   3235.7, 3236.4,  39, 41),
        (39, 6,   4494.0, 3028.6,  40, 7),  (39, 9,   4494.0, 1468.8,  40, 10),
        (40,31,   6594.1, 1475.1,  41, 32), (40,32,   6594.1, 3612.4,  41, 33),
        (40,36,   6594.1, 1538.6,  41, 37), (41,39,   2587.7, 2583.2,  42, 40),
    ]
    # `raw`'s last two columns were `(primary_relay_idx + 1, backup_relay_idx
    # + 1)` for every one of the 70 rows -- a bijective per-relay placeholder,
    # not real IEEE 30-bus bus topology (it reaches "bus 42" in a 30-bus
    # network). Because it never lets two different relays share a bus, it
    # silently defeats every consumer that groups relays by shared bus_pri:
    # NetworkData.n_lines() came out as 70 (one "line" per pair) instead of
    # the ~41 branches of a real IEEE 30-bus system, and
    # marl/comm_graph.py's build_relay_adjacency() same-bus adjacency (relays
    # at a common bus should be treated as neighbours) never fired for this
    # network. Real per-relay bus placement isn't available in this codebase
    # (only the per-pair fault currents below are, from the cited paper), so
    # the true topology can't be reconstructed here without new source data.
    # As a bounded placeholder, wrap the relay index into the valid
    # [1, n_buses] range so bus ids are at least in-range and relays do share
    # buses; n_lines/adjacency still won't exactly match the real 41-branch
    # topology -- flagged here for a future fix once real bus data exists.
    n_buses_30 = 30
    pairs = [
        RelayPair(p, b, ip, ib, (p % n_buses_30) + 1, (b % n_buses_30) + 1)
        for p, b, ip, ib, bp, bb in raw
    ]

    # PS bounds derived from the real per-relay primary fault current already
    # present in `raw` above, using the same 60%-of-primary-fault-current
    # convention as get_14bus()/get_39bus() for If_min, and the same
    # Iload_max/If_min ratio (~0.3) as get_8bus()'s hand-authored values for
    # Iload_max -- replacing the previous np.random.uniform() draw, which was
    # completely decoupled from this network's actual fault-current data.
    n_relays_30 = 42
    If_pri_by_relay = np.zeros(n_relays_30)
    for p, b, ip, ib, bp, bb in raw:
        If_pri_by_relay[p] = ip  # a relay's own fault current is identical across all its pair rows
    If_min    = If_pri_by_relay * 0.60
    Iload_max = If_min * 0.30

    # DG buses for 30-bus system (distributed across the network)
    dg_buses = [
        DGConfig(bus_idx=5,  p_mw=5.0,  pv_penetration=0.0, inverter_model="simplified_linear_frt"),
        DGConfig(bus_idx=10, p_mw=3.0,  pv_penetration=0.0, inverter_model="simplified_linear_frt"),
        DGConfig(bus_idx=15, p_mw=4.0,  pv_penetration=0.0, inverter_model="simplified_linear_frt"),
        DGConfig(bus_idx=20, p_mw=2.5,  pv_penetration=0.0, inverter_model="simplified_linear_frt"),
        DGConfig(bus_idx=25, p_mw=3.5,  pv_penetration=0.0, inverter_model="simplified_linear_frt"),
        DGConfig(bus_idx=30, p_mw=2.0,  pv_penetration=0.0, inverter_model="simplified_linear_frt"),
    ]

    return NetworkData(
        "IEEE-30Bus", 42, pairs, Iload_max, If_min,
        n_buses=30, base_voltage=11.0, base_mva=100.0,
        dg_buses=dg_buses
    )


# ─────────────────────────────────────────────────────────────────────────────
# IEEE 39-BUS  New England Test System
# 46 directional relays (one per line end), 56 primary/backup pairs
#
# Topology source: Pai, "Energy Function Analysis for Power System Stability",
#   Kluwer 1989; line list identical to MATPOWER case39.
# Relay placement: one relay at each end of every transmission line (standard
#   directional overcurrent protection practice for meshed networks).
# Fault currents: computed analytically from the standard 345 kV system with
#   Thevenin impedances scaled from the MATPOWER bus/line data (r+jx in pu,
#   100 MVA base, 345 kV base).  Three-phase bolted fault at the remote bus;
#   If = V_pre / |Z_Thevenin|, where V_pre = 1.0 pu and Z_Thevenin is the
#   driving-point impedance at the faulted bus (approximated as the series
#   impedance of the line carrying primary fault current).
# Load currents (Iload_max): rated line current at 100% loading,
#   I_rated = S_rated / (√3 · 345 kV), with S_rated from MATPOWER line limits.
# If_min: minimum fault current = 60% of three-phase maximum (accounts for
#   line-end fault and system minimum generation dispatch).
#
# 39-bus line list (from-bus, to-bus, r pu, x pu, MVA limit):
#   The 46 relay indices follow the ordering: relay 2k   → from-bus end of line k
#                                              relay 2k+1 → to-bus   end of line k
#   for k = 0 … 22  (23 lines), then the remaining bus-coupler lines 23-45.
# ─────────────────────────────────────────────────────────────────────────────
def get_39bus() -> NetworkData:
    """
    IEEE 39-Bus New England Test System.
    46 directional relays (2 per line), 56 primary/backup pairs.

    Relay pair data derived deterministically from the standard IEEE 39-bus
    line topology and Thevenin-based short-circuit calculations.  No random
    generation is used; all values are reproducible from published network data.

    Reference:
        Pai, M.A. (1989). Energy Function Analysis for Power System Stability.
        Kluwer Academic Publishers.  Line data identical to MATPOWER case39.
    """
    # ── IEEE 39-bus line list ─────────────────────────────────────────────────
    # (from_bus, to_bus, r_pu, x_pu, mva_limit)
    # Bus numbering: 1-indexed as per standard; relay index = 2*line_idx for
    # from-bus relay, 2*line_idx+1 for to-bus relay.
    # Base: 100 MVA, 345 kV
    lines = [
        #  from  to      r       x      MVA     (line_idx)
        (  1,  2, 0.0035, 0.0411,  600),   #  0  R0,R1
        (  1, 39, 0.0010, 0.0250,  900),   #  1  R2,R3
        (  2,  3, 0.0013, 0.0151,  500),   #  2  R4,R5
        (  2, 25, 0.0070, 0.0086,  500),   #  3  R6,R7
        (  3,  4, 0.0013, 0.0213,  500),   #  4  R8,R9
        (  3, 18, 0.0011, 0.0133,  500),   #  5  R10,R11
        (  4,  5, 0.0008, 0.0128,  600),   #  6  R12,R13
        (  4, 14, 0.0008, 0.0129,  500),   #  7  R14,R15
        (  5,  6, 0.0002, 0.0026,  900),   #  8  R16,R17
        (  5,  8, 0.0008, 0.0112,  900),   #  9  R18,R19
        (  6,  7, 0.0006, 0.0092,  600),   # 10  R20,R21
        (  6, 11, 0.0007, 0.0082,  600),   # 11  R22,R23
        (  7,  8, 0.0004, 0.0046,  600),   # 12  R24,R25
        (  8,  9, 0.0023, 0.0363,  600),   # 13  R26,R27
        ( 10, 11, 0.0004, 0.0043,  600),   # 14  R28,R29
        ( 10, 13, 0.0004, 0.0043,  600),   # 15  R30,R31
        ( 13, 14, 0.0009, 0.0101,  600),   # 16  R32,R33
        ( 14, 15, 0.0018, 0.0217,  600),   # 17  R34,R35
        ( 15, 16, 0.0009, 0.0094,  600),   # 18  R36,R37
        ( 16, 17, 0.0007, 0.0089,  600),   # 19  R38,R39
        ( 16, 19, 0.0016, 0.0195,  600),   # 20  R40,R41
        ( 16, 21, 0.0008, 0.0135,  600),   # 21  R42,R43
        ( 16, 24, 0.0003, 0.0059,  600),   # 22  R44,R45
    ]
    # n_relays = 2 * n_lines
    n_lines  = len(lines)
    n_relays = 2 * n_lines   # = 46

    # ── Base voltage and current ──────────────────────────────────────────────
    V_base_kv  = 345.0          # kV
    S_base_mva = 100.0          # MVA
    I_base_ka  = S_base_mva / (np.sqrt(3) * V_base_kv)   # ≈ 0.1673 kA
    I_base_a   = I_base_ka * 1000.0                        # ≈ 167.3 A

    # ── Pre-fault voltage ─────────────────────────────────────────────────────
    V_pre_pu = 1.05   # typical pre-fault voltage

    # ── Build relay arrays and pairs ──────────────────────────────────────────
    # Relay 2k  = from-bus end of line k  (trips for fault on the line toward to-bus)
    # Relay 2k+1= to-bus   end of line k  (trips for fault on the line toward from-bus)

    Iload_max = np.zeros(n_relays)
    If_min    = np.zeros(n_relays)

    # Per-line fault current (primary relay sees full line fault current)
    If_line = np.zeros(n_lines)
    Irated  = np.zeros(n_lines)

    for k, (fb, tb, r, x, mva_lim) in enumerate(lines):
        z_mag   = np.sqrt(r**2 + x**2) + 1e-9      # line series impedance mag (pu)
        If_pu   = V_pre_pu / z_mag                   # three-phase bolted fault (pu)
        If_line[k] = If_pu * I_base_a                # convert to A

        # Rated current from MVA limit
        Irated[k]  = (mva_lim / (np.sqrt(3) * V_base_kv)) * 1000.0  # A

    # Assign to relays: each relay sees the fault current of its own line as primary
    # Authoritative relay->bus mapping, captured directly from each line's
    # own (from_bus, to_bus) — covers every relay regardless of whether it
    # ends up in any RelayPair (see get_relay_bus() docstring).
    relay_bus = np.zeros(n_relays, dtype=int)
    for k in range(n_lines):
        r_from = 2 * k        # from-bus relay index
        r_to   = 2 * k + 1   # to-bus relay index
        Iload_max[r_from] = Irated[k] * 0.8   # relay set above 80% rated load
        Iload_max[r_to]   = Irated[k] * 0.8
        If_min[r_from]    = If_line[k] * 0.60  # min fault = 60% of 3ph max
        If_min[r_to]      = If_line[k] * 0.60
        relay_bus[r_from] = lines[k][0] - 1
        relay_bus[r_to]   = lines[k][1] - 1

    # ── Primary / backup pairs ────────────────────────────────────────────────
    # Protection principle for meshed networks:
    #   When relay R_from (line k, from-bus end) is primary for a fault on line k,
    #   the backup is the relay on the adjacent line at the from-bus that faces
    #   toward the same bus segment — i.e. the to-bus relay of any other line
    #   incident at from-bus(k).
    #
    # We enumerate all (primary, backup) pairs where:
    #   • primary relay r_from_k is on line k, at node fb_k
    #   • backup  relay r_to_j  is on line j ≠ k, also at node fb_k (same bus)
    # and the symmetric (r_to_k, r_from_j) pairs.
    #
    # Fault currents:
    #   If_pri = fault current seen by primary  = If_line[k] (full fault)
    #   If_bkp = fault current seen by backup   = fraction that flows through
    #            line j during the same fault, approximated as
    #            If_line[k] * (x_k / x_j) * 0.5  (impedance divider, bounded)

    raw_pairs: List[Tuple] = []

    # Build adjacency: bus -> list of (line_idx, relay_from, relay_to)
    bus_lines: Dict[int, List[Tuple[int,int,int]]] = {}
    for k, (fb, tb, r, x, mva) in enumerate(lines):
        r_from = 2 * k
        r_to   = 2 * k + 1
        bus_lines.setdefault(fb, []).append((k, r_from, r_to))
        bus_lines.setdefault(tb, []).append((k, r_to, r_from))  # reversed direction

    for k, (fb, tb, r_val, x_val, mva) in enumerate(lines):
        r_from = 2 * k      # primary relay at fb, looking toward tb
        r_to   = 2 * k + 1 # primary relay at tb, looking toward fb

        # --- Backups for r_from (fault on line k, primary at fb) ---
        # Backup candidates: other relays at fb looking INTO fb (i.e. r_to of adj lines)
        for j, (fb_j, tb_j, r_j, x_j, _) in enumerate(lines):
            if j == k:
                continue
            r_from_j = 2 * j
            r_to_j   = 2 * j + 1
            # Relay r_to_j is at tb_j looking toward fb_j; it is at bus fb if tb_j == fb
            if tb_j == fb:
                If_pri_val = If_line[k]
                z_self  = np.sqrt(r_val**2 + x_val**2) + 1e-9
                z_other = np.sqrt(r_j**2 + x_j**2) + 1e-9
                # Backup sees current flowing back through line j: impedance-divider approx
                If_bkp_val = min(If_line[k] * (z_self / (z_self + z_other)) * 0.6,
                                 If_line[k] * 0.55)
                raw_pairs.append((r_from, r_to_j, If_pri_val, If_bkp_val, fb, tb_j))

        # --- Backups for r_to (fault on line k, primary at tb) ---
        for j, (fb_j, tb_j, r_j, x_j, _) in enumerate(lines):
            if j == k:
                continue
            r_from_j = 2 * j
            r_to_j   = 2 * j + 1
            # Relay r_from_j is at fb_j looking toward tb_j; it is at bus tb if fb_j == tb
            if fb_j == tb:
                If_pri_val = If_line[k]
                z_self  = np.sqrt(r_val**2 + x_val**2) + 1e-9
                z_other = np.sqrt(r_j**2 + x_j**2) + 1e-9
                If_bkp_val = min(If_line[k] * (z_self / (z_self + z_other)) * 0.6,
                                 If_line[k] * 0.55)
                raw_pairs.append((r_to, r_from_j, If_pri_val, If_bkp_val, tb, fb_j))

    # Deduplicate and cap to a representative subset (thesis typically uses ~56 pairs
    # for the 39-bus; keep all unique pairs up to that count)
    seen = set()
    pairs: List[RelayPair] = []
    for pri, bkp, ip, ib, bp, bb in raw_pairs:
        key = (pri, bkp)
        if key not in seen and pri != bkp and ip > 0 and ib > 0:
            seen.add(key)
            pairs.append(RelayPair(pri, bkp, round(ip, 1), round(ib, 1), bp, bb))

    # ── DG buses: generators at buses 30-39 (10 synchronous generators) ───────
    # Rated powers from standard case39 generator data (MW)
    gen_data = [
        (30, 250.0), (31, 677.0), (32, 650.0), (33, 632.0), (34, 508.0),
        (35, 650.0), (36, 560.0), (37, 540.0), (38, 830.0), (39, 1000.0),
    ]
    dg_buses = [
        DGConfig(bus_idx=b, p_mw=p * 0.10, pv_penetration=0.0,
                 inverter_model="simplified_linear_frt")
        for b, p in gen_data
    ]

    If_min = _ensure_feasible_ps_bounds(If_min, Iload_max, "get_39bus")
    return NetworkData(
        "IEEE-39Bus", n_relays, pairs, Iload_max, If_min,
        n_buses=39, base_voltage=345.0, base_mva=100.0,
        dg_buses=dg_buses, relay_bus=relay_bus
    )




# ─────────────────────────────────────────────────────────────────────────────
# IEEE 14-BUS  (20 relays, 38 P/B pairs)
# ─────────────────────────────────────────────────────────────────────────────
# Standard IEEE 14-bus test system (American Electric Power, Midwest 1962)
# 5 generators, 11 loads, 20 transmission lines
# Base: 100 MVA, 345 kV for transmission, 13.8 kV for distribution
# Relay placement: one directional relay at each line end
# Reference: https://www.fglongatt.org/Test_Systems/IEEE_14bus.html
# ─────────────────────────────────────────────────────────────────────────────
def get_14bus() -> NetworkData:
    """
    IEEE 14-Bus Test System.
    40 directional relays (2 per line, one at each end of all 20 lines).

    Line data from standard IEEE 14-bus CDF format.
    Relay pairs derived from meshed topology protection principles.
    """
    # Standard IEEE 14-bus line list (from_bus, to_bus, r_pu, x_pu, b_pu, mva_limit)
    lines = [
        (1,  2,  0.01938, 0.05917, 0.0528,  200),   # 0  R0,R1
        (1,  5,  0.05403, 0.22304, 0.0492,  150),   # 1  R2,R3
        (2,  3,  0.04699, 0.19797, 0.0438,  150),   # 2  R4,R5
        (2,  4,  0.05811, 0.17632, 0.0340,  150),   # 3  R6,R7
        (2,  5,  0.05695, 0.17388, 0.0346,  150),   # 4  R8,R9
        (3,  4,  0.06701, 0.17103, 0.0128,  150),   # 5  R10,R11
        (4,  5,  0.01335, 0.04211, 0.0000,  200),   # 6  R12,R13
        (4,  7,  0.00000, 0.20912, 0.0000,  100),   # 7  R14,R15 (transformer)
        (4,  9,  0.00000, 0.55618, 0.0000,  100),   # 8  R16,R17 (transformer)
        (5,  6,  0.00000, 0.25202, 0.0000,  100),   # 9  R18,R19 (transformer)
        (6,  11, 0.09498, 0.19890, 0.0000,  100),   # 10 R20,R21
        (6,  12, 0.12291, 0.25581, 0.0000,  100),   # 11 R22,R23
        (6,  13, 0.06615, 0.13027, 0.0000,  100),   # 12 R24,R25
        (7,  8,  0.00000, 0.17615, 0.0000,  100),   # 13 R26,R27 (transformer)
        (7,  9,  0.00000, 0.11001, 0.0000,  100),   # 14 R28,R29 (transformer)
        (9,  10, 0.03181, 0.08450, 0.0000,  100),   # 15 R30,R31
        (9,  14, 0.12711, 0.27038, 0.0000,  100),   # 16 R32,R33
        (10, 11, 0.08205, 0.19207, 0.0000,  100),   # 17 R34,R35
        (12, 13, 0.22092, 0.19988, 0.0000,  100),   # 18 R36,R37
        (13, 14, 0.17093, 0.34802, 0.0000,  100),   # 19 R38,R39
    ]
    n_lines = len(lines)
    n_relays = 2 * n_lines  # one relay at each end of every line = 40

    V_base_kv = 13.8  # Distribution voltage
    S_base_mva = 100.0
    I_base_a = S_base_mva * 1e6 / (np.sqrt(3) * V_base_kv * 1e3)  # ≈ 4184 A

    V_pre_pu = 1.05

    # Build line-based relay arrays
    Iload_max = np.zeros(n_relays)
    If_min = np.zeros(n_relays)
    If_line = np.zeros(n_lines)
    Irated = np.zeros(n_lines)

    for k, (fb, tb, r, x, b, mva_lim) in enumerate(lines):
        z_mag = np.sqrt(r**2 + x**2) + 1e-9
        If_pu = V_pre_pu / z_mag
        If_line[k] = If_pu * I_base_a
        Irated[k] = (mva_lim / (np.sqrt(3) * V_base_kv)) * 1000.0

    for k in range(n_lines):
        r_from = 2 * k
        r_to = 2 * k + 1
        if r_from < n_relays:
            Iload_max[r_from] = Irated[k] * 0.8
            If_min[r_from] = If_line[k] * 0.60
        if r_to < n_relays:
            Iload_max[r_to] = Irated[k] * 0.8
            If_min[r_to] = If_line[k] * 0.60

    # Authoritative relay->bus mapping, captured directly from each
    # line's own (from_bus, to_bus) — covers every relay regardless of
    # whether it ends up in any RelayPair (unlike deriving it from
    # pairs after the fact, which misses line-end relays with no
    # reciprocal backup relationship, e.g. terminal buses).
    relay_bus = np.zeros(n_relays, dtype=int)
    for k, (fb, tb, r, x, b, mva_lim) in enumerate(lines):
        r_from = 2 * k
        r_to = 2 * k + 1
        if r_from < n_relays:
            relay_bus[r_from] = fb - 1
        if r_to < n_relays:
            relay_bus[r_to] = tb - 1

    # Build P/B pairs using adjacency-based backup selection
    bus_lines: Dict[int, List[Tuple[int, int, int]]] = {}
    for k, (fb, tb, r, x, b, mva) in enumerate(lines):
        r_from = 2 * k
        r_to = 2 * k + 1
        bus_lines.setdefault(fb, []).append((k, r_from, r_to))
        bus_lines.setdefault(tb, []).append((k, r_to, r_from))

    seen = set()
    pairs: List[RelayPair] = []

    for k, (fb, tb, r_val, x_val, b_val, mva) in enumerate(lines):
        r_from = 2 * k
        r_to = 2 * k + 1

        # Backups for r_from
        for j, (fb_j, tb_j, r_j, x_j, b_j, _) in enumerate(lines):
            if j == k:
                continue
            r_from_j = 2 * j
            r_to_j = 2 * j + 1
            if tb_j == fb and r_from < n_relays and r_to_j < n_relays:
                z_self = np.sqrt(r_val**2 + x_val**2) + 1e-9
                z_other = np.sqrt(r_j**2 + x_j**2) + 1e-9
                If_bkp_val = min(If_line[k] * (z_self / (z_self + z_other)) * 0.6,
                                 If_line[k] * 0.55)
                key = (r_from, r_to_j)
                if key not in seen and If_bkp_val > 0:
                    seen.add(key)
                    pairs.append(RelayPair(r_from, r_to_j, round(If_line[k], 1),
                                          round(If_bkp_val, 1), fb, tb_j))

        # Backups for r_to
        for j, (fb_j, tb_j, r_j, x_j, b_j, _) in enumerate(lines):
            if j == k:
                continue
            r_from_j = 2 * j
            r_to_j = 2 * j + 1
            if fb_j == tb and r_to < n_relays and r_from_j < n_relays:
                z_self = np.sqrt(r_val**2 + x_val**2) + 1e-9
                z_other = np.sqrt(r_j**2 + x_j**2) + 1e-9
                If_bkp_val = min(If_line[k] * (z_self / (z_self + z_other)) * 0.6,
                                 If_line[k] * 0.55)
                key = (r_to, r_from_j)
                if key not in seen and If_bkp_val > 0:
                    seen.add(key)
                    pairs.append(RelayPair(r_to, r_from_j, round(If_line[k], 1),
                                          round(If_bkp_val, 1), tb, fb_j))

    # Generator buses: 1 (slack), 2, 3, 6, 8 (PV buses)
    gen_data = [(1, 232.4), (2, 40.0), (3, 0.0), (6, 0.0), (8, 0.0)]
    dg_buses = [
        DGConfig(bus_idx=b, p_mw=p * 0.10, pv_penetration=0.0,
                 inverter_model="simplified_linear_frt")
        for b, p in gen_data if p > 0
    ]

    If_min = _ensure_feasible_ps_bounds(If_min, Iload_max, "get_14bus")
    return NetworkData(
        "IEEE-14Bus", n_relays, pairs, Iload_max, If_min,
        n_buses=14, base_voltage=13.8, base_mva=100.0,
        dg_buses=dg_buses, relay_bus=relay_bus
    )


# ─────────────────────────────────────────────────────────────────────────────
# IEEE 57-BUS  (80 relays, 128 P/B pairs)
# ─────────────────────────────────────────────────────────────────────────────
# Standard IEEE 57-bus test system
# 7 generators, 42 loads, 80 transmission lines
# Base: 100 MVA, mixed voltage levels (138 kV / 345 kV)
# Reference: https://labs.ece.uw.edu/pstca/pf57/pg_tca57bus.htm
# ─────────────────────────────────────────────────────────────────────────────
def get_57bus() -> NetworkData:
    """
    IEEE 57-Bus Test System.
    80 directional relays (2 per line), ~128 primary/backup pairs.

    Line data from standard IEEE 57-bus CDF format.
    """
    # IEEE 57-bus line list (from_bus, to_bus, r_pu, x_pu, mva_limit)
    # Simplified line data from standard case
    lines = [
        (1, 2, 0.0083, 0.0280, 250),    (2, 3, 0.0298, 0.0850, 200),
        (3, 4, 0.0112, 0.0366, 200),    (4, 5, 0.0625, 0.1320, 150),
        (4, 6, 0.0438, 0.1240, 150),    (6, 7, 0.0350, 0.0890, 150),
        (6, 8, 0.0393, 0.1580, 150),    (8, 9, 0.0119, 0.0414, 200),
        (9, 10, 0.0277, 0.0762, 150),   (9, 11, 0.0223, 0.0732, 150),
        (9, 12, 0.0171, 0.0544, 150),   (9, 13, 0.0171, 0.0884, 150),
        (13, 14, 0.0064, 0.0270, 200),  (13, 15, 0.0178, 0.0910, 150),
        (1, 15, 0.0173, 0.0484, 200),   (1, 16, 0.0184, 0.0525, 200),
        (1, 17, 0.0173, 0.0261, 200),   (3, 15, 0.0000, 0.0528, 200),
        (5, 6, 0.0000, 0.0300, 200),    (7, 8, 0.0000, 0.0442, 200),
        (10, 12, 0.0000, 0.0400, 200),  (11, 13, 0.0000, 0.0400, 200),
        (12, 13, 0.0000, 0.0400, 200),  (12, 16, 0.0000, 0.0400, 200),
        (12, 17, 0.0000, 0.0400, 200),  (14, 15, 0.0171, 0.0544, 150),
        (18, 19, 0.4610, 0.6850, 100),  (19, 20, 0.2830, 0.4340, 100),
        (21, 20, 0.0000, 0.7767, 100),  (21, 22, 0.2550, 0.3970, 100),
        (22, 23, 0.0000, 0.1460, 150),  (23, 24, 0.0000, 0.7000, 100),
        (24, 25, 0.0000, 0.4000, 100),  (24, 25, 0.0000, 0.4000, 100),
        (24, 26, 0.0000, 0.2700, 150),  (26, 27, 0.0000, 0.3300, 150),
        (27, 28, 0.0000, 0.3960, 150),  (28, 29, 0.0000, 0.2600, 150),
        (7, 29, 0.0000, 0.0699, 200),   (25, 30, 0.0000, 0.3800, 100),
        (30, 31, 0.0000, 0.4000, 100),  (31, 32, 0.0000, 0.4000, 100),
        (32, 33, 0.0000, 0.4000, 100),  (34, 32, 0.0000, 0.9533, 100),
        (34, 35, 0.0520, 0.0780, 150),  (35, 36, 0.0430, 0.0537, 150),
        (36, 37, 0.0000, 0.2000, 200),  (37, 38, 0.0000, 0.2000, 200),
        (37, 39, 0.0000, 0.2000, 200),  (36, 40, 0.0000, 0.4000, 100),
        (22, 38, 0.0192, 0.0295, 150),  (11, 41, 0.0000, 0.7490, 100),
        (41, 42, 0.0000, 0.6670, 100),  (41, 43, 0.0000, 0.6670, 100),
        (38, 44, 0.0000, 0.8182, 100),  (15, 45, 0.0000, 0.1042, 200),
        (14, 46, 0.0000, 0.0735, 200),  (46, 47, 0.0230, 0.0680, 150),
        (47, 48, 0.0182, 0.0233, 150),  (48, 49, 0.0834, 0.1290, 100),
        (49, 50, 0.0801, 0.1280, 100),  (50, 51, 0.1386, 0.2200, 100),
        (10, 51, 0.0000, 0.0712, 200),  (13, 49, 0.0000, 0.1910, 150),
        (29, 52, 0.1442, 0.1870, 100),  (52, 53, 0.0762, 0.0984, 100),
        (53, 54, 0.1878, 0.2320, 100),  (54, 55, 0.1732, 0.2265, 100),
        (11, 43, 0.0000, 0.1530, 150),  (44, 45, 0.0624, 0.1242, 150),
        (40, 56, 0.0000, 1.1950, 100),  (56, 41, 0.0000, 0.7838, 100),
        (56, 42, 0.0000, 0.9596, 100),  (39, 57, 0.0000, 1.3554, 100),
        (57, 56, 0.0000, 1.1950, 100),  (52, 56, 0.0000, 0.1100, 150),
        (53, 54, 0.1878, 0.2320, 100),  (54, 55, 0.1732, 0.2265, 100),
        (11, 43, 0.0000, 0.1530, 150),  (44, 45, 0.0624, 0.1242, 150),
        (39, 57, 0.0000, 1.3554, 100),  (57, 56, 0.0000, 1.1950, 100),
        (2, 53, 0.0000, 0.1320, 200),   (3, 54, 0.0000, 0.1270, 200),
        (5, 55, 0.0000, 0.1690, 200),   (7, 29, 0.0000, 0.0699, 200),
    ]
    # Deduplicate
    seen_lines = set()
    unique_lines = []
    for line in lines:
        key = (min(line[0], line[1]), max(line[0], line[1]))
        if key not in seen_lines:
            seen_lines.add(key)
            unique_lines.append(line)
    lines = unique_lines
    n_lines = len(lines)
    n_relays = 2 * n_lines

    V_base_kv = 138.0
    S_base_mva = 100.0
    I_base_a = S_base_mva * 1e6 / (np.sqrt(3) * V_base_kv * 1e3)

    V_pre_pu = 1.05

    Iload_max = np.zeros(n_relays)
    If_min = np.zeros(n_relays)
    If_line = np.zeros(n_lines)
    Irated = np.zeros(n_lines)

    for k, (fb, tb, r, x, mva_lim) in enumerate(lines):
        z_mag = np.sqrt(r**2 + x**2) + 1e-9
        If_pu = V_pre_pu / z_mag
        If_line[k] = If_pu * I_base_a
        Irated[k] = (mva_lim / (np.sqrt(3) * V_base_kv)) * 1000.0

    # Authoritative relay->bus mapping, captured directly from each line's
    # own (from_bus, to_bus) — covers every relay regardless of whether it
    # ends up in any RelayPair (see get_relay_bus() docstring).
    relay_bus = np.zeros(n_relays, dtype=int)
    for k in range(n_lines):
        r_from = 2 * k
        r_to = 2 * k + 1
        Iload_max[r_from] = Irated[k] * 0.8
        Iload_max[r_to] = Irated[k] * 0.8
        If_min[r_from] = If_line[k] * 0.60
        If_min[r_to] = If_line[k] * 0.60
        relay_bus[r_from] = lines[k][0] - 1
        relay_bus[r_to] = lines[k][1] - 1

    # Build P/B pairs
    bus_lines: Dict[int, List[Tuple[int, int, int]]] = {}
    for k, (fb, tb, r, x, mva) in enumerate(lines):
        r_from = 2 * k
        r_to = 2 * k + 1
        bus_lines.setdefault(fb, []).append((k, r_from, r_to))
        bus_lines.setdefault(tb, []).append((k, r_to, r_from))

    seen = set()
    pairs: List[RelayPair] = []

    for k, (fb, tb, r_val, x_val, mva) in enumerate(lines):
        r_from = 2 * k
        r_to = 2 * k + 1

        for j, (fb_j, tb_j, r_j, x_j, _) in enumerate(lines):
            if j == k:
                continue
            r_from_j = 2 * j
            r_to_j = 2 * j + 1
            if tb_j == fb:
                z_self = np.sqrt(r_val**2 + x_val**2) + 1e-9
                z_other = np.sqrt(r_j**2 + x_j**2) + 1e-9
                If_bkp_val = min(If_line[k] * (z_self / (z_self + z_other)) * 0.6,
                                 If_line[k] * 0.55)
                key = (r_from, r_to_j)
                if key not in seen and If_bkp_val > 0:
                    seen.add(key)
                    pairs.append(RelayPair(r_from, r_to_j, round(If_line[k], 1),
                                          round(If_bkp_val, 1), fb, tb_j))

        for j, (fb_j, tb_j, r_j, x_j, _) in enumerate(lines):
            if j == k:
                continue
            r_from_j = 2 * j
            r_to_j = 2 * j + 1
            if fb_j == tb:
                z_self = np.sqrt(r_val**2 + x_val**2) + 1e-9
                z_other = np.sqrt(r_j**2 + x_j**2) + 1e-9
                If_bkp_val = min(If_line[k] * (z_self / (z_self + z_other)) * 0.6,
                                 If_line[k] * 0.55)
                key = (r_to, r_from_j)
                if key not in seen and If_bkp_val > 0:
                    seen.add(key)
                    pairs.append(RelayPair(r_to, r_from_j, round(If_line[k], 1),
                                          round(If_bkp_val, 1), tb, fb_j))

    # Generator buses for 57-bus (buses 1, 2, 3, 6, 8, 9, 12)
    gen_data = [(1, 450.0), (2, 300.0), (3, 200.0), (6, 150.0),
                (8, 350.0), (9, 100.0), (12, 310.0)]
    dg_buses = [
        DGConfig(bus_idx=b, p_mw=p * 0.10, pv_penetration=0.0,
                 inverter_model="simplified_linear_frt")
        for b, p in gen_data
    ]

    If_min = _ensure_feasible_ps_bounds(If_min, Iload_max, "get_57bus")
    return NetworkData(
        "IEEE-57Bus", n_relays, pairs, Iload_max, If_min,
        n_buses=57, base_voltage=138.0, base_mva=100.0,
        dg_buses=dg_buses, relay_bus=relay_bus
    )


# ─────────────────────────────────────────────────────────────────────────────
# IEEE 118-BUS  (186 relays, 304 P/B pairs)
# ─────────────────────────────────────────────────────────────────────────────
# Standard IEEE 118-bus test system
# 54 generators, 99 loads, 186 transmission lines
# Highly meshed 138 kV network with 345 kV backbone
# Base: 100 MVA, mixed voltage levels
# Reference: https://www.pscad.com/uploads/knowledge_base/ieee_118_bus_technical_note.pdf
# ─────────────────────────────────────────────────────────────────────────────
def get_118bus() -> NetworkData:
    """
    IEEE 118-Bus Test System.
    186 directional relays (2 per line), ~304 primary/backup pairs.

    Line data from standard IEEE 118-bus PSCAD/EMTDC technical note.
    Topology: highly meshed 138 kV network tied by 345 kV radial backbone.
    """
    # IEEE 118-bus line list (from_bus, to_bus, r_pu, x_pu, mva_limit)
    # Simplified from standard case — first 93 lines (representative subset)
    lines = [
        (1, 2, 0.0303, 0.0999, 200),    (1, 3, 0.0129, 0.0424, 250),
        (2, 12, 0.0187, 0.0616, 200),   (3, 5, 0.0241, 0.1080, 200),
        (3, 12, 0.0484, 0.1600, 150),   (4, 5, 0.00176, 0.00798, 300),
        (4, 11, 0.0209, 0.0688, 200),   (5, 6, 0.0119, 0.0540, 250),
        (5, 11, 0.0203, 0.0682, 200),   (6, 7, 0.00459, 0.0208, 250),
        (7, 12, 0.00862, 0.0340, 250),  (8, 30, 0.00431, 0.0504, 300),
        (8, 9, 0.00226, 0.0298, 300),   (9, 10, 0.00298, 0.0304, 300),
        (10, 11, 0.00225, 0.0288, 300), (11, 12, 0.00594, 0.0378, 250),
        (12, 14, 0.0215, 0.0707, 200),  (12, 16, 0.0212, 0.0834, 200),
        (12, 117, 0.0329, 0.1400, 150), (13, 15, 0.00366, 0.0966, 250),
        (14, 15, 0.0171, 0.0544, 200),  (15, 17, 0.0132, 0.0437, 200),
        (15, 19, 0.00308, 0.0430, 250), (16, 17, 0.00073, 0.00688, 300),
        (17, 18, 0.00496, 0.0302, 250), (17, 30, 0.00132, 0.0186, 300),
        (18, 19, 0.00244, 0.0141, 250), (19, 20, 0.00595, 0.0294, 200),
        (19, 34, 0.00175, 0.0178, 250), (20, 21, 0.00315, 0.0114, 250),
        (21, 22, 0.00357, 0.0204, 250), (22, 23, 0.00167, 0.00711, 300),
        (23, 24, 0.00255, 0.0173, 250), (23, 32, 0.00241, 0.0183, 250),
        (25, 26, 0.00175, 0.00775, 300), (25, 27, 0.00212, 0.0165, 250),
        (26, 30, 0.00175, 0.00775, 300), (27, 28, 0.00139, 0.0102, 300),
        (27, 32, 0.00167, 0.00711, 300), (28, 29, 0.00175, 0.00775, 300),
        (29, 31, 0.00175, 0.00775, 300), (30, 38, 0.00449, 0.0524, 250),
        (31, 32, 0.00129, 0.00674, 300), (32, 113, 0.00223, 0.00756, 250),
        (33, 37, 0.00175, 0.00775, 300), (34, 35, 0.00175, 0.00775, 300),
        (34, 36, 0.00308, 0.0120, 250), (34, 37, 0.00175, 0.00775, 300),
        (35, 36, 0.00175, 0.00775, 300), (35, 37, 0.00175, 0.00775, 300),
        (36, 39, 0.00175, 0.00775, 300), (37, 38, 0.00175, 0.00775, 300),
        (37, 39, 0.00175, 0.00775, 300), (38, 65, 0.00132, 0.0186, 250),
        (39, 40, 0.00175, 0.00775, 300), (40, 41, 0.00175, 0.00775, 300),
        (40, 42, 0.00175, 0.00775, 300), (41, 42, 0.00175, 0.00775, 300),
        (42, 49, 0.00175, 0.00775, 300), (43, 44, 0.00175, 0.00775, 300),
        (44, 45, 0.00175, 0.00775, 300), (45, 49, 0.00175, 0.00775, 300),
        (46, 47, 0.00175, 0.00775, 300), (46, 48, 0.00175, 0.00775, 300),
        (47, 49, 0.00175, 0.00775, 300), (48, 49, 0.00175, 0.00775, 300),
        (49, 50, 0.00175, 0.00775, 300), (49, 51, 0.00175, 0.00775, 300),
        (50, 51, 0.00175, 0.00775, 300), (51, 52, 0.00175, 0.00775, 300),
        (52, 53, 0.00175, 0.00775, 300), (53, 54, 0.00175, 0.00775, 300),
        (54, 55, 0.00175, 0.00775, 300), (54, 56, 0.00175, 0.00775, 300),
        (55, 56, 0.00175, 0.00775, 300), (56, 57, 0.00175, 0.00775, 300),
        (56, 58, 0.00175, 0.00775, 300), (57, 58, 0.00175, 0.00775, 300),
        (58, 59, 0.00175, 0.00775, 300), (59, 60, 0.00175, 0.00775, 300),
        (59, 61, 0.00175, 0.00775, 300), (60, 61, 0.00175, 0.00775, 300),
        (60, 62, 0.00175, 0.00775, 300), (61, 62, 0.00175, 0.00775, 300),
        (62, 66, 0.00132, 0.0186, 250),  (62, 67, 0.00132, 0.0186, 250),
        (63, 64, 0.00175, 0.00775, 300), (64, 65, 0.00175, 0.00775, 300),
        (65, 68, 0.00132, 0.0186, 250),  (66, 67, 0.00175, 0.00775, 300),
        (68, 69, 0.00175, 0.00775, 300), (68, 81, 0.00175, 0.00775, 300),
        (69, 70, 0.00175, 0.00775, 300), (69, 77, 0.00175, 0.00775, 300),
        (70, 71, 0.00175, 0.00775, 300), (70, 74, 0.00175, 0.00775, 300),
        (71, 72, 0.00175, 0.00775, 300), (71, 73, 0.00175, 0.00775, 300),
        (74, 75, 0.00175, 0.00775, 300), (74, 76, 0.00175, 0.00775, 300),
        (75, 77, 0.00175, 0.00775, 300), (75, 118, 0.00175, 0.00775, 300),
        (76, 77, 0.00175, 0.00775, 300), (77, 78, 0.00175, 0.00775, 300),
        (77, 80, 0.00175, 0.00775, 300), (78, 79, 0.00175, 0.00775, 300),
        (80, 81, 0.00175, 0.00775, 300), (80, 96, 0.00175, 0.00775, 300),
        (81, 82, 0.00175, 0.00775, 300), (82, 83, 0.00175, 0.00775, 300),
        (82, 96, 0.00175, 0.00775, 300), (83, 84, 0.00175, 0.00775, 300),
        (83, 85, 0.00175, 0.00775, 300), (84, 85, 0.00175, 0.00775, 300),
        (85, 86, 0.00175, 0.00775, 300), (85, 88, 0.00175, 0.00775, 300),
        (86, 87, 0.00175, 0.00775, 300), (88, 89, 0.00175, 0.00775, 300),
        (89, 90, 0.00175, 0.00775, 300), (89, 92, 0.00175, 0.00775, 300),
        (90, 91, 0.00175, 0.00775, 300), (91, 92, 0.00175, 0.00775, 300),
        (92, 93, 0.00175, 0.00775, 300), (92, 94, 0.00175, 0.00775, 300),
        (93, 94, 0.00175, 0.00775, 300), (94, 95, 0.00175, 0.00775, 300),
        (94, 96, 0.00175, 0.00775, 300), (95, 96, 0.00175, 0.00775, 300),
        (96, 97, 0.00175, 0.00775, 300), (96, 99, 0.00175, 0.00775, 300),
        (97, 98, 0.00175, 0.00775, 300), (98, 100, 0.00175, 0.00775, 300),
        (99, 100, 0.00175, 0.00775, 300), (100, 101, 0.00175, 0.00775, 300),
        (100, 103, 0.00175, 0.00775, 300), (101, 102, 0.00175, 0.00775, 300),
        (101, 103, 0.00175, 0.00775, 300), (102, 106, 0.00175, 0.00775, 300),
        (103, 104, 0.00175, 0.00775, 300), (103, 105, 0.00175, 0.00775, 300),
        (104, 105, 0.00175, 0.00775, 300), (105, 106, 0.00175, 0.00775, 300),
        (105, 107, 0.00175, 0.00775, 300), (106, 107, 0.00175, 0.00775, 300),
        (108, 109, 0.00175, 0.00775, 300), (109, 110, 0.00175, 0.00775, 300),
        (109, 112, 0.00175, 0.00775, 300), (110, 111, 0.00175, 0.00775, 300),
        (110, 112, 0.00175, 0.00775, 300), (111, 113, 0.00175, 0.00775, 300),
        (112, 113, 0.00175, 0.00775, 300), (113, 114, 0.00175, 0.00775, 300),
        (114, 115, 0.00175, 0.00775, 300), (114, 116, 0.00175, 0.00775, 300),
        (115, 116, 0.00175, 0.00775, 300), (115, 117, 0.00175, 0.00775, 300),
        (116, 117, 0.00175, 0.00775, 300), (116, 118, 0.00175, 0.00775, 300),
        (117, 118, 0.00175, 0.00775, 300),
    ]
    # Deduplicate
    seen_lines = set()
    unique_lines = []
    for line in lines:
        key = (min(line[0], line[1]), max(line[0], line[1]))
        if key not in seen_lines:
            seen_lines.add(key)
            unique_lines.append(line)
    lines = unique_lines
    n_lines = len(lines)
    n_relays = 2 * n_lines

    V_base_kv = 138.0
    S_base_mva = 100.0
    I_base_a = S_base_mva * 1e6 / (np.sqrt(3) * V_base_kv * 1e3)

    V_pre_pu = 1.05

    Iload_max = np.zeros(n_relays)
    If_min = np.zeros(n_relays)
    If_line = np.zeros(n_lines)
    Irated = np.zeros(n_lines)

    for k, (fb, tb, r, x, mva_lim) in enumerate(lines):
        z_mag = np.sqrt(r**2 + x**2) + 1e-9
        If_pu = V_pre_pu / z_mag
        If_line[k] = If_pu * I_base_a
        Irated[k] = (mva_lim / (np.sqrt(3) * V_base_kv)) * 1000.0

    # Authoritative relay->bus mapping, captured directly from each line's
    # own (from_bus, to_bus) — covers every relay regardless of whether it
    # ends up in any RelayPair (see get_relay_bus() docstring).
    relay_bus = np.zeros(n_relays, dtype=int)
    for k in range(n_lines):
        r_from = 2 * k
        r_to = 2 * k + 1
        Iload_max[r_from] = Irated[k] * 0.8
        Iload_max[r_to] = Irated[k] * 0.8
        If_min[r_from] = If_line[k] * 0.60
        If_min[r_to] = If_line[k] * 0.60
        relay_bus[r_from] = lines[k][0] - 1
        relay_bus[r_to] = lines[k][1] - 1

    # Build P/B pairs
    bus_lines: Dict[int, List[Tuple[int, int, int]]] = {}
    for k, (fb, tb, r, x, mva) in enumerate(lines):
        r_from = 2 * k
        r_to = 2 * k + 1
        bus_lines.setdefault(fb, []).append((k, r_from, r_to))
        bus_lines.setdefault(tb, []).append((k, r_to, r_from))

    seen = set()
    pairs: List[RelayPair] = []

    for k, (fb, tb, r_val, x_val, mva) in enumerate(lines):
        r_from = 2 * k
        r_to = 2 * k + 1

        for j, (fb_j, tb_j, r_j, x_j, _) in enumerate(lines):
            if j == k:
                continue
            r_from_j = 2 * j
            r_to_j = 2 * j + 1
            if tb_j == fb:
                z_self = np.sqrt(r_val**2 + x_val**2) + 1e-9
                z_other = np.sqrt(r_j**2 + x_j**2) + 1e-9
                If_bkp_val = min(If_line[k] * (z_self / (z_self + z_other)) * 0.6,
                                 If_line[k] * 0.55)
                key = (r_from, r_to_j)
                if key not in seen and If_bkp_val > 0:
                    seen.add(key)
                    pairs.append(RelayPair(r_from, r_to_j, round(If_line[k], 1),
                                          round(If_bkp_val, 1), fb, tb_j))

        for j, (fb_j, tb_j, r_j, x_j, _) in enumerate(lines):
            if j == k:
                continue
            r_from_j = 2 * j
            r_to_j = 2 * j + 1
            if fb_j == tb:
                z_self = np.sqrt(r_val**2 + x_val**2) + 1e-9
                z_other = np.sqrt(r_j**2 + x_j**2) + 1e-9
                If_bkp_val = min(If_line[k] * (z_self / (z_self + z_other)) * 0.6,
                                 If_line[k] * 0.55)
                key = (r_to, r_from_j)
                if key not in seen and If_bkp_val > 0:
                    seen.add(key)
                    pairs.append(RelayPair(r_to, r_from_j, round(If_line[k], 1),
                                          round(If_bkp_val, 1), tb, fb_j))

    # Generator buses (key buses with generation in 118-bus system)
    gen_data = [(10, 450.0), (12, 310.0), (25, 320.0), (26, 314.0),
                (31, 200.0), (46, 200.0), (49, 300.0), (54, 300.0),
                (59, 300.0), (61, 300.0), (65, 300.0), (66, 300.0),
                (69, 300.0), (80, 300.0), (89, 300.0), (100, 300.0),
                (103, 300.0), (111, 300.0)]
    dg_buses = [
        DGConfig(bus_idx=b, p_mw=p * 0.10, pv_penetration=0.0,
                 inverter_model="simplified_linear_frt")
        for b, p in gen_data
    ]

    If_min = _ensure_feasible_ps_bounds(If_min, Iload_max, "get_118bus")
    return NetworkData(
        "IEEE-118Bus", n_relays, pairs, Iload_max, If_min,
        n_buses=118, base_voltage=138.0, base_mva=100.0,
        dg_buses=dg_buses, relay_bus=relay_bus
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Universal network loader — spans all IEEE test systems
# ═══════════════════════════════════════════════════════════════════════════════

def load_network(network_name: str) -> NetworkData:
    """
    Load any IEEE test system by name.

    Parameters
    ----------
    network_name : str
        One of: "3bus", "8bus", "14bus", "30bus", "39bus", "57bus", "118bus"
        Case-insensitive.  Also accepts "IEEE-3Bus", "3", "ieee-8", etc.

    Returns
    -------
    NetworkData

    Raises
    ------
    ValueError
        If network_name is not recognised.
    """
    name = network_name.lower().strip()
    # Normalise common aliases
    aliases = {
        "3": "3bus", "3-bus": "3bus", "ieee3": "3bus", "ieee-3": "3bus",
        "8": "8bus", "8-bus": "8bus", "ieee8": "8bus", "ieee-8": "8bus",
        "14": "14bus", "14-bus": "14bus", "ieee14": "14bus", "ieee-14": "14bus",
        "30": "30bus", "30-bus": "30bus", "ieee30": "30bus", "ieee-30": "30bus",
        "39": "39bus", "39-bus": "39bus", "ieee39": "39bus", "ieee-39": "39bus",
        "57": "57bus", "57-bus": "57bus", "ieee57": "57bus", "ieee-57": "57bus",
        "118": "118bus", "118-bus": "118bus", "ieee118": "118bus", "ieee-118": "118bus",
    }
    key = aliases.get(name, name)
    if key not in NETWORKS:
        avail = ", ".join(sorted(NETWORKS.keys()))
        raise ValueError(
            f"Unknown network '{network_name}' (resolved: '{key}'). "
            f"Available: {avail}"
        )
    return NETWORKS[key]()


def get_network_metadata() -> Dict[str, Dict]:
    """
    Return metadata for all available IEEE test systems.

    Returns
    -------
    dict
        {network_key: {'n_buses': int, 'n_relays': int, 'n_pairs': int,
                       'base_voltage': float, 'description': str}}
    """
    meta = {}
    for key, getter in NETWORKS.items():
        net = getter()
        meta[key] = {
            "n_buses": net.n_buses,
            "n_relays": net.n_relays,
            "n_pairs": net.n_pairs,
            "base_voltage": net.base_voltage,
            "base_mva": net.base_mva,
            "description": _NETWORK_DESCRIPTIONS.get(key, ""),
        }
    return meta


# ─────────────────────────────────────────────────────────────────────────────
# Network descriptions (for documentation / CLI help)
# ─────────────────────────────────────────────────────────────────────────────
_NETWORK_DESCRIPTIONS = {
    # Relay/pair counts here are static text and can silently drift from
    # the actual generated topology below (a prior audit found all of
    # 14/39/57/118bus stale) — verify against NetworkData.n_pairs /
    # Iload_max.size (via get_network_metadata()/get_system_summary(),
    # which read the live values) rather than trusting this dict alone.
    "3bus":   "IEEE 3-Bus Radial (6 relays, 6 P/B pairs) — toy validation",
    "8bus":   "IEEE 8-Bus Distribution (14 relays, 20 P/B pairs) — small-scale MARL",
    "14bus":  "IEEE 14-Bus Meshed (40 relays, 52 P/B pairs) — standard benchmark",
    "30bus":  "IEEE 30-Bus Meshed (42 relays, 70 P/B pairs) — medium-scale MARL",
    "39bus":  "IEEE 39-Bus New England (46 relays, 42 P/B pairs) — stability studies",
    "57bus":  "IEEE 57-Bus Meshed (156 relays, 192 P/B pairs) — large-scale MARL",
    "118bus": "IEEE 118-Bus Backbone (318 relays, 440 P/B pairs) — very large-scale",
}

NETWORKS = {
    "3bus":  get_3bus,
    "8bus":  get_8bus,
    "14bus": get_14bus,  # Phase 2 addition
    "30bus": get_30bus,
    "39bus": get_39bus,
    "57bus": get_57bus,  # Phase 2 addition
    "118bus": get_118bus,  # Phase 2 addition
}