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

        # Add external grid (slack bus). Defaults to bus 0 (1-indexed bus
        # 1) for systems without a real designated reference bus; systems
        # with real line_data can override via "slack_bus"/"slack_vm_pu"
        # (1-indexed bus number, per-unit voltage setpoint). This mattered
        # for 39bus specifically: MATPOWER case39's mpc.bus marks bus 31
        # (not bus 1) as the type=3 reference bus, at Vg=0.982 pu (its
        # gen row's Vg) -- always placing the slack at bus 0 regardless
        # put pandapower's idealized infinite-source slack on a bus that,
        # in the real system, is just an ordinary PQ load bus, and left
        # the ACTUAL reference bus (31) to be modeled as if it were an
        # ordinary generator -- part of why power flow wouldn't converge
        # independent of every other fix above.
        #
        # s_sc_max_mva/rx_max are required by pandapower's short-circuit
        # module (sc.calc_sc) for both case="max" and case="min" studies
        # -- without them, any SC study (Fig 44's fault-contribution
        # analysis, run_short_circuit() below) fails with "short circuit
        # apparent power s_sc_max_mva needs to be specified for external
        # grid". 1000 MVA (10x this system's base_mva=100.0) is a typical
        # utility-substation short-circuit strength for a distribution-
        # class system, matching the value already used for case="min" in
        # the unwired experiments/exp_fig44_fault_contribution.py.
        slack_bus0 = (self.net_data.line_data.get("slack_bus", 1) - 1
                      if self.net_data.line_data else 0)
        slack_vm_pu = (self.net_data.line_data.get("slack_vm_pu", 1.0)
                        if self.net_data.line_data else 1.0)
        pp.create_ext_grid(net, bus=bus_map[slack_bus0], vm_pu=slack_vm_pu, va_degree=0.0,
                            s_sc_max_mva=1000.0, rx_max=0.1,
                            s_sc_min_mva=1000.0, rx_min=0.1)

        n = self.net_data.n_buses
        if self.net_data.line_data and self.net_data.line_data.get("lines"):
            # Real per-line topology AND impedance, sourced from this
            # system's own standard CDF/MATPOWER line list (see
            # get_14bus()/get_39bus() -- the same data those functions
            # already use for fault-current calculations, just not
            # previously exposed to pandapower). Replaces two separate
            # problems the branch below has: (1) a fake ring topology for
            # every "junction-bus convention" network (bus_pri==bus_bkp
            # for every RelayPair, so no real from/to edge is recoverable
            # from pairs alone), and (2) an arbitrary cable std_type
            # picked by a voltage threshold regardless of the system's
            # actual base voltage -- e.g. a 12-20kV-rated cable modeling
            # 39-bus's 345kV transmission lines, and a 0.4kV-rated cable
            # modeling 8-bus's 11kV lines. Both produced power flow that
            # either failed to converge (14bus) or "converged" with
            # 1000%+ line loading (8bus, 39bus) -- an artifact of the
            # placeholder line model, not the actual network.
            V_base_kv = self.net_data.base_voltage
            S_base_mva = self.net_data.base_mva
            z_base_ohm = V_base_kv ** 2 / S_base_mva
            y_base_s = 1.0 / z_base_ohm
            f_hz = 50.0
            for ln in self.net_data.line_data["lines"]:
                from_bus0 = ln["from_bus"] - 1
                to_bus0 = ln["to_bus"] - 1
                if not (0 <= from_bus0 < n and 0 <= to_bus0 < n):
                    continue
                ratio = ln.get("ratio", 0.0)
                if ratio:
                    # A nonzero MATPOWER `ratio` (including exactly 1.0,
                    # a unity-ratio isolation transformer) marks a
                    # transformer branch, not a plain line. These have
                    # near-zero series resistance (r_pu often exactly 0)
                    # -- pandapower's Newton-Raphson solver is
                    # numerically ill-conditioned by that when modeled
                    # as create_line_from_parameters (confirmed via
                    # pp.diagnostic(): "impedance values close to zero",
                    # non-convergence persisting until reindexed/rescaled
                    # away). create_transformer_from_parameters handles
                    # a small series impedance robustly since that's the
                    # normal case for a real transformer.
                    #
                    # sn_mva/vk_percent/vkr_percent are on the
                    # transformer's OWN rated power, not the system's
                    # S_base_mva these r_pu/x_pu values are expressed on
                    # -- rescale by (sn_mva / S_base_mva). Cross-checked
                    # directly against pandapower's own built-in
                    # pandapower.networks.case39() reference conversion
                    # of this exact standard system: it independently
                    # produces the identical vk_percent (e.g. 16.29% for
                    # the bus2-bus30 transformer, 45.0% for bus6-bus31)
                    # using this same formula, and that reference network
                    # converges without issue -- pandapower's
                    # diagnostic() flagging vk_percent>20% as an
                    # "invalid_values" candidate is only an advisory
                    # plausibility heuristic, not a hard error; a version
                    # of this fix that instead set sn_mva=S_base_mva to
                    # silence that warning was actually wrong, not a fix.
                    # The off-nominal tap ratio is baked directly into
                    # vn_hv_kv (both sides otherwise share this system's
                    # single nominal voltage in this codebase's model,
                    # so there's no genuine HV/LV voltage-class split to
                    # reflect here beyond the tap itself). No magnetizing
                    # branch is modeled (pfe_kw=i0_percent=0), matching
                    # b_pu=0 on every transformer branch in the source
                    # data -- these are series-impedance-only in
                    # MATPOWER's own model too.
                    sn_mva = ln["mva_limit"]
                    base_ratio = sn_mva / S_base_mva
                    vk_percent = max(
                        np.hypot(ln["r_pu"], ln["x_pu"]) * base_ratio * 100.0, 0.01
                    )
                    vkr_percent = ln["r_pu"] * base_ratio * 100.0
                    pp.create_transformer_from_parameters(
                        net,
                        hv_bus=bus_map[from_bus0],
                        lv_bus=bus_map[to_bus0],
                        sn_mva=sn_mva,
                        vn_hv_kv=V_base_kv * ratio,
                        vn_lv_kv=V_base_kv,
                        vkr_percent=vkr_percent,
                        vk_percent=vk_percent,
                        pfe_kw=0.0,
                        i0_percent=0.0,
                        name=f"Trafo_{ln['from_bus']}_{ln['to_bus']}",
                    )
                    continue
                r_ohm = ln["r_pu"] * z_base_ohm
                x_ohm = ln["x_pu"] * z_base_ohm
                b_s = ln.get("b_pu", 0.0) * y_base_s
                c_nf = max(b_s / (2 * np.pi * f_hz) * 1e9, 0.0)
                max_i_ka = ln["mva_limit"] / (np.sqrt(3) * V_base_kv)
                pp.create_line_from_parameters(
                    net,
                    from_bus=bus_map[from_bus0],
                    to_bus=bus_map[to_bus0],
                    length_km=1.0,
                    r_ohm_per_km=max(r_ohm, 1e-6),  # pandapower rejects r=0
                    x_ohm_per_km=x_ohm,
                    c_nf_per_km=c_nf,
                    max_i_ka=max_i_ka,
                    name=f"Line_{ln['from_bus']}_{ln['to_bus']}",
                )
        else:
            # Derive the pandapower line topology from this network's own
            # relay-pair bus data (RelayPair.bus_pri/bus_bkp, 1-indexed)
            # instead of a topology guess keyed only on bus count --
            # previously this was a hardcoded stand-in completely
            # decoupled from the actual relay network topology used
            # everywhere else (network/network_data.py, marl/comm_graph.py,
            # etc.), so pandapower's power flow / fault study described a
            # different graph than the one relays actually coordinate on.
            # This path has no real per-line impedance data available
            # (net_data.line_data is None for these systems), so it still
            # falls back to a generic cable std_type -- kept for systems
            # without a line_data source (8bus/30bus/57bus/118bus); its
            # power-flow *loading magnitudes* aren't calibrated against a
            # real line rating and shouldn't be trusted quantitatively,
            # only the topology/direction structure RelayPair data gives.
            edges = set()
            for p in self.net_data.pairs:
                b1, b2 = p.bus_pri - 1, p.bus_bkp - 1
                if b1 != b2 and 0 <= b1 < n and 0 <= b2 < n:
                    edges.add(tuple(sorted((b1, b2))))
            line_pairs = sorted(edges)
            if not line_pairs:
                # Junction-bus convention networks (bus_pri == bus_bkp for
                # every pair, e.g. get_57bus()/get_118bus() -- get_14bus()/
                # get_39bus() now have real line_data above and never reach
                # here) don't encode a from/to edge per pair at all, so
                # there is no real topology to recover from RelayPair data.
                # Fall back to a simple ring so every bus still connects to
                # something; an intentionally simplified stand-in pending
                # real per-line bus endpoint data for these systems too.
                line_pairs = [(i, (i+1) % n) for i in range(n)]

            for from_bus, to_bus in line_pairs:
                if from_bus < self.net_data.n_buses and to_bus < self.net_data.n_buses:
                    pp.create_line(
                        net,
                        from_bus=bus_map[from_bus],
                        to_bus=bus_map[to_bus],
                        length_km=1.0,
                        std_type="NAYY 4x150 SE" if self.net_data.base_voltage <= 20 else "NA2XS2Y 1x185 RM/25 12/20 kV"
                    )

        if self.net_data.line_data and self.net_data.line_data.get("loads"):
            # Real per-bus load (standard CDF/MATPOWER case data, same
            # source as line_data["lines"] above). Iload_max is a
            # per-relay CT/protection pickup rating, not a bus's real
            # power demand -- summing it into per-bus MW below produced
            # system-wide loads several times the real system's actual
            # demand (see get_14bus()/get_39bus()'s comments), which
            # alone was enough to prevent power flow from converging
            # regardless of how correct the line model was.
            for ld in self.net_data.line_data["loads"]:
                bus0 = ld["bus"] - 1
                if 0 <= bus0 < self.net_data.n_buses and (ld["p_mw"] or ld["q_mvar"]):
                    pp.create_load(
                        net,
                        bus=bus_map[bus0],
                        p_mw=ld["p_mw"],
                        q_mvar=ld["q_mvar"],
                        name=f"Load_Bus{ld['bus']}",
                    )
        else:
            # No real per-bus load data available for this system --
            # fall back to the Iload_max-derived heuristic. Not
            # calibrated against real system demand (see the line-model
            # else-branch's identical caveat above); kept for systems
            # without a line_data source.
            #
            # get_relay_bus(i) is a legitimate many-to-one mapping -- any
            # bus with more than one incident line has more than one
            # relay on it. Iload_max[i] rates that specific relay's own
            # outgoing branch's protection pickup, so several relays at
            # one bus are independent measurements of the SAME bus's
            # approximate load-current scale, not additive independent
            # loads. Previously this loop called pp.create_load() once
            # per relay, so pandapower's per-bus aggregation summed those
            # measurements and multiplied a bus's load by however many
            # relays/lines happened to converge there (verified: 3 of
            # 8bus's 8 buses were triple-counted this way, producing
            # 17.99/20.58/22.86 MW there vs 4.27-9.14 MW at single-relay
            # buses, which alone was enough to push max line loading to
            # 865-886%). Aggregate by bus first -- using max() across a
            # bus's relays as the defensible reading of "this bus's
            # approximate load current, per whichever relay's rating is
            # largest" -- so exactly one load row is created per bus.
            bus_p_mw: dict = {}
            for i in range(self.net_data.n_relays):
                # Real bus this relay is located at (from its RelayPair's
                # bus_pri/bus_bkp) — the previous placeholder
                # min(i, n_buses-1) ignored real topology, dumping
                # multiple relays' loads onto whichever bus happened to
                # match their index.
                bus_idx = self.net_data.get_relay_bus(i)
                # P(MW) = sqrt(3) * V_LL(kV) * I(A) / 1000 for 3-phase real
                # power (V_LL(kV)*1000 = volts, so P(W) =
                # sqrt(3)*V_LL(kV)*1000*I(A), and P(MW) = that / 1e6 =
                # sqrt(3)*V_LL(kV)*I(A) / 1e3). The previous "/ 1e6"
                # divisor left the kV->V conversion undone, undersizing
                # every load by 1000x.
                p_mw = self.net_data.Iload_max[i] * self.net_data.base_voltage * 1.732 / 1e3  # 3-phase
                if 0 <= bus_idx < self.net_data.n_buses:
                    bus_p_mw[bus_idx] = max(bus_p_mw.get(bus_idx, 0.0), p_mw)

            for bus_idx, p_mw in bus_p_mw.items():
                p_load = p_mw * 0.8  # 0.8 pf
                q_load = p_mw * 0.6
                if not (p_load or q_load):
                    # Mirrors the real-load-data branch above, which skips
                    # rows whose p_mw/q_mvar both round to zero.
                    continue
                pp.create_load(
                    net,
                    bus=bus_map[bus_idx],
                    p_mw=p_load,
                    q_mvar=q_load,
                    name=f"Load_Bus{bus_idx + 1}",
                )

        # Add DG/PV generators. Assumes dg.p_mw is the DG's full RATED
        # capacity (dg.pv_penetration==0.0, the NetworkData constructors'
        # default) -- NOT a NetworkData already scaled by
        # NetworkData.apply_pv_penetration(level), which bakes `level`
        # into dg.p_mw AND sets dg.pv_penetration=level on the same
        # object. Passing a pre-scaled NetworkData here (or into
        # update_pv_penetration() below) would double-apply the scaling
        # (p_mw * level again on top of the already-scaled p_mw).
        # Currently dormant: every DistributionNetwork(use_pandapower=True)
        # call site in this repo passes the raw, unscaled NetworkData and
        # scales via update_pv_penetration()/reset(pv_level=...) instead.
        for dg in self.net_data.dg_buses:
            # dg.bus_idx is 1-indexed (same convention as RelayPair.bus_pri/
            # bus_bkp), but bus_map is keyed 0-indexed (built via
            # `for i in range(n_buses)` above, same as the load-placement
            # loop's own `bus_idx = get_relay_bus(i)` two loops up). Indexing
            # bus_map directly with dg.bus_idx placed every sgen one bus
            # downstream of the real DG, and silently dropped a DG at the
            # network's highest-numbered (1-indexed) bus entirely, since
            # `dg.bus_idx < n_buses` is False exactly when dg.bus_idx==n_buses.
            dg_bus0 = dg.bus_idx - 1
            if 0 <= dg_bus0 < self.net_data.n_buses:
                # sn_mva (rated apparent power) is required by
                # pandapower's short-circuit module (sc.calc_sc) for every
                # sgen, same category of gap as the ext_grid's
                # s_sc_max_mva above -- without it, Fig 44's fault-
                # contribution study fails with "sn_mva needs to be
                # specified for all sgens in net.sgen.sn_mva". Use the
                # DG's rated (unscaled) capacity, not its current
                # pv_penetration-scaled output -- a generator's nameplate
                # rating doesn't shrink just because it's dispatched below
                # full output.
                # always_on generators (e.g. 39-bus's 10 real synchronous
                # machines) already have their dispatched output baked
                # into dg.p_mw itself (rated * a fixed loading factor, see
                # get_39bus()) -- they don't scale with pv_penetration the
                # way genuine distributed-PV entries do.
                output_p_mw = dg.p_mw if dg.always_on else dg.p_mw * dg.pv_penetration
                output_q_mvar = dg.q_mvar if dg.always_on else dg.q_mvar * dg.pv_penetration
                if dg.always_on and dg.vm_pu is not None:
                    # Real generators regulate local voltage via reactive
                    # power (a voltage-controlled PV bus in power-flow
                    # terms), not a fixed PQ injection -- confirmed by
                    # direct comparison against pandapower's own built-in
                    # pandapower.networks.case39() reference, which
                    # models these same 9 buses with create_gen (not
                    # create_sgen) and converges; an all-PQ version of
                    # the same data does not. Distributed-PV entries
                    # below (always_on=False) keep the PQ/sgen model,
                    # matching how real inverter-based PV lacking
                    # dedicated voltage control is actually operated.
                    pp.create_gen(
                        net,
                        bus=bus_map[dg_bus0],
                        p_mw=output_p_mw,
                        vm_pu=dg.vm_pu,
                        sn_mva=float(np.hypot(dg.p_mw, dg.q_mvar)) or 1.0,
                        name=f"Gen_Bus{dg.bus_idx}",
                    )
                    continue
                pp.create_sgen(
                    net,
                    bus=bus_map[dg_bus0],
                    p_mw=output_p_mw,
                    q_mvar=output_q_mvar,
                    sn_mva=float(np.hypot(dg.p_mw, dg.q_mvar)) or 1.0,
                    name=f"PV_Bus{dg.bus_idx}",
                    type="PV",
                    # Short-circuit-specific fields the sc.calc_sc()
                    # module needs for any sgen: modeled as a full-size
                    # converter (IEC 60909-0:2016 "current_source" type,
                    # the standard model for grid-following PV
                    # inverters), with k=1.0 (short-circuit current
                    # limited to rated current -- the conservative,
                    # standard assumption for inverter-based generation
                    # without manufacturer-specific overcurrent data).
                    generator_type="current_source",
                    k=1.0,
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

        # If_min is indexed by RELAY (len == n_relays), not by bus, so it
        # can't be read directly with the bus loop index `i` below (n_buses
        # != n_relays for every supported network) -- that mismatch used to
        # silently pull the wrong relay's bound for every bus. Look up the
        # relays actually located at bus `i` via RelayPair.bus_pri/bus_bkp
        # (1-indexed, hence the `bus_no = i + 1`) and use their fault
        # currents instead.
        default_ik = float(np.min(self.net_data.If_min)) if len(self.net_data.If_min) > 0 else 5000.0
        for i in range(self.net_data.n_buses):
            bus_no = i + 1
            relay_currents = (
                [p.If_pri for p in self.net_data.pairs if p.bus_pri == bus_no] +
                [p.If_bkp for p in self.net_data.pairs if p.bus_bkp == bus_no]
            )
            base_ik = min(relay_currents) if relay_currents else default_ik
            fc[f"bus_{i}"] = base_ik * ft_scale

        return fc

    def _apply_topology_changes(self, changes: Dict):
        """Apply N-1 contingencies, line outages, bus splits, and outage factors."""
        if not PANDAPOWER_AVAILABLE or self.pp_net is None:
            return

        # Line outages
        if "line_outages" in changes:
            for line_idx in changes["line_outages"]:
                if isinstance(line_idx, int) and 0 <= line_idx < len(self.pp_net.line):
                    self.pp_net.line.at[line_idx, "in_service"] = False
                elif isinstance(line_idx, str) and line_idx.startswith("L"):
                    # Handle "L1", "L2" format
                    try:
                        idx = int(line_idx[1:]) - 1
                        if 0 <= idx < len(self.pp_net.line):
                            self.pp_net.line.at[idx, "in_service"] = False
                    except ValueError:
                        pass

        # Bus splits (open switches)
        if "bus_splits" in changes:
            for bus_idx in changes["bus_splits"]:
                if bus_idx in self.bus_map:
                    pp_bus = self.bus_map[bus_idx]
                    for line_idx in range(len(self.pp_net.line)):
                        if (self.pp_net.line.at[line_idx, "from_bus"] == pp_bus or
                            self.pp_net.line.at[line_idx, "to_bus"] == pp_bus):
                            self.pp_net.line.at[line_idx, "in_service"] = False

        # Outage factor (scale all line impedances)
        if "outage_factor" in changes:
            factor = changes["outage_factor"]
            if isinstance(factor, (int, float)) and factor > 0:
                for line_idx in range(len(self.pp_net.line)):
                    if self.pp_net.line.at[line_idx, "in_service"]:
                        self.pp_net.line.at[line_idx, "r_ohm_per_km"] *= factor
                        self.pp_net.line.at[line_idx, "x_ohm_per_km"] *= factor

    def update_pv_penetration(self, pv_level: float):
        """Update PV generation scaling for all genuine distributed-PV DG
        buses. always_on generators (e.g. 39-bus's real synchronous
        machines) are skipped -- they dispatch at a fixed level
        independent of pv_level (see DGConfig.always_on)."""
        if not PANDAPOWER_AVAILABLE or self.pp_net is None:
            return

        for dg in self.net_data.dg_buses:
            if dg.always_on:
                continue
            # Find sgen corresponding to this DG bus. Exact match on the
            # "PV_Bus{n}" name, not substring containment -- `"PV_Bus3" in
            # "PV_Bus39"` was previously True, so a system with both a bus
            # 3 and a bus 39 DG would have silently cross-applied one
            # bus's pv_level scaling to the other's sgen.
            for sgen_idx in range(len(self.pp_net.sgen)):
                if str(self.pp_net.sgen.at[sgen_idx, "name"]) == f"PV_Bus{dg.bus_idx}":
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

    def get_line_flows(self) -> Dict[int, float]:
        """Return real power flow (MW) leaving the 'from' bus of each
        in-service line, after run_power_flow(). Sign follows
        pandapower's convention: positive means flow from `from_bus`
        toward `to_bus`; a sign flip relative to a 0%-PV baseline at the
        same line indicates the line's real power direction has
        reversed (e.g. a downstream PV injection now exceeds local load
        and pushes power back toward the substation)."""
        if not PANDAPOWER_AVAILABLE or self.pp_net is None:
            return {}

        flows = {}
        for line_idx in range(len(self.pp_net.line)):
            if self.pp_net.line.at[line_idx, "in_service"]:
                flows[line_idx] = float(self.pp_net.res_line.at[line_idx, "p_from_mw"])
        return flows


# Supported network sizes for validation
SUPPORTED_NETWORK_SIZES = {3, 8, 14, 30, 39, 57, 118}


def build_pandapower_network(net_data: NetworkData) -> PandapowerNetwork:
    """Factory function to create PandapowerNetwork from NetworkData."""
    if net_data.n_buses not in SUPPORTED_NETWORK_SIZES:
        raise ValueError(
            f"Network size {net_data.n_buses} not supported. "
            f"Supported: {sorted(SUPPORTED_NETWORK_SIZES)}"
        )
    return PandapowerNetwork(net_data)


def get_supported_networks() -> Dict[str, int]:
    """Return mapping of network names to bus counts."""
    return {
        "3bus": 3, "8bus": 8, "14bus": 14, "30bus": 30,
        "39bus": 39, "57bus": 57, "118bus": 118,
    }