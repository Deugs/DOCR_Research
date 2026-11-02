"""
tests/test_vectorized_power_flow.py
------------------------------------
Verification for network/vectorized_power_flow.py's batched Newton-Raphson
AC power flow solver -- Phase V-PF of the GPU-vectorized-environment plan.

This is the highest-stakes correctness gate in that whole plan: unlike
every other phase (which verifies new torch code against this project's
own prior NumPy implementation), this module is brand-new physics with no
prior version to diff against -- its bar is direct numerical agreement
with real pandapower.runpp() output, reusing the same reference-comparison
pattern as test_pandapower_model.py::test_39bus_matches_pandapower_reference.

Key findings from building this test (documented here, not just in commit
history, since they explain non-obvious assertions below):
- Index correspondence pitfall: a line_data["lines"] index does NOT match
  pandapower's internal net.line DataFrame row index once transformers are
  present (transformers go into the separate net.trafo table, shifting
  every line index after them) -- outage_to_pandapower_line_idx() below
  exists specifically to avoid silently mis-comparing against the wrong
  physical element, a mistake made (and caught) while writing this test.
- A random single-branch outage can genuinely ISLAND a bus (no remaining
  path to the rest of the network) -- mathematically singular for any
  power-flow method, not a bug. Pandapower's own response is to silently
  drop the bus and report NaN while still claiming converged=True; this
  solver's `damping` parameter avoids a hard crash on the same case without
  claiming the isolated bus's resulting voltage is meaningful.
- Non-convergence on extreme random (load, PV, topology) combinations is
  expected behavior for ANY power-flow method -- pandapower itself fails
  to converge on a real fraction of the randomized scenarios tested here.
  The `converged` flag's job is catching that reliably (zero false
  negatives -- claiming converged when the answer doesn't match reality --
  is the only unacceptable failure mode; occasional false positives, being
  overly conservative, are safe).
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
import torch

from network.network_data import get_8bus, get_14bus, get_39bus
from network.pandapower_network import build_pandapower_network, PANDAPOWER_AVAILABLE
from network.vectorized_power_flow import VectorizedPowerFlowSolver

pytestmark = pytest.mark.skipif(not PANDAPOWER_AVAILABLE, reason="pandapower not installed")

SYSTEMS = [("8bus", get_8bus), ("14bus", get_14bus), ("39bus", get_39bus)]


def outage_to_pandapower_line_idx(net, line_data_idx):
    """Map an index into net.line_data['lines'] to the corresponding row in
    pandapower's internal net.line DataFrame, which excludes transformer
    entries (they go into net.trafo instead, built in the same iteration
    order -- see pandapower_network.py's _build_network()). Returns None if
    line_data_idx refers to a transformer, which callers should handle via
    net.trafo instead of net.line."""
    lines = net.line_data["lines"]
    pp_line_idx = 0
    for i, ln in enumerate(lines):
        is_trafo = bool(ln.get("ratio", 0.0))
        if i == line_data_idx:
            return None if is_trafo else pp_line_idx
        if not is_trafo:
            pp_line_idx += 1
    return None


def _random_scenario(rng, n_lines):
    pv_level = float(rng.uniform(0, 1))
    n_out = int(rng.integers(0, 2))
    outages = rng.choice(n_lines, size=n_out, replace=False).tolist() if n_out > 0 else []
    return pv_level, outages


@pytest.mark.parametrize("name,get_net", SYSTEMS)
def test_matches_pandapower_base_case(name, get_net):
    """No outages, several PV levels -- the simplest case, must match
    pandapower's own bus voltages essentially to machine precision."""
    net = get_net()
    solver = VectorizedPowerFlowSolver(net, n_envs=1, device="cpu", max_iter=30)
    for pv_level in [0.0, 0.3, 0.6, 1.0]:
        pp_net = build_pandapower_network(net)
        pp_net.update_pv_penetration(pv_level)
        assert pp_net.run_power_flow(), f"{name} pv={pv_level}: pandapower itself failed to converge"
        real_vm = pp_net.get_bus_voltages()
        real_vm_arr = np.array([real_vm[i] for i in range(net.n_buses)])

        line_in_service = torch.ones((1, solver.n_lines), dtype=torch.bool)
        vm, va, conv = solver.solve(line_in_service, torch.tensor([pv_level], dtype=torch.float64))
        assert bool(conv[0]), f"{name} pv={pv_level}: solver reported non-convergence on the base case"
        diff = np.max(np.abs(real_vm_arr - vm[0].numpy()))
        assert diff < 1e-6, f"{name} pv={pv_level}: max bus-voltage diff {diff:.2e} vs pandapower"


@pytest.mark.parametrize("name,get_net", SYSTEMS)
def test_matches_pandapower_randomized_scenarios(name, get_net):
    """Randomized (PV level, single-line-or-transformer outage) scenarios,
    including N-1 contingency and (for 39-bus) PV-bus cases. Zero false
    negatives required (solver claims converged but answer is wrong);
    occasional non-convergence -- on either side -- is tolerated and
    expected, matching real pandapower's own behavior on sufficiently
    adversarial random inputs."""
    net = get_net()
    solver = VectorizedPowerFlowSolver(net, n_envs=1, device="cpu", max_iter=30)
    # Fixed per-system seed, NOT hash(name) -- Python randomizes string
    # hashes per-process by default (a security feature, PYTHONHASHSEED),
    # so a hash-derived seed silently picks different random scenarios on
    # every single test run. Caught this directly: this test passed
    # standalone, then failed on a later run with a genuinely different
    # scenario, which hash-based "determinism" made look like flakiness
    # instead of what it actually was -- no fixed seed at all.
    seed = {"8bus": 101, "14bus": 102, "39bus": 103}[name]
    rng = np.random.default_rng(seed)
    n_lines = solver.n_lines

    n_checked = 0
    for _ in range(20):
        pv_level, outages = _random_scenario(rng, n_lines)
        pp_idxs, skip = [], False
        for idx in outages:
            pidx = outage_to_pandapower_line_idx(net, idx)
            if pidx is None:
                skip = True
                break
            pp_idxs.append(pidx)
        if skip:
            continue  # transformer-outage scenarios covered by their own test below

        pp_net = build_pandapower_network(net)
        pp_net.update_pv_penetration(pv_level)
        for pidx in pp_idxs:
            pp_net.pp_net.line.at[pidx, "in_service"] = False
        pp_ok = pp_net.run_power_flow()

        line_in_service = torch.ones((1, n_lines), dtype=torch.bool)
        for idx in outages:
            line_in_service[0, idx] = False
        vm, va, conv = solver.solve(line_in_service, torch.tensor([pv_level], dtype=torch.float64))
        my_ok = bool(conv[0])

        if not pp_ok:
            assert not my_ok, (
                f"{name} pv={pv_level:.3f} outages={outages}: pandapower failed to converge "
                f"but solver reported converged=True -- false negative, the one unacceptable case"
            )
            continue
        if not my_ok:
            continue  # solver being conservative on a case pandapower handled -- acceptable

        # A single line outage can island a bus even when pandapower's
        # overall `converged` flag is True (confirmed: it still marks that
        # specific bus's own voltage NaN rather than failing the whole
        # solve -- same behavior as the dedicated transformer-islanding
        # case in test_islanding_does_not_crash, just triggered here by a
        # random scenario instead of a hand-picked one). Compare only
        # buses pandapower itself reports a finite voltage for, same
        # "trust pandapower's own opinion of what's well-defined" approach
        # used there.
        real_vm = pp_net.get_bus_voltages()
        real_vm_arr = np.array([real_vm.get(i, np.nan) for i in range(net.n_buses)])
        finite = np.isfinite(real_vm_arr)
        if not finite.any():
            continue
        diff = np.max(np.abs(real_vm_arr[finite] - vm[0].numpy()[finite]))
        assert diff < 1e-4, (
            f"{name} pv={pv_level:.3f} outages={outages}: max bus-voltage diff {diff:.2e} vs pandapower"
        )
        n_checked += 1

    assert n_checked >= 10, f"{name}: too few both-converged scenarios checked ({n_checked}/20) to trust this test"


def test_39bus_pv_buses_match_pandapower():
    """39-bus specifically exercises PV-bus handling (9 always-on generator
    buses, voltage-controlled) -- the case this project's own
    pandapower_network.py comments confirm an all-PQ formulation fails to
    converge for. Verifies every PV bus's voltage is held exactly at its
    specified setpoint (by construction) and the solve still matches
    pandapower everywhere else."""
    net = get_39bus()
    solver = VectorizedPowerFlowSolver(net, n_envs=1, device="cpu", max_iter=30)
    assert len(solver.pv_bus0) == 9, f"expected 9 PV buses on 39-bus, found {len(solver.pv_bus0)}"

    pp_net = build_pandapower_network(net)
    pp_net.update_pv_penetration(0.3)
    assert pp_net.run_power_flow()
    real_vm = pp_net.get_bus_voltages()
    real_vm_arr = np.array([real_vm[i] for i in range(net.n_buses)])

    line_in_service = torch.ones((1, solver.n_lines), dtype=torch.bool)
    vm, va, conv = solver.solve(line_in_service, torch.tensor([0.3], dtype=torch.float64))
    assert bool(conv[0])

    for i, bus0 in enumerate(solver.pv_bus0):
        assert abs(vm[0, bus0].item() - solver.pv_vm[i].item()) < 1e-10, (
            f"PV bus {bus0} voltage not held at its setpoint"
        )
    diff = np.max(np.abs(real_vm_arr - vm[0].numpy()))
    assert diff < 1e-6


def test_transformer_outage_matches_pandapower():
    """Outaging a transformer branch (not a plain line) -- a separate code
    path in both pandapower (net.trafo, not net.line) and this solver
    (transformers are just another entry in the shared branch list with a
    non-unity tap, no special-casing needed).

    Uses line_data index 20 specifically, confirmed (via direct check
    against every one of 39-bus's 12 transformers) to be one of only two
    whose outage does NOT island a bus -- most of 39-bus's transformers are
    generator step-up transformers on radial spurs, so outaging them
    islands the generator (see test_islanding_does_not_crash, which
    deliberately exercises that degenerate case instead; the first attempt
    at writing this test picked an islanding transformer by accident, which
    is its own lesson in why this comment exists)."""
    net = get_39bus()
    solver = VectorizedPowerFlowSolver(net, n_envs=1, device="cpu", max_iter=30)
    trafo_line_data_idx = 20
    trafo_net_idx = 3  # net.trafo row index for this same branch (see module comment on index correspondence)
    assert net.line_data["lines"][trafo_line_data_idx].get("ratio", 0.0), "expected index 20 to be a transformer"

    pp_net = build_pandapower_network(net)
    pp_net.update_pv_penetration(0.4)
    pp_net.pp_net.trafo.at[trafo_net_idx, "in_service"] = False
    assert pp_net.run_power_flow()
    real_vm = pp_net.get_bus_voltages()
    assert len(real_vm) == net.n_buses, "outage unexpectedly islanded a bus -- picked the wrong transformer"
    real_vm_arr = np.array([real_vm[i] for i in range(net.n_buses)])

    line_in_service = torch.ones((1, solver.n_lines), dtype=torch.bool)
    line_in_service[0, trafo_line_data_idx] = False
    vm, va, conv = solver.solve(line_in_service, torch.tensor([0.4], dtype=torch.float64))
    assert bool(conv[0])
    diff = np.max(np.abs(real_vm_arr - vm[0].numpy()))
    assert diff < 1e-4


def test_islanding_does_not_crash():
    """A specific real degenerate case found during development: outaging
    39-bus's first transformer genuinely islands its generator bus (bus 29,
    0-indexed) -- pandapower's own response is vm_pu=NaN for that bus while
    still reporting converged=True for the rest. This solver must not raise
    (a hard crash would take down an entire batch of otherwise-healthy
    environments over one degenerate one) -- it's fine for the islanded
    bus's own voltage to be physically meaningless, but every OTHER bus in
    the same solve must still come back sane."""
    net = get_39bus()
    solver = VectorizedPowerFlowSolver(net, n_envs=1, device="cpu", max_iter=30)
    line_in_service = torch.ones((1, solver.n_lines), dtype=torch.bool)
    line_in_service[0, 4] = False  # line_data index 4: the islanding transformer

    vm, va, conv = solver.solve(line_in_service, torch.tensor([0.4], dtype=torch.float64))
    assert torch.isfinite(vm).all(), "islanded-bus case produced non-finite voltages"
    non_islanded = [i for i in range(net.n_buses) if i != 29]
    assert np.all((vm[0, non_islanded].numpy() > 0.5) & (vm[0, non_islanded].numpy() < 1.5)), (
        "a bus NOT involved in the islanding came back with an implausible voltage"
    )


def test_batched_matches_individual_solves():
    """The actual GPU-relevant use case: solving many environments (with
    DIFFERENT topologies/PV levels) in one batched call must match solving
    each individually -- no cross-environment leakage in the batched
    Jacobian/linear-solve path."""
    net = get_39bus()
    rng = np.random.default_rng(99)
    n_envs = 8
    solver_batch = VectorizedPowerFlowSolver(net, n_envs=n_envs, device="cpu", max_iter=30)
    n_lines = solver_batch.n_lines

    line_in_service = torch.ones((n_envs, n_lines), dtype=torch.bool)
    pv_level = torch.as_tensor(rng.uniform(0, 1, n_envs), dtype=torch.float64)
    for e in range(n_envs):
        n_out = int(rng.integers(0, 2))
        if n_out > 0:
            idxs = rng.choice(n_lines, size=n_out, replace=False)
            line_in_service[e, idxs] = False

    vm_batch, va_batch, conv_batch = solver_batch.solve(line_in_service, pv_level)

    solver_single = VectorizedPowerFlowSolver(net, n_envs=1, device="cpu", max_iter=30)
    for e in range(n_envs):
        if not conv_batch[e]:
            continue
        vm_e, va_e, conv_e = solver_single.solve(line_in_service[e:e + 1], pv_level[e:e + 1])
        diff = np.max(np.abs(vm_batch[e].numpy() - vm_e[0].numpy()))
        assert diff < 1e-6, f"env {e}: batched vs individual solve diff {diff:.2e}"


if __name__ == "__main__":
    test_matches_pandapower_base_case("8bus", get_8bus)
    test_matches_pandapower_base_case("14bus", get_14bus)
    test_matches_pandapower_base_case("39bus", get_39bus)
    test_matches_pandapower_randomized_scenarios("8bus", get_8bus)
    test_matches_pandapower_randomized_scenarios("14bus", get_14bus)
    test_matches_pandapower_randomized_scenarios("39bus", get_39bus)
    test_39bus_pv_buses_match_pandapower()
    test_transformer_outage_matches_pandapower()
    test_islanding_does_not_crash()
    test_batched_matches_individual_solves()
    print("All vectorized power flow tests passed.")
