"""
ga_sqp.py
─────────
Hybrid GA → SQP optimizer replicating Al-Bhadely & İnan (Energies 2023).

Phase 1 Enhancement: Supports dynamic fault currents, multiple fault types,
PV penetration levels, and topology-aware constraint checking.

Phase 1 — Genetic Algorithm (global exploration)
  • real-valued chromosome [TMS_1..N, PS_1..N]
  • tournament selection, uniform crossover, Gaussian mutation
  • penalty-augmented objective (Eq. 9-11)

Phase 2 — Sequential Quadratic Programming (local refinement)
  • scipy.optimize.minimize(method='SLSQP')
  • warm-started from GA best individual
  • explicit CTI inequality constraints (≥ 0.3s)
"""

import numpy as np
from scipy.optimize import minimize, Bounds
import time

from network.network_data import NetworkData
from network.relay_physics import (
    objective_function, check_constraints,
    compute_operating_times,
    TMS_MIN, TMS_MAX, CTI,
)


def pack(TMS, PS):       return np.concatenate([TMS, PS])
def unpack(x, n):        return x[:n], x[n:]


def fitness(x, net, alpha1=1.0, alpha2=100.0, beta=50.0, 
            fault_type="3ph", topology_state=None):
    """Phase 1: Enhanced fitness with fault type and topology support."""
    TMS, PS = unpack(x, net.n_relays)
    OF, _, _ = objective_function(net, TMS, PS, alpha1, alpha2, beta,
                                   fault_type=fault_type, topology_state=topology_state)
    return OF


# ── Genetic Algorithm ─────────────────────────────────────────────────────────
def run_ga(
    net,
    pop_size = 100,
    n_gen    = 200,
    p_cross  = 0.8,
    p_mut    = 0.05,
    tourn_k  = 3,
    seed     = 42,
    alpha1   = 1.0,
    alpha2   = 100.0,
    beta     = 50.0,
    verbose  = False,
    fault_type = "3ph",         # Phase 1: fault type
    topology_state = None,      # Phase 1: topology state
    pv_level = 0.0,             # Phase 1: PV penetration (for logging)
):
    rng = np.random.default_rng(seed)
    n   = net.n_relays
    ps_min, ps_max = net.ps_bounds()

    lo = np.concatenate([np.full(n, TMS_MIN), ps_min])
    hi = np.concatenate([np.full(n, TMS_MAX), ps_max])
    hi = np.maximum(hi, lo + 1e-6)

    pop   = rng.uniform(0, 1, (pop_size, 2*n)) * (hi - lo) + lo
    f_pop = np.array([fitness(ind, net, alpha1, alpha2, beta, fault_type, topology_state) 
                      for ind in pop])

    best_idx = int(np.argmin(f_pop))
    best_ind = pop[best_idx].copy()
    best_fit = float(f_pop[best_idx])
    history  = [best_fit]

    for gen in range(n_gen):
        def tournament():
            idx = rng.integers(0, pop_size, tourn_k)
            return pop[idx[np.argmin(f_pop[idx])]].copy()

        new_pop = np.empty_like(pop)
        for i in range(0, pop_size, 2):
            p1, p2 = tournament(), tournament()
            if rng.random() < p_cross:
                mask = rng.random(2*n) < 0.5
                c1   = np.where(mask, p1, p2)
                c2   = np.where(mask, p2, p1)
            else:
                c1, c2 = p1.copy(), p2.copy()
            for c in (c1, c2):
                m = rng.random(2*n) < p_mut
                c[m] += rng.normal(0, 0.05*(hi-lo)[m])
                np.clip(c, lo, hi, out=c)
            new_pop[i] = c1
            if i+1 < pop_size:
                new_pop[i+1] = c2

        new_pop[0] = best_ind          # elitism
        pop   = new_pop
        f_pop = np.array([fitness(ind, net, alpha1, alpha2, beta, fault_type, topology_state) 
                          for ind in pop])

        bi = int(np.argmin(f_pop))
        if f_pop[bi] < best_fit:
            best_fit = float(f_pop[bi])
            best_ind = pop[bi].copy()
        history.append(best_fit)

        if verbose and gen % 50 == 0:
            print(f"  GA gen {gen:4d}  best_OF={best_fit:.4f}  PV={pv_level*100:.0f}%")

    return {"best": best_ind, "best_fit": best_fit, "history": history}


# ── SQP ───────────────────────────────────────────────────────────────────────
def run_sqp(net, x0, alpha1=1.0, alpha2=100.0, beta=50.0, maxiter=1000,
            fault_type="3ph", topology_state=None):
    """Phase 1: Enhanced SQP with fault type and topology support."""
    n = net.n_relays
    ps_min, ps_max = net.ps_bounds()
    lo = np.concatenate([np.full(n, TMS_MIN), ps_min])
    hi = np.concatenate([np.full(n, TMS_MAX), ps_max])
    hi = np.maximum(hi, lo + 1e-6)

    def cti_constraints(x):
        TMS, PS = unpack(x, n)
        t_pri, t_bkp = compute_operating_times(net, TMS, PS, 
                                                fault_type=fault_type, 
                                                topology_state=topology_state)
        return t_bkp - t_pri - CTI       # >= 0

    result = minimize(
        lambda x: fitness(x, net, alpha1, alpha2, beta, fault_type, topology_state),
        x0,
        method="SLSQP",
        bounds=Bounds(lo, hi),
        constraints={"type": "ineq", "fun": cti_constraints},
        options={"maxiter": maxiter, "ftol": 1e-9, "disp": False},
    )
    return {"x": result.x, "OF": result.fun,
            "success": result.success, "message": result.message}


# ── Full pipeline ─────────────────────────────────────────────────────────────
def run_ga_sqp(net, ga_pop=100, ga_gen=200, seed=42, verbose=True,
               fault_type="3ph", topology_state=None, pv_level=0.0):
    """
    Phase 1: Enhanced GA-SQP with dynamic network support.

    Parameters:
        net: NetworkData instance
        ga_pop: GA population size
        ga_gen: GA generations
        seed: Random seed
        verbose: Print progress
        fault_type: Fault type ("3ph", "LL", "1ph", "2ph-g")
        topology_state: Topology state dict for dynamic fault currents
        pv_level: PV penetration level (0.0-1.0) for logging
    """
    t0 = time.perf_counter()
    if verbose:
        print(f"\n{'─'*55}")
        print(f"  GA-SQP  ·  {net.name}  ·  "
              f"{net.n_relays} relays  ·  {net.n_pairs} pairs")
        if pv_level > 0:
            print(f"  PV Penetration: {pv_level*100:.0f}%  ·  Fault: {fault_type}")
        print(f"{'─'*55}")
        print(f"  [Phase 1] Genetic Algorithm  (pop={ga_pop}, gen={ga_gen})")

    ga_res  = run_ga(net, pop_size=ga_pop, n_gen=ga_gen,
                     seed=seed, verbose=verbose,
                     fault_type=fault_type, topology_state=topology_state,
                     pv_level=pv_level)

    if verbose:
        print(f"  GA best OF = {ga_res['best_fit']:.4f} s")
        print(f"  [Phase 2] SQP refinement ...")

    sqp_res        = run_sqp(net, x0=ga_res["best"], 
                             fault_type=fault_type, topology_state=topology_state)
    TMS_opt, PS_opt = unpack(sqp_res["x"], net.n_relays)
    constraints     = check_constraints(net, TMS_opt, PS_opt,
                                        fault_type=fault_type, topology_state=topology_state)
    elapsed         = time.perf_counter() - t0

    if verbose:
        print(f"  SQP OF  = {sqp_res['OF']:.4f} s  "
              f"|  CTI ok: {constraints['cti_satisfied']}")
        print(f"  Elapsed : {elapsed:.1f} s")

    return {
        "method":      "GA-SQP",
        "network":     net.name,
        "TMS":         TMS_opt,
        "PS":          PS_opt,
        "OF":          sqp_res["OF"],
        "OF_ga":       ga_res["best_fit"],
        "ga_history":  ga_res["history"],
        "constraints": constraints,
        "elapsed":     elapsed,
        "sqp_success": sqp_res["success"],
        "fault_type":  fault_type,
        "pv_level":    pv_level,
    }
