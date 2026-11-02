"""
metaheuristics.py
-----------------
Phase 4: Benchmark metaheuristic algorithms for DOCR coordination.

Implements:
1. HWOA  - Hybrid Whale Optimization Algorithm
2. HHO   - Harris Hawks Optimization
3. JAYA  - JAYA Algorithm (parameter-free)
4. EEO   - Enhanced Equilibrium Optimizer

All algorithms use the same unified objective function and bounds
as GA-SQP for fair comparison.

References:
- Mirjalili & Lewis, "The Whale Optimization Algorithm", Adv Eng Softw 2016
- Heidari et al., "Harris Hawks Optimization", Future Generation Comp Sys 2019
- Rao, "Jaya: A simple and new optimization algorithm", 2016
- Faramarzi et al., "Equilibrium Optimizer", Knowl-Based Syst 2020

2026-10-26: all four algorithms' population loops are now vectorized
(previously a Python `for i in range(pop_size)` loop calling
network.relay_physics.objective_function once per individual per
iteration -- 30 x 500 = 15,000 scalar calls per run). Every per-individual
update formula is unchanged; only the loop structure and fitness
evaluation changed -- see _fitness_batch() and each run_*() function's own
docstring for what was vectorized and why. Fitness evaluation now routes
through relay_physics.objective_function_batch (real torch, device-gated
by the project's DOCR_USE_GPU=1 opt-in policy -- see that module's
docstring) instead of a per-individual Python loop calling the scalar
objective_function. The position-update math itself stays plain NumPy
array ops (cheap regardless of framework, same principle already
established for GA's crossover/mutation in ga_sqp.py). Verified via
statistical equivalence against the pre-vectorization implementation
across 15-20 matched seeds per algorithm (final OF and CTI-feasibility
rate, not bit-identical trajectories -- vectorizing necessarily changes
the RNG draw order, see each function's docstring).
"""

import numpy as np
import scipy.special
from scipy.optimize import minimize, Bounds
from typing import Dict, List, Tuple, Optional, Callable
import time

from network.network_data import NetworkData
from network.relay_physics import (
    objective_function, check_constraints,
    compute_operating_times,
    TMS_MIN, TMS_MAX, CTI,
)


def pack(TMS, PS):       return np.concatenate([TMS, PS])
def unpack(x, n):        return x[:n], x[n:]


def _get_bounds(net: NetworkData) -> Tuple[np.ndarray, np.ndarray]:
    """Return (lo, hi) bounds for [TMS_1..N, PS_1..N]."""
    n = net.n_relays
    ps_min, ps_max = net.ps_bounds()
    lo = np.concatenate([np.full(n, TMS_MIN), ps_min])
    hi = np.concatenate([np.full(n, TMS_MAX), ps_max])
    hi = np.maximum(hi, lo + 1e-6)
    return lo, hi


def _fitness(x: np.ndarray, net: NetworkData,
             alpha1: float = 1.0, alpha2: float = 100.0, beta: float = 50.0,
             fault_type: str = "3ph", topology_state=None) -> float:
    """Unified fitness function for all metaheuristics."""
    TMS, PS = unpack(x, net.n_relays)
    OF, _, _ = objective_function(net, TMS, PS, alpha1, alpha2, beta,
                                   fault_type=fault_type, topology_state=topology_state)
    return OF


def _fitness_batch(pop: np.ndarray, net: NetworkData,
                    alpha1: float = 1.0, alpha2: float = 100.0, beta: float = 50.0,
                    fault_type: str = "3ph", topology_state=None) -> np.ndarray:
    """Vectorized counterpart of _fitness(): evaluates a whole population
    (pop_size, 2*n_relays) at once via relay_physics.objective_function_batch
    instead of looping _fitness() once per individual -- 2026-10-26, part of
    vectorizing HWOA/HHO/JAYA/EEO's population loops (see each run_*
    function's docstring). Falls back to the per-individual loop only when
    topology_state is given (dynamic fault currents aren't supported by the
    batched path), same pattern as baselines/ga_sqp.py's fitness_batch."""
    if topology_state is None:
        from network.relay_physics import objective_function_batch
        n = net.n_relays
        TMS_batch, PS_batch = pop[:, :n], pop[:, n:]
        return objective_function_batch(
            net, TMS_batch, PS_batch, alpha1, alpha2, beta, fault_type=fault_type)
    return np.array([
        _fitness(ind, net, alpha1, alpha2, beta, fault_type, topology_state)
        for ind in pop
    ])


# -- Helper: SQP refinement (shared by all metaheuristics) -----------
def _sqp_refine(net: NetworkData, x0: np.ndarray, 
                fault_type: str = "3ph", topology_state=None,
                alpha1: float = 1.0, alpha2: float = 100.0, beta: float = 50.0,
                maxiter: int = 500) -> Dict:
    """SQP local refinement warm-started from metaheuristic best."""
    n = net.n_relays
    lo, hi = _get_bounds(net)

    def cti_constraints(x):
        TMS, PS = unpack(x, n)
        t_pri, t_bkp = compute_operating_times(net, TMS, PS, 
                                                fault_type=fault_type, 
                                                topology_state=topology_state)
        return t_bkp - t_pri - CTI

    result = minimize(
        lambda x: _fitness(x, net, alpha1, alpha2, beta, fault_type, topology_state),
        x0,
        method="SLSQP",
        bounds=Bounds(lo, hi),
        constraints={"type": "ineq", "fun": cti_constraints},
        options={"maxiter": maxiter, "ftol": 1e-9, "disp": False},
    )
    return {"x": result.x, "OF": result.fun, "success": result.success, "message": result.message}


# ============================================================================
# 1. HWOA -- Hybrid Whale Optimization Algorithm
# ============================================================================

def run_hwoa(
    net: NetworkData,
    n_whales: int = 30,
    max_iter: int = 500,
    seed: int = 42,
    verbose: bool = False,
    fault_type: str = "3ph",
    topology_state=None,
    pv_level: float = 0.0,
    use_sqp: bool = True,
) -> Dict:
    """
    Hybrid Whale Optimization Algorithm (HWOA) for DOCR.
    
    WOA simulates humpback whales' bubble-net hunting strategy.
    Hybrid variant: WOA global exploration -> SQP local refinement.
    
    Parameters:
        n_whales: Population size (whales)
        max_iter: Maximum iterations
        seed: Random seed
        verbose: Print progress
        fault_type: Fault type for evaluation
        topology_state: Topology state dict
        pv_level: PV penetration for logging
        use_sqp: Enable SQP refinement after WOA
    
    Returns:
        Dict with TMS, PS, OF, history, constraints, elapsed time
    """
    rng = np.random.default_rng(seed)
    n = net.n_relays
    lo, hi = _get_bounds(net)
    dim = 2 * n
    
    t0 = time.perf_counter()
    
    # Initialize whale positions
    whales = rng.uniform(0, 1, (n_whales, dim)) * (hi - lo) + lo
    fitness = _fitness_batch(whales, net, fault_type=fault_type, topology_state=topology_state)

    best_idx = int(np.argmin(fitness))
    best_whale = whales[best_idx].copy()
    best_fit = float(fitness[best_idx])
    history = [best_fit]

    for t in range(max_iter):
        a = 2.0 - 2.0 * t / max_iter  # Decreases from 2 to 0
        a2 = -1.0 + t * (-1.0) / max_iter  # Decreases from -1 to -2

        # 2026-10-26: vectorized across the whole population -- same three
        # candidate-position formulas as before (encircling / searching /
        # bubble-net), computed for every whale at once via NumPy array
        # ops, then selected per-whale with np.where based on each whale's
        # own (p, A) draw, instead of a Python `for i in range(n_whales)`
        # loop evaluating one whale's formula and calling _fitness() on it
        # individually. r1/r2/l/p are still drawn once per whale (not once
        # per whale per dimension), matching the original's per-whale
        # scalar semantics -- shape (n_whales, 1) so they broadcast across
        # the dim axis exactly like the scalar did in the original loop.
        # RNG call order necessarily differs from the sequential loop (all
        # whales' r1 drawn together, then all r2, etc., rather than one
        # whale's full draw-then-update before the next) -- not expected
        # to reproduce the old per-seed trajectory bit-for-bit, but every
        # formula and selection rule is unchanged (see module-level
        # verification notes).
        r1 = rng.random((n_whales, 1))
        r2 = rng.random((n_whales, 1))
        l = (a2 - 1) * rng.random((n_whales, 1)) + 1
        p = rng.random((n_whales, 1))
        A = 2 * a * r1 - a
        C = 2 * r2
        b = 1.0

        rand_idx = rng.integers(0, n_whales, n_whales)
        rand_whales = whales[rand_idx]  # (n_whales, dim)

        D_encircle = np.abs(C * best_whale[None, :] - whales)
        cand_encircle = best_whale[None, :] - A * D_encircle

        D_search = np.abs(C * rand_whales - whales)
        cand_search = rand_whales - A * D_search

        D_bubble = np.abs(best_whale[None, :] - whales)
        cand_bubble = D_bubble * np.exp(b * l) * np.cos(2 * np.pi * l) + best_whale[None, :]

        explore = p < 0.5                      # (n_whales, 1)
        encircle_mask = explore & (np.abs(A) < 1)
        search_mask = explore & ~(np.abs(A) < 1)
        # exploit (p >= 0.5) is the remaining case -- the innermost
        # np.where's default branch below.
        new_whales = np.where(
            encircle_mask, cand_encircle,
            np.where(search_mask, cand_search, cand_bubble)
        )

        # Boundary handling: reflect back
        whales = np.clip(new_whales, lo, hi)

        # Evaluate new positions -- one batched call instead of n_whales
        # individual _fitness() calls.
        fitness = _fitness_batch(whales, net, fault_type=fault_type, topology_state=topology_state)

        bi = int(np.argmin(fitness))
        if fitness[bi] < best_fit:
            best_fit = float(fitness[bi])
            best_whale = whales[bi].copy()
        
        history.append(best_fit)
        
        if verbose and t % 50 == 0:
            print(f"  HWOA iter {t:4d}  best_OF={best_fit:.4f}")
    
    # SQP refinement
    if use_sqp:
        if verbose:
            print(f"  HWOA best = {best_fit:.4f}  |  SQP refinement ...")
        sqp_res = _sqp_refine(net, best_whale, fault_type, topology_state)
        best_whale = sqp_res["x"]
        best_fit = sqp_res["OF"]
    
    TMS_opt, PS_opt = unpack(best_whale, n)
    constraints = check_constraints(net, TMS_opt, PS_opt, 
                                    fault_type=fault_type, topology_state=topology_state)
    elapsed = time.perf_counter() - t0
    
    if verbose:
        print(f"  HWOA OF = {best_fit:.4f} s  |  CTI ok: {constraints['cti_satisfied']}")
        print(f"  Elapsed : {elapsed:.1f} s")
    
    return {
        "method": "HWOA",
        "network": net.name,
        "TMS": TMS_opt,
        "PS": PS_opt,
        "OF": best_fit,
        "OF_meta": history[-1] if use_sqp else best_fit,
        "history": history,
        "constraints": constraints,
        "elapsed": elapsed,
        "fault_type": fault_type,
        "pv_level": pv_level,
        "n_evals": n_whales * max_iter,
    }


# ============================================================================
# 2. HHO -- Harris Hawks Optimization
# ============================================================================

def run_hho(
    net: NetworkData,
    n_hawks: int = 30,
    max_iter: int = 500,
    seed: int = 42,
    verbose: bool = False,
    fault_type: str = "3ph",
    topology_state=None,
    pv_level: float = 0.0,
    use_sqp: bool = True,
) -> Dict:
    """
    Harris Hawks Optimization (HHO) for DOCR.
    
    HHO simulates Harris hawks cooperative hunting behavior:
    - Exploration: hawks monitor/search for prey (rabbit)
    - Exploitation: surprise pounce with 4 strategies based on prey energy
    
    Parameters: Same interface as HWOA.
    """
    rng = np.random.default_rng(seed)
    n = net.n_relays
    lo, hi = _get_bounds(net)
    dim = 2 * n
    
    t0 = time.perf_counter()
    
    # Initialize hawk positions
    hawks = rng.uniform(0, 1, (n_hawks, dim)) * (hi - lo) + lo
    fitness = _fitness_batch(hawks, net, fault_type=fault_type, topology_state=topology_state)

    best_idx = int(np.argmin(fitness))
    rabbit = hawks[best_idx].copy()  # Best position (the rabbit)
    rabbit_fit = float(fitness[best_idx])
    history = [rabbit_fit]

    for t in range(max_iter):
        # 2026-10-26: vectorized across the whole population. This is the
        # most complex of the four metaheuristics to vectorize -- 6
        # distinct per-hawk strategies (2 exploration + 4 exploitation),
        # two of which (soft/hard besiege with progressive rapid dives)
        # each need up to 2 EXTRA candidate-position fitness evaluations
        # (Y, Z) before a per-hawk greedy accept/reject decides the hawk's
        # actual new position. Computes every strategy's candidate
        # position for every hawk (cheap NumPy array math, no Python
        # per-hawk loop), batches the handful of resulting fitness
        # evaluations (one call per candidate ARRAY, not per hawk), then
        # selects each hawk's actual outcome via boolean masks matching
        # the original if/elif branch structure exactly.
        #
        # One deliberate correction along the way: the original sequential
        # `for i in range(n_hawks)` loop read `hawks[rand_idx]` (strategy
        # A1) and `hawks.mean(axis=0)` (A2, and the hard-besiege-progressive
        # strategy) from the array WHILE it was being mutated in place --
        # so a hawk late in loop order could see OTHER hawks' positions
        # already updated earlier in that same iteration, an artifact of
        # sequential execution order, not part of Heidari et al. 2019's
        # published algorithm (which defines the population mean X_m(t) as
        # a single snapshot at the start of iteration t, used consistently
        # by every hawk that iteration). A vectorized update has no
        # "earlier in the loop" to begin with, so every hawk's formula here
        # consistently reads the pre-iteration snapshot -- the behavior the
        # paper actually specifies, not the sequential-loop's incidental
        # one. This has no effect on a hawk's reference to its OWN current
        # position (hawks[i]), which was never touched before hawk i's own
        # turn in the original loop either.
        hawks_snap = hawks  # pre-iteration snapshot (not reassigned until the end)
        mean_snap = hawks_snap.mean(axis=0)

        E0 = 2 * rng.random(n_hawks) - 1               # Initial energy, per hawk
        E = 2 * E0 * (1 - t / max_iter)                 # Escaping energy, per hawk
        absE = np.abs(E)
        explore = absE >= 1                              # (n_hawks,)

        # --- Exploration strategy 1: random hawk ---
        rand_idx = rng.integers(0, n_hawks, n_hawks)
        rand_hawks = hawks_snap[rand_idx]                # (n_hawks, dim)
        ra1 = rng.random((n_hawks, 1))
        rb1 = rng.random((n_hawks, 1))
        cand_explore_hawk = rand_hawks - ra1 * np.abs(rand_hawks - 2 * rb1 * hawks_snap)

        # --- Exploration strategy 2: random position in search space ---
        rc = rng.random((n_hawks, 1))
        rd = rng.random((n_hawks, 1))
        cand_explore_pos = (rabbit[None, :] - mean_snap[None, :]) - rc * (
            lo[None, :] + rd * (hi[None, :] - lo[None, :])
        )

        explore_sub = rng.random(n_hawks) < 0.5          # (n_hawks,)
        mask_explore_hawk = explore & explore_sub
        mask_explore_pos = explore & ~explore_sub

        # --- Exploitation branch selection ---
        r_branch = rng.random(n_hawks)
        mask_soft = ~explore & (r_branch < 0.5) & (absE >= 0.5)
        mask_hard = ~explore & (r_branch < 0.5) & (absE < 0.5)
        mask_soft_prog = ~explore & (r_branch >= 0.5) & (absE >= 0.5)
        mask_hard_prog = ~explore & (r_branch >= 0.5) & (absE < 0.5)

        # --- Soft besiege (Heidari et al. 2019, Eq. 6) ---
        J_soft = 2 * (1 - rng.random((n_hawks, 1)))
        cand_soft = (rabbit[None, :] - hawks_snap) - E[:, None] * np.abs(
            J_soft * rabbit[None, :] - hawks_snap
        )

        # --- Hard besiege ---
        cand_hard = rabbit[None, :] - np.abs(rabbit[None, :] - hawks_snap) * E[:, None]

        # --- Soft besiege with progressive rapid dives ---
        J_sp = 2 * (1 - rng.random((n_hawks, 1)))
        Y_sp = rabbit[None, :] - E[:, None] * np.abs(J_sp * rabbit[None, :] - hawks_snap)
        step_sp = _levy_flight_batch(dim, rng, n_hawks)
        rz_sp = rng.random((n_hawks, 1))
        Z_sp = np.clip(Y_sp + rz_sp * step_sp, lo, hi)

        # --- Hard besiege with progressive rapid dives (Eq. 11) ---
        J_hp = 2 * (1 - rng.random((n_hawks, 1)))
        Y_hp = rabbit[None, :] - E[:, None] * np.abs(J_hp * rabbit[None, :] - mean_snap[None, :])
        step_hp = _levy_flight_batch(dim, rng, n_hawks)
        rz_hp = rng.random((n_hawks, 1))
        Z_hp = np.clip(Y_hp + rz_hp * step_hp, lo, hi)

        # The 4 "simple" strategies (both exploration sub-strategies, soft
        # and hard besiege) each directly become the hawk's new position,
        # clipped and evaluated once -- exactly the final
        # `hawks[i] = clip(...); fitness[i] = _fitness(...)` block the
        # original loop fell through to for these 4 cases.
        simple_mask = mask_explore_hawk | mask_explore_pos | mask_soft | mask_hard
        simple_cand = np.where(
            mask_explore_hawk[:, None], cand_explore_hawk,
            np.where(mask_explore_pos[:, None], cand_explore_pos,
            np.where(mask_soft[:, None], cand_soft, cand_hard))
        )
        simple_cand = np.clip(simple_cand, lo, hi)
        simple_fit = _fitness_batch(simple_cand, net, fault_type=fault_type, topology_state=topology_state)

        # The 2 progressive-dive strategies each need Y/Z evaluated against
        # the hawk's CURRENT fitness before deciding -- batched once across
        # the whole population rather than per-hawk, matching the
        # original's own Y_fit/Z_fit calls but now vectorized.
        Y_sp_fit = _fitness_batch(Y_sp, net, fault_type=fault_type, topology_state=topology_state)
        Z_sp_fit = _fitness_batch(Z_sp, net, fault_type=fault_type, topology_state=topology_state)
        Y_hp_fit = _fitness_batch(Y_hp, net, fault_type=fault_type, topology_state=topology_state)
        Z_hp_fit = _fitness_batch(Z_hp, net, fault_type=fault_type, topology_state=topology_state)

        keep_Y_sp = Y_sp_fit < fitness
        keep_Z_sp = ~keep_Y_sp & (Z_sp_fit < fitness)
        # Neither improves -> hawk stays at its pre-iteration position
        # (hawks_snap), exactly as the original loop's `continue` without
        # reassigning hawks[i] left it unchanged.
        pos_soft_prog = np.where(keep_Y_sp[:, None], Y_sp,
                         np.where(keep_Z_sp[:, None], Z_sp, hawks_snap))
        fit_soft_prog = np.where(keep_Y_sp, Y_sp_fit, np.where(keep_Z_sp, Z_sp_fit, fitness))

        keep_Y_hp = Y_hp_fit < fitness
        keep_Z_hp = ~keep_Y_hp & (Z_hp_fit < fitness)
        pos_hard_prog = np.where(keep_Y_hp[:, None], Y_hp,
                         np.where(keep_Z_hp[:, None], Z_hp, hawks_snap))
        fit_hard_prog = np.where(keep_Y_hp, Y_hp_fit, np.where(keep_Z_hp, Z_hp_fit, fitness))

        # Final per-hawk combination across all 6 strategies.
        hawks = np.where(
            simple_mask[:, None], simple_cand,
            np.where(mask_soft_prog[:, None], pos_soft_prog, pos_hard_prog)
        )
        fitness = np.where(
            simple_mask, simple_fit,
            np.where(mask_soft_prog, fit_soft_prog, fit_hard_prog)
        )

        bi = int(np.argmin(fitness))
        if fitness[bi] < rabbit_fit:
            rabbit_fit = float(fitness[bi])
            rabbit = hawks[bi].copy()
        
        history.append(rabbit_fit)
        
        if verbose and t % 50 == 0:
            print(f"  HHO iter {t:4d}  best_OF={rabbit_fit:.4f}")
    
    # SQP refinement
    if use_sqp:
        if verbose:
            print(f"  HHO best = {rabbit_fit:.4f}  |  SQP refinement ...")
        sqp_res = _sqp_refine(net, rabbit, fault_type, topology_state)
        rabbit = sqp_res["x"]
        rabbit_fit = sqp_res["OF"]
    
    TMS_opt, PS_opt = unpack(rabbit, n)
    constraints = check_constraints(net, TMS_opt, PS_opt, 
                                    fault_type=fault_type, topology_state=topology_state)
    elapsed = time.perf_counter() - t0
    
    if verbose:
        print(f"  HHO OF = {rabbit_fit:.4f} s  |  CTI ok: {constraints['cti_satisfied']}")
        print(f"  Elapsed : {elapsed:.1f} s")
    
    return {
        "method": "HHO",
        "network": net.name,
        "TMS": TMS_opt,
        "PS": PS_opt,
        "OF": rabbit_fit,
        "OF_meta": history[-1] if use_sqp else rabbit_fit,
        "history": history,
        "constraints": constraints,
        "elapsed": elapsed,
        "fault_type": fault_type,
        "pv_level": pv_level,
        "n_evals": n_hawks * max_iter,
    }


def _levy_flight(dim: int, rng: np.random.Generator, beta: float = 1.5) -> np.ndarray:
    """Levy flight step for HHO exploitation."""
    sigma = (scipy.special.gamma(1 + beta) * np.sin(np.pi * beta / 2) /
             (scipy.special.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    u = rng.normal(0, sigma, dim)
    v = rng.normal(0, 1, dim)
    return u / (np.abs(v) ** (1 / beta))


def _levy_flight_batch(dim: int, rng: np.random.Generator, n: int, beta: float = 1.5) -> np.ndarray:
    """Vectorized counterpart of _levy_flight(): draws an independent Levy
    step for each of n hawks at once, shape (n, dim). Same formula/
    distribution as _levy_flight, just batched -- used by run_hho's
    2026-10-26 vectorization (see that function's docstring)."""
    sigma = (scipy.special.gamma(1 + beta) * np.sin(np.pi * beta / 2) /
             (scipy.special.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    u = rng.normal(0, sigma, (n, dim))
    v = rng.normal(0, 1, (n, dim))
    return u / (np.abs(v) ** (1 / beta))


# ============================================================================
# 3. JAYA -- JAYA Algorithm (parameter-free)
# ============================================================================

def run_jaya(
    net: NetworkData,
    n_pop: int = 30,
    max_iter: int = 500,
    seed: int = 42,
    verbose: bool = False,
    fault_type: str = "3ph",
    topology_state=None,
    pv_level: float = 0.0,
    use_sqp: bool = True,
) -> Dict:
    """
    JAYA Algorithm for DOCR.
    
    JAYA is parameter-free: it moves toward the best solution and away
    from the worst solution. No algorithm-specific parameters needed.
    
    Formula: x_new = x + r1*(x_best - |x|) - r2*(x_worst - |x|)
    
    Parameters: Same interface as HWOA.
    """
    rng = np.random.default_rng(seed)
    n = net.n_relays
    lo, hi = _get_bounds(net)
    dim = 2 * n
    
    t0 = time.perf_counter()
    
    # Initialize population
    pop = rng.uniform(0, 1, (n_pop, dim)) * (hi - lo) + lo
    fitness = _fitness_batch(pop, net, fault_type=fault_type, topology_state=topology_state)

    best_idx = int(np.argmin(fitness))
    worst_idx = int(np.argmax(fitness))
    best = pop[best_idx].copy()
    worst = pop[worst_idx].copy()
    best_fit = float(fitness[best_idx])
    history = [best_fit]

    for t in range(max_iter):
        # 2026-10-26: vectorized across the whole population -- same JAYA
        # update formula as before (move toward best, away from worst),
        # computed for every individual at once. r1/r2 are still drawn
        # per-(individual, dimension) exactly as in the original loop
        # (shape (n_pop, dim), matching rng.random(dim) called once per
        # individual there). The original's per-individual greedy
        # accept/reject (only keep new_pos if it strictly improves that
        # individual's own fitness) is preserved via a boolean mask instead
        # of an `if` inside a Python loop -- every individual's candidate
        # is still evaluated and compared independently, just all at once.
        r1 = rng.random((n_pop, dim))
        r2 = rng.random((n_pop, dim))

        new_pop = pop + r1 * (best[None, :] - np.abs(pop)) - r2 * (worst[None, :] - np.abs(pop))
        new_pop = np.clip(new_pop, lo, hi)

        new_fit = _fitness_batch(new_pop, net, fault_type=fault_type, topology_state=topology_state)

        improve = new_fit < fitness                     # (n_pop,)
        pop = np.where(improve[:, None], new_pop, pop)
        fitness = np.where(improve, new_fit, fitness)

        # Update best and worst
        best_idx = int(np.argmin(fitness))
        worst_idx = int(np.argmax(fitness))
        if fitness[best_idx] < best_fit:
            best_fit = float(fitness[best_idx])
            best = pop[best_idx].copy()
        worst = pop[worst_idx].copy()
        
        history.append(best_fit)
        
        if verbose and t % 50 == 0:
            print(f"  JAYA iter {t:4d}  best_OF={best_fit:.4f}")
    
    # SQP refinement
    if use_sqp:
        if verbose:
            print(f"  JAYA best = {best_fit:.4f}  |  SQP refinement ...")
        sqp_res = _sqp_refine(net, best, fault_type, topology_state)
        best = sqp_res["x"]
        best_fit = sqp_res["OF"]
    
    TMS_opt, PS_opt = unpack(best, n)
    constraints = check_constraints(net, TMS_opt, PS_opt, 
                                    fault_type=fault_type, topology_state=topology_state)
    elapsed = time.perf_counter() - t0
    
    if verbose:
        print(f"  JAYA OF = {best_fit:.4f} s  |  CTI ok: {constraints['cti_satisfied']}")
        print(f"  Elapsed : {elapsed:.1f} s")
    
    return {
        "method": "JAYA",
        "network": net.name,
        "TMS": TMS_opt,
        "PS": PS_opt,
        "OF": best_fit,
        "OF_meta": history[-1] if use_sqp else best_fit,
        "history": history,
        "constraints": constraints,
        "elapsed": elapsed,
        "fault_type": fault_type,
        "pv_level": pv_level,
        "n_evals": n_pop * max_iter,
    }


# ============================================================================
# 4. EEO -- Enhanced Equilibrium Optimizer
# ============================================================================

def run_eeo(
    net: NetworkData,
    n_particles: int = 30,
    max_iter: int = 500,
    seed: int = 42,
    verbose: bool = False,
    fault_type: str = "3ph",
    topology_state=None,
    pv_level: float = 0.0,
    use_sqp: bool = True,
) -> Dict:
    """
    Enhanced Equilibrium Optimizer (EEO) for DOCR.
    
    EEO is based on control volume mass balance models used to estimate
    both dynamic and equilibrium states. Particles are concentrations
    that update toward equilibrium (best solutions).
    
    Enhancement: Adaptive a (generation rate) and memory pool.
    
    Parameters: Same interface as HWOA.
    """
    rng = np.random.default_rng(seed)
    n = net.n_relays
    lo, hi = _get_bounds(net)
    dim = 2 * n
    
    t0 = time.perf_counter()
    
    # Initialize particles (concentrations)
    particles = rng.uniform(0, 1, (n_particles, dim)) * (hi - lo) + lo
    fitness = _fitness_batch(particles, net, fault_type=fault_type, topology_state=topology_state)

    # Sort and select equilibrium pool (top 4)
    sorted_idx = np.argsort(fitness)
    eq_pool = particles[sorted_idx[:4]].copy()  # C_eq1, C_eq2, C_eq3, C_eq4
    best_fit = float(fitness[sorted_idx[0]])
    history = [best_fit]

    # Memory pool for diversity
    memory = particles.copy()
    mem_fitness = fitness.copy()

    for t in range(max_iter):
        # Average equilibrium candidate
        C_eq_avg = np.mean(eq_pool, axis=0)
        # Faramarzi et al. 2020, Eq. 9: C_eq is drawn uniformly from a
        # 5-member pool {C_eq1..C_eq4, C_eq_avg}. C_eq_avg was computed
        # every iteration but never included in the draw below (eq_idx
        # only ranged over 0-3, i.e. the top-4 pool) -- silently omitting
        # the averaged candidate the algorithm's own formula requires.
        eq_candidates = np.vstack([eq_pool, C_eq_avg[None, :]])

        # Generation rate (adaptive) -- depends only on t/max_iter, same
        # scalar for every particle (unchanged from the original).
        a1 = 2.0 * np.exp(-(t / max_iter))
        a2 = 2.0 * (1 - t / max_iter)
        t_vec = (1 - t / max_iter) ** (a2 * (t / max_iter))

        # 2026-10-26: vectorized across the whole population -- EEO has no
        # per-particle branching or greedy accept/reject (every particle's
        # concentration update is applied unconditionally each iteration,
        # unlike JAYA's), which makes this a direct elementwise port: every
        # random draw that was per-particle-per-dimension in the original
        # loop (eq_idx, r, mem_idx, sign_vec's draw, r1, r2) is now drawn
        # once for the whole (n_particles[, dim]) array instead of once per
        # particle inside `for i in range(n_particles)`, and every formula
        # operates on the full (n_particles, dim) arrays at once.
        eq_idx = rng.integers(0, 5, n_particles)
        C_eq = eq_candidates[eq_idx]                          # (n_particles, dim)

        r = rng.random((n_particles, dim))
        lambda_vec = np.log(1.0 / r)

        mem_idx = rng.integers(0, n_particles, n_particles)
        mem_influence = 0.1 * (memory[mem_idx] - particles)   # (n_particles, dim)

        sign_vec = np.sign(rng.random((n_particles, dim)) - 0.5)
        F = a1 * sign_vec * (np.exp(-lambda_vec * t_vec) - 1.0)

        GP = 0.5
        r1 = rng.random((n_particles, dim))
        r2 = rng.random((n_particles, dim))
        GCP = 0.5 * r1 * (r2 >= GP).astype(float)

        G0 = GCP * (C_eq - lambda_vec * particles)
        G = G0 * F

        # Eq. 13: C(t+1) = C_eq + (C-C_eq)*F + (G/(lambda*V))*(1-F), V=1.
        particles = (
            C_eq
            + (particles - C_eq) * F
            + (G / (lambda_vec + 1e-10)) * (1.0 - F)
            + mem_influence
        )
        particles = np.clip(particles, lo, hi)

        fitness = _fitness_batch(particles, net, fault_type=fault_type, topology_state=topology_state)

        # Update equilibrium pool with top 4
        sorted_idx = np.argsort(fitness)
        eq_pool = particles[sorted_idx[:4]].copy()
        
        # Update memory (keep best half)
        combined = np.vstack([memory, particles])
        combined_fit = np.concatenate([mem_fitness, fitness])
        mem_sorted = np.argsort(combined_fit)
        memory = combined[mem_sorted[:n_particles]]
        mem_fitness = combined_fit[mem_sorted[:n_particles]]
        
        current_best = float(fitness[sorted_idx[0]])
        if current_best < best_fit:
            best_fit = current_best
        
        history.append(best_fit)
        
        if verbose and t % 50 == 0:
            print(f"  EEO iter {t:4d}  best_OF={best_fit:.4f}")
    
    # SQP refinement -- warm-start from the true best-ever individual
    # (memory, an elitist archive merged/sorted every iteration), not from
    # sorted_idx[0] (the best of only the FINAL iteration's population).
    # EEO's concentration update is applied unconditionally with no
    # greedy/elitist selection at the per-particle level, so the final
    # population's best can be (and empirically is, on some seeds) worse
    # than best_fit / memory[0], which otherwise makes the returned OF
    # inconsistent with the returned TMS/PS, and (with use_sqp=True)
    # warm-starts SQP from an inferior point.
    best_mem_idx = int(np.argmin(mem_fitness))
    best_particle = memory[best_mem_idx].copy()
    best_fit = float(mem_fitness[best_mem_idx])
    if use_sqp:
        if verbose:
            print(f"  EEO best = {best_fit:.4f}  |  SQP refinement ...")
        sqp_res = _sqp_refine(net, best_particle, fault_type, topology_state)
        best_particle = sqp_res["x"]
        best_fit = sqp_res["OF"]
    
    TMS_opt, PS_opt = unpack(best_particle, n)
    constraints = check_constraints(net, TMS_opt, PS_opt, 
                                    fault_type=fault_type, topology_state=topology_state)
    elapsed = time.perf_counter() - t0
    
    if verbose:
        print(f"  EEO OF = {best_fit:.4f} s  |  CTI ok: {constraints['cti_satisfied']}")
        print(f"  Elapsed : {elapsed:.1f} s")
    
    return {
        "method": "EEO",
        "network": net.name,
        "TMS": TMS_opt,
        "PS": PS_opt,
        "OF": best_fit,
        "OF_meta": history[-1] if use_sqp else best_fit,
        "history": history,
        "constraints": constraints,
        "elapsed": elapsed,
        "fault_type": fault_type,
        "pv_level": pv_level,
        "n_evals": n_particles * max_iter,
    }


# ============================================================================
# Unified runner: run all metaheuristics with consistent settings
# ============================================================================

METAHEURISTICS = {
    "HWOA": run_hwoa,
    "HHO": run_hho,
    "JAYA": run_jaya,
    "EEO": run_eeo,
}

# Each runner names its population-size kwarg differently; run_metaheuristic
# used to pass all four names to every runner regardless of method, which
# raised TypeError: run_hwoa() got an unexpected keyword argument 'n_hawks'
# (never caught because nothing called run_metaheuristic/run_all_metaheuristics
# until the Fig 5 pipeline was wired up to use them).
_POP_SIZE_KWARG = {
    "HWOA": "n_whales",
    "HHO": "n_hawks",
    "JAYA": "n_pop",
    "EEO": "n_particles",
}


def run_metaheuristic(
    method: str,
    net: NetworkData,
    pop_size: int = 30,
    max_iter: int = 500,
    seed: int = 42,
    verbose: bool = False,
    fault_type: str = "3ph",
    topology_state=None,
    pv_level: float = 0.0,
    use_sqp: bool = True,
) -> Dict:
    """Run a single metaheuristic by name."""
    if method not in METAHEURISTICS:
        raise ValueError(f"Unknown method: {method}. Available: {list(METAHEURISTICS.keys())}")

    runner = METAHEURISTICS[method]
    return runner(
        net,
        **{_POP_SIZE_KWARG[method]: pop_size},
        max_iter=max_iter,
        seed=seed,
        verbose=verbose,
        fault_type=fault_type,
        topology_state=topology_state,
        pv_level=pv_level,
        use_sqp=use_sqp,
    )


def run_all_metaheuristics(
    net: NetworkData,
    pop_size: int = 30,
    max_iter: int = 500,
    seed: int = 42,
    verbose: bool = False,
    fault_type: str = "3ph",
    topology_state=None,
    pv_level: float = 0.0,
    use_sqp: bool = True,
) -> Dict[str, Dict]:
    """Run all 4 metaheuristics and return results dict."""
    results = {}
    for method in METAHEURISTICS:
        if verbose:
            print(f"\\n{'-'*55}")
            print(f"  Running {method} on {net.name}...")
            print(f"{'-'*55}")
        
        # Use consistent pop_size and max_iter, but handle method-specific param names
        kwargs = {
            "net": net,
            "max_iter": max_iter,
            "seed": seed + list(METAHEURISTICS.keys()).index(method),  # Different seed per method
            "verbose": verbose,
            "fault_type": fault_type,
            "topology_state": topology_state,
            "pv_level": pv_level,
            "use_sqp": use_sqp,
        }
        
        # Add population size with correct parameter name
        if method == "HWOA":
            kwargs["n_whales"] = pop_size
        elif method == "HHO":
            kwargs["n_hawks"] = pop_size
        elif method == "JAYA":
            kwargs["n_pop"] = pop_size
        elif method == "EEO":
            kwargs["n_particles"] = pop_size
        
        results[method] = METAHEURISTICS[method](**kwargs)
    
    return results