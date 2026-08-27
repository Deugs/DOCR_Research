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
    fitness = np.array([_fitness(w, net, fault_type=fault_type, topology_state=topology_state) 
                        for w in whales])
    
    best_idx = int(np.argmin(fitness))
    best_whale = whales[best_idx].copy()
    best_fit = float(fitness[best_idx])
    history = [best_fit]
    
    for t in range(max_iter):
        a = 2.0 - 2.0 * t / max_iter  # Decreases from 2 to 0
        a2 = -1.0 + t * (-1.0) / max_iter  # Decreases from -1 to -2
        
        for i in range(n_whales):
            r1, r2 = rng.random(), rng.random()
            A = 2 * a * r1 - a
            C = 2 * r2
            b = 1.0
            l = (a2 - 1) * rng.random() + 1
            p = rng.random()
            
            if p < 0.5:
                if abs(A) < 1:
                    # Encircling prey
                    D = abs(C * best_whale - whales[i])
                    whales[i] = best_whale - A * D
                else:
                    # Search for prey (exploration)
                    rand_idx = rng.integers(0, n_whales)
                    rand_whale = whales[rand_idx]
                    D = abs(C * rand_whale - whales[i])
                    whales[i] = rand_whale - A * D
            else:
                # Bubble-net attacking (exploitation)
                D = abs(best_whale - whales[i])
                whales[i] = D * np.exp(b * l) * np.cos(2 * np.pi * l) + best_whale
            
            # Boundary handling: reflect back
            whales[i] = np.clip(whales[i], lo, hi)
        
        # Evaluate new positions
        fitness = np.array([_fitness(w, net, fault_type=fault_type, topology_state=topology_state) 
                            for w in whales])
        
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
    fitness = np.array([_fitness(h, net, fault_type=fault_type, topology_state=topology_state) 
                        for h in hawks])
    
    best_idx = int(np.argmin(fitness))
    rabbit = hawks[best_idx].copy()  # Best position (the rabbit)
    rabbit_fit = float(fitness[best_idx])
    history = [rabbit_fit]
    
    for t in range(max_iter):
        for i in range(n_hawks):
            # Escaping energy (Heidari et al. 2019, Algorithm 1): drawn
            # fresh PER HAWK each iteration, not once per iteration and
            # shared by the whole population -- sharing one E value
            # synchronized which of the exploration/exploitation branches
            # (and which of the four besiege strategies) every hawk took
            # in a given iteration, removing the per-agent stochastic
            # diversity the algorithm relies on.
            E0 = 2 * rng.random() - 1  # Initial energy
            E = 2 * E0 * (1 - t / max_iter)  # Escaping energy

            if abs(E) >= 1:
                # Exploration: random hawk or random position
                if rng.random() < 0.5:
                    # Random hawk
                    rand_idx = rng.integers(0, n_hawks)
                    hawks[i] = hawks[rand_idx] - rng.random() * abs(
                        hawks[rand_idx] - 2 * rng.random() * hawks[i]
                    )
                else:
                    # Random position in search space
                    hawks[i] = (rabbit - hawks.mean(axis=0)) - rng.random() * (
                        lo + rng.random() * (hi - lo)
                    )
            else:
                # Exploitation: 4 strategies based on rabbit escaping
                r = rng.random()
                
                if r < 0.5 and abs(E) >= 0.5:
                    # Soft besiege (Heidari et al. 2019, Eq. 6):
                    # X(t+1) = (X_rabbit - X) - E*|J*X_rabbit - X|.
                    # Previous code omitted the escaping-energy multiplier E
                    # entirely and used a spurious "2*rng.random()" scale on
                    # the X term (an artifact copy-pasted from the
                    # exploration-phase formula above), so exploitation never
                    # actually shrank with decaying energy.
                    J = 2 * (1 - rng.random())
                    hawks[i] = (rabbit - hawks[i]) - E * abs(J * rabbit - hawks[i])
                elif r < 0.5 and abs(E) < 0.5:
                    # Hard besiege
                    hawks[i] = rabbit - abs(rabbit - hawks[i]) * E
                elif r >= 0.5 and abs(E) >= 0.5:
                    # Soft besiege with progressive rapid dives
                    J = 2 * (1 - rng.random())
                    Y = rabbit - E * abs(J * rabbit - hawks[i])
                    Y_fit = _fitness(Y, net, fault_type=fault_type, topology_state=topology_state)
                    
                    # Levy flight
                    step = _levy_flight(dim, rng)
                    Z = Y + rng.random() * step
                    Z = np.clip(Z, lo, hi)
                    Z_fit = _fitness(Z, net, fault_type=fault_type, topology_state=topology_state)
                    
                    if Y_fit < fitness[i]:
                        hawks[i] = Y
                        fitness[i] = Y_fit
                    elif Z_fit < fitness[i]:
                        hawks[i] = Z
                        fitness[i] = Z_fit
                    continue
                else:
                    # Hard besiege with progressive rapid dives
                    # (Heidari et al. 2019, Eq. 11): Y = X_rabbit -
                    # E*|J*X_rabbit - X_m(t)|. The jump-strength factor J was
                    # missing entirely (only E was applied), which drops one
                    # of the two randomization terms the paper uses to model
                    # the rabbit's erratic escape jumps.
                    J = 2 * (1 - rng.random())
                    Y = rabbit - E * abs(J * rabbit - hawks.mean(axis=0))
                    Y_fit = _fitness(Y, net, fault_type=fault_type, topology_state=topology_state)
                    
                    step = _levy_flight(dim, rng)
                    Z = Y + rng.random() * step
                    Z = np.clip(Z, lo, hi)
                    Z_fit = _fitness(Z, net, fault_type=fault_type, topology_state=topology_state)
                    
                    if Y_fit < fitness[i]:
                        hawks[i] = Y
                        fitness[i] = Y_fit
                    elif Z_fit < fitness[i]:
                        hawks[i] = Z
                        fitness[i] = Z_fit
                    continue
            
            hawks[i] = np.clip(hawks[i], lo, hi)
            fitness[i] = _fitness(hawks[i], net, fault_type=fault_type, topology_state=topology_state)
        
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
    fitness = np.array([_fitness(p, net, fault_type=fault_type, topology_state=topology_state) 
                        for p in pop])
    
    best_idx = int(np.argmin(fitness))
    worst_idx = int(np.argmax(fitness))
    best = pop[best_idx].copy()
    worst = pop[worst_idx].copy()
    best_fit = float(fitness[best_idx])
    history = [best_fit]
    
    for t in range(max_iter):
        for i in range(n_pop):
            r1, r2 = rng.random(dim), rng.random(dim)
            
            # JAYA update: move toward best, away from worst
            new_pos = pop[i] + r1 * (best - np.abs(pop[i])) - r2 * (worst - np.abs(pop[i]))
            new_pos = np.clip(new_pos, lo, hi)
            
            new_fit = _fitness(new_pos, net, fault_type=fault_type, topology_state=topology_state)
            
            # Greedy selection
            if new_fit < fitness[i]:
                pop[i] = new_pos
                fitness[i] = new_fit
        
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
    fitness = np.array([_fitness(p, net, fault_type=fault_type, topology_state=topology_state) 
                        for p in particles])
    
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

        for i in range(n_particles):
            # Randomly select one equilibrium candidate (5-member pool)
            eq_idx = rng.integers(0, 5)
            C_eq = eq_candidates[eq_idx]
            
            # Generation rate (adaptive)
            a1 = 2.0 * np.exp(-(t / max_iter))
            a2 = 2.0 * (1 - t / max_iter)
            
            # Exponential term (Faramarzi et al. 2020, Eq. 9):
            # T = (1 - Iter/Max_iter)^(a2 * Iter/Max_iter) -- a deterministic
            # function of iteration progress, giving the intended
            # exploration->exploitation decay schedule. The previous
            # `a1 * rng.random(dim)` substituted a fresh per-dimension
            # random draw for the Iter/Max_iter term, making the exponent
            # uncorrelated with iteration progress (resampled independently
            # every particle/iteration instead of decaying over the run).
            t_vec = (1 - t / max_iter) ** (a2 * (t / max_iter))
            
            # Unit vector
            r = rng.random(dim)
            lambda_vec = np.log(1.0 / r)
            
            # Memory influence
            mem_idx = rng.integers(0, n_particles)
            mem_influence = 0.1 * (memory[mem_idx] - particles[i])

            # Update concentration (Faramarzi et al. 2020, Eq. 9): the
            # exponential exploitation term F MUST decay via t_vec so that
            # search intensity shrinks as iterations progress -- t_vec was
            # previously computed and discarded, which meant F never varied
            # with iteration count (only with fresh randoms each step), so
            # EEO never actually transitioned from exploration to
            # exploitation like the real Equilibrium Optimizer does.
            #
            # Eq. 9 defines F = a1 * sign(r-0.5) * (exp(-lambda*t) - 1) --
            # a1 (not a2; a2 is only used above in t_vec, per Eq. 8). F
            # previously reused a2 a second time here, leaving a1 computed
            # every iteration but never referenced anywhere in the function.
            sign_vec = np.sign(rng.random(dim) - 0.5)
            F = a1 * sign_vec * (np.exp(-lambda_vec * t_vec) - 1.0)

            # Generation-rate control parameter (Faramarzi et al. 2020, Eq. 12):
            # GCP = 0.5 * r1 * 1(r2 >= GP). The previous code fixed the scale at
            # a flat 0.5 with no r1 factor, which biases the generation term's
            # magnitude and drops the intended randomization.
            GP = 0.5
            r1 = rng.random(dim)
            r2 = rng.random(dim)
            GCP = 0.5 * r1 * (r2 >= GP).astype(float)

            # Generation term (Eq. 15-16): G0 = GCP*(C_eq - lambda*C), and the
            # rate actually applied must decay through the same F used in the
            # concentration update: G = G0 * F. The previous code computed G0
            # but never multiplied by F, and divided by C_eq_avg instead of
            # using the (1-F) closing factor from Eq. 13, so the generation
            # term neither tracked the exploration/exploitation schedule nor
            # matched the published update rule.
            G0 = GCP * (C_eq - lambda_vec * particles[i])
            G = G0 * F

            # Eq. 13: C(t+1) = C_eq + (C-C_eq)*F + (G/(lambda*V))*(1-F), V=1.
            particles[i] = (
                C_eq
                + (particles[i] - C_eq) * F
                + (G / (lambda_vec + 1e-10)) * (1.0 - F)
                + mem_influence
            )
            particles[i] = np.clip(particles[i], lo, hi)
            
            fitness[i] = _fitness(particles[i], net, fault_type=fault_type, topology_state=topology_state)
        
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