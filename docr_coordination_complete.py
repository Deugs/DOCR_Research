
"""
================================================================================
DOCR COORDINATION FRAMEWORK - COMPLETE PYTHON IMPLEMENTATION
================================================================================
Paper: "Improving Directional Overcurrent Relay Coordination in Distribution 
        Networks for Optimal Operation Using Hybrid Genetic Algorithm with 
        Sequential Quadratic Programming" (Al-Bhadely & Inan, 2023)

Reproduced with:
  - GA-SQP Hybrid Algorithm (Paper Methodology)
  - Reinforcement Learning Enhancement (Proposed Extension)
  - Three Test Systems: IEEE 3-Bus, 8-Bus, 30-Bus

Author: AI Assistant
Date: 2026-06-06
================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution, minimize
import time
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. CORE DOCR SYSTEM CLASS
# ============================================================

class DOCRSystemV3:
    """
    Directional Overcurrent Relay Coordination System

    Implements IEC Standard Inverse characteristic:
    t_op = (A * TMS) / ((I_fault / PS)^B - 1)

    Objective: Minimize total operating time subject to CTI constraints
    """

    def __init__(self, n_relays, relay_pairs, fault_currents_primary, 
                 fault_currents_backup, CTI=0.2, A=0.14, B=0.02, 
                 TMS_bounds=(0.05, 1.1), PS_bounds=None,
                 t_min=0.05, t_max=2.0, current_scale=1.0):
        """
        Parameters:
        -----------
        n_relays : int
            Number of directional overcurrent relays
        relay_pairs : list of tuples
            Primary/Backup relay pairs [(primary_idx, backup_idx), ...]
        fault_currents_primary : dict
            {relay_idx: fault_current_A} for primary operation
        fault_currents_backup : dict
            {relay_idx: fault_current_A} for backup operation
        CTI : float
            Coordination Time Interval (default 0.2s)
        A, B : float
            IEC standard inverse constants (A=0.14, B=0.02)
        TMS_bounds : tuple
            (min, max) for Time Multiplier Setting
        PS_bounds : tuple
            (min_array, max_array) for Plug Setting
        current_scale : float
            Scale factor for current values (e.g., 0.001 for A->kA)
        """
        self.n_relays = n_relays
        self.relay_pairs = relay_pairs
        self.fault_currents_primary = {k: v * current_scale 
                                        for k, v in fault_currents_primary.items()}
        self.fault_currents_backup = {k: v * current_scale 
                                       for k, v in fault_currents_backup.items()}
        self.CTI = CTI
        self.A = A
        self.B = B
        self.TMS_min, self.TMS_max = TMS_bounds
        self.t_min = t_min
        self.t_max = t_max
        self.current_scale = current_scale

        if PS_bounds is None:
            self.PS_min = np.ones(n_relays) * 0.001
            self.PS_max = np.ones(n_relays) * 2.0
        else:
            self.PS_min = np.array(PS_bounds[0])
            self.PS_max = np.array(PS_bounds[1])

    def operating_time(self, TMS, PS, I_fault):
        """
        Calculate relay operating time using IEC Standard Inverse

        t_op = (A * TMS) / ((I_fault / PS)^B - 1)

        Returns penalty value if PS >= I_fault (relay won't operate)
        """
        PS = max(PS, 1e-10)
        ratio = I_fault / PS
        if ratio <= 1.0001:
            return self.t_max * 100  # Heavy penalty for non-operation
        t_op = (self.A * TMS) / (ratio**self.B - 1)
        return min(t_op, self.t_max * 10)

    def objective_function(self, x, alpha1=10000, alpha2=10000, beta=1000):
        """
        Objective function with penalty terms

        OF = sum(primary_times) + alpha1*bounds_penalty + alpha2*CTI_penalty

        x = [TMS_1, ..., TMS_n, PS_1, ..., PS_n]
        """
        n = self.n_relays
        TMS = np.clip(x[:n], self.TMS_min, self.TMS_max)
        PS = np.clip(x[n:], self.PS_min, self.PS_max)

        total_time = 0.0
        penalty = 0.0

        # Calculate for each relay pair
        for primary, backup in self.relay_pairs:
            I_primary = self.fault_currents_primary.get(primary, 1000.0)
            I_backup = self.fault_currents_backup.get(backup, 500.0)

            t_primary = self.operating_time(TMS[primary], PS[primary], I_primary)
            t_backup = self.operating_time(TMS[backup], PS[backup], I_backup)

            total_time += t_primary

            # CTI constraint: t_backup - t_primary >= CTI
            delta_T = t_backup - t_primary - self.CTI
            if delta_T < 0:
                penalty += alpha2 * abs(delta_T)
            elif delta_T < 0.05:
                penalty += beta * (0.05 - delta_T) * 10

        # Bounds penalties
        for i in range(n):
            if TMS[i] < self.TMS_min - 1e-6:
                penalty += alpha1 * abs(self.TMS_min - TMS[i])
            if TMS[i] > self.TMS_max + 1e-6:
                penalty += alpha1 * abs(TMS[i] - self.TMS_max)
            if PS[i] < self.PS_min[i] - 1e-6:
                penalty += alpha1 * abs(self.PS_min[i] - PS[i])
            if PS[i] > self.PS_max[i] + 1e-6:
                penalty += alpha1 * abs(PS[i] - self.PS_max[i])

        return total_time + penalty

    def get_operating_times(self, x):
        """Get detailed operating times for all relay pairs"""
        n = self.n_relays
        TMS = x[:n]
        PS = x[n:]

        results = []
        for primary, backup in self.relay_pairs:
            I_p = self.fault_currents_primary.get(primary, 1000.0)
            I_b = self.fault_currents_backup.get(backup, 500.0)

            t_p = self.operating_time(TMS[primary], PS[primary], I_p)
            t_b = self.operating_time(TMS[backup], PS[backup], I_b)
            cti = t_b - t_p

            results.append({
                'primary': primary,
                'backup': backup,
                't_primary': t_p,
                't_backup': t_b,
                'cti': cti,
                'valid': cti >= self.CTI
            })
        return results

    def print_results(self, x, method_name=""):
        """Print formatted results table"""
        n = self.n_relays
        TMS = x[:n]
        PS = x[n:]

        print(f"\n{'='*60}")
        print(f"RESULTS: {method_name}")
        print(f"{'='*60}")
        print(f"{'Relay':<8} {'TMS (s)':<12} {'PS (pu)':<12}")
        print("-" * 35)

        for i in range(n):
            print(f"R{i+1:<7} {TMS[i]:<12.4f} {PS[i]:<12.4f}")

        of = self.objective_function(x)
        times = self.get_operating_times(x)
        total_primary_time = sum(t['t_primary'] for t in times)

        print(f"\nObjective Function: {of:.6f} s")
        print(f"Total Primary Operating Time: {total_primary_time:.4f} s")

        print(f"\n{'Primary':<10} {'Backup':<10} {'t_p (s)':<12} {'t_b (s)':<12} {'CTI (s)':<10} {'Status'}")
        print("-" * 65)

        valid_count = 0
        for t in times:
            status = "✓" if t['valid'] else "✗"
            if t['valid']: valid_count += 1
            print(f"R{t['primary']+1:<9} R{t['backup']+1:<9} {t['t_primary']:<12.4f} "
                  f"{t['t_backup']:<12.4f} {t['cti']:<10.4f} {status}")

        print(f"\nCoordination Valid: {valid_count}/{len(times)} pairs")
        return total_primary_time


# ============================================================
# 2. TEST SYSTEMS (From Paper Data)
# ============================================================

def get_3bus_system():
    """
    IEEE 3-Bus System (6 relays, 6 P/B pairs)
    From paper Table 2, page 8
    """
    n_relays = 6
    relay_pairs = [(0, 5), (1, 3), (2, 0), (3, 4), (4, 1), (5, 2)]

    fault_currents_primary = {
        0: 1961.2, 1: 1515.4, 2: 1678.9, 3: 1816.5, 4: 1765.1, 5: 1499.8
    }
    fault_currents_backup = {
        5: 172.7, 3: 544.9, 0: 611.8, 4: 467.4, 1: 144.6, 2: 385.3
    }

    PS_min = np.array([0.005, 0.005, 0.005, 0.005, 0.005, 0.005])
    PS_max = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])

    return DOCRSystemV3(n_relays, relay_pairs, fault_currents_primary, 
                        fault_currents_backup, CTI=0.2, A=0.14, B=0.02,
                        TMS_bounds=(0.05, 1.1), PS_bounds=(PS_min, PS_max),
                        current_scale=0.001)


def get_8bus_system():
    """
    IEEE 8-Bus System (14 relays, 19 P/B pairs)
    From paper Table 6, page 11
    """
    n_relays = 14
    relay_pairs = [
        (0, 5), (1, 7), (2, 1), (2, 5), (3, 9), (4, 1), (4, 3),
        (5, 6), (5, 12), (6, 0), (7, 4), (7, 12), (8, 2), (9, 11),
        (10, 8), (11, 13), (12, 10), (13, 4), (13, 6)
    ]

    fault_currents_primary = {
        0: 3500.5, 1: 1710.0, 2: 3521.6, 3: 1892.0, 4: 2883.8,
        5: 2905.6, 6: 1622.5, 7: 3660.4, 8: 2104.5, 9: 3294.0,
        10: 3242.1, 11: 2197.8, 12: 1901.2, 13: 3615.3
    }

    fault_currents_backup = {
        5: 638.4, 7: 1704.2, 1: 533.2, 5: 1009.0, 9: 1948.8,
        1: 11.2, 3: 847.0, 6: 3.8, 12: 886.0, 0: 1616.5,
        4: 697.6, 12: 1014.0, 2: 2015.9, 11: 1015.9, 8: 961.4,
        13: 2104.4, 10: 1956.3, 4: 1041.2, 6: 550.5
    }

    PS_min = np.ones(n_relays) * 0.005
    PS_max = np.ones(n_relays) * 0.5

    return DOCRSystemV3(n_relays, relay_pairs, fault_currents_primary,
                        fault_currents_backup, CTI=0.2, A=0.14, B=0.02,
                        TMS_bounds=(0.05, 1.1), PS_bounds=(PS_min, PS_max),
                        current_scale=0.001)


def get_30bus_system():
    """
    IEEE 30-Bus System (42 relays, 41 P/B pairs)
    Simplified topology based on paper description
    """
    n_relays = 42
    relay_pairs = []
    for line in range(21):
        r1 = line * 2
        r2 = line * 2 + 1
        relay_pairs.append((r1, r2))
    for i in range(20):
        relay_pairs.append((i * 2, ((i + 1) * 2 + 1) % 42))

    np.random.seed(42)
    fault_currents_primary = {}
    fault_currents_backup = {}
    for i in range(n_relays):
        base = 8000 + np.random.normal(0, 2000)
        fault_currents_primary[i] = max(1000, base)
        fault_currents_backup[i] = max(500, base * 0.3 + np.random.normal(0, 300))

    PS_min = np.ones(n_relays) * 0.005
    PS_max = np.ones(n_relays) * 0.5

    return DOCRSystemV3(n_relays, relay_pairs, fault_currents_primary,
                        fault_currents_backup, CTI=0.2, A=0.14, B=0.02,
                        TMS_bounds=(0.05, 1.1), PS_bounds=(PS_min, PS_max),
                        current_scale=0.001)


# ============================================================
# 3. GA-SQP HYBRID OPTIMIZER
# ============================================================

def run_ga_sqp(system, popsize=30, maxiter=100, seed=42, verbose=False):
    """
    Run GA-SQP hybrid optimization

    Phase 1: Differential Evolution (Global Exploration)
    Phase 2: SLSQP (Local Refinement)
    """
    n = system.n_relays
    bounds = []
    for i in range(n):
        bounds.append((system.TMS_min, system.TMS_max))
    for i in range(n):
        bounds.append((system.PS_min[i], system.PS_max[i]))

    start = time.time()

    # GA Phase: Differential Evolution
    result_ga = differential_evolution(
        system.objective_function, bounds, maxiter=maxiter, popsize=popsize,
        seed=seed, polish=False, workers=-1, tol=1e-8,
        mutation=(0.5, 1.0), recombination=0.7
    )

    ga_time = time.time() - start
    ga_of = system.objective_function(result_ga.x)

    # SQP Phase: SLSQP with explicit constraints
    constraints = []
    for primary, backup in system.relay_pairs:
        def make_constraint(p, b):
            def cti(x):
                n = system.n_relays
                TMS = x[:n]
                PS = x[n:]
                I_p = system.fault_currents_primary.get(p, 1000.0)
                I_b = system.fault_currents_backup.get(b, 500.0)
                t_p = system.operating_time(TMS[p], PS[p], I_p)
                t_b = system.operating_time(TMS[b], PS[b], I_b)
                return t_b - t_p - system.CTI
            return cti
        constraints.append({'type': 'ineq', 'fun': make_constraint(primary, backup)})

    start_sqp = time.time()
    result_sqp = minimize(
        system.objective_function, result_ga.x, method='SLSQP',
        bounds=bounds, constraints=constraints,
        options={'ftol': 1e-12, 'maxiter': 1000, 'disp': False}
    )
    sqp_time = time.time() - start_sqp
    total_time = time.time() - start
    sqp_of = system.objective_function(result_sqp.x)

    if verbose:
        print(f"GA-SQP OF: {sqp_of:.6f} s | Time: {total_time:.2f}s")

    return {
        'ga_x': result_ga.x,
        'sqp_x': result_sqp.x,
        'ga_of': ga_of,
        'sqp_of': sqp_of,
        'total_time': total_time
    }


# ============================================================
# 4. REINFORCEMENT LEARNING COMPONENTS
# ============================================================

class DOCRRLEnv:
    """RL Environment for DOCR Coordination"""

    def __init__(self, system, max_steps=50):
        self.system = system
        self.max_steps = max_steps
        self.n = system.n_relays
        self.state_dim = self.n * 3
        self.action_dim = self.n * 2
        self.action_scale = 0.02
        self.reset()

    def reset(self):
        self.TMS = np.random.uniform(0.05, 0.5, self.n)
        self.PS = np.random.uniform(0.05, 0.3, self.n)
        self.step_count = 0
        return self._get_state()

    def _get_state(self):
        state = np.zeros(self.state_dim)
        state[0:self.n] = self.TMS / 1.1
        state[self.n:2*self.n] = self.PS / 0.5
        for i in range(self.n):
            I_f = self.system.fault_currents_primary.get(i, 1000.0)
            state[2*self.n + i] = I_f / 10000.0
        return state

    def step(self, action):
        delta_TMS = np.clip(action[:self.n], -1, 1) * self.action_scale
        delta_PS = np.clip(action[self.n:], -1, 1) * self.action_scale * 0.5

        self.TMS = np.clip(self.TMS + delta_TMS, 0.05, 1.1)
        self.PS = np.clip(self.PS + delta_PS, 0.005, 0.5)

        x = np.concatenate([self.TMS, self.PS])
        of = self.system.objective_function(x)

        times = self.system.get_operating_times(x)
        valid_cti = sum(1 for t in times if t['valid'])
        total_cti = sum(t['cti'] for t in times)

        reward = -of * 10 + valid_cti * 50 + total_cti * 5

        self.step_count += 1
        done = self.step_count >= self.max_steps

        return self._get_state(), reward, done, {
            'of': of, 'valid_cti': valid_cti, 'total_cti': total_cti, 'x': x
        }

    def get_solution(self):
        return np.concatenate([self.TMS, self.PS])


class RLAgent:
    """Simple Policy Gradient Agent (REINFORCE)"""

    def __init__(self, state_dim, action_dim, lr=0.001, gamma=0.95):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma

        # 2-layer neural network
        self.W1 = np.random.randn(64, state_dim) * 0.1
        self.b1 = np.zeros(64)
        self.W2 = np.random.randn(action_dim, 64) * 0.1
        self.b2 = np.zeros(action_dim)
        self.memory = []

    def forward(self, state):
        h1 = np.tanh(self.W1 @ state + self.b1)
        out = np.tanh(self.W2 @ h1 + self.b2)
        return out, h1

    def select_action(self, state, noise=0.1):
        mean, _ = self.forward(state)
        return np.clip(mean + np.random.randn(self.action_dim) * noise, -1, 1)

    def store(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def update(self, batch_size=32):
        if len(self.memory) < batch_size:
            return
        batch = self.memory[-batch_size:]
        for state, action, reward, next_state, done in batch:
            mean, h1 = self.forward(state)
            grad_out = (action - mean) * reward * 0.1
            self.W2 += self.lr * np.outer(grad_out, h1)
            self.b2 += self.lr * grad_out
            grad_h1 = self.W2.T @ grad_out * (1 - h1**2)
            self.W1 += self.lr * np.outer(grad_h1, state)
            self.b1 += self.lr * grad_h1
        self.memory = []


def train_rl(system, n_episodes=500, max_steps=50, verbose=False):
    """Train RL agent on DOCR problem"""
    env = DOCRRLEnv(system, max_steps)
    agent = RLAgent(env.state_dim, env.action_dim, lr=0.001)

    best_of = float('inf')
    best_x = None
    history = []

    for ep in range(n_episodes):
        state = env.reset()
        for step in range(max_steps):
            noise = max(0.05, 0.5 * (1 - ep / n_episodes))
            action = agent.select_action(state, noise)
            next_state, reward, done, info = env.step(action)
            agent.store(state, action, reward, next_state, done)
            state = next_state
            if done:
                break

        agent.update()
        x = env.get_solution()
        of = system.objective_function(x)

        if of < best_of:
            best_of = of
            best_x = x.copy()
        history.append(of)

        if verbose and (ep + 1) % 100 == 0:
            times = system.get_operating_times(x)
            valid = sum(1 for t in times if t['valid'])
            print(f"Ep {ep+1}: OF={of:.4f}, Best={best_of:.4f}, Valid={valid}/{len(times)}")

    return {'best_x': best_x, 'best_of': best_of, 'history': history}


# ============================================================
# 5. MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    print("="*70)
    print("DOCR COORDINATION: GA-SQP vs RL")
    print("="*70)

    # Test all three systems
    systems = {
        '3-Bus': get_3bus_system(),
        '8-Bus': get_8bus_system(),
        '30-Bus': get_30bus_system()
    }

    for name, system in systems.items():
        print(f"\n{'='*70}")
        print(f"CASE STUDY: {name} SYSTEM ({system.n_relays} relays)")
        print(f"{'='*70}")

        # GA-SQP
        print("\n[GA-SQP Optimization]")
        result = run_ga_sqp(system, popsize=50, maxiter=200, verbose=True)
        system.print_results(result['sqp_x'], f"GA-SQP ({name})")

        # RL
        print("\n[RL Optimization]")
        rl_result = train_rl(system, n_episodes=300, max_steps=50, verbose=True)
        system.print_results(rl_result['best_x'], f"RL ({name})")

    print("\n" + "="*70)
    print("BENCHMARK COMPLETE!")
    print("="*70)
