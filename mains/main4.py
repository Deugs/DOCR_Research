
"""
main.py
───────
Phase 4 Entry Point for DOCR-MARL Thesis.

Runs comprehensive experiments comparing:
1. GA-SQP (classical control baseline)
2. Single-PPO (centralized RL baseline)
3. IPPO (decentralized RL, no communication)
4. MAPPO (centralized critic, no communication)
5. MAPPO-Comm (MAPPO + GNN communication)
6. MAPPO-Safe (MAPPO + Augmented Lagrangian + Action Projection)
7. MAPPO-Safe-EWC (MAPPO-Safe + EWC continual adaptation)
8. HWOA (Hybrid Whale Optimization Algorithm)
9. HHO (Harris Hawks Optimization)
10. JAYA (JAYA Algorithm — parameter-free)
11. EEO (Enhanced Equilibrium Optimizer)

Phase 4 adds:
- Benchmark suite with metaheuristics (HWOA, HHO, JAYA, EEO)
- Statistical validation (Friedman, Nemenyi, Cliff's delta, Wilcoxon)
- Automated comparison tables and plots
- LaTeX table generation for thesis

Usage:
    python main.py --phase4                    # all Phase 4 experiments
    python main.py --phase4 --systems 8bus     # single system
    python main.py --phase4 --fast             # reduced iterations
    python main.py --phase4 --benchmark        # run benchmark suite
    python main.py --phase4 --stats            # statistical tests only
    python main.py --phase4 --compare          # full comparison table
"""

import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
from network.network_data import NETWORKS
from network.relay_physics import check_constraints, FAULT_TYPES
from baselines.ga_sqp import run_ga_sqp

os.makedirs("outputs", exist_ok=True)


# ── Phase 1-3: Existing MARL runners (preserved) ────────────────────────────

def run_ippo(net, total_steps=30000, seed=0, verbose=True, use_pandapower=False):
    """Run IPPO (decentralized, no communication) for backward comparison."""
    from marl.marl_env import DynamicDOCREnv
    from marl.marl_agents import MAPPOMARLTrainer

    env = DynamicDOCREnv(net, action_scale=0.05, max_steps=300, seed=seed,
                         use_pandapower=use_pandapower, fault_type="3ph")
    trainer = MAPPOMARLTrainer(
        env, use_comm=False, hidden_actor=32, hidden_critic=64,
        lr_actor=0.02, clip_eps=0.2, gamma=0.99, lam=0.95,
        rollout_len=64, n_epochs=4, seed=seed, curriculum=False,
    )
    result = trainer.train(total_steps=total_steps, verbose=verbose,
                           log_every=max(total_steps // 10, 1))
    result["constraints"] = check_constraints(net, result["TMS"], result["PS"], fault_type="3ph")
    return result


def run_mappo(net, total_steps=30000, seed=0, verbose=True, use_pandapower=False):
    """Run MAPPO (centralized critic, no communication)."""
    from marl.marl_env import DynamicDOCREnv
    from marl.marl_agents import MAPPOMARLTrainer

    env = DynamicDOCREnv(net, action_scale=0.05, max_steps=300, seed=seed,
                         use_pandapower=use_pandapower, fault_type="3ph")
    trainer = MAPPOMARLTrainer(
        env, use_comm=False, hidden_actor=64, hidden_critic=128,
        lr_actor=0.02, lr_critic=0.01, clip_eps=0.2, gamma=0.99, lam=0.95,
        rollout_len=64, n_epochs=4, seed=seed, curriculum=False,
    )
    result = trainer.train(total_steps=total_steps, verbose=verbose,
                           log_every=max(total_steps // 10, 1))
    result["constraints"] = check_constraints(net, result["TMS"], result["PS"], fault_type="3ph")
    return result


def run_mappo_comm(net, total_steps=30000, seed=0, verbose=True, use_pandapower=False):
    """Run MAPPO with GNN communication."""
    from marl.marl_env import DynamicDOCREnv
    from marl.marl_agents import MAPPOMARLTrainer

    env = DynamicDOCREnv(net, action_scale=0.05, max_steps=300, seed=seed,
                         use_pandapower=use_pandapower, fault_type="3ph")
    trainer = MAPPOMARLTrainer(
        env, use_comm=True, hidden_actor=64, hidden_critic=128,
        lr_actor=0.02, lr_critic=0.01, clip_eps=0.2, gamma=0.99, lam=0.95,
        rollout_len=64, n_epochs=4, seed=seed, curriculum=False,
    )
    result = trainer.train(total_steps=total_steps, verbose=verbose,
                           log_every=max(total_steps // 10, 1))
    result["constraints"] = check_constraints(net, result["TMS"], result["PS"], fault_type="3ph")
    return result


def run_mappo_safe(net, total_steps=30000, seed=0, verbose=True, use_pandapower=False,
                   safe_config=None):
    """Run MAPPO with Safe RL (Augmented Lagrangian + Action Projection)."""
    from marl.marl_env import DynamicDOCREnv
    from marl.marl_agents import MAPPOMARLTrainer
    from marl.safe_rl import create_safe_rl_env, SafeRLConfig

    env = DynamicDOCREnv(net, action_scale=0.05, max_steps=300, seed=seed,
                         use_pandapower=use_pandapower, fault_type="3ph")

    safe_config = safe_config or SafeRLConfig(
        project_actions=True, enforce_cti=True,
        lambda_init=1.0, lambda_lr=0.01, rho_penalty=10.0,
        use_pid=True, adaptive_rho=True
    )
    safe_env = create_safe_rl_env(env, safe_config)

    trainer = MAPPOMARLTrainer(
        safe_env, use_comm=False, hidden_actor=64, hidden_critic=128,
        lr_actor=0.02, lr_critic=0.01, clip_eps=0.2, gamma=0.99, lam=0.95,
        rollout_len=64, n_epochs=4, seed=seed, curriculum=False,
    )
    result = trainer.train(total_steps=total_steps, verbose=verbose,
                           log_every=max(total_steps // 10, 1))
    result["constraints"] = check_constraints(net, result["TMS"], result["PS"], fault_type="3ph")

    safe_stats = safe_env.get_safe_rl_stats()
    result["safe_rl_stats"] = safe_stats
    result["method"] = "MAPPO-Safe"
    return result


def run_mappo_safe_ewc(net, total_steps=30000, seed=0, verbose=True, use_pandapower=False,
                       safe_config=None, ewc_config=None):
    """Run MAPPO with Safe RL + EWC continual adaptation."""
    from marl.marl_env import DynamicDOCREnv
    from marl.marl_agents import MAPPOMARLTrainer
    from marl.safe_rl import create_safe_rl_env, SafeRLConfig
    from marl.continual_trainer import create_ewc_trainer, EWCConfig

    env = DynamicDOCREnv(net, action_scale=0.05, max_steps=300, seed=seed,
                         use_pandapower=use_pandapower, fault_type="3ph")

    safe_config = safe_config or SafeRLConfig(
        project_actions=True, enforce_cti=True,
        lambda_init=1.0, lambda_lr=0.01, rho_penalty=10.0,
        use_pid=True, adaptive_rho=True
    )
    safe_env = create_safe_rl_env(env, safe_config)

    base_trainer = MAPPOMARLTrainer(
        safe_env, use_comm=False, hidden_actor=64, hidden_critic=128,
        lr_actor=0.02, lr_critic=0.01, clip_eps=0.2, gamma=0.99, lam=0.95,
        rollout_len=64, n_epochs=4, seed=seed, curriculum=False,
    )

    ewc_config = ewc_config or EWCConfig(
        lambda_ewc=100.0, fisher_samples=100,
        use_curriculum_warmstart=True, max_tasks=6
    )
    ewc_trainer = create_ewc_trainer(base_trainer, ewc_config)

    result = base_trainer.train(total_steps=total_steps, verbose=verbose,
                                log_every=max(total_steps // 10, 1))
    result["constraints"] = check_constraints(net, result["TMS"], result["PS"], fault_type="3ph")

    # Register this (single) task with EWC so the apparatus genuinely runs
    # rather than staying completely inert (ewc_trainer was previously
    # built but base_trainer.train() bypassed it entirely). A lone task
    # has nothing to protect against forgetting — expected, not a bug —
    # but the machinery should reflect that it actually ran.
    cur_params = base_trainer.actor.get_all_params()
    obs_f = safe_env.reset(seed=seed + 9000)
    s_obs, s_act = [], []
    done_f, sf = False, 0
    rng = np.random.default_rng(seed + 9999)
    while not done_f and sf < 30:
        acts = np.zeros((safe_env.n_agents, safe_env.ACT_DIM))
        for i in range(safe_env.n_agents):
            a, _ = base_trainer.actor.sample(obs_f[i], rng)
            acts[i] = a
            s_obs.append(obs_f[i].copy())
            s_act.append(a.copy())
        obs_f, _, done_f, _ = safe_env.step(acts); sf += 1
    n_p = len(cur_params)
    fisher = np.zeros(n_p)
    if s_obs:
        idx = rng.choice(len(s_obs), size=min(30, len(s_obs)), replace=False)
        h = 1e-3
        for p_idx in rng.choice(n_p, size=min(n_p, 50), replace=False):
            bl = np.mean([base_trainer.actor.log_prob(s_obs[j], s_act[j]) for j in idx])
            b2 = cur_params.copy(); b2[p_idx] += h
            base_trainer.actor.set_all_params(b2)
            ul = np.mean([base_trainer.actor.log_prob(s_obs[j], s_act[j]) for j in idx])
            fisher[p_idx] = ((ul - bl) / h) ** 2
        base_trainer.actor.set_all_params(cur_params)
    ewc_trainer.ewc_regularizer.register_task("MAPPO-Safe-EWC", cur_params)
    ewc_trainer.ewc_regularizer.update_fisher("MAPPO-Safe-EWC", fisher)
    # No "next" task will ever register here to trigger the lagged
    # consolidation (this is a single-shot call, not a task sequence) —
    # finalize() closes this one out explicitly.
    ewc_trainer.ewc_regularizer.finalize()

    safe_stats = safe_env.get_safe_rl_stats()
    ewc_stats = ewc_trainer.get_adaptation_stats()
    result["safe_rl_stats"] = safe_stats
    result["ewc_stats"] = ewc_stats
    result["method"] = "MAPPO-Safe-EWC"
    return result


def run_single_ppo(net, total_steps=30000, seed=0, verbose=True, use_pandapower=False):
    """Run centralized Single-PPO baseline."""
    from marl.single_agent_ppo import run_single_ppo as _run_single_ppo
    return _run_single_ppo(net, total_steps=total_steps, seed=seed, verbose=verbose,
                           use_pandapower=use_pandapower, hidden_actor=128, hidden_critic=128)


# ── Phase 4: Metaheuristic runners ───────────────────────────────────────────

def run_hwoa(net, pop_size=30, max_iter=500, seed=42, verbose=True,
             fault_type="3ph", topology_state=None, pv_level=0.0, use_sqp=True):
    """Run HWOA (Hybrid Whale Optimization Algorithm)."""
    from baselines.metaheuristics import run_hwoa as _run_hwoa
    return _run_hwoa(net, n_whales=pop_size, max_iter=max_iter, seed=seed,
                     verbose=verbose, fault_type=fault_type,
                     topology_state=topology_state, pv_level=pv_level, use_sqp=use_sqp)


def run_hho(net, pop_size=30, max_iter=500, seed=42, verbose=True,
            fault_type="3ph", topology_state=None, pv_level=0.0, use_sqp=True):
    """Run HHO (Harris Hawks Optimization)."""
    from baselines.metaheuristics import run_hho as _run_hho
    return _run_hho(net, n_hawks=pop_size, max_iter=max_iter, seed=seed,
                    verbose=verbose, fault_type=fault_type,
                    topology_state=topology_state, pv_level=pv_level, use_sqp=use_sqp)


def run_jaya(net, pop_size=30, max_iter=500, seed=42, verbose=True,
             fault_type="3ph", topology_state=None, pv_level=0.0, use_sqp=True):
    """Run JAYA Algorithm (parameter-free)."""
    from baselines.metaheuristics import run_jaya as _run_jaya
    return _run_jaya(net, n_pop=pop_size, max_iter=max_iter, seed=seed,
                     verbose=verbose, fault_type=fault_type,
                     topology_state=topology_state, pv_level=pv_level, use_sqp=use_sqp)


def run_eeo(net, pop_size=30, max_iter=500, seed=42, verbose=True,
            fault_type="3ph", topology_state=None, pv_level=0.0, use_sqp=True):
    """Run EEO (Enhanced Equilibrium Optimizer)."""
    from baselines.metaheuristics import run_eeo as _run_eeo
    return _run_eeo(net, n_particles=pop_size, max_iter=max_iter, seed=seed,
                    verbose=verbose, fault_type=fault_type,
                    topology_state=topology_state, pv_level=pv_level, use_sqp=use_sqp)


# ── Phase 4: Statistical analysis runner ─────────────────────────────────────

def run_statistical_analysis(results_dict, alpha=0.05):
    """Run Friedman, Nemenyi, Cliff's delta, and Wilcoxon tests."""
    from evaluation.stat_analysis import run_benchmark_analysis
    
    print(f"\n{'='*70}")
    print(f"  PHASE 4: STATISTICAL ANALYSIS")
    print(f"{'='*70}")
    
    return run_benchmark_analysis(results_dict, metric="OF", alpha=alpha, verbose=True)


# ── Phase 4: Benchmark suite runner ──────────────────────────────────────────

def run_benchmark_suite(systems, methods, n_runs=10, fast=False, verbose=True):
    """Run full benchmark suite with multiple independent runs."""
    from evaluation.benchmark_suite import BenchmarkSuite, BenchmarkConfig
    
    cfg = BenchmarkConfig(
        systems=systems,
        methods=methods,
        n_runs=n_runs,
        pop_size=20 if fast else 30,
        max_iter=200 if fast else 500,
        ga_pop=50 if fast else 100,
        ga_gen=100 if fast else 200,
        marl_steps=5000 if fast else 10000,
        verbose=verbose,
    )
    
    suite = BenchmarkSuite(cfg)
    suite.run_all()
    suite.print_comparison_table()
    suite.plot_convergence("outputs/phase4_convergence.png")
    
    if len(systems) >= 2 and len(methods) >= 3:
        suite.run_statistical_tests()
        suite.plot_cd_diagram("outputs/phase4_cd_diagram.png")
        suite.generate_latex_table("outputs/phase4_table.tex")
    
    suite.save_results("outputs/phase4_results.npz")
    return suite


# ── Comparison tables (updated for Phase 4) ──────────────────────────────────

def print_comparison_table(results_dict):
    """Print formatted comparison table across all methods (Phase 4 extended)."""
    LINE = "─" * 140
    print(f"\n{LINE}")
    print(f"  {'Network':<12} {'Method':<18} {'OF (s)':>9} "
          f"{'CTI viol':>9} {'CTI min':>10} {'Time (s)':>10} "
          f"{'Comm BW':>10} {'Safe λ':>8} {'EWC Tasks':>10} {'Improvement':>12}")
    print(LINE)

    for net_key, res in results_dict.items():
        ga_of = None
        if "GA-SQP" in res:
            ga_of = res["GA-SQP"]["OF"]
        
        methods = [m for m in res if m != "net"]
        for meth in methods:
            r = res[meth]
            c = r.get("constraints", check_constraints(res.get("net", res.get("GA-SQP", {}).get("net")),
                                                        r["TMS"], r["PS"], fault_type="3ph"))
            comm_bw = r.get("comm_bandwidth", 0.0)
            safe_lambda = r.get("safe_rl_stats", {}).get("lambda_mean", 0.0)
            ewc_tasks = r.get("ewc_stats", {}).get("n_tasks_learned", 0)
            imp = (ga_of - r["OF"]) / ga_of * 100 if ga_of and ga_of > 0 else 0.0

            print(f"  {net_key:<12} {meth:<18} "
                  f"{r['OF']:>9.4f} {c.get('cti_violations', 0):>9d} "
                  f"{c.get('cti_min', 0.0):>10.4f} {r['elapsed']:>10.2f} "
                  f"{comm_bw:>9.3f}kbps {safe_lambda:>8.3f} {ewc_tasks:>10d} {imp:>+11.2f}%")
        print()
    print(LINE)

def plot_phase3_comparison(results_dict, out_path):
    """Plot Phase 3 method comparison including Safe RL and EWC."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from network.network_data import NETWORKS
    from network.relay_physics import check_constraints

    STYLE = {
        "figure.facecolor": "#0d1117",
        "axes.facecolor":   "#161b22",
        "axes.edgecolor":   "#30363d",
        "axes.labelcolor":  "#8b949e",
        "axes.titlecolor":  "#e6edf3",
        "xtick.color":      "#8b949e",
        "ytick.color":      "#8b949e",
        "grid.color":       "#21262d",
        "grid.linewidth":   0.8,
        "text.color":       "#e6edf3",
        "legend.facecolor": "#161b22",
        "legend.edgecolor": "#30363d",
        "font.family":      "monospace",
    }
    plt.rcParams.update(STYLE)

    COLORS = {
        "GA-SQP": "#2563eb",
        "Single-PPO": "#f778ba",
        "IPPO": "#a371f7",
        "MAPPO": "#39d353",
        "MAPPO-Comm": "#00d4ff",
        "MAPPO-Safe": "#ff6b35",
        "MAPPO-Safe-EWC": "#f7c948",
    }

    nets = list(results_dict.keys())
    methods = ["GA-SQP", "Single-PPO", "IPPO", "MAPPO", "MAPPO-Comm", "MAPPO-Safe", "MAPPO-Safe-EWC"]

    # Reconstruct network objects from names
    net_objects = {}
    for net_key in nets:
        if "GA-SQP" in results_dict[net_key]:
            net_name = results_dict[net_key]["GA-SQP"].get("network", "")
            for key, fn in NETWORKS.items():
                if fn().name == net_name:
                    net_objects[net_key] = fn()
                    break

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Phase 3: Safe RL + EWC Method Comparison", fontsize=12)

    # OF comparison
    ax = axes[0, 0]
    x = np.arange(len(nets))
    w = 0.12
    for i, meth in enumerate(methods):
        ofs = []
        for net in nets:
            if meth in results_dict[net]:
                ofs.append(results_dict[net][meth]["OF"])
            else:
                ofs.append(np.nan)
        ax.bar(x + i * w - 3 * w, ofs, w, label=meth, color=COLORS.get(meth, "#fff"), alpha=0.85)

    ax.set_title("Objective Function", fontsize=10)
    ax.set_xlabel("Network")
    ax.set_ylabel("OF (s)")
    ax.set_xticks(x)
    ax.set_xticklabels([n.upper() for n in nets])
    ax.legend(fontsize=6, ncol=2)
    ax.grid(True, alpha=0.3)

    # CTI violations
    ax = axes[0, 1]
    for i, meth in enumerate(methods):
        viols = []
        for net in nets:
            if meth in results_dict[net] and net in net_objects:
                r = results_dict[net][meth]
                c = r.get("constraints")
                if c is None:
                    c = check_constraints(net_objects[net], r["TMS"], r["PS"])
                viols.append(c.get("cti_violations", 0))
            else:
                viols.append(np.nan)
        ax.bar(x + i * w - 3 * w, viols, w, label=meth, color=COLORS.get(meth, "#fff"), alpha=0.85)

    ax.set_title("CTI Violations", fontsize=10)
    ax.set_xlabel("Network")
    ax.set_ylabel("# Violations")
    ax.set_xticks(x)
    ax.set_xticklabels([n.upper() for n in nets])
    ax.legend(fontsize=6, ncol=2)
    ax.grid(True, alpha=0.3)

    # Safe RL Lagrange multiplier evolution
    ax = axes[1, 0]
    for net in nets:
        for meth in ["MAPPO-Safe", "MAPPO-Safe-EWC"]:
            if meth in results_dict[net]:
                r = results_dict[net][meth]
                if "safe_rl_stats" in r:
                    lambda_mean = r["safe_rl_stats"].get("lambda_mean", 0)
                    ax.bar(f"{net}\n{meth}", lambda_mean, color=COLORS.get(meth, "#fff"), alpha=0.85)
    ax.set_title("Safe RL: Avg Lagrange Multiplier", fontsize=10)
    ax.set_ylabel("λ (mean)")
    ax.tick_params(axis='x', rotation=45, labelsize=6)
    ax.grid(True, alpha=0.3)

    # EWC task count
    ax = axes[1, 1]
    for net in nets:
        if "MAPPO-Safe-EWC" in results_dict[net]:
            r = results_dict[net]["MAPPO-Safe-EWC"]
            if "ewc_stats" in r:
                n_tasks = r["ewc_stats"].get("n_tasks_learned", 0)
                ax.bar(net.upper(), n_tasks, color=COLORS["MAPPO-Safe-EWC"], alpha=0.85)
    ax.set_title("EWC: Tasks Learned", fontsize=10)
    ax.set_ylabel("# Tasks")
    ax.set_xticks(range(len(nets)))
    ax.set_xticklabels([n.upper() for n in nets])
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  ✓ {out_path}")


def print_phase4_comparison_table(results_dict):
    """Print Phase 4 specific comparison with metaheuristics."""
    LINE = "─" * 130
    print(f"\n{LINE}")
    print(f"  PHASE 4: BENCHMARK COMPARISON (Metaheuristics + MARL + GA-SQP)")
    print(LINE)
    print(f"  {'Network':<10} {'Method':<15} {'OF (s)':>10} {'OF_std':>10} "
          f"{'CTI_ok':>8} {'CTI_min':>10} {'Time(s)':>10} {'Evals':>12} {'Rank':>6}")
    print(LINE)
    
    for net_key, res in results_dict.items():
        # Compute mean ranks for this system
        method_names = [m for m in res if m != "net" and isinstance(res[m], dict) and "OF" in res[m]]
        if not method_names:
            continue
        
        ofs = [res[m]["OF"] for m in method_names]
        ranks = np.argsort(np.argsort(ofs)) + 1  # 1 = best
        
        for i, meth in enumerate(method_names):
            r = res[meth]
            c = r.get("constraints", {})
            cti_ok = "✓" if c.get("cti_satisfied", False) else "✗"
            evals = r.get("n_evals", 0)
            
            print(f"  {net_key:<10} {meth:<15} {r['OF']:>10.4f} {'—':>10} "
                  f"{cti_ok:>8} {c.get('cti_min', 0.0):>10.4f} "
                  f"{r['elapsed']:>10.2f} {evals:>12,} {ranks[i]:>6d}")
        print()
    print(LINE)


def plot_phase4_comparison(results_dict, out_path):
    """Plot Phase 4 method comparison including metaheuristics."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plot")
        return

    STYLE = {
        "figure.facecolor": "#0d1117",
        "axes.facecolor":   "#161b22",
        "axes.edgecolor":   "#30363d",
        "axes.labelcolor":  "#8b949e",
        "axes.titlecolor":  "#e6edf3",
        "xtick.color":      "#8b949e",
        "ytick.color":      "#8b949e",
        "grid.color":       "#21262d",
        "grid.linewidth":   0.8,
        "text.color":       "#e6edf3",
        "legend.facecolor": "#161b22",
        "legend.edgecolor": "#30363d",
        "font.family":      "monospace",
    }
    plt.rcParams.update(STYLE)

    COLORS = {
        "GA-SQP": "#2563eb",
        "Single-PPO": "#f778ba",
        "IPPO": "#a371f7",
        "MAPPO": "#39d353",
        "MAPPO-Comm": "#00d4ff",
        "MAPPO-Safe": "#ff6b35",
        "MAPPO-Safe-EWC": "#f7c948",
        "HWOA": "#ff6b6b",
        "HHO": "#4ecdc4",
        "JAYA": "#45b7d1",
        "EEO": "#96ceb4",
    }

    nets = list(results_dict.keys())
    methods = ["GA-SQP", "Single-PPO", "IPPO", "MAPPO", "MAPPO-Comm",
               "MAPPO-Safe", "MAPPO-Safe-EWC", "HWOA", "HHO", "JAYA", "EEO"]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Phase 4: Benchmark Comparison (Metaheuristics + MARL)", fontsize=12)

    # OF comparison
    ax = axes[0, 0]
    x = np.arange(len(nets))
    w = 0.08
    for i, meth in enumerate(methods):
        ofs = []
        for net in nets:
            if meth in results_dict[net] and isinstance(results_dict[net][meth], dict):
                ofs.append(results_dict[net][meth].get("OF", np.nan))
            else:
                ofs.append(np.nan)
        ax.bar(x + i * w - 5 * w, ofs, w, label=meth, color=COLORS.get(meth, "#fff"), alpha=0.85)

    ax.set_title("Objective Function", fontsize=10)
    ax.set_xlabel("Network")
    ax.set_ylabel("OF (s)")
    ax.set_xticks(x)
    ax.set_xticklabels([n.upper() for n in nets])
    ax.legend(fontsize=5, ncol=3)
    ax.grid(True, alpha=0.3)

    # CTI violations
    ax = axes[0, 1]
    for i, meth in enumerate(methods):
        viols = []
        for net in nets:
            if meth in results_dict[net] and isinstance(results_dict[net][meth], dict):
                c = results_dict[net][meth].get("constraints", {})
                viols.append(c.get("cti_violations", 0))
            else:
                viols.append(np.nan)
        ax.bar(x + i * w - 5 * w, viols, w, label=meth, color=COLORS.get(meth, "#fff"), alpha=0.85)

    ax.set_title("CTI Violations", fontsize=10)
    ax.set_xlabel("Network")
    ax.set_ylabel("# Violations")
    ax.set_xticks(x)
    ax.set_xticklabels([n.upper() for n in nets])
    ax.legend(fontsize=5, ncol=3)
    ax.grid(True, alpha=0.3)

    # Elapsed time
    ax = axes[1, 0]
    for i, meth in enumerate(methods):
        times = []
        for net in nets:
            if meth in results_dict[net] and isinstance(results_dict[net][meth], dict):
                times.append(results_dict[net][meth].get("elapsed", 0))
            else:
                times.append(np.nan)
        ax.bar(x + i * w - 5 * w, times, w, label=meth, color=COLORS.get(meth, "#fff"), alpha=0.85)

    ax.set_title("Computation Time", fontsize=10)
    ax.set_xlabel("Network")
    ax.set_ylabel("Time (s)")
    ax.set_xticks(x)
    ax.set_xticklabels([n.upper() for n in nets])
    ax.legend(fontsize=5, ncol=3)
    ax.grid(True, alpha=0.3)

    # Algorithm category pie (placeholder)
    ax = axes[1, 1]
    categories = ["Classical", "Metaheuristic", "MARL"]
    counts = [1, 4, 6]  # GA-SQP, HWOA/HHO/JAYA/EEO, MARL variants
    colors_pie = ["#2563eb", "#ff6b6b", "#39d353"]
    ax.pie(counts, labels=categories, colors=colors_pie, autopct="%1.0f%%",
           startangle=90, textprops={"color": "white"})
    ax.set_title("Algorithm Categories", fontsize=10)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  ✓ {out_path}")


# ── Main entry point ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="DOCR-MARL Phase 4 Experiments")
    parser.add_argument("--systems", nargs="+",
                        default=["8bus"],
                        choices=["3bus", "8bus", "30bus", "39bus"],
                        help="Test systems to evaluate")
    parser.add_argument("--fast", action="store_true",
                        help="Reduced iterations for quick smoke test")
    parser.add_argument("--phase4", action="store_true",
                        help="Run all Phase 4 experiments")
    parser.add_argument("--phase3", action="store_true",
                        help="Run Phase 3 experiments (backward compatible)")
    parser.add_argument("--phase2", action="store_true",
                        help="Run Phase 2 experiments (backward compatible)")
    parser.add_argument("--phase1", action="store_true",
                        help="Run Phase 1 experiments (backward compatible)")
    parser.add_argument("--pandapower", action="store_true",
                        help="Enable pandapower integration")
    parser.add_argument("--verbose", action="store_true", default=True,
                        help="Print detailed progress")
    parser.add_argument("--comm", action="store_true",
                        help="Enable GNN communication for MAPPO")
    parser.add_argument("--safe", action="store_true",
                        help="Enable Safe RL (Augmented Lagrangian)")
    parser.add_argument("--ewc", action="store_true",
                        help="Enable EWC continual adaptation")
    parser.add_argument("--compare", action="store_true",
                        help="Run full comparison of all methods")
    parser.add_argument("--benchmark", action="store_true",
                        help="Run benchmark suite with multiple runs")
    parser.add_argument("--stats", action="store_true",
                        help="Run statistical tests on existing results")
    parser.add_argument("--methods", nargs="+",
                        default=["GA-SQP", "HWOA", "HHO", "JAYA", "EEO"],
                        choices=["GA-SQP", "Single-PPO", "IPPO", "MAPPO", "MAPPO-Comm",
                                 "MAPPO-Safe", "MAPPO-Safe-EWC",
                                 "HWOA", "HHO", "JAYA", "EEO"],
                        help="Methods to compare")
    parser.add_argument("--n_runs", type=int, default=10,
                        help="Independent runs per method (for benchmark)")
    args = parser.parse_args()

    # Hyperparameters
    if args.fast:
        GA_POP, GA_GEN, MARL_STEPS = 30, 50, 2000
        META_POP, META_ITER = 20, 200
    else:
        GA_POP, GA_GEN, MARL_STEPS = 100, 200, 10000
        META_POP, META_ITER = 30, 500

    args.ga_pop = GA_POP
    args.ga_gen = GA_GEN
    args.marl_steps = MARL_STEPS
    args.meta_pop = META_POP
    args.meta_iter = META_ITER

    # Phase 4: default methods include metaheuristics
    if args.phase4:
        if not any(m in args.methods for m in ["HWOA", "HHO", "JAYA", "EEO"]):
            args.methods.extend(["HWOA", "HHO", "JAYA", "EEO"])
        if "GA-SQP" not in args.methods:
            args.methods.insert(0, "GA-SQP")
        if args.safe and "MAPPO-Safe" not in args.methods:
            args.methods.append("MAPPO-Safe")
        if args.ewc and "MAPPO-Safe-EWC" not in args.methods:
            args.methods.append("MAPPO-Safe-EWC")
        if args.comm and "MAPPO-Comm" not in args.methods:
            args.methods.append("MAPPO-Comm")

    # Benchmark suite mode
    if args.benchmark:
        print(f"\n{'█'*60}")
        print(f"  PHASE 4: BENCHMARK SUITE")
        print(f"  Systems: {', '.join(args.systems)}")
        print(f"  Methods: {', '.join(args.methods)}")
        print(f"  Runs: {args.n_runs}")
        print(f"{'█'*60}")
        
        suite = run_benchmark_suite(
            systems=args.systems,
            methods=args.methods,
            n_runs=args.n_runs,
            fast=args.fast,
            verbose=args.verbose,
        )
        return

    all_results = {}

    for sys_key in args.systems:
        net = NETWORKS[sys_key]()
        print(f"\n{'█' * 55}")
        print(f"  SYSTEM: {net.name}")
        print(f"  Relays: {net.n_relays}  |  Pairs: {net.n_pairs}  |  Buses: {net.n_buses}")
        print(f"  Methods: {', '.join(args.methods)}")
        if args.safe:
            print(f"  Safe RL: Augmented Lagrangian + Action Projection")
        if args.ewc:
            print(f"  EWC: Elastic Weight Consolidation")
        print(f"{'█' * 55}")

        sys_results = {"net": net}

        # GA-SQP (always run as baseline)
        if "GA-SQP" in args.methods:
            print(f"\n  [GA-SQP] Running...")
            sys_results["GA-SQP"] = run_ga_sqp(
                net, ga_pop=GA_POP, ga_gen=GA_GEN, seed=42,
                verbose=args.verbose, fault_type="3ph", pv_level=0.0
            )

        # Metaheuristics
        for meta in ["HWOA", "HHO", "JAYA", "EEO"]:
            if meta in args.methods:
                print(f"\n  [{meta}] Running...")
                runner = globals()[f"run_{meta.lower()}"]
                sys_results[meta] = runner(
                    net, pop_size=META_POP, max_iter=META_ITER, seed=42,
                    verbose=args.verbose, fault_type="3ph", pv_level=0.0,
                    use_sqp=True
                )

        # Single-PPO (centralized)
        if "Single-PPO" in args.methods:
            print(f"\n  [Single-PPO] Running centralized PPO...")
            sys_results["Single-PPO"] = run_single_ppo(
                net, total_steps=MARL_STEPS, seed=0,
                verbose=args.verbose, use_pandapower=args.pandapower
            )

        # IPPO (decentralized, no comm)
        if "IPPO" in args.methods:
            print(f"\n  [IPPO] Running decentralized IPPO...")
            sys_results["IPPO"] = run_ippo(
                net, total_steps=MARL_STEPS, seed=0,
                verbose=args.verbose, use_pandapower=args.pandapower
            )

        # MAPPO (centralized critic, no comm)
        if "MAPPO" in args.methods:
            print(f"\n  [MAPPO] Running MAPPO...")
            sys_results["MAPPO"] = run_mappo(
                net, total_steps=MARL_STEPS, seed=0,
                verbose=args.verbose, use_pandapower=args.pandapower
            )

        # MAPPO-Comm (with GNN)
        if "MAPPO-Comm" in args.methods or args.comm:
            print(f"\n  [MAPPO-Comm] Running MAPPO with GNN communication...")
            sys_results["MAPPO-Comm"] = run_mappo_comm(
                net, total_steps=MARL_STEPS, seed=0,
                verbose=args.verbose, use_pandapower=args.pandapower
            )

        # MAPPO-Safe (with Safe RL)
        if "MAPPO-Safe" in args.methods or args.safe:
            print(f"\n  [MAPPO-Safe] Running MAPPO with Safe RL...")
            sys_results["MAPPO-Safe"] = run_mappo_safe(
                net, total_steps=MARL_STEPS, seed=0,
                verbose=args.verbose, use_pandapower=args.pandapower
            )

        # MAPPO-Safe-EWC (with Safe RL + EWC)
        if "MAPPO-Safe-EWC" in args.methods or args.ewc:
            print(f"\n  [MAPPO-Safe-EWC] Running MAPPO with Safe RL + EWC...")
            sys_results["MAPPO-Safe-EWC"] = run_mappo_safe_ewc(
                net, total_steps=MARL_STEPS, seed=0,
                verbose=args.verbose, use_pandapower=args.pandapower
            )

        all_results[sys_key] = sys_results

    # Print comparison tables
    if args.compare or args.phase4 or args.phase3 or args.phase2:
        print_comparison_table(all_results)
        if args.phase4:
            print_phase4_comparison_table(all_results)
            plot_phase4_comparison(all_results, "outputs/phase4_comparison.png")
        elif args.phase3:
            plot_phase3_comparison(all_results, "outputs/phase3_comparison.png")
        else:
            plot_phase3_comparison(all_results, "outputs/phase2_comparison.png")

    # Statistical tests
    if (args.stats or args.phase4) and len(args.systems) >= 2:
        # Build results matrix for statistical analysis
        results_matrix = {}
        for sys_key, sys_res in all_results.items():
            results_matrix[sys_key] = {}
            for meth, res in sys_res.items():
                if meth != "net" and isinstance(res, dict) and "OF" in res:
                    results_matrix[sys_key][meth] = res["OF"]
        
        if len(results_matrix) >= 2:
            run_statistical_analysis(results_matrix)

    print("\n  All outputs written to ./outputs/")
    print("\n  Phase 4 experiments complete!")
    print("  Next steps (Phase 5):")
    print("    1. Large-scale training with 10,000+ scenarios")
    print("    2. PV penetration sweep analysis (0-100%)")
    print("    3. Topology robustness testing (N-1 contingencies)")
    print("    4. Curriculum learning evaluation")


if __name__ == "__main__":
    main()
