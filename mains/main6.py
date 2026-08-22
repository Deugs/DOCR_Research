
"""
main.py
──────    # Phase 7: Robustness & Deployment
    if args.phase7 or args.robustness:
        if not args.methods:
            args.methods = ["GA-SQP", "JAYA"]
        return run_phase7_robustness(args)

    if args.deployment:
        if not args.methods:
            args.methods = ["MAPPO-Comm", "MAPPO-Safe-EWC"]
        return run_phase7_deployment(args)

    # Phase 8: Statistical Validation & Thesis
    if args.phase8 or args.validate:
        return run_phase8_validation(args)

─
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
    
    print(f"\\n{'='*70}")
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
    print(f"\\n{LINE}")
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
    print(f"\\n{LINE}")
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


# ── Phase 5: Large-Scale Training & Scenarios ───────────────────────────────

def run_phase5_experiments(args):
    """Run Phase 5 large-scale training experiments."""
    from experiments.scenario_generator import generate_scenario_dataset
    from experiments.training_pipeline import TrainingPipeline, create_experiment_matrix
    from experiments.results_aggregator import ResultsAggregator
    from evaluation.large_scale_eval import LargeScaleEvaluator

    print(f"\n{'█'*70}")
    print(f"  PHASE 5: LARGE-SCALE TRAINING & SCENARIOS")
    print(f"{'█'*70}")

    # Step 1: Generate scenarios
    if not args.skip_scenarios:
        print("\n  [Step 1] Generating scenario dataset...")
        train_path, val_path = generate_scenario_dataset(
            output_dir=os.path.join(args.output_dir, "scenarios"),
            n_train=args.n_train_scenarios,
            n_val=args.n_val_scenarios,
            seed=args.seed,
        )

    # Step 2: Create experiment configurations
    print("\n  [Step 2] Creating experiment matrix...")
    configs = create_experiment_matrix(
        systems=args.systems,
        methods=args.methods,
        pv_levels=args.pv_levels if args.pv_levels else [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        fault_types=args.fault_types if args.fault_types else ["3ph"],
        n_runs=args.n_runs,
        fast=args.fast,
    )

    # Step 3: Run training pipeline
    print("\n  [Step 3] Running training pipeline...")
    pipeline = TrainingPipeline(
        configs=configs,
        n_workers=args.n_workers,
        checkpoint_dir=os.path.join(args.output_dir, "checkpoints"),
        verbose=args.verbose,
    )
    results = pipeline.run_all()

    # Step 4: Save raw results
    results_path = os.path.join(args.output_dir, "phase5_results.json")
    pipeline.save_results(results_path)

    # Step 5: Aggregate and analyze
    print("\n  [Step 4] Aggregating results...")
    aggregator = ResultsAggregator(results)
    aggregator.aggregate()
    aggregator.print_summary_table()

    # Generate plots
    plot_dir = os.path.join(args.output_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    aggregator.plot_pv_sweep(os.path.join(plot_dir, "pv_sweep.png"))
    aggregator.plot_curriculum_progress(os.path.join(plot_dir, "curriculum_progress.png"))
    aggregator.generate_latex_table(os.path.join(plot_dir, "results_table.tex"))
    aggregator.save_summary_json(os.path.join(args.output_dir, "summary.json"))

    # Step 6: Comprehensive evaluation
    print("\n  [Step 5] Computing evaluation metrics...")
    evaluator = LargeScaleEvaluator(results)
    report = evaluator.generate_report()

    report_path = os.path.join(args.output_dir, "evaluation_report.txt")
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"  ✓ Evaluation report: {report_path}")

    print(f"\n{'█'*70}")
    print(f"  PHASE 5 COMPLETE")
    print(f"  Results saved to: {args.output_dir}")
    print(f"{'█'*70}")

    return results


def run_pv_sweep_analysis(args):
    """Run PV penetration sweep analysis."""
    from experiments.training_pipeline import TrainingPipeline, TrainingConfig
    from experiments.results_aggregator import ResultsAggregator

    print(f"\n{'█'*70}")
    print(f"  PHASE 5: PV PENETRATION SWEEP ANALYSIS")
    print(f"{'█'*70}")

    configs = []
    for system in args.systems:
        for method in args.methods:
            for pv_level in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
                for run in range(args.n_runs):
                    configs.append(TrainingConfig(
                        method=method,
                        system=system,
                        pv_level=pv_level,
                        fault_type="3ph",
                        n_runs=args.n_runs,
                        seed_offset=run * 1000,
                        ga_pop=30 if args.fast else 100,
                        ga_gen=50 if args.fast else 200,
                        meta_pop=20 if args.fast else 30,
                        meta_iter=200 if args.fast else 500,
                        marl_steps=5000 if args.fast else 30000,
                        use_safe_rl="Safe" in method,
                        use_ewc="EWC" in method,
                        use_comm="Comm" in method,
                    ))

    pipeline = TrainingPipeline(configs, n_workers=args.n_workers, verbose=args.verbose)
    results = pipeline.run_all()

    # Analyze
    aggregator = ResultsAggregator(results)
    aggregator.plot_pv_sweep(os.path.join(args.output_dir, "pv_sweep.png"))
    aggregator.save_summary_json(os.path.join(args.output_dir, "pv_sweep_summary.json"))

    print(f"\n  PV Sweep Analysis Complete!")
    print(f"  Results: {args.output_dir}")

    return results


def run_curriculum_evaluation(args):
    """Run curriculum learning evaluation."""
    from experiments.training_pipeline import TrainingPipeline, TrainingConfig
    from experiments.results_aggregator import ResultsAggregator

    print(f"\n{'█'*70}")
    print(f"  PHASE 5: CURRICULUM LEARNING EVALUATION")
    print(f"{'█'*70}")

    configs = []
    for system in args.systems:
        for stage, pv_level in enumerate([0.0, 0.2, 0.4, 0.6, 0.8, 1.0]):
            for run in range(args.n_runs):
                configs.append(TrainingConfig(
                    method="MAPPO-Safe-EWC",
                    system=system,
                    pv_level=pv_level,
                    fault_type="3ph",
                    n_runs=args.n_runs,
                    seed_offset=run * 1000,
                    use_curriculum=True,
                    curriculum_stage=stage,
                    use_ewc=True,
                    use_safe_rl=True,
                    marl_steps=5000 if args.fast else 30000,
                ))

    pipeline = TrainingPipeline(configs, n_workers=args.n_workers, verbose=args.verbose)
    results = pipeline.run_all()

    aggregator = ResultsAggregator(results)
    aggregator.plot_curriculum_progress(os.path.join(args.output_dir, "curriculum_progress.png"))
    aggregator.save_summary_json(os.path.join(args.output_dir, "curriculum_summary.json"))

    print(f"\n  Curriculum Evaluation Complete!")

    return results


def run_topology_robustness(args):
    """Run topology robustness testing (N-1 contingencies)."""
    from experiments.training_pipeline import TrainingPipeline, TrainingConfig
    from experiments.results_aggregator import ResultsAggregator

    print(f"\n{'█'*70}")
    print(f"  PHASE 5: TOPOLOGY ROBUSTNESS TESTING")
    print(f"{'█'*70}")

    configs = []
    for system in args.systems:
        for method in args.methods:
            # Baseline (no outage)
            for run in range(args.n_runs):
                configs.append(TrainingConfig(
                    method=method,
                    system=system,
                    pv_level=0.5,
                    fault_type="3ph",
                    topology_changes={},
                    n_runs=args.n_runs,
                    seed_offset=run * 1000,
                ))
            # With N-1 outage
            for run in range(args.n_runs):
                configs.append(TrainingConfig(
                    method=method,
                    system=system,
                    pv_level=0.5,
                    fault_type="3ph",
                    topology_changes={"type": "n1_outage", "line_idx": run % 5, "outage_factor": 0.7},
                    n_runs=args.n_runs,
                    seed_offset=run * 1000 + 500,
                ))

    pipeline = TrainingPipeline(configs, n_workers=args.n_workers, verbose=args.verbose)
    results = pipeline.run_all()

    aggregator = ResultsAggregator(results)
    robustness = aggregator.get_topology_robustness()

    print("\n  Topology Robustness Results:")
    for system, metrics in robustness.items():
        print(f"    {system}:")
        print(f"      Baseline OF: {metrics['without_outage_of_mean']:.4f}")
        print(f"      With N-1 OF: {metrics['with_outage_of_mean']:.4f}")
        print(f"      Degradation: {metrics['degradation_percent']:.2f}%")

    aggregator.save_summary_json(os.path.join(args.output_dir, "robustness_summary.json"))

    return results



# ── Phase 7: Robustness & Deployment ────────────────────────────────────────

def run_phase7_robustness(args):
    """Run Phase 7 robustness testing experiments."""
    from deployment.robustness_tests import RobustnessTester, RobustnessConfig
    from network.network_data import NETWORKS
    from baselines.metaheuristics import run_jaya
    from baselines.ga_sqp import run_ga_sqp

    print(f"\n{'█'*70}")
    print(f"  PHASE 7: ROBUSTNESS & DEPLOYMENT TESTING")
    print(f"{'█'*70}")

    results = {}

    for sys_key in args.systems:
        net = NETWORKS[sys_key]()
        print(f"\n  System: {net.name} ({net.n_relays} relays)")

        tester = RobustnessTester(RobustnessConfig(seed=args.seed))

        # Test GA-SQP
        if "GA-SQP" in args.methods:
            print(f"\n  Testing GA-SQP robustness...")
            def ga_runner(net, seed=42, config=None):
                return run_ga_sqp(net, ga_pop=30, ga_gen=50, seed=seed, verbose=False)

            tester.test_method("GA-SQP", ga_runner, net, n_runs=args.n_runs, verbose=args.verbose)
            results[f"{sys_key}_GA-SQP"] = tester.compute_degradation_metrics("GA-SQP")
            print(tester.generate_report("GA-SQP"))

        # Test JAYA (fast metaheuristic)
        if "JAYA" in args.methods:
            print(f"\n  Testing JAYA robustness...")
            def jaya_runner(net, seed=42, config=None):
                return run_jaya(net, n_pop=20, max_iter=100, seed=seed, verbose=False, use_sqp=False)

            tester.test_method("JAYA", jaya_runner, net, n_runs=args.n_runs, verbose=args.verbose)
            results[f"{sys_key}_JAYA"] = tester.compute_degradation_metrics("JAYA")
            print(tester.generate_report("JAYA"))

    # Save results
    import json
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "phase7_robustness.json"), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'█'*70}")
    print(f"  PHASE 7 COMPLETE")
    print(f"  Results saved to: {args.output_dir}/phase7_robustness.json")
    print(f"{'█'*70}")

    return results


def run_phase7_deployment(args):
    """Run Phase 7 deployment pipeline experiments."""
    from deployment.inference_engine import InferenceEngine, InferenceConfig, export_marl_deployment
    from deployment.hils_bridge import HILSBridge, HILSConfig
    from deployment.byzantine_resilience import ByzantineResilientAggregator
    from network.network_data import NETWORKS

    print(f"\n{'█'*70}")
    print(f"  PHASE 7: DEPLOYMENT PIPELINE")
    print(f"{'█'*70}")

    for sys_key in args.systems:
        net = NETWORKS[sys_key]()
        print(f"\n  System: {net.name} ({net.n_relays} relays)")

        # 1. Inference latency benchmark
        print(f"\n  [1] Inference Latency Benchmark")
        config = InferenceConfig(target_latency_ms=100.0, benchmark_runs=100)
        engine = InferenceEngine("dummy_model.onnx", config)

        if engine.initialize():
            stats = engine.benchmark(n_runs=100)
            print(f"    Mean latency: {stats['mean_ms']:.2f} ms")
            print(f"    P99 latency: {stats['p99_ms']:.2f} ms")
            print(f"    Meets target (<100ms): {'YES ✓' if stats['meets_target'] else 'NO ✗'}")
            print(engine.get_latency_report())

        # 2. HILS bridge test
        print(f"\n  [2] HILS Bridge Test")
        hils_config = HILSConfig(simulation_time_step_ms=4.0)
        hils = HILSBridge(hils_config)

        if hils.start():
            settings = {i: (0.5, 1.2) for i in range(min(5, net.n_relays))}
            for step in range(20):
                result = hils.run_step(settings)

            stats = hils.get_statistics()
            print(f"    Cycles: {stats['cycles']}, Errors: {stats['errors']}")
            print(hils.generate_report())
            hils.stop()

        # 3. Byzantine resilience test
        print(f"\n  [3] Byzantine Resilience Test")
        agg = ByzantineResilientAggregator(n_agents=net.n_relays)

        # Simulate normal operation. Uses an explicitly seeded Generator
        # (not the legacy global np.random state, which every other
        # stochastic path in this file already avoids per CLAUDE.md's
        # seeding convention) so this demo block is reproducible run-to-run.
        byz_rng = np.random.default_rng(args.seed)
        for i in range(net.n_relays):
            neighbors = byz_rng.normal(1.0, 0.1, min(4, net.n_relays - 1))
            val, meta = agg.aggregate(i, 1.0 + i * 0.01, neighbors.tolist())

        # Inject Byzantine agent
        for _ in range(10):
            val, meta = agg.aggregate(2, 10.0, [1.0, 1.1, 1.2, 1.15])

        health = agg.get_system_health()
        print(f"    System status: {health['status']}")
        print(f"    Suspicious agents: {health['trust']['suspicious_agents']}")

    print(f"\n{'█'*70}")
    print(f"  PHASE 7 DEPLOYMENT COMPLETE")
    print(f"{'█'*70}")


# ── Phase 8: Statistical Validation & Thesis ─────────────────────────────────

def run_phase8_validation(args):
    """Run Phase 8 thesis validation and reproducibility checks."""
    from evaluation.thesis_validation import ThesisValidator, ReproducibilityChecklist
    from evaluation.thesis_validation import generate_dockerfile, generate_github_actions

    print(f"\n{'█'*70}")
    print(f"  PHASE 8: STATISTICAL VALIDATION & THESIS")
    print(f"{'█'*70}")

    # 1. Validate all novelty claims
    print(f"\n  [1] Validating Novelty Claims")
    validator = ThesisValidator(results_dir=args.output_dir)

    # Load results if available
    results_file = os.path.join(args.output_dir, "phase5_results.json")
    if os.path.exists(results_file):
        validator.load_results()
    else:
        print(f"    Warning: No Phase 5 results found at {results_file}")
        print(f"    Using placeholder validation (run --phase5 first for full validation)")

    results = validator.validate_all_claims()

    # Print report
    report = validator.generate_validation_report()
    print(report)

    # Save report
    report_path = os.path.join(args.output_dir, "phase8_validation_report.txt")
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"  ✓ Validation report: {report_path}")

    # 2. Generate LaTeX tables
    print(f"\n  [2] Generating LaTeX Tables")
    latex_path = os.path.join(args.output_dir, "phase8_tables.tex")
    validator.generate_latex_tables(latex_path)

    # 3. Reproducibility checklist
    print(f"\n  [3] Reproducibility Checklist")
    checklist = ReproducibilityChecklist(repo_dir=".")
    checklist_report = checklist.generate_report()
    print(checklist_report)

    checklist_path = os.path.join(args.output_dir, "phase8_reproducibility.txt")
    with open(checklist_path, 'w') as f:
        f.write(checklist_report)
    print(f"  ✓ Reproducibility report: {checklist_path}")

    # 4. Generate Docker and CI files
    print(f"\n  [4] Generating Deployment Files")
    generate_dockerfile(os.path.join(args.output_dir, "Dockerfile"))
    generate_github_actions(os.path.join(args.output_dir, ".github/workflows/ci.yml"))
    print(f"  ✓ Dockerfile and CI config generated")

    # 5. Summary
    passed = sum(1 for r in results if r.passed)
    total = len(results)

    print(f"\n{'█'*70}")
    print(f"  PHASE 8 COMPLETE")
    print(f"  Novelty Claims: {passed}/{total} validated")
    print(f"  Output: {args.output_dir}")
    print(f"{'█'*70}")

    return results

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

    # Phase 5 arguments
    parser.add_argument("--phase5", action="store_true",
                        help="Run Phase 5 large-scale experiments")
    parser.add_argument("--pv-sweep", action="store_true",
                        help="Run PV penetration sweep analysis")
    parser.add_argument("--curriculum-eval", action="store_true",
                        help="Run curriculum learning evaluation")
    parser.add_argument("--topology-robustness", action="store_true",
                        help="Run topology robustness testing")
    parser.add_argument("--skip-scenarios", action="store_true",
                        help="Skip scenario generation (use existing)")
    parser.add_argument("--n-train-scenarios", type=int, default=10000,
                        help="Number of training scenarios to generate")
    parser.add_argument("--n-val-scenarios", type=int, default=2000,
                        help="Number of validation scenarios to generate")
    parser.add_argument("--pv-levels", nargs="+", type=float,
                        default=None,
                        help="PV penetration levels to test")
    parser.add_argument("--fault-types", nargs="+",
                        default=None,
                        choices=["3ph", "LL", "1ph", "2ph-g"],
                        help="Fault types to test")
    parser.add_argument("--output-dir", type=str, default="outputs/phase5",
                        help="Output directory for Phase 5 results")
    parser.add_argument("--n-workers", type=int, default=-1,
                        help="Number of parallel workers (-1 = auto)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")

    # Phase 7 arguments
    parser.add_argument("--phase7", action="store_true",
                        help="Run Phase 7 robustness & deployment experiments")
    parser.add_argument("--robustness", action="store_true",
                        help="Run robustness testing only")
    parser.add_argument("--deployment", action="store_true",
                        help="Run deployment pipeline only")

    # Phase 8 arguments
    parser.add_argument("--phase8", action="store_true",
                        help="Run Phase 8 thesis validation & reproducibility")
    parser.add_argument("--validate", action="store_true",
                        help="Run novelty claim validation only")

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

    # Phase 5: Large-scale experiments
    if args.phase5:
        if not any(m in args.methods for m in ["MAPPO-Safe", "MAPPO-Safe-EWC", "HWOA", "JAYA"]):
            args.methods = ["GA-SQP", "MAPPO", "MAPPO-Safe", "MAPPO-Safe-EWC", "HWOA", "JAYA"]
        return run_phase5_experiments(args)

    if args.pv_sweep:
        if not args.methods:
            args.methods = ["GA-SQP", "MAPPO", "MAPPO-Safe"]
        return run_pv_sweep_analysis(args)

    if args.curriculum_eval:
        args.methods = ["MAPPO-Safe-EWC"]
        return run_curriculum_evaluation(args)

    if args.topology_robustness:
        if not args.methods:
            args.methods = ["GA-SQP", "MAPPO-Safe"]
        return run_topology_robustness(args)

    # Benchmark suite mode
    if args.benchmark:
        print(f"\\n{'█'*60}")
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
        print(f"\\n{'█' * 55}")
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
            print(f"\\n  [GA-SQP] Running...")
            sys_results["GA-SQP"] = run_ga_sqp(
                net, ga_pop=GA_POP, ga_gen=GA_GEN, seed=42,
                verbose=args.verbose, fault_type="3ph", pv_level=0.0
            )

        # Metaheuristics
        for meta in ["HWOA", "HHO", "JAYA", "EEO"]:
            if meta in args.methods:
                print(f"\\n  [{meta}] Running...")
                runner = globals()[f"run_{meta.lower()}"]
                sys_results[meta] = runner(
                    net, pop_size=META_POP, max_iter=META_ITER, seed=42,
                    verbose=args.verbose, fault_type="3ph", pv_level=0.0,
                    use_sqp=True
                )

        # Single-PPO (centralized)
        if "Single-PPO" in args.methods:
            print(f"\\n  [Single-PPO] Running centralized PPO...")
            sys_results["Single-PPO"] = run_single_ppo(
                net, total_steps=MARL_STEPS, seed=0,
                verbose=args.verbose, use_pandapower=args.pandapower
            )

        # IPPO (decentralized, no comm)
        if "IPPO" in args.methods:
            print(f"\\n  [IPPO] Running decentralized IPPO...")
            sys_results["IPPO"] = run_ippo(
                net, total_steps=MARL_STEPS, seed=0,
                verbose=args.verbose, use_pandapower=args.pandapower
            )

        # MAPPO (centralized critic, no comm)
        if "MAPPO" in args.methods:
            print(f"\\n  [MAPPO] Running MAPPO...")
            sys_results["MAPPO"] = run_mappo(
                net, total_steps=MARL_STEPS, seed=0,
                verbose=args.verbose, use_pandapower=args.pandapower
            )

        # MAPPO-Comm (with GNN)
        if "MAPPO-Comm" in args.methods or args.comm:
            print(f"\\n  [MAPPO-Comm] Running MAPPO with GNN communication...")
            sys_results["MAPPO-Comm"] = run_mappo_comm(
                net, total_steps=MARL_STEPS, seed=0,
                verbose=args.verbose, use_pandapower=args.pandapower
            )

        # MAPPO-Safe (with Safe RL)
        if "MAPPO-Safe" in args.methods or args.safe:
            print(f"\\n  [MAPPO-Safe] Running MAPPO with Safe RL...")
            sys_results["MAPPO-Safe"] = run_mappo_safe(
                net, total_steps=MARL_STEPS, seed=0,
                verbose=args.verbose, use_pandapower=args.pandapower
            )

        # MAPPO-Safe-EWC (with Safe RL + EWC)
        if "MAPPO-Safe-EWC" in args.methods or args.ewc:
            print(f"\\n  [MAPPO-Safe-EWC] Running MAPPO with Safe RL + EWC...")
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

    print("\\n  All outputs written to ./outputs/")
    print("\\n  Phase 4 experiments complete!")
    print("  Next steps (Phase 5):")
    print("    1. Large-scale training with 10,000+ scenarios")
    print("    2. PV penetration sweep analysis (0-100%)")
    print("    3. Topology robustness testing (N-1 contingencies)")
    print("    4. Curriculum learning evaluation")


if __name__ == "__main__":
    main()
