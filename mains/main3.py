"""
main.py
───────
Phase 3 Entry Point for DOCR-MARL Thesis.

Runs comprehensive experiments comparing:
1. GA-SQP (classical control baseline)
2. Single-PPO (centralized RL baseline)
3. IPPO (decentralized RL, no communication)
4. MAPPO (centralized critic, no communication)
5. MAPPO-Comm (MAPPO + GNN communication)
6. MAPPO-Safe (MAPPO + Augmented Lagrangian + Action Projection)
7. MAPPO-Safe-EWC (MAPPO-Safe + EWC continual adaptation)

Usage:
    python main.py --phase3                    # all Phase 3 experiments
    python main.py --phase3 --systems 8bus     # single system
    python main.py --phase3 --fast             # reduced iterations
    python main.py --phase3 --safe             # enable Safe RL
    python main.py --phase3 --ewc              # enable EWC
    python main.py --phase3 --compare          # full comparison table
"""

import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
from network.network_data import NETWORKS
from network.relay_physics import check_constraints, FAULT_TYPES
from baselines.ga_sqp import run_ga_sqp

os.makedirs("outputs", exist_ok=True)


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

    # Add safe RL stats
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
    # built but base_trainer.train() bypassed it entirely — register_task
    # was never called and get_adaptation_stats() always reported 0 tasks
    # learned). A lone task has nothing to protect against forgetting —
    # that's expected, not a bug — but the machinery should reflect that
    # it actually ran rather than looking identical to never having been
    # invoked at all.
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

    # Add safe RL + EWC stats
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


def print_comparison_table(results_dict):
    """Print formatted comparison table across all methods."""
    LINE = "─" * 130
    print(f"\n{LINE}")
    print(f"  {'Network':<12} {'Method':<18} {'OF (s)':>9} "
          f"{'CTI viol':>9} {'CTI min':>10} {'Time (s)':>10} "
          f"{'Comm BW':>10} {'Safe λ':>8} {'EWC Tasks':>10} {'Improvement':>12}")
    print(LINE)

    for net_key, res in results_dict.items():
        ga_of = res["GA-SQP"]["OF"]
        for meth in ("GA-SQP", "Single-PPO", "IPPO", "MAPPO", "MAPPO-Comm", "MAPPO-Safe", "MAPPO-Safe-EWC"):
            if meth not in res:
                continue
            r = res[meth]
            c = r.get("constraints", check_constraints(res.get("net", res["GA-SQP"].get("net")),
                                                        r["TMS"], r["PS"], fault_type="3ph"))
            comm_bw = r.get("comm_bandwidth", 0.0)
            safe_lambda = r.get("safe_rl_stats", {}).get("lambda_mean", 0.0)
            ewc_tasks = r.get("ewc_stats", {}).get("n_tasks_learned", 0)
            imp = (ga_of - r["OF"]) / ga_of * 100 if ga_of > 0 else 0.0

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


def main():
    parser = argparse.ArgumentParser(description="DOCR-MARL Phase 3 Experiments")
    parser.add_argument("--systems", nargs="+",
                        default=["8bus"],
                        choices=["3bus", "8bus", "30bus", "39bus"],
                        help="Test systems to evaluate")
    parser.add_argument("--fast", action="store_true",
                        help="Reduced iterations for quick smoke test")
    parser.add_argument("--phase3", action="store_true",
                        help="Run all Phase 3 experiments")
    parser.add_argument("--phase2", action="store_true",
                        help="Run Phase 2 experiments")
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
    parser.add_argument("--methods", nargs="+",
                        default=["GA-SQP", "IPPO", "MAPPO"],
                        choices=["GA-SQP", "Single-PPO", "IPPO", "MAPPO", "MAPPO-Comm",
                                 "MAPPO-Safe", "MAPPO-Safe-EWC"],
                        help="Methods to compare")
    args = parser.parse_args()

    # Hyperparameters
    if args.fast:
        GA_POP, GA_GEN, MARL_STEPS = 30, 50, 2000
    else:
        GA_POP, GA_GEN, MARL_STEPS = 100, 200, 10000

    args.ga_pop = GA_POP
    args.ga_gen = GA_GEN
    args.marl_steps = MARL_STEPS

    # Phase 3: default methods include Safe RL and EWC if enabled
    if args.phase3:
        if "MAPPO-Safe" not in args.methods and args.safe:
            args.methods.append("MAPPO-Safe")
        if "MAPPO-Safe-EWC" not in args.methods and args.ewc:
            args.methods.append("MAPPO-Safe-EWC")
        if "MAPPO-Comm" not in args.methods and args.comm:
            args.methods.append("MAPPO-Comm")

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

    # Print comparison table
    if args.compare or args.phase3 or args.phase2:
        print_comparison_table(all_results)
        if args.phase3:
            plot_phase3_comparison(all_results, "outputs/phase3_comparison.png")
        else:
            plot_phase3_comparison(all_results, "outputs/phase2_comparison.png")

    print("\n  All outputs written to ./outputs/")
    print("\n  Phase 3 experiments complete!")
    print("  Next steps (Phase 4):")
    print("    1. Implement benchmark comparison suite (HWOA, HHO, JAYA, EEO)")
    print("    2. Add statistical validation (Friedman, Nemenyi, Cliff's delta)")
    print("    3. Create deployment pipeline (ONNX + IEC 61850)")


if __name__ == "__main__":
    main()
