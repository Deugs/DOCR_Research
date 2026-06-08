"""
main.py
───────
Phase 2 Entry Point for DOCR-MARL Thesis.

Runs comprehensive experiments comparing:
1. GA-SQP (classical control baseline)
2. Single-PPO (centralized RL baseline)
3. IPPO (decentralized RL, no communication)
4. MAPPO (centralized critic, no communication)
5. MAPPO-Comm (MAPPO + GNN communication)

Usage:
    python main.py --phase2                    # all Phase 2 experiments
    python main.py --phase2 --systems 8bus     # single system
    python main.py --phase2 --fast             # reduced iterations
    python main.py --phase2 --comm             # enable GNN communication
    python main.py --phase2 --compare          # full comparison table
"""

import sys, os, argparse
sys.path.insert(0, os.path.dirname(__file__))

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


def run_single_ppo(net, total_steps=30000, seed=0, verbose=True, use_pandapower=False):
    """Run centralized Single-PPO baseline."""
    from baselines.single_agent_ppo import run_single_ppo as _run_single_ppo
    return _run_single_ppo(net, total_steps=total_steps, seed=seed, verbose=verbose,
                           use_pandapower=use_pandapower, hidden_actor=128, hidden_critic=128)


def print_comparison_table(results_dict):
    """Print formatted comparison table across all methods."""
    LINE = "─" * 120
    print(f"\n{LINE}")
    print(f"  {'Network':<12} {'Method':<18} {'OF (s)':>9} "
          f"{'CTI viol':>9} {'CTI min':>10} {'Time (s)':>10} "
          f"{'Comm BW':>10} {'Improvement':>12}")
    print(LINE)

    for net_key, res in results_dict.items():
        ga_of = res["GA-SQP"]["OF"]
        for meth in ("GA-SQP", "Single-PPO", "IPPO", "MAPPO", "MAPPO-Comm"):
            if meth not in res:
                continue
            r = res[meth]
            c = r.get("constraints", check_constraints(res.get("net", res["GA-SQP"].get("net")),
                                                        r["TMS"], r["PS"], fault_type="3ph"))
            comm_bw = r.get("comm_bandwidth", 0.0)
            imp = (ga_of - r["OF"]) / ga_of * 100 if ga_of > 0 else 0.0

            print(f"  {net_key:<12} {meth:<18} "
                  f"{r['OF']:>9.4f} {c.get('cti_violations', 0):>9d} "
                  f"{c.get('cti_min', 0.0):>10.4f} {r['elapsed']:>10.2f} "
                  f"{comm_bw:>9.3f}kbps {imp:>+11.2f}%")
        print()
    print(LINE)


def plot_phase2_comparison(results_dict, out_path):
    """Plot Phase 2 method comparison."""
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
    }

    nets = list(results_dict.keys())
    methods = ["GA-SQP", "Single-PPO", "IPPO", "MAPPO", "MAPPO-Comm"]

    # Reconstruct network objects from names
    net_objects = {}
    for net_key in nets:
        if "GA-SQP" in results_dict[net_key]:
            net_name = results_dict[net_key]["GA-SQP"].get("network", "")
            for key, fn in NETWORKS.items():
                if fn().name == net_name:
                    net_objects[net_key] = fn()
                    break

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Phase 2: Method Comparison", fontsize=12)

    # OF comparison
    ax = axes[0]
    x = np.arange(len(nets))
    w = 0.15
    for i, meth in enumerate(methods):
        ofs = []
        for net in nets:
            if meth in results_dict[net]:
                ofs.append(results_dict[net][meth]["OF"])
            else:
                ofs.append(np.nan)
        ax.bar(x + i * w - 2 * w, ofs, w, label=meth, color=COLORS.get(meth, "#fff"), alpha=0.85)

    ax.set_title("Objective Function", fontsize=10)
    ax.set_xlabel("Network")
    ax.set_ylabel("OF (s)")
    ax.set_xticks(x)
    ax.set_xticklabels([n.upper() for n in nets])
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # CTI violations
    ax = axes[1]
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
        ax.bar(x + i * w - 2 * w, viols, w, label=meth, color=COLORS.get(meth, "#fff"), alpha=0.85)

    ax.set_title("CTI Violations", fontsize=10)
    ax.set_xlabel("Network")
    ax.set_ylabel("# Violations")
    ax.set_xticks(x)
    ax.set_xticklabels([n.upper() for n in nets])
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Computation time
    ax = axes[2]
    for i, meth in enumerate(methods):
        times = []
        for net in nets:
            if meth in results_dict[net]:
                times.append(results_dict[net][meth]["elapsed"])
            else:
                times.append(np.nan)
        ax.bar(x + i * w - 2 * w, times, w, label=meth, color=COLORS.get(meth, "#fff"), alpha=0.85)

    ax.set_title("Computation Time", fontsize=10)
    ax.set_xlabel("Network")
    ax.set_ylabel("Seconds")
    ax.set_xticks(x)
    ax.set_xticklabels([n.upper() for n in nets])
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  ✓ {out_path}")


def main():
    parser = argparse.ArgumentParser(description="DOCR-MARL Phase 2 Experiments")
    parser.add_argument("--systems", nargs="+",
                        default=["8bus"],
                        choices=["3bus", "8bus", "30bus", "39bus"],
                        help="Test systems to evaluate")
    parser.add_argument("--fast", action="store_true",
                        help="Reduced iterations for quick smoke test")
    parser.add_argument("--phase2", action="store_true",
                        help="Run all Phase 2 experiments")
    parser.add_argument("--phase1", action="store_true",
                        help="Run Phase 1 experiments (backward compatible)")
    parser.add_argument("--pandapower", action="store_true",
                        help="Enable pandapower integration")
    parser.add_argument("--verbose", action="store_true", default=True,
                        help="Print detailed progress")
    parser.add_argument("--comm", action="store_true",
                        help="Enable GNN communication for MAPPO")
    parser.add_argument("--compare", action="store_true",
                        help="Run full comparison of all methods")
    parser.add_argument("--methods", nargs="+",
                        default=["GA-SQP", "IPPO", "MAPPO"],
                        choices=["GA-SQP", "Single-PPO", "IPPO", "MAPPO", "MAPPO-Comm"],
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

    all_results = {}

    for sys_key in args.systems:
        net = NETWORKS[sys_key]()
        print(f"\n{'█' * 55}")
        print(f"  SYSTEM: {net.name}")
        print(f"  Relays: {net.n_relays}  |  Pairs: {net.n_pairs}  |  Buses: {net.n_buses}")
        print(f"  Methods: {', '.join(args.methods)}")
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

        all_results[sys_key] = sys_results

    # Print comparison table
    if args.compare or args.phase2:
        print_comparison_table(all_results)
        plot_phase2_comparison(all_results, "outputs/phase2_comparison.png")

    print("\n  All outputs written to ./outputs/")
    print("\n  Phase 2 experiments complete!")
    print("  Next steps (Phase 3):")
    print("    1. Implement Safe RL (Augmented Lagrangian + action projection)")
    print("    2. Add EWC continual adaptation")
    print("    3. Create benchmark comparison suite (HWOA, HHO, JAYA, EEO)")


if __name__ == "__main__":
    main()