"""
main.py
───────
Phase 1 Entry Point for DOCR-MARL Thesis.

Runs comprehensive experiments:
1. Static baseline (0% PV, 3ph fault) — validates implementation
2. PV penetration sweep (0-100%) — quantifies CTI violations
3. Fault type generalization (3ph, LL, 1ph, 2ph-g)
4. Topology robustness (N-1 contingencies)
5. Curriculum learning demonstration

Usage:
    python main.py                      # all systems, full experiments
    python main.py --systems 8bus       # single system
    python main.py --fast               # reduced iterations (quick test)
    python main.py --phase1             # Phase 1 experiments only
"""

import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
from network.network_data import NETWORKS
from network.relay_physics import check_constraints, FAULT_TYPES, compute_fault_currents_dynamic
from baselines.ga_sqp import run_ga_sqp
from marl.marl_env import DynamicDOCREnv
from marl.marl_agents import MAPPOMARLTrainer
from evaluation.evaluation import (
    plot_convergence, plot_relay_times,
    plot_cti_distribution, plot_relay_settings,
    plot_summary, print_comparison_table,
    plot_pv_penetration_analysis,
    plot_fault_type_comparison,
    plot_curriculum_progress,
    plot_topology_robustness,
    print_pv_analysis_table,
)

os.makedirs("outputs", exist_ok=True)


def run_marl(
    net,
    total_steps=30000,
    seed=0,
    verbose=True,
    curriculum=False,
    curriculum_stage=0,
    fault_type="3ph",
    use_pandapower=False,
):
    """Run MARL-IPPO with Phase 1 features."""
    env = DynamicDOCREnv(
        net,
        action_scale=0.05,
        max_steps=300,
        seed=seed,
        use_pandapower=use_pandapower,
        curriculum_stage=curriculum_stage,
        fault_type=fault_type,
    )
    trainer = MAPPOMARLTrainer(
        env,
        hidden_actor=32,
        hidden_critic=64,
        lr_actor=0.02,
        clip_eps=0.2,
        gamma=0.99,
        lam=0.95,
        rollout_len=64,
        n_epochs=4,
        seed=seed,
        curriculum=curriculum,
        curriculum_stages=6,
    )
    result = trainer.train(
        total_steps=total_steps,
        verbose=verbose,
        log_every=max(total_steps // 10, 1),
    )
    result["constraints"] = check_constraints(
        net, result["TMS"], result["PS"], fault_type=fault_type
    )
    return result


def experiment_static_baseline(args, sys_key, net):
    """Experiment 1: Static baseline (0% PV, 3ph fault) — validates implementation."""
    print(f"\n{'═'*55}")
    print(f"  EXPERIMENT 1: Static Baseline — {net.name}")
    print(f"  0% PV, 3ph fault, no topology changes")
    print(f"{'═'*55}")

    # GA-SQP (control)
    res_ga = run_ga_sqp(
        net,
        ga_pop=args.ga_pop,
        ga_gen=args.ga_gen,
        seed=42,
        verbose=args.verbose,
        fault_type="3ph",
        pv_level=0.0,
    )

    # MARL-IPPO (proposed)
    res_marl = run_marl(
        net,
        total_steps=args.marl_steps,
        seed=0,
        verbose=args.verbose,
        curriculum=False,
        fault_type="3ph",
        use_pandapower=args.pandapower,
    )

    return {"GA-SQP": res_ga, "MARL": res_marl}


def experiment_pv_sweep(args, sys_key, net):
    """Experiment 2: PV penetration sweep (0%, 20%, 40%, 60%, 80%, 100%)."""
    print(f"\n{'═'*55}")
    print(f"  EXPERIMENT 2: PV Penetration Sweep — {net.name}")
    print(f"  6 levels: 0%, 20%, 40%, 60%, 80%, 100%")
    print(f"{'═'*55}")

    pv_levels = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    results_by_pv = {}

    for stage_idx, pv in enumerate(pv_levels):
        print(f"\n  --- PV Level: {int(pv*100)}% ---")

        # objective_function()/compute_operating_times() only see PV effects
        # via an explicit topology_state["fault_currents"] override -- pv_level
        # alone (passed to run_ga_sqp below) is genuinely logging-only (see
        # ga_sqp.py's own docstring). Without this, GA-SQP got bit-identical
        # fault currents at every PV level. Matches the same pattern already
        # used correctly in baselines/experiments/unified_exp_fig05.py.
        topology_state = {"fault_currents": compute_fault_currents_dynamic(
            net, pv_penetration=pv, fault_type="3ph")}

        # GA-SQP with PV level
        res_ga = run_ga_sqp(
            net,
            ga_pop=args.ga_pop,
            ga_gen=args.ga_gen,
            seed=42,
            verbose=args.verbose,
            fault_type="3ph",
            pv_level=pv,
            topology_state=topology_state,
        )

        # MARL with PV level: DynamicDOCREnv has no direct pv_level float
        # param -- PV is selected via curriculum_stage indexing into its own
        # pv_levels list, which is exactly this loop's pv_levels. Previously
        # curriculum_stage was never passed (defaulted to 0), so every
        # iteration trained at 0% PV regardless of the loop variable.
        res_marl = run_marl(
            net,
            total_steps=args.marl_steps,
            seed=0,
            verbose=args.verbose,
            curriculum=False,
            curriculum_stage=stage_idx,
            fault_type="3ph",
            use_pandapower=args.pandapower,
        )

        results_by_pv[pv] = {"GA-SQP": res_ga, "MARL": res_marl}

    # Plot PV analysis
    plot_pv_penetration_analysis(
        results_by_pv, net.name,
        f"outputs/{sys_key}_pv_analysis.png"
    )
    print_pv_analysis_table(results_by_pv, net.name)

    return results_by_pv


def experiment_fault_types(args, sys_key, net):
    """Experiment 3: Fault type generalization (3ph, LL, 1ph, 2ph-g)."""
    print(f"\n{'═'*55}")
    print(f"  EXPERIMENT 3: Fault Type Generalization — {net.name}")
    print(f"  Testing: 3ph (train), LL, 1ph, 2ph-g (generalization)")
    print(f"{'═'*55}")

    fault_types = ["3ph", "LL", "1ph", "2ph-g"]
    results_by_fault = {}

    for ft in fault_types:
        print(f"\n  --- Fault Type: {ft} ---")

        res_ga = run_ga_sqp(
            net,
            ga_pop=args.ga_pop,
            ga_gen=args.ga_gen,
            seed=42,
            verbose=args.verbose,
            fault_type=ft,
            pv_level=0.0,
        )

        res_marl = run_marl(
            net,
            total_steps=args.marl_steps,
            seed=0,
            verbose=args.verbose,
            curriculum=False,
            fault_type=ft,
            use_pandapower=args.pandapower,
        )

        results_by_fault[ft] = {"GA-SQP": res_ga, "MARL": res_marl}

    plot_fault_type_comparison(
        results_by_fault, net.name,
        f"outputs/{sys_key}_fault_types.png"
    )

    return results_by_fault


def experiment_topology_robustness(args, sys_key, net):
    """Experiment 4: Topology robustness (N-1 contingencies)."""
    print(f"\n{'═'*55}")
    print(f"  EXPERIMENT 4: Topology Robustness — {net.name}")
    print(f"  Testing: Base, N-1 line outage, N-1 switch open")
    print(f"{'═'*55}")

    from environment.topology_manager import TopologyManager

    n_lines = max(1, net.n_buses - 1)
    top_mgr = TopologyManager(n_buses=net.n_buses, n_lines=n_lines)

    topologies = {
        "Base": None,
        "N-1_Line0": {"line_outages": [0]},
        "N-1_Line1": {"line_outages": [1]},
        "N-2_Lines01": {"line_outages": [0, 1]},
    }

    results_by_topology = {}

    for top_name, top_changes in topologies.items():
        print(f"\n  --- Topology: {top_name} ---")

        # GA-SQP with topology
        res_ga = run_ga_sqp(
            net,
            ga_pop=args.ga_pop,
            ga_gen=args.ga_gen,
            seed=42,
            verbose=args.verbose,
            fault_type="3ph",
            topology_state=top_changes,
            pv_level=0.0,
        )

        # MARL with topology (simplified: no topology changes during training)
        res_marl = run_marl(
            net,
            total_steps=args.marl_steps,
            seed=0,
            verbose=args.verbose,
            curriculum=False,
            fault_type="3ph",
            use_pandapower=args.pandapower,
        )

        results_by_topology[top_name] = {"GA-SQP": res_ga, "MARL": res_marl}

    plot_topology_robustness(
        results_by_topology, net.name,
        f"outputs/{sys_key}_topology.png"
    )

    return results_by_topology


def experiment_curriculum(args, sys_key, net):
    """Experiment 5: Curriculum learning demonstration (6 stages)."""
    print(f"\n{'═'*55}")
    print(f"  EXPERIMENT 5: Curriculum Learning — {net.name}")
    print(f"  6 stages: Static → PV@20% → PV@40% → PV@60% → PV@80% → PV@100% + topology")
    print(f"{'═'*55}")

    env = DynamicDOCREnv(
        net,
        action_scale=0.05,
        max_steps=300,
        seed=0,
        use_pandapower=args.pandapower,
        curriculum_stage=0,
    )

    trainer = MAPPOMARLTrainer(
        env,
        hidden_actor=32,
        hidden_critic=64,
        lr_actor=0.02,
        clip_eps=0.2,
        gamma=0.99,
        lam=0.95,
        rollout_len=64,
        n_epochs=4,
        seed=0,
        curriculum=True,
        curriculum_stages=6,
    )

    result = trainer.train(
        total_steps=args.marl_steps,
        verbose=args.verbose,
        log_every=max(args.marl_steps // 10, 1),
    )

    result["constraints"] = check_constraints(
        net, result["TMS"], result["PS"]
    )

    # Plot curriculum progress
    if result.get("stage_history"):
        plot_curriculum_progress(
            result["stage_history"],
            f"outputs/{sys_key}_curriculum.png"
        )

    print(f"\n  Curriculum final stage: {result.get('curriculum_stage', 0)}")
    print(f"  Best OF: {result['OF']:.4f}")
    print(f"  CTI satisfied: {result['constraints']['cti_satisfied']}")

    return result


def main():
    parser = argparse.ArgumentParser(description="DOCR-MARL Phase 1 Experiments")
    parser.add_argument("--systems", nargs="+",
                        default=["8bus", "30bus"],
                        choices=["3bus", "8bus", "30bus", "39bus"],
                        help="Test systems to evaluate")
    parser.add_argument("--fast", action="store_true",
                        help="Reduced iterations for quick smoke test")
    parser.add_argument("--phase1", action="store_true",
                        help="Run all Phase 1 experiments")
    parser.add_argument("--pandapower", action="store_true",
                        help="Enable pandapower integration (requires installation)")
    parser.add_argument("--verbose", action="store_true", default=True,
                        help="Print detailed progress")
    parser.add_argument("--experiments", nargs="+",
                        default=["static", "pv_sweep", "fault_types"],
                        choices=["static", "pv_sweep", "fault_types", "topology", "curriculum"],
                        help="Experiments to run")
    args = parser.parse_args()

    # ── Hyperparameters ────────────────────────────────────────────
    if args.fast:
        GA_POP, GA_GEN, MARL_STEPS = 30, 50, 5000
    else:
        GA_POP, GA_GEN, MARL_STEPS = 100, 200, 30000

    args.ga_pop = GA_POP
    args.ga_gen = GA_GEN
    args.marl_steps = MARL_STEPS

    all_results = {}

    for sys_key in args.systems:
        net = NETWORKS[sys_key]()
        print(f"\n{'█'*55}")
        print(f"  SYSTEM: {net.name}")
        print(f"  Relays: {net.n_relays}  |  Pairs: {net.n_pairs}  |  Buses: {net.n_buses}")
        print(f"  DG Buses: {len(net.dg_buses)}  |  PV Levels: 6 (0-100%)")
        print(f"{'█'*55}")

        sys_results = {}

        # Experiment 1: Static baseline (always run)
        if "static" in args.experiments:
            baseline = experiment_static_baseline(args, sys_key, net)
            sys_results["baseline"] = baseline

            # Per-system plots for baseline
            plot_relay_times(
                baseline["GA-SQP"], baseline["MARL"], net,
                f"outputs/{sys_key}_relay_times.png"
            )
            plot_cti_distribution(
                baseline["GA-SQP"], baseline["MARL"], net,
                f"outputs/{sys_key}_cti_dist.png"
            )
            plot_relay_settings(
                baseline["GA-SQP"], baseline["MARL"], net,
                f"outputs/{sys_key}_settings.png"
            )

        # Experiment 2: PV penetration sweep
        if "pv_sweep" in args.experiments:
            pv_results = experiment_pv_sweep(args, sys_key, net)
            sys_results["pv_sweep"] = pv_results

        # Experiment 3: Fault type generalization
        if "fault_types" in args.experiments:
            fault_results = experiment_fault_types(args, sys_key, net)
            sys_results["fault_types"] = fault_results

        # Experiment 4: Topology robustness
        if "topology" in args.experiments:
            topology_results = experiment_topology_robustness(args, sys_key, net)
            sys_results["topology"] = topology_results

        # Experiment 5: Curriculum learning
        if "curriculum" in args.experiments:
            curriculum_result = experiment_curriculum(args, sys_key, net)
            sys_results["curriculum"] = curriculum_result

        all_results[sys_key] = sys_results

    # ── Cross-system plots ─────────────────────────────────────────
    if len(args.systems) > 1 and "static" in args.experiments:
        baseline_results = {
            k: v["baseline"] for k, v in all_results.items() if "baseline" in v
        }
        if baseline_results:
            plot_convergence(baseline_results, "outputs/convergence.png")
            plot_summary(baseline_results, "outputs/summary.png")
            print_comparison_table(baseline_results)

    print("\n  All outputs written to ./outputs/")
    print("\n  Phase 1 experiments complete!")
    print("  Next steps:")
    print("    1. Install pandapower: pip install pandapower")
    print("    2. Validate IEEE 8-bus system loads correctly")
    print("    3. Run: python main.py --phase1 --pandapower")


if __name__ == "__main__":
    main()