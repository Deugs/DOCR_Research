"""
run_all_networks.py
───────────────────
Run curriculum learning experiments across ALL IEEE test systems.

Usage:
    python run_all_networks.py --networks 3bus 8bus 14bus 30bus 39bus 57bus 118bus
    python run_all_networks.py --networks 8bus 14bus --episodes_per_task 500
    python run_all_networks.py --all  # Run on all 7 test systems
"""

import argparse
import sys
import os
from pathlib import Path
import logging

sys.path.insert(0, os.path.dirname(__file__))

from network.network_data import NETWORKS, get_network_metadata
from marl.exp_fig07_curriculum_learning import main as run_curriculum

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger(__name__)


def print_network_table():
    """Print a formatted table of all available networks."""
    meta = get_network_metadata()
    print("\n" + "=" * 80)
    print("IEEE TEST SYSTEMS AVAILABLE FOR DOCR-MARL")
    print("=" * 80)
    print(f"{'Key':>8} {'Buses':>6} {'Relays':>7} {'Pairs':>7} {'Voltage':>9} {'Description'}")
    print("-" * 80)
    for key in sorted(meta.keys()):
        m = meta[key]
        print(f"{key:>8} {m['n_buses']:>6} {m['n_relays']:>7} {m['n_pairs']:>7} "
              f"{m['base_voltage']:>7.1f} kV  {m['description']}")
    print("=" * 80)


def main():
    p = argparse.ArgumentParser(
        description="Run curriculum learning across IEEE test systems",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_all_networks.py --all
  python run_all_networks.py --networks 8bus 14bus 30bus
  python run_all_networks.py --networks 118bus --episodes_per_task 200 --n_seeds 1
        """
    )
    p.add_argument("--all", action="store_true",
                   help="Run on all 7 IEEE test systems")
    p.add_argument("--networks", nargs="+", choices=list(NETWORKS.keys()),
                   help="Specific networks to run (default: 8bus)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--episodes_per_task", type=int, default=1000)
    p.add_argument("--n_seeds", type=int, default=3)
    p.add_argument("--output_dir", type=str, default="outputs")
    p.add_argument("--list", action="store_true",
                   help="List all available networks and exit")

    args = p.parse_args()

    if args.list:
        print_network_table()
        return

    # Determine which networks to run
    if args.all:
        networks = list(NETWORKS.keys())
    elif args.networks:
        networks = args.networks
    else:
        networks = ["8bus"]

    print_network_table()
    logger.info(f"Running curriculum learning on: {', '.join(networks)}")

    # Run experiments sequentially
    results = {}
    for net_key in networks:
        meta = get_network_metadata()[net_key]
        logger.info(f"\n{'='*60}")
        logger.info(f"Starting: {net_key} ({meta['n_buses']} buses, "
                   f"{meta['n_relays']} relays, {meta['n_pairs']} pairs)")
        logger.info(f"{'='*60}")

        try:
            run_curriculum(
                network=net_key,
                seed=args.seed,
                episodes_per_task=args.episodes_per_task,
                n_seeds=args.n_seeds,
                output_dir=args.output_dir,
            )
            results[net_key] = "SUCCESS"
            logger.info(f"Completed: {net_key}")
        except Exception as e:
            results[net_key] = f"FAILED: {str(e)}"
            logger.error(f"Failed on {net_key}: {e}")

    # Final summary
    print("\n" + "=" * 80)
    print("EXPERIMENT SUMMARY")
    print("=" * 80)
    for net_key, status in results.items():
        meta = get_network_metadata()[net_key]
        marker = "✓" if status == "SUCCESS" else "✗"
        print(f"{marker} {net_key:>8} ({meta['n_buses']:>3} buses): {status}")
    print("=" * 80)


if __name__ == "__main__":
    main()