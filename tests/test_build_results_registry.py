"""
tests/test_build_results_registry.py
─────────────────────────────────────
Tests for experiments/build_results_registry.py's ingestion logic: parsing
the Fig 5/6 sweep npz files and the Core C cache into ResultsAggregator
records, without needing a real (multi-hour) experiment run.
"""
import json
import os
import pickle
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from experiments.build_results_registry import (
    load_fig05, load_fig06, load_core_c, build_registry, CORE_C_PV_LEVEL,
)


class TestLoadFig05(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.system_dir = Path(self.tmp.name) / "8bus"
        self.system_dir.mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    def _write_npz(self, n_seeds_in_npz=True):
        save_dict = {
            "pv_levels": np.array([0.0, 0.2]),
            "MAPPO_mean_OF": np.array([100.0, 110.0]),
            "MAPPO_std_OF": np.array([5.0, 6.0]),
            "MAPPO_mean_cti": np.array([2.0, 1.0]),
            "MAPPO_norm_OF": np.array([100.0, 110.0]),
        }
        if n_seeds_in_npz:
            save_dict["n_seeds"] = np.array(3)
        np.savez(self.system_dir / "fig05_pv_sweep.npz", **save_dict)

    def test_missing_file_returns_empty(self):
        self.assertEqual(load_fig05(self.system_dir, "8bus"), [])

    def test_parses_method_and_pv_rows(self):
        self._write_npz()
        records = load_fig05(self.system_dir, "8bus")
        self.assertEqual(len(records), 2)
        r0 = next(r for r in records if r["pv_level"] == 0.0)
        self.assertEqual(r0["method"], "MAPPO")
        self.assertEqual(r0["system"], "8bus")
        self.assertAlmostEqual(r0["of_mean"], 100.0)
        self.assertAlmostEqual(r0["of_std"], 5.0)
        self.assertAlmostEqual(r0["cti_violations_mean"], 2.0)
        self.assertEqual(r0["n_runs"], 3)
        # Fig 5 must never write a violation *rate* field -- it only has counts.
        self.assertNotIn("cti_violation_rate_pct", r0)

    def test_n_seeds_falls_back_to_stats_txt_when_npz_lacks_it(self):
        self._write_npz(n_seeds_in_npz=False)
        (self.system_dir / "fig05_pv_sweep_stats.txt").write_text(
            "PV level 0% -- pairwise significance across 7 seeds\n"
        )
        records = load_fig05(self.system_dir, "8bus")
        self.assertEqual(records[0]["n_runs"], 7)

    def test_n_runs_omitted_when_truly_unknown(self):
        self._write_npz(n_seeds_in_npz=False)
        records = load_fig05(self.system_dir, "8bus")
        self.assertNotIn("n_runs", records[0])


class TestLoadFig06(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.system_dir = Path(self.tmp.name) / "8bus"
        self.system_dir.mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    def test_violation_rate_is_a_percentage_not_a_fraction(self):
        np.savez(
            self.system_dir / "fig06_cti_violations.npz",
            pv_levels=np.array([0.0]),
            n_seeds=np.array(2),
            MAPPO_violation_rate=np.array([33.475]),
            MAPPO_violation_rate_std=np.array([1.0]),
            MAPPO_mild_rate=np.array([6.1]),
            MAPPO_mild_rate_std=np.array([0.5]),
        )
        records = load_fig06(self.system_dir, "8bus")
        self.assertEqual(len(records), 1)
        r = records[0]
        self.assertAlmostEqual(r["cti_violation_rate_pct"], 33.475)
        # cti_ok_rate must be a proper [0,1] fraction: (100 - pct) / 100.
        self.assertAlmostEqual(r["cti_ok_rate"], 0.66525)
        self.assertTrue(0.0 <= r["cti_ok_rate"] <= 1.0)
        # Fig 6 never writes cti_violations_mean -- that's Fig 5's count field.
        self.assertNotIn("cti_violations_mean", r)


class TestLoadCoreC(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cache_dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _write_cache(self, index):
        (self.cache_dir / "cache_index.json").write_text(json.dumps(index))

    def test_missing_index_returns_empty(self):
        self.assertEqual(load_core_c(self.cache_dir), [])

    def test_excludes_pre_rename_orphan_entry(self):
        """An entry named plain 'core_C_baselines' (no system suffix) predates
        Core C's per-system parametrization and never declared which system
        it was for -- it must be excluded, not guessed at."""
        self._write_cache({
            "deadbeef": {"name": "core_C_baselines", "config_keys": ["name", "methods"]},
        })
        with open(self.cache_dir / "deadbeef.pkl", "wb") as f:
            pickle.dump({"MAPPO": np.zeros((2, 50))}, f)
        self.assertEqual(load_core_c(self.cache_dir), [])

    def test_includes_system_suffixed_entry(self):
        self._write_cache({
            "abc123": {"name": "core_C_baselines_14bus", "config_keys": ["name", "system", "methods"]},
        })
        arr = np.concatenate([np.full((1, 150), 5.0), np.full((1, 150), 7.0)], axis=0)
        with open(self.cache_dir / "abc123.pkl", "wb") as f:
            pickle.dump({"MAPPO": arr}, f)

        records = load_core_c(self.cache_dir)
        self.assertEqual(len(records), 1)
        r = records[0]
        self.assertEqual(r["method"], "MAPPO")
        self.assertEqual(r["system"], "14bus")
        self.assertEqual(r["pv_level"], CORE_C_PV_LEVEL)
        self.assertEqual(r["n_runs"], 2)
        # mean of the last 100 columns of each row: row0=5.0, row1=7.0 -> mean 6.0
        self.assertAlmostEqual(r["of_mean"], 6.0)


class TestBuildRegistry(unittest.TestCase):
    def test_empty_dirs_produce_empty_registry(self):
        with tempfile.TemporaryDirectory() as outputs_dir, tempfile.TemporaryDirectory() as cache_dir:
            agg = build_registry(Path(outputs_dir), Path(cache_dir))
            self.assertEqual(len(agg.aggregated), 0)

    def test_fig05_and_fig06_for_same_key_merge_not_clobber(self):
        with tempfile.TemporaryDirectory() as outputs_dir, tempfile.TemporaryDirectory() as cache_dir:
            system_dir = Path(outputs_dir) / "8bus"
            system_dir.mkdir()
            np.savez(system_dir / "fig05_pv_sweep.npz",
                     pv_levels=np.array([0.0]), n_seeds=np.array(3),
                     MAPPO_mean_OF=np.array([100.0]), MAPPO_std_OF=np.array([5.0]),
                     MAPPO_mean_cti=np.array([2.0]), MAPPO_norm_OF=np.array([100.0]))
            np.savez(system_dir / "fig06_cti_violations.npz",
                     pv_levels=np.array([0.0]),
                     MAPPO_violation_rate=np.array([10.0]), MAPPO_violation_rate_std=np.array([1.0]))

            agg = build_registry(Path(outputs_dir), Path(cache_dir))
            entry = agg.aggregated["method=MAPPO|system=8bus|pv_level=0.0|fault_type=3ph"]
            self.assertEqual(entry.n_runs, 3)  # from Fig 5, not clobbered by Fig 6's unknown n_seeds
            self.assertAlmostEqual(entry.of_mean, 100.0)
            self.assertAlmostEqual(entry.cti_violation_rate_pct, 10.0)


if __name__ == "__main__":
    unittest.main()
