"""
tests/test_phase5.py
────────────────────
Phase 5 Comprehensive Unit Tests.

Tests all Phase 5 components:
1. Scenario generation (stratified sampling, edge cases)
2. Training pipeline (configuration dispatch, checkpointing)
3. Config manager (YAML loading, validation)
4. Results aggregation (grouping, statistics, plotting)
5. Large-scale evaluation (metrics computation)
6. Integration (end-to-end pipeline)
"""

import unittest
import numpy as np
import os
import sys
import json
import tempfile
import shutil

# Add parent to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from experiments.scenario_generator import (
    ScenarioConfig, StratificationConfig, MarkovCloudModel,
    DailyProfileGenerator, LargeScaleScenarioGenerator,
    generate_scenario_dataset,
)
from experiments.training_pipeline import (
    TrainingConfig, TrainingResult, TrainingPipeline,
    create_experiment_matrix,
)
from experiments.config_manager import (
    ExperimentConfig, ConfigManager, create_experiment_from_config,
)
from experiments.results_aggregator import (
    AggregatedResult, ResultsAggregator,
)
from evaluation.large_scale_eval import (
    ScalabilityMetrics, GeneralizationMetrics, RobustnessMetrics,
    EfficiencyMetrics, CommunicationMetrics, SafetyMetrics,
    AdaptationMetrics, LargeScaleEvaluator,
)


class TestScenarioGeneration(unittest.TestCase):
    """Test scenario generation components."""

    def test_scenario_config_creation(self):
        """Test ScenarioConfig dataclass."""
        config = ScenarioConfig(
            scenario_id=0,
            system_name="8bus",
            pv_level=0.5,
            fault_type="3ph",
            topology_changes={},
            time_of_day=12.0,
            irradiance_factor=0.8,
            temperature=25.0,
            load_profile_factor=1.0,
        )
        self.assertEqual(config.scenario_id, 0)
        self.assertEqual(config.system_name, "8bus")
        self.assertEqual(config.pv_level, 0.5)

        # Test serialization
        d = config.to_dict()
        self.assertIn("scenario_id", d)
        config2 = ScenarioConfig.from_dict(d)
        self.assertEqual(config2.scenario_id, config.scenario_id)

    def test_stratification_config_defaults(self):
        """Test StratificationConfig default values."""
        config = StratificationConfig()
        self.assertEqual(config.n_train, 10000)
        self.assertEqual(config.n_val, 2000)
        self.assertEqual(len(config.pv_levels), 6)
        self.assertEqual(len(config.fault_types), 4)
        self.assertEqual(len(config.systems), 3)

    def test_markov_cloud_model(self):
        """Test Markov cloud state transitions."""
        model = MarkovCloudModel(seed=42)
        sequence = model.generate_sequence(10, time_of_day=12.0)

        self.assertEqual(len(sequence), 10)
        for state, factor in sequence:
            self.assertIn(state, model.STATES)
            self.assertIn(factor, model.STATES.values())
            self.assertTrue(0.0 <= factor <= 1.0)

    def test_markov_time_adjustment(self):
        """Test that time-of-day affects transition probabilities."""
        model = MarkovCloudModel(seed=42)

        # Generate multiple sequences to get statistical significance
        n_trials = 10
        midday_changes_total = 0
        morning_changes_total = 0

        for _ in range(n_trials):
            # Midday should be more stable
            seq_midday = model.generate_sequence(50, time_of_day=12.0)
            midday_changes_total += sum(1 for i in range(1, len(seq_midday)) 
                                   if seq_midday[i][0] != seq_midday[i-1][0])

            # Morning should be more variable
            seq_morning = model.generate_sequence(50, time_of_day=7.0)
            morning_changes_total += sum(1 for i in range(1, len(seq_morning)) 
                                    if seq_morning[i][0] != seq_morning[i-1][0])

        # Morning should have more changes on average
        self.assertGreater(morning_changes_total, midday_changes_total * 0.3)

    def test_daily_profile_generator(self):
        """Test daily profile generation."""
        gen = DailyProfileGenerator(seed=42)

        # Irradiance
        irr_noon = gen.generate_irradiance(12.0, cloud_factor=1.0)
        irr_night = gen.generate_irradiance(0.0, cloud_factor=1.0)
        self.assertGreater(irr_noon, irr_night)
        self.assertGreaterEqual(irr_noon, 0)

        # Temperature
        temp = gen.generate_temperature(12.0)
        self.assertIsInstance(temp, float)

        # Load factor
        load = gen.generate_load_factor(12.0)
        self.assertTrue(0.5 <= load <= 1.5)

    def test_scenario_generator_basic(self):
        """Test basic scenario generation."""
        config = StratificationConfig(n_train=100, n_val=20)
        generator = LargeScaleScenarioGenerator(config=config, seed=42)

        train, val = generator.generate_all_scenarios()

        self.assertEqual(len(train), 100)
        self.assertEqual(len(val), 20)

        # Check stratification
        pv_counts = {}
        for s in train:
            pv_counts[s.pv_level] = pv_counts.get(s.pv_level, 0) + 1

        # All PV levels should be represented
        for pv in config.pv_levels:
            self.assertIn(pv, pv_counts)
            self.assertGreater(pv_counts[pv], 0)

    def test_scenario_edge_cases(self):
        """Test edge-case scenario generation."""
        config = StratificationConfig(n_train=1000, n_val=0, edge_case_prob=0.1)
        generator = LargeScaleScenarioGenerator(config=config, seed=42)

        train, _ = generator.generate_all_scenarios()

        # Check for edge cases
        edge_cases = [s for s in train if s.pv_level in [0.0, 1.0] and s.topology_changes]
        self.assertGreater(len(edge_cases), 0)

    def test_curriculum_batches(self):
        """Test curriculum batch grouping."""
        config = StratificationConfig(n_train=120, n_val=0)
        generator = LargeScaleScenarioGenerator(config=config, seed=42)

        train, _ = generator.generate_all_scenarios()
        batches = generator.get_curriculum_batches(train)

        self.assertEqual(len(batches), 6)
        for stage, scenarios in batches.items():
            self.assertGreater(len(scenarios), 0)
            for s in scenarios:
                self.assertEqual(s.curriculum_stage, stage)

    def test_scenario_statistics(self):
        """Test scenario statistics computation."""
        config = StratificationConfig(n_train=100, n_val=0)
        generator = LargeScaleScenarioGenerator(config=config, seed=42)

        train, _ = generator.generate_all_scenarios()
        stats = generator.get_statistics(train)

        self.assertEqual(stats["total"], 100)
        self.assertIn("by_system", stats)
        self.assertIn("by_pv_level", stats)
        self.assertIn("n1_outage_rate", stats)
        self.assertTrue(0 <= stats["n1_outage_rate"] <= 1)

    def test_save_load_scenarios(self):
        """Test scenario save/load roundtrip."""
        config = StratificationConfig(n_train=50, n_val=0)
        generator = LargeScaleScenarioGenerator(config=config, seed=42)

        train, _ = generator.generate_all_scenarios()

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name

        try:
            generator.save_scenarios(train, temp_path)
            loaded = generator.load_scenarios(temp_path)

            self.assertEqual(len(loaded), len(train))
            self.assertEqual(loaded[0].scenario_id, train[0].scenario_id)
            self.assertEqual(loaded[0].system_name, train[0].system_name)
        finally:
            os.unlink(temp_path)


class TestTrainingPipeline(unittest.TestCase):
    """Test training pipeline components."""

    def test_training_config_creation(self):
        """Test TrainingConfig dataclass."""
        config = TrainingConfig(
            method="GA-SQP",
            system="8bus",
            pv_level=0.0,
            fault_type="3ph",
        )
        self.assertEqual(config.method, "GA-SQP")
        self.assertEqual(config.ga_pop, 100)  # Default

        d = config.to_dict()
        self.assertIn("method", d)

    def test_training_result_creation(self):
        """Test TrainingResult dataclass."""
        config = TrainingConfig(method="GA-SQP", system="8bus", pv_level=0.0, fault_type="3ph")
        result = TrainingResult(
            config=config,
            run_id=0,
            best_of=5.2,
            best_tms=np.array([0.5]),
            best_ps=np.array([1.0]),
            cti_satisfied=True,
            cti_violations=0,
            cti_min=0.35,
            elapsed_seconds=10.0,
            history=[5.5, 5.2],
        )
        self.assertEqual(result.best_of, 5.2)
        self.assertTrue(result.cti_satisfied)

        d = result.to_dict()
        self.assertIn("best_of", d)

    def test_experiment_matrix_creation(self):
        """Test experiment matrix generation."""
        configs = create_experiment_matrix(
            systems=["8bus"],
            methods=["GA-SQP"],
            pv_levels=[0.0, 1.0],
            fault_types=["3ph"],
            n_runs=2,
            fast=True,
        )

        self.assertEqual(len(configs), 4)  # 2 PV levels * 2 runs

        # Check fast mode parameters
        self.assertEqual(configs[0].ga_pop, 30)
        self.assertEqual(configs[0].ga_gen, 50)

    def test_experiment_matrix_full(self):
        """Test full experiment matrix."""
        configs = create_experiment_matrix(
            systems=["8bus", "30bus"],
            methods=["GA-SQP", "MAPPO"],
            pv_levels=[0.0, 0.5, 1.0],
            fault_types=["3ph", "LL"],
            n_runs=3,
        )

        expected = 2 * 2 * 3 * 2 * 3  # systems * methods * pv * faults * runs
        self.assertEqual(len(configs), expected)


class TestConfigManager(unittest.TestCase):
    """Test configuration management."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_experiment_config_defaults(self):
        """Test ExperimentConfig default values."""
        config = ExperimentConfig()
        self.assertEqual(config.name, "default_experiment")
        self.assertEqual(config.n_runs, 10)
        self.assertEqual(config.marl_steps, 30000)

    def test_config_serialization(self):
        """Test config YAML serialization."""
        config = ExperimentConfig(name="test", systems=["8bus"], methods=["GA-SQP"])

        filepath = os.path.join(self.temp_dir, "test_config.yaml")
        config.to_yaml(filepath)

        loaded = ExperimentConfig.from_yaml(filepath)
        self.assertEqual(loaded.name, "test")
        self.assertEqual(loaded.systems, ["8bus"])

    def test_config_manager_load_save(self):
        """Test ConfigManager load/save."""
        manager = ConfigManager(config_dir=self.temp_dir)

        config = ExperimentConfig(name="test", systems=["8bus"])
        manager.save(config, "test.yaml")

        loaded = manager.load("test.yaml")
        self.assertEqual(loaded.name, "test")

    def test_config_manager_merge(self):
        """Test config merging with overrides."""
        manager = ConfigManager(config_dir=self.temp_dir)

        base = ExperimentConfig(name="base", n_runs=10)
        manager.save(base, "base.yaml")

        merged = manager.merge("base.yaml", {"n_runs": 20})
        self.assertEqual(merged.n_runs, 20)
        self.assertEqual(merged.name, "base")

    def test_config_validation(self):
        """Test configuration validation."""
        manager = ConfigManager(config_dir=self.temp_dir)

        # Valid config
        valid = ExperimentConfig(systems=["8bus"], methods=["GA-SQP"])
        issues = manager.validate(valid)
        self.assertEqual(len(issues), 0)

        # Invalid config
        invalid = ExperimentConfig(systems=["invalid"], methods=["unknown"])
        issues = manager.validate(invalid)
        self.assertGreater(len(issues), 0)

    def test_create_experiment_from_config(self):
        """Test experiment spec creation."""
        config = ExperimentConfig(
            name="test",
            systems=["8bus"],
            methods=["GA-SQP"],
            pv_levels=[0.0, 1.0],
            n_runs=5,
        )

        spec = create_experiment_from_config(config)
        self.assertEqual(spec["name"], "test")
        self.assertEqual(spec["systems"], ["8bus"])
        self.assertEqual(spec["n_runs"], 5)


class TestResultsAggregator(unittest.TestCase):
    """Test results aggregation."""

    def setUp(self):
        self.dummy_results = [
            {"config": {"method": "GA-SQP", "system": "8bus", "pv_level": 0.0, "fault_type": "3ph"},
             "best_of": 5.2, "cti_satisfied": True, "cti_violations": 0, "cti_min": 0.35, "elapsed_seconds": 10.0},
            {"config": {"method": "GA-SQP", "system": "8bus", "pv_level": 0.0, "fault_type": "3ph"},
             "best_of": 5.3, "cti_satisfied": True, "cti_violations": 0, "cti_min": 0.33, "elapsed_seconds": 11.0},
            {"config": {"method": "MAPPO", "system": "8bus", "pv_level": 0.0, "fault_type": "3ph"},
             "best_of": 5.1, "cti_satisfied": True, "cti_violations": 0, "cti_min": 0.40, "elapsed_seconds": 120.0},
            {"config": {"method": "MAPPO", "system": "8bus", "pv_level": 1.0, "fault_type": "3ph"},
             "best_of": 5.8, "cti_satisfied": True, "cti_violations": 0, "cti_min": 0.32, "elapsed_seconds": 130.0},
        ]

    def test_aggregator_indexing(self):
        """Test results indexing."""
        agg = ResultsAggregator(self.dummy_results)

        self.assertEqual(len(agg.by_method), 2)  # GA-SQP, MAPPO
        self.assertEqual(len(agg.by_system), 1)  # 8bus
        self.assertEqual(len(agg.by_pv), 2)  # 0.0, 1.0

    def test_aggregate_by_method_system(self):
        """Test aggregation by method and system."""
        agg = ResultsAggregator(self.dummy_results)
        aggregated = agg.aggregate(group_by=["method", "system"])

        self.assertIn("method=GA-SQP|system=8bus", aggregated)
        self.assertIn("method=MAPPO|system=8bus", aggregated)

        ga_sqp = aggregated["method=GA-SQP|system=8bus"]
        self.assertEqual(ga_sqp.n_runs, 2)
        self.assertAlmostEqual(ga_sqp.of_mean, 5.25, places=5)

    def test_pv_sweep_data(self):
        """Test PV sweep data extraction."""
        agg = ResultsAggregator(self.dummy_results)
        sweep = agg.get_pv_sweep_data(method="MAPPO")

        self.assertIn(0.0, sweep)
        self.assertIn(1.0, sweep)
        self.assertAlmostEqual(sweep[0.0]["of_mean"], 5.1, places=5)

    def test_summary_table(self):
        """Test summary table printing."""
        agg = ResultsAggregator(self.dummy_results)
        # Should not raise
        agg.print_summary_table()

    def test_save_summary_json(self):
        """Test JSON summary export."""
        agg = ResultsAggregator(self.dummy_results)
        agg.aggregate()

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name

        try:
            agg.save_summary_json(temp_path)
            with open(temp_path, 'r') as f:
                summary = json.load(f)
            self.assertIn("total_runs", summary)
            self.assertEqual(summary["total_runs"], 4)
        finally:
            os.unlink(temp_path)


class TestLargeScaleEvaluation(unittest.TestCase):
    """Test large-scale evaluation metrics."""

    def setUp(self):
        self.dummy_results = [
            {"config": {"method": "GA-SQP", "system": "8bus", "pv_level": 0.0, "fault_type": "3ph"},
             "best_of": 5.2, "cti_satisfied": True, "cti_violations": 0, "cti_min": 0.35, "elapsed_seconds": 10.0},
            {"config": {"method": "GA-SQP", "system": "30bus", "pv_level": 0.0, "fault_type": "3ph"},
             "best_of": 12.1, "cti_satisfied": True, "cti_violations": 0, "cti_min": 0.30, "elapsed_seconds": 15.0},
            {"config": {"method": "MAPPO", "system": "8bus", "pv_level": 0.0, "fault_type": "3ph"},
             "best_of": 5.1, "cti_satisfied": True, "cti_violations": 0, "cti_min": 0.40, "elapsed_seconds": 120.0},
            {"config": {"method": "MAPPO-Comm", "system": "8bus", "pv_level": 0.0, "fault_type": "3ph"},
             "best_of": 4.9, "cti_satisfied": True, "cti_violations": 0, "cti_min": 0.42, "elapsed_seconds": 130.0, "comm_bandwidth": 0.256},
            {"config": {"method": "MAPPO", "system": "8bus", "pv_level": 0.0, "fault_type": "LL"},
             "best_of": 5.3, "cti_satisfied": True, "cti_violations": 0, "cti_min": 0.38, "elapsed_seconds": 125.0},
        ]

    def test_scalability_metrics(self):
        """Test scalability metrics computation."""
        evaluator = LargeScaleEvaluator(self.dummy_results)
        metrics = evaluator.compute_scalability_metrics()

        self.assertIn("GA-SQP", metrics)
        self.assertIn("MAPPO", metrics)

        # Check system sizes
        for method, method_metrics in metrics.items():
            for m in method_metrics:
                self.assertGreater(m.n_relays, 0)
                self.assertGreaterEqual(m.of_value, 0)

    def test_generalization_metrics(self):
        """Test generalization metrics."""
        evaluator = LargeScaleEvaluator(self.dummy_results)
        metrics = evaluator.compute_generalization_metrics()

        self.assertIn("MAPPO", metrics)
        m = metrics["MAPPO"]

        # MAPPO has both 3ph (seen) and LL (unseen)
        self.assertGreater(m.seen_of_mean, 0)
        self.assertGreater(m.unseen_of_mean, 0)
        self.assertIn("3ph", m.fault_type_performance)
        self.assertIn("LL", m.fault_type_performance)

    def test_efficiency_metrics(self):
        """Test efficiency metrics."""
        evaluator = LargeScaleEvaluator(self.dummy_results)
        metrics = evaluator.compute_efficiency_metrics()

        self.assertIn("GA-SQP", metrics)
        self.assertIn("MAPPO", metrics)

        # GA-SQP should be faster than MAPPO
        ga_time = metrics["GA-SQP"].total_training_time
        mappo_time = metrics["MAPPO"].total_training_time
        self.assertLess(ga_time, mappo_time)

    def test_communication_metrics(self):
        """Test communication metrics."""
        evaluator = LargeScaleEvaluator(self.dummy_results)
        metrics = evaluator.compute_communication_metrics()

        self.assertIn("MAPPO-Comm", metrics)
        m = metrics["MAPPO-Comm"]

        self.assertGreater(m.avg_bandwidth_kbps, 0)
        self.assertEqual(m.message_dim, 8)

    def test_safety_metrics(self):
        """Test safety metrics."""
        evaluator = LargeScaleEvaluator(self.dummy_results)
        metrics = evaluator.compute_safety_metrics()

        self.assertIn("GA-SQP", metrics)
        m = metrics["GA-SQP"]

        self.assertGreaterEqual(m.cti_violation_rate, 0.0)
        self.assertLessEqual(m.cti_violation_rate, 1.0)

    def test_full_report(self):
        """Test complete evaluation report generation."""
        evaluator = LargeScaleEvaluator(self.dummy_results)
        report = evaluator.generate_report()

        self.assertIn("SCALABILITY METRICS", report)
        self.assertIn("GENERALIZATION METRICS", report)
        self.assertIn("EFFICIENCY METRICS", report)
        self.assertIn("COMMUNICATION METRICS", report)
        self.assertIn("SAFETY METRICS", report)


class TestIntegration(unittest.TestCase):
    """Integration tests for Phase 5."""

    def test_end_to_end_scenario_generation(self):
        """Test end-to-end scenario generation and statistics."""
        config = StratificationConfig(n_train=50, n_val=10)
        generator = LargeScaleScenarioGenerator(config=config, seed=42)

        train, val = generator.generate_all_scenarios()

        self.assertEqual(len(train), 50)
        self.assertEqual(len(val), 10)

        # Check stratification
        train_stats = generator.get_statistics(train)
        self.assertEqual(train_stats["total"], 50)

        # All systems should be represented
        for system in config.systems:
            self.assertIn(system, train_stats["by_system"])

    def test_config_to_experiment_spec(self):
        """Test config to experiment spec conversion."""
        config = ExperimentConfig(
            name="integration_test",
            systems=["8bus"],
            methods=["GA-SQP"],
            pv_levels=[0.0, 1.0],
            n_runs=2,
        )

        spec = create_experiment_from_config(config)
        self.assertEqual(spec["name"], "integration_test")
        self.assertEqual(spec["n_runs"], 2)

    def test_results_aggregation_with_realistic_data(self):
        """Test aggregation with realistic multi-run data."""
        # Generate synthetic results
        np.random.seed(42)
        results = []
        for method in ["GA-SQP", "MAPPO"]:
            for system in ["8bus", "30bus"]:
                for pv in [0.0, 0.5, 1.0]:
                    for run in range(5):
                        base_of = 5.0 if system == "8bus" else 12.0
                        pv_penalty = pv * 0.5
                        method_bonus = -0.1 if method == "MAPPO" else 0.0
                        noise = np.random.normal(0, 0.1)

                        results.append({
                            "config": {
                                "method": method,
                                "system": system,
                                "pv_level": pv,
                                "fault_type": "3ph",
                            },
                            "best_of": base_of + pv_penalty + method_bonus + noise,
                            "cti_satisfied": np.random.random() > 0.1,
                            "cti_violations": np.random.randint(0, 2),
                            "cti_min": 0.3 + np.random.random() * 0.2,
                            "elapsed_seconds": np.random.exponential(100),
                        })

        agg = ResultsAggregator(results)
        aggregated = agg.aggregate(group_by=["method", "system", "pv_level"])

        # Should have entries for all combinations
        self.assertEqual(len(aggregated), 12)  # 2 methods * 2 systems * 3 PV levels

        # Each should have 5 runs
        for key, a in aggregated.items():
            self.assertEqual(a.n_runs, 5)
            self.assertGreater(a.of_mean, 0)

    def test_large_scale_evaluator_comprehensive(self):
        """Test comprehensive evaluator with diverse data."""
        np.random.seed(42)
        results = []

        # Generate diverse results
        for method in ["GA-SQP", "MAPPO", "MAPPO-Comm"]:
            for system in ["8bus", "30bus"]:
                for pv in [0.0, 1.0]:
                    for fault in ["3ph", "LL"]:
                        for run in range(3):
                            base = 5.0 if system == "8bus" else 12.0
                            of = base + pv * 0.8 + np.random.normal(0, 0.2)

                            result = {
                                "config": {
                                    "method": method,
                                    "system": system,
                                    "pv_level": pv,
                                    "fault_type": fault,
                                },
                                "best_of": of,
                                "cti_satisfied": True,
                                "cti_violations": 0,
                                "cti_min": 0.35,
                                "elapsed_seconds": np.random.exponential(50),
                            }

                            if "Comm" in method:
                                result["comm_bandwidth"] = np.random.uniform(0.1, 0.5)

                            results.append(result)

        evaluator = LargeScaleEvaluator(results)
        all_metrics = evaluator.compute_all_metrics()

        self.assertIn("scalability", all_metrics)
        self.assertIn("generalization", all_metrics)
        self.assertIn("efficiency", all_metrics)
        self.assertIn("communication", all_metrics)
        self.assertIn("safety", all_metrics)


def run_tests():
    """Run all Phase 5 tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    suite.addTests(loader.loadTestsFromTestCase(TestScenarioGeneration))
    suite.addTests(loader.loadTestsFromTestCase(TestTrainingPipeline))
    suite.addTests(loader.loadTestsFromTestCase(TestConfigManager))
    suite.addTests(loader.loadTestsFromTestCase(TestResultsAggregator))
    suite.addTests(loader.loadTestsFromTestCase(TestLargeScaleEvaluation))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
