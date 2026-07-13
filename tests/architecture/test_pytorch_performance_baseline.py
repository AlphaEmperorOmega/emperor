from __future__ import annotations

import json
import unittest
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASELINE_PATH = (
    PROJECT_ROOT
    / "docs"
    / "architecture"
    / "pytorch-performance-baseline-2026-07-10.json"
)
NARRATIVE_PATH = (
    PROJECT_ROOT / "docs" / "architecture" / "pytorch-performance-baseline.md"
)
REQUIRED_CATEGORIES = {
    "batch_placement",
    "dataloader",
    "item_synchronization",
    "repeated_to_device",
    "routing_halting",
    "ttt_inner_loop",
}
REQUIRED_SCENARIOS = {
    "batch_placement_forced_copy",
    "batch_placement_host_to_device",
    "dataloader_tensor_dataset_workers_0",
    "dataloader_tensor_dataset_workers_2",
    "model_forward_experts_linear_baseline",
    "model_forward_experts_linear_top1_switch_aux",
    "model_forward_linears_linear_baseline",
    "model_forward_linears_linear_halting",
    "model_forward_linears_linear_memory_ttt_1_steps",
    "model_forward_linears_linear_memory_ttt_2_steps",
    "model_forward_linears_linear_memory_ttt_4_steps",
    "model_forward_linears_linear_memory_ttt_disabled",
    "same_device_to_once",
    "same_device_to_repeated_8",
    "scalar_item_repeated_8",
    "scalar_stack_single_host_transfer",
}
TIMING_FIELDS = {
    "coefficient_of_variation",
    "max_ms",
    "mean_ms",
    "median_ms",
    "min_ms",
    "p95_ms",
    "stdev_ms",
}


class PytorchPerformanceBaselineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.baseline: dict[str, Any] = json.loads(
            BASELINE_PATH.read_text(encoding="utf-8")
        )
        cls.devices = {
            device["metadata"]["device"]: device for device in cls.baseline["devices"]
        }

    def test_reproduction_conditions_are_explicit(self) -> None:
        self.assertEqual(self.baseline["schema_version"], 1)
        conditions = self.baseline["conditions"]
        self.assertGreaterEqual(conditions["warmup"], 5)
        self.assertGreaterEqual(conditions["repetitions"], 30)
        self.assertEqual(conditions["seed"], 20260710)
        self.assertEqual(conditions["torch_threads"], 1)
        self.assertEqual(conditions["torch_interop_threads"], 1)
        self.assertIn("before and after", conditions["cuda_sync_policy"])

        environment = self.baseline["environment"]
        for field in ("hostname", "platform", "python", "torch"):
            with self.subTest(field=field):
                self.assertTrue(environment[field])
        self.assertIn("cuda_available", environment)
        self.assertIn("cuda_execution_supported", environment)

    def test_cpu_baseline_covers_every_audited_hot_path(self) -> None:
        cpu = self.devices["cpu"]
        self.assertEqual(cpu["status"], "passed")
        self.assertEqual(cpu["metadata"]["dtype"], "float32")
        self.assertTrue(cpu["metadata"]["name"])
        self.assertGreater(cpu["metadata"]["logical_cpu_count"], 0)

        results = cpu["results"]
        self.assertEqual({result["name"] for result in results}, REQUIRED_SCENARIOS)
        self.assertEqual(
            {result["category"] for result in results},
            REQUIRED_CATEGORIES,
        )
        self.assertTrue(all(result["status"] == "passed" for result in results))

    def test_each_cpu_result_records_timing_memory_and_identity(self) -> None:
        for result in self.devices["cpu"]["results"]:
            with self.subTest(scenario=result["name"]):
                self.assertEqual(result["device"], "cpu")
                self.assertEqual(result["dtype"], "float32")
                self.assertTrue(result["identity"])
                self.assertGreaterEqual(result["warmup"], 1)
                self.assertGreaterEqual(result["repetitions"], 2)
                self.assertEqual(
                    len(result["samples_ms"]),
                    result["repetitions"],
                )
                self.assertTrue(all(sample >= 0 for sample in result["samples_ms"]))
                self.assertIn("eager CPU", result["synchronization"])

                timing = result["timing"]
                self.assertEqual(set(timing), TIMING_FIELDS)
                self.assertLessEqual(timing["min_ms"], timing["median_ms"])
                self.assertLessEqual(timing["median_ms"], timing["max_ms"])
                self.assertLessEqual(timing["min_ms"], timing["p95_ms"])
                self.assertLessEqual(timing["p95_ms"], timing["max_ms"])
                self.assertGreaterEqual(timing["stdev_ms"], 0)
                self.assertGreaterEqual(timing["coefficient_of_variation"], 0)

                memory = result["memory"]
                self.assertIsNone(memory["cuda_peak_allocated_bytes"])
                self.assertGreater(memory["process_peak_rss_bytes"], 0)
                self.assertGreaterEqual(memory["python_tracemalloc_peak_bytes"], 0)
                self.assertIn("high-water mark", memory["process_peak_rss_scope"])
                self.assertIn("untimed", memory["python_tracemalloc_scope"])

    def test_model_and_dataloader_identities_are_reproducible(self) -> None:
        results = self.devices["cpu"]["results"]
        model_results = [
            result for result in results if result["identity"].get("model_package")
        ]
        self.assertEqual(len(model_results), 8)
        for result in model_results:
            with self.subTest(scenario=result["name"]):
                identity = result["identity"]
                self.assertEqual(identity["batch_size"], 32)
                self.assertEqual(identity["input_shape"], [32, 1, 28, 28])
                self.assertTrue(identity["dataset"])
                self.assertTrue(identity["task"])
                self.assertGreater(identity["parameter_count"], 0)
                self.assertIn("runtime_overrides", identity)

        dataloaders = [
            result for result in results if result["category"] == "dataloader"
        ]
        self.assertEqual(
            {result["identity"]["num_workers"] for result in dataloaders},
            {0, 2},
        )
        for result in dataloaders:
            self.assertEqual(result["identity"]["sample_count"], 2048)
            self.assertGreater(result["throughput_samples_per_second"], 0)

    def test_cuda_is_measured_or_skipped_with_hardware_evidence(self) -> None:
        cuda = self.devices["cuda"]
        environment = self.baseline["environment"]
        if cuda["status"] == "passed":
            self.assertTrue(environment["cuda_execution_supported"])
            self.assertEqual(
                {result["name"] for result in cuda["results"]},
                REQUIRED_SCENARIOS,
            )
            for result in cuda["results"]:
                with self.subTest(scenario=result["name"]):
                    self.assertIn("before and after", result["synchronization"])
                    self.assertIsNotNone(result["memory"]["cuda_peak_allocated_bytes"])
        else:
            self.assertEqual(cuda["status"], "skipped")
            self.assertFalse(environment["cuda_execution_supported"])
            self.assertEqual(cuda["results"], [])
            self.assertTrue(cuda["reason"])
            self.assertTrue(cuda["metadata"]["name"])
            self.assertEqual(len(cuda["metadata"]["capability"]), 2)
            self.assertTrue(cuda["metadata"]["compiled_architectures"])

    def test_narrative_and_reproduction_tool_accompany_baseline(self) -> None:
        narrative = NARRATIVE_PATH.read_text(encoding="utf-8")
        self.assertIn(BASELINE_PATH.name, narrative)
        self.assertIn("tools/benchmark_pytorch_hotspots.py", narrative)
        self.assertIn("No production implementation was changed", narrative)
        self.assertTrue(
            (PROJECT_ROOT / "tools" / "benchmark_pytorch_hotspots.py").is_file()
        )


if __name__ == "__main__":
    unittest.main()
