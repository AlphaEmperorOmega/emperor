from __future__ import annotations

import os
import unittest

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from viewer.backend.inspector.discovery import (
    discover_models,
    list_model_datasets,
    list_model_monitors,
    list_model_presets,
)
from viewer.backend.inspector.service import inspect_model


class InspectorDiscoveryTests(unittest.TestCase):
    def test_model_discovery_lists_expected_packages(self) -> None:
        models = set(discover_models())
        self.assertGreaterEqual(
            models,
            {
                "bert_linear",
                "experts_linear",
                "experts_linear_adaptive",
                "linear",
                "linear_adaptive",
                "parametric_generator",
                "parametric_matrix",
                "parametric_vector",
                "vit_linear",
            },
        )

    def test_preset_discovery_for_linear(self) -> None:
        presets = list_model_presets("linear")
        preset_names = {preset["name"] for preset in presets}
        self.assertIn("baseline", preset_names)
        self.assertIn("recurrent-gating-halting", preset_names)

    def test_dataset_discovery_for_linear(self) -> None:
        datasets = list_model_datasets("linear")
        dataset_by_name = {dataset["name"]: dataset for dataset in datasets}

        self.assertIn("Mnist", dataset_by_name)
        self.assertEqual(dataset_by_name["Mnist"]["inputDim"], 784)
        self.assertEqual(dataset_by_name["Mnist"]["outputDim"], 10)
        self.assertIn("Cifar10", dataset_by_name)

    def test_monitor_discovery_for_model_packages(self) -> None:
        linear_monitors = list_model_monitors("linear")
        adaptive_monitor_by_name = {
            monitor["name"]: monitor
            for monitor in list_model_monitors("experts_linear_adaptive")
        }

        self.assertEqual([monitor["name"] for monitor in linear_monitors], ["linear"])
        self.assertFalse(linear_monitors[0]["defaultEnabled"])
        self.assertEqual(linear_monitors[0]["kinds"], ["scalar"])
        self.assertEqual(
            set(adaptive_monitor_by_name),
            {"linear", "adaptive", "sampler"},
        )
        self.assertIn("image", adaptive_monitor_by_name["sampler"]["kinds"])

    def test_one_preset_in_every_model_package_is_inspectable(self) -> None:
        for model in discover_models():
            with self.subTest(model=model):
                preset = list_model_presets(model)[0]["name"]
                result = inspect_model(model, preset)
                self.assertGreater(len(result["nodes"]), 0)
                self.assertGreater(len(result["edges"]), 0)


if __name__ == "__main__":
    unittest.main()
