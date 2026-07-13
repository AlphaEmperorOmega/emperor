from __future__ import annotations

import os
import unittest

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from workbench.backend.inspection_errors import InspectionFailure
from workbench.backend.tests.inspection_support import (
    discover_models,
    inspect_model,
    list_model_datasets,
    list_model_monitors,
    list_model_presets,
)


class InspectorDiscoveryTests(unittest.TestCase):
    def test_model_discovery_lists_expected_packages(self) -> None:
        models = set(discover_models())
        self.assertGreaterEqual(
            models,
            {
                "bert/linear",
                "bert/linear_adaptive",
                "bert/expert_linear",
                "bert/expert_linear_adaptive",
                "experts/linear",
                "experts/linear_adaptive",
                "gpt/linear",
                "gpt/linear_adaptive",
                "gpt/expert_linear",
                "gpt/expert_linear_adaptive",
                "linears/linear",
                "linears/linear_adaptive",
                "neuron/linear",
                "neuron/linear_adaptive",
                "neuron/expert_linear",
                "neuron/expert_linear_adaptive",
                "parametric/parametric_generator",
                "parametric/parametric_matrix",
                "parametric/parametric_vector",
                "transformer/linear",
                "transformer/linear_adaptive",
                "transformer/expert_linear",
                "transformer/expert_linear_adaptive",
                "vit/linear",
                "vit/linear_adaptive",
                "vit/expert_linear",
                "vit/expert_linear_adaptive",
            },
        )

    def test_preset_discovery_for_linear(self) -> None:
        presets = list_model_presets("linears/linear")
        preset_names = {preset["name"] for preset in presets}
        self.assertIn("baseline", preset_names)
        self.assertIn("recurrent-gating-halting", preset_names)

    def test_flat_model_ids_are_rejected(self) -> None:
        with self.assertRaisesRegex(InspectionFailure, "Invalid model name"):
            list_model_presets("linear")

    def test_removed_neuron_model_ids_are_rejected(self) -> None:
        removed_flat_name = "neuron" + "_linear"
        with self.assertRaisesRegex(InspectionFailure, "Unknown model"):
            list_model_presets(f"neuron/{removed_flat_name}")

        with self.assertRaisesRegex(InspectionFailure, "Invalid model name"):
            list_model_presets(removed_flat_name)

    def test_dataset_discovery_for_linear(self) -> None:
        payload = list_model_datasets("linears/linear")
        self.assertEqual(payload["defaultExperimentTask"], "image-classification")
        self.assertEqual(len(payload["datasetGroups"]), 1)

        group = payload["datasetGroups"][0]
        self.assertEqual(group["experimentTask"], "image-classification")
        datasets = group["datasets"]
        dataset_by_name = {dataset["name"]: dataset for dataset in datasets}

        self.assertIn("Mnist", dataset_by_name)
        self.assertEqual(dataset_by_name["Mnist"]["inputDim"], 784)
        self.assertEqual(dataset_by_name["Mnist"]["outputDim"], 10)
        self.assertIn("Cifar10", dataset_by_name)

    def test_dataset_discovery_for_gpt_uses_causal_language_modeling(self) -> None:
        payload = list_model_datasets("gpt/linear")

        self.assertEqual(
            payload["defaultExperimentTask"],
            "causal-language-modeling",
        )
        self.assertEqual(len(payload["datasetGroups"]), 1)
        group = payload["datasetGroups"][0]
        self.assertEqual(group["experimentTask"], "causal-language-modeling")
        self.assertEqual(
            [dataset["name"] for dataset in group["datasets"]],
            ["WikiText2", "PennTreebank"],
        )

    def test_transformer_discovery_exposes_translation_directions_and_dims(
        self,
    ) -> None:
        for model in (
            "transformer/linear",
            "transformer/linear_adaptive",
            "transformer/expert_linear",
            "transformer/expert_linear_adaptive",
        ):
            with self.subTest(model=model):
                payload = list_model_datasets(model)
                self.assertEqual(
                    payload["defaultExperimentTask"],
                    "text-translation",
                )
                self.assertEqual(len(payload["datasetGroups"]), 1)
                group = payload["datasetGroups"][0]
                self.assertEqual(group["experimentTask"], "text-translation")
                datasets = group["datasets"]
                self.assertEqual(
                    [dataset["name"] for dataset in datasets],
                    ["Multi30kDeEn", "Multi30kEnDe"],
                )
                self.assertTrue(
                    all(dataset["inputDim"] == 8192 for dataset in datasets)
                )
                self.assertTrue(
                    all(dataset["outputDim"] == 8192 for dataset in datasets)
                )

    def test_monitor_discovery_for_model_packages(self) -> None:
        linear_monitors = list_model_monitors("linears/linear")
        linear_monitor_by_name = {
            monitor["name"]: monitor for monitor in linear_monitors
        }
        adaptive_monitor_by_name = {
            monitor["name"]: monitor
            for monitor in list_model_monitors("experts/linear_adaptive")
        }
        attention_monitor_by_name = {
            monitor["name"]: monitor for monitor in list_model_monitors("bert/linear")
        }
        parametric_monitor_by_name = {
            monitor["name"]: monitor
            for monitor in list_model_monitors("parametric/parametric_matrix")
        }

        self.assertEqual(
            [monitor["name"] for monitor in linear_monitors],
            [
                "linear",
                "recurrent-layer",
                "layer-controller",
                "halting",
                "memory",
            ],
        )
        for monitor in linear_monitors:
            self.assertFalse(monitor["defaultEnabled"])
        self.assertEqual(linear_monitor_by_name["linear"]["kinds"], ["scalar"])
        self.assertEqual(
            linear_monitor_by_name["recurrent-layer"]["kinds"],
            ["scalar", "histogram", "image"],
        )
        self.assertEqual(
            linear_monitor_by_name["layer-controller"]["kinds"],
            ["scalar"],
        )
        self.assertEqual(
            linear_monitor_by_name["halting"]["kinds"],
            ["scalar", "histogram", "image"],
        )
        self.assertEqual(
            linear_monitor_by_name["memory"]["kinds"],
            ["scalar"],
        )
        self.assertEqual(
            set(adaptive_monitor_by_name),
            {
                "linear",
                "adaptive",
                "sampler",
                "weight-bank",
                "recurrent-layer",
                "layer-controller",
                "memory",
            },
        )
        self.assertIn("image", adaptive_monitor_by_name["sampler"]["kinds"])
        self.assertEqual(
            set(attention_monitor_by_name),
            {"attention", "recurrent-layer", "layer-controller", "memory"},
        )
        self.assertFalse(attention_monitor_by_name["attention"]["defaultEnabled"])
        self.assertIn("image", attention_monitor_by_name["attention"]["kinds"])
        self.assertEqual(
            set(parametric_monitor_by_name),
            {"parametric", "layer-controller"},
        )
        self.assertFalse(parametric_monitor_by_name["parametric"]["defaultEnabled"])
        self.assertIn("histogram", parametric_monitor_by_name["parametric"]["kinds"])

    def test_one_preset_in_every_model_package_is_inspectable(self) -> None:
        for model in discover_models():
            with self.subTest(model=model):
                preset = list_model_presets(model)[0]["name"]
                result = inspect_model(model, preset)
                self.assertGreater(len(result["nodes"]), 0)
                self.assertGreater(len(result["edges"]), 0)


if __name__ == "__main__":
    unittest.main()
