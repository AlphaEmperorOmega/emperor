from __future__ import annotations

import os
import unittest
from types import SimpleNamespace

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from viewer.backend.inspector.discovery import (
    dataset_cli_name,
    dataset_label,
    dataset_name,
    discover_models,
    list_model_datasets,
    list_model_monitors,
    list_model_presets,
    normalize_dataset_name,
    resolve_dataset,
    resolve_datasets,
)
from viewer.backend.inspector.errors import InspectorError
from viewer.backend.inspector.service import inspect_model


class FakeDatasetA:
    flattened_input_dim = 4
    num_classes = 2


class FakeDatasetB:
    flattened_input_dim = 8
    num_classes = 3


class FashionMnist:
    pass


class Mnist:
    pass


def fake_parts():
    return SimpleNamespace(
        name="fake_model",
        dataset_options=[FakeDatasetA, FakeDatasetB],
        dataset=FakeDatasetA,
    )


def fake_mnist_parts():
    return SimpleNamespace(
        name="fake_model",
        dataset_options=[Mnist],
        dataset=Mnist,
    )


class InspectorDiscoveryTests(unittest.TestCase):
    def test_dataset_naming_helpers_characterize_current_format(self) -> None:
        self.assertEqual(dataset_name(FakeDatasetA), "FakeDatasetA")
        self.assertEqual(dataset_label(FakeDatasetA), "Fake Dataset A")
        self.assertEqual(dataset_cli_name(FakeDatasetA), "fake-dataset-a")
        self.assertEqual(dataset_name(FashionMnist), "FashionMnist")
        self.assertEqual(dataset_label(FashionMnist), "Fashion Mnist")
        self.assertEqual(dataset_cli_name(FashionMnist), "fashion-mnist")
        self.assertEqual(normalize_dataset_name(" Fashion MNIST!! "), "fashion-mnist")
        self.assertEqual(normalize_dataset_name("fake_dataset_a"), "fake-dataset-a")

    def test_resolve_dataset_accepts_existing_aliases(self) -> None:
        parts = fake_parts()

        self.assertIs(resolve_dataset(parts, None), FakeDatasetA)
        for alias in (
            "FakeDatasetB",
            "fakedatasetb",
            "fake-dataset-b",
            "Fake Dataset B",
            " fake_dataset_b!! ",
        ):
            with self.subTest(alias=alias):
                self.assertIs(resolve_dataset(parts, alias), FakeDatasetB)

    def test_resolve_dataset_rejects_path_like_inputs(self) -> None:
        parts = fake_mnist_parts()

        for dataset in (
            "./Mnist",
            "../Mnist",
            "/tmp/Mnist",
            "data/Mnist",
            "C:\\data\\Mnist",
        ):
            with self.subTest(dataset=dataset):
                with self.assertRaises(InspectorError) as error:
                    resolve_dataset(parts, dataset)

                message = str(error.exception)
                self.assertIn(dataset, message)
                self.assertIn("filesystem path", message)
                self.assertIn("server-known dataset name", message)

    def test_resolve_datasets_preserves_order_and_removes_duplicates(self) -> None:
        resolved = resolve_datasets(
            fake_parts(),
            [
                "fake-dataset-b",
                "FakeDatasetB",
                "FakeDatasetA",
                "fake-dataset-a",
            ],
        )

        self.assertEqual(resolved, [FakeDatasetB, FakeDatasetA])

    def test_resolve_dataset_unknown_error_text(self) -> None:
        with self.assertRaises(InspectorError) as error:
            resolve_dataset(fake_parts(), "UnknownDataset")

        self.assertEqual(
            str(error.exception),
            "Unknown dataset 'UnknownDataset' for model 'fake_model'. "
            "Valid datasets: FakeDatasetA, FakeDatasetB.",
        )

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
