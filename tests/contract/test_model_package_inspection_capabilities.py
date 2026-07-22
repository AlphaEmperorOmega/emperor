from __future__ import annotations

import unittest

from emperor.experiments import ExperimentTask
from model_runtime.packages import (
    config_key_to_model_param,
    dataset_cli_name,
    dataset_label,
    dataset_name,
    iter_supported_config_keys,
    normalize_dataset_name,
    parse_config_value,
)
from models.catalog import model_package


class ModelPackageInspectionCapabilityTests(unittest.TestCase):
    def setUp(self) -> None:
        package = model_package("linears/linear")
        self.assertIsNotNone(package)
        assert package is not None
        self.package = package

    def test_selected_package_resolves_presets_without_catalog_lookup(self) -> None:
        preset = self.package.resolve_preset("gating")

        self.assertEqual(self.package.preset_name(preset), "gating")
        self.assertTrue(self.package.preset_description(preset))
        self.assertEqual(
            self.package.preset_locks(preset)["stack_gate_flag"].value,
            True,
        )

    def test_selected_package_resolves_task_compatible_datasets(self) -> None:
        task = self.package.resolve_experiment_task("image-classification")
        dataset = self.package.resolve_dataset("fashion-mnist", task)

        self.assertEqual(task, ExperimentTask.IMAGE_CLASSIFICATION)
        self.assertEqual(dataset_name(dataset), "FashionMNIST")
        self.assertEqual(dataset_cli_name(dataset), "fashion-mnist")
        self.assertEqual(dataset_label(dataset), "Fashion M N I S T")
        self.assertEqual(normalize_dataset_name("Fashion_MNIST"), "fashion-mnist")

    def test_dataset_metadata_naming_is_transport_neutral(self) -> None:
        class FakeDatasetA:
            pass

        class FashionMnist:
            pass

        self.assertEqual(dataset_name(FakeDatasetA), "FakeDatasetA")
        self.assertEqual(dataset_label(FakeDatasetA), "Fake Dataset A")
        self.assertEqual(dataset_cli_name(FakeDatasetA), "fake-dataset-a")
        self.assertEqual(dataset_name(FashionMnist), "FashionMnist")
        self.assertEqual(dataset_label(FashionMnist), "Fashion Mnist")
        self.assertEqual(dataset_cli_name(FashionMnist), "fashion-mnist")
        self.assertEqual(normalize_dataset_name(" Fashion MNIST!! "), "fashion-mnist")

    def test_selected_package_resolves_aliases_and_deduplicates_datasets(
        self,
    ) -> None:
        resolved = self.package.resolve_datasets(
            ["fashion-mnist", "FashionMNIST", "mnist", "Mnist"]
        )

        self.assertEqual(
            [dataset_name(dataset) for dataset in resolved],
            ["FashionMNIST", "Mnist"],
        )

    def test_selected_package_rejects_path_and_incompatible_task_inputs(self) -> None:
        for dataset in (
            "./Mnist",
            "../Mnist",
            "/tmp/Mnist",
            "data/Mnist",
            "C:\\data\\Mnist",
        ):
            with self.subTest(dataset=dataset):
                with self.assertRaisesRegex(ValueError, "filesystem path"):
                    self.package.resolve_dataset(dataset)
        with self.assertRaisesRegex(ValueError, "Valid tasks: image-classification"):
            self.package.resolve_experiment_task("causal-language-modeling")

    def test_selected_package_reports_unknown_dataset_choices(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "Valid datasets: Mnist, FashionMNIST, Cifar10, Cifar100",
        ):
            self.package.resolve_dataset("UnknownDataset")

    def test_runtime_configuration_parsing_is_public_package_behavior(self) -> None:
        keys = iter_supported_config_keys(self.package.runtime_defaults)

        self.assertIn("HIDDEN_DIM", keys)
        self.assertEqual(
            parse_config_value(
                self.package.runtime_defaults,
                "HIDDEN_DIM",
                "128",
            ),
            128,
        )
        self.assertEqual(
            config_key_to_model_param("STACK_GATE_FLAG"), "stack_gate_flag"
        )

    def test_selected_package_supplies_ordered_configuration_field_metadata(
        self,
    ) -> None:
        metadata = self.package.configuration_field_metadata()
        search_metadata = self.package.configuration_field_metadata(
            include_search_space=True
        )

        self.assertEqual(metadata["BATCH_SIZE"]["sectionPath"], ["Global"])
        self.assertIn("SEARCH_SPACE_HIDDEN_DIM", search_metadata)
        self.assertTrue(metadata["BATCH_SIZE"]["sortKey"])

    def test_selected_package_validates_monitor_metadata(self) -> None:
        options = self.package.monitor_options()

        self.assertEqual(options[0].name, "linear")
        self.assertEqual(len({option.name for option in options}), len(options))


if __name__ == "__main__":
    unittest.main()
