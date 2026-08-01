from __future__ import annotations

import unittest

from emperor.experiments import ExperimentTask
from model_runtime.inspection import InspectionRequest, inspect_model
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

    def test_selected_package_resolves_new_presets_and_projects_their_locks(
        self,
    ) -> None:
        expected_locked_fields = {
            "weighted-residual": {"stack_residual_connection_option"},
            "weighted-blend-residual": {"stack_residual_connection_option"},
            "attention-residual": {"stack_residual_connection_option"},
            "recurrent-layer-gating": {"recurrent_flag", "stack_gate_flag"},
            "recurrent-dual-gating": {
                "recurrent_flag",
                "stack_gate_flag",
                "recurrent_stack_gate_flag",
            },
            "recurrent-layer-halting": {"recurrent_flag", "stack_halting_flag"},
            "recurrent-dual-halting": {
                "recurrent_flag",
                "stack_halting_flag",
                "recurrent_stack_halting_flag",
            },
            "weighted-memory": {"memory_flag", "memory_option"},
            "element-wise-weighted-memory": {"memory_flag", "memory_option"},
            "no-norm": {"layer_norm_position"},
            "pre-activation-norm": {"layer_norm_position"},
        }

        for name, locked_fields in expected_locked_fields.items():
            with self.subTest(preset=name):
                preset = self.package.resolve_preset(name)
                self.assertEqual(self.package.preset_name(preset), name)
                self.assertTrue(self.package.preset_description(preset))
                self.assertEqual(
                    set(self.package.preset_locks(preset)),
                    locked_fields,
                )

    def test_new_preset_inspection_reports_scope_and_config_identity(self) -> None:
        residual = inspect_model(
            self.package,
            InspectionRequest(preset="weighted-residual"),
        )
        residual_nodes = {node.path: node for node in residual.nodes}
        residual_node = residual_nodes["main_model.layers.0.residual_connection"]
        self.assertEqual(residual_node.type_name, "WeightedResidual")
        self.assertEqual(
            residual_node.configuration.type_name,
            "WeightedResidualConfig",
        )

        dual_gating = inspect_model(
            self.package,
            InspectionRequest(preset="recurrent-dual-gating"),
        )
        gating_nodes = {node.path: node for node in dual_gating.nodes}
        self.assertIn("main_model.recurrent_gate", gating_nodes)
        self.assertTrue(gating_nodes["main_model"].details["recurrent"]["gate"])
        self.assertTrue(gating_nodes["main_model.block_model.layers.0"].details["gate"])

        dual_halting = inspect_model(
            self.package,
            InspectionRequest(preset="recurrent-dual-halting"),
        )
        halting_nodes = {node.path: node for node in dual_halting.nodes}
        self.assertIn("main_model.halting_model", halting_nodes)
        self.assertTrue(halting_nodes["main_model"].details["recurrent"]["halting"])
        self.assertTrue(
            halting_nodes["main_model.block_model.layers.0"].details["halting"]
        )

        memory = inspect_model(
            self.package,
            InspectionRequest(preset="element-wise-weighted-memory"),
        )
        memory_nodes = {node.path: node for node in memory.nodes}
        memory_node = memory_nodes["main_model.layers.0.memory_model"]
        self.assertEqual(memory_node.type_name, "ElementWiseWeightedDynamicMemory")
        self.assertEqual(
            memory_node.configuration.type_name,
            "ElementWiseWeightedDynamicMemoryConfig",
        )

        pre_activation_norm = inspect_model(
            self.package,
            InspectionRequest(preset="pre-activation-norm"),
        )
        norm_nodes = {node.path: node for node in pre_activation_norm.nodes}
        self.assertEqual(
            norm_nodes["main_model.layers.0"].details["layer_norm"],
            "DEFAULT",
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
