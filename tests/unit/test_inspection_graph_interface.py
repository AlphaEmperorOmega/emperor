from __future__ import annotations

import os
import unittest
from dataclasses import FrozenInstanceError
from unittest.mock import patch

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from torch import nn

from emperor.layers import (
    LayerConfig,
    ResidualConfig,
    ResidualConnectionOptions,
)
from emperor.linears import LinearLayerConfig
from model_runtime.inspection import (
    InspectionError,
    InspectionRequest,
    InspectionResult,
    inspect_model,
    inspect_model_graph,
)
from model_runtime.packages import ModelIdentity, ModelPackage
from models.catalog import model_package


class _BrokenPackageAdapter:
    @staticmethod
    def _missing(*_args, **_kwargs):
        raise ModuleNotFoundError("No module named 'models.__inspection_missing__'")

    load_metadata = _missing
    load_runtime_options_type = _missing
    bind_runtime_defaults = _missing
    load_preset_type = _missing
    load_presets = _missing
    build_configurations = _missing
    build_model = _missing
    build_experiment = _missing


def _broken_package() -> ModelPackage:
    return ModelPackage(
        ModelIdentity("broken", "missing"),
        _BrokenPackageAdapter(),
    )


class InspectionGraphInterfaceTests(unittest.TestCase):
    def test_broken_package_construction_failure_is_transport_neutral(self) -> None:
        package = _broken_package()

        with self.assertRaisesRegex(
            InspectionError,
            "Failed to import model package 'broken/missing'",
        ):
            inspect_model(package, InspectionRequest(preset="baseline"))

    def test_selected_package_produces_frozen_semantic_graph_records(self) -> None:
        package = model_package("linears/linear")
        self.assertIsNotNone(package)
        assert package is not None

        result = inspect_model(
            package,
            InspectionRequest(
                preset="baseline",
                dataset="Mnist",
                experiment_task="image-classification",
            ),
        )

        self.assertIsInstance(result, InspectionResult)
        self.assertEqual(result.identity, package.identity)
        self.assertEqual(result.preset, "baseline")
        self.assertIsInstance(result.nodes, tuple)
        self.assertIsInstance(result.edges, tuple)
        root = result.nodes[0]
        self.assertEqual(root.id, "__root__")
        self.assertEqual(root.path, "model")
        self.assertEqual(root.type_name, "Model")
        self.assertEqual(root.graph_role, "architecture")
        self.assertGreater(root.parameter_count, 0)
        self.assertNotIn("parameterCount", root.details)
        with self.assertRaises(FrozenInstanceError):
            result.preset = "gating"  # type: ignore[misc]
        with self.assertRaises(TypeError):
            root.details["changed"] = True  # type: ignore[index]

    def test_empty_module_graph_has_only_the_stable_root(self) -> None:
        graph = inspect_model_graph(nn.Sequential())

        self.assertEqual(len(graph.nodes), 1)
        self.assertEqual(graph.nodes[0].id, "__root__")
        self.assertEqual(graph.nodes[0].parameter_count, 0)
        self.assertEqual(graph.edges, ())

    def test_layer_residual_config_preserves_flat_inspection_fields(self) -> None:
        configured_module = nn.Module()
        configured_module.cfg = LayerConfig(
            residual_config=ResidualConfig(
                option=ResidualConnectionOptions.WEIGHTED_BLEND,
                model_config=LinearLayerConfig(bias_flag=True),
            )
        )

        configuration = inspect_model_graph(configured_module).nodes[0].configuration

        self.assertIsNotNone(configuration)
        assert configuration is not None
        serialized_fields = {field.key: field.value for field in configuration.fields}
        self.assertNotIn("residual_config", serialized_fields)
        self.assertEqual(
            serialized_fields["residual_connection_option"],
            "WEIGHTED_BLEND",
        )
        self.assertEqual(
            serialized_fields["residual_model_config"],
            "LinearLayerConfig",
        )

    def test_semantic_graph_preserves_stable_paths_and_referential_integrity(
        self,
    ) -> None:
        package = model_package("linears/linear")
        assert package is not None

        result = inspect_model(package, InspectionRequest(preset="baseline"))

        node_ids = [node.id for node in result.nodes]
        edge_ids = [edge.id for edge in result.edges]
        self.assertEqual(len(node_ids), len(set(node_ids)))
        self.assertEqual(len(edge_ids), len(set(edge_ids)))
        self.assertEqual(len(result.edges), len(result.nodes) - 1)
        self.assertTrue(
            all(
                edge.source in node_ids and edge.target in node_ids
                for edge in result.edges
            )
        )

    def test_override_and_lock_validation_flow_through_the_same_interface(self) -> None:
        package = model_package("linears/linear")
        assert package is not None

        result = inspect_model(
            package,
            InspectionRequest(
                preset="baseline",
                overrides={"hidden-dim": "128"},
            ),
        )
        nodes = {node.id: node for node in result.nodes}
        self.assertEqual(nodes["main_model.layers.0"].details["dims"], "128 -> 128")

        with self.assertRaisesRegex(InspectionError, "locked fields"):
            inspect_model(
                package,
                InspectionRequest(
                    preset="gating",
                    overrides={"stack_gate_flag": "false"},
                ),
            )

    def test_oversized_structure_is_rejected_before_model_constructor_lookup(
        self,
    ) -> None:
        package = model_package("linears/linear")
        assert package is not None
        limits = package.inspection_construction_limits

        with patch.object(
            ModelPackage,
            "build_model",
            side_effect=AssertionError("model constructor was observed"),
        ):
            with self.assertRaisesRegex(
                InspectionError,
                "HIDDEN_DIM.*exceeds.*maximum",
            ):
                inspect_model(
                    package,
                    InspectionRequest(
                        preset="baseline",
                        overrides={
                            "hidden_dim": limits.maximum_hidden_dimension + 1,
                        },
                    ),
                )

    def test_obvious_parameter_growth_is_rejected_before_construction(self) -> None:
        package = model_package("linears/linear")
        assert package is not None
        limits = package.inspection_construction_limits

        with patch.object(
            ModelPackage,
            "build_model",
            side_effect=AssertionError("model constructor was observed"),
        ):
            with self.assertRaisesRegex(
                InspectionError,
                "estimated parameter count.*exceeds",
            ):
                inspect_model(
                    package,
                    InspectionRequest(
                        preset="baseline",
                        overrides={
                            "hidden_dim": limits.maximum_hidden_dimension,
                            "stack_num_layers": limits.maximum_layer_count,
                        },
                    ),
                )


if __name__ == "__main__":
    unittest.main()
