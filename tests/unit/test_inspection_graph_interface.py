from __future__ import annotations

import os
import unittest
from dataclasses import FrozenInstanceError
from unittest.mock import PropertyMock, patch

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from models.catalog import model_package
from torch import nn

from model_runtime.inspection import (
    InspectionError,
    InspectionRequest,
    InspectionResult,
    inspect_model,
    inspect_model_graph,
)
from model_runtime.packages import ModelPackage


class InspectionGraphInterfaceTests(unittest.TestCase):
    def test_broken_package_construction_failure_is_transport_neutral(self) -> None:
        package = ModelPackage(
            "broken",
            "missing",
            "models.__inspection_missing__",
        )

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
                    overrides={"gate_flag": "false"},
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
            "model_class",
            new_callable=PropertyMock,
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
                            "hidden_dim": limits.maximum_dimension + 1,
                        },
                    ),
                )

    def test_obvious_parameter_growth_is_rejected_before_construction(self) -> None:
        package = model_package("linears/linear")
        assert package is not None
        limits = package.inspection_construction_limits

        with patch.object(
            ModelPackage,
            "model_class",
            new_callable=PropertyMock,
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
                            "hidden_dim": limits.maximum_dimension,
                            "stack_num_layers": limits.maximum_layer_count,
                        },
                    ),
                )


if __name__ == "__main__":
    unittest.main()
