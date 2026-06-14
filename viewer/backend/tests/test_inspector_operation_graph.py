from __future__ import annotations

import os
import unittest
from unittest.mock import patch

import torch
from pydantic import ValidationError
from torch import nn

from viewer.backend.inspector.operation_graph import (
    inspect_operation_graph,
    serialize_exported_program,
)
from viewer.backend.schemas import OperationGraphResponse

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

SMALL_LINEAR_OVERRIDES = {"hidden_dim": "16", "stack_num_layers": "1"}


class TinyOperationFixture(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 3)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.linear(inputs))


def user_input_node(result: dict[str, object]) -> dict[str, object]:
    for node in result["nodes"]:
        if node["details"].get("inputKind") == "user_input":
            return node
    raise AssertionError("Expected user input operation node")


class InspectorOperationGraphTests(unittest.TestCase):
    def test_operation_graph_schema_rejects_extra_fields(self) -> None:
        with self.assertRaises(ValidationError):
            OperationGraphResponse.model_validate(
                {
                    "model": "linears/linear",
                    "preset": "baseline",
                    "source": "torch-export",
                    "status": "unsupported",
                    "nodes": [],
                    "edges": [],
                    "warnings": [],
                    "unexpected": True,
                }
            )

    def test_tiny_fixture_uses_deterministic_graph_order_ids_and_edges(self) -> None:
        model = TinyOperationFixture().eval()
        exported_program = torch.export.export(
            model,
            (torch.zeros((1, 2), dtype=torch.float32),),
        )

        nodes, edges = serialize_exported_program(exported_program, model)

        self.assertEqual(
            [node["id"] for node in nodes],
            [f"op_{index:04d}" for index in range(len(nodes))],
        )
        self.assertEqual(nodes[-1]["opKind"], "output")
        self.assertTrue(edges)
        self.assertTrue(
            all(edge["source"] < edge["target"] for edge in edges),
            edges,
        )

    def test_linear_baseline_exports_placeholder_call_and_output_nodes(self) -> None:
        result = inspect_operation_graph(
            "linears/linear",
            "baseline",
            SMALL_LINEAR_OVERRIDES,
            dataset="Mnist",
        )

        OperationGraphResponse.model_validate(result)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["source"], "torch-export")
        op_kinds = {node["opKind"] for node in result["nodes"]}
        self.assertIn("placeholder", op_kinds)
        self.assertIn("call_function", op_kinds)
        self.assertIn("output", op_kinds)
        self.assertTrue(result["edges"])
        self.assertTrue(
            any(node["groupId"] == "input_model.model" for node in result["nodes"])
        )
        self.assertEqual(user_input_node(result)["details"]["shape"], [1, 1, 28, 28])

    def test_dataset_shape_changes_operation_input_metadata(self) -> None:
        mnist = inspect_operation_graph(
            "linears/linear",
            "baseline",
            SMALL_LINEAR_OVERRIDES,
            dataset="Mnist",
        )
        cifar = inspect_operation_graph(
            "linears/linear",
            "baseline",
            SMALL_LINEAR_OVERRIDES,
            dataset="Cifar100",
        )

        self.assertEqual(user_input_node(mnist)["details"]["shape"], [1, 1, 28, 28])
        self.assertEqual(user_input_node(cifar)["details"]["shape"], [1, 3, 32, 32])

    def test_export_failure_returns_unsupported_response(self) -> None:
        with patch(
            "viewer.backend.inspector.operation_graph.torch.export.export",
            side_effect=RuntimeError("cannot export fixture"),
        ):
            result = inspect_operation_graph(
                "linears/linear",
                "baseline",
                SMALL_LINEAR_OVERRIDES,
                dataset="Mnist",
            )

        OperationGraphResponse.model_validate(result)
        self.assertEqual(result["status"], "unsupported")
        self.assertEqual(result["nodes"], [])
        self.assertEqual(result["edges"], [])
        self.assertTrue(result["warnings"])
        self.assertIn("torch.export.export failed", result["warnings"][0])

    def test_export_serialization_failure_returns_unsupported_response(self) -> None:
        with patch(
            "viewer.backend.inspector.operation_graph.serialize_exported_program",
            side_effect=RuntimeError("cannot serialize fixture"),
        ):
            result = inspect_operation_graph(
                "linears/linear",
                "baseline",
                SMALL_LINEAR_OVERRIDES,
                dataset="Mnist",
            )

        OperationGraphResponse.model_validate(result)
        self.assertEqual(result["status"], "unsupported")
        self.assertEqual(result["nodes"], [])
        self.assertEqual(result["edges"], [])
        self.assertTrue(result["warnings"])
        self.assertIn("Failed to serialize torch.export graph", result["warnings"][0])


if __name__ == "__main__":
    unittest.main()
