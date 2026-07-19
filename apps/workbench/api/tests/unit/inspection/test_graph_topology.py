from __future__ import annotations

import os
import unittest

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from torch import nn

from emperor.layers import (
    ActivationOptions,
    LastLayerBiasOptions,
    LayerConfig,
    LayerNormPositionOptions,
    LayerStackConfig,
)
from emperor.linears import LinearLayerConfig
from emperor_workbench.inspection import (
    InspectionFailure,
)
from models.catalog import model_package
from tests.support.inspection import (
    inspect_model,
    serialize_graph,
)
from tests.unit.inspection._graph_support import (
    GraphEdgePayload,
    SharedParameterGraph,
    TinyGraphFixture,
    nodes_by_id,
)


class InspectionGraphTopologyTests(unittest.TestCase):
    def test_graph_serializer_preserves_depth_first_named_child_order(self) -> None:
        nodes, edges = serialize_graph(TinyGraphFixture())
        expected_edges: list[GraphEdgePayload] = [
            {"id": "__root__-encoder", "source": "__root__", "target": "encoder"},
            {
                "id": "encoder-encoder.linear",
                "source": "encoder",
                "target": "encoder.linear",
            },
            {
                "id": "encoder-encoder.relu",
                "source": "encoder",
                "target": "encoder.relu",
            },
            {"id": "__root__-head", "source": "__root__", "target": "head"},
        ]

        self.assertEqual(
            [node["id"] for node in nodes],
            ["__root__", "encoder", "encoder.linear", "encoder.relu", "head"],
        )
        self.assertEqual(edges, expected_edges)

    def test_graph_serializer_preserves_node_shape_for_small_modules(self) -> None:
        nodes, _edges = serialize_graph(TinyGraphFixture())
        node_by_id = nodes_by_id(nodes)
        expected_node_keys = {
            "id",
            "label",
            "typeName",
            "path",
            "graphRole",
            "parameterCount",
            "parameterSizeBytes",
            "details",
            "config",
        }
        optional_node_keys = {"description"}

        for node in nodes:
            with self.subTest(node=node["id"]):
                self.assertEqual(set(node) - optional_node_keys, expected_node_keys)
                self.assertEqual(node["graphRole"], "architecture")

        self.assertEqual(node_by_id["__root__"]["path"], "model")
        self.assertEqual(node_by_id["__root__"]["parameterCount"], 12)
        self.assertEqual(node_by_id["__root__"]["parameterSizeBytes"], 48)
        self.assertEqual(node_by_id["encoder"]["parameterCount"], 9)
        self.assertEqual(node_by_id["encoder"]["parameterSizeBytes"], 36)
        self.assertEqual(node_by_id["encoder.linear"]["parameterCount"], 9)
        self.assertEqual(node_by_id["encoder.linear"]["parameterSizeBytes"], 36)
        self.assertEqual(node_by_id["encoder.relu"]["parameterCount"], 0)
        self.assertEqual(node_by_id["encoder.relu"]["parameterSizeBytes"], 0)
        self.assertEqual(node_by_id["head"]["parameterCount"], 3)
        self.assertEqual(node_by_id["head"]["parameterSizeBytes"], 12)
        self.assertEqual(
            node_by_id["encoder.linear"]["details"],
            {"weightShape": "3 x 2", "biasShape": "3"},
        )
        self.assertEqual(node_by_id["encoder.relu"]["details"], {})
        self.assertIsNone(node_by_id["encoder.relu"]["config"])

    def test_graph_serializer_uses_stable_paths_and_edges(self) -> None:
        result = inspect_model("linears/linear", "baseline")
        node_ids = {node["id"] for node in result["nodes"]}
        edge_ids = {edge["id"] for edge in result["edges"]}
        node_by_id = nodes_by_id(result["nodes"])

        self.assertNotIn("main_model.layers", node_ids)
        self.assertIn("main_model.layers.0.model", node_ids)
        self.assertNotIn("main_model-main_model.layers", edge_ids)
        self.assertNotIn("main_model.layers-main_model.layers.0", edge_ids)
        for index in range(5):
            node_id = f"main_model.layers.{index}"
            with self.subTest(node_id=node_id):
                self.assertIn(node_id, node_ids)
                self.assertEqual(node_by_id[node_id]["path"], node_id)
                self.assertIn(f"main_model-{node_id}", edge_ids)

    def test_graph_serializer_keeps_ordinary_module_lists_visible(self) -> None:
        class ExpertModulesFixture(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.expert_modules = nn.ModuleList([nn.Linear(2, 2)])

        nodes, edges = serialize_graph(ExpertModulesFixture())
        node_ids = {node["id"] for node in nodes}
        edge_ids = {edge["id"] for edge in edges}

        self.assertIn("expert_modules", node_ids)
        self.assertIn("expert_modules.0", node_ids)
        self.assertIn("__root__-expert_modules", edge_ids)
        self.assertIn("expert_modules-expert_modules.0", edge_ids)

    def test_graph_serializer_uses_class_names_for_semantic_layer_stack_paths(
        self,
    ) -> None:
        class NestedStackFixture(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.main_model = nn.Module()
                self.main_model.block_model = LayerStackConfig(
                    input_dim=4,
                    hidden_dim=4,
                    output_dim=2,
                    num_layers=1,
                    last_layer_bias_option=LastLayerBiasOptions.DEFAULT,
                    apply_output_pipeline_flag=True,
                    layer_config=LayerConfig(
                        input_dim=4,
                        output_dim=2,
                        activation=ActivationOptions.DISABLED,
                        residual_config=None,
                        dropout_probability=0.0,
                        layer_norm_position=LayerNormPositionOptions.DISABLED,
                        gate_config=None,
                        halting_config=None,
                        memory_config=None,
                        layer_model_config=LinearLayerConfig(
                            input_dim=4,
                            output_dim=2,
                            bias_flag=True,
                        ),
                    ),
                ).build()

        nodes, edges = serialize_graph(NestedStackFixture())
        node_by_id = nodes_by_id(nodes)
        edge_ids = {edge["id"] for edge in edges}
        stack_node = node_by_id["main_model.block_model"]

        self.assertEqual(stack_node["label"], "LayerStack")
        self.assertEqual(stack_node["typeName"], "LayerStack")
        self.assertEqual(stack_node["path"], "main_model.block_model")
        self.assertNotEqual(stack_node["label"], "Block Model")
        self.assertNotIn("main_model.block_model.layers", node_by_id)
        self.assertIn("main_model.block_model.layers.0", node_by_id)
        self.assertIn(
            "main_model.block_model-main_model.block_model.layers.0",
            edge_ids,
        )

    def test_inspector_rejects_unknown_memory_linear_graph(self) -> None:
        with self.assertRaises(InspectionFailure) as raised:
            inspect_model("memory/memory_linear", "gated-residual")

        self.assertIn("Unknown model", raised.exception.detail)

    def test_graph_serializer_marks_internal_modules(self) -> None:
        nodes, _edges = serialize_graph(nn.Sequential(nn.Dropout(), nn.LayerNorm(4)))
        role_by_type = {node["typeName"]: node["graphRole"] for node in nodes}
        description_by_type = {
            node["typeName"]: node.get("description") for node in nodes
        }

        self.assertEqual(role_by_type["Dropout"], "internal")
        self.assertEqual(role_by_type["LayerNorm"], "internal")
        self.assertEqual(
            description_by_type["Dropout"],
            (
                "Regularization module that randomly zeroes activations during "
                "training and is inactive during evaluation."
            ),
        )
        self.assertEqual(
            description_by_type["LayerNorm"],
            (
                "Normalizes features within each sample to stabilize hidden-state "
                "scale before or after a layer block."
            ),
        )

    def test_graph_serializer_marks_runtime_modules(self) -> None:
        result = inspect_model("linears/linear", "baseline")
        role_by_id = {node["id"]: node["graphRole"] for node in result["nodes"]}
        description_by_id = {
            node["id"]: node.get("description") for node in result["nodes"]
        }

        self.assertEqual(role_by_id["loss_fn"], "runtime")
        self.assertEqual(role_by_id["metrics"], "runtime")
        self.assertEqual(role_by_id["metrics.train_accuracy"], "runtime")
        self.assertEqual(role_by_id["metrics.train_f1_score"], "runtime")
        self.assertEqual(
            description_by_id["loss_fn"],
            "Runtime loss module for multi-class classification targets.",
        )
        self.assertEqual(
            description_by_id["metrics"],
            (
                "Runtime module that groups classifier metrics for train, "
                "validation, and test reporting."
            ),
        )

    def test_graph_serializer_marks_architecture_modules(self) -> None:
        result = inspect_model("linears/linear", "baseline")
        role_by_id = {node["id"]: node["graphRole"] for node in result["nodes"]}

        self.assertEqual(role_by_id["__root__"], "architecture")
        self.assertEqual(role_by_id["main_model"], "architecture")
        self.assertNotIn("main_model.layers", role_by_id)
        self.assertEqual(role_by_id["main_model.layers.0"], "architecture")
        self.assertEqual(role_by_id["main_model.layers.0.model"], "architecture")

    def test_graph_serializer_reports_parameter_counts(self) -> None:
        nodes, _edges = serialize_graph(nn.Sequential(nn.Linear(4, 3), nn.ReLU()))
        count_by_id = {node["id"]: node["parameterCount"] for node in nodes}

        self.assertEqual(count_by_id["__root__"], 15)
        self.assertEqual(count_by_id["0"], 15)
        self.assertEqual(count_by_id["1"], 0)

    def test_graph_serializer_counts_shared_parameters_once_at_root(self) -> None:
        model = SharedParameterGraph()
        nodes, _edges = serialize_graph(model)
        node_by_id = nodes_by_id(nodes)
        expected_count = model.left.weight.numel()
        expected_size_bytes = expected_count * model.left.weight.element_size()

        self.assertIs(model.left.weight, model.right.weight)
        self.assertEqual(node_by_id["__root__"]["parameterCount"], expected_count)
        self.assertEqual(
            node_by_id["__root__"]["parameterSizeBytes"],
            expected_size_bytes,
        )
        self.assertEqual(node_by_id["left"]["parameterCount"], expected_count)
        self.assertEqual(node_by_id["right"]["parameterCount"], expected_count)

    def test_inspect_linear_baseline_model_size_matches_registered_parameters(
        self,
    ) -> None:
        package = model_package("linears/linear")
        assert package is not None
        preset = package.resolve_preset("baseline")
        dataset = package.resolve_dataset(None)
        cfg = package.build_configurations(preset, dataset)[0]
        model = package.build_model(cfg)
        result = inspect_model("linears/linear", "baseline")
        node_by_id = nodes_by_id(result["nodes"])
        expected_count = sum(parameter.numel() for parameter in model.parameters())
        expected_size_bytes = sum(
            parameter.numel() * parameter.element_size()
            for parameter in model.parameters()
        )

        self.assertEqual(result["parameterCount"], expected_count)
        self.assertEqual(result["parameterSizeBytes"], expected_size_bytes)
        self.assertEqual(
            node_by_id["__root__"]["parameterCount"],
            result["parameterCount"],
        )
        self.assertEqual(
            node_by_id["__root__"]["parameterSizeBytes"],
            result["parameterSizeBytes"],
        )
        for node_id in ("input_model", "main_model", "output_model"):
            with self.subTest(node_id=node_id):
                self.assertGreater(node_by_id[node_id]["parameterCount"], 0)
                self.assertGreater(node_by_id[node_id]["parameterSizeBytes"], 0)

    def test_inspect_experts_then_linear_keeps_linear_graph_identity(self) -> None:
        experts_result = inspect_model("experts/linear", "baseline")
        experts_type_names = {node["typeName"] for node in experts_result["nodes"]}
        self.assertIn("MixtureOfExperts", experts_type_names)

        linear_result = inspect_model("linears/linear", "baseline")
        linear_type_names = {node["typeName"] for node in linear_result["nodes"]}

        self.assertEqual(linear_result["modelType"], "linears")
        self.assertEqual(linear_result["model"], "linear")
        self.assertEqual(linear_result["preset"], "baseline")
        self.assertIn("LinearLayer", linear_type_names)
        self.assertFalse(
            any(
                type_name.startswith("MixtureOfExperts")
                for type_name in linear_type_names
            )
        )

if __name__ == "__main__":
    unittest.main()
