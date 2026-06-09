from __future__ import annotations

import os
import unittest
from collections import Counter
from typing import Any, TypeAlias

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from torch import nn

from emperor.augmentations.adaptive_parameters.config import (
    AdaptiveParameterAugmentationConfig,
)
from emperor.base.layer.config import LayerConfig, LayerStackConfig
from emperor.base.options import (
    ActivationOptions,
    LastLayerBiasOptions,
    LayerNormPositionOptions,
)
from emperor.linears.core.config import AdaptiveLinearLayerConfig, LinearLayerConfig
from viewer.backend.inspector.discovery import discover_models, list_model_presets
from viewer.backend.inspector.graph import serialize_graph
from viewer.backend.inspector.service import inspect_model

GraphNodePayload: TypeAlias = dict[str, Any]
GraphEdgePayload: TypeAlias = dict[str, str]


class ConfiguredModule(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg


class TinyGraphFixture(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential()
        self.encoder.add_module("linear", nn.Linear(2, 3))
        self.encoder.add_module("relu", nn.ReLU())
        self.head = nn.Linear(3, 1, bias=False)


def config_fields(node: GraphNodePayload) -> dict[str, object]:
    config = node["config"]
    if config is None:
        return {}
    return {field["key"]: field["value"] for field in config["fields"]}


def nodes_by_id(nodes: list[GraphNodePayload]) -> dict[str, GraphNodePayload]:
    return {str(node["id"]): node for node in nodes}


class InspectorGraphTests(unittest.TestCase):
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
            "details",
            "config",
        }

        for node in nodes:
            with self.subTest(node=node["id"]):
                self.assertEqual(set(node), expected_node_keys)
                self.assertEqual(node["graphRole"], "architecture")

        self.assertEqual(node_by_id["__root__"]["path"], "model")
        self.assertEqual(node_by_id["__root__"]["parameterCount"], 12)
        self.assertEqual(node_by_id["encoder"]["parameterCount"], 9)
        self.assertEqual(node_by_id["encoder.linear"]["parameterCount"], 9)
        self.assertEqual(node_by_id["encoder.relu"]["parameterCount"], 0)
        self.assertEqual(node_by_id["head"]["parameterCount"], 3)
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
        self.assertIn("main_model.0.model", node_ids)
        self.assertIn("main_model-main_model.0", edge_ids)

    def test_graph_serializer_marks_internal_modules(self) -> None:
        nodes, _edges = serialize_graph(nn.Sequential(nn.Dropout(), nn.LayerNorm(4)))
        role_by_type = {node["typeName"]: node["graphRole"] for node in nodes}

        self.assertEqual(role_by_type["Dropout"], "internal")
        self.assertEqual(role_by_type["LayerNorm"], "internal")

    def test_graph_serializer_marks_runtime_modules(self) -> None:
        result = inspect_model("linears/linear", "baseline")
        role_by_id = {node["id"]: node["graphRole"] for node in result["nodes"]}

        self.assertEqual(role_by_id["loss_fn"], "runtime")
        self.assertEqual(role_by_id["metrics"], "runtime")
        self.assertEqual(role_by_id["metrics.train_accuracy"], "runtime")
        self.assertEqual(role_by_id["metrics.train_f1_score"], "runtime")

    def test_graph_serializer_marks_architecture_modules(self) -> None:
        result = inspect_model("linears/linear", "baseline")
        role_by_id = {node["id"]: node["graphRole"] for node in result["nodes"]}

        self.assertEqual(role_by_id["__root__"], "architecture")
        self.assertEqual(role_by_id["main_model"], "architecture")
        self.assertEqual(role_by_id["main_model.0"], "architecture")
        self.assertEqual(role_by_id["main_model.0.model"], "architecture")

    def test_graph_serializer_reports_parameter_counts(self) -> None:
        nodes, _edges = serialize_graph(nn.Sequential(nn.Linear(4, 3), nn.ReLU()))
        count_by_id = {node["id"]: node["parameterCount"] for node in nodes}

        self.assertEqual(count_by_id["__root__"], 15)
        self.assertEqual(count_by_id["0"], 15)
        self.assertEqual(count_by_id["1"], 0)

    def test_graph_serializer_reports_direct_weight_and_bias_shapes(self) -> None:
        nodes, _edges = serialize_graph(nn.Sequential(nn.Linear(4, 3), nn.ReLU()))
        details_by_id = {node["id"]: node["details"] for node in nodes}

        self.assertNotIn("weightShape", details_by_id["__root__"])
        self.assertNotIn("biasShape", details_by_id["__root__"])
        self.assertEqual(details_by_id["0"]["weightShape"], "3 x 4")
        self.assertEqual(details_by_id["0"]["biasShape"], "3")
        self.assertNotIn("weightShape", details_by_id["1"])
        self.assertNotIn("biasShape", details_by_id["1"])

    def test_graph_serializer_includes_linear_config_fields(self) -> None:
        model = LinearLayerConfig(
            input_dim=4,
            output_dim=3,
            bias_flag=False,
        ).build()
        nodes, _edges = serialize_graph(nn.Sequential(model))
        linear_node = {node["id"]: node for node in nodes}["0"]

        self.assertEqual(linear_node["config"]["typeName"], "LinearLayerConfig")
        self.assertEqual(
            config_fields(linear_node),
            {"input_dim": 4, "output_dim": 3, "bias_flag": False},
        )
        self.assertIn("weightShape", linear_node["details"])
        self.assertNotIn("biasShape", linear_node["details"])

    def test_graph_serializer_serializes_adaptive_config_references(self) -> None:
        no_augmentation = ConfiguredModule(
            AdaptiveLinearLayerConfig(
                input_dim=4,
                output_dim=3,
                bias_flag=True,
                adaptive_augmentation_config=None,
            )
        )
        with_augmentation = ConfiguredModule(
            AdaptiveLinearLayerConfig(
                input_dim=4,
                output_dim=3,
                bias_flag=True,
                adaptive_augmentation_config=AdaptiveParameterAugmentationConfig(
                    input_dim=4,
                    output_dim=3,
                ),
            )
        )

        nodes, _edges = serialize_graph(
            nn.Sequential(no_augmentation, with_augmentation)
        )
        node_by_id = {node["id"]: node for node in nodes}

        self.assertIsNone(
            config_fields(node_by_id["0"])["adaptive_augmentation_config"]
        )
        self.assertEqual(
            config_fields(node_by_id["1"])["adaptive_augmentation_config"],
            "AdaptiveParameterAugmentationConfig",
        )

    def test_graph_serializer_includes_layer_config_fields(self) -> None:
        cfg = LayerConfig(
            input_dim=4,
            output_dim=3,
            activation=ActivationOptions.GELU,
            residual_flag=False,
            dropout_probability=0.1,
            layer_norm_position=LayerNormPositionOptions.BEFORE,
            gate_config=None,
            halting_config=None,
            memory_config=None,
            shared_halting_flag=False,
            layer_model_config=LinearLayerConfig(
                input_dim=4,
                output_dim=3,
                bias_flag=True,
            ),
        )
        nodes, _edges = serialize_graph(ConfiguredModule(cfg))
        root = nodes[0]

        self.assertEqual(root["config"]["typeName"], "LayerConfig")
        self.assertEqual(
            [field["key"] for field in root["config"]["fields"]],
            [
                "input_dim",
                "output_dim",
                "activation",
                "residual_flag",
                "dropout_probability",
                "layer_norm_position",
                "gate_config",
                "halting_config",
                "memory_config",
                "shared_halting_flag",
                "layer_model_config",
            ],
        )
        self.assertEqual(config_fields(root)["activation"], "GELU")
        self.assertEqual(config_fields(root)["layer_norm_position"], "BEFORE")
        self.assertEqual(config_fields(root)["layer_model_config"], "LinearLayerConfig")
        self.assertIsNone(config_fields(root)["gate_config"])
        self.assertIsNone(config_fields(root)["halting_config"])
        self.assertIsNone(config_fields(root)["memory_config"])

    def test_graph_serializer_preserves_layer_stack_config_on_sequential(self) -> None:
        stack_config = LayerStackConfig(
            input_dim=4,
            hidden_dim=5,
            output_dim=2,
            num_layers=2,
            last_layer_bias_option=LastLayerBiasOptions.DISABLED,
            apply_output_pipeline_flag=True,
            layer_config=LayerConfig(
                input_dim=4,
                output_dim=5,
                activation=ActivationOptions.GELU,
                residual_flag=False,
                dropout_probability=0.0,
                layer_norm_position=LayerNormPositionOptions.DISABLED,
                gate_config=None,
                halting_config=None,
                memory_config=None,
                shared_halting_flag=False,
                layer_model_config=LinearLayerConfig(
                    input_dim=4,
                    output_dim=5,
                    bias_flag=True,
                ),
            ),
        )

        model = stack_config.build()
        nodes, _edges = serialize_graph(model)
        root = nodes[0]

        self.assertIsInstance(model, nn.Sequential)
        self.assertEqual(root["config"]["typeName"], "LayerStackConfig")
        self.assertEqual(config_fields(root)["input_dim"], 4)
        self.assertEqual(config_fields(root)["hidden_dim"], 5)
        self.assertEqual(config_fields(root)["output_dim"], 2)
        self.assertEqual(config_fields(root)["num_layers"], 2)
        self.assertEqual(config_fields(root)["last_layer_bias_option"], "DISABLED")
        self.assertTrue(config_fields(root)["apply_output_pipeline_flag"])
        self.assertEqual(config_fields(root)["layer_config"], "LayerConfig")

    def test_graph_serializer_reports_expert_metadata(self) -> None:
        result = inspect_model("experts/experts_linear", "baseline")
        expert_model = next(
            node for node in result["nodes"] if node["typeName"] == "MixtureOfExperts"
        )
        expert_stack_model = next(
            node
            for node in result["nodes"]
            if node["typeName"] == "MixtureOfExpertsModel"
        )

        for node in (expert_model, expert_stack_model):
            with self.subTest(node=node["id"]):
                self.assertEqual(node["details"]["topK"], 2)
                self.assertEqual(node["details"]["numExperts"], 4)
                self.assertEqual(node["details"]["routingMode"], "LAYER")

    def test_graph_serializer_reports_neuron_cluster_grid(self) -> None:
        result = inspect_model(
            "neuron/neuron_linear",
            "baseline",
            {
                "cluster_x_axis_total_neurons": "3",
                "cluster_y_axis_total_neurons": "3",
                "cluster_initial_x_axis_total_neurons": "2",
                "cluster_initial_y_axis_total_neurons": "2",
            },
        )
        cluster_node = next(
            node for node in result["nodes"] if node["typeName"] == "NeuronCluster"
        )
        cluster = cluster_node["details"]["cluster"]

        self.assertEqual(cluster_node["id"], "neuron_cluster")
        self.assertEqual(cluster["capacity"], [3, 3, 1])
        self.assertEqual(cluster["initial"], [2, 2, 1])
        self.assertEqual(cluster["instantiated"], 4)
        self.assertEqual(
            cluster["coordinates"],
            [[1, 1, 1], [1, 2, 1], [2, 1, 1], [2, 2, 1]],
        )
        self.assertNotIn("recurrent", cluster_node["details"])

    def test_graph_serializer_reports_terminal_reachable_area(self) -> None:
        result = inspect_model("neuron/neuron_linear", "baseline")
        terminal_node = next(
            node for node in result["nodes"] if node["typeName"] == "Terminal"
        )
        reach = terminal_node["details"]["terminalReach"]

        self.assertEqual(reach["position"], [1, 1, 1])
        self.assertEqual(reach["total"], len(reach["connections"]))
        self.assertIn([1, 1, 1], reach["connections"])
        self.assertTrue(
            all(len(coordinate) == 3 for coordinate in reach["connections"])
        )

    def test_graph_serializer_skips_uninitialized_lazy_parameters(self) -> None:
        nodes, _edges = serialize_graph(nn.Sequential(nn.LazyLinear(3)))
        count_by_id = {node["id"]: node["parameterCount"] for node in nodes}
        details_by_id = {node["id"]: node["details"] for node in nodes}

        self.assertEqual(count_by_id["__root__"], 0)
        self.assertEqual(count_by_id["0"], 0)
        self.assertNotIn("weightShape", details_by_id["0"])
        self.assertNotIn("biasShape", details_by_id["0"])

    def test_graph_serializer_outputs_unique_node_and_edge_ids(self) -> None:
        for model in discover_models():
            with self.subTest(model=model):
                preset = list_model_presets(model)[0]["name"]
                result = inspect_model(model, preset)
                node_counts = Counter(node["id"] for node in result["nodes"])
                edge_counts = Counter(edge["id"] for edge in result["edges"])
                self.assertEqual(
                    [],
                    [node_id for node_id, count in node_counts.items() if count > 1],
                )
                self.assertEqual(
                    [],
                    [edge_id for edge_id, count in edge_counts.items() if count > 1],
                )


if __name__ == "__main__":
    unittest.main()
