from __future__ import annotations

import json
import os
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from typing import Any, TypeAlias

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import torch
from emperor.augmentations.adaptive_parameters.config import (
    AdaptiveParameterAugmentationConfig,
)
from emperor.base.layer import (
    Layer,
    LayerStack,
)
from emperor.base.layer.config import (
    LayerConfig,
    LayerStackConfig,
    RecurrentLayerConfig,
)
from emperor.base.layer.gate import GateConfig, LayerGateOptions
from emperor.base.layer.residual import ResidualConnectionOptions
from emperor.base.options import (
    ActivationOptions,
    LastLayerBiasOptions,
    LayerNormPositionOptions,
)
from emperor.linears.core.config import AdaptiveLinearLayerConfig, LinearLayerConfig
from torch import nn

from viewer.backend.inspector.discovery import discover_models, list_model_presets
from viewer.backend.inspector.errors import InspectorError
from viewer.backend.inspector.graph import serialize_graph
from viewer.backend.inspector.service import build_config, inspect_model
from viewer.backend.log_runs import LogRunIndex
from viewer.backend.repositories.log_runs import LogRunRepository
from viewer.backend.services.inspection import InspectionService
from viewer.backend.tests.helpers import write_tensorboard_run

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


class SharedParameterBranch(nn.Module):
    def __init__(self, parameter: nn.Parameter) -> None:
        super().__init__()
        self.weight = parameter


class SharedParameterGraph(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        shared_parameter = nn.Parameter(torch.ones(2, 3))
        self.left = SharedParameterBranch(shared_parameter)
        self.right = SharedParameterBranch(shared_parameter)


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
            "parameterSizeBytes",
            "details",
            "config",
        }

        for node in nodes:
            with self.subTest(node=node["id"]):
                self.assertEqual(set(node), expected_node_keys)
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
                        residual_connection_option=ResidualConnectionOptions.DISABLED,
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
        with self.assertRaises(InspectorError) as raised:
            inspect_model("memory/memory_linear", "gated-residual")

        self.assertIn("Unknown model", raised.exception.detail)

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
        parts, _option, cfg = build_config("linears/linear", "baseline")
        model = parts.model_type(cfg)
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

    def test_inspection_service_uses_saved_log_run_params(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logs_root = os.path.join(tmp, "logs")
            run_dir = write_tensorboard_run(
                Path(logs_root),
                [
                    "exp_linear",
                    "linears",
                    "linear",
                    "BASELINE",
                    "Mnist",
                    "run_20260601_010203",
                    "version_0",
                ],
            )
            (run_dir / "result.json").write_text(
                json.dumps({"params": {"stack_hidden_dim": 12}}),
                encoding="utf-8",
            )
            run = LogRunIndex(logs_root=logs_root).list_runs()[0]
            service = InspectionService(
                LogRunRepository(LogRunIndex(logs_root=logs_root))
            )

            default_result = service.inspect(
                model_type="linears",
                model="linear",
                preset="baseline",
                dataset="Mnist",
                overrides={},
            )
            historical_result = service.inspect(
                model_type="linears",
                model="linear",
                preset="baseline",
                dataset="Mnist",
                overrides={},
                log_run_id=run.id,
            )
            expected_result = inspect_model(
                "linears/linear",
                "baseline",
                {"stack_hidden_dim": 12},
                dataset="Mnist",
            )

        self.assertNotEqual(
            historical_result["parameterCount"],
            default_result["parameterCount"],
        )
        self.assertEqual(
            historical_result["parameterCount"],
            expected_result["parameterCount"],
        )

    def test_inspection_service_ignores_unknown_saved_log_run_params(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logs_root = os.path.join(tmp, "logs")
            run_dir = write_tensorboard_run(
                Path(logs_root),
                [
                    "exp_linear",
                    "linears",
                    "linear",
                    "BASELINE",
                    "Cifar10",
                    "run_20260601_010203",
                    "version_0",
                ],
            )
            (run_dir / "result.json").write_text(
                json.dumps(
                    {
                        "params": {
                            "input_dim": 3072,
                            "output_dim": 10,
                            "stack_hidden_dim": 12,
                            "gather_frequency_flag": False,
                        }
                    }
                ),
                encoding="utf-8",
            )
            run = LogRunIndex(logs_root=logs_root).list_runs()[0]
            service = InspectionService(
                LogRunRepository(LogRunIndex(logs_root=logs_root))
            )

            historical_result = service.inspect(
                model_type="linears",
                model="linear",
                preset="baseline",
                dataset="Cifar10",
                overrides={},
                log_run_id=run.id,
            )
            expected_result = inspect_model(
                "linears/linear",
                "baseline",
                {
                    "input_dim": 3072,
                    "output_dim": 10,
                    "stack_hidden_dim": 12,
                },
                dataset="Cifar10",
            )

        self.assertEqual(
            historical_result["parameterCount"],
            expected_result["parameterCount"],
        )

    def test_inspection_service_keeps_request_overrides_strict_with_log_run(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logs_root = os.path.join(tmp, "logs")
            write_tensorboard_run(
                Path(logs_root),
                [
                    "exp_linear",
                    "linears",
                    "linear",
                    "BASELINE",
                    "Mnist",
                    "run_20260601_010203",
                    "version_0",
                ],
            )
            run = LogRunIndex(logs_root=logs_root).list_runs()[0]
            service = InspectionService(
                LogRunRepository(LogRunIndex(logs_root=logs_root))
            )

            with self.assertRaises(InspectorError) as raised:
                service.inspect(
                    model_type="linears",
                    model="linear",
                    preset="baseline",
                    dataset="Mnist",
                    overrides={"gather_frequency_flag": False},
                    log_run_id=run.id,
                )

        self.assertIn(
            "Unknown override 'gather_frequency_flag'",
            raised.exception.detail,
        )

    def test_inspection_service_rejects_mismatched_log_run_identity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logs_root = os.path.join(tmp, "logs")
            write_tensorboard_run(
                Path(logs_root),
                [
                    "exp_linear",
                    "linears",
                    "linear",
                    "BASELINE",
                    "Mnist",
                    "run_20260601_010203",
                    "version_0",
                ],
            )
            run = LogRunIndex(logs_root=logs_root).list_runs()[0]
            service = InspectionService(
                LogRunRepository(LogRunIndex(logs_root=logs_root))
            )

            with self.assertRaises(InspectorError) as raised:
                service.inspect(
                    model_type="linears",
                    model="linear",
                    preset="baseline",
                    dataset="Cifar10",
                    overrides={},
                    log_run_id=run.id,
                )

        self.assertIn("belongs to dataset 'Mnist'", raised.exception.detail)

    def test_inspection_service_preserves_preset_inspect_without_log_run(self) -> None:
        service = InspectionService()

        self.assertEqual(
            service.inspect(
                model_type="linears",
                model="linear",
                preset="baseline",
                dataset="Mnist",
                overrides={},
            ),
            inspect_model("linears/linear", "baseline", {}, dataset="Mnist"),
        )

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
            residual_connection_option=ResidualConnectionOptions.DISABLED,
            dropout_probability=0.1,
            layer_norm_position=LayerNormPositionOptions.BEFORE,
            gate_config=None,
            halting_config=None,
            memory_config=None,
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
                "residual_connection_option",
                "dropout_probability",
                "layer_norm_position",
                "gate_config",
                "halting_config",
                "memory_config",
                "layer_model_config",
            ],
        )
        self.assertEqual(config_fields(root)["activation"], "GELU")
        self.assertEqual(config_fields(root)["layer_norm_position"], "BEFORE")
        self.assertEqual(config_fields(root)["layer_model_config"], "LinearLayerConfig")
        self.assertIsNone(config_fields(root)["gate_config"])
        self.assertIsNone(config_fields(root)["halting_config"])
        self.assertIsNone(config_fields(root)["memory_config"])

    def test_graph_serializer_reports_layer_gate_option_details(self) -> None:
        gate_model_config = LayerStackConfig(
            input_dim=4,
            hidden_dim=4,
            output_dim=4,
            num_layers=1,
            last_layer_bias_option=LastLayerBiasOptions.DEFAULT,
            apply_output_pipeline_flag=False,
            layer_config=LayerConfig(
                input_dim=4,
                output_dim=4,
                activation=ActivationOptions.DISABLED,
                residual_connection_option=ResidualConnectionOptions.DISABLED,
                dropout_probability=0.0,
                layer_norm_position=LayerNormPositionOptions.DISABLED,
                gate_config=None,
                halting_config=None,
                memory_config=None,
                layer_model_config=LinearLayerConfig(
                    input_dim=4,
                    output_dim=4,
                    bias_flag=True,
                ),
            ),
        )
        cfg = LayerConfig(
            input_dim=4,
            output_dim=4,
            activation=ActivationOptions.DISABLED,
            residual_connection_option=ResidualConnectionOptions.DISABLED,
            dropout_probability=0.0,
            layer_norm_position=LayerNormPositionOptions.DISABLED,
            gate_config=GateConfig(
                model_config=gate_model_config,
                option=LayerGateOptions.MULTIPLIER,
            ),
            halting_config=None,
            memory_config=None,
            layer_model_config=LinearLayerConfig(
                input_dim=4,
                output_dim=4,
                bias_flag=True,
            ),
        )

        nodes, _edges = serialize_graph(Layer(cfg))
        root = nodes[0]

        self.assertEqual(root["details"]["gateOption"], "MULTIPLIER")
        self.assertTrue(root["details"]["gate"])

    def test_graph_serializer_reports_recurrent_gate_option_details(self) -> None:
        block_config = LayerConfig(
            input_dim=4,
            output_dim=4,
            activation=ActivationOptions.DISABLED,
            residual_connection_option=ResidualConnectionOptions.DISABLED,
            dropout_probability=0.0,
            layer_norm_position=LayerNormPositionOptions.DISABLED,
            gate_config=None,
            halting_config=None,
            memory_config=None,
            layer_model_config=LinearLayerConfig(
                input_dim=4,
                output_dim=4,
                bias_flag=True,
            ),
        )
        cfg = RecurrentLayerConfig(
            input_dim=4,
            output_dim=4,
            max_steps=2,
            recurrent_layer_norm_position=LayerNormPositionOptions.AFTER,
            block_config=block_config,
            gate_config=GateConfig(
                model_config=LayerStackConfig(
                    input_dim=4,
                    hidden_dim=4,
                    output_dim=4,
                    num_layers=1,
                    last_layer_bias_option=LastLayerBiasOptions.DEFAULT,
                    apply_output_pipeline_flag=False,
                    layer_config=block_config,
                ),
                option=LayerGateOptions.MULTIPLIER,
            ),
            residual_connection_option=ResidualConnectionOptions.DISABLED,
        )

        nodes, _edges = serialize_graph(cfg.build())
        root = nodes[0]

        self.assertEqual(root["details"]["recurrent"]["gateOption"], "MULTIPLIER")
        self.assertEqual(root["details"]["recurrent"]["layerNorm"], "AFTER")
        self.assertTrue(root["details"]["recurrent"]["gate"])

    def test_graph_serializer_omits_disabled_layer_residual_module(self) -> None:
        def layer_config(
            residual_connection_option: ResidualConnectionOptions,
        ) -> LayerConfig:
            return LayerConfig(
                input_dim=4,
                output_dim=4,
                activation=ActivationOptions.DISABLED,
                residual_connection_option=residual_connection_option,
                dropout_probability=0.0,
                layer_norm_position=LayerNormPositionOptions.DISABLED,
                gate_config=None,
                halting_config=None,
                memory_config=None,
                layer_model_config=LinearLayerConfig(
                    input_dim=4,
                    output_dim=4,
                    bias_flag=True,
                ),
            )

        disabled_layer = Layer(layer_config(ResidualConnectionOptions.DISABLED))
        enabled_layer = Layer(layer_config(ResidualConnectionOptions.RESIDUAL))

        disabled_nodes, _disabled_edges = serialize_graph(disabled_layer)
        enabled_nodes, _enabled_edges = serialize_graph(enabled_layer)

        self.assertIsNone(disabled_layer.residual_connection)
        self.assertNotIn(
            "ResidualConnection",
            {node["typeName"] for node in disabled_nodes},
        )
        self.assertIn(
            "ResidualConnection",
            {node["typeName"] for node in enabled_nodes},
        )

    def test_graph_serializer_preserves_layer_stack_config_on_layer_stack(self) -> None:
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
                residual_connection_option=ResidualConnectionOptions.DISABLED,
                dropout_probability=0.0,
                layer_norm_position=LayerNormPositionOptions.DISABLED,
                gate_config=None,
                halting_config=None,
                memory_config=None,
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

        self.assertIsInstance(model, LayerStack)
        self.assertEqual(root["config"]["typeName"], "LayerStackConfig")
        self.assertEqual(config_fields(root)["input_dim"], 4)
        self.assertEqual(config_fields(root)["hidden_dim"], 5)
        self.assertEqual(config_fields(root)["output_dim"], 2)
        self.assertEqual(config_fields(root)["num_layers"], 2)
        self.assertEqual(config_fields(root)["last_layer_bias_option"], "DISABLED")
        self.assertTrue(config_fields(root)["apply_output_pipeline_flag"])
        self.assertIsNone(config_fields(root)["shared_gate_config"])
        self.assertEqual(config_fields(root)["layer_config"], "LayerConfig")

    def test_graph_serializer_reports_shared_gate_on_layer_details(self) -> None:
        gate_config = LayerStackConfig(
            input_dim=4,
            hidden_dim=4,
            output_dim=4,
            num_layers=1,
            last_layer_bias_option=LastLayerBiasOptions.DEFAULT,
            apply_output_pipeline_flag=False,
            layer_config=LayerConfig(
                input_dim=4,
                output_dim=4,
                activation=ActivationOptions.DISABLED,
                residual_connection_option=ResidualConnectionOptions.DISABLED,
                dropout_probability=0.0,
                layer_norm_position=LayerNormPositionOptions.DISABLED,
                gate_config=None,
                halting_config=None,
                memory_config=None,
                layer_model_config=LinearLayerConfig(
                    input_dim=4,
                    output_dim=4,
                    bias_flag=True,
                ),
            ),
        )
        stack_config = LayerStackConfig(
            input_dim=5,
            hidden_dim=4,
            output_dim=4,
            num_layers=3,
            last_layer_bias_option=LastLayerBiasOptions.DEFAULT,
            apply_output_pipeline_flag=True,
            shared_gate_config=GateConfig(
                model_config=gate_config,
                option=LayerGateOptions.MULTIPLIER,
            ),
            layer_config=LayerConfig(
                input_dim=5,
                output_dim=4,
                activation=ActivationOptions.DISABLED,
                residual_connection_option=ResidualConnectionOptions.DISABLED,
                dropout_probability=0.0,
                layer_norm_position=LayerNormPositionOptions.DISABLED,
                gate_config=None,
                halting_config=None,
                memory_config=None,
                layer_model_config=LinearLayerConfig(
                    input_dim=5,
                    output_dim=4,
                    bias_flag=True,
                ),
            ),
        )

        nodes, _edges = serialize_graph(stack_config.build())
        node_by_id = nodes_by_id(nodes)

        self.assertEqual(
            config_fields(node_by_id["__root__"])["shared_gate_config"],
            "GateConfig",
        )
        for node_id in ("layers.0", "layers.1", "layers.2"):
            with self.subTest(node_id=node_id):
                self.assertTrue(node_by_id[node_id]["details"]["gate"])

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

    def test_inspect_experts_then_linear_keeps_linear_graph_identity(self) -> None:
        experts_result = inspect_model("experts/experts_linear", "baseline")
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

    def test_graph_serializer_reports_neuron_cluster_grid(self) -> None:
        result = inspect_model(
            "neuron/neuron_linear",
            "baseline",
            {
                "cluster_x_axis_total_neurons": "3",
                "cluster_y_axis_total_neurons": "3",
                "cluster_z_axis_total_neurons": "1",
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
        self.assertEqual(cluster["initialStart"], [1, 1, 1])
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

        self.assertEqual(
            terminal_node["id"],
            "neuron_cluster.cluster.neuron_4_4_1.terminal",
        )
        self.assertEqual(reach["position"], [4, 4, 1])
        self.assertEqual(reach["total"], len(reach["connections"]))
        self.assertIn([4, 4, 1], reach["connections"])
        self.assertIn([3, 3, 1], reach["connections"])
        self.assertTrue(
            all(len(coordinate) == 3 for coordinate in reach["connections"])
        )

    def test_graph_serializer_skips_uninitialized_lazy_parameters(self) -> None:
        nodes, _edges = serialize_graph(nn.Sequential(nn.LazyLinear(3)))
        count_by_id = {node["id"]: node["parameterCount"] for node in nodes}
        size_by_id = {node["id"]: node["parameterSizeBytes"] for node in nodes}
        details_by_id = {node["id"]: node["details"] for node in nodes}

        self.assertEqual(count_by_id["__root__"], 0)
        self.assertEqual(count_by_id["0"], 0)
        self.assertEqual(size_by_id["__root__"], 0)
        self.assertEqual(size_by_id["0"], 0)
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
