from __future__ import annotations

import json
import os
import tempfile
import unittest
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeAlias
from unittest import mock

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import models.experts.linear.config as experts_linear_config
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

from workbench.backend.inspector.checkpoint_shapes import (
    MAX_CHECKPOINT_GRAPH_SHAPE_BYTES,
    checkpoint_graph_shapes_from_state_dict,
)
from workbench.backend.inspector.discovery import discover_models, list_model_presets
from workbench.backend.inspector.errors import InspectorError
from workbench.backend.inspector.graph import serialize_graph
from workbench.backend.inspector.service import build_config, inspect_model
from workbench.backend.log_runs import LogRunIndex
from workbench.backend.repositories.log_runs import LogRunRepository
from workbench.backend.services.inspection import InspectionService
from workbench.backend.tests.helpers import write_tensorboard_run

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


@dataclass
class PlainConfig:
    width: int


class DocumentedModule(nn.Module):
    """Documented component description."""


def checkpoint_state_dict(
    *,
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    layer_count: int,
    stack_prefix: str = "main_model.layers",
) -> dict[str, torch.Tensor]:
    state_dict: dict[str, torch.Tensor] = {
        "input_model.model.weight_params": torch.zeros(input_dim, hidden_dim),
        "input_model.model.bias_params": torch.zeros(hidden_dim),
        "output_model.model.weight_params": torch.zeros(hidden_dim, output_dim),
        "output_model.model.bias_params": torch.zeros(output_dim),
    }
    for index in range(layer_count):
        state_dict[f"{stack_prefix}.{index}.model.weight_params"] = torch.zeros(
            hidden_dim,
            hidden_dim,
        )
        state_dict[f"{stack_prefix}.{index}.model.bias_params"] = torch.zeros(
            hidden_dim,
        )
    return state_dict


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
                json.dumps({"params": {"hidden_dim": 12}}),
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
                {"hidden_dim": 12},
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

    def test_inspection_service_uses_checkpoint_shapes_for_log_run_graph(
        self,
    ) -> None:
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
                json.dumps(
                    {
                        "params": {
                            "input_dim": 20,
                            "output_dim": 9,
                            "hidden_dim": 12,
                            "stack_num_layers": 4,
                        },
                        "metrics": {},
                    }
                ),
                encoding="utf-8",
            )
            (run_dir / "checkpoints" / "last.ckpt").write_text(
                "placeholder",
                encoding="utf-8",
            )
            torch.save(
                {
                    "state_dict": checkpoint_state_dict(
                        input_dim=20,
                        hidden_dim=12,
                        output_dim=9,
                        layer_count=4,
                    )
                },
                run_dir / "checkpoints" / "epoch=0-step=1.ckpt",
            )
            torch.save(
                {
                    "state_dict": checkpoint_state_dict(
                        input_dim=14,
                        hidden_dim=32,
                        output_dim=6,
                        layer_count=3,
                    )
                },
                run_dir / "checkpoints" / "epoch=2-step=300.ckpt",
            )
            run = LogRunIndex(logs_root=logs_root).list_runs()[0]
            service = InspectionService(
                LogRunRepository(LogRunIndex(logs_root=logs_root))
            )

            preset_result = service.inspect(
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
            checkpoint_result = inspect_model(
                "linears/linear",
                "baseline",
                {
                    "input_dim": 14,
                    "output_dim": 6,
                    "hidden_dim": 32,
                    "stack_num_layers": 3,
                },
                dataset="Mnist",
            )
            stale_result = inspect_model(
                "linears/linear",
                "baseline",
                {
                    "input_dim": 20,
                    "output_dim": 9,
                    "hidden_dim": 12,
                    "stack_num_layers": 4,
                },
                dataset="Mnist",
            )

        node_by_id = nodes_by_id(historical_result["nodes"])

        self.assertEqual(
            preset_result,
            inspect_model("linears/linear", "baseline", {}, dataset="Mnist"),
        )
        self.assertEqual(
            historical_result["parameterCount"],
            checkpoint_result["parameterCount"],
        )
        self.assertNotEqual(
            historical_result["parameterCount"],
            stale_result["parameterCount"],
        )
        self.assertEqual(node_by_id["input_model"]["details"]["dims"], "14 -> 32")
        self.assertEqual(node_by_id["main_model"]["details"]["numLayers"], 3)
        self.assertEqual(
            node_by_id["main_model.layers.0"]["details"]["dims"],
            "32 -> 32",
        )
        self.assertEqual(node_by_id["output_model"]["details"]["dims"], "32 -> 6")
        self.assertNotIn("main_model.layers.3", node_by_id)
        self.assertEqual(
            node_by_id["input_model.model"]["details"]["weightShape"],
            "14 x 32",
        )
        self.assertEqual(
            node_by_id["input_model.model"]["details"]["biasShape"],
            "32",
        )
        self.assertEqual(
            node_by_id["main_model.layers.2.model"]["details"]["weightShape"],
            "32 x 32",
        )
        self.assertEqual(
            node_by_id["output_model.model"]["details"]["weightShape"],
            "32 x 6",
        )
        self.assertEqual(
            node_by_id["output_model.model"]["details"]["biasShape"],
            "6",
        )

    def test_checkpoint_shape_extractor_counts_recurrent_block_layers(self) -> None:
        checkpoint_shapes = checkpoint_graph_shapes_from_state_dict(
            checkpoint_state_dict(
                input_dim=8,
                hidden_dim=16,
                output_dim=4,
                layer_count=2,
                stack_prefix="main_model.block_model.layers",
            )
        )

        self.assertIsNotNone(checkpoint_shapes)
        if checkpoint_shapes is None:
            self.fail("Expected checkpoint shape extraction to succeed.")
        self.assertEqual(
            checkpoint_shapes.config_overrides,
            {
                "input_dim": 8,
                "output_dim": 4,
                "hidden_dim": 16,
                "stack_num_layers": 2,
            },
        )

    def test_checkpoint_shape_extractor_infers_optional_controller_counts(
        self,
    ) -> None:
        state_dict = checkpoint_state_dict(
            input_dim=8,
            hidden_dim=16,
            output_dim=4,
            layer_count=2,
        )
        for outer_index in range(2):
            for gate_index in range(3):
                state_dict[
                    "main_model.layers."
                    f"{outer_index}.gate_model.model.layers.{gate_index}."
                    "model.weight_params"
                ] = torch.zeros(16, 16)
            for halting_index in range(4):
                state_dict[
                    "main_model.layers."
                    f"{outer_index}.halting_model.halting_gate_model.layers."
                    f"{halting_index}.model.weight_params"
                ] = torch.zeros(16, 2 if halting_index == 3 else 16)
            for memory_index in range(2):
                state_dict[
                    "main_model.layers."
                    f"{outer_index}.memory_model.memory_model.layers."
                    f"{memory_index}.model.weight_params"
                ] = torch.zeros(16, 16)

        checkpoint_shapes = checkpoint_graph_shapes_from_state_dict(state_dict)

        self.assertIsNotNone(checkpoint_shapes)
        if checkpoint_shapes is None:
            self.fail("Expected checkpoint shape extraction to succeed.")
        self.assertEqual(checkpoint_shapes.config_overrides["stack_num_layers"], 2)
        self.assertTrue(checkpoint_shapes.config_overrides["stack_gate_flag"])
        self.assertEqual(
            checkpoint_shapes.config_overrides["gate_stack_num_layers"],
            3,
        )
        self.assertTrue(checkpoint_shapes.config_overrides["stack_halting_flag"])
        self.assertEqual(
            checkpoint_shapes.config_overrides["halting_stack_num_layers"],
            4,
        )
        self.assertTrue(checkpoint_shapes.config_overrides["memory_flag"])
        self.assertEqual(
            checkpoint_shapes.config_overrides["memory_stack_num_layers"],
            2,
        )

    def test_checkpoint_shape_extractor_maps_boundary_generator_counts_to_global(
        self,
    ) -> None:
        state_dict = checkpoint_state_dict(
            input_dim=8,
            hidden_dim=16,
            output_dim=4,
            layer_count=2,
        )
        for prefix in (
            "input_model.model.adaptive_behaviour.weight_generator.model",
            "output_model.model.adaptive_behaviour.weight_generator.model",
        ):
            for layer_index in range(3):
                state_dict[
                    f"{prefix}.layers.{layer_index}.model.weight_params"
                ] = torch.zeros(16, 16)

        checkpoint_shapes = checkpoint_graph_shapes_from_state_dict(state_dict)

        self.assertIsNotNone(checkpoint_shapes)
        if checkpoint_shapes is None:
            self.fail("Expected checkpoint shape extraction to succeed.")
        self.assertEqual(
            checkpoint_shapes.config_overrides["adaptive_generator_stack_num_layers"],
            3,
        )
        self.assertEqual(
            [
                key
                for key in checkpoint_shapes.config_overrides
                if key.endswith("_layer_adaptive_generator_stack_num_layers")
            ],
            [],
        )

    def test_checkpoint_shape_extractor_infers_controller_hidden_dims_safely(
        self,
    ) -> None:
        state_dict = checkpoint_state_dict(
            input_dim=8,
            hidden_dim=16,
            output_dim=4,
            layer_count=2,
        )
        for outer_index in range(2):
            state_dict[
                "main_model.layers."
                f"{outer_index}.gate_model.model.layers.0.model.weight_params"
            ] = torch.zeros(16, 32)
            state_dict[
                "main_model.layers."
                f"{outer_index}.gate_model.model.layers.1.model.weight_params"
            ] = torch.zeros(32, 16)
            state_dict[
                "main_model.layers."
                f"{outer_index}.memory_model.memory_model.layers.0."
                "model.weight_params"
            ] = torch.zeros(16, 32)
            state_dict[
                "main_model.layers."
                f"{outer_index}.memory_model.memory_model.layers.1."
                "model.weight_params"
            ] = torch.zeros(32, 16)
            state_dict[
                "main_model.layers."
                f"{outer_index}.halting_model.halting_gate_model.layers.0."
                "model.weight_params"
            ] = torch.zeros(16, 24)
            state_dict[
                "main_model.layers."
                f"{outer_index}.halting_model.halting_gate_model.layers.2."
                "model.weight_params"
            ] = torch.zeros(24, 2)

        checkpoint_shapes = checkpoint_graph_shapes_from_state_dict(state_dict)

        self.assertIsNotNone(checkpoint_shapes)
        if checkpoint_shapes is None:
            self.fail("Expected checkpoint shape extraction to succeed.")
        self.assertTrue(
            checkpoint_shapes.config_overrides["gate_stack_independent_flag"]
        )
        self.assertEqual(
            checkpoint_shapes.config_overrides["gate_stack_hidden_dim"],
            32,
        )
        self.assertEqual(checkpoint_shapes.config_overrides["gate_stack_num_layers"], 2)
        self.assertTrue(
            checkpoint_shapes.config_overrides["memory_stack_independent_flag"]
        )
        self.assertEqual(
            checkpoint_shapes.config_overrides["memory_stack_hidden_dim"],
            32,
        )
        self.assertEqual(
            checkpoint_shapes.config_overrides["memory_stack_num_layers"],
            2,
        )
        self.assertEqual(
            checkpoint_shapes.config_overrides["submodule_stack_hidden_dim"],
            32,
        )
        self.assertNotIn(
            "halting_stack_hidden_dim",
            checkpoint_shapes.config_overrides,
        )
        self.assertNotIn(
            "halting_stack_num_layers",
            checkpoint_shapes.config_overrides,
        )
        self.assertIn(
            "halting_stack_num_layers:nonContiguous",
            checkpoint_shapes.diagnostics.structural_fallback_reasons,
        )

    def test_checkpoint_shape_extractor_infers_expert_and_sampler_counts(
        self,
    ) -> None:
        state_dict: dict[str, torch.Tensor] = {}
        for outer_index in range(2):
            for router_index in range(4):
                output_dim = 3 if router_index == 3 else 16
                state_dict[
                    "main_model.expert_stack.layers."
                    f"{outer_index}.model.sampler.router.model.layers."
                    f"{router_index}.model.weight_params"
                ] = torch.zeros(16, output_dim)
            state_dict[
                "main_model.expert_stack.layers."
                f"{outer_index}.model.sampler.router.model.layers.3."
                "model.bias_params"
            ] = torch.zeros(3)
            for expert_index in range(3):
                for layer_index in range(2):
                    state_dict[
                        "main_model.expert_stack.layers."
                        f"{outer_index}.model.expert_modules.{expert_index}."
                        f"layers.{layer_index}.model.weight_params"
                    ] = torch.zeros(16, 16)

        checkpoint_shapes = checkpoint_graph_shapes_from_state_dict(state_dict)

        self.assertIsNotNone(checkpoint_shapes)
        if checkpoint_shapes is None:
            self.fail("Expected checkpoint shape extraction to succeed.")
        self.assertEqual(checkpoint_shapes.config_overrides["stack_num_layers"], 2)
        self.assertEqual(
            checkpoint_shapes.config_overrides["expert_stack_num_layers"],
            2,
        )
        self.assertEqual(
            checkpoint_shapes.config_overrides["router_stack_num_layers"],
            4,
        )
        self.assertNotIn(
            "router_stack_independent_flag",
            checkpoint_shapes.config_overrides,
        )
        self.assertEqual(checkpoint_shapes.config_overrides["expert_num_experts"], 3)

    def test_checkpoint_shape_extractor_counts_transformer_layers(self) -> None:
        checkpoint_shapes = checkpoint_graph_shapes_from_state_dict(
            {
                "transformer.layers.0.model.feed_forward_model.model.layers.0."
                "model.weight_params": torch.zeros(8, 8),
                "transformer.layers.1.model.feed_forward_model.model.layers.0."
                "model.weight_params": torch.zeros(8, 8),
                "transformer.layers.2.model.feed_forward_model.model.layers.0."
                "model.weight_params": torch.zeros(8, 8),
            }
        )

        self.assertIsNotNone(checkpoint_shapes)
        if checkpoint_shapes is None:
            self.fail("Expected checkpoint shape extraction to succeed.")
        self.assertEqual(checkpoint_shapes.config_overrides["stack_num_layers"], 3)

    def test_checkpoint_shape_extractor_omits_non_contiguous_layer_counts(
        self,
    ) -> None:
        checkpoint_shapes = checkpoint_graph_shapes_from_state_dict(
            {
                "main_model.layers.0.model.weight_params": torch.zeros(8, 8),
                "main_model.layers.2.model.weight_params": torch.zeros(8, 8),
            }
        )

        self.assertIsNotNone(checkpoint_shapes)
        if checkpoint_shapes is None:
            self.fail("Expected checkpoint shape extraction to succeed.")
        self.assertNotIn("stack_num_layers", checkpoint_shapes.config_overrides)
        self.assertIn(
            "stack_num_layers:nonContiguous",
            checkpoint_shapes.diagnostics.structural_fallback_reasons,
        )

    def test_checkpoint_shape_extractor_maps_direct_tensor_shape_details(
        self,
    ) -> None:
        checkpoint_shapes = checkpoint_graph_shapes_from_state_dict(
            {
                "linear.model.weight_params": torch.zeros(32, 64),
                "linear.model.bias_params": torch.zeros(128),
                "token_embedding.weight": torch.zeros(11, 7),
                "conv.weight": torch.zeros(5, 3, 2, 2),
                "projection.weight": torch.zeros(3, 4),
                "batch.running_mean": torch.zeros(4),
                "batch.num_batches_tracked": torch.tensor(0),
            }
        )

        self.assertIsNotNone(checkpoint_shapes)
        if checkpoint_shapes is None:
            self.fail("Expected checkpoint shape extraction to succeed.")
        self.assertEqual(
            checkpoint_shapes.parameter_shapes["linear.model"]["weightShape"],
            "32 x 64",
        )
        self.assertEqual(
            checkpoint_shapes.parameter_shapes["linear.model"]["biasShape"],
            "128",
        )
        self.assertEqual(
            checkpoint_shapes.parameter_shapes["linear.model"]["inputDim"],
            32,
        )
        self.assertEqual(
            checkpoint_shapes.parameter_shapes["linear.model"]["outputDim"],
            64,
        )
        self.assertEqual(
            checkpoint_shapes.parameter_shapes["linear.model"]["dims"],
            "32 -> 64",
        )
        self.assertEqual(
            checkpoint_shapes.parameter_shapes["token_embedding"]["weightShape"],
            "11 x 7",
        )
        self.assertNotIn("dims", checkpoint_shapes.parameter_shapes["token_embedding"])
        self.assertEqual(
            checkpoint_shapes.parameter_shapes["token_embedding"]["numEmbeddings"],
            11,
        )
        self.assertEqual(
            checkpoint_shapes.parameter_shapes["token_embedding"]["embeddingDim"],
            7,
        )
        self.assertEqual(
            checkpoint_shapes.parameter_shapes["conv"]["kernelShape"],
            "2 x 2",
        )
        self.assertNotIn("dims", checkpoint_shapes.parameter_shapes["conv"])
        self.assertEqual(
            checkpoint_shapes.parameter_shapes["projection"]["weightShape"],
            "3 x 4",
        )
        self.assertNotIn("dims", checkpoint_shapes.parameter_shapes["projection"])
        self.assertEqual(
            checkpoint_shapes.parameter_shapes["batch"]["tensorShapes"],
            {
                "num_batches_tracked": "scalar",
                "running_mean": "4",
            },
        )

    def test_inspection_service_enables_safe_checkpoint_controller_structure(
        self,
    ) -> None:
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
            state_dict = checkpoint_state_dict(
                input_dim=8,
                hidden_dim=16,
                output_dim=4,
                layer_count=2,
            )
            for outer_index in range(2):
                for gate_index in range(2):
                    state_dict[
                        "main_model.layers."
                        f"{outer_index}.gate_model.model.layers.{gate_index}."
                        "model.weight_params"
                    ] = torch.zeros(16, 16)
            torch.save(
                {"state_dict": state_dict},
                run_dir / "checkpoints" / "epoch=0-step=1.ckpt",
            )
            run = LogRunIndex(logs_root=logs_root).list_runs()[0]
            service = InspectionService(
                LogRunRepository(LogRunIndex(logs_root=logs_root))
            )

            result = service.inspect(
                model_type="linears",
                model="linear",
                preset="baseline",
                dataset="Mnist",
                overrides={},
                log_run_id=run.id,
            )

        node_by_id = nodes_by_id(result["nodes"])
        self.assertEqual(node_by_id["main_model"]["details"]["numLayers"], 2)
        self.assertTrue(node_by_id["main_model.layers.0"]["details"]["gate"])
        self.assertIn("main_model.layers.0.gate_model", node_by_id)
        self.assertEqual(
            node_by_id["__root__"]["details"]["checkpoint"],
            {"status": "matched", "tensorCount": len(state_dict)},
        )
        self.assertEqual(
            node_by_id["input_model"]["details"]["checkpoint"]["status"],
            "matched",
        )
        self.assertEqual(
            node_by_id["loss_fn"]["details"]["checkpoint"],
            {
                "status": "missing",
                "tensorCount": 0,
                "reason": "noCheckpointTensor",
            },
        )

    def test_inspection_service_request_overrides_beat_checkpoint_structure(
        self,
    ) -> None:
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
            torch.save(
                {
                    "state_dict": checkpoint_state_dict(
                        input_dim=8,
                        hidden_dim=16,
                        output_dim=4,
                        layer_count=3,
                    )
                },
                run_dir / "checkpoints" / "epoch=0-step=1.ckpt",
            )
            run = LogRunIndex(logs_root=logs_root).list_runs()[0]
            service = InspectionService(
                LogRunRepository(LogRunIndex(logs_root=logs_root))
            )

            result = service.inspect(
                model_type="linears",
                model="linear",
                preset="baseline",
                dataset="Mnist",
                overrides={"hidden_dim": 32, "stack_num_layers": 1},
                log_run_id=run.id,
            )

        node_by_id = nodes_by_id(result["nodes"])
        self.assertEqual(node_by_id["main_model"]["details"]["numLayers"], 1)
        self.assertNotIn("main_model.layers.1", node_by_id)
        layer_details = node_by_id["main_model.layers.0.model"]["details"]
        self.assertEqual(layer_details["weightShape"], "16 x 16")
        self.assertEqual(layer_details["biasShape"], "16")
        self.assertEqual(layer_details["inputDim"], 16)
        self.assertEqual(layer_details["outputDim"], 16)
        self.assertEqual(layer_details["dims"], "16 -> 16")

    def test_inspection_service_uses_checkpoint_dims_for_controller_linear_node(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logs_root = os.path.join(tmp, "logs")
            run_dir = write_tensorboard_run(
                Path(logs_root),
                [
                    "exp_linear",
                    "linears",
                    "linear",
                    "GATING",
                    "Mnist",
                    "run_20260601_010203",
                    "version_0",
                ],
            )
            (run_dir / "result.json").write_text(
                json.dumps(
                    {
                        "params": {
                            "hidden_dim": 32,
                            "stack_num_layers": 1,
                            "submodule_stack_hidden_dim": 256,
                        }
                    }
                ),
                encoding="utf-8",
            )
            state_dict = {
                "main_model.layers.0.gate_model.model.layers.0."
                "model.weight_params": torch.zeros(32, 32),
                "main_model.layers.0.gate_model.model.layers.0."
                "model.bias_params": torch.zeros(32),
                "main_model.layers.0.gate_model.model.layers.1."
                "model.weight_params": torch.zeros(32, 32),
                "main_model.layers.0.gate_model.model.layers.1."
                "model.bias_params": torch.zeros(32),
            }
            torch.save(
                {"state_dict": state_dict},
                run_dir / "checkpoints" / "epoch=0-step=1.ckpt",
            )
            run = LogRunIndex(logs_root=logs_root).list_runs()[0]
            service = InspectionService(
                LogRunRepository(LogRunIndex(logs_root=logs_root))
            )

            result = service.inspect(
                model_type="linears",
                model="linear",
                preset="gating",
                dataset="Mnist",
                overrides={},
                log_run_id=run.id,
            )

        node_by_id = nodes_by_id(result["nodes"])
        gate_details = node_by_id[
            "main_model.layers.0.gate_model.model.layers.0.model"
        ]["details"]
        self.assertEqual(gate_details["weightShape"], "32 x 32")
        self.assertEqual(gate_details["biasShape"], "32")
        self.assertEqual(gate_details["inputDim"], 32)
        self.assertEqual(gate_details["outputDim"], 32)
        self.assertEqual(gate_details["dims"], "32 -> 32")

    def test_inspection_service_marks_locked_checkpoint_override_fallback(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logs_root = os.path.join(tmp, "logs")
            run_dir = write_tensorboard_run(
                Path(logs_root),
                [
                    "exp_linear",
                    "linears",
                    "linear",
                    "GATING",
                    "Mnist",
                    "run_20260601_010203",
                    "version_0",
                ],
            )
            state_dict = checkpoint_state_dict(
                input_dim=8,
                hidden_dim=16,
                output_dim=4,
                layer_count=1,
            )
            state_dict[
                "main_model.layers.0.gate_model.model.layers.0."
                "model.weight_params"
            ] = torch.zeros(16, 16)
            torch.save(
                {"state_dict": state_dict},
                run_dir / "checkpoints" / "epoch=0-step=1.ckpt",
            )
            run = LogRunIndex(logs_root=logs_root).list_runs()[0]
            service = InspectionService(
                LogRunRepository(LogRunIndex(logs_root=logs_root))
            )

            result = service.inspect(
                model_type="linears",
                model="linear",
                preset="gating",
                dataset="Mnist",
                overrides={},
                log_run_id=run.id,
            )

        root = nodes_by_id(result["nodes"])["__root__"]
        self.assertEqual(
            root["details"]["checkpoint"],
            {
                "status": "matched",
                "tensorCount": len(state_dict),
                "reason": "structuralFallback",
            },
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
                            "hidden_dim": 12,
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
                    "hidden_dim": 12,
                },
                dataset="Cifar10",
            )

        self.assertEqual(
            historical_result["parameterCount"],
            expected_result["parameterCount"],
        )

    def test_inspection_service_skips_oversized_checkpoint_shape_load(self) -> None:
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
                            "hidden_dim": 12,
                        }
                    }
                ),
                encoding="utf-8",
            )
            checkpoint_path = run_dir / "checkpoints" / "epoch=0-step=1.ckpt"
            os.truncate(checkpoint_path, MAX_CHECKPOINT_GRAPH_SHAPE_BYTES + 1)
            run = LogRunIndex(logs_root=logs_root).list_runs()[0]
            service = InspectionService(
                LogRunRepository(LogRunIndex(logs_root=logs_root))
            )

            with mock.patch(
                "workbench.backend.inspector.checkpoint_shapes.torch.load",
                side_effect=AssertionError("oversized checkpoint was loaded"),
            ) as torch_load:
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
                    "hidden_dim": 12,
                },
                dataset="Cifar10",
            )

        torch_load.assert_not_called()
        root_checkpoint = nodes_by_id(historical_result["nodes"])["__root__"][
            "details"
        ]["checkpoint"]
        self.assertEqual(
            historical_result["parameterCount"],
            expected_result["parameterCount"],
        )
        self.assertEqual(root_checkpoint["status"], "missing")
        self.assertEqual(root_checkpoint["tensorCount"], 0)
        self.assertEqual(root_checkpoint["reason"], "structuralFallback")
        self.assertTrue(
            any(
                reason.startswith("checkpointTooLarge:")
                for reason in root_checkpoint["fallbackReasons"]
            )
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
            linear_node["description"],
            (
                "Applies a learned linear projection with configured input/output "
                "dimensions and optional bias."
            ),
        )
        self.assertEqual(
            config_fields(linear_node),
            {"input_dim": 4, "output_dim": 3, "bias_flag": False},
        )
        field_by_key = {
            field["key"]: field for field in linear_node["config"]["fields"]
        }
        self.assertEqual(
            field_by_key["input_dim"]["description"],
            "Input feature dimension.",
        )
        self.assertEqual(
            field_by_key["output_dim"]["description"],
            "Output feature dimension.",
        )
        self.assertEqual(
            field_by_key["bias_flag"]["description"],
            "Add a learnable bias to the output.",
        )
        self.assertIn("weightShape", linear_node["details"])
        self.assertNotIn("biasShape", linear_node["details"])

    def test_graph_serializer_uses_docstring_description_fallback(self) -> None:
        nodes, _edges = serialize_graph(DocumentedModule())

        self.assertEqual(nodes[0]["description"], "Documented component description.")

    def test_graph_serializer_omits_unknown_descriptions_quietly(self) -> None:
        nodes, _edges = serialize_graph(ConfiguredModule(PlainConfig(width=7)))
        root = nodes[0]

        self.assertNotIn("description", root)
        self.assertEqual(
            root["config"],
            {
                "typeName": "PlainConfig",
                "fields": [{"key": "width", "value": 7}],
            },
        )

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
        result = inspect_model("experts/linear", "baseline")
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
                self.assertEqual(
                    node["details"]["numExperts"],
                    experts_linear_config.EXPERT_NUM_EXPERTS,
                )
                self.assertEqual(node["details"]["routingMode"], "LAYER")

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

    def test_graph_serializer_reports_neuron_cluster_grid(self) -> None:
        result = inspect_model(
            "neuron/linear",
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
        result = inspect_model("neuron/linear", "baseline")
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

    def test_graph_serializer_serializes_all_neuron_variants(self) -> None:
        for model_name in (
            "neuron/linear",
            "neuron/linear_adaptive",
            "neuron/expert_linear",
            "neuron/expert_linear_adaptive",
        ):
            with self.subTest(model_name=model_name):
                result = inspect_model(model_name, "baseline")
                type_names = {node["typeName"] for node in result["nodes"]}

                self.assertEqual(result["modelType"], "neuron")
                self.assertIn("NeuronCluster", type_names)

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
