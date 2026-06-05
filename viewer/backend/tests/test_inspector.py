from __future__ import annotations

import importlib
import asyncio
import json
import os
import tempfile
import unittest
from collections import Counter
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

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
from emperor.experiments.base import GridSearch, RandomSearch
from viewer.backend.inspector.discovery import (
    discover_models,
    list_model_datasets,
    list_model_monitors,
    list_model_presets,
)
from viewer.backend.inspector.errors import InspectorError
from viewer.backend.inspector.graph import serialize_graph
from viewer.backend.inspector.schema import config_schema, search_space_schema
from viewer.backend.inspector.service import inspect_model
from viewer.backend.inspector.search import parse_training_search
from viewer.backend.log_runs import LogRunDeleteFilters, LogRunIndex
from viewer.backend.training_jobs import TrainingJobManager
from viewer.backend.training_worker import search_mode_from_parsed_search


class FakeProcess:
    pid = 1234

    def __init__(self, exit_code: int | None = None) -> None:
        self.exit_code = exit_code
        self.terminated = False

    def poll(self) -> int | None:
        return self.exit_code

    def terminate(self) -> None:
        self.terminated = True
        self.exit_code = -15


class FakeRunner:
    def __init__(self, process: FakeProcess | None = None) -> None:
        self.process = process or FakeProcess()
        self.commands: list[list[str]] = []

    def start(self, command, *, cwd, env, log_path):
        self.commands.append(command)
        log_path.write_text("fake training log\n", encoding="utf-8")
        return self.process


class ConfiguredModule(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg


def config_fields(node: dict) -> dict[str, object]:
    config = node["config"]
    if config is None:
        return {}
    return {field["key"]: field["value"] for field in config["fields"]}


def write_tensorboard_run(
    logs_root: Path,
    relative_parts: list[str],
    *,
    scalars: dict[str, list[tuple[int, float]]] | None = None,
    metrics: dict[str, object] | None = None,
    hparams: bool = True,
    checkpoint: bool = True,
) -> Path:
    run_dir = logs_root.joinpath(*relative_parts)
    writer = SummaryWriter(log_dir=str(run_dir))
    for tag, points in (scalars or {"train/loss": [(1, 0.5)]}).items():
        for step, value in points:
            writer.add_scalar(tag, value, step)
    writer.flush()
    writer.close()

    if hparams:
        (run_dir / "hparams.yaml").write_text("batch_size: 4\n", encoding="utf-8")
    if checkpoint:
        checkpoint_dir = run_dir / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)
        (checkpoint_dir / "epoch=0-step=1.ckpt").write_text(
            "checkpoint", encoding="utf-8"
        )
    if metrics is not None:
        (run_dir / "result.json").write_text(
            json.dumps({"metrics": metrics}),
            encoding="utf-8",
        )
    return run_dir


def delete_filters_for_runs(
    runs,
    *,
    experiments: list[str] | None = None,
    datasets: list[str] | None = None,
    models: list[str] | None = None,
    presets: list[str] | None = None,
    run_ids: list[str] | None = None,
) -> LogRunDeleteFilters:
    return LogRunDeleteFilters(
        experiments=(
            experiments
            if experiments is not None
            else sorted({run.experiment for run in runs})
        ),
        datasets=(
            datasets if datasets is not None else sorted({run.dataset for run in runs})
        ),
        models=models if models is not None else sorted({run.model for run in runs}),
        presets=(
            presets if presets is not None else sorted({run.preset for run in runs})
        ),
        runIds=run_ids if run_ids is not None else sorted({run.id for run in runs}),
    )


class InspectorTests(unittest.TestCase):
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

    def test_config_schema_exposes_supported_field_types(self) -> None:
        linear_fields = {
            field["key"]: field for field in config_schema("linear")["fields"]
        }
        self.assertEqual(linear_fields["hidden_dim"]["type"], "int")
        self.assertEqual(linear_fields["hidden_dim"]["choices"], [])
        self.assertEqual(linear_fields["stack_num_layers"]["type"], "int")
        self.assertEqual(linear_fields["stack_num_layers"]["default"], 5)
        self.assertEqual(linear_fields["stack_num_layers"]["choices"], [])
        self.assertEqual(linear_fields["learning_rate"]["type"], "float")
        self.assertEqual(linear_fields["gate_flag"]["type"], "bool")
        self.assertEqual(linear_fields["stack_activation"]["type"], "enum")
        self.assertIn("GELU", linear_fields["stack_activation"]["choices"])
        self.assertEqual(linear_fields["hidden_dim"]["section"], "Layer Stack Options")
        self.assertEqual(
            linear_fields["recurrent_flag"]["section"],
            "Recurrent Layer Options",
        )

        vit_fields = {
            field["key"]: field for field in config_schema("vit_linear")["fields"]
        }
        self.assertEqual(vit_fields["positional_embedding_option"]["type"], "class")
        self.assertIn(
            "ImageLearnedPositionalEmbeddingConfig",
            vit_fields["positional_embedding_option"]["choices"],
        )
        self.assertNotIn(
            "AbsolutePositionalEmbeddingConfig",
            vit_fields["positional_embedding_option"]["choices"],
        )

    def test_config_schema_excludes_abstract_class_choices(self) -> None:
        fields = {
            field["key"]: field for field in config_schema("linear_adaptive")["fields"]
        }

        self.assertNotIn("DynamicWeightConfig", fields["weight_option"]["choices"])
        self.assertIn(
            "SingleModelDynamicWeightConfig",
            fields["weight_option"]["choices"],
        )
        self.assertNotIn("DynamicBiasConfig", fields["bias_option"]["choices"])
        self.assertIn(
            "AdditiveDynamicBiasConfig",
            fields["bias_option"]["choices"],
        )
        self.assertNotIn("DynamicDiagonalConfig", fields["diagonal_option"]["choices"])
        self.assertIn(
            "StandardDynamicDiagonalConfig",
            fields["diagonal_option"]["choices"],
        )
        self.assertNotIn("AxisMaskConfig", fields["row_mask_option"]["choices"])
        self.assertIn(
            "DiagonalAxisMaskConfig",
            fields["row_mask_option"]["choices"],
        )

    def test_config_schema_exposes_boundary_projector_choices(self) -> None:
        fields = {
            field["key"]: field for field in config_schema("linear_adaptive")["fields"]
        }

        self.assertEqual(
            fields["input_layer_model_option"]["section"],
            "Input Boundary Projector Options",
        )
        self.assertEqual(
            fields["output_layer_model_option"]["section"],
            "Output Boundary Projector Options",
        )
        self.assertEqual(fields["input_layer_model_option"]["type"], "class")
        self.assertTrue(fields["input_layer_model_option"]["nullable"])
        self.assertEqual(
            fields["input_layer_model_option"]["choices"],
            ["AdaptiveLinearLayerConfig"],
        )
        self.assertNotIn("searchChoices", fields["input_layer_model_option"])

    def test_config_schema_marks_preset_owned_fields_locked(self) -> None:
        baseline_fields = {
            field["key"]: field
            for field in config_schema("linear", "baseline")["fields"]
        }
        gating_fields = {
            field["key"]: field for field in config_schema("linear", "gating")["fields"]
        }

        self.assertFalse(baseline_fields["gate_flag"]["locked"])
        self.assertTrue(gating_fields["gate_flag"]["locked"])
        self.assertEqual(gating_fields["gate_flag"]["lockedValue"], True)
        self.assertIn("GATING preset", gating_fields["gate_flag"]["lockedReason"])

    def test_search_space_schema_exposes_linear_axes(self) -> None:
        axes = {
            axis["key"]: axis
            for axis in search_space_schema("linear", "baseline")["axes"]
        }

        self.assertIn("learning_rate", axes)
        self.assertIn("hidden_dim", axes)
        self.assertIn("stack_activation", axes)
        self.assertEqual(axes["hidden_dim"]["section"], "Layer Stack Options")
        self.assertEqual(axes["hidden_dim"]["type"], "int")
        self.assertEqual(axes["hidden_dim"]["values"], [16, 32, 64, 128, 256, 512])
        self.assertEqual(axes["stack_activation"]["type"], "enum")
        self.assertIn("GELU", axes["stack_activation"]["values"])
        self.assertFalse(axes["hidden_dim"]["locked"])

    def test_search_space_schema_marks_preset_owned_axes_locked(self) -> None:
        axes = {
            axis["key"]: axis
            for axis in search_space_schema("linear", "post-norm")["axes"]
        }

        self.assertTrue(axes["layer_norm_position"]["locked"])
        self.assertEqual(axes["layer_norm_position"]["lockedValue"], "AFTER")
        self.assertIn("POST_NORM preset", axes["layer_norm_position"]["lockedReason"])

    def test_override_parsing_changes_linear_hidden_dim_graph_details(self) -> None:
        result = inspect_model("linear", "baseline", {"hidden_dim": "128"})
        node_by_id = {node["id"]: node for node in result["nodes"]}
        main_layer_details = node_by_id["main_model.0"]["details"]
        output_layer_details = node_by_id["output_model"]["details"]

        self.assertEqual(main_layer_details["dims"], "128 -> 128")
        self.assertEqual(output_layer_details["dims"], "128 -> 10")
        self.assertNotIn("inputShape", main_layer_details)
        self.assertNotIn("outputShape", main_layer_details)
        self.assertNotIn("shapeTransition", main_layer_details)
        self.assertNotIn("inputShape", output_layer_details)
        self.assertNotIn("outputShape", output_layer_details)
        self.assertNotIn("shapeTransition", output_layer_details)

    def test_inspect_uses_selected_dataset_dimensions(self) -> None:
        result = inspect_model("linear", "baseline", dataset="Cifar100")
        node_by_id = {node["id"]: node for node in result["nodes"]}

        self.assertEqual(node_by_id["output_model"]["details"]["dims"], "256 -> 100")

    def test_locked_preset_override_is_rejected_for_inspect(self) -> None:
        with self.assertRaises(InspectorError) as context:
            inspect_model("linear", "gating", {"gate_flag": "false"})

        self.assertIn("locked fields", str(context.exception))
        self.assertIn("stack_gate_flag", str(context.exception))

    def test_inspect_response_includes_top_level_parameter_count(self) -> None:
        result = inspect_model("linear", "baseline")

        self.assertGreater(result["parameterCount"], 0)
        self.assertEqual(result["parameterCount"], result["nodes"][0]["parameterCount"])

    def test_inspect_reports_local_linear_weight_and_bias_shapes_on_owner(self) -> None:
        result = inspect_model("linear", "baseline")
        node_by_id = {node["id"]: node for node in result["nodes"]}

        self.assertNotIn("weightShape", node_by_id["main_model.0"]["details"])
        self.assertNotIn("biasShape", node_by_id["main_model.0"]["details"])
        self.assertEqual(
            node_by_id["main_model.0.model"]["details"]["weightShape"],
            "256 x 256",
        )
        self.assertEqual(
            node_by_id["main_model.0.model"]["details"]["biasShape"],
            "256",
        )

    def test_config_override_aliases_match_builder_parameter_names(self) -> None:
        result = inspect_model("linear_adaptive", "baseline", {"gate_flag": "true"})
        self.assertTrue(
            any(node["details"].get("gate") is True for node in result["nodes"])
        )

    def test_abstract_config_override_is_rejected_before_model_instantiation(
        self,
    ) -> None:
        with self.assertRaises(InspectorError) as context:
            inspect_model(
                "linear_adaptive",
                "baseline",
                {"row_mask_option": "AxisMaskConfig"},
            )

        message = str(context.exception)
        self.assertIn("Invalid value for override 'row_mask_option'", message)
        self.assertNotIn("Failed to instantiate model", message)

    def test_graph_serializer_uses_stable_paths_and_edges(self) -> None:
        result = inspect_model("linear", "baseline")
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
        result = inspect_model("linear", "baseline")
        role_by_id = {node["id"]: node["graphRole"] for node in result["nodes"]}

        self.assertEqual(role_by_id["loss_fn"], "runtime")
        self.assertEqual(role_by_id["metrics"], "runtime")
        self.assertEqual(role_by_id["metrics.train_accuracy"], "runtime")
        self.assertEqual(role_by_id["metrics.train_f1_score"], "runtime")

    def test_graph_serializer_marks_architecture_modules(self) -> None:
        result = inspect_model("linear", "baseline")
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
        result = inspect_model("experts_linear", "baseline")
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
            "neuron_linear",
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
        result = inspect_model("neuron_linear", "baseline")
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

    def test_api_health_and_inspect(self) -> None:
        import httpx
        from viewer.backend.api import app

        async def call_api() -> tuple[
            httpx.Response,
            httpx.Response,
            httpx.Response,
            httpx.Response,
        ]:
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport,
                base_url="http://testserver",
            ) as client:
                health = await client.get("/health")
                monitors = await client.get("/models/linear/monitors")
                search_space = await client.get(
                    "/models/linear/search-space?preset=baseline"
                )
                inspect_response = await client.post(
                    "/inspect",
                    json={
                        "model": "linear",
                        "preset": "baseline",
                        "dataset": "Mnist",
                        "overrides": {"hidden_dim": "128"},
                    },
                )
                return health, monitors, search_space, inspect_response

        health_response, monitors_response, search_space_response, response = (
            asyncio.run(call_api())
        )
        self.assertEqual(health_response.json(), {"status": "ok"})
        self.assertEqual(monitors_response.status_code, 200)
        self.assertEqual(monitors_response.json()["monitors"][0]["name"], "linear")
        self.assertEqual(search_space_response.status_code, 200)
        search_space_payload = search_space_response.json()
        self.assertIn(
            "hidden_dim", {axis["key"] for axis in search_space_payload["axes"]}
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["model"], "linear")
        self.assertTrue(payload["nodes"])
        self.assertTrue(payload["edges"])
        self.assertIn("parameterCount", payload)
        self.assertIn("parameterCount", payload["nodes"][0])

    def test_api_routes_declare_response_models(self) -> None:
        from fastapi.routing import APIRoute
        from viewer.backend.api import app

        missing = [
            f"{sorted(route.methods)} {route.path}"
            for route in app.routes
            if isinstance(route, APIRoute) and route.response_model is None
        ]

        self.assertEqual(missing, [])

    def test_api_rejects_extra_request_fields(self) -> None:
        import httpx
        from viewer.backend.api import app

        async def call_api() -> httpx.Response:
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport,
                base_url="http://testserver",
            ) as client:
                return await client.post(
                    "/inspect",
                    json={
                        "model": "linear",
                        "preset": "baseline",
                        "overrides": {},
                        "command": ["python", "-m", "viewer.backend.training_worker"],
                    },
                )

        response = asyncio.run(call_api())

        self.assertEqual(response.status_code, 422)
        self.assertIn("extra_forbidden", response.text)

    def test_api_dependency_overrides_can_replace_route_services(self) -> None:
        import httpx
        from viewer.backend.api import create_app
        from viewer.backend.dependencies import get_model_catalog_service

        class FakeModelCatalogService:
            def list_models(self) -> list[str]:
                return ["override_model"]

        async def override_model_catalog_service() -> FakeModelCatalogService:
            return FakeModelCatalogService()

        test_app = create_app()
        test_app.dependency_overrides[get_model_catalog_service] = (
            override_model_catalog_service
        )

        async def call_api() -> httpx.Response:
            transport = httpx.ASGITransport(app=test_app)
            async with httpx.AsyncClient(
                transport=transport,
                base_url="http://testserver",
            ) as client:
                return await client.get("/models")

        response = asyncio.run(call_api())

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"models": ["override_model"]})

    def test_training_api_response_does_not_expose_manager_internals(self) -> None:
        import httpx
        from viewer.backend.api import ViewerApiSettings, create_app

        with tempfile.TemporaryDirectory() as tmp:
            logs_root = Path(tmp) / "logs"
            manager = TrainingJobManager(
                root=Path(tmp) / "jobs",
                logs_root=logs_root,
                runner=FakeRunner(),
            )

            async def call_api() -> httpx.Response:
                transport = httpx.ASGITransport(
                    app=create_app(
                        ViewerApiSettings(logs_root=str(logs_root)),
                        training_manager=manager,
                    )
                )
                async with httpx.AsyncClient(
                    transport=transport,
                    base_url="http://testserver",
                ) as client:
                    return await client.post(
                        "/training/jobs",
                        json={
                            "model": "linear",
                            "preset": "baseline",
                            "presets": ["baseline", "gating"],
                            "datasets": ["Mnist"],
                            "overrides": {"hidden_dim": "128"},
                            "logFolder": "test_model",
                            "monitors": ["linear"],
                        },
                    )

            response = asyncio.run(call_api())

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "running")
        self.assertEqual(payload["preset"], "baseline")
        self.assertEqual(payload["presets"], ["baseline", "gating"])
        self.assertEqual(payload["pid"], 1234)
        for internal_key in ("command", "root", "process"):
            self.assertNotIn(internal_key, payload)

    def test_training_job_creation_uses_fake_process_runner(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runner = FakeRunner()
            logs_root = Path(tmp) / "logs"
            manager = TrainingJobManager(
                root=Path(tmp), logs_root=logs_root, runner=runner
            )

            payload = manager.create_job(
                model="linear",
                preset="baseline",
                datasets=["Mnist"],
                overrides={"hidden_dim": "128"},
                log_folder="test_model",
                monitors=["linear"],
            )
            payload_path = Path(tmp) / payload["id"] / "payload.json"
            worker_payload = json.loads(payload_path.read_text())

            self.assertEqual(payload["status"], "running")
            self.assertEqual(payload["preset"], "baseline")
            self.assertEqual(payload["presets"], ["baseline"])
            self.assertEqual(payload["datasets"], ["Mnist"])
            self.assertEqual(payload["monitors"], ["linear"])
            self.assertEqual(payload["logFolder"], "test_model")
            self.assertEqual(worker_payload["monitors"], ["linear"])
            self.assertEqual(worker_payload["preset"], "baseline")
            self.assertEqual(worker_payload["presets"], ["baseline"])
            self.assertEqual(worker_payload["logFolder"], "test_model")
            self.assertEqual(payload["pid"], 1234)
            self.assertTrue((logs_root / "test_model").is_dir())
            self.assertTrue(runner.commands)
            self.assertIn("viewer.backend.training_worker", runner.commands[0])

    def test_training_job_manager_active_jobs_excludes_terminal_statuses(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            running_manager = TrainingJobManager(
                root=root / "running",
                logs_root=root / "logs-running",
                runner=FakeRunner(FakeProcess()),
            )
            running_job = running_manager.create_job(
                model="linear",
                preset="baseline",
                datasets=["Mnist"],
                overrides={},
                log_folder="running_model",
                monitors=[],
            )
            self.assertEqual(
                running_manager.active_jobs(),
                [
                    {
                        "id": running_job["id"],
                        "status": "running",
                        "logFolder": "running_model",
                    }
                ],
            )

            completed_manager = TrainingJobManager(
                root=root / "completed",
                logs_root=root / "logs-completed",
                runner=FakeRunner(FakeProcess(exit_code=0)),
            )
            completed_manager.create_job(
                model="linear",
                preset="baseline",
                datasets=["Mnist"],
                overrides={},
                log_folder="completed_model",
                monitors=[],
            )
            self.assertEqual(completed_manager.active_jobs(), [])

            failed_manager = TrainingJobManager(
                root=root / "failed",
                logs_root=root / "logs-failed",
                runner=FakeRunner(FakeProcess(exit_code=1)),
            )
            failed_manager.create_job(
                model="linear",
                preset="baseline",
                datasets=["Mnist"],
                overrides={},
                log_folder="failed_model",
                monitors=[],
            )
            self.assertEqual(failed_manager.active_jobs(), [])

            cancelled_manager = TrainingJobManager(
                root=root / "cancelled",
                logs_root=root / "logs-cancelled",
                runner=FakeRunner(FakeProcess()),
            )
            cancelled_job = cancelled_manager.create_job(
                model="linear",
                preset="baseline",
                datasets=["Mnist"],
                overrides={},
                log_folder="cancelled_model",
                monitors=[],
            )
            cancelled_manager.cancel_job(cancelled_job["id"])
            self.assertEqual(cancelled_manager.active_jobs(), [])

    def test_training_job_accepts_multiple_presets_and_multiplies_run_count(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manager = TrainingJobManager(
                root=Path(tmp) / "jobs",
                logs_root=Path(tmp) / "logs",
                runner=FakeRunner(),
            )

            payload = manager.create_job(
                model="linear",
                preset="baseline",
                presets=["baseline", "gating", "baseline"],
                datasets=["Mnist", "Cifar10"],
                overrides={},
                log_folder="multi_preset",
            )
            payload_path = Path(tmp) / "jobs" / payload["id"] / "payload.json"
            worker_payload = json.loads(payload_path.read_text())

        self.assertEqual(payload["preset"], "baseline")
        self.assertEqual(payload["presets"], ["baseline", "gating"])
        self.assertEqual(payload["plannedRunCount"], 4)
        self.assertEqual(worker_payload["presets"], ["baseline", "gating"])

    def test_training_job_rejects_unknown_selected_preset(self) -> None:
        manager = TrainingJobManager(runner=FakeRunner())

        with self.assertRaises(InspectorError):
            manager.create_job(
                model="linear",
                preset="baseline",
                presets=["baseline", "missing-preset"],
                datasets=["Mnist"],
                overrides={},
                log_folder="test_model",
            )

    def test_training_job_accepts_grid_search_and_strips_conflicting_override(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manager = TrainingJobManager(
                root=Path(tmp) / "jobs",
                logs_root=Path(tmp) / "logs",
                runner=FakeRunner(),
            )

            payload = manager.create_job(
                model="linear",
                preset="baseline",
                presets=["baseline", "gating"],
                datasets=["Mnist"],
                overrides={"hidden_dim": "999", "stack_num_layers": "4"},
                search={
                    "mode": "grid",
                    "values": {"hidden_dim": [64, 128]},
                },
                log_folder="grid_search",
            )
            payload_path = Path(tmp) / "jobs" / payload["id"] / "payload.json"
            worker_payload = json.loads(payload_path.read_text())

        self.assertEqual(payload["overrides"], {"stack_num_layers": "4"})
        self.assertEqual(payload["search"]["mode"], "grid")
        self.assertEqual(payload["search"]["values"], {"hidden_dim": [64, 128]})
        self.assertEqual(payload["plannedRunCount"], 4)
        self.assertEqual(worker_payload["overrides"], {"stack_num_layers": "4"})
        self.assertEqual(worker_payload["search"], payload["search"])

    def test_training_run_plan_materializes_grid_rows_and_commands(self) -> None:
        manager = TrainingJobManager(runner=FakeRunner())

        plan = manager.create_run_plan(
            model="linear",
            preset="baseline",
            presets=["baseline", "gating"],
            datasets=["Mnist"],
            overrides={"hidden_dim": "999", "stack_num_layers": "4"},
            search={
                "mode": "grid",
                "values": {"hidden_dim": [64, 128]},
            },
            log_folder="",
        )

        self.assertEqual(plan["summary"]["totalRuns"], 4)
        self.assertEqual(plan["summary"]["remainingEpochs"], 120)
        self.assertFalse(plan["isRandomSearch"])
        self.assertEqual(plan["runs"][0]["preset"], "baseline")
        self.assertEqual(plan["runs"][0]["dataset"], "Mnist")
        self.assertEqual(plan["runs"][0]["status"], "Pending")
        self.assertEqual(plan["runs"][0]["overrides"]["hidden_dim"], 64)
        self.assertEqual(plan["runs"][0]["overrides"]["stack_num_layers"], "4")
        self.assertEqual(
            [change["source"] for change in plan["runs"][0]["changes"]],
            ["override", "search"],
        )
        self.assertIn("--datasets Mnist", plan["runs"][0]["command"])
        self.assertIn("--hidden-dim 64", plan["runs"][0]["command"])
        self.assertIn("--stack-num-layers 4", plan["runs"][0]["command"])
        self.assertNotIn("--logdir", plan["runs"][0]["command"])

    def test_training_job_accepts_random_search_sample_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manager = TrainingJobManager(
                root=Path(tmp) / "jobs",
                logs_root=Path(tmp) / "logs",
                runner=FakeRunner(),
            )

            payload = manager.create_job(
                model="linear",
                preset="baseline",
                datasets=["Mnist", "Cifar10"],
                overrides={},
                search={
                    "mode": "random",
                    "values": {
                        "hidden_dim": [64, 128],
                        "stack_activation": ["RELU", "GELU"],
                    },
                    "randomSamples": 3,
                },
                log_folder="random_search",
            )

        self.assertEqual(payload["search"]["mode"], "random")
        self.assertEqual(payload["search"]["randomSamples"], 3)
        self.assertEqual(payload["plannedRunCount"], 6)

    def test_training_run_plan_materializes_random_search_before_start(self) -> None:
        manager = TrainingJobManager(runner=FakeRunner())

        plan = manager.create_run_plan(
            model="linear",
            preset="baseline",
            datasets=["Mnist", "Cifar10"],
            overrides={},
            search={
                "mode": "random",
                "values": {
                    "hidden_dim": [64, 128],
                    "stack_activation": ["RELU", "GELU"],
                },
                "randomSamples": 3,
            },
            log_folder="random_search",
        )

        self.assertTrue(plan["isRandomSearch"])
        self.assertEqual(plan["summary"]["totalRuns"], 6)
        self.assertEqual(plan["summary"]["remainingEpochs"], 180)
        self.assertTrue(
            all(
                any(change["source"] == "search" for change in run["changes"])
                for run in plan["runs"]
            )
        )
        self.assertTrue(
            all("--logdir random_search" in run["command"] for run in plan["runs"])
        )

    def test_training_job_accepts_submitted_run_plan(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manager = TrainingJobManager(
                root=Path(tmp) / "jobs",
                logs_root=Path(tmp) / "logs",
                runner=FakeRunner(),
            )
            plan = manager.create_run_plan(
                model="linear",
                preset="baseline",
                datasets=["Mnist"],
                overrides={"hidden_dim": "128"},
                log_folder="",
            )
            plan["runs"][0]["snapshotId"] = "snapshot-1"
            plan["runs"][0]["snapshotName"] = "wide hidden"

            payload = manager.create_job(
                model="linear",
                preset="baseline",
                datasets=["Mnist"],
                overrides={"hidden_dim": "128"},
                log_folder="submitted_plan",
                run_plan=plan,
            )
            payload_path = Path(tmp) / "jobs" / payload["id"] / "payload.json"
            worker_payload = json.loads(payload_path.read_text())

        self.assertEqual(payload["runPlan"]["summary"]["totalRuns"], 1)
        self.assertIn(
            "--logdir submitted_plan", payload["runPlan"]["runs"][0]["command"]
        )
        self.assertEqual(payload["runPlan"]["runs"][0]["snapshotId"], "snapshot-1")
        self.assertEqual(payload["runPlan"]["runs"][0]["snapshotName"], "wide hidden")
        self.assertNotIn("snapshot-1", payload["runPlan"]["runs"][0]["command"])
        self.assertNotIn("wide hidden", payload["runPlan"]["runs"][0]["command"])
        self.assertEqual(
            worker_payload["runPlan"]["runs"][0]["id"],
            plan["runs"][0]["id"],
        )
        self.assertEqual(
            worker_payload["runPlan"]["runs"][0]["snapshotName"],
            "wide hidden",
        )

    def test_training_run_plan_preserves_error_traceback_from_progress_events(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manager = TrainingJobManager(
                root=Path(tmp) / "jobs",
                logs_root=Path(tmp) / "logs",
                runner=FakeRunner(),
            )
            payload = manager.create_job(
                model="linear",
                preset="baseline",
                datasets=["Mnist"],
                overrides={},
                log_folder="traceback_test",
            )
            job = manager.jobs[payload["id"]]
            manager._write_event(
                job,
                {
                    "type": "error",
                    "status": "failed",
                    "dataset": "Mnist",
                    "preset": "baseline",
                    "runId": payload["runPlan"]["runs"][0]["id"],
                    "error": "scalar conversion failed",
                    "traceback": "Traceback (most recent call last):\nRuntimeError: scalar conversion failed",
                },
            )

            failed_payload = manager.get_job(payload["id"])

        failed_run = failed_payload["runPlan"]["runs"][0]
        self.assertEqual(failed_run["status"], "Failed")
        self.assertEqual(failed_run["error"], "scalar conversion failed")
        self.assertIn("RuntimeError", failed_run["errorTraceback"])

    def test_training_job_rejects_invalid_search_requests(self) -> None:
        manager = TrainingJobManager(runner=FakeRunner())

        invalid_searches = [
            {"mode": "grid", "values": {"missing_axis": [1]}},
            {"mode": "grid", "values": {"hidden_dim": []}},
            {"mode": "random", "values": {"hidden_dim": [64]}, "randomSamples": 0},
            {"mode": "grid", "values": {"hidden_dim": [999]}},
            {"mode": "grid", "values": {}},
        ]

        for search in invalid_searches:
            with self.subTest(search=search):
                with self.assertRaises(InspectorError):
                    manager.create_job(
                        model="linear",
                        preset="baseline",
                        datasets=["Mnist"],
                        overrides={},
                        search=search,
                        log_folder="invalid_search",
                    )

    def test_training_job_rejects_locked_search_axis(self) -> None:
        manager = TrainingJobManager(runner=FakeRunner())

        with self.assertRaises(InspectorError) as context:
            manager.create_job(
                model="linear",
                preset="baseline",
                presets=["baseline", "post-norm"],
                datasets=["Mnist"],
                overrides={},
                search={
                    "mode": "grid",
                    "values": {"layer_norm_position": ["BEFORE", "AFTER"]},
                },
                log_folder="locked_search",
            )

        self.assertIn("locked", str(context.exception))

    def test_worker_search_mode_conversion_uses_experiment_search_types(self) -> None:
        grid_search = parse_training_search(
            "linear",
            "baseline",
            {"mode": "grid", "values": {"hidden_dim": [64]}},
            dataset_count=1,
        )
        random_search = parse_training_search(
            "linear",
            "baseline",
            {
                "mode": "random",
                "values": {"hidden_dim": [64, 128]},
                "randomSamples": 2,
            },
            dataset_count=1,
        )

        self.assertIsInstance(search_mode_from_parsed_search(grid_search), GridSearch)
        converted_random_search = search_mode_from_parsed_search(random_search)
        self.assertIsInstance(converted_random_search, RandomSearch)
        self.assertEqual(converted_random_search.num_samples, 2)

    def test_training_job_rejects_invalid_log_folders(self) -> None:
        manager = TrainingJobManager(runner=FakeRunner())

        for log_folder in (
            "",
            "my experiment",
            "my-experiment",
            "my.folder",
            "my/folder",
            "_my_folder",
            "my_folder_",
            "my__folder",
        ):
            with self.subTest(log_folder=log_folder):
                with self.assertRaises(InspectorError):
                    manager.create_job(
                        model="linear",
                        preset="baseline",
                        datasets=["Mnist"],
                        overrides={},
                        log_folder=log_folder,
                    )

    def test_training_job_rejects_unknown_monitor(self) -> None:
        manager = TrainingJobManager(runner=FakeRunner())

        with self.assertRaises(InspectorError):
            manager.create_job(
                model="linear",
                preset="baseline",
                datasets=["Mnist"],
                overrides={},
                log_folder="test_model",
                monitors=["sampler"],
            )

    def test_training_job_rejects_locked_overrides(self) -> None:
        manager = TrainingJobManager(runner=FakeRunner())

        with self.assertRaises(InspectorError):
            manager.create_job(
                model="linear",
                preset="baseline",
                presets=["baseline", "gating"],
                datasets=["Mnist"],
                overrides={"gate_flag": "false"},
                log_folder="test_model",
            )

    def test_training_job_monitor_data_filters_tensorboard_tags(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "jobs"
            log_dir = Path(tmp) / "logs" / "run"
            writer = SummaryWriter(log_dir=str(log_dir))
            writer.add_scalar("main_model.0.model/output/mean", 0.12, 100)
            writer.add_scalar("main_model.1.model/output/mean", 0.99, 100)
            writer.add_histogram(
                "main_model.0.model/histogram/usage_fraction",
                torch.tensor([0.05, 0.15, 0.2]),
                100,
            )
            writer.add_image(
                "main_model.0.model/heatmap/usage_fraction",
                torch.ones(1, 2, 2),
                100,
                dataformats="CHW",
            )
            writer.flush()
            writer.close()

            manager = TrainingJobManager(
                root=root,
                logs_root=Path(tmp) / "logs",
                runner=FakeRunner(),
            )
            payload = manager.create_job(
                model="linear",
                preset="baseline",
                datasets=["Mnist"],
                overrides={},
                log_folder="test_model",
                monitors=["linear"],
            )
            job = manager.jobs[payload["id"]]
            manager._write_event(
                job,
                {
                    "type": "dataset_started",
                    "status": "running",
                    "dataset": "Mnist",
                    "logDir": str(log_dir),
                },
            )

            data = manager.get_monitor_data(
                payload["id"],
                node_path="main_model.0.model",
                dataset="Mnist",
            )
            unmatched = manager.get_monitor_data(
                payload["id"],
                node_path="missing",
                dataset="Mnist",
            )

        self.assertEqual(data["jobId"], payload["id"])
        self.assertEqual(data["nodePath"], "main_model.0.model")
        self.assertEqual(data["dataset"], "Mnist")
        self.assertEqual(
            data["scalarSeries"][0]["tag"], "main_model.0.model/output/mean"
        )
        self.assertEqual(data["scalarSeries"][0]["label"], "output/mean")
        self.assertEqual(data["scalarSeries"][0]["points"][0]["step"], 100)
        self.assertEqual(len(data["histograms"]), 1)
        self.assertEqual(len(data["images"]), 1)
        self.assertTrue(
            data["images"][0]["dataUrl"].startswith("data:image/png;base64,")
        )
        self.assertEqual(unmatched["scalarSeries"], [])
        self.assertEqual(unmatched["histograms"], [])
        self.assertEqual(unmatched["images"], [])

    def test_training_job_monitor_data_filters_by_preset_and_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "jobs"
            baseline_dir = Path(tmp) / "logs" / "baseline"
            gating_dir = Path(tmp) / "logs" / "gating"
            baseline_writer = SummaryWriter(log_dir=str(baseline_dir))
            baseline_writer.add_scalar("main_model.0.model/output/mean", 0.12, 100)
            baseline_writer.flush()
            baseline_writer.close()
            gating_writer = SummaryWriter(log_dir=str(gating_dir))
            gating_writer.add_scalar("main_model.0.model/output/mean", 0.88, 100)
            gating_writer.flush()
            gating_writer.close()

            manager = TrainingJobManager(
                root=root,
                logs_root=Path(tmp) / "logs",
                runner=FakeRunner(),
            )
            payload = manager.create_job(
                model="linear",
                preset="baseline",
                presets=["baseline", "gating"],
                datasets=["Mnist"],
                overrides={},
                log_folder="test_model",
                monitors=["linear"],
            )
            job = manager.jobs[payload["id"]]
            manager._write_event(
                job,
                {
                    "type": "dataset_started",
                    "status": "running",
                    "preset": "baseline",
                    "dataset": "Mnist",
                    "logDir": str(baseline_dir),
                },
            )
            manager._write_event(
                job,
                {
                    "type": "dataset_started",
                    "status": "running",
                    "preset": "gating",
                    "dataset": "Mnist",
                    "logDir": str(gating_dir),
                },
            )

            data = manager.get_monitor_data(
                payload["id"],
                node_path="main_model.0.model",
                preset="gating",
                dataset="Mnist",
            )

        self.assertEqual(data["preset"], "gating")
        self.assertEqual(data["dataset"], "Mnist")
        self.assertAlmostEqual(data["scalarSeries"][0]["points"][0]["value"], 0.88)

    def test_log_run_monitor_data_filters_tensorboard_tags(self) -> None:
        import httpx
        from viewer.backend.api import ViewerApiSettings, create_app

        with tempfile.TemporaryDirectory() as tmp:
            logs_root = Path(tmp) / "logs"
            log_dir = logs_root.joinpath(
                "test_model",
                "linear",
                "BASELINE",
                "Mnist",
                "historical_20260601_010203",
                "version_0",
            )
            writer = SummaryWriter(log_dir=str(log_dir))
            writer.add_scalar("main_model.0.model/output/mean", 0.12, 100)
            writer.add_scalar("main_model.1.model/output/mean", 0.99, 100)
            writer.add_histogram(
                "main_model.0.model/histogram/usage_fraction",
                torch.tensor([0.05, 0.15, 0.2]),
                100,
            )
            writer.add_image(
                "main_model.0.model/heatmap/usage_fraction",
                torch.ones(1, 2, 2),
                100,
                dataformats="CHW",
            )
            writer.flush()
            writer.close()

            run_id = LogRunIndex(logs_root=logs_root).list_runs()[0].id

            async def call_api() -> (
                tuple[httpx.Response, httpx.Response, httpx.Response]
            ):
                transport = httpx.ASGITransport(
                    app=create_app(ViewerApiSettings(logs_root=str(logs_root)))
                )
                async with httpx.AsyncClient(
                    transport=transport,
                    base_url="http://testserver",
                ) as client:
                    data_response = await client.get(
                        f"/logs/runs/{run_id}/monitor-data",
                        params={"nodePath": "main_model.0.model"},
                    )
                    unmatched_response = await client.get(
                        f"/logs/runs/{run_id}/monitor-data",
                        params={"nodePath": "missing"},
                    )
                    unknown_response = await client.get(
                        "/logs/runs/not-a-run/monitor-data",
                        params={"nodePath": "main_model.0.model"},
                    )
                    return data_response, unmatched_response, unknown_response

            data_response, unmatched_response, unknown_response = asyncio.run(
                call_api()
            )

        self.assertEqual(data_response.status_code, 200)
        data = data_response.json()
        self.assertEqual(data["jobId"], run_id)
        self.assertEqual(data["nodePath"], "main_model.0.model")
        self.assertEqual(data["dataset"], "Mnist")
        self.assertEqual(
            data["scalarSeries"][0]["tag"], "main_model.0.model/output/mean"
        )
        self.assertEqual(data["scalarSeries"][0]["label"], "output/mean")
        self.assertEqual(data["scalarSeries"][0]["points"][0]["step"], 100)
        self.assertEqual(len(data["histograms"]), 1)
        self.assertEqual(len(data["images"]), 1)
        self.assertTrue(
            data["images"][0]["dataUrl"].startswith("data:image/png;base64,")
        )

        self.assertEqual(unmatched_response.status_code, 200)
        unmatched = unmatched_response.json()
        self.assertEqual(unmatched["jobId"], run_id)
        self.assertEqual(unmatched["dataset"], "Mnist")
        self.assertEqual(unmatched["scalarSeries"], [])
        self.assertEqual(unmatched["histograms"], [])
        self.assertEqual(unmatched["images"], [])

        self.assertEqual(unknown_response.status_code, 400)
        self.assertIn("Unknown log run id", unknown_response.json()["detail"])

    def test_log_run_index_parses_supported_log_shapes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logs_root = Path(tmp) / "logs"
            default_run = write_tensorboard_run(
                logs_root,
                ["linear", "BASELINE", "Mnist", "aaa_20260601_010203", "version_0"],
                metrics={"test/accuracy": 0.9},
            )
            custom_run = write_tensorboard_run(
                logs_root,
                [
                    "test_model",
                    "linear_adaptive",
                    "DUAL_MODEL_WEIGHT",
                    "Cifar10",
                    "bbb_20260601_020304",
                    "version_1",
                ],
                metrics=None,
                hparams=False,
                checkpoint=False,
            )
            viewer_run = write_tensorboard_run(
                logs_root,
                [
                    "viewer-training",
                    "job-123",
                    "linear",
                    "BASELINE",
                    "FashionMNIST",
                    "ccc_20260601_030405",
                    "version_0",
                ],
                metrics={"validation/accuracy": 0.8},
            )
            no_event_run = logs_root.joinpath(
                "linear",
                "BASELINE",
                "Mnist",
                "ddd_20260601_040506",
                "version_2",
            )
            no_event_run.mkdir(parents=True)
            malformed_result_run = write_tensorboard_run(
                logs_root,
                [
                    "linear",
                    "BASELINE",
                    "Mnist",
                    "eee_20260601_050607",
                    "version_0",
                ],
                metrics=None,
            )
            (malformed_result_run / "result.json").write_text(
                "{not valid json",
                encoding="utf-8",
            )
            outside_run = write_tensorboard_run(
                Path(tmp) / "outside",
                [
                    "linear",
                    "BASELINE",
                    "Mnist",
                    "fff_20260601_060708",
                    "version_0",
                ],
                metrics={"test/accuracy": 1.0},
            )
            escaped_run_parent = logs_root.joinpath(
                "linear",
                "BASELINE",
                "Mnist",
                "escaped_20260601_070809",
            )
            escaped_run_parent.mkdir(parents=True)
            escaped_run = escaped_run_parent / "version_99"
            escaped_run.symlink_to(outside_run, target_is_directory=True)

            runs = LogRunIndex(logs_root=logs_root).list_runs()

        by_path = {run.relativePath: run for run in runs}
        default_summary = by_path[default_run.relative_to(logs_root).as_posix()]
        custom_summary = by_path[custom_run.relative_to(logs_root).as_posix()]
        viewer_summary = by_path[viewer_run.relative_to(logs_root).as_posix()]
        no_event_summary = by_path[no_event_run.relative_to(logs_root).as_posix()]
        malformed_result_summary = by_path[
            malformed_result_run.relative_to(logs_root).as_posix()
        ]

        self.assertIsNone(default_summary.group)
        self.assertEqual(default_summary.experiment, "linear")
        self.assertEqual(default_summary.model, "linear")
        self.assertEqual(default_summary.preset, "BASELINE")
        self.assertEqual(default_summary.dataset, "Mnist")
        self.assertEqual(default_summary.timestamp, "2026-06-01 01:02:03")
        self.assertTrue(default_summary.hasResult)
        self.assertGreater(default_summary.eventFileCount, 0)
        self.assertEqual(default_summary.checkpointCount, 1)
        self.assertTrue(default_summary.hasHparams)
        self.assertEqual(default_summary.metrics["test/accuracy"], 0.9)

        self.assertEqual(custom_summary.group, "test_model")
        self.assertEqual(custom_summary.experiment, "test_model")
        self.assertEqual(custom_summary.model, "linear_adaptive")
        self.assertFalse(custom_summary.hasResult)
        self.assertFalse(custom_summary.hasHparams)
        self.assertEqual(custom_summary.checkpointCount, 0)

        self.assertEqual(viewer_summary.group, "viewer-training/job-123")
        self.assertEqual(viewer_summary.experiment, "viewer-training")
        self.assertEqual(viewer_summary.model, "linear")
        self.assertEqual(viewer_summary.dataset, "FashionMNIST")

        self.assertFalse(no_event_summary.hasResult)
        self.assertEqual(no_event_summary.eventFileCount, 0)
        self.assertEqual(no_event_summary.checkpointCount, 0)
        self.assertFalse(no_event_summary.hasHparams)
        self.assertEqual(no_event_summary.metrics, {})

        self.assertTrue(malformed_result_summary.hasResult)
        self.assertGreater(malformed_result_summary.eventFileCount, 0)
        self.assertEqual(malformed_result_summary.metrics, {})
        self.assertNotIn(
            "linear/BASELINE/Mnist/escaped_20260601_070809/version_99",
            by_path,
        )

    def test_log_run_index_deletes_experiment_tree(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            logs_root = root / "logs"
            deleted_run = write_tensorboard_run(
                logs_root,
                [
                    "test_model",
                    "linear",
                    "BASELINE",
                    "Mnist",
                    "aaa_20260601_010203",
                    "version_0",
                ],
            )
            second_deleted_run = write_tensorboard_run(
                logs_root,
                [
                    "test_model",
                    "linear",
                    "BASELINE",
                    "Cifar10",
                    "bbb_20260601_020304",
                    "version_0",
                ],
            )
            remaining_run = write_tensorboard_run(
                logs_root,
                [
                    "test_model_2",
                    "linear",
                    "BASELINE",
                    "Mnist",
                    "ccc_20260601_030405",
                    "version_0",
                ],
            )
            outside_target = root / "outside.txt"
            outside_target.write_text("outside", encoding="utf-8")
            logs_root.joinpath("test_model", "outside-link").symlink_to(outside_target)

            index = LogRunIndex(logs_root=logs_root)
            run_ids_by_path = {run.relativePath: run.id for run in index.list_runs()}
            result = index.delete_experiment("test_model")
            remaining_paths = {run.relativePath for run in index.list_runs()}

            self.assertEqual(result.experiment, "test_model")
            self.assertEqual(result.deletedRunCount, 2)
            self.assertEqual(result.deletedRelativePath, "test_model")
            self.assertEqual(
                set(result.deletedRunIds),
                {
                    run_ids_by_path[deleted_run.relative_to(logs_root).as_posix()],
                    run_ids_by_path[
                        second_deleted_run.relative_to(logs_root).as_posix()
                    ],
                },
            )
            self.assertFalse(logs_root.joinpath("test_model").exists())
            self.assertTrue(remaining_run.exists())
            self.assertTrue(outside_target.exists())
            self.assertEqual(
                remaining_paths,
                {remaining_run.relative_to(logs_root).as_posix()},
            )

    def test_log_run_index_deletes_filtered_version_dirs_and_prunes_empty_parents(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logs_root = Path(tmp) / "logs"
            mnist_run = write_tensorboard_run(
                logs_root,
                [
                    "test_model",
                    "linear",
                    "BASELINE",
                    "Mnist",
                    "aaa_20260601_010203",
                    "version_0",
                ],
            )
            cifar_run = write_tensorboard_run(
                logs_root,
                [
                    "test_model",
                    "linear",
                    "BASELINE",
                    "Cifar10",
                    "bbb_20260601_020304",
                    "version_0",
                ],
            )
            gating_run = write_tensorboard_run(
                logs_root,
                [
                    "test_model",
                    "linear",
                    "GATING",
                    "Mnist",
                    "ccc_20260601_030405",
                    "version_0",
                ],
            )
            other_experiment_run = write_tensorboard_run(
                logs_root,
                [
                    "other_model",
                    "linear",
                    "BASELINE",
                    "Mnist",
                    "ddd_20260601_040506",
                    "version_0",
                ],
            )

            index = LogRunIndex(logs_root=logs_root)
            runs = index.list_runs()
            filters = delete_filters_for_runs(
                runs,
                experiments=["test_model"],
                datasets=["Mnist"],
                presets=["BASELINE"],
            )
            plan = index.create_delete_plan(filters, active_jobs=[])
            result = index.delete_runs(filters, active_jobs=[])
            remaining_paths = {run.relativePath for run in index.list_runs()}

            self.assertTrue(plan.canDelete)
            self.assertEqual(plan.to_response()["candidateCount"], 1)
            self.assertEqual(len(result.deletedRunIds), 1)
            self.assertEqual(
                result.deletedRelativePaths,
                [mnist_run.relative_to(logs_root).as_posix()],
            )
            self.assertFalse(mnist_run.exists())
            self.assertFalse(mnist_run.parent.exists())
            self.assertTrue(cifar_run.exists())
            self.assertTrue(gating_run.exists())
            self.assertTrue(other_experiment_run.exists())
            self.assertEqual(
                remaining_paths,
                {
                    cifar_run.relative_to(logs_root).as_posix(),
                    gating_run.relative_to(logs_root).as_posix(),
                    other_experiment_run.relative_to(logs_root).as_posix(),
                },
            )

            second_filters = delete_filters_for_runs(
                index.list_runs(),
                experiments=["test_model"],
                datasets=["Cifar10"],
                presets=["BASELINE"],
            )
            index.delete_runs(second_filters, active_jobs=[])
            self.assertFalse(cifar_run.exists())
            self.assertFalse(
                logs_root.joinpath(
                    "test_model",
                    "linear",
                    "BASELINE",
                    "Cifar10",
                ).exists()
            )
            self.assertTrue(logs_root.joinpath("test_model").exists())

    def test_log_run_index_deletes_exact_run_id_filter_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logs_root = Path(tmp) / "logs"
            first_run = write_tensorboard_run(
                logs_root,
                [
                    "test_model",
                    "linear",
                    "BASELINE",
                    "Mnist",
                    "aaa_20260601_010203",
                    "version_0",
                ],
            )
            second_run = write_tensorboard_run(
                logs_root,
                [
                    "test_model",
                    "linear",
                    "BASELINE",
                    "Mnist",
                    "bbb_20260601_020304",
                    "version_0",
                ],
            )

            index = LogRunIndex(logs_root=logs_root)
            runs = index.list_runs()
            first_run_id = next(
                run.id
                for run in runs
                if run.relativePath == first_run.relative_to(logs_root).as_posix()
            )
            filters = delete_filters_for_runs(runs, run_ids=[first_run_id])
            result = index.delete_runs(filters, active_jobs=[])

            self.assertEqual(result.deletedRunIds, [first_run_id])
            self.assertFalse(first_run.exists())
            self.assertTrue(second_run.exists())

    def test_log_run_delete_empty_filters_match_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logs_root = Path(tmp) / "logs"
            write_tensorboard_run(
                logs_root,
                [
                    "test_model",
                    "linear",
                    "BASELINE",
                    "Mnist",
                    "aaa_20260601_010203",
                    "version_0",
                ],
            )

            index = LogRunIndex(logs_root=logs_root)
            filters = LogRunDeleteFilters(
                experiments=["test_model"],
                datasets=[],
                models=["linear"],
                presets=["BASELINE"],
                runIds=[index.list_runs()[0].id],
            )
            plan = index.create_delete_plan(filters, active_jobs=[])

            self.assertFalse(plan.canDelete)
            self.assertEqual(plan.candidates, [])
            with self.assertRaisesRegex(InspectorError, "No log runs match"):
                index.delete_runs(filters, active_jobs=[])

    def test_log_run_index_lists_safe_top_level_experiments(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            logs_root = root / "logs"
            logs_root.mkdir()
            logs_root.joinpath("empty_experiment").mkdir()
            logs_root.joinpath("bad-name").mkdir()
            logs_root.joinpath("_bad_name").mkdir()
            outside_experiment = root / "outside_experiment"
            outside_experiment.mkdir()
            logs_root.joinpath("linked_experiment").symlink_to(
                outside_experiment,
                target_is_directory=True,
            )
            write_tensorboard_run(
                logs_root,
                [
                    "test_model",
                    "linear",
                    "BASELINE",
                    "Mnist",
                    "aaa_20260601_010203",
                    "version_0",
                ],
            )
            write_tensorboard_run(
                logs_root,
                [
                    "viewer-training",
                    "job-123",
                    "linear",
                    "BASELINE",
                    "Mnist",
                    "bbb_20260601_020304",
                    "version_0",
                ],
            )

            experiments = LogRunIndex(logs_root=logs_root).list_experiments()

        self.assertEqual(
            [experiment.experiment for experiment in experiments],
            ["empty_experiment", "test_model"],
        )
        by_name = {experiment.experiment: experiment for experiment in experiments}
        self.assertEqual(by_name["empty_experiment"].runCount, 0)
        self.assertEqual(by_name["test_model"].runCount, 1)
        self.assertEqual(by_name["test_model"].relativePath, "test_model")

    def test_log_run_index_rejects_invalid_delete_experiments(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            logs_root = root / "logs"
            write_tensorboard_run(
                logs_root,
                ["linear", "BASELINE", "Mnist", "aaa_20260601_010203", "version_0"],
            )
            outside_experiment = root / "outside_experiment"
            write_tensorboard_run(
                outside_experiment,
                ["linear", "BASELINE", "Mnist", "bbb_20260601_020304", "version_0"],
            )
            logs_root.joinpath("linked").symlink_to(
                outside_experiment,
                target_is_directory=True,
            )

            index = LogRunIndex(logs_root=logs_root)
            for experiment in (
                "",
                "../outside",
                "linear/BASELINE",
                ".",
                "..",
                "missing",
            ):
                with self.subTest(experiment=experiment):
                    with self.assertRaises(InspectorError):
                        index.delete_experiment(experiment)

            with self.assertRaisesRegex(InspectorError, "symlink"):
                index.delete_experiment("linked")

            self.assertTrue(logs_root.joinpath("linear").exists())
            self.assertTrue(outside_experiment.exists())

    def test_log_api_deletes_experiment_and_refreshes_runs(self) -> None:
        import httpx
        from viewer.backend.api import ViewerApiSettings, create_app

        with tempfile.TemporaryDirectory() as tmp:
            logs_root = Path(tmp) / "logs"
            write_tensorboard_run(
                logs_root,
                [
                    "test_model",
                    "linear",
                    "BASELINE",
                    "Mnist",
                    "aaa_20260601_010203",
                    "version_0",
                ],
            )
            write_tensorboard_run(
                logs_root,
                [
                    "test_model_2",
                    "linear",
                    "BASELINE",
                    "Cifar10",
                    "bbb_20260601_020304",
                    "version_0",
                ],
            )

            async def call_api() -> (
                tuple[httpx.Response, httpx.Response, httpx.Response]
            ):
                transport = httpx.ASGITransport(
                    app=create_app(ViewerApiSettings(logs_root=str(logs_root)))
                )
                async with httpx.AsyncClient(
                    transport=transport,
                    base_url="http://testserver",
                ) as client:
                    before_response = await client.get("/logs/runs")
                    delete_response = await client.delete(
                        "/logs/experiments/test_model",
                    )
                    after_response = await client.get("/logs/runs")
                    return before_response, delete_response, after_response

            before_response, delete_response, after_response = asyncio.run(call_api())

            self.assertEqual(before_response.status_code, 200)
            self.assertEqual(delete_response.status_code, 200)
            self.assertFalse(logs_root.joinpath("test_model").exists())
            self.assertTrue(logs_root.joinpath("test_model_2").exists())

        delete_payload = delete_response.json()
        self.assertEqual(delete_payload["experiment"], "test_model")
        self.assertEqual(delete_payload["deletedRunCount"], 1)
        self.assertEqual(delete_payload["deletedRelativePath"], "test_model")
        self.assertEqual(len(delete_payload["deletedRunIds"]), 1)
        self.assertEqual(
            [run["experiment"] for run in after_response.json()["runs"]],
            ["test_model_2"],
        )

    def test_log_api_deletes_valid_empty_experiment_folder(self) -> None:
        import httpx
        from viewer.backend.api import ViewerApiSettings, create_app

        with tempfile.TemporaryDirectory() as tmp:
            logs_root = Path(tmp) / "logs"
            logs_root.mkdir()
            logs_root.joinpath("new_empty").mkdir()

            async def call_api() -> httpx.Response:
                transport = httpx.ASGITransport(
                    app=create_app(ViewerApiSettings(logs_root=str(logs_root)))
                )
                async with httpx.AsyncClient(
                    transport=transport,
                    base_url="http://testserver",
                ) as client:
                    return await client.delete("/logs/experiments/new_empty")

            response = asyncio.run(call_api())

            self.assertEqual(response.status_code, 200)
            self.assertEqual(
                response.json(),
                {
                    "experiment": "new_empty",
                    "deletedRunIds": [],
                    "deletedRunCount": 0,
                    "deletedRelativePath": "new_empty",
                },
            )
            self.assertFalse(logs_root.joinpath("new_empty").exists())

    def test_log_api_plans_and_deletes_filtered_runs_with_active_job_guard(
        self,
    ) -> None:
        import httpx
        from viewer.backend.api import ViewerApiSettings, create_app

        with tempfile.TemporaryDirectory() as tmp:
            logs_root = Path(tmp) / "logs"
            run_dir = write_tensorboard_run(
                logs_root,
                [
                    "test_model",
                    "linear",
                    "BASELINE",
                    "Mnist",
                    "aaa_20260601_010203",
                    "version_0",
                ],
            )
            manager = TrainingJobManager(
                root=Path(tmp) / "jobs",
                logs_root=logs_root,
                runner=FakeRunner(),
            )
            manager.create_job(
                model="linear",
                preset="baseline",
                datasets=["Mnist"],
                overrides={},
                log_folder="test_model",
                monitors=[],
            )

            async def call_api() -> tuple[httpx.Response, httpx.Response]:
                transport = httpx.ASGITransport(
                    app=create_app(
                        ViewerApiSettings(logs_root=str(logs_root)),
                        training_manager=manager,
                    )
                )
                async with httpx.AsyncClient(
                    transport=transport,
                    base_url="http://testserver",
                ) as client:
                    runs_response = await client.get("/logs/runs")
                    run = runs_response.json()["runs"][0]
                    filters = {
                        "experiments": [run["experiment"]],
                        "datasets": [run["dataset"]],
                        "models": [run["model"]],
                        "presets": [run["preset"]],
                        "runIds": [run["id"]],
                    }
                    plan_response = await client.post(
                        "/logs/runs/delete-plan",
                        json=filters,
                    )
                    delete_response = await client.post(
                        "/logs/runs/delete",
                        json=filters,
                    )
                    return plan_response, delete_response

            plan_response, delete_response = asyncio.run(call_api())

            self.assertEqual(plan_response.status_code, 200)
            plan_payload = plan_response.json()
            self.assertEqual(plan_payload["candidateCount"], 1)
            self.assertFalse(plan_payload["canDelete"])
            self.assertEqual(
                plan_payload["blockedByActiveJobs"][0]["logFolder"],
                "test_model",
            )
            self.assertEqual(delete_response.status_code, 400)
            self.assertIn(
                "A training job is still writing to this log folder.",
                delete_response.text,
            )
            self.assertTrue(run_dir.exists())

    def test_log_api_lists_experiments_including_empty_folders(self) -> None:
        import httpx
        from viewer.backend.api import ViewerApiSettings, create_app

        with tempfile.TemporaryDirectory() as tmp:
            logs_root = Path(tmp) / "logs"
            logs_root.mkdir()
            logs_root.joinpath("new_empty").mkdir()
            logs_root.joinpath("bad-name").mkdir()
            write_tensorboard_run(
                logs_root,
                [
                    "test_model",
                    "linear",
                    "BASELINE",
                    "Mnist",
                    "aaa_20260601_010203",
                    "version_0",
                ],
            )

            async def call_api() -> httpx.Response:
                transport = httpx.ASGITransport(
                    app=create_app(ViewerApiSettings(logs_root=str(logs_root)))
                )
                async with httpx.AsyncClient(
                    transport=transport,
                    base_url="http://testserver",
                ) as client:
                    return await client.get("/logs/experiments")

            response = asyncio.run(call_api())

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json()["experiments"],
            [
                {
                    "experiment": "new_empty",
                    "runCount": 0,
                    "relativePath": "new_empty",
                },
                {
                    "experiment": "test_model",
                    "runCount": 1,
                    "relativePath": "test_model",
                },
            ],
        )

    def test_log_api_paginates_unbounded_list_endpoints(self) -> None:
        import httpx
        from viewer.backend.api import ViewerApiSettings, create_app

        with tempfile.TemporaryDirectory() as tmp:
            logs_root = Path(tmp) / "logs"
            for index, experiment in enumerate(("exp_a", "exp_b", "exp_c"), start=1):
                write_tensorboard_run(
                    logs_root,
                    [
                        experiment,
                        "linear",
                        "BASELINE",
                        "Mnist",
                        f"run_{index}_2026060{index}_010203",
                        "version_0",
                    ],
                )

            async def call_api() -> tuple[httpx.Response, httpx.Response]:
                transport = httpx.ASGITransport(
                    app=create_app(ViewerApiSettings(logs_root=str(logs_root)))
                )
                async with httpx.AsyncClient(
                    transport=transport,
                    base_url="http://testserver",
                ) as client:
                    runs_response = await client.get(
                        "/logs/runs",
                        params={"limit": 2, "offset": 1},
                    )
                    experiments_response = await client.get(
                        "/logs/experiments",
                        params={"limit": 1, "offset": 1},
                    )
                    return runs_response, experiments_response

            runs_response, experiments_response = asyncio.run(call_api())

        self.assertEqual(runs_response.status_code, 200)
        runs_payload = runs_response.json()
        self.assertEqual(runs_payload["total"], 3)
        self.assertEqual(runs_payload["limit"], 2)
        self.assertEqual(runs_payload["offset"], 1)
        self.assertFalse(runs_payload["hasMore"])
        self.assertEqual(len(runs_payload["runs"]), 2)

        self.assertEqual(experiments_response.status_code, 200)
        experiments_payload = experiments_response.json()
        self.assertEqual(experiments_payload["total"], 3)
        self.assertEqual(experiments_payload["limit"], 1)
        self.assertEqual(experiments_payload["offset"], 1)
        self.assertTrue(experiments_payload["hasMore"])
        self.assertEqual(len(experiments_payload["experiments"]), 1)

    def test_log_api_reads_tags_scalars_and_rejects_unknown_run_ids(self) -> None:
        import httpx
        from viewer.backend.api import ViewerApiSettings, create_app

        with tempfile.TemporaryDirectory() as tmp:
            logs_root = Path(tmp) / "logs"
            write_tensorboard_run(
                logs_root,
                ["linear", "BASELINE", "Mnist", "aaa_20260601_010203", "version_0"],
                scalars={
                    "train/loss": [(1, 0.5), (2, 0.25)],
                    "validation/accuracy": [(2, 0.75)],
                },
                metrics={"test/accuracy": 0.9},
            )
            write_tensorboard_run(
                logs_root,
                [
                    "test_model_2",
                    "linear_adaptive",
                    "DUAL_MODEL_WEIGHT",
                    "Cifar10",
                    "bbb_20260601_020304",
                    "version_0",
                ],
                metrics=None,
                hparams=False,
                checkpoint=False,
            )
            no_event_run = logs_root.joinpath(
                "linear",
                "BASELINE",
                "Mnist",
                "no_events_20260601_040506",
                "version_0",
            )
            no_event_run.mkdir(parents=True)
            malformed_result_run = write_tensorboard_run(
                logs_root,
                [
                    "linear",
                    "BASELINE",
                    "Mnist",
                    "malformed_20260601_050607",
                    "version_0",
                ],
                metrics=None,
            )
            (malformed_result_run / "result.json").write_text(
                "{not valid json",
                encoding="utf-8",
            )

            async def call_api() -> tuple[
                httpx.Response,
                httpx.Response,
                httpx.Response,
                httpx.Response,
                httpx.Response,
            ]:
                transport = httpx.ASGITransport(
                    app=create_app(ViewerApiSettings(logs_root=str(logs_root)))
                )
                async with httpx.AsyncClient(
                    transport=transport,
                    base_url="http://testserver",
                ) as client:
                    runs_response = await client.get("/logs/runs")
                    run_id = next(
                        run["id"]
                        for run in runs_response.json()["runs"]
                        if run["relativePath"]
                        == "linear/BASELINE/Mnist/aaa_20260601_010203/version_0"
                    )
                    tags_response = await client.post(
                        "/logs/tags",
                        json={"runIds": [run_id]},
                    )
                    scalars_response = await client.post(
                        "/logs/scalars",
                        json={"runIds": [run_id], "tags": ["train/loss"]},
                    )
                    unknown_response = await client.post(
                        "/logs/tags",
                        json={"runIds": ["not-a-run"]},
                    )
                    raw_path_response = await client.post(
                        "/logs/scalars",
                        json={
                            "runIds": [
                                "linear/BASELINE/Mnist/aaa_20260601_010203/version_0"
                            ],
                            "tags": ["train/loss"],
                        },
                    )
                    return (
                        runs_response,
                        tags_response,
                        scalars_response,
                        unknown_response,
                        raw_path_response,
                    )

            (
                runs_response,
                tags_response,
                scalars_response,
                unknown_response,
                raw_path_response,
            ) = asyncio.run(call_api())

        self.assertEqual(runs_response.status_code, 200)
        runs_payload = runs_response.json()["runs"]
        by_path = {run["relativePath"]: run for run in runs_payload}
        run_payload = by_path["linear/BASELINE/Mnist/aaa_20260601_010203/version_0"]
        incomplete_payload = by_path[
            "test_model_2/linear_adaptive/DUAL_MODEL_WEIGHT/Cifar10/"
            "bbb_20260601_020304/version_0"
        ]
        no_event_payload = by_path[
            "linear/BASELINE/Mnist/no_events_20260601_040506/version_0"
        ]
        malformed_result_payload = by_path[
            "linear/BASELINE/Mnist/malformed_20260601_050607/version_0"
        ]
        self.assertEqual(run_payload["experiment"], "linear")
        self.assertEqual(run_payload["dataset"], "Mnist")
        self.assertTrue(run_payload["hasResult"])
        self.assertGreater(run_payload["eventFileCount"], 0)
        self.assertEqual(run_payload["metrics"]["test/accuracy"], 0.9)
        self.assertEqual(incomplete_payload["experiment"], "test_model_2")
        self.assertFalse(incomplete_payload["hasResult"])
        self.assertFalse(no_event_payload["hasResult"])
        self.assertEqual(no_event_payload["eventFileCount"], 0)
        self.assertEqual(no_event_payload["metrics"], {})
        self.assertTrue(malformed_result_payload["hasResult"])
        self.assertEqual(malformed_result_payload["metrics"], {})

        self.assertEqual(tags_response.status_code, 200)
        self.assertEqual(
            set(tags_response.json()["runs"][0]["scalarTags"]),
            {"train/loss", "validation/accuracy"},
        )

        self.assertEqual(scalars_response.status_code, 200)
        series = scalars_response.json()["series"][0]
        self.assertEqual(series["tag"], "train/loss")
        self.assertEqual([point["step"] for point in series["points"]], [1, 2])
        self.assertEqual(series["points"][1]["value"], 0.25)

        self.assertEqual(unknown_response.status_code, 400)
        self.assertIn("Unknown log run id", unknown_response.json()["detail"])
        self.assertEqual(raw_path_response.status_code, 400)
        self.assertIn("Unknown log run id", raw_path_response.json()["detail"])

    def test_api_default_cors_settings_allow_local_dev_frontends(self) -> None:
        import httpx
        from viewer.backend.api import create_app

        async def call_api(origin: str) -> httpx.Response:
            transport = httpx.ASGITransport(app=create_app())
            async with httpx.AsyncClient(
                transport=transport,
                base_url="http://testserver",
            ) as client:
                return await client.options(
                    "/health",
                    headers={
                        "origin": origin,
                        "access-control-request-method": "GET",
                    },
                )

        for origin in (
            "http://localhost:9000",
            "http://127.0.0.1:9000",
            "http://0.0.0.0:9000",
            "http://localhost:9001",
            "http://127.0.0.1:9001",
            "http://0.0.0.0:9001",
            "http://localhost:3000",
            "http://127.0.0.1:3000",
            "http://0.0.0.0:3000",
        ):
            with self.subTest(origin=origin):
                response = asyncio.run(call_api(origin))
                self.assertEqual(response.status_code, 200)
                self.assertEqual(
                    response.headers["access-control-allow-origin"],
                    origin,
                )
                self.assertIn(
                    "DELETE",
                    response.headers["access-control-allow-methods"],
                )

    def test_api_factory_applies_custom_cors_settings(self) -> None:
        import httpx
        from viewer.backend.api import ViewerApiSettings, create_app

        async def call_api() -> httpx.Response:
            transport = httpx.ASGITransport(
                app=create_app(ViewerApiSettings(cors_origins=["http://frontend.test"]))
            )
            async with httpx.AsyncClient(
                transport=transport,
                base_url="http://testserver",
            ) as client:
                return await client.options(
                    "/health",
                    headers={
                        "origin": "http://frontend.test",
                        "access-control-request-method": "GET",
                    },
                )

        response = asyncio.run(call_api())
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.headers["access-control-allow-origin"],
            "http://frontend.test",
        )

    def test_api_inspector_errors_use_shared_handler(self) -> None:
        import httpx
        from viewer.backend.api import app

        async def call_api() -> httpx.Response:
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport,
                base_url="http://testserver",
            ) as client:
                return await client.get("/models/not_a_model/presets")

        response = asyncio.run(call_api())
        self.assertEqual(response.status_code, 400)
        self.assertIn("Unknown model", response.json()["detail"])

    def test_one_preset_in_every_model_package_is_inspectable(self) -> None:
        for model in discover_models():
            with self.subTest(model=model):
                preset = list_model_presets(model)[0]["name"]
                result = inspect_model(model, preset)
                self.assertGreater(len(result["nodes"]), 0)
                self.assertGreater(len(result["edges"]), 0)

    def test_bert_linear_and_vit_linear_no_deleted_transformer_utils_imports(
        self,
    ) -> None:
        for module_name in ("models.bert_linear.presets", "models.vit_linear.presets"):
            with self.subTest(module=module_name):
                module = importlib.import_module(module_name)
                self.assertTrue(hasattr(module, "ExperimentPresets"))
