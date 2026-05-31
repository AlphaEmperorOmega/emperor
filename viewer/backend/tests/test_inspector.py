from __future__ import annotations

import importlib
import asyncio
import os
import unittest
from collections import Counter

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from torch import nn

from viewer.backend.inspector.discovery import discover_models, list_model_presets
from viewer.backend.inspector.errors import InspectorError
from viewer.backend.inspector.graph import serialize_graph
from viewer.backend.inspector.schema import config_schema
from viewer.backend.inspector.service import inspect_model


class InspectorTests(unittest.TestCase):
    def test_model_discovery_lists_expected_packages(self) -> None:
        models = set(discover_models())
        self.assertGreaterEqual(
            models,
            {
                "bert",
                "experts_linear",
                "experts_linear_adaptive",
                "linear",
                "linear_adaptive",
                "parametric_generator",
                "parametric_matrix",
                "parametric_vector",
                "vit",
            },
        )

    def test_preset_discovery_for_linear(self) -> None:
        presets = list_model_presets("linear")
        preset_names = {preset["name"] for preset in presets}
        self.assertIn("baseline", preset_names)
        self.assertIn("recurrent-gating-halting", preset_names)

    def test_config_schema_exposes_supported_field_types(self) -> None:
        linear_fields = {
            field["key"]: field for field in config_schema("linear")["fields"]
        }
        self.assertEqual(linear_fields["hidden_dim"]["type"], "int")
        self.assertEqual(linear_fields["learning_rate"]["type"], "float")
        self.assertEqual(linear_fields["gate_flag"]["type"], "bool")
        self.assertEqual(linear_fields["stack_activation"]["type"], "enum")
        self.assertIn("GELU", linear_fields["stack_activation"]["choices"])
        self.assertEqual(linear_fields["hidden_dim"]["section"], "Layer Stack Options")
        self.assertEqual(
            linear_fields["recurrent_flag"]["section"],
            "Recurrent Layer Options",
        )

        vit_fields = {field["key"]: field for field in config_schema("vit")["fields"]}
        self.assertEqual(vit_fields["adaptive_attn_bias_option"]["type"], "class")
        self.assertNotIn(
            "DynamicBiasConfig",
            vit_fields["adaptive_attn_bias_option"]["choices"],
        )

    def test_config_schema_excludes_abstract_class_choices(self) -> None:
        fields = {
            field["key"]: field
            for field in config_schema("linear_adaptive")["fields"]
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

    def test_override_parsing_changes_linear_hidden_dim_graph_details(self) -> None:
        result = inspect_model("linear", "baseline", {"hidden_dim": "128"})
        node_by_id = {node["id"]: node for node in result["nodes"]}
        self.assertEqual(node_by_id["main_model.0"]["details"]["dims"], "128 -> 128")
        self.assertEqual(node_by_id["output_model"]["details"]["dims"], "128 -> 10")

    def test_inspect_response_includes_top_level_parameter_count(self) -> None:
        result = inspect_model("linear", "baseline")

        self.assertGreater(result["parameterCount"], 0)
        self.assertEqual(result["parameterCount"], result["nodes"][0]["parameterCount"])

    def test_config_override_aliases_match_builder_parameter_names(self) -> None:
        result = inspect_model("linear_adaptive", "baseline", {"gate_flag": "true"})
        self.assertTrue(
            any(node["details"].get("gate") is True for node in result["nodes"])
        )

    def test_abstract_config_override_is_rejected_before_model_instantiation(self) -> None:
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

    def test_graph_serializer_skips_uninitialized_lazy_parameters(self) -> None:
        nodes, _edges = serialize_graph(nn.Sequential(nn.LazyLinear(3)))
        count_by_id = {node["id"]: node["parameterCount"] for node in nodes}

        self.assertEqual(count_by_id["__root__"], 0)
        self.assertEqual(count_by_id["0"], 0)

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

        async def call_api() -> tuple[httpx.Response, httpx.Response]:
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport,
                base_url="http://testserver",
            ) as client:
                health = await client.get("/health")
                inspect_response = await client.post(
                    "/inspect",
                    json={
                        "model": "linear",
                        "preset": "baseline",
                        "overrides": {"hidden_dim": "128"},
                    },
                )
                return health, inspect_response

        health_response, response = asyncio.run(call_api())
        self.assertEqual(health_response.json(), {"status": "ok"})
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["model"], "linear")
        self.assertTrue(payload["nodes"])
        self.assertTrue(payload["edges"])

    def test_one_preset_in_every_model_package_is_inspectable(self) -> None:
        for model in discover_models():
            with self.subTest(model=model):
                preset = list_model_presets(model)[0]["name"]
                result = inspect_model(model, preset)
                self.assertGreater(len(result["nodes"]), 0)
                self.assertGreater(len(result["edges"]), 0)

    def test_bert_and_vit_no_deleted_transformer_utils_imports(self) -> None:
        for module_name in ("models.bert.presets", "models.vit.presets"):
            with self.subTest(module=module_name):
                module = importlib.import_module(module_name)
                self.assertTrue(hasattr(module, "ExperimentPresets"))
