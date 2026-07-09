from __future__ import annotations

import os
import unittest

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import models.experts.linear.config as expert_linear_config
import models.linears.linear.config as linears_linear_config
import models.neuron.linear.config as neuron_config
import models.vit.linear.config as vit_config
from emperor.base.layer.gate import LayerGateOptions
from emperor.base.options import LayerNormPositionOptions

from workbench.backend.inspector.errors import InspectorError
from workbench.backend.inspector.overrides import parse_override_mapping
from workbench.backend.inspector.service import inspect_model


class InspectorServiceTests(unittest.TestCase):
    def assert_neuron_graph_dims(
        self,
        result,
        hidden_dim: int,
    ) -> None:
        node_by_id = {node["id"]: node for node in result["nodes"]}

        self.assertEqual(
            node_by_id["input_model"]["details"]["dims"],
            f"784 -> {hidden_dim}",
        )
        self.assertEqual(
            node_by_id["neuron_cluster"]["details"]["inputDim"],
            hidden_dim,
        )
        self.assertEqual(
            node_by_id["output_model"]["details"]["dims"],
            f"{hidden_dim} -> 10",
        )

    def test_override_parsing_changes_linear_hidden_dim_graph_details(self) -> None:
        result = inspect_model(
            "linears/linear",
            "baseline",
            {"hidden_dim": "128"},
        )
        node_by_id = {node["id"]: node for node in result["nodes"]}
        main_layer_details = node_by_id["main_model.layers.0"]["details"]
        output_layer_details = node_by_id["output_model"]["details"]

        self.assertEqual(main_layer_details["dims"], "128 -> 128")
        self.assertEqual(output_layer_details["dims"], "128 -> 10")
        self.assertNotIn("inputShape", main_layer_details)
        self.assertNotIn("outputShape", main_layer_details)
        self.assertNotIn("shapeTransition", main_layer_details)
        self.assertNotIn("inputShape", output_layer_details)
        self.assertNotIn("outputShape", output_layer_details)
        self.assertNotIn("shapeTransition", output_layer_details)

    def test_neuron_baseline_uses_wrapper_source_defaults(self) -> None:
        result = inspect_model("neuron/linear", "baseline")

        self.assert_neuron_graph_dims(
            result,
            hidden_dim=neuron_config.HIDDEN_DIM,
        )

    def test_neuron_hidden_dim_override_flows_through_graph(self) -> None:
        result = inspect_model(
            "neuron/linear",
            "baseline",
            {"hidden_dim": "64"},
        )

        self.assert_neuron_graph_dims(result, hidden_dim=64)

    def test_inspect_uses_selected_dataset_dimensions(self) -> None:
        result = inspect_model("linears/linear", "baseline", dataset="Cifar100")
        node_by_id = {node["id"]: node for node in result["nodes"]}

        self.assertEqual(
            node_by_id["output_model"]["details"]["dims"],
            f"{linears_linear_config.HIDDEN_DIM} -> 100",
        )

    def test_inspect_rejects_path_like_dataset_input(self) -> None:
        with self.assertRaises(InspectorError) as context:
            inspect_model("linears/linear", "baseline", dataset="./Mnist")

        message = str(context.exception)
        self.assertIn("./Mnist", message)
        self.assertIn("filesystem path", message)
        self.assertIn("server-known dataset name", message)

    def test_locked_preset_override_is_rejected_for_inspect(self) -> None:
        with self.assertRaises(InspectorError) as context:
            inspect_model("linears/linear", "gating", {"gate_flag": "false"})

        self.assertIn("locked fields", str(context.exception))
        self.assertIn("stack_gate_flag", str(context.exception))

    def test_inspect_response_includes_top_level_parameter_count_and_size(self) -> None:
        result = inspect_model("linears/linear", "baseline")

        self.assertGreater(result["parameterCount"], 0)
        self.assertGreater(result["parameterSizeBytes"], 0)
        self.assertEqual(result["parameterCount"], result["nodes"][0]["parameterCount"])
        self.assertEqual(
            result["parameterSizeBytes"],
            result["nodes"][0]["parameterSizeBytes"],
        )

    def test_inspect_reports_local_linear_weight_and_bias_shapes_on_owner(self) -> None:
        result = inspect_model("linears/linear", "baseline")
        node_by_id = {node["id"]: node for node in result["nodes"]}

        self.assertNotIn("weightShape", node_by_id["main_model.layers.0"]["details"])
        self.assertNotIn("biasShape", node_by_id["main_model.layers.0"]["details"])
        self.assertEqual(
            node_by_id["main_model.layers.0.model"]["details"]["weightShape"],
            (
                f"{linears_linear_config.HIDDEN_DIM} x "
                f"{linears_linear_config.HIDDEN_DIM}"
            ),
        )
        self.assertEqual(
            node_by_id["main_model.layers.0.model"]["details"]["biasShape"],
            str(linears_linear_config.HIDDEN_DIM),
        )

    def test_config_override_aliases_match_builder_parameter_names(self) -> None:
        result = inspect_model(
            "linears/linear_adaptive", "baseline", {"gate_flag": "true"}
        )
        self.assertTrue(
            any(node["details"].get("gate") is True for node in result["nodes"])
        )

    def test_gate_option_overrides_parse_to_builder_parameter_names(self) -> None:
        parsed = parse_override_mapping(
            linears_linear_config,
            {
                "gate_option": "ADDITION",
                "recurrent_gate_option": "MULTIPLIER",
            },
        )

        self.assertIs(parsed["gate_option"], LayerGateOptions.ADDITION)
        self.assertIs(parsed["recurrent_gate_option"], LayerGateOptions.MULTIPLIER)

    def test_stack_alias_overrides_parse_to_builder_parameter_names(self) -> None:
        parsed = parse_override_mapping(
            linears_linear_config,
            {
                "hidden_dim": "128",
                "stack_layer_norm_position": "AFTER",
                "stack_bias_flag": "false",
                "gate_stack_independent_flag": "true",
                "gate_stack_hidden_dim": "32",
                "gate_stack_layer_norm_position": "BEFORE",
                "gate_stack_bias_flag": "true",
                "halting_stack_independent_flag": "true",
                "halting_stack_hidden_dim": "48",
                "memory_stack_independent_flag": "true",
                "memory_stack_hidden_dim": "64",
                "recurrent_gate_stack_independent_flag": "true",
                "recurrent_gate_stack_hidden_dim": "96",
                "recurrent_halting_stack_independent_flag": "true",
                "recurrent_halting_stack_hidden_dim": "112",
            },
        )

        self.assertEqual(parsed["hidden_dim"], 128)
        self.assertIs(parsed["layer_norm_position"], LayerNormPositionOptions.AFTER)
        self.assertFalse(parsed["stack_bias_flag"])
        self.assertTrue(parsed["gate_stack_independent_flag"])
        self.assertEqual(parsed["gate_stack_hidden_dim"], 32)
        self.assertIs(
            parsed["gate_stack_layer_norm_position"],
            LayerNormPositionOptions.BEFORE,
        )
        self.assertTrue(parsed["gate_stack_bias_flag"])
        self.assertTrue(parsed["halting_stack_independent_flag"])
        self.assertEqual(parsed["halting_stack_hidden_dim"], 48)
        self.assertTrue(parsed["memory_stack_independent_flag"])
        self.assertEqual(parsed["memory_stack_hidden_dim"], 64)
        self.assertTrue(parsed["recurrent_gate_stack_independent_flag"])
        self.assertEqual(parsed["recurrent_gate_stack_hidden_dim"], 96)
        self.assertTrue(parsed["recurrent_halting_stack_independent_flag"])
        self.assertEqual(parsed["recurrent_halting_stack_hidden_dim"], 112)

    def test_legacy_controller_stack_overrides_are_rejected(self) -> None:
        for override_key in (
            "gate_hidden_dim",
            "gate_layer_norm_position",
            "gate_bias_flag",
        ):
            with self.subTest(override_key=override_key):
                with self.assertRaisesRegex(InspectorError, "Unknown override"):
                    parse_override_mapping(
                        linears_linear_config,
                        {override_key: "32"},
                    )

    def test_legacy_controller_stack_overrides_are_rejected_for_experts(self) -> None:
        for override_key in (
            "gate_hidden_dim",
            "gate_layer_norm_position",
            "gate_bias_flag",
            "halting_hidden_dim",
            "halting_layer_norm_position",
            "halting_bias_flag",
        ):
            with self.subTest(override_key=override_key):
                with self.assertRaisesRegex(InspectorError, "Unknown override"):
                    parse_override_mapping(
                        expert_linear_config,
                        {override_key: "32"},
                    )

    def test_nullable_empty_string_override_parses_as_none(self) -> None:
        parsed = parse_override_mapping(
            vit_config,
            {"positional_embedding_padding_idx": ""},
        )

        self.assertIsNone(parsed["positional_embedding_padding_idx"])

    def test_empty_string_override_still_rejects_non_nullable_fields(self) -> None:
        with self.assertRaises(InspectorError):
            parse_override_mapping(vit_config, {"hidden_dim": ""})

    def test_legacy_residual_flag_override_maps_to_connection_option(self) -> None:
        cases = (
            ("true", "RESIDUAL", True),
            ("false", "DISABLED", False),
        )
        for raw_value, expected_option, expects_residual_node in cases:
            with self.subTest(raw_value=raw_value):
                result = inspect_model(
                    "linears/linear",
                    "baseline",
                    {"stack_residual_flag": raw_value},
                )

                node_by_id = {node["id"]: node for node in result["nodes"]}
                layer_config_fields = {
                    field["key"]: field["value"]
                    for field in node_by_id["main_model.layers.0"]["config"]["fields"]
                }
                self.assertEqual(
                    layer_config_fields["residual_connection_option"],
                    expected_option,
                )
                self.assertEqual(
                    "main_model.layers.0.residual_connection" in node_by_id,
                    expects_residual_node,
                )

    def test_abstract_config_override_is_rejected_before_model_instantiation(
        self,
    ) -> None:
        with self.assertRaises(InspectorError) as context:
            inspect_model(
                "linears/linear_adaptive",
                "baseline",
                {"row_mask_option": "AxisMaskConfig"},
            )

        message = str(context.exception)
        self.assertIn("Invalid value for override 'row_mask_option'", message)
        self.assertNotIn("Failed to instantiate model", message)


if __name__ == "__main__":
    unittest.main()
