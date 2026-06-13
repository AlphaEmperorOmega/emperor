from __future__ import annotations

import os
import unittest

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import models.neuron.neuron_linear.config as neuron_linear_config

from viewer.backend.inspector.errors import InspectorError
from viewer.backend.inspector.service import inspect_model


class InspectorServiceTests(unittest.TestCase):
    def assert_neuron_linear_graph_dims(
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
        result = inspect_model("linears/linear", "baseline", {"hidden_dim": "128"})
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

    def test_neuron_linear_baseline_uses_wrapper_source_defaults(self) -> None:
        result = inspect_model("neuron/neuron_linear", "baseline")

        self.assert_neuron_linear_graph_dims(
            result,
            hidden_dim=neuron_linear_config.HIDDEN_DIM,
        )

    def test_neuron_linear_hidden_dim_override_flows_through_graph(self) -> None:
        result = inspect_model(
            "neuron/neuron_linear",
            "baseline",
            {"hidden_dim": "64"},
        )

        self.assert_neuron_linear_graph_dims(result, hidden_dim=64)

    def test_inspect_uses_selected_dataset_dimensions(self) -> None:
        result = inspect_model("linears/linear", "baseline", dataset="Cifar100")
        node_by_id = {node["id"]: node for node in result["nodes"]}

        self.assertEqual(node_by_id["output_model"]["details"]["dims"], "256 -> 100")

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
            "256 x 256",
        )
        self.assertEqual(
            node_by_id["main_model.layers.0.model"]["details"]["biasShape"],
            "256",
        )

    def test_config_override_aliases_match_builder_parameter_names(self) -> None:
        result = inspect_model(
            "linears/linear_adaptive", "baseline", {"gate_flag": "true"}
        )
        self.assertTrue(
            any(node["details"].get("gate") is True for node in result["nodes"])
        )

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
