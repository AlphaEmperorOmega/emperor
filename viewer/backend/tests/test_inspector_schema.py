from __future__ import annotations

import os
import unittest

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from viewer.backend.inspector.schema import config_schema, search_space_schema


class InspectorSchemaTests(unittest.TestCase):
    def test_config_schema_exposes_supported_field_types(self) -> None:
        linear_fields = {
            field["key"]: field for field in config_schema("linears/linear")["fields"]
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
            field["key"]: field
            for field in config_schema("transformer_encoder/vit_linear")["fields"]
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

    def test_config_schema_serializes_value_defaults(self) -> None:
        linear_fields = {
            field["key"]: field for field in config_schema("linears/linear")["fields"]
        }
        adaptive_fields = {
            field["key"]: field
            for field in config_schema("linears/linear_adaptive")["fields"]
        }
        vit_fields = {
            field["key"]: field
            for field in config_schema("transformer_encoder/vit_linear")["fields"]
        }

        self.assertEqual(linear_fields["hidden_dim"]["default"], 256)
        self.assertEqual(linear_fields["stack_activation"]["default"], "GELU")
        self.assertEqual(
            vit_fields["positional_embedding_option"]["default"],
            "ImageLearnedPositionalEmbeddingConfig",
        )
        self.assertIsNone(adaptive_fields["input_layer_model_option"]["default"])

    def test_config_schema_excludes_abstract_class_choices(self) -> None:
        fields = {
            field["key"]: field
            for field in config_schema("linears/linear_adaptive")["fields"]
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
            field["key"]: field
            for field in config_schema("linears/linear_adaptive")["fields"]
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
            for field in config_schema("linears/linear", "baseline")["fields"]
        }
        gating_fields = {
            field["key"]: field
            for field in config_schema("linears/linear", "gating")["fields"]
        }

        self.assertFalse(baseline_fields["gate_flag"]["locked"])
        self.assertTrue(gating_fields["gate_flag"]["locked"])
        self.assertEqual(gating_fields["gate_flag"]["lockedValue"], True)
        self.assertIn("GATING preset", gating_fields["gate_flag"]["lockedReason"])

    def test_search_space_schema_exposes_linear_axes(self) -> None:
        axes = {
            axis["key"]: axis
            for axis in search_space_schema("linears/linear", "baseline")["axes"]
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

    def test_search_space_schema_serializes_axis_values(self) -> None:
        linear_axes = {
            axis["key"]: axis
            for axis in search_space_schema("linears/linear", "baseline")["axes"]
        }
        adaptive_axes = {
            axis["key"]: axis
            for axis in search_space_schema("linears/linear_adaptive", "baseline")[
                "axes"
            ]
        }

        self.assertEqual(
            linear_axes["hidden_dim"]["values"],
            [16, 32, 64, 128, 256, 512],
        )
        self.assertEqual(
            linear_axes["stack_activation"]["values"],
            ["RELU", "LEAKY_RELU", "ELU", "GELU", "TANH"],
        )
        self.assertEqual(
            adaptive_axes["input_layer_model_option"]["values"],
            [None, "AdaptiveLinearLayerConfig"],
        )

    def test_search_space_schema_marks_preset_owned_axes_locked(self) -> None:
        axes = {
            axis["key"]: axis
            for axis in search_space_schema("linears/linear", "post-norm")["axes"]
        }

        self.assertTrue(axes["layer_norm_position"]["locked"])
        self.assertEqual(axes["layer_norm_position"]["lockedValue"], "AFTER")
        self.assertIn("POST_NORM preset", axes["layer_norm_position"]["lockedReason"])


if __name__ == "__main__":
    unittest.main()
