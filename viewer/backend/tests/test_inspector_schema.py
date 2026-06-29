from __future__ import annotations

import argparse
import importlib
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import models.linears.linear.config as linear_config
from emperor.base.layer.gate import LayerGateOptions
from emperor.base.options import ActivationOptions
from emperor.memory.config import WeightedDynamicMemoryConfig
from models.config_overrides import parse_config_value

from viewer.backend.inspector.schema import config_schema, search_space_schema


def _fields_by_key(payload: dict) -> dict[str, dict]:
    return {field["key"].lower(): field for field in payload["fields"]}


def _axes_by_key(payload: dict) -> dict[str, dict]:
    return {axis["key"].lower(): axis for axis in payload["axes"]}


class InspectorSchemaTests(unittest.TestCase):
    def test_config_schema_emits_uppercase_public_keys(self) -> None:
        fields = config_schema("linears/linear")["fields"]

        stack_field = next(
            field
            for field in fields
            if field["configKey"] == "STACK_APPLY_OUTPUT_PIPELINE_FLAG"
        )
        self.assertEqual(stack_field["key"], "STACK_APPLY_OUTPUT_PIPELINE_FLAG")
        self.assertEqual(stack_field["key"], stack_field["configKey"])
        self.assertEqual(stack_field["flag"], "--stack-apply-output-pipeline-flag")
        for field in fields:
            self.assertEqual(field["key"], field["configKey"])
            self.assertTrue(field["key"].isupper())

    def test_search_schema_emits_uppercase_axis_keys(self) -> None:
        axes = search_space_schema("linears/linear", "baseline")["axes"]

        stack_axis = next(
            axis
            for axis in axes
            if axis["searchKey"] == "SEARCH_SPACE_STACK_HIDDEN_DIM"
        )
        self.assertEqual(stack_axis["key"], "STACK_HIDDEN_DIM")
        self.assertEqual(stack_axis["key"], stack_axis["configKey"])
        self.assertEqual(stack_axis["searchKey"], "SEARCH_SPACE_STACK_HIDDEN_DIM")
        for axis in axes:
            self.assertEqual(axis["key"], axis["configKey"])
            self.assertTrue(axis["key"].isupper())

    def test_config_schema_sections_imported_trainer_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            shared_module_name = "shared_trainer_config_fixture"
            model_module_name = "model_config_fixture"
            (temp_path / f"{shared_module_name}.py").write_text(
                "\n".join(
                    [
                        '"""Shared trainer defaults for schema tests."""',
                        "TRAINER_MAX_STEPS: int = -1",
                        "TRAINER_PRECISION: str = '32-true'",
                        "CALLBACK_CHECKPOINT_FLAG: bool = False",
                    ]
                ),
                encoding="utf-8",
            )
            (temp_path / f"{model_module_name}.py").write_text(
                "\n".join(
                    [
                        f"from {shared_module_name} import *",
                        "",
                        "# Trainer",
                        "TRAINER_ACCELERATOR: str = 'cpu'",
                        "",
                        "# Model",
                        "STACK_HIDDEN_DIM: int = 128",
                    ]
                ),
                encoding="utf-8",
            )
            sys.path.insert(0, temp_dir)
            try:
                importlib.invalidate_caches()
                config_module = importlib.import_module(model_module_name)
                parts = SimpleNamespace(config_module=config_module)
                with patch(
                    "viewer.backend.inspector.schema.load_model_parts",
                    return_value=parts,
                ):
                    fields = _fields_by_key(config_schema("test/model"))
            finally:
                sys.path.remove(temp_dir)
                sys.modules.pop(model_module_name, None)
                sys.modules.pop(shared_module_name, None)

        self.assertEqual(fields["trainer_accelerator"]["section"], "Trainer")
        self.assertEqual(fields["trainer_max_steps"]["section"], "Trainer")
        self.assertEqual(fields["trainer_precision"]["section"], "Trainer")
        self.assertEqual(fields["callback_checkpoint_flag"]["section"], "Callback")
        self.assertEqual(fields["stack_hidden_dim"]["section"], "Model")

    def test_config_schema_exposes_supported_field_types(self) -> None:
        linear_fields = _fields_by_key(config_schema("linears/linear"))
        self.assertEqual(linear_fields["stack_hidden_dim"]["type"], "int")
        self.assertEqual(linear_fields["stack_hidden_dim"]["default"], 256)
        self.assertEqual(linear_fields["stack_layer_norm_position"]["type"], "enum")
        self.assertEqual(linear_fields["stack_bias_flag"]["type"], "bool")
        adaptive_fields = _fields_by_key(config_schema("linears/linear_adaptive"))
        self.assertEqual(adaptive_fields["stack_bias_flag"]["type"], "bool")
        self.assertEqual(
            adaptive_fields["stack_bias_flag"]["section"],
            "Layer Stack Options",
        )
        self.assertTrue(adaptive_fields["stack_bias_flag"]["default"])
        self.assertEqual(linear_fields["stack_num_layers"]["type"], "int")
        self.assertEqual(linear_fields["stack_num_layers"]["default"], 5)
        self.assertEqual(linear_fields["stack_num_layers"]["choices"], [])
        self.assertEqual(linear_fields["learning_rate"]["type"], "float")
        self.assertEqual(linear_fields["gate_flag"]["type"], "bool")
        self.assertEqual(linear_fields["stack_activation"]["type"], "enum")
        self.assertIn("GELU", linear_fields["stack_activation"]["choices"])
        self.assertEqual(linear_fields["memory_flag"]["type"], "bool")
        self.assertEqual(linear_fields["gate_option"]["type"], "enum")
        self.assertEqual(linear_fields["gate_option"]["default"], "MULTIPLIER")
        self.assertTrue(linear_fields["gate_option"]["nullable"])
        self.assertIn("MULTIPLIER", linear_fields["gate_option"]["choices"])
        self.assertIn("ADDITION", linear_fields["gate_option"]["choices"])
        self.assertNotIn("INTERPOLATION", linear_fields["gate_option"]["choices"])
        self.assertNotIn("DISABLED", linear_fields["gate_option"]["choices"])
        self.assertEqual(linear_fields["gate_activation"]["type"], "enum")
        self.assertEqual(linear_fields["gate_activation"]["default"], "SIGMOID")
        self.assertTrue(linear_fields["gate_activation"]["nullable"])
        self.assertIn("TANH", linear_fields["gate_activation"]["choices"])
        self.assertEqual(linear_fields["gate_stack_independent_flag"]["type"], "bool")
        self.assertFalse(linear_fields["gate_stack_independent_flag"]["default"])
        self.assertEqual(linear_fields["gate_stack_hidden_dim"]["type"], "int")
        self.assertIsNone(linear_fields["gate_stack_hidden_dim"]["default"])
        self.assertTrue(linear_fields["gate_stack_hidden_dim"]["nullable"])
        self.assertEqual(linear_fields["gate_stack_bias_flag"]["type"], "bool")
        self.assertTrue(linear_fields["gate_stack_bias_flag"]["default"])
        self.assertTrue(linear_fields["gate_stack_bias_flag"]["nullable"])
        self.assertNotIn("gate_hidden_dim", linear_fields)
        self.assertNotIn("gate_layer_norm_position", linear_fields)
        self.assertNotIn("gate_bias_flag", linear_fields)
        self.assertEqual(
            linear_fields["gate_option"]["section"],
            "Gate Options",
        )
        self.assertEqual(
            linear_fields["gate_stack_hidden_dim"]["section"],
            "Gate Stack Options",
        )
        self.assertEqual(linear_fields["recurrent_gate_option"]["type"], "enum")
        self.assertEqual(
            linear_fields["recurrent_gate_option"]["default"],
            "MULTIPLIER",
        )
        self.assertTrue(linear_fields["recurrent_gate_option"]["nullable"])
        self.assertIn(
            "MULTIPLIER",
            linear_fields["recurrent_gate_option"]["choices"],
        )
        self.assertIn(
            "ADDITION",
            linear_fields["recurrent_gate_option"]["choices"],
        )
        self.assertNotIn(
            "INTERPOLATION",
            linear_fields["recurrent_gate_option"]["choices"],
        )
        self.assertNotIn(
            "DISABLED",
            linear_fields["recurrent_gate_option"]["choices"],
        )
        self.assertEqual(linear_fields["recurrent_gate_activation"]["type"], "enum")
        self.assertEqual(
            linear_fields["recurrent_gate_activation"]["default"],
            "SIGMOID",
        )
        self.assertTrue(linear_fields["recurrent_gate_activation"]["nullable"])
        self.assertIn(
            "TANH",
            linear_fields["recurrent_gate_activation"]["choices"],
        )
        self.assertEqual(
            linear_fields["recurrent_gate_stack_independent_flag"]["type"],
            "bool",
        )
        self.assertFalse(
            linear_fields["recurrent_gate_stack_independent_flag"]["default"],
        )
        self.assertEqual(
            linear_fields["recurrent_gate_stack_hidden_dim"]["type"],
            "int",
        )
        self.assertIsNone(linear_fields["recurrent_gate_stack_hidden_dim"]["default"])
        self.assertTrue(linear_fields["recurrent_gate_stack_hidden_dim"]["nullable"])
        self.assertEqual(
            linear_fields["recurrent_gate_stack_num_layers"]["type"],
            "int",
        )
        self.assertIsNone(
            linear_fields["recurrent_gate_stack_num_layers"]["default"],
        )
        self.assertTrue(
            linear_fields["recurrent_gate_stack_num_layers"]["nullable"],
        )
        self.assertEqual(
            linear_fields["recurrent_gate_stack_activation"]["type"],
            "enum",
        )
        self.assertIsNone(
            linear_fields["recurrent_gate_stack_activation"]["default"],
        )
        self.assertTrue(
            linear_fields["recurrent_gate_stack_activation"]["nullable"],
        )
        self.assertIn(
            "TANH",
            linear_fields["recurrent_gate_stack_activation"]["choices"],
        )
        self.assertNotIn("recurrent_gate_hidden_dim", linear_fields)
        self.assertNotIn("recurrent_gate_layer_norm_position", linear_fields)
        self.assertNotIn("recurrent_gate_bias_flag", linear_fields)
        self.assertEqual(
            linear_fields["recurrent_gate_option"]["section"],
            "Recurrent Gate Options",
        )
        self.assertEqual(
            linear_fields["recurrent_gate_stack_hidden_dim"]["section"],
            "Recurrent Gate Stack Options",
        )
        self.assertEqual(linear_fields["recurrent_layer_norm_position"]["type"], "enum")
        self.assertEqual(
            linear_fields["recurrent_layer_norm_position"]["default"],
            "DISABLED",
        )
        self.assertFalse(linear_fields["recurrent_layer_norm_position"]["nullable"])
        self.assertIn(
            "BEFORE",
            linear_fields["recurrent_layer_norm_position"]["choices"],
        )
        self.assertIn(
            "DEFAULT",
            linear_fields["recurrent_layer_norm_position"]["choices"],
        )
        self.assertIn(
            "AFTER",
            linear_fields["recurrent_layer_norm_position"]["choices"],
        )
        self.assertEqual(
            linear_fields["recurrent_layer_norm_position"]["section"],
            "Recurrent Layer Options",
        )
        self.assertEqual(linear_fields["memory_option"]["type"], "class")
        self.assertIn(
            "WeightedDynamicMemoryConfig",
            linear_fields["memory_option"]["choices"],
        )
        self.assertEqual(linear_fields["memory_position_option"]["type"], "enum")
        self.assertEqual(
            linear_fields["memory_test_time_training_learning_rate"]["type"],
            "float",
        )
        self.assertEqual(
            linear_fields["memory_test_time_training_num_inner_steps"]["type"],
            "int",
        )
        self.assertEqual(
            linear_fields["stack_hidden_dim"]["section"],
            "Layer Stack Options",
        )
        self.assertEqual(linear_fields["memory_flag"]["section"], "Memory Options")
        self.assertEqual(
            linear_fields["recurrent_flag"]["section"],
            "Recurrent Layer Options",
        )
        self.assertEqual(
            linear_fields["submodule_stack_hidden_dim"]["section"],
            "Layer Stack Submodule Options",
        )

        vit_fields = _fields_by_key(config_schema("transformer_encoder/vit_linear"))
        self.assertEqual(vit_fields["positional_embedding_option"]["type"], "class")
        self.assertIn(
            "ImageLearnedPositionalEmbeddingConfig",
            vit_fields["positional_embedding_option"]["choices"],
        )
        self.assertNotIn(
            "AbsolutePositionalEmbeddingConfig",
            vit_fields["positional_embedding_option"]["choices"],
        )

    def test_config_schema_exposes_field_descriptions(self) -> None:
        adaptive_fields = _fields_by_key(config_schema("linears/linear_adaptive"))

        self.assertTrue(
            all(field["description"].strip() for field in adaptive_fields.values())
        )
        self.assertIn(
            "optimizer steps",
            adaptive_fields["trainer_max_steps"]["description"],
        )
        self.assertIn(
            "validation data",
            adaptive_fields["trainer_limit_val_batches"]["description"],
        )
        self.assertIn(
            "model summary",
            adaptive_fields["trainer_enable_model_summary"]["description"],
        )
        self.assertIn(
            "profiling adds overhead",
            adaptive_fields["trainer_profiler"]["description"],
        )
        self.assertIn(
            "dedicated halting stack",
            adaptive_fields["halting_stack_bias_flag"]["description"],
        )
        self.assertIn(
            "output pipeline",
            adaptive_fields["halting_stack_apply_output_pipeline_flag"][
                "description"
            ],
        )
        self.assertIn(
            "smoke tests",
            adaptive_fields["run_test_after_fit"]["description"],
        )

    def test_config_schema_serializes_value_defaults(self) -> None:
        linear_fields = _fields_by_key(config_schema("linears/linear"))
        adaptive_fields = _fields_by_key(config_schema("linears/linear_adaptive"))
        vit_fields = _fields_by_key(config_schema("transformer_encoder/vit_linear"))

        self.assertEqual(linear_fields["stack_hidden_dim"]["default"], 256)
        self.assertEqual(linear_fields["stack_activation"]["default"], "GELU")
        self.assertEqual(
            vit_fields["positional_embedding_option"]["default"],
            "ImageLearnedPositionalEmbeddingConfig",
        )
        self.assertFalse(
            any(key.endswith("_layer_adaptive_flag") for key in adaptive_fields)
        )

    def test_linear_schemas_do_not_expose_halting_output_dims(self) -> None:
        removed_field_keys = {
            "halting_output_dim",
            "recurrent_halting_output_dim",
        }
        removed_config_keys = {
            "HALTING_OUTPUT_DIM",
            "RECURRENT_HALTING_OUTPUT_DIM",
        }

        for model_name in ("linears/linear", "linears/linear_adaptive"):
            payload = config_schema(model_name)
            fields = _fields_by_key(payload)
            config_keys = {field["configKey"] for field in payload["fields"]}

            with self.subTest(model_name=model_name):
                for field_key in removed_field_keys:
                    self.assertNotIn(field_key, fields)
                for config_key in removed_config_keys:
                    self.assertNotIn(config_key, config_keys)

    def test_linear_adaptive_schema_uses_controller_stack_field_names(self) -> None:
        fields = _fields_by_key(config_schema("linears/linear_adaptive"))
        expected_stack_fields = {
            "gate_stack_hidden_dim",
            "gate_stack_layer_norm_position",
            "gate_stack_bias_flag",
            "halting_stack_hidden_dim",
            "halting_stack_layer_norm_position",
            "halting_stack_bias_flag",
            "memory_stack_hidden_dim",
            "memory_stack_layer_norm_position",
            "memory_stack_bias_flag",
            "recurrent_gate_stack_hidden_dim",
            "recurrent_gate_stack_layer_norm_position",
            "recurrent_gate_stack_bias_flag",
            "recurrent_halting_stack_hidden_dim",
            "recurrent_halting_stack_layer_norm_position",
            "recurrent_halting_stack_bias_flag",
        }
        legacy_fields = {
            name.replace("_stack_", "_") for name in expected_stack_fields
        }

        for field_name in expected_stack_fields:
            with self.subTest(field_name=field_name):
                self.assertIn(field_name, fields)

        for field_name in legacy_fields:
            with self.subTest(field_name=field_name):
                self.assertNotIn(field_name, fields)

    def test_model_schemas_use_controller_stack_field_names(self) -> None:
        expected_by_model = {
            "experts/experts_linear": {
                "gate_stack_hidden_dim",
                "gate_stack_layer_norm_position",
                "gate_stack_bias_flag",
                "halting_stack_hidden_dim",
                "halting_stack_layer_norm_position",
                "halting_stack_bias_flag",
            },
            "experts/experts_linear_adaptive": {
                "gate_stack_hidden_dim",
                "gate_stack_layer_norm_position",
                "gate_stack_bias_flag",
                "halting_stack_hidden_dim",
                "halting_stack_layer_norm_position",
                "halting_stack_bias_flag",
            },
            "neuron/neuron_linear": {
                "gate_stack_hidden_dim",
                "gate_stack_layer_norm_position",
                "gate_stack_bias_flag",
                "halting_stack_hidden_dim",
                "halting_stack_layer_norm_position",
                "halting_stack_bias_flag",
                "cluster_halting_stack_hidden_dim",
                "cluster_halting_stack_layer_norm_position",
                "cluster_halting_stack_bias_flag",
            },
        }

        for model_name, expected_stack_fields in expected_by_model.items():
            fields = _fields_by_key(config_schema(model_name))
            legacy_fields = {
                field_name.replace("_stack_", "_")
                for field_name in expected_stack_fields
            }
            with self.subTest(model_name=model_name):
                for field_name in expected_stack_fields:
                    self.assertIn(field_name, fields)
                for field_name in legacy_fields:
                    self.assertNotIn(field_name, fields)

    def test_config_schema_excludes_abstract_class_choices(self) -> None:
        fields = _fields_by_key(config_schema("linears/linear_adaptive"))

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

    def test_config_schema_orders_adaptive_class_choices_from_search_space(
        self,
    ) -> None:
        fields = _fields_by_key(config_schema("linears/linear_adaptive", "baseline"))

        self.assertEqual(
            fields["weight_option"]["choices"][0],
            "SingleModelDynamicWeightConfig",
        )
        self.assertEqual(
            fields["bias_option"]["choices"][0],
            "AffineTransformDynamicBiasConfig",
        )
        self.assertEqual(
            fields["diagonal_option"]["choices"][0],
            "StandardDynamicDiagonalConfig",
        )
        self.assertEqual(
            fields["row_mask_option"]["choices"][0],
            "DiagonalAxisMaskConfig",
        )

    def test_config_schema_exposes_adaptive_component_option_flags(self) -> None:
        fields = _fields_by_key(config_schema("linears/linear_adaptive"))

        expected_flags = {
            "weight_option_flag": "Weight Generator Options",
            "bias_option_flag": "Bias Generator Options",
            "diagonal_option_flag": "Diagonal Generator Options",
            "mask_option_flag": "Mask Options",
        }
        for field_key, section in expected_flags.items():
            with self.subTest(field_key=field_key):
                self.assertEqual(fields[field_key]["type"], "bool")
                self.assertFalse(fields[field_key]["default"])
                self.assertEqual(fields[field_key]["choices"], [True, False])
                self.assertEqual(fields[field_key]["section"], section)

        self.assertEqual(
            fields["adaptive_submodule_stack_hidden_dim"]["section"],
            "Adaptive Submodule Stack Options",
        )
        self.assertEqual(
            fields["adaptive_submodule_stack_num_layers"]["section"],
            "Adaptive Submodule Stack Options",
        )
        self.assertNotEqual(
            fields["adaptive_submodule_stack_hidden_dim"]["section"],
            "Mask Options",
        )

        self.assertEqual(fields["weight_option"]["type"], "class")
        self.assertIn(
            "SingleModelDynamicWeightConfig",
            fields["weight_option"]["choices"],
        )
        self.assertEqual(fields["bias_option"]["type"], "class")
        self.assertIn(
            "AdditiveDynamicBiasConfig",
            fields["bias_option"]["choices"],
        )
        self.assertEqual(fields["diagonal_option"]["type"], "class")
        self.assertIn(
            "StandardDynamicDiagonalConfig",
            fields["diagonal_option"]["choices"],
        )
        self.assertEqual(fields["row_mask_option"]["type"], "class")
        self.assertEqual(fields["row_mask_option"]["section"], "Mask Options")
        self.assertIn(
            "WeightInformedScoreAxisMaskConfig",
            fields["row_mask_option"]["choices"],
        )
        self.assertEqual(
            fields["weight_generator_stack_independent_flag"]["section"],
            "Weight Generator Stack Options",
        )
        self.assertEqual(
            fields["bias_generator_stack_independent_flag"]["section"],
            "Bias Generator Stack Options",
        )
        self.assertEqual(
            fields["diagonal_generator_stack_independent_flag"]["section"],
            "Diagonal Generator Stack Options",
        )
        self.assertEqual(
            fields["mask_generator_stack_independent_flag"]["section"],
            "Mask Stack Options",
        )
        self.assertEqual(
            fields["mask_generator_stack_hidden_dim"]["section"],
            "Mask Stack Options",
        )

    def test_config_schema_exposes_boundary_projector_adaptive_options(self) -> None:
        fields = _fields_by_key(config_schema("linears/linear_adaptive"))

        self.assertNotIn("input_layer_model_option", fields)
        self.assertNotIn("output_layer_model_option", fields)
        self.assertFalse(any(key.endswith("_layer_adaptive_flag") for key in fields))
        expected_options = {
            "input_layer_weight_option": "Input Boundary Projector Options",
            "input_layer_bias_option": "Input Boundary Projector Options",
            "input_layer_diagonal_option": "Input Boundary Projector Options",
            "input_layer_row_mask_option": "Input Boundary Projector Options",
            "output_layer_weight_option": "Output Boundary Projector Options",
            "output_layer_bias_option": "Output Boundary Projector Options",
            "output_layer_diagonal_option": "Output Boundary Projector Options",
            "output_layer_row_mask_option": "Output Boundary Projector Options",
        }
        for field_key, section in expected_options.items():
            with self.subTest(field_key=field_key):
                self.assertEqual(fields[field_key]["section"], section)
                self.assertEqual(fields[field_key]["type"], "class")
                self.assertIsNone(fields[field_key]["default"])
                self.assertTrue(fields[field_key]["nullable"])
                self.assertNotIn("searchChoices", fields[field_key])

    def test_config_schema_exposes_nullable_controller_stack_overrides(self) -> None:
        fields = _fields_by_key(config_schema("linears/linear"))

        self.assertEqual(
            fields["submodule_stack_hidden_dim"]["section"],
            "Layer Stack Submodule Options",
        )
        self.assertEqual(fields["submodule_stack_hidden_dim"]["type"], "int")
        self.assertFalse(fields["submodule_stack_hidden_dim"]["nullable"])

        self.assertEqual(fields["memory_stack_independent_flag"]["type"], "bool")
        self.assertFalse(fields["memory_stack_independent_flag"]["default"])
        self.assertEqual(fields["memory_stack_hidden_dim"]["type"], "int")
        self.assertIsNone(fields["memory_stack_hidden_dim"]["default"])
        self.assertTrue(fields["memory_stack_hidden_dim"]["nullable"])
        self.assertEqual(fields["memory_stack_hidden_dim"]["choices"], [])
        self.assertNotIn("memory_hidden_dim", fields)
        self.assertNotIn("memory_layer_norm_position", fields)
        self.assertNotIn("memory_bias_flag", fields)
        self.assertEqual(fields["memory_stack_dropout_probability"]["type"], "float")
        self.assertIsNone(fields["memory_stack_dropout_probability"]["default"])
        self.assertTrue(fields["memory_stack_dropout_probability"]["nullable"])
        self.assertEqual(fields["memory_stack_activation"]["type"], "enum")
        self.assertIsNone(fields["memory_stack_activation"]["default"])
        self.assertTrue(fields["memory_stack_activation"]["nullable"])
        self.assertIn("GELU", fields["memory_stack_activation"]["choices"])
        self.assertEqual(
            fields["memory_stack_apply_output_pipeline_flag"]["type"],
            "bool",
        )
        self.assertIsNone(fields["memory_stack_apply_output_pipeline_flag"]["default"])
        self.assertTrue(fields["memory_stack_apply_output_pipeline_flag"]["nullable"])
        self.assertEqual(
            fields["memory_stack_apply_output_pipeline_flag"]["choices"],
            [True, False],
        )
        self.assertEqual(fields["gate_stack_activation"]["type"], "enum")
        self.assertEqual(fields["gate_stack_activation"]["default"], "TANH")
        self.assertTrue(fields["gate_stack_activation"]["nullable"])
        self.assertIn("TANH", fields["gate_stack_activation"]["choices"])
        self.assertEqual(fields["halting_stack_independent_flag"]["type"], "bool")
        self.assertFalse(fields["halting_stack_independent_flag"]["default"])
        self.assertNotIn("halting_hidden_dim", fields)
        self.assertNotIn("halting_layer_norm_position", fields)
        self.assertNotIn("halting_bias_flag", fields)
        self.assertEqual(
            fields["recurrent_halting_stack_independent_flag"]["type"],
            "bool",
        )
        self.assertFalse(
            fields["recurrent_halting_stack_independent_flag"]["default"],
        )
        self.assertNotIn("recurrent_halting_hidden_dim", fields)
        self.assertNotIn("recurrent_halting_layer_norm_position", fields)
        self.assertNotIn("recurrent_halting_bias_flag", fields)
        self.assertEqual(
            fields["halting_stack_last_layer_bias_option"]["default"],
            "DISABLED",
        )
        self.assertTrue(fields["halting_stack_last_layer_bias_option"]["nullable"])
        self.assertIn(
            "DEFAULT",
            fields["halting_stack_last_layer_bias_option"]["choices"],
        )

    def test_config_schema_stack_equivalent_fields_match_canonical_metadata(
        self,
    ) -> None:
        stack_equivalent_suffixes = {
            "hidden_dim",
            "num_layers",
            "activation",
            "layer_norm_position",
            "apply_output_pipeline_flag",
            "bias_flag",
        }

        for model_name in ("linears/linear", "linears/linear_adaptive"):
            fields = _fields_by_key(config_schema(model_name))
            for field_key, field in fields.items():
                if "_stack_" not in field_key or field_key.startswith("stack_"):
                    continue
                suffix = field_key.split("_stack_", 1)[1]
                if suffix not in stack_equivalent_suffixes:
                    continue
                canonical_key = f"stack_{suffix}"
                if canonical_key not in fields:
                    continue

                with self.subTest(
                    model_name=model_name,
                    field_key=field_key,
                    canonical_key=canonical_key,
                ):
                    canonical = fields[canonical_key]
                    self.assertEqual(field["type"], canonical["type"])
                    self.assertEqual(field["choices"], canonical["choices"])
                    self.assertEqual(field["key"].lower(), field_key)
                    self.assertNotEqual(field["section"], canonical["section"])

        linear_fields = _fields_by_key(config_schema("linears/linear"))
        self.assertEqual(linear_fields["stack_hidden_dim"]["default"], 256)
        self.assertFalse(linear_fields["stack_hidden_dim"]["nullable"])
        self.assertIsNone(linear_fields["gate_stack_hidden_dim"]["default"])
        self.assertTrue(linear_fields["gate_stack_hidden_dim"]["nullable"])
        self.assertTrue(
            linear_fields["stack_apply_output_pipeline_flag"]["default"]
        )
        self.assertFalse(
            linear_fields["stack_apply_output_pipeline_flag"]["nullable"]
        )
        self.assertIsNone(
            linear_fields["memory_stack_apply_output_pipeline_flag"]["default"]
        )
        self.assertTrue(
            linear_fields["memory_stack_apply_output_pipeline_flag"]["nullable"]
        )

    def test_parse_config_value_supports_nullable_memory_primitives(self) -> None:
        self.assertEqual(
            parse_config_value(
                linear_config,
                "MEMORY_TEST_TIME_TRAINING_LEARNING_RATE",
                "0.02",
            ),
            0.02,
        )
        self.assertEqual(
            parse_config_value(
                linear_config,
                "MEMORY_TEST_TIME_TRAINING_NUM_INNER_STEPS",
                "2",
            ),
            2,
        )
        self.assertIsNone(
            parse_config_value(
                linear_config,
                "MEMORY_TEST_TIME_TRAINING_LEARNING_RATE",
                "None",
            )
        )
        self.assertIs(
            parse_config_value(
                linear_config,
                "MEMORY_OPTION",
                "WeightedDynamicMemoryConfig",
            ),
            WeightedDynamicMemoryConfig,
        )

    def test_parse_config_value_supports_none_for_nullable_overrides(self) -> None:
        self.assertIsNone(
            parse_config_value(linear_config, "MEMORY_STACK_HIDDEN_DIM", "None")
        )
        self.assertIsNone(
            parse_config_value(
                linear_config,
                "MEMORY_STACK_DROPOUT_PROBABILITY",
                "None",
            )
        )
        self.assertIsNone(
            parse_config_value(linear_config, "MEMORY_STACK_ACTIVATION", "None")
        )
        self.assertIsNone(
            parse_config_value(
                linear_config,
                "MEMORY_STACK_APPLY_OUTPUT_PIPELINE_FLAG",
                "None",
            )
        )
        self.assertIs(
            parse_config_value(linear_config, "GATE_OPTION", "MULTIPLIER"),
            LayerGateOptions.MULTIPLIER,
        )
        self.assertIs(
            parse_config_value(linear_config, "RECURRENT_GATE_OPTION", "MULTIPLIER"),
            LayerGateOptions.MULTIPLIER,
        )
        self.assertIs(
            parse_config_value(linear_config, "GATE_OPTION", "ADDITION"),
            LayerGateOptions.ADDITION,
        )
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_config_value(linear_config, "GATE_OPTION", "INTERPOLATION")
        self.assertIs(
            parse_config_value(linear_config, "GATE_ACTIVATION", "TANH"),
            ActivationOptions.TANH,
        )
        self.assertIsNone(parse_config_value(linear_config, "GATE_OPTION", "None"))

    def test_config_schema_marks_preset_owned_fields_locked(self) -> None:
        baseline_fields = _fields_by_key(config_schema("linears/linear", "baseline"))
        gating_fields = _fields_by_key(config_schema("linears/linear", "gating"))

        self.assertFalse(baseline_fields["gate_flag"]["locked"])
        self.assertTrue(gating_fields["gate_flag"]["locked"])
        self.assertEqual(gating_fields["gate_flag"]["lockedValue"], True)
        self.assertIn("GATING preset", gating_fields["gate_flag"]["lockedReason"])

    def test_config_schema_locks_adaptive_component_flags_for_presets(self) -> None:
        fields = _fields_by_key(
            config_schema("linears/linear_adaptive", "full-stack")
        )

        expected_locked_fields = {
            "weight_option_flag": True,
            "weight_option": "DualModelDynamicWeightConfig",
            "bias_option_flag": True,
            "bias_option": "AdditiveDynamicBiasConfig",
            "diagonal_option_flag": True,
            "diagonal_option": "CombinedDynamicDiagonalConfig",
            "mask_option_flag": True,
            "row_mask_option": "WeightInformedScoreAxisMaskConfig",
        }
        for field_key, locked_value in expected_locked_fields.items():
            with self.subTest(field_key=field_key):
                self.assertTrue(fields[field_key]["locked"])
                self.assertEqual(fields[field_key]["lockedValue"], locked_value)
                self.assertIn("FULL_STACK preset", fields[field_key]["lockedReason"])

    def test_search_space_schema_exposes_linear_axes(self) -> None:
        axes = _axes_by_key(search_space_schema("linears/linear", "baseline"))

        self.assertIn("learning_rate", axes)
        self.assertIn("stack_hidden_dim", axes)
        self.assertIn("stack_activation", axes)
        self.assertEqual(axes["stack_hidden_dim"]["section"], "Layer Stack Options")
        self.assertEqual(axes["stack_hidden_dim"]["type"], "int")
        self.assertEqual(
            axes["stack_hidden_dim"]["values"],
            [16, 32, 64, 128, 256, 512],
        )
        self.assertEqual(axes["stack_activation"]["type"], "enum")
        self.assertIn("GELU", axes["stack_activation"]["values"])
        self.assertFalse(axes["stack_hidden_dim"]["locked"])

    def test_search_space_schema_serializes_axis_values(self) -> None:
        linear_axes = _axes_by_key(search_space_schema("linears/linear", "baseline"))
        adaptive_axes = _axes_by_key(
            search_space_schema("linears/linear_adaptive", "baseline")
        )

        self.assertEqual(
            linear_axes["stack_hidden_dim"]["values"],
            [16, 32, 64, 128, 256, 512],
        )
        self.assertEqual(
            linear_axes["stack_activation"]["values"],
            ["RELU", "LEAKY_RELU", "ELU", "GELU", "TANH"],
        )
        self.assertFalse(
            any(key.endswith("_layer_adaptive_flag") for key in adaptive_axes)
        )
        self.assertEqual(
            adaptive_axes["adaptive_generator_stack_num_layers"]["values"],
            [1, 2, 3],
        )

    def test_search_space_schema_marks_preset_owned_axes_locked(self) -> None:
        axes = _axes_by_key(search_space_schema("linears/linear", "post-norm"))

        self.assertTrue(axes["stack_layer_norm_position"]["locked"])
        self.assertEqual(axes["stack_layer_norm_position"]["lockedValue"], "AFTER")
        self.assertEqual(
            axes["stack_layer_norm_position"]["lockedByPresets"],
            ["POST_NORM"],
        )
        self.assertEqual(len(axes["stack_layer_norm_position"]["lockReasons"]), 1)
        self.assertIn(
            "POST_NORM preset",
            axes["stack_layer_norm_position"]["lockedReason"],
        )

    def test_search_space_schema_locks_union_of_selected_presets(self) -> None:
        axes = _axes_by_key(
            search_space_schema(
                "linears/linear_adaptive",
                "baseline",
                ["single-model-weight", "additive-bias"],
            )
        )

        self.assertTrue(axes["weight_option"]["locked"])
        self.assertEqual(
            axes["weight_option"]["lockedByPresets"],
            ["SINGLE_MODEL_WEIGHT"],
        )
        self.assertEqual(
            axes["weight_option"]["lockedValue"],
            "SingleModelDynamicWeightConfig",
        )
        self.assertTrue(axes["bias_option"]["locked"])
        self.assertEqual(
            axes["bias_option"]["lockedByPresets"],
            ["ADDITIVE_BIAS"],
        )
        self.assertEqual(
            axes["bias_option"]["lockedValue"],
            "AdditiveDynamicBiasConfig",
        )
        self.assertFalse(axes["stack_hidden_dim"]["locked"])
        self.assertNotIn("weight_option_flag", axes)
        self.assertNotIn("bias_option_flag", axes)


if __name__ == "__main__":
    unittest.main()
