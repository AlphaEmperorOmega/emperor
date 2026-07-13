import argparse
import contextlib
import io
import unittest
from dataclasses import fields

import models.bert.linear.config as bert_config
import models.experts.linear.config as expert_linear_config
import models.gpt.linear.config as gpt_config
import models.linears.linear.config as linears_linear_config
import models.linears.linear_adaptive.config as linear_adaptive_config
import models.vit.linear.config as vit_config
from emperor.augmentations.adaptive_parameters.core.monitor import (
    AdaptiveParameterMonitorCallback,
)
from emperor.augmentations.adaptive_parameters.core.weight import (
    LowRankDynamicWeightConfig,
)
from emperor.base.layer.gate import LayerGateOptions
from emperor.base.options import ActivationOptions
from emperor.datasets.image.classification.mnist import Mnist
from emperor.halting.core.monitor import HaltingMonitorCallback
from emperor.linears.core.monitor import LinearMonitorCallback
from lightning.pytorch.callbacks import EarlyStopping
from models.config_overrides import iter_supported_config_keys, print_config_options
from models.experts.linear import (
    ExperimentPreset as ExpertLinearExperimentPreset,
)
from models.linears.linear import ExperimentPreset as LinearExperimentPreset
from models.linears.linear.presets import (
    ExperimentPresets as LinearExperimentPresets,
)
from models.linears.linear_adaptive import ExperimentPreset
from models.linears.linear_adaptive.presets import Experiment, ExperimentPresets
from models.parametric.parametric_vector import (
    ExperimentPreset as ParametricVectorExperimentPreset,
)
from models.parser import (
    ExperimentMode,
    get_experiment_parser,
    resolve_experiment_mode,
    resolve_monitor_callbacks,
)

from model_runtime.packages import GridSearch


class ExperimentConfigOverrideTestCase:
    def make_parser(self):
        return get_experiment_parser(
            ExperimentPreset.names(),
            "models.linears.linear_adaptive",
        )

    def resolve_args(self, args):
        return resolve_experiment_mode(args, ExperimentPreset)


class TestExperimentConfigOverrideParsing(
    ExperimentConfigOverrideTestCase,
    unittest.TestCase,
):
    def test_resolve_preset_returns_primary_without_selected_presets(self):
        args = self.make_parser().parse_args(["--preset", "single-model-weight"])

        self.assertEqual(args.preset, "single-model-weight")
        self.assertFalse(hasattr(args, "opt" + "ion"))

        mode = self.resolve_args(args)

        self.assertIsInstance(mode, ExperimentMode)
        self.assertIs(mode.preset, ExperimentPreset.SINGLE_MODEL_WEIGHT)
        self.assertIsNone(mode.selected_presets)
        self.assertIsNone(mode.search_mode)

    def test_resolve_presets_returns_primary_and_ordered_selected_presets(self):
        args = self.make_parser().parse_args(
            ["--presets", "single-model-weight", "dual-model-weight"]
        )

        self.assertEqual(args.presets, ["single-model-weight", "dual-model-weight"])
        self.assertFalse(hasattr(args, "options"))

        mode = self.resolve_args(args)

        self.assertIs(mode.preset, ExperimentPreset.SINGLE_MODEL_WEIGHT)
        self.assertEqual(
            mode.selected_presets,
            [
                ExperimentPreset.SINGLE_MODEL_WEIGHT,
                ExperimentPreset.DUAL_MODEL_WEIGHT,
            ],
        )
        self.assertIsNone(mode.search_mode)

    def test_resolve_all_presets_preserves_all_presets_mode(self):
        args = self.make_parser().parse_args(["--all-presets"])

        self.assertTrue(args.all_presets)
        self.assertFalse(hasattr(args, "all_" + "options"))

        mode = self.resolve_args(args)

        self.assertIsNone(mode.preset)
        self.assertIsNone(mode.selected_presets)
        self.assertIsNone(mode.search_mode)

    def test_experiment_mode_fields_document_help_metadata(self):
        expected_names = [
            "experiment_task",
            "preset",
            "selected_presets",
            "search_mode",
            "search_keys",
            "config_overrides",
            "search_overrides",
            "monitor_names",
            "monitor_callbacks",
        ]

        mode_fields = fields(ExperimentMode)

        self.assertEqual(
            [mode_field.name for mode_field in mode_fields],
            expected_names,
        )
        for mode_field in mode_fields:
            with self.subTest(field=mode_field.name):
                self.assertIsInstance(mode_field.metadata["help"], str)
                self.assertTrue(mode_field.metadata["help"])

    def test_experiment_mode_does_not_support_tuple_unpacking(self):
        args = self.make_parser().parse_args(["--preset", "single-model-weight"])
        mode = self.resolve_args(args)

        with self.assertRaises(TypeError):
            (
                preset,
                selected_presets,
                search_mode,
                search_keys,
                config_overrides,
                search_overrides,
                monitor_names,
                monitor_callbacks,
            ) = mode
            self.fail(
                (
                    preset,
                    selected_presets,
                    search_mode,
                    search_keys,
                    config_overrides,
                    search_overrides,
                    monitor_names,
                    monitor_callbacks,
                )
            )

    def test_named_config_flags_parse_supported_value_types(self):
        args = self.make_parser().parse_args(
            [
                "--preset",
                "single-model-weight",
                "--num-epochs",
                "30",
                "--callback-early-stopping-patience",
                "0",
                "--learning-rate",
                "1e-4",
                "--hidden-dim",
                "128",
                "--stack-activation",
                "GELU",
                "--gate-option",
                "ADDITION",
                "--recurrent-gate-option",
                "MULTIPLIER",
                "--weight-option",
                "LowRankDynamicWeightConfig",
                "--input-layer-weight-option",
                "LowRankDynamicWeightConfig",
            ]
        )

        mode = self.resolve_args(args)

        self.assertEqual(mode.config_overrides["num_epochs"], 30)
        self.assertEqual(mode.config_overrides["callback_early_stopping_patience"], 0)
        self.assertEqual(mode.config_overrides["learning_rate"], 1e-4)
        self.assertEqual(mode.config_overrides["hidden_dim"], 128)
        self.assertIs(mode.config_overrides["stack_activation"], ActivationOptions.GELU)
        self.assertIs(mode.config_overrides["gate_option"], LayerGateOptions.ADDITION)
        self.assertIs(
            mode.config_overrides["recurrent_gate_option"],
            LayerGateOptions.MULTIPLIER,
        )
        self.assertIs(
            mode.config_overrides["weight_option"],
            LowRankDynamicWeightConfig,
        )
        self.assertIs(
            mode.config_overrides["input_layer_weight_option"],
            LowRankDynamicWeightConfig,
        )
        self.assertEqual(mode.search_overrides, {})

    def test_hidden_dim_flag_sets_hidden_dim_override(self):
        args = self.make_parser().parse_args(
            [
                "--preset",
                "single-model-weight",
                "--hidden-dim",
                "128",
            ]
        )

        mode = self.resolve_args(args)

        self.assertEqual(mode.config_overrides["hidden_dim"], 128)

    def test_legacy_stack_hidden_dim_flag_is_rejected(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                self.make_parser().parse_args(
                    [
                        "--preset",
                        "single-model-weight",
                        "--stack-hidden-dim",
                        "128",
                    ]
                )

    def test_boundary_adaptive_generator_stack_flags_are_rejected(self):
        removed_flags = [
            "--input-layer-adaptive-generator-stack-hidden-dim",
            "--output-layer-adaptive-generator-stack-hidden-dim",
        ]

        for removed_flag in removed_flags:
            with self.subTest(flag=removed_flag):
                with contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit):
                        self.make_parser().parse_args(
                            [
                                "--preset",
                                "single-model-weight",
                                removed_flag,
                                "128",
                            ]
                        )

    def test_linear_adaptive_config_hides_boundary_generator_stack_keys(self):
        supported_keys = set(iter_supported_config_keys(linear_adaptive_config))
        boundary_stack_prefixes = tuple(
            f"{boundary}_LAYER_ADAPTIVE_GENERATOR_STACK"
            for boundary in ("INPUT", "OUTPUT")
        )

        self.assertFalse(
            any(key.startswith(boundary_stack_prefixes) for key in supported_keys)
        )
        self.assertIn("ADAPTIVE_GENERATOR_STACK_HIDDEN_DIM", supported_keys)
        self.assertIn("ADAPTIVE_GENERATOR_STACK_BIAS_FLAG", supported_keys)
        self.assertFalse(
            any(key.endswith("_LAYER_ADAPTIVE_FLAG") for key in supported_keys)
        )

    def test_global_adaptive_generator_stack_flags_parse_to_builder_params(self):
        args = self.make_parser().parse_args(
            [
                "--preset",
                "single-model-weight",
                "--adaptive-generator-stack-hidden-dim",
                "21",
                "--adaptive-generator-stack-num-layers",
                "3",
                "--adaptive-generator-stack-bias-flag",
                "false",
            ]
        )

        mode = self.resolve_args(args)

        self.assertEqual(
            mode.config_overrides["adaptive_generator_stack_hidden_dim"],
            21,
        )
        self.assertEqual(
            mode.config_overrides["adaptive_generator_stack_num_layers"],
            3,
        )
        self.assertIs(
            mode.config_overrides["adaptive_generator_stack_bias_flag"],
            False,
        )

    def test_removed_adaptive_generator_alias_flags_are_rejected(self):
        old_submodule_stack_flag = "adaptive-" + "submodule-stack"
        old_stack_flag = "adaptive-" + "stack"
        removed_flags = [
            f"--{old_submodule_stack_flag}-hidden-dim",
            f"--{old_stack_flag}-hidden-dim",
        ]

        for removed_flag in removed_flags:
            with self.subTest(flag=removed_flag):
                with contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit):
                        self.make_parser().parse_args(
                            [
                                "--preset",
                                "single-model-weight",
                                removed_flag,
                                "21",
                            ]
                        )

    def test_named_config_flags_parse_bool_and_string_values(self):
        args = self.make_parser().parse_args(
            [
                "--preset",
                "single-model-weight",
                "--stack-bias-flag",
                "false",
                "--trainer-accelerator",
                "mps",
                "--data-num-workers",
                "0",
                "--run-test-after-fit",
                "false",
            ]
        )

        mode = self.resolve_args(args)

        self.assertIs(mode.config_overrides["stack_bias_flag"], False)
        self.assertEqual(mode.config_overrides["trainer_accelerator"], "mps")
        self.assertEqual(mode.config_overrides["data_num_workers"], 0)
        self.assertIs(mode.config_overrides["run_test_after_fit"], False)
        self.assertEqual(mode.search_overrides, {})

    def test_config_marker_only_groups_overrides(self):
        plain_args = self.make_parser().parse_args(
            [
                "--preset",
                "single-model-weight",
                "--num-epochs",
                "30",
                "--stack-bias-flag",
                "false",
            ]
        )
        grouped_args = self.make_parser().parse_args(
            [
                "--preset",
                "single-model-weight",
                "--config",
                "--num-epochs",
                "30",
                "--stack-bias-flag",
                "false",
            ]
        )

        plain_mode = self.resolve_args(plain_args)
        grouped_mode = self.resolve_args(grouped_args)

        self.assertEqual(grouped_mode.config_overrides, plain_mode.config_overrides)
        self.assertEqual(grouped_mode.search_overrides, plain_mode.search_overrides)

    def test_top_level_bias_flag_alias_is_rejected(self):
        removed_flag = "--" + "bias-flag"

        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                self.make_parser().parse_args(
                    [
                        "--preset",
                        "single-model-weight",
                        removed_flag,
                        "false",
                    ]
                )

    def test_controller_stack_flags_parse_to_builder_params(self):
        parser = get_experiment_parser(
            ExpertLinearExperimentPreset.names(),
            "models.experts.linear",
        )
        args = parser.parse_args(
            [
                "--preset",
                "baseline",
                "--gate-stack-hidden-dim",
                "32",
                "--gate-stack-layer-norm-position",
                "AFTER",
                "--gate-stack-bias-flag",
                "false",
                "--halting-stack-hidden-dim",
                "48",
                "--halting-stack-layer-norm-position",
                "BEFORE",
                "--halting-stack-bias-flag",
                "false",
            ]
        )

        mode = resolve_experiment_mode(args, ExpertLinearExperimentPreset)

        self.assertEqual(mode.config_overrides["gate_stack_hidden_dim"], 32)
        self.assertIs(
            mode.config_overrides["gate_stack_layer_norm_position"],
            expert_linear_config.LayerNormPositionOptions.AFTER,
        )
        self.assertFalse(mode.config_overrides["gate_stack_bias_flag"])
        self.assertEqual(mode.config_overrides["halting_stack_hidden_dim"], 48)
        self.assertIs(
            mode.config_overrides["halting_stack_layer_norm_position"],
            expert_linear_config.LayerNormPositionOptions.BEFORE,
        )
        self.assertFalse(mode.config_overrides["halting_stack_bias_flag"])

    def test_router_stack_flags_parse_without_independent_flag(self):
        parser = get_experiment_parser(
            ExpertLinearExperimentPreset.names(),
            "models.experts.linear",
        )
        args = parser.parse_args(
            [
                "--preset",
                "baseline",
                "--router-stack-hidden-dim",
                "40",
                "--router-stack-num-layers",
                "3",
                "--router-bias-flag",
                "false",
            ]
        )

        mode = resolve_experiment_mode(args, ExpertLinearExperimentPreset)

        self.assertEqual(mode.config_overrides["router_stack_hidden_dim"], 40)
        self.assertEqual(mode.config_overrides["router_stack_num_layers"], 3)
        self.assertFalse(mode.config_overrides["router_bias_flag"])

    def test_router_stack_independent_flag_is_rejected(self):
        parser = get_experiment_parser(
            ExpertLinearExperimentPreset.names(),
            "models.experts.linear",
        )

        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parser.parse_args(
                    [
                        "--preset",
                        "baseline",
                        "--router-stack-independent-flag",
                        "true",
                    ]
                )

    def test_legacy_controller_stack_flags_are_rejected(self):
        parser = get_experiment_parser(
            ExpertLinearExperimentPreset.names(),
            "models.experts.linear",
        )
        removed_flags = [
            "--gate-" + "hidden-dim",
            "--gate-" + "layer-norm-position",
            "--gate-" + "bias-flag",
            "--halting-" + "hidden-dim",
            "--halting-" + "layer-norm-position",
            "--halting-" + "bias-flag",
        ]

        for removed_flag in removed_flags:
            with self.subTest(flag=removed_flag):
                with contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit):
                        parser.parse_args(
                            [
                                "--preset",
                                "baseline",
                                removed_flag,
                                "32",
                            ]
                        )

    def test_monitor_resolver_builds_callbacks_for_valid_names(self):
        callbacks = resolve_monitor_callbacks(
            linears_linear_config,
            ["linear", "halting"],
        )

        self.assertEqual(
            [type(callback) for callback in callbacks],
            [
                LinearMonitorCallback,
                HaltingMonitorCallback,
            ],
        )

    def test_monitor_resolver_deduplicates_repeated_names(self):
        callbacks = resolve_monitor_callbacks(
            linears_linear_config,
            ["linear", "linear", "halting", "linear"],
        )

        self.assertEqual(
            [type(callback) for callback in callbacks],
            [
                LinearMonitorCallback,
                HaltingMonitorCallback,
            ],
        )

    def test_monitor_resolver_rejects_unknown_names_with_valid_monitors(self):
        with self.assertRaisesRegex(
            ValueError,
            (
                r"Unknown --monitors: \['does-not-exist'\]\. "
                r"Valid monitors: halting, layer-controller, linear, memory, "
                r"recurrent-layer"
            ),
        ):
            resolve_monitor_callbacks(
                linears_linear_config,
                ["linear", "does-not-exist"],
            )

    def test_cli_monitor_names_are_deduplicated_on_experiment_mode(self):
        parser = get_experiment_parser(
            LinearExperimentPreset.names(),
            "models.linears.linear",
        )
        args = parser.parse_args(
            [
                "--preset",
                "baseline",
                "--monitors",
                "linear",
                "halting",
                "linear",
            ]
        )

        mode = resolve_experiment_mode(args, LinearExperimentPreset)

        self.assertEqual(mode.monitor_names, ["linear", "halting"])
        self.assertEqual(
            [type(callback) for callback in mode.monitor_callbacks],
            [
                LinearMonitorCallback,
                HaltingMonitorCallback,
            ],
        )

    def test_runs_cli_can_defer_monitor_callback_construction(self):
        parser = get_experiment_parser(
            LinearExperimentPreset.names(),
            "models.linears.linear",
        )
        args = parser.parse_args(
            ["--preset", "baseline", "--monitors", "linear", "halting"]
        )

        mode = resolve_experiment_mode(
            args,
            LinearExperimentPreset,
            build_monitor_callbacks=False,
        )

        self.assertEqual(mode.monitor_names, ["linear", "halting"])
        self.assertEqual(mode.monitor_callbacks, [])

    def test_list_config_prints_shared_runtime_options(self):
        output = io.StringIO()

        with contextlib.redirect_stdout(output):
            print_config_options("linears/linear")

        rendered = output.getvalue()
        self.assertIn("--data-num-workers", rendered)
        self.assertIn("--run-test-after-fit", rendered)
        self.assertIn("--trainer-profiler", rendered)

    def test_search_set_parses_lists_using_config_value_types(self):
        args = self.make_parser().parse_args(
            [
                "--preset",
                "single-model-weight",
                "--grid-search",
                "--search-set",
                "hidden-dim=64,128",
                "--search-set",
                "stack_activation=RELU,GELU",
                "--search-set",
                "gate-option=MULTIPLIER,ADDITION",
                "--search-set",
                "recurrent_gate_option=MULTIPLIER,ADDITION",
                "--search-set",
                "weight_option=None,LowRankDynamicWeightConfig",
                "--search-set",
                "input_layer_weight_option=None,LowRankDynamicWeightConfig",
            ]
        )

        mode = self.resolve_args(args)

        self.assertIsInstance(mode.search_mode, GridSearch)
        self.assertEqual(mode.search_overrides["hidden_dim"], [64, 128])
        self.assertEqual(
            mode.search_overrides["stack_activation"],
            [ActivationOptions.RELU, ActivationOptions.GELU],
        )
        self.assertEqual(
            mode.search_overrides["gate_option"],
            [
                LayerGateOptions.MULTIPLIER,
                LayerGateOptions.ADDITION,
            ],
        )
        self.assertEqual(
            mode.search_overrides["recurrent_gate_option"],
            [
                LayerGateOptions.MULTIPLIER,
                LayerGateOptions.ADDITION,
            ],
        )
        self.assertEqual(
            mode.search_overrides["weight_option"],
            [None, LowRankDynamicWeightConfig],
        )
        self.assertEqual(
            mode.search_overrides["input_layer_weight_option"],
            [None, LowRankDynamicWeightConfig],
        )

    def test_search_set_requires_search_mode(self):
        args = self.make_parser().parse_args(
            [
                "--preset",
                "single-model-weight",
                "--search-set",
                "hidden_dim=64,128",
            ]
        )

        with self.assertRaises(ValueError):
            self.resolve_args(args)

    def test_explicit_no_search_presets_blocks_preset_search(self):
        parser = get_experiment_parser(
            ParametricVectorExperimentPreset.names(),
            "models.parametric.parametric_vector",
        )
        args = parser.parse_args(["--preset", "preset", "--grid-search"])

        with self.assertRaises(ValueError):
            resolve_experiment_mode(
                args,
                ParametricVectorExperimentPreset,
                no_search_presets=["PRESET"],
            )

    def test_preset_search_is_allowed_by_default(self):
        parser = get_experiment_parser(
            ParametricVectorExperimentPreset.names(),
            "models.parametric.parametric_vector",
        )
        args = parser.parse_args(["--preset", "preset", "--grid-search"])

        mode = resolve_experiment_mode(
            args,
            ParametricVectorExperimentPreset,
        )

        self.assertIs(mode.preset, ParametricVectorExperimentPreset.PRESET)
        self.assertIsInstance(mode.search_mode, GridSearch)

    def test_malformed_search_set_raises_argument_error(self):
        args = self.make_parser().parse_args(
            [
                "--preset",
                "single-model-weight",
                "--grid-search",
                "--search-set",
                "hidden_dim",
            ]
        )

        with self.assertRaises(argparse.ArgumentTypeError):
            self.resolve_args(args)

    def test_unknown_search_set_key_raises_argument_error(self):
        args = self.make_parser().parse_args(
            [
                "--preset",
                "single-model-weight",
                "--grid-search",
                "--search-set",
                "does_not_exist=1,2",
            ]
        )

        with self.assertRaises(argparse.ArgumentTypeError):
            self.resolve_args(args)

    def test_legacy_search_set_keys_raise_argument_error(self):
        cases = [
            (
                self.make_parser(),
                ExperimentPreset,
                "single-model-weight",
                "adaptive_" + "submodule_stack_hidden_dim=21,22",
            ),
            (
                get_experiment_parser(
                    ExpertLinearExperimentPreset.names(),
                    "models.experts.linear",
                ),
                ExpertLinearExperimentPreset,
                "baseline",
                "gate_" + "hidden_dim=32,64",
            ),
        ]

        for parser, preset_enum, preset, search_set in cases:
            with self.subTest(search_set=search_set):
                args = parser.parse_args(
                    [
                        "--preset",
                        preset,
                        "--grid-search",
                        "--search-set",
                        search_set,
                    ]
                )
                with self.assertRaises(argparse.ArgumentTypeError):
                    resolve_experiment_mode(args, preset_enum)

    def test_fixed_override_and_search_axis_conflict_raises(self):
        args = self.make_parser().parse_args(
            [
                "--preset",
                "single-model-weight",
                "--hidden-dim",
                "128",
                "--grid-search",
                "--search-set",
                "hidden_dim=64,128",
            ]
        )

        with self.assertRaises(ValueError):
            self.resolve_args(args)

    def test_transformer_config_override_keys_match_supported_builder_surface(self):
        bert_keys = set(iter_supported_config_keys(bert_config))
        gpt_keys = set(iter_supported_config_keys(gpt_config))
        vit_keys = set(iter_supported_config_keys(vit_config))
        removed_bias_constant = "BIAS" + "_FLAG"

        self.assertIn("GATE_FLAG", bert_keys)
        self.assertNotIn("GATE_HIDDEN_DIM", bert_keys)
        self.assertIn("GATE_STACK_HIDDEN_DIM", bert_keys)
        self.assertIn("HALTING_FLAG", bert_keys)
        self.assertNotIn("HALTING_HIDDEN_DIM", bert_keys)
        self.assertIn("HALTING_STACK_HIDDEN_DIM", bert_keys)
        self.assertIn("MEMORY_FLAG", bert_keys)
        self.assertIn("RECURRENT_FLAG", bert_keys)
        self.assertIn("STACK_RESIDUAL_CONNECTION_OPTION", bert_keys)
        self.assertIn("STACK_LAST_LAYER_BIAS_OPTION", bert_keys)
        self.assertIn("STACK_APPLY_OUTPUT_PIPELINE_FLAG", bert_keys)
        self.assertIn("STACK_BIAS_FLAG", bert_keys)
        self.assertIn("TOKEN_TYPE_VOCAB_SIZE", bert_keys)
        self.assertIn("EMBEDDING_LAYER_NORM_FLAG", bert_keys)
        self.assertIn("EMBEDDING_DROPOUT_PROBABILITY", bert_keys)
        self.assertIn("MLM_ACTIVATION", bert_keys)
        self.assertIn("MLM_DENSE_BIAS_FLAG", bert_keys)
        self.assertIn("MLM_LAYER_NORM_FLAG", bert_keys)
        self.assertIn("MLM_DECODER_BIAS_FLAG", bert_keys)
        self.assertIn("MLM_DECODER_WEIGHT_TYING_FLAG", bert_keys)
        self.assertIn("NSP_POOLER_ACTIVATION", bert_keys)
        self.assertIn("NSP_POOLER_BIAS_FLAG", bert_keys)
        self.assertIn("NSP_OUTPUT_DIM", bert_keys)
        self.assertIn("NSP_HEAD_BIAS_FLAG", bert_keys)
        self.assertNotIn("BERT_PRETRAINING_TARGET_VOCAB_SIZE", bert_keys)
        self.assertIn("SEQUENCE_LENGTH", gpt_keys)
        self.assertIn("EMBEDDING_LAYER_NORM_FLAG", gpt_keys)
        self.assertIn("EMBEDDING_DROPOUT_PROBABILITY", gpt_keys)
        self.assertIn("LM_HEAD_BIAS_FLAG", gpt_keys)
        self.assertIn("LM_HEAD_WEIGHT_TYING_FLAG", gpt_keys)
        self.assertNotIn("TOKEN_TYPE_VOCAB_SIZE", gpt_keys)
        self.assertFalse(any(key.startswith("MLM_") for key in gpt_keys))
        self.assertFalse(any(key.startswith("NSP_") for key in gpt_keys))
        self.assertFalse(any("CAUSAL" in key for key in gpt_keys))
        self.assertNotIn(removed_bias_constant, bert_keys)
        self.assertNotIn("SEQUENCE_LENGTH", vit_keys)
        self.assertNotIn(removed_bias_constant, vit_keys)
        self.assertNotIn("TRANSFORMER_NUM_LAYERS", vit_keys)
        self.assertNotIn("ACTIVATION_FUNCTION", vit_keys)
        self.assertNotIn("DROPOUT_PROBABILITY", vit_keys)
        self.assertIn("HIDDEN_DIM", vit_keys)
        self.assertIn("IMAGE_PATCH_SIZE", vit_keys)
        self.assertIn("ATTN_NUM_HEADS", vit_keys)
        self.assertIn("LAYER_NORM_POSITION", vit_keys)
        self.assertIn("STACK_NUM_LAYERS", vit_keys)
        self.assertIn("STACK_ACTIVATION", vit_keys)
        self.assertIn("STACK_DROPOUT_PROBABILITY", vit_keys)


class TestExperimentConfigOverrideApplication(unittest.TestCase):
    def test_preset_lock_reasons_use_exact_backticked_field_names(self):
        locks = LinearExperimentPresets().locks_for_preset(
            LinearExperimentPreset.GATING
        )

        self.assertEqual(
            locks["stack_gate_flag"].reason,
            (
                "Locked by the GATING preset because this preset locks "
                "`stack_gate_flag`."
            ),
        )

        adaptive_locks = ExperimentPresets().locks_for_preset(
            ExperimentPreset.SINGLE_MODEL_WEIGHT
        )

        self.assertEqual(
            adaptive_locks["weight_option"].reason,
            (
                "Locked by the SINGLE_MODEL_WEIGHT preset because this preset "
                "locks `weight_option`."
            ),
        )

    def test_unlocked_override_wins_but_preset_locked_override_raises(self):
        cfg = ExperimentPresets().get_config(
            ExperimentPreset.SINGLE_MODEL_WEIGHT,
            Mnist,
            config_overrides={
                "hidden_dim": 64,
            },
        )[0]

        self.assertEqual(cfg.hidden_dim, 64)

        with self.assertRaisesRegex(ValueError, "SINGLE_MODEL_WEIGHT.*weight_option"):
            ExperimentPresets().get_config(
                ExperimentPreset.SINGLE_MODEL_WEIGHT,
                Mnist,
                config_overrides={
                    "weight_option": LowRankDynamicWeightConfig,
                },
            )

    def test_search_overrides_create_expected_configs(self):
        configs = ExperimentPresets().get_config(
            ExperimentPreset.SINGLE_MODEL_WEIGHT,
            Mnist,
            search_mode=GridSearch(),
            search_overrides={
                "hidden_dim": [64, 128],
                "stack_num_layers": [2, 4],
            },
        )

        self.assertEqual(len(configs), 4)
        self.assertEqual(
            sorted({cfg.hidden_dim for cfg in configs}),
            [64, 128],
        )
        self.assertEqual(
            sorted({cfg.experiment_config.model_config.num_layers for cfg in configs}),
            [2, 4],
        )

    def test_global_adaptive_generator_stack_overrides_apply(self):
        cfg = ExperimentPresets().get_config(
            ExperimentPreset.SINGLE_MODEL_WEIGHT,
            Mnist,
            config_overrides={
                "adaptive_generator_stack_hidden_dim": 21,
                "adaptive_generator_stack_num_layers": 3,
                "adaptive_generator_stack_bias_flag": False,
            },
        )[0]
        layer_model_config = (
            cfg.experiment_config.model_config.layer_config.layer_model_config
        )
        augmentation_config = layer_model_config.adaptive_augmentation_config
        stack_config = augmentation_config.model_config

        self.assertEqual(stack_config.hidden_dim, 21)
        self.assertEqual(stack_config.num_layers, 3)
        self.assertFalse(stack_config.layer_config.layer_model_config.bias_flag)

    def test_trainer_overrides_disable_early_stopping_without_static_monitors(self):
        experiment = Experiment(ExperimentPreset.SINGLE_MODEL_WEIGHT)

        trainer_config = experiment._load_trainer_config(
            {
                "callback_early_stopping_patience": 0,
                "trainer_devices": 1,
            }
        )
        callback_types = {type(callback) for callback in trainer_config["callbacks"]}

        self.assertNotIn(EarlyStopping, callback_types)
        self.assertNotIn(AdaptiveParameterMonitorCallback, callback_types)
        self.assertEqual(trainer_config["trainer_args"]["devices"], 1)

    def test_runtime_overrides_are_not_passed_to_model_builder(self):
        cfg = ExperimentPresets().get_config(
            ExperimentPreset.SINGLE_MODEL_WEIGHT,
            Mnist,
            config_overrides={
                "data_num_workers": 0,
                "run_test_after_fit": False,
            },
        )[0]

        self.assertIsNotNone(cfg.experiment_config)

    def test_runtime_config_reads_defaults_and_overrides(self):
        experiment = Experiment(ExperimentPreset.SINGLE_MODEL_WEIGHT)

        defaults = experiment._load_runtime_config({})
        overrides = experiment._load_runtime_config(
            {
                "data_num_workers": 0,
                "run_test_after_fit": False,
            }
        )

        self.assertEqual(
            defaults["data_num_workers"],
            linear_adaptive_config.DATA_NUM_WORKERS,
        )
        self.assertIs(
            defaults["run_test_after_fit"],
            linear_adaptive_config.RUN_TEST_AFTER_FIT,
        )
        self.assertEqual(overrides["data_num_workers"], 0)
        self.assertIs(overrides["run_test_after_fit"], False)


if __name__ == "__main__":
    unittest.main()
