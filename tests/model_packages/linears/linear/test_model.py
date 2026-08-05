import ast
import contextlib
import importlib
import inspect
import io
import runpy
import sys
import unittest
from dataclasses import FrozenInstanceError, dataclass, replace
from pathlib import Path
from unittest.mock import patch

import torch

import models.linears.linear.config as config
import models.linears.linear.dataset_options as dataset_options
import models.linears.linear.monitor_options as monitor_options
import models.linears.linear.search_space as search_space
from emperor.config import BaseOptions
from emperor.datasets.image.classification import Cifar10, Cifar100, FashionMNIST, Mnist
from emperor.experiments import ExperimentTask
from emperor.layers import (
    ActivationOptions,
    AdditiveResidualConfig,
    AttentionResidualConfig,
    GateConfig,
    LastLayerBiasOptions,
    LayerConfig,
    LayerGateOptions,
    LayerNormPositionOptions,
    LayerStackConfig,
    RecurrentLayerConfig,
    WeightedBlendResidualConfig,
    WeightedResidualConfig,
)
from emperor.linears import LinearLayerConfig
from emperor.memory import (
    ElementWiseWeightedDynamicMemoryConfig,
    MemoryPositionOptions,
    WeightedDynamicMemoryConfig,
)
from model_runtime.packages import (
    PresetDefinition,
    config_key_to_model_param,
    iter_supported_config_keys,
)
from models.catalog import model_package
from models.cli_selection import resolve_cli_selection
from models.experiment_cli_parser import get_experiment_parser
from models.linears.linear._projection_config_factory import ProjectionConfigFactory
from models.linears.linear._residual import ResidualStackOptions
from models.linears.linear.config_builder import LinearConfigBuilder
from models.linears.linear.model import Model
from models.linears.linear.presets import (
    _PRESET_DEFINITIONS,
    Experiment,
    ExperimentPreset,
)
from models.linears.linear.runtime_defaults import DEFAULT_RUNTIME, runtime_from_flat
from models.linears.linear.runtime_options import (
    ControllerStackOptions,
    GateOptions,
    HaltingOptions,
    MainStackOptions,
    MemoryOptions,
    RecurrenceOptions,
    RuntimeOptions,
)
from models.training_test_utils import (
    RandomImageClassificationDataModule,
    tiny_cpu_trainer,
)

_NON_MODEL_PREFIXES = (
    "TRAINER_",
    "CALLBACK_",
    "DATA_",
    "RUN_",
    "MONITOR_",
)
_NON_MODEL_KEYS = {
    "HALTING_OPTION",
    "NUM_EPOCHS",
    "RECURRENT_HALTING_OPTION",
}
_NEW_PRESET_EXPECTATIONS = {
    "WEIGHTED_RESIDUAL": (
        25,
        {"stack_residual_connection_option": WeightedResidualConfig},
        "Default config with a learned tanh-scaled current contribution composed "
        "with the previous hidden state at each hidden layer.",
    ),
    "WEIGHTED_BLEND_RESIDUAL": (
        26,
        {"stack_residual_connection_option": WeightedBlendResidualConfig},
        "Default config with a learned bounded convex blend between current and "
        "previous hidden states at each hidden layer.",
    ),
    "ATTENTION_RESIDUAL": (
        27,
        {"stack_residual_connection_option": AttentionResidualConfig},
        "Default config with depth-local attention over compatible residual sources "
        "in the hidden stack.",
    ),
    "RECURRENT_LAYER_GATING": (
        28,
        {
            "recurrent_flag": True,
            "stack_gate_flag": True,
        },
        "Recurrent config with per-layer gating inside the reused hidden stack and "
        "no outer recurrent-step gate.",
    ),
    "RECURRENT_DUAL_GATING": (
        29,
        {
            "recurrent_flag": True,
            "stack_gate_flag": True,
            "recurrent_stack_gate_flag": True,
        },
        "Recurrent config with separate inner per-layer gates and an outer "
        "recurrent-step gate.",
    ),
    "RECURRENT_LAYER_HALTING": (
        30,
        {
            "recurrent_flag": True,
            "stack_halting_flag": True,
        },
        "Recurrent config with halting inside the reused hidden stack while the outer "
        "recurrence retains a fixed maximum step budget.",
    ),
    "RECURRENT_DUAL_HALTING": (
        31,
        {
            "recurrent_flag": True,
            "stack_halting_flag": True,
            "recurrent_stack_halting_flag": True,
        },
        "Recurrent config with separate inner-stack and outer recurrent-step "
        "stick-breaking halting controllers.",
    ),
    "WEIGHTED_MEMORY": (
        32,
        {
            "memory_flag": True,
            "memory_option": WeightedDynamicMemoryConfig,
        },
        "Default config with shared stack memory using a sample- and position-level "
        "weighted merge.",
    ),
    "ELEMENT_WISE_WEIGHTED_MEMORY": (
        33,
        {
            "memory_flag": True,
            "memory_option": ElementWiseWeightedDynamicMemoryConfig,
        },
        "Default config with shared stack memory using a per-feature weighted merge.",
    ),
    "NO_NORM": (
        34,
        {"layer_norm_position": LayerNormPositionOptions.DISABLED},
        "Default config with no layer normalization in the main hidden stack.",
    ),
    "PRE_ACTIVATION_NORM": (
        35,
        {"layer_norm_position": LayerNormPositionOptions.DEFAULT},
        "Default config with normalization after affine and memory work but before "
        "activation, distinct from pre-layer and post-layer normalization.",
    ),
}
_HISTORICAL_PRESET_SNAPSHOT = (
    (
        "BASELINE",
        1,
        {},
        "Default config: a GELU hidden linear stack with pre-layer norm and dropout.",
    ),
    (
        "GATING",
        2,
        {"stack_gate_flag": True},
        "Default config with per-layer gating enabled, so each hidden layer output "
        "is modulated by a learned sigmoid gate.",
    ),
    (
        "HALTING",
        3,
        {"stack_halting_flag": True},
        "Default config with stack halting enabled, so examples can stop early as "
        "they move through the hidden stack.",
    ),
    (
        "MEMORY",
        4,
        {"memory_flag": True},
        "Default config with shared stack memory enabled across the hidden layers.",
    ),
    (
        "GATING_HALTING",
        5,
        {"stack_gate_flag": True, "stack_halting_flag": True},
        "Default config with both per-layer gating and stack halting enabled.",
    ),
    (
        "GATING_MEMORY",
        6,
        {"stack_gate_flag": True, "memory_flag": True},
        "Default config with both per-layer gating and shared stack memory enabled.",
    ),
    (
        "HALTING_MEMORY",
        7,
        {"stack_halting_flag": True, "memory_flag": True},
        "Default config with both stack halting and shared stack memory enabled.",
    ),
    (
        "GATING_HALTING_MEMORY",
        8,
        {
            "stack_gate_flag": True,
            "stack_halting_flag": True,
            "memory_flag": True,
        },
        "Default config with per-layer gating, stack halting, and shared stack "
        "memory enabled.",
    ),
    (
        "RESIDUAL",
        9,
        {"stack_residual_connection_option": AdditiveResidualConfig},
        "Default config with residual skip connections enabled between same-width "
        "hidden layers.",
    ),
    (
        "POST_NORM",
        10,
        {"layer_norm_position": LayerNormPositionOptions.AFTER},
        "Default config with layer norm applied after each layer instead of before it.",
    ),
    (
        "RESIDUAL_POST_NORM",
        11,
        {
            "stack_residual_connection_option": AdditiveResidualConfig,
            "layer_norm_position": LayerNormPositionOptions.AFTER,
        },
        "Default config with residual skip connections and post-layer normalization "
        "enabled.",
    ),
    (
        "RESIDUAL_GATING",
        12,
        {
            "stack_residual_connection_option": AdditiveResidualConfig,
            "stack_gate_flag": True,
        },
        "Default config with residual skip connections and per-layer gating enabled.",
    ),
    (
        "RESIDUAL_HALTING",
        13,
        {
            "stack_residual_connection_option": AdditiveResidualConfig,
            "stack_halting_flag": True,
        },
        "Default config with residual skip connections and stack halting enabled.",
    ),
    (
        "RESIDUAL_MEMORY",
        14,
        {
            "stack_residual_connection_option": AdditiveResidualConfig,
            "memory_flag": True,
        },
        "Default config with residual skip connections and shared stack memory enabled.",
    ),
    (
        "RECURRENT",
        15,
        {"recurrent_flag": True},
        "Default config wrapped in fixed-step recurrence, reusing the hidden stack "
        "for each recurrent step.",
    ),
    (
        "RECURRENT_GATING",
        16,
        {"recurrent_flag": True, "recurrent_stack_gate_flag": True},
        "Default recurrent config with step-level gating enabled after each recurrent "
        "update.",
    ),
    (
        "RECURRENT_HALTING",
        17,
        {"recurrent_flag": True, "recurrent_stack_halting_flag": True},
        "Default recurrent config with recurrent halting enabled, allowing early "
        "stopping before the max step count.",
    ),
    (
        "RECURRENT_MEMORY",
        18,
        {"recurrent_flag": True, "memory_flag": True},
        "Default recurrent config whose reused hidden stack has shared memory enabled.",
    ),
    (
        "RECURRENT_GATING_HALTING",
        19,
        {
            "recurrent_flag": True,
            "recurrent_stack_gate_flag": True,
            "recurrent_stack_halting_flag": True,
        },
        "Default recurrent config with both step-level gating and recurrent halting "
        "enabled.",
    ),
    (
        "RECURRENT_GATING_MEMORY",
        20,
        {
            "recurrent_flag": True,
            "recurrent_stack_gate_flag": True,
            "memory_flag": True,
        },
        "Default recurrent config with step-level gating and shared memory in the "
        "reused hidden stack.",
    ),
    (
        "RECURRENT_HALTING_MEMORY",
        21,
        {
            "recurrent_flag": True,
            "recurrent_stack_halting_flag": True,
            "memory_flag": True,
        },
        "Default recurrent config with recurrent halting and shared memory in the "
        "reused hidden stack.",
    ),
    (
        "RECURRENT_GATING_HALTING_MEMORY",
        22,
        {
            "recurrent_flag": True,
            "recurrent_stack_gate_flag": True,
            "recurrent_stack_halting_flag": True,
            "memory_flag": True,
        },
        "Default recurrent config with step-level gating, recurrent halting, and "
        "shared memory in the reused hidden stack.",
    ),
    (
        "RECURRENT_RESIDUAL",
        23,
        {
            "recurrent_flag": True,
            "stack_residual_connection_option": AdditiveResidualConfig,
        },
        "Default recurrent config using a residual hidden stack at each recurrent step.",
    ),
    (
        "RECURRENT_POST_NORM",
        24,
        {
            "recurrent_flag": True,
            "layer_norm_position": LayerNormPositionOptions.AFTER,
        },
        "Default recurrent config using a post-normalized hidden stack at each "
        "recurrent step.",
    ),
)


@dataclass(frozen=True, slots=True)
class ForeignRuntimeOptions:
    value: object


def _shared_gate_config(dim: int = 16) -> GateConfig:
    return GateConfig(
        model_config=LayerStackConfig(
            input_dim=dim,
            hidden_dim=dim,
            output_dim=dim,
            num_layers=1,
            last_layer_bias_option=LastLayerBiasOptions.DEFAULT,
            apply_output_pipeline_flag=False,
            layer_config=LayerConfig(
                input_dim=dim,
                output_dim=dim,
                activation=ActivationOptions.DISABLED,
                residual_config=None,
                dropout_probability=0.0,
                layer_norm_position=LayerNormPositionOptions.DISABLED,
                gate_config=None,
                halting_config=None,
                memory_config=None,
                layer_model_config=LinearLayerConfig(
                    input_dim=dim,
                    output_dim=dim,
                    bias_flag=True,
                ),
            ),
        ),
        option=LayerGateOptions.MULTIPLIER,
        activation=ActivationOptions.SIGMOID,
    )


class TestLinearRuntimeDefaults(unittest.TestCase):
    def test_default_runtime_is_the_empty_flat_translation(self):
        self.assertEqual(DEFAULT_RUNTIME, runtime_from_flat({}))

    def test_runtime_option_types_are_frozen_and_slotted(self):
        option_types = (
            RuntimeOptions,
            MainStackOptions,
            ControllerStackOptions,
            GateOptions,
            HaltingOptions,
            MemoryOptions,
            RecurrenceOptions,
            ResidualStackOptions,
        )
        for option_type in option_types:
            with self.subTest(option_type=option_type.__name__):
                params = option_type.__dataclass_params__
                self.assertTrue(params.frozen)
                self.assertNotIn("__dict__", option_type.__dict__)
                self.assertTrue(option_type.__slots__)

        with self.assertRaises(FrozenInstanceError):
            DEFAULT_RUNTIME.hidden_dim = 64

    def test_every_supported_model_config_key_resolves_and_builds(self):
        model_keys = [
            key
            for key in iter_supported_config_keys(config)
            if key not in _NON_MODEL_KEYS
            and not any(key.startswith(prefix) for prefix in _NON_MODEL_PREFIXES)
        ]

        self.assertEqual(len(model_keys), 112)
        for key in model_keys:
            with self.subTest(key=key):
                flat_key = config_key_to_model_param(key)
                runtime = runtime_from_flat({flat_key: getattr(config, key)})
                LinearConfigBuilder(runtime=runtime).build()

    def test_canonical_runtime_fields_bind_to_typed_options(self):
        runtime = runtime_from_flat(
            {
                "stack_gate_flag": True,
                "stack_halting_flag": True,
                "layer_norm_position": LayerNormPositionOptions.AFTER,
            }
        )

        self.assertTrue(runtime.gate.enabled)
        self.assertTrue(runtime.halting.enabled)
        self.assertIs(
            runtime.stack.layer_norm_position,
            LayerNormPositionOptions.AFTER,
        )

    def test_noncanonical_runtime_key_spellings_are_rejected(self):
        for retired_key in (
            "stack_layer_norm_position",
            "layer-norm-position",
            "LAYER_NORM_POSITION",
            " layer_norm_position",
        ):
            with self.subTest(retired_key=retired_key):
                with self.assertRaisesRegex(ValueError, "unknown runtime override"):
                    runtime_from_flat({retired_key: LayerNormPositionOptions.AFTER})

    def test_unknown_key_names_package_invalid_key_and_accepted_keys(self):
        with self.assertRaises(ValueError) as raised:
            runtime_from_flat({"unknown_width": 12})

        message = str(raised.exception)
        self.assertIn("models.linears.linear", message)
        self.assertIn("unknown_width", message)
        self.assertIn("accepted keys", message)
        self.assertIn("hidden_dim", message)

    def test_incorrect_types_name_key_actual_type_and_expected_type(self):
        cases = (
            ("hidden_dim", "64", "str", "int"),
            ("learning_rate", 1, "int", "float"),
            ("stack_gate_flag", 1, "int", "bool"),
            ("stack_activation", "GELU", "str", "ActivationOptions"),
            ("memory_option", str, "type", "type[DynamicMemoryConfig]"),
        )
        for key, value, actual_type, expected_type in cases:
            with self.subTest(key=key):
                with self.assertRaises(TypeError) as raised:
                    runtime_from_flat({key: value})
                message = str(raised.exception)
                self.assertIn(key, message)
                self.assertIn(actual_type, message)
                self.assertIn(expected_type, message)

    def test_positive_values_are_validated_before_construction(self):
        cases = (
            ("batch_size", 0),
            ("learning_rate", 0.0),
            ("input_dim", 0),
            ("hidden_dim", -1),
            ("output_dim", 0),
            ("stack_num_layers", 0),
            ("submodule_stack_num_layers", -1),
            ("gate_stack_hidden_dim", 0),
            ("recurrent_max_steps", 0),
            ("memory_test_time_training_learning_rate", 0.0),
            ("memory_test_time_training_num_inner_steps", 0),
        )
        for key, value in cases:
            with self.subTest(key=key):
                with self.assertRaisesRegex(ValueError, key):
                    runtime_from_flat({key: value})

    def test_probabilities_and_thresholds_are_validated(self):
        cases = (
            ("stack_dropout_probability", -0.1),
            ("submodule_stack_dropout_probability", 1.0),
            ("gate_stack_dropout_probability", 1.1),
            ("halting_dropout", -0.1),
            ("recurrent_halting_dropout", 1.0),
            ("halting_threshold", 0.0),
            ("halting_threshold", 1.1),
            ("recurrent_halting_threshold", 0.0),
        )
        for key, value in cases:
            with self.subTest(key=key, value=value):
                with self.assertRaisesRegex(ValueError, key):
                    runtime_from_flat({key: value})

    def test_enabled_gates_require_a_concrete_option(self):
        for flag, option in (
            ("stack_gate_flag", "gate_option"),
            ("recurrent_stack_gate_flag", "recurrent_gate_option"),
        ):
            with self.subTest(flag=flag):
                with self.assertRaisesRegex(ValueError, option):
                    runtime_from_flat({flag: True, option: None})

    def test_controller_stacks_inherit_submodule_defaults(self):
        runtime = runtime_from_flat(
            {
                "submodule_stack_hidden_dim": 37,
                "submodule_stack_num_layers": 4,
                "submodule_stack_activation": ActivationOptions.MISH,
                "submodule_stack_layer_norm_position": (LayerNormPositionOptions.AFTER),
                "submodule_stack_residual_connection_option": (
                    AdditiveResidualConfig
                ),
                "submodule_stack_dropout_probability": 0.12,
                "submodule_stack_last_layer_bias_option": (
                    LastLayerBiasOptions.DEFAULT
                ),
                "submodule_stack_apply_output_pipeline_flag": True,
                "submodule_stack_bias_flag": False,
                "gate_stack_hidden_dim": 99,
                "halting_stack_num_layers": 8,
                "memory_stack_activation": ActivationOptions.RELU,
            }
        )

        self.assertEqual(runtime.gate.stack.hidden_dim, 37)
        self.assertEqual(runtime.gate.stack.num_layers, 4)
        self.assertIs(runtime.gate.stack.activation, ActivationOptions.MISH)
        self.assertEqual(runtime.memory.stack, runtime.gate.stack)
        self.assertEqual(
            runtime.halting.stack,
            replace(
                runtime.gate.stack,
                last_layer_bias_option=LastLayerBiasOptions.DISABLED,
            ),
        )

    def test_independent_controller_stacks_apply_partial_overrides(self):
        runtime = runtime_from_flat(
            {
                "submodule_stack_hidden_dim": 17,
                "submodule_stack_num_layers": 4,
                "submodule_stack_activation": ActivationOptions.MISH,
                "gate_stack_independent_flag": True,
                "gate_stack_hidden_dim": 29,
                "gate_stack_num_layers": None,
                "gate_stack_activation": None,
                "halting_stack_independent_flag": True,
                "halting_stack_num_layers": 5,
                "memory_stack_independent_flag": True,
                "memory_stack_activation": ActivationOptions.TANH,
            }
        )

        self.assertEqual(runtime.gate.stack.hidden_dim, 29)
        self.assertEqual(runtime.gate.stack.num_layers, 4)
        self.assertIs(runtime.gate.stack.activation, ActivationOptions.MISH)
        self.assertEqual(runtime.halting.stack.hidden_dim, 17)
        self.assertEqual(runtime.halting.stack.num_layers, 5)
        self.assertIs(runtime.memory.stack.activation, ActivationOptions.TANH)

    def test_recurrent_stacks_inherit_resolved_controller_stacks(self):
        runtime = runtime_from_flat(
            {
                "gate_stack_independent_flag": True,
                "gate_stack_hidden_dim": 31,
                "gate_stack_activation": ActivationOptions.SILU,
                "halting_stack_independent_flag": True,
                "halting_stack_hidden_dim": 41,
                "halting_stack_num_layers": 4,
            }
        )

        self.assertEqual(runtime.recurrence.gate.stack, runtime.gate.stack)
        self.assertEqual(runtime.recurrence.halting.stack, runtime.halting.stack)

        independent = runtime_from_flat(
            {
                "gate_stack_independent_flag": True,
                "gate_stack_hidden_dim": 31,
                "recurrent_gate_stack_independent_flag": True,
                "recurrent_gate_stack_hidden_dim": 64,
            }
        )
        self.assertEqual(independent.gate.stack.hidden_dim, 31)
        self.assertEqual(independent.recurrence.gate.stack.hidden_dim, 64)

    def test_typed_callers_customize_concrete_runtime_with_replace(self):
        runtime = replace(
            DEFAULT_RUNTIME,
            hidden_dim=24,
            stack=replace(
                DEFAULT_RUNTIME.stack,
                num_layers=3,
                activation=ActivationOptions.RELU,
            ),
            gate=replace(DEFAULT_RUNTIME.gate, enabled=True),
        )

        self.assertEqual(runtime.hidden_dim, 24)
        self.assertEqual(runtime.stack.num_layers, 3)
        self.assertIs(runtime.stack.activation, ActivationOptions.RELU)
        self.assertTrue(runtime.gate.enabled)


class TestLinearConstruction(unittest.TestCase):
    def test_builder_exposes_only_keyword_runtime(self):
        signature = inspect.signature(LinearConfigBuilder)

        self.assertEqual(list(signature.parameters), ["runtime"])
        self.assertIs(
            signature.parameters["runtime"].kind,
            inspect.Parameter.KEYWORD_ONLY,
        )
        self.assertIs(signature.parameters["runtime"].default, DEFAULT_RUNTIME)

        with self.assertRaises(TypeError):
            LinearConfigBuilder(DEFAULT_RUNTIME)
        for kwargs in (
            {"hidden_dim": 12},
            {"stack_options": object()},
            {"layer_controller_options": object()},
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(TypeError):
                    LinearConfigBuilder(**kwargs)

    def test_builder_rejects_foreign_runtime_objects(self):
        with self.assertRaises(TypeError) as raised:
            LinearConfigBuilder(runtime=ForeignRuntimeOptions(DEFAULT_RUNTIME))

        message = str(raised.exception)
        self.assertIn("models.linears.linear", message)
        self.assertIn("RuntimeOptions", message)
        self.assertIn("ForeignRuntimeOptions", message)

    def test_default_builder_constructs_expected_model_config(self):
        cfg = LinearConfigBuilder().build()

        self.assertEqual(cfg.batch_size, DEFAULT_RUNTIME.batch_size)
        self.assertEqual(cfg.learning_rate, DEFAULT_RUNTIME.learning_rate)
        self.assertEqual(cfg.input_dim, DEFAULT_RUNTIME.input_dim)
        self.assertEqual(cfg.hidden_dim, DEFAULT_RUNTIME.hidden_dim)
        self.assertEqual(cfg.output_dim, DEFAULT_RUNTIME.output_dim)
        self.assertIsInstance(cfg.experiment_config.input_model_config, LayerConfig)
        self.assertIsInstance(cfg.experiment_config.model_config, LayerStackConfig)
        self.assertIsInstance(cfg.experiment_config.output_model_config, LayerConfig)

    def test_flat_and_custom_typed_runtimes_build_equivalent_configs(self):
        flat_runtime = runtime_from_flat(
            {
                "hidden_dim": 24,
                "stack_num_layers": 3,
                "stack_activation": ActivationOptions.RELU,
                "stack_gate_flag": True,
            }
        )
        typed_runtime = replace(
            DEFAULT_RUNTIME,
            hidden_dim=24,
            stack=replace(
                DEFAULT_RUNTIME.stack,
                num_layers=3,
                activation=ActivationOptions.RELU,
            ),
            gate=replace(DEFAULT_RUNTIME.gate, enabled=True),
        )

        self.assertEqual(flat_runtime, typed_runtime)
        self.assertEqual(
            LinearConfigBuilder(runtime=flat_runtime).build(),
            LinearConfigBuilder(runtime=typed_runtime).build(),
        )

    def test_builder_uses_one_projection_factory_instance(self):
        with patch(
            "models.linears.linear.config_builder.ProjectionConfigFactory",
            wraps=ProjectionConfigFactory,
        ) as factory_type:
            LinearConfigBuilder().build()

        factory_type.assert_called_once_with(DEFAULT_RUNTIME)

    def test_projection_configs_are_separate_plain_linear_layers(self):
        runtime = runtime_from_flat({"stack_activation": ActivationOptions.MISH})
        experiment = LinearConfigBuilder(runtime=runtime).build().experiment_config

        self.assertIsNot(experiment.input_model_config, experiment.output_model_config)
        for projection, activation in (
            (experiment.input_model_config, ActivationOptions.MISH),
            (experiment.output_model_config, ActivationOptions.DISABLED),
        ):
            with self.subTest(activation=activation):
                self.assertIs(projection.activation, activation)
                self.assertIs(
                    projection.layer_norm_position,
                    LayerNormPositionOptions.DISABLED,
                )
                self.assertIsNone(projection.residual_config)
                self.assertEqual(projection.dropout_probability, 0.0)
                self.assertIsNone(projection.gate_config)
                self.assertIsNone(projection.halting_config)
                self.assertIsInstance(projection.layer_model_config, LinearLayerConfig)
                self.assertTrue(projection.layer_model_config.bias_flag)

    def test_gate_halting_and_memory_combinations_construct_independently(self):
        for gate_enabled in (False, True):
            for halting_enabled in (False, True):
                for memory_enabled in (False, True):
                    with self.subTest(
                        gate=gate_enabled,
                        halting=halting_enabled,
                        memory=memory_enabled,
                    ):
                        runtime = runtime_from_flat(
                            {
                                "stack_gate_flag": gate_enabled,
                                "stack_halting_flag": halting_enabled,
                                "memory_flag": memory_enabled,
                            }
                        )
                        stack = (
                            LinearConfigBuilder(runtime=runtime)
                            .build()
                            .experiment_config.model_config
                        )
                        self.assertEqual(
                            stack.layer_config.gate_config is not None,
                            gate_enabled,
                        )
                        self.assertEqual(
                            stack.layer_config.halting_config is not None,
                            halting_enabled,
                        )
                        self.assertEqual(
                            stack.shared_memory_config is not None,
                            memory_enabled,
                        )

    def test_control_configs_use_resolved_runtime_stacks(self):
        runtime = runtime_from_flat(
            {
                "stack_gate_flag": True,
                "stack_halting_flag": True,
                "memory_flag": True,
                "gate_stack_independent_flag": True,
                "gate_stack_hidden_dim": 22,
                "gate_stack_activation": ActivationOptions.SILU,
                "halting_stack_independent_flag": True,
                "halting_stack_hidden_dim": 33,
                "halting_stack_num_layers": 5,
                "memory_stack_independent_flag": True,
                "memory_stack_hidden_dim": 44,
                "memory_stack_activation": ActivationOptions.TANH,
            }
        )
        stack = (
            LinearConfigBuilder(runtime=runtime).build().experiment_config.model_config
        )

        self.assertEqual(stack.layer_config.gate_config.model_config.hidden_dim, 22)
        self.assertIs(
            stack.layer_config.gate_config.model_config.layer_config.activation,
            ActivationOptions.SILU,
        )
        self.assertEqual(
            stack.layer_config.halting_config.halting_gate_config.hidden_dim,
            33,
        )
        self.assertEqual(
            stack.layer_config.halting_config.halting_gate_config.output_dim,
            2,
        )
        self.assertEqual(
            stack.layer_config.halting_config.halting_gate_config.num_layers,
            5,
        )
        self.assertEqual(stack.shared_memory_config.model_config.hidden_dim, 44)
        self.assertIs(
            stack.shared_memory_config.model_config.layer_config.activation,
            ActivationOptions.TANH,
        )

    def test_custom_memory_options_are_forwarded(self):
        runtime = runtime_from_flat(
            {
                "hidden_dim": 8,
                "memory_flag": True,
                "memory_option": WeightedDynamicMemoryConfig,
                "memory_position_option": MemoryPositionOptions.BEFORE_AFFINE,
                "memory_test_time_training_learning_rate": 0.02,
                "memory_test_time_training_num_inner_steps": 2,
            }
        )
        memory = (
            LinearConfigBuilder(runtime=runtime)
            .build()
            .experiment_config.model_config.shared_memory_config
        )

        self.assertIsInstance(memory, WeightedDynamicMemoryConfig)
        self.assertIs(
            memory.memory_position_option,
            MemoryPositionOptions.BEFORE_AFFINE,
        )
        self.assertEqual(memory.test_time_training_learning_rate, 0.02)
        self.assertEqual(memory.test_time_training_num_inner_steps, 2)

    def test_recurrence_wraps_the_hidden_stack_and_uses_own_controllers(self):
        runtime = runtime_from_flat(
            {
                "recurrent_flag": True,
                "recurrent_max_steps": 3,
                "recurrent_layer_norm_position": LayerNormPositionOptions.AFTER,
                "recurrent_stack_gate_flag": True,
                "recurrent_gate_stack_independent_flag": True,
                "recurrent_gate_stack_hidden_dim": 64,
                "recurrent_stack_halting_flag": True,
                "recurrent_halting_threshold": 0.65,
                "recurrent_halting_stack_independent_flag": True,
                "recurrent_halting_stack_hidden_dim": 72,
                "memory_flag": True,
            }
        )
        recurrent = (
            LinearConfigBuilder(runtime=runtime).build().experiment_config.model_config
        )

        self.assertIsInstance(recurrent, RecurrentLayerConfig)
        self.assertEqual(recurrent.max_steps, 3)
        self.assertIs(
            recurrent.recurrent_layer_norm_position,
            LayerNormPositionOptions.AFTER,
        )
        self.assertEqual(recurrent.gate_config.model_config.hidden_dim, 64)
        self.assertEqual(
            recurrent.halting_config.halting_gate_config.hidden_dim,
            72,
        )
        self.assertEqual(recurrent.halting_config.threshold, 0.65)
        self.assertIsNotNone(recurrent.block_config.shared_memory_config)
        self.assertIsNone(recurrent.memory_config)

    def test_shared_gate_is_a_typed_runtime_customization(self):
        shared_gate = _shared_gate_config()
        runtime = replace(
            DEFAULT_RUNTIME,
            gate=replace(
                DEFAULT_RUNTIME.gate,
                enabled=False,
                shared_config=shared_gate,
            ),
        )
        cfg = LinearConfigBuilder(runtime=runtime).build()
        stack = cfg.experiment_config.model_config

        self.assertIs(stack.shared_gate_config, shared_gate)
        self.assertIsNone(stack.layer_config.gate_config)

        invalid_runtime = replace(
            runtime,
            gate=replace(runtime.gate, enabled=True),
        )
        invalid_cfg = LinearConfigBuilder(runtime=invalid_runtime).build()
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            Model(invalid_cfg)

    def test_factories_depend_only_on_package_runtime_not_flat_config(self):
        package_dir = Path(config.__file__).resolve().parent
        factory_paths = (
            package_dir / "_control_config_factory.py",
            package_dir / "_hidden_model_config_factory.py",
            package_dir / "_projection_config_factory.py",
        )
        controller_stack_implementations = 0
        for path in factory_paths:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            imported_modules = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported_modules.update(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imported_modules.add(node.module)
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    controller_stack_implementations += node.name == "_controller_stack"

            self.assertNotIn("models.linears.linear.config", imported_modules)
            self.assertFalse(
                {
                    module
                    for module in imported_modules
                    if module.startswith("models.linears.")
                    and not module.startswith("models.linears.linear.")
                },
                path,
            )

        self.assertEqual(controller_stack_implementations, 1)


class TestLinearPresetsAndMetadata(unittest.TestCase):
    def test_public_modules_and_catalog_identity_remain_stable(self):
        for module_name in (
            "models.linears.linear.config",
            "models.linears.linear.presets",
            "models.linears.linear.model",
            "models.linears.linear.config_builder",
            "models.linears.linear.experiment_config",
        ):
            with self.subTest(module_name=module_name):
                self.assertEqual(
                    importlib.import_module(module_name).__name__,
                    module_name,
                )

        experiment = Experiment(model_package=model_package("linears/linear"))
        self.assertEqual(
            experiment.model_package.identity.catalog_key,
            "linears/linear",
        )

    def test_preset_enum_values_and_definitions_remain_stable(self):
        self.assertTrue(issubclass(ExperimentPreset, BaseOptions))
        self.assertEqual(
            [preset.value for preset in ExperimentPreset],
            list(range(1, 36)),
        )
        self.assertEqual(set(_PRESET_DEFINITIONS), set(ExperimentPreset))

        presets = model_package("linears/linear").presets
        for preset in ExperimentPreset:
            with self.subTest(preset=preset.name):
                definition = presets.definition_for_preset(preset)
                self.assertIsInstance(definition, PresetDefinition)
                self.assertEqual(definition, _PRESET_DEFINITIONS[preset])
                self.assertEqual(
                    presets.overrides_for_preset(preset),
                    definition.preset_values,
                )
                self.assertEqual(
                    presets.description_for_preset(preset),
                    definition.description,
                )
                self.assertTrue(definition.description)

    def test_historical_presets_and_default_remain_exact(self):
        presets = model_package("linears/linear").presets

        self.assertIs(presets.default_preset, ExperimentPreset.BASELINE)
        self.assertEqual(
            [preset.name for preset in list(ExperimentPreset)[:24]],
            [name for name, _, _, _ in _HISTORICAL_PRESET_SNAPSHOT],
        )
        for name, value, override_map, description in _HISTORICAL_PRESET_SNAPSHOT:
            with self.subTest(preset=name):
                preset = ExperimentPreset[name]
                self.assertEqual(preset.value, value)
                self.assertEqual(
                    presets.overrides_for_preset(preset),
                    override_map,
                )
                self.assertEqual(
                    presets.description_for_preset(preset),
                    description,
                )

    def test_every_preset_builds_with_its_declared_values(self):
        presets = model_package("linears/linear").presets
        for preset in ExperimentPreset:
            with self.subTest(preset=preset.name):
                cfg = presets.get_config(preset)[0]
                self.assertIsNotNone(cfg.experiment_config.model_config)

    def test_new_preset_values_and_override_maps_are_exact(self):
        presets = model_package("linears/linear").presets

        for name, (
            value,
            override_map,
            description,
        ) in _NEW_PRESET_EXPECTATIONS.items():
            with self.subTest(preset=name):
                preset = ExperimentPreset[name]
                self.assertEqual(preset.value, value)
                self.assertEqual(
                    presets.overrides_for_preset(preset),
                    override_map,
                )
                self.assertEqual(
                    presets.description_for_preset(preset),
                    description,
                )

    def test_all_resolved_preset_maps_are_pairwise_unique(self):
        presets = model_package("linears/linear").presets
        resolved_maps = [
            frozenset(presets.overrides_for_preset(preset).items())
            for preset in ExperimentPreset
        ]

        self.assertEqual(len(resolved_maps), 35)
        self.assertEqual(len(set(resolved_maps)), 35)

    def test_new_preset_locks_reject_conflicts_and_allow_unrelated_overrides(self):
        presets = model_package("linears/linear").presets
        conflicting_values = {
            "stack_residual_connection_option": AdditiveResidualConfig,
            "recurrent_flag": False,
            "stack_gate_flag": False,
            "recurrent_stack_gate_flag": False,
            "stack_halting_flag": False,
            "recurrent_stack_halting_flag": False,
            "memory_flag": False,
            "layer_norm_position": LayerNormPositionOptions.BEFORE,
        }

        for name, (_, override_map, _) in _NEW_PRESET_EXPECTATIONS.items():
            preset = ExperimentPreset[name]
            for field, locked_value in override_map.items():
                if field == "memory_option":
                    conflict = (
                        ElementWiseWeightedDynamicMemoryConfig
                        if locked_value is WeightedDynamicMemoryConfig
                        else WeightedDynamicMemoryConfig
                    )
                else:
                    conflict = conflicting_values[field]
                with (
                    self.subTest(preset=name, locked_field=field),
                    self.assertRaisesRegex(ValueError, field),
                ):
                    presets.get_config(
                        preset,
                        config_overrides={field: conflict},
                    )

            cfg = presets.get_config(
                preset,
                config_overrides={"hidden_dim": 19},
            )[0]
            self.assertEqual(cfg.hidden_dim, 19)

        memory_cfg = presets.get_config(
            ExperimentPreset.WEIGHTED_MEMORY,
            config_overrides={
                "memory_position_option": MemoryPositionOptions.BEFORE_AFFINE,
            },
        )[0]
        self.assertIs(
            memory_cfg.experiment_config.model_config.shared_memory_config.memory_position_option,
            MemoryPositionOptions.BEFORE_AFFINE,
        )

    def test_controller_presets_wire_expected_configs(self):
        expected = {
            ExperimentPreset.BASELINE: (False, False, False),
            ExperimentPreset.GATING: (True, False, False),
            ExperimentPreset.HALTING: (False, True, False),
            ExperimentPreset.MEMORY: (False, False, True),
            ExperimentPreset.GATING_HALTING: (True, True, False),
            ExperimentPreset.GATING_MEMORY: (True, False, True),
            ExperimentPreset.HALTING_MEMORY: (False, True, True),
            ExperimentPreset.GATING_HALTING_MEMORY: (True, True, True),
        }
        for preset, (gate, halting, memory) in expected.items():
            with self.subTest(preset=preset.name):
                stack = (
                    model_package("linears/linear")
                    .presets.get_config(preset)[0]
                    .experiment_config.model_config
                )
                self.assertEqual(stack.layer_config.gate_config is not None, gate)
                self.assertEqual(stack.layer_config.halting_config is not None, halting)
                self.assertEqual(stack.shared_memory_config is not None, memory)

    def test_residual_and_post_norm_presets_wire_expected_layers(self):
        cases = {
            ExperimentPreset.RESIDUAL: (
                AdditiveResidualConfig,
                config.LAYER_NORM_POSITION,
            ),
            ExperimentPreset.POST_NORM: (
                None,
                LayerNormPositionOptions.AFTER,
            ),
            ExperimentPreset.RESIDUAL_POST_NORM: (
                AdditiveResidualConfig,
                LayerNormPositionOptions.AFTER,
            ),
            ExperimentPreset.RESIDUAL_GATING: (
                AdditiveResidualConfig,
                config.LAYER_NORM_POSITION,
            ),
            ExperimentPreset.RESIDUAL_HALTING: (
                AdditiveResidualConfig,
                config.LAYER_NORM_POSITION,
            ),
            ExperimentPreset.RESIDUAL_MEMORY: (
                AdditiveResidualConfig,
                config.LAYER_NORM_POSITION,
            ),
            ExperimentPreset.WEIGHTED_RESIDUAL: (
                WeightedResidualConfig,
                config.LAYER_NORM_POSITION,
            ),
            ExperimentPreset.WEIGHTED_BLEND_RESIDUAL: (
                WeightedBlendResidualConfig,
                config.LAYER_NORM_POSITION,
            ),
            ExperimentPreset.ATTENTION_RESIDUAL: (
                AttentionResidualConfig,
                config.LAYER_NORM_POSITION,
            ),
        }
        for preset, (residual, norm) in cases.items():
            with self.subTest(preset=preset.name):
                layer = (
                    model_package("linears/linear")
                    .presets.get_config(preset)[0]
                    .experiment_config.model_config.layer_config
                )
                actual_residual = (
                    None
                    if layer.residual_config is None
                    else type(layer.residual_config)
                )
                self.assertIs(actual_residual, residual)
                self.assertIs(layer.layer_norm_position, norm)

    def test_attention_residual_preset_preserves_halting_incompatibility(self):
        cfg = model_package("linears/linear").presets.get_config(
            ExperimentPreset.ATTENTION_RESIDUAL,
            config_overrides={"stack_halting_flag": True},
        )[0]

        with self.assertRaisesRegex(
            ValueError,
            "halting.*AttentionResidualConfig|AttentionResidualConfig.*halting",
        ):
            Model(cfg)

    def test_recurrent_presets_wire_optional_controllers(self):
        expected = {
            ExperimentPreset.RECURRENT: (False, False, False),
            ExperimentPreset.RECURRENT_GATING: (True, False, False),
            ExperimentPreset.RECURRENT_HALTING: (False, True, False),
            ExperimentPreset.RECURRENT_MEMORY: (False, False, True),
            ExperimentPreset.RECURRENT_GATING_HALTING: (True, True, False),
            ExperimentPreset.RECURRENT_GATING_MEMORY: (True, False, True),
            ExperimentPreset.RECURRENT_HALTING_MEMORY: (False, True, True),
            ExperimentPreset.RECURRENT_GATING_HALTING_MEMORY: (True, True, True),
            ExperimentPreset.RECURRENT_RESIDUAL: (False, False, False),
            ExperimentPreset.RECURRENT_POST_NORM: (False, False, False),
        }
        for preset, (gate, halting, memory) in expected.items():
            with self.subTest(preset=preset.name):
                recurrent = (
                    model_package("linears/linear")
                    .presets.get_config(preset)[0]
                    .experiment_config.model_config
                )
                self.assertIsInstance(recurrent, RecurrentLayerConfig)
                self.assertEqual(recurrent.gate_config is not None, gate)
                self.assertEqual(recurrent.halting_config is not None, halting)
                self.assertEqual(
                    recurrent.block_config.shared_memory_config is not None,
                    memory,
                )

    def test_recurrent_gating_presets_place_controllers_by_scope(self):
        presets = model_package("linears/linear").presets
        cases = {
            ExperimentPreset.RECURRENT_LAYER_GATING: (True, False),
            ExperimentPreset.RECURRENT_DUAL_GATING: (True, True),
        }

        for preset, (has_inner_gate, has_outer_gate) in cases.items():
            with self.subTest(preset=preset.name):
                recurrent = presets.get_config(preset)[0].experiment_config.model_config
                self.assertIsInstance(recurrent, RecurrentLayerConfig)
                inner_gate = recurrent.block_config.layer_config.gate_config
                outer_gate = recurrent.gate_config
                self.assertEqual(inner_gate is not None, has_inner_gate)
                self.assertEqual(outer_gate is not None, has_outer_gate)
                if outer_gate is not None:
                    self.assertIsNot(inner_gate, outer_gate)

    def test_recurrent_halting_presets_place_controllers_by_scope(self):
        presets = model_package("linears/linear").presets
        cases = {
            ExperimentPreset.RECURRENT_LAYER_HALTING: (True, False),
            ExperimentPreset.RECURRENT_DUAL_HALTING: (True, True),
        }

        for preset, (has_inner_halting, has_outer_halting) in cases.items():
            with self.subTest(preset=preset.name):
                recurrent = presets.get_config(preset)[0].experiment_config.model_config
                self.assertIsInstance(recurrent, RecurrentLayerConfig)
                inner_halting = recurrent.block_config.layer_config.halting_config
                outer_halting = recurrent.halting_config
                self.assertEqual(inner_halting is not None, has_inner_halting)
                self.assertEqual(outer_halting is not None, has_outer_halting)
                if outer_halting is not None:
                    self.assertIsNot(inner_halting, outer_halting)

    def test_memory_presets_wire_exact_shared_implementations(self):
        presets = model_package("linears/linear").presets
        cases = {
            ExperimentPreset.WEIGHTED_MEMORY: WeightedDynamicMemoryConfig,
            ExperimentPreset.ELEMENT_WISE_WEIGHTED_MEMORY: (
                ElementWiseWeightedDynamicMemoryConfig
            ),
        }

        for preset, expected_type in cases.items():
            with self.subTest(preset=preset.name):
                stack = presets.get_config(preset)[0].experiment_config.model_config
                memory = stack.shared_memory_config
                self.assertIs(type(memory), expected_type)
                self.assertIs(
                    memory.memory_position_option,
                    MemoryPositionOptions.AFTER_AFFINE,
                )
                self.assertIsNone(memory.test_time_training_learning_rate)
                self.assertIsNone(memory.test_time_training_num_inner_steps)

    def test_normalization_presets_preserve_all_main_stack_positions(self):
        presets = model_package("linears/linear").presets
        cases = {
            ExperimentPreset.BASELINE: LayerNormPositionOptions.BEFORE,
            ExperimentPreset.POST_NORM: LayerNormPositionOptions.AFTER,
            ExperimentPreset.NO_NORM: LayerNormPositionOptions.DISABLED,
            ExperimentPreset.PRE_ACTIVATION_NORM: LayerNormPositionOptions.DEFAULT,
        }

        for preset, expected_position in cases.items():
            with self.subTest(preset=preset.name):
                cfg = presets.get_config(preset)[0]
                stack = cfg.experiment_config.model_config
                self.assertIs(
                    stack.layer_config.layer_norm_position,
                    expected_position,
                )
                layers = Model(cfg).main_model.layers
                self.assertTrue(layers)
                self.assertEqual(
                    all(layer.layer_norm_module is None for layer in layers),
                    expected_position is LayerNormPositionOptions.DISABLED,
                )

    def test_preset_locks_keep_values_and_reasons_together(self):
        presets = model_package("linears/linear").presets
        for preset in ExperimentPreset:
            with self.subTest(preset=preset.name):
                locks = presets.locks_for_preset(preset)
                self.assertEqual(
                    set(locks),
                    set(_PRESET_DEFINITIONS[preset].preset_values),
                )
                for field, lock in locks.items():
                    self.assertEqual(
                        lock.value,
                        _PRESET_DEFINITIONS[preset].preset_values[field],
                    )
                    self.assertIn(f"`{field}`", lock.reason)

    def test_locked_and_unlocked_overrides_keep_existing_behavior(self):
        presets = model_package("linears/linear").presets
        with self.assertRaisesRegex(ValueError, "stack_gate_flag"):
            presets.get_config(
                ExperimentPreset.GATING,
                config_overrides={"stack_gate_flag": False},
            )

        cfg = presets.get_config(
            ExperimentPreset.GATING,
            config_overrides={"hidden_dim": 19},
        )[0]
        self.assertEqual(cfg.hidden_dim, 19)

    def test_search_metadata_and_alias_remain_stable(self):
        self.assertEqual(search_space.SEARCH_SPACE_LEARNING_RATE, [1e-4, 1e-3, 1e-2])
        self.assertEqual(
            search_space.SEARCH_SPACE_HIDDEN_DIM,
            [16, 32, 64, 128, 256, 512],
        )
        self.assertEqual(search_space.SEARCH_SPACE_STACK_NUM_LAYERS, [2, 4, 8, 16, 32])
        self.assertIs(
            search_space.SEARCH_SPACE_LAYER_NORM_POSITION,
            search_space.SEARCH_SPACE_LAYER_NORM_POSITION,
        )

    def test_dataset_metadata_remains_task_grouped(self):
        self.assertIs(
            dataset_options.DEFAULT_EXPERIMENT_TASK,
            ExperimentTask.IMAGE_CLASSIFICATION,
        )
        self.assertEqual(
            dataset_options.DATASET_OPTIONS_BY_TASK,
            {
                ExperimentTask.IMAGE_CLASSIFICATION: [
                    Mnist,
                    FashionMNIST,
                    Cifar10,
                    Cifar100,
                ]
            },
        )

    def test_monitor_metadata_remains_available(self):
        self.assertEqual(
            [option.name for option in monitor_options.MONITOR_OPTIONS],
            ["linear", "recurrent-layer", "layer-controller", "halting", "memory"],
        )
        for option in monitor_options.MONITOR_OPTIONS:
            with self.subTest(option=option.name):
                self.assertTrue(option.label)
                self.assertTrue(option.kinds)
                self.assertIsNotNone(option.build_callback())

    def test_cli_exposes_only_canonical_runtime_override_flags(self):
        package = model_package("linears/linear")
        parser = get_experiment_parser(package)
        cases = (
            ("--hidden-dim", "64", "hidden_dim", 64),
            (
                "--layer-norm-position",
                "AFTER",
                "layer_norm_position",
                LayerNormPositionOptions.AFTER,
            ),
        )
        for flag, value, key, expected in cases:
            with self.subTest(flag=flag):
                args = parser.parse_args(["--preset", "baseline", flag, value])
                mode = resolve_cli_selection(args, package, ExperimentPreset)
                self.assertEqual(mode.config_overrides[key], expected)

        for removed_flag in (
            "--stack-layer-norm-position",
            "--stack-hidden-dim",
            "--bias-flag",
        ):
            with self.subTest(removed_flag=removed_flag):
                with contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit):
                        parser.parse_args(["--preset", "baseline", removed_flag, "64"])

    def test_module_entrypoint_resolves_without_training(self):
        with (
            patch.object(sys, "argv", ["linear", "--preset", "baseline"]),
            patch(
                "models.package_cli.execute_runs",
                return_value=(),
            ) as execute_runs,
            self.assertRaises(SystemExit) as exit_context,
        ):
            runpy.run_module("models.linears.linear.__main__", run_name="__main__")

        self.assertEqual(exit_context.exception.code, 0)
        execute_runs.assert_called_once()
        package, plan = execute_runs.call_args.args
        self.assertEqual(package.catalog_key, "linears/linear")
        self.assertEqual(plan.presets, ("baseline",))
        self.assertIsNone(plan.search)
        self.assertEqual(dict(plan.overrides), {})
        self.assertEqual(execute_runs.call_args.kwargs["monitors"], ())


class TestLinearModelBehavior(unittest.TestCase):
    def _fake_batch(self, dataset: type, batch_size: int) -> torch.Tensor:
        return torch.randn(
            batch_size,
            dataset.num_channels,
            dataset.default_height,
            dataset.default_width,
        )

    def test_model_preserves_full_config(self):
        cfg = LinearConfigBuilder().build()
        model = Model(cfg)

        self.assertIs(model.cfg, cfg)
        self.assertIs(model.experiment_config, cfg.experiment_config)

    def test_every_preset_forwards_one_mnist_batch(self):
        batch_size = 4
        dataset = dataset_options.DATASET_OPTIONS_BY_TASK[
            dataset_options.DEFAULT_EXPERIMENT_TASK
        ][0]
        presets = model_package("linears/linear").presets

        for preset in ExperimentPreset:
            with self.subTest(preset=preset.name):
                cfg = presets.get_config(preset, dataset)[0]
                output = Model(cfg)(self._fake_batch(dataset, batch_size))
                logits = output[0] if isinstance(output, tuple) else output
                self.assertEqual(logits.shape, (batch_size, dataset.num_classes))

    def test_baseline_forwards_all_declared_datasets(self):
        batch_size = 4
        presets = model_package("linears/linear").presets

        for dataset in dataset_options.DATASET_OPTIONS_BY_TASK[
            dataset_options.DEFAULT_EXPERIMENT_TASK
        ]:
            with self.subTest(dataset=dataset.__name__):
                cfg = presets.get_config(ExperimentPreset.BASELINE, dataset)[0]
                logits = Model(cfg)(self._fake_batch(dataset, batch_size))
                self.assertEqual(logits.shape, (batch_size, dataset.num_classes))

    def test_custom_dimensions_forward(self):
        runtime = runtime_from_flat(
            {
                "input_dim": 8,
                "hidden_dim": 12,
                "output_dim": 4,
                "stack_num_layers": 2,
            }
        )
        model = Model(LinearConfigBuilder(runtime=runtime).build())

        logits = model(torch.randn(3, 1, 2, 4))

        self.assertEqual(logits.shape, (3, 4))

    def test_memory_forward_and_backward_produce_memory_gradients(self):
        runtime = runtime_from_flat(
            {
                "input_dim": 8,
                "hidden_dim": 8,
                "output_dim": 4,
                "stack_num_layers": 2,
                "memory_flag": True,
            }
        )
        model = Model(LinearConfigBuilder(runtime=runtime).build())
        output = model(torch.randn(2, 1, 2, 4))
        logits = output[0] if isinstance(output, tuple) else output

        self.assertEqual(logits.shape, (2, 4))
        logits.sum().backward()

        memory_parameters = [
            parameter
            for name, parameter in model.named_parameters()
            if "memory_model" in name and parameter.requires_grad
        ]
        self.assertTrue(memory_parameters)
        self.assertTrue(
            any(
                parameter.grad is not None and torch.any(parameter.grad.abs() > 0)
                for parameter in memory_parameters
            )
        )

    def test_new_residual_presets_are_forward_local_and_backpropagate(self):
        presets = model_package("linears/linear").presets
        for preset in (
            ExperimentPreset.WEIGHTED_RESIDUAL,
            ExperimentPreset.WEIGHTED_BLEND_RESIDUAL,
            ExperimentPreset.ATTENTION_RESIDUAL,
        ):
            with self.subTest(preset=preset.name):
                cfg = presets.get_config(
                    preset,
                    config_overrides={
                        "input_dim": 8,
                        "hidden_dim": 8,
                        "output_dim": 4,
                        "stack_num_layers": 2,
                        "stack_dropout_probability": 0.0,
                    },
                )[0]
                model = Model(cfg)
                model.eval()
                batch = torch.randn(3, 1, 2, 4)

                first_output = model(batch)
                second_output = model(batch)
                first_logits = (
                    first_output[0] if isinstance(first_output, tuple) else first_output
                )
                second_logits = (
                    second_output[0]
                    if isinstance(second_output, tuple)
                    else second_output
                )

                torch.testing.assert_close(second_logits, first_logits)
                second_logits.square().mean().backward()
                residual_parameters = [
                    parameter
                    for name, parameter in model.named_parameters()
                    if "residual_connection" in name and parameter.requires_grad
                ]
                self.assertTrue(residual_parameters)
                self.assertTrue(
                    any(
                        parameter.grad is not None
                        and torch.isfinite(parameter.grad).all()
                        and torch.any(parameter.grad.abs() > 0)
                        for parameter in residual_parameters
                    )
                )

    def test_new_memory_presets_forward_and_backpropagate(self):
        presets = model_package("linears/linear").presets
        for preset in (
            ExperimentPreset.WEIGHTED_MEMORY,
            ExperimentPreset.ELEMENT_WISE_WEIGHTED_MEMORY,
        ):
            with self.subTest(preset=preset.name):
                cfg = presets.get_config(
                    preset,
                    config_overrides={
                        "input_dim": 8,
                        "hidden_dim": 8,
                        "output_dim": 4,
                        "stack_num_layers": 2,
                        "stack_dropout_probability": 0.0,
                    },
                )[0]
                model = Model(cfg)

                output = model(torch.randn(3, 1, 2, 4))
                logits = output[0] if isinstance(output, tuple) else output

                self.assertEqual(logits.shape, (3, 4))
                logits.square().mean().backward()
                memory_parameters = [
                    parameter
                    for name, parameter in model.named_parameters()
                    if "memory_model" in name and parameter.requires_grad
                ]
                self.assertTrue(memory_parameters)
                self.assertTrue(
                    any(
                        parameter.grad is not None
                        and torch.isfinite(parameter.grad).all()
                        and torch.any(parameter.grad.abs() > 0)
                        for parameter in memory_parameters
                    )
                )

    def test_recurrent_controller_combination_forwards(self):
        runtime = runtime_from_flat(
            {
                "input_dim": 8,
                "hidden_dim": 8,
                "output_dim": 4,
                "stack_num_layers": 2,
                "recurrent_flag": True,
                "recurrent_stack_gate_flag": True,
                "recurrent_stack_halting_flag": True,
                "memory_flag": True,
            }
        )
        model = Model(LinearConfigBuilder(runtime=runtime).build())
        output = model(torch.randn(2, 1, 2, 4))
        logits = output[0] if isinstance(output, tuple) else output

        self.assertEqual(logits.shape, (2, 4))

    def test_recurrent_dual_gating_backpropagates_through_both_scopes(self):
        cfg = model_package("linears/linear").presets.get_config(
            ExperimentPreset.RECURRENT_DUAL_GATING,
            config_overrides={
                "input_dim": 8,
                "hidden_dim": 8,
                "output_dim": 4,
                "stack_num_layers": 2,
                "recurrent_max_steps": 2,
                "stack_dropout_probability": 0.0,
            },
        )[0]
        model = Model(cfg)

        output = model(torch.randn(3, 1, 2, 4))
        logits = output[0] if isinstance(output, tuple) else output
        logits.square().mean().backward()

        inner_gate_parameters = [
            parameter
            for name, parameter in model.named_parameters()
            if "block_model" in name
            and "gate_model" in name
            and parameter.requires_grad
        ]
        outer_gate_parameters = [
            parameter
            for name, parameter in model.named_parameters()
            if "recurrent_gate" in name and parameter.requires_grad
        ]
        for scope, parameters in (
            ("inner", inner_gate_parameters),
            ("outer", outer_gate_parameters),
        ):
            with self.subTest(scope=scope):
                self.assertTrue(parameters)
                self.assertTrue(
                    any(
                        parameter.grad is not None
                        and torch.isfinite(parameter.grad).all()
                        and torch.any(parameter.grad.abs() > 0)
                        for parameter in parameters
                    )
                )

    def test_recurrent_dual_halting_is_forward_local_and_backpropagates(self):
        cfg = model_package("linears/linear").presets.get_config(
            ExperimentPreset.RECURRENT_DUAL_HALTING,
            config_overrides={
                "input_dim": 8,
                "hidden_dim": 8,
                "output_dim": 4,
                "stack_num_layers": 2,
                "recurrent_max_steps": 2,
                "stack_dropout_probability": 0.0,
                "halting_dropout": 0.0,
                "recurrent_halting_dropout": 0.0,
            },
        )[0]
        model = Model(cfg)
        model.eval()
        batch = torch.randn(3, 1, 2, 4)

        first_logits, first_ponder_loss = model(batch)
        second_logits, second_ponder_loss = model(batch)

        torch.testing.assert_close(second_logits, first_logits)
        torch.testing.assert_close(second_ponder_loss, first_ponder_loss)
        self.assertTrue(torch.isfinite(second_ponder_loss))
        self.assertTrue(second_ponder_loss.requires_grad)
        (second_logits.square().mean() + second_ponder_loss).backward()

        inner_halting_parameters = [
            parameter
            for name, parameter in model.named_parameters()
            if "block_model" in name
            and "halting_model" in name
            and parameter.requires_grad
        ]
        outer_halting_parameters = [
            parameter
            for name, parameter in model.named_parameters()
            if name.startswith("main_model.halting_model") and parameter.requires_grad
        ]
        for scope, parameters in (
            ("inner", inner_halting_parameters),
            ("outer", outer_halting_parameters),
        ):
            with self.subTest(scope=scope):
                self.assertTrue(parameters)
                self.assertTrue(
                    any(
                        parameter.grad is not None
                        and torch.isfinite(parameter.grad).all()
                        and torch.any(parameter.grad.abs() > 0)
                        for parameter in parameters
                    )
                )

    def test_new_preset_state_dicts_round_trip_strictly(self):
        presets = model_package("linears/linear").presets
        batch = torch.randn(3, 1, 2, 4)
        for name in _NEW_PRESET_EXPECTATIONS:
            with self.subTest(preset=name):
                cfg = presets.get_config(
                    ExperimentPreset[name],
                    config_overrides={
                        "input_dim": 8,
                        "hidden_dim": 8,
                        "output_dim": 4,
                        "stack_num_layers": 2,
                        "recurrent_max_steps": 2,
                        "stack_dropout_probability": 0.0,
                        "halting_dropout": 0.0,
                        "recurrent_halting_dropout": 0.0,
                    },
                )[0]
                source = Model(cfg).eval()
                restored = Model(cfg).eval()

                load_result = restored.load_state_dict(
                    source.state_dict(),
                    strict=True,
                )

                self.assertEqual(load_result.missing_keys, [])
                self.assertEqual(load_result.unexpected_keys, [])
                source_output = source(batch)
                restored_output = restored(batch)
                source_tensors = (
                    source_output
                    if isinstance(source_output, tuple)
                    else (source_output,)
                )
                restored_tensors = (
                    restored_output
                    if isinstance(restored_output, tuple)
                    else (restored_output,)
                )
                self.assertEqual(len(restored_tensors), len(source_tensors))
                for actual, expected in zip(
                    restored_tensors,
                    source_tensors,
                    strict=True,
                ):
                    torch.testing.assert_close(actual, expected)

    def test_all_presets_train_one_tiny_epoch(self):
        dataset = dataset_options.DATASET_OPTIONS_BY_TASK[
            dataset_options.DEFAULT_EXPERIMENT_TASK
        ][0]
        presets = model_package("linears/linear").presets

        for preset in ExperimentPreset:
            with self.subTest(preset=preset.name):
                cfg = presets.get_config(preset, dataset)[0]
                tiny_cpu_trainer().fit(
                    Model(cfg),
                    datamodule=RandomImageClassificationDataModule(dataset),
                )


if __name__ == "__main__":
    unittest.main()
