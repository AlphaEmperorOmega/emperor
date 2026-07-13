import ast
import inspect
import unittest
from dataclasses import FrozenInstanceError, fields, replace

import models.linears.linear_adaptive.dataset_options as dataset_options
import models.linears.linear_adaptive.monitor_options as monitor_options
import models.linears.linear_adaptive.search_space as search_space
import torch
from emperor.augmentations.adaptive_parameters.core.weight import (
    DualModelDynamicWeightConfig,
)
from emperor.base.layer.config import LayerStackConfig, RecurrentLayerConfig
from emperor.base.options import ActivationOptions, LayerNormPositionOptions
from emperor.linears.core.config import AdaptiveLinearLayerConfig
from models.linears.linear_adaptive import (
    _adaptive_parameter_config_factory,
    _control_config_factory,
    _hidden_model_config_factory,
    _projection_config_factory,
    runtime_defaults,
)
from models.linears.linear_adaptive.config_builder import (
    LinearAdaptiveConfigBuilder,
)
from models.linears.linear_adaptive.model import Model
from models.linears.linear_adaptive.presets import (
    ExperimentPreset,
    ExperimentPresets,
)
from models.linears.linear_adaptive.runtime_defaults import (
    DEFAULT_RUNTIME,
    runtime_from_flat,
)
from models.training_test_utils import (
    RandomImageClassificationDataModule,
    tiny_cpu_trainer,
)

from model_runtime.packages import PresetDefinition


class TestRuntimeDefaults(unittest.TestCase):
    def test_default_runtime_is_the_empty_flat_translation(self):
        self.assertEqual(DEFAULT_RUNTIME, runtime_from_flat({}))

    def test_runtime_types_are_frozen_and_slotted(self):
        with self.assertRaises(FrozenInstanceError):
            DEFAULT_RUNTIME.hidden_dim = 12
        self.assertFalse(hasattr(DEFAULT_RUNTIME, "__dict__"))
        self.assertGreater(len(fields(DEFAULT_RUNTIME)), 1)

    def test_every_supported_flat_default_translates(self):
        for key, value in runtime_defaults._FLAT_DEFAULTS.items():
            with self.subTest(key=key):
                self.assertIsNotNone(runtime_from_flat({key: value}))

    def test_aliases_translate_and_conflicts_fail(self):
        runtime = runtime_from_flat(
            {"stack_layer_norm_position": DEFAULT_RUNTIME.stack.layer_norm_position}
        )
        self.assertEqual(
            runtime.stack.layer_norm_position,
            DEFAULT_RUNTIME.stack.layer_norm_position,
        )
        with self.assertRaisesRegex(ValueError, "conflicting aliases"):
            runtime_from_flat(
                {
                    "layer_norm_position": DEFAULT_RUNTIME.stack.layer_norm_position,
                    "stack_layer_norm_position": LayerNormPositionOptions.AFTER,
                }
            )

    def test_unknown_type_and_invalid_values_fail_atomically(self):
        with self.assertRaisesRegex(ValueError, "linear_adaptive.*unknown.*bogus"):
            runtime_from_flat({"bogus": 1})
        with self.assertRaisesRegex(TypeError, "hidden_dim.*str.*int"):
            runtime_from_flat({"hidden_dim": "wide"})
        for key, value in (
            ("hidden_dim", 0),
            ("batch_size", 0),
            ("learning_rate", 0.0),
            ("stack_num_layers", 0),
            ("recurrent_max_steps", 0),
            ("stack_dropout_probability", 1.1),
            ("halting_threshold", -0.1),
            ("mask_threshold", 2.0),
        ):
            with self.subTest(key=key), self.assertRaises(ValueError):
                runtime_from_flat({key: value})

    def test_controller_and_recurrent_stacks_are_fully_resolved(self):
        inherited = runtime_from_flat(
            {
                "submodule_stack_activation": ActivationOptions.RELU,
                "gate_stack_activation": ActivationOptions.TANH,
                "recurrent_gate_stack_activation": ActivationOptions.SIGMOID,
            }
        )
        self.assertEqual(inherited.gate.stack.activation, ActivationOptions.RELU)
        self.assertEqual(
            inherited.recurrence.gate.stack.activation,
            ActivationOptions.RELU,
        )
        independent = runtime_from_flat(
            {
                "gate_stack_independent_flag": True,
                "gate_stack_activation": ActivationOptions.TANH,
                "recurrent_gate_stack_independent_flag": True,
                "recurrent_gate_stack_activation": ActivationOptions.SIGMOID,
            }
        )
        self.assertEqual(independent.gate.stack.activation, ActivationOptions.TANH)
        self.assertEqual(
            independent.recurrence.gate.stack.activation,
            ActivationOptions.SIGMOID,
        )

    def test_adaptive_generators_resolve_shared_and_independent_stacks(self):
        inherited = runtime_from_flat(
            {
                "adaptive_generator_stack_activation": ActivationOptions.RELU,
                "weight_generator_stack_activation": ActivationOptions.TANH,
            }
        )
        self.assertFalse(inherited.weight.generator_stack.independent)
        self.assertEqual(
            inherited.weight.generator_stack.stack.activation,
            ActivationOptions.RELU,
        )
        independent = runtime_from_flat(
            {
                "adaptive_generator_stack_activation": ActivationOptions.RELU,
                "weight_generator_stack_independent_flag": True,
                "weight_generator_stack_activation": ActivationOptions.TANH,
            }
        )
        self.assertTrue(independent.weight.generator_stack.independent)
        self.assertEqual(
            independent.weight.generator_stack.stack.activation,
            ActivationOptions.TANH,
        )

    def test_enabled_adaptive_options_require_implementations(self):
        for flag in (
            "weight_option_flag",
            "bias_option_flag",
            "diagonal_option_flag",
            "mask_option_flag",
        ):
            with self.subTest(flag=flag), self.assertRaises(ValueError):
                runtime_from_flat({flag: True})


class TestConstruction(unittest.TestCase):
    def test_default_and_typed_custom_runtime_build(self):
        default = LinearAdaptiveConfigBuilder().build()
        custom_runtime = replace(
            DEFAULT_RUNTIME,
            batch_size=3,
            hidden_dim=12,
            stack=replace(DEFAULT_RUNTIME.stack, hidden_dim=12),
        )
        custom = LinearAdaptiveConfigBuilder(runtime=custom_runtime).build()
        self.assertEqual(default.hidden_dim, DEFAULT_RUNTIME.hidden_dim)
        self.assertEqual(custom.batch_size, 3)
        self.assertEqual(custom.hidden_dim, 12)

    def test_builder_rejects_foreign_runtime_objects(self):
        with self.assertRaisesRegex(TypeError, "RuntimeOptions"):
            LinearAdaptiveConfigBuilder(runtime=object())

    def test_factories_do_not_import_or_consult_config(self):
        for module in (
            _adaptive_parameter_config_factory,
            _control_config_factory,
            _hidden_model_config_factory,
            _projection_config_factory,
        ):
            with self.subTest(module=module.__name__):
                tree = ast.parse(inspect.getsource(module))
                imported = {
                    node.module
                    for node in ast.walk(tree)
                    if isinstance(node, ast.ImportFrom) and node.module
                }
                self.assertNotIn(
                    "models.linears.linear_adaptive.config",
                    imported,
                )

    def test_controller_and_adaptive_configs_are_wired(self):
        runtime = runtime_from_flat(
            {
                "stack_gate_flag": True,
                "recurrent_flag": True,
                "recurrent_gate_flag": True,
                "recurrent_halting_flag": True,
                "weight_option_flag": True,
                "weight_option": DualModelDynamicWeightConfig,
            }
        )
        cfg = LinearAdaptiveConfigBuilder(runtime=runtime).build()
        recurrent = cfg.experiment_config.model_config
        self.assertIsInstance(recurrent, RecurrentLayerConfig)
        self.assertIsNotNone(recurrent.gate_config)
        self.assertIsNotNone(recurrent.halting_config)
        self.assertIsInstance(recurrent.block_config, LayerStackConfig)
        layer_model = recurrent.block_config.layer_config.layer_model_config
        self.assertIsInstance(layer_model, AdaptiveLinearLayerConfig)
        self.assertIsInstance(
            layer_model.adaptive_augmentation_config.weight_config,
            DualModelDynamicWeightConfig,
        )

    def test_input_and_output_projections_are_distinct(self):
        cfg = LinearAdaptiveConfigBuilder().build().experiment_config
        self.assertIsNot(cfg.input_model_config, cfg.output_model_config)
        self.assertNotEqual(
            cfg.input_model_config.activation,
            cfg.output_model_config.activation,
        )


class TestPresetsAndMetadata(unittest.TestCase):
    def test_every_preset_builds(self):
        presets = ExperimentPresets()
        for preset in ExperimentPreset:
            with self.subTest(preset=preset.name):
                self.assertEqual(len(presets.get_config(preset)), 1)

    def test_preset_identity_values_and_descriptions_are_stable(self):
        presets = ExperimentPresets()
        self.assertEqual(
            [preset.value for preset in ExperimentPreset],
            list(range(1, len(ExperimentPreset) + 1)),
        )
        for preset in ExperimentPreset:
            definition = presets.definition_for_preset(preset)
            self.assertIsInstance(definition, PresetDefinition)
            self.assertTrue(definition.description.strip())

    def test_metadata_groups_remain_declared(self):
        self.assertIn(
            dataset_options.DEFAULT_EXPERIMENT_TASK,
            dataset_options.DATASET_OPTIONS_BY_TASK,
        )
        self.assertIsInstance(monitor_options.MONITOR_OPTIONS, list)
        self.assertTrue(
            any(name.startswith("SEARCH_SPACE_") for name in vars(search_space))
        )


class TestModelBehavior(unittest.TestCase):
    @staticmethod
    def _fake_batch(dataset: type, batch_size: int = 2) -> torch.Tensor:
        return torch.randn(
            batch_size,
            dataset.num_channels,
            dataset.default_height,
            dataset.default_width,
        )

    def test_every_preset_forwards_one_batch(self):
        presets = ExperimentPresets()
        dataset = dataset_options.DATASET_OPTIONS_BY_TASK[
            dataset_options.DEFAULT_EXPERIMENT_TASK
        ][0]
        for preset in ExperimentPreset:
            with self.subTest(preset=preset.name):
                model = Model(presets.get_config(preset, dataset)[0])
                output = model(self._fake_batch(dataset))
                logits = output[0] if isinstance(output, tuple) else output
                self.assertEqual(logits.shape, (2, dataset.num_classes))

    def test_baseline_forwards_every_declared_dataset(self):
        presets = ExperimentPresets()
        for dataset in dataset_options.DATASET_OPTIONS_BY_TASK[
            dataset_options.DEFAULT_EXPERIMENT_TASK
        ]:
            with self.subTest(dataset=dataset.__name__):
                model = Model(presets.get_config(ExperimentPreset.BASELINE, dataset)[0])
                output = model(self._fake_batch(dataset))
                logits = output[0] if isinstance(output, tuple) else output
                self.assertEqual(logits.shape, (2, dataset.num_classes))

    def test_every_preset_trains_for_one_smoke_epoch(self):
        presets = ExperimentPresets()
        dataset = dataset_options.DATASET_OPTIONS_BY_TASK[
            dataset_options.DEFAULT_EXPERIMENT_TASK
        ][0]
        for preset in ExperimentPreset:
            with self.subTest(preset=preset.name):
                model = Model(presets.get_config(preset, dataset)[0])
                tiny_cpu_trainer().fit(
                    model,
                    datamodule=RandomImageClassificationDataModule(dataset),
                )


if __name__ == "__main__":
    unittest.main()
