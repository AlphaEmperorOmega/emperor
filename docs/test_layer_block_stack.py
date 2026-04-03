from emperor.linears.utils import presets
import torch
import unittest

from dataclasses import asdict
from emperor.parametric.utils.presets import ParametricLayerPresets
from emperor.linears.utils.presets import LinearPresets
from docs.config import default_unittest_config
from emperor.base.enums import ActivationOptions, LastLayerBiasOptions
from emperor.linears.options import LinearLayerOptions
from torch.nn import Sequential
from emperor.base.layer import Layer, LayerStack, LayerStackConfig, LayerState
from emperor.linears.utils.config import LinearLayerConfig
from emperor.augmentations.adaptive_parameters.config import (
    AdaptiveParameterAugmentationConfig,
)
from emperor.base.enums import LayerNormPositionOptions
from emperor.augmentations.adaptive_parameters.options import DynamicDepthOptions

from emperor.halting.config import HaltingConfig
from emperor.halting.options import HaltingOptions, HaltingHiddenStateModeOptions


class TestLayerStack(unittest.TestCase):
    def preset(
        self,
        input_dim: int = 12,
        hidden_dim: int = 24,
        output_dim: int = 6,
        bias_flag: bool = True,
        layer_norm_position: LayerNormPositionOptions = LayerNormPositionOptions.NONE,
        generator_depth: DynamicDepthOptions = DynamicDepthOptions.DISABLED,
        stack_num_layers: int = 2,
        stack_activation: ActivationOptions = ActivationOptions.RELU,
        stack_residual_flag: bool = True,
        stack_dropout_probability: float = 0.2,
        shared_halting_flag: bool = False,
        last_layer_bias_option: LastLayerBiasOptions | None = None,
    ) -> "LayerStackConfig":

        gate_config = LayerStackConfig(
            model_type=LinearLayerOptions.BASE,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=stack_num_layers,
            activation=stack_activation,
            layer_norm_position=layer_norm_position,
            residual_flag=stack_residual_flag,
            adaptive_computation_flag=False,
            dropout_probability=stack_dropout_probability,
            halting_config=None,
            shared_halting_flag=False,
            last_layer_bias_option=last_layer_bias_option,
            override_config=LinearLayerConfig(
                input_dim=input_dim,
                output_dim=output_dim,
                bias_flag=bias_flag,
                data_monitor=None,
                parameter_monitor=None,
                override_config=AdaptiveParameterAugmentationConfig(
                    generator_depth=generator_depth,
                ),
            ),
        )

        halting_config = HaltingConfig(
            halting_option=HaltingOptions.SOFT_HALTING,
            input_dim=output_dim,
            threshold=0.99,
            halting_dropout=0.0,
            hidden_state_mode=HaltingHiddenStateModeOptions.RAW,
            override_config=LayerStackConfig(
                model_type=LinearLayerOptions.BASE,
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                num_layers=stack_num_layers,
                activation=stack_activation,
                layer_norm_position=layer_norm_position,
                residual_flag=stack_residual_flag,
                adaptive_computation_flag=False,
                dropout_probability=stack_dropout_probability,
                halting_config=None,
                shared_halting_flag=False,
                last_layer_bias_option=last_layer_bias_option,
                override_config=LinearLayerConfig(
                    input_dim=input_dim,
                    output_dim=output_dim,
                    bias_flag=bias_flag,
                    data_monitor=None,
                    parameter_monitor=None,
                    override_config=AdaptiveParameterAugmentationConfig(
                        generator_depth=generator_depth,
                    ),
                ),
            ),
        )

        return LayerStackConfig(
            model_type=LinearLayerOptions.BASE,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=stack_num_layers,
            activation=stack_activation,
            layer_norm_position=layer_norm_position,
            residual_flag=stack_residual_flag,
            adaptive_computation_flag=False,
            dropout_probability=stack_dropout_probability,
            halting_config=None,
            shared_halting_flag=shared_halting_flag,
            last_layer_bias_option=last_layer_bias_option,
            gate_config=gate_config,
            override_config=LinearLayerConfig(
                input_dim=input_dim,
                output_dim=output_dim,
                bias_flag=bias_flag,
                data_monitor=None,
                parameter_monitor=None,
                override_config=AdaptiveParameterAugmentationConfig(
                    generator_depth=generator_depth,
                ),
            ),
        )

    def test_init_stores_all_config_attributes(self):
        cfg = self.preset()
        model = LayerStack(cfg)

        self.assertIsInstance(model, LayerStack)

        self.assertEqual(model.model_type, cfg.model_type)
        self.assertEqual(model.activation, cfg.activation)
        self.assertEqual(model.residual_flag, cfg.residual_flag)
        self.assertEqual(model.adaptive_computation_flag, cfg.adaptive_computation_flag)
        self.assertEqual(model.dropout_probability, cfg.dropout_probability)
        self.assertEqual(model.layer_norm_position, cfg.layer_norm_position)
        self.assertEqual(model.halting_config, cfg.halting_config)
        self.assertEqual(model.shared_halting_flag, cfg.shared_halting_flag)
        self.assertEqual(model.input_dim, cfg.input_dim)
        self.assertEqual(model.hidden_dim, cfg.hidden_dim)
        self.assertEqual(model.output_dim, cfg.output_dim)
        self.assertEqual(model.num_layers, cfg.num_layers)
        self.assertEqual(model.last_layer_bias_option, cfg.last_layer_bias_option)

    def test_build_returns_correct_type_for_num_layers(self):
        num_layers_options = [1, 2, 3, 4]
        for num_layers in num_layers_options:
            with self.subTest(num_layers=num_layers):
                cfg = self.preset(stack_num_layers=num_layers)
                model = LayerStack(cfg).build()

                if num_layers == 1:
                    self.assertIsInstance(model, Layer)
                    self.assertIsInstance(model.gate_module, Layer)
                else:
                    self.assertIsInstance(model, Sequential)
                    for layer in model:
                        self.assertIsInstance(layer.gate_module, Sequential)

    def test_build_forward_pass_output_shape(self):
        batch_size = 4
        num_layers_options = [1, 2, 3]
        input_dims = [6, 12]
        output_dims = [6, 8]
        for num_layers in num_layers_options:
            for input_dim in input_dims:
                for output_dim in output_dims:
                    message = f"num_layers={num_layers}, input_dim={input_dim}, output_dim={output_dim}"
                    with self.subTest(msg=message):
                        cfg = self.preset(
                            stack_num_layers=num_layers,
                            input_dim=input_dim,
                            output_dim=output_dim,
                        )
                        model = LayerStack(cfg).build()
                        input = torch.randn(batch_size, input_dim)
                        state = LayerState(hidden=input)
                        output_state = model(state)

                        if num_layers == 1:
                            self.assertIsInstance(model.gate_module, Layer)
                        else:
                            for layer in model:
                                self.assertIsInstance(layer.gate_module, Sequential)

                        self.assertEqual(
                            output_state.hidden.shape, (batch_size, output_dim)
                        )


# class Test___resolve_model_type_overrides(TestLayerStack):
#     def test_if_config_is_updated_for_linear_model(self):
#         config = LayerStackConfig(
#             model_type=LinearLayerOptions.ADAPTIVE,
#         )
#         self.rebuild_presets(config)
#
#         input_dim = 8
#         output_dim = 16
#
#         updated_config = self.model._LayerStack__resolve_model_type_overrides(
#             input_dim, output_dim
#         )
#
#         self.assertEqual(updated_config.input_dim, input_dim)
#         self.assertEqual(updated_config.output_dim, output_dim)
#
#     def test_if_config_is_updated_for_parameter_generator_model(self):
#         config = ParametricLayerPresets.parametric_layer_stack_preset()
#
#         model = LayerStack(config)
#
#         input_dim = 8
#         output_dim = 16
#
#         updated_config = model._LayerStack__resolve_model_type_overrides(
#             input_dim, output_dim
#         )
#
#         self.assertEqual(updated_config.input_dim, input_dim)
#         self.assertEqual(updated_config.output_dim, output_dim)
#
#
# class Test___create_layer(TestLayerStack):
#     def test_if_all_model_types_are_computed(self):
#         config_types = [
#             LinearPresets.base_linear_layer_stack_preset,
#             LinearPresets.adaptive_linear_layer_stack_preset,
#         ]
#         flag_options = [False, True]
#         for config_type in config_types:
#             for residual_flag in flag_options:
#                 message = f"Failed for config type: {config_type.__name__}, residual_flag: {residual_flag}"
#                 with self.subTest(msg=message):
#                     try:
#                         c = config_type(
#                             input_dim=8,
#                             output_dim=8,
#                             residual_flag=residual_flag,
#                         )
#                     except Exception as e:
#                         c = config_type(
#                             input_dim=8,
#                             output_dim=8,
#                             stack_residual_flag=residual_flag,
#                         )
#
#                     m = LayerStack(c)
#                     batch_size = 4
#
#                     model = m._LayerStack__create_layer(
#                         c.input_dim, c.output_dim, c.residual_flag
#                     )
#
#                     input = torch.randn(batch_size, c.input_dim)
#                     output = model(input)
#                     if isinstance(output, tuple):
#                         output, _ = model(input)
#
#                     expected_output = (batch_size, c.output_dim)
#
#                     self.assertIsInstance(model, Layer)
#                     self.assertEqual(model.activation_function, ActivationOptions.RELU)
#                     self.assertEqual(model.layer_norm_dim, None)
#                     self.assertEqual(model.residual_connection_flag, c.residual_flag)
#                     self.assertEqual(
#                         model.is_adaptive_computation, c.adaptive_computation_flag
#                     )
#                     self.assertEqual(model.dropout_probability, c.dropout_probability)
#                     self.assertEqual(output.shape, expected_output)
#

# class Test___add_initial_layer(TestLayerStack):
#     def test_no_layer_is_added_when_num_layers_is_one(self):
#         types = (LinearLayerOptions, ParameterGeneratorTypes)
#         for type in types:
#             for layer_type in type:
#                 config = LayerStackConfig(
#                     input_dim=8,
#                     output_dim=16,
#                     num_layers=1,
#                     model_type=layer_type,
#                 )
#                 self.rebuild_presets(config)
#
#                 model_list = []
#                 adjustment = self.model._LayerStack__add_initial_layer(model_list)
#
#                 self.assertEqual(len(model_list), 0)
#                 self.assertEqual(
#                     adjustment, LayerStackAdjustments.SHARED_INPUT_OUTPUT_DIM
#                 )
#
#     def test_no_layer_is_added_when_num_layers_is_three_with_same_input_output_dim(
#         self,
#     ):
#         types = (LinearLayerOptions, ParameterGeneratorTypes)
#         for type in types:
#             for layer_type in type:
#                 config = LayerStackConfig(
#                     input_dim=16,
#                     output_dim=16,
#                     num_layers=1,
#                     model_type=layer_type,
#                 )
#                 self.rebuild_presets(config)
#
#                 model_list = []
#                 adjustment = self.model._LayerStack__add_initial_layer(model_list)
#
#                 self.assertEqual(len(model_list), 0)
#                 self.assertEqual(
#                     adjustment, LayerStackAdjustments.SHARED_INPUT_OUTPUT_DIM
#                 )
#
#     def test_initial_layer_is_added_with_multiple_layers_and_different_input_output_dim(
#         self,
#     ):
#         num_layers_array = [2, 3, 4]
#         types = (LinearLayerOptions, ParameterGeneratorTypes)
#         for type in types:
#             for layer_type in type:
#                 for num_layer in num_layers_array:
#                     config = LayerStackConfig(
#                         input_dim=8,
#                         output_dim=16,
#                         num_layers=num_layer,
#                         model_type=layer_type,
#                     )
#                     self.rebuild_presets(config)
#
#                     model_list = []
#                     adjustment = self.model._LayerStack__add_initial_layer(model_list)
#                     model = nn.Sequential(*model_list)
#
#                     input = torch.randn(self.batch_size, self.input_dim)
#                     output = model(input)
#                     if isinstance(output, tuple):
#                         output, _ = model(input)
#
#                     expected_output = (self.batch_size, self.hidden_dim)
#
#                     self.assertEqual(
#                         adjustment, LayerStackAdjustments.SEPARATE_INPUT_OUTPUT_DIM
#                     )
#                     self.assertEqual(len(model_list), 1)
#                     self.assertEqual(output.shape, expected_output)


# class Test___add_hidden_layers(TestLayerStack):
#     def test_no_layer_added_when_num_layers_is_one(self):
#         types = (LinearLayerOptions, ParameterGeneratorTypes)
#         for type in types:
#             for layer_type in type:
#                 config = LayerStackConfig(
#                     input_dim=8,
#                     output_dim=16,
#                     num_layers=1,
#                     model_type=layer_type,
#                 )
#                 self.rebuild_presets(config)
#
#                 model_list = []
#                 adjustment = LayerStackAdjustments.SHARED_INPUT_OUTPUT_DIM
#                 self.model._LayerStack__add_hidden_layers(model_list, adjustment)
#
#                 self.assertEqual(len(model_list), 0)
#
#     def test_no_layer_added_when_num_layers_is_two_with_initial_layer_added(self):
#         types = (LinearLayerOptions, ParameterGeneratorTypes)
#         for type in types:
#             for layer_type in type:
#                 config = LayerStackConfig(
#                     input_dim=8,
#                     output_dim=16,
#                     num_layers=2,
#                     model_type=layer_type,
#                 )
#                 self.rebuild_presets(config)
#
#                 model_list = []
#                 adjustment = LayerStackAdjustments.SEPARATE_INPUT_OUTPUT_DIM
#                 self.model._LayerStack__add_hidden_layers(model_list, adjustment)
#
#                 self.assertEqual(len(model_list), 0)
#
#     def test_hidden_layers_are_added_multiple_hidden_layers(self):
#         num_layers_array = [2, 3, 4]
#         types = (LinearLayerOptions, ParameterGeneratorTypes)
#         for type in types:
#             for layer_type in type:
#                 for num_layer in num_layers_array:
#                     config = LayerStackConfig(
#                         input_dim=8,
#                         output_dim=16,
#                         num_layers=num_layer,
#                         model_type=layer_type,
#                     )
#                     self.rebuild_presets(config)
#
#                     model_list = []
#
#                     adjustment = LayerStackAdjustments.SHARED_INPUT_OUTPUT_DIM
#                     self.model._LayerStack__add_hidden_layers(model_list, adjustment)
#                     model = nn.Sequential(*model_list)
#
#                     input = torch.randn(self.batch_size, self.hidden_dim)
#                     output = model(input)
#                     if isinstance(output, tuple):
#                         output, _ = model(input)
#
#                     expected_output = (self.batch_size, self.hidden_dim)
#
#                     self.assertEqual(len(model_list), num_layer - adjustment.value)
#                     self.assertEqual(output.shape, expected_output)
#
#
# class Test___add_output_layer(TestLayerStack):
#     def test_ensure_input_layer_is_returned_when_num_layers_is_one(self):
#         types = (LinearLayerOptions, ParameterGeneratorTypes)
#         for type in types:
#             for layer_type in type:
#                 config = LayerStackConfig(
#                     input_dim=8,
#                     output_dim=16,
#                     num_layers=1,
#                     model_type=layer_type,
#                 )
#                 self.rebuild_presets(config)
#
#                 model_list = []
#                 self.model._LayerStack__add_output_layer(model_list)
#                 model = model_list[0]
#
#                 input = torch.randn(self.batch_size, self.input_dim)
#                 output = model(input)
#                 if isinstance(output, tuple):
#                     output, _ = model(input)
#
#                 expected_output = (self.batch_size, self.output_dim)
#
#                 self.assertEqual(len(model_list), 1)
#                 self.assertEqual(output.shape, expected_output)
#
#     def test_ensure_output_layer_is_added_when_multiple_layers_are_added(self):
#         types = (LinearLayerOptions, ParameterGeneratorTypes)
#         for type in types:
#             for layer_type in type:
#                 config = LayerStackConfig(
#                     input_dim=8,
#                     output_dim=16,
#                     num_layers=2,
#                     model_type=layer_type,
#                 )
#                 self.rebuild_presets(config)
#
#                 model_list = []
#                 self.model._LayerStack__add_output_layer(model_list)
#                 model = model_list[0]
#
#                 input = torch.randn(self.batch_size, self.hidden_dim)
#                 output = model(input)
#                 if isinstance(output, tuple):
#                     output, _ = model(input)
#
#                 expected_output = (self.batch_size, self.output_dim)
#
#                 self.assertEqual(len(model_list), 1)
#                 self.assertEqual(output.shape, expected_output)
#
#
# class Test_build_model(TestLayerStack):
#     def test_model_layer_block_returned_when_num_layers_is_one(self):
#         types = (LinearLayerOptions, ParameterGeneratorTypes)
#         for type in types:
#             for layer_type in type:
#                 config = LayerStackConfig(
#                     input_dim=8,
#                     output_dim=16,
#                     num_layers=1,
#                     model_type=layer_type,
#                 )
#                 self.rebuild_presets(config)
#
#                 model = self.model.build_model()
#
#                 input = torch.randn(self.batch_size, self.input_dim)
#                 output = model(input)
#                 if isinstance(output, tuple):
#                     output, _ = model(input)
#
#                 expected_output = (self.batch_size, self.output_dim)
#
#                 self.assertIsInstance(model, Layer)
#                 self.assertEqual(output.shape, expected_output)
#
#     def test_sequential_is_returned_when_num_layers_is_greater_than_one(self):
#         num_layers_array = [2, 3, 4]
#
#         types = (LinearLayerOptions, ParameterGeneratorTypes)
#         for type in types:
#             for layer_type in type:
#                 for num_layers in num_layers_array:
#                     config = LayerStackConfig(
#                         input_dim=8,
#                         output_dim=16,
#                         num_layers=num_layers,
#                         model_type=layer_type,
#                     )
#                     self.rebuild_presets(config)
#
#                     model = self.model.build_model()
#
#                     input = torch.randn(self.batch_size, self.input_dim)
#                     output = model(input)
#                     if isinstance(output, tuple):
#                         output, _ = model(input)
#
#                     expected_output = (self.batch_size, self.output_dim)
#
#                     self.assertIsInstance(model, nn.Sequential)
#
