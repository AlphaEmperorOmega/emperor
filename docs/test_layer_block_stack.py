import torch
import unittest

from torch.nn import Sequential
from dataclasses import asdict
from emperor.parametric.utils.presets import ParametricLayerPresets
from emperor.linears.utils.presets import LinearPresets
from docs.config import default_unittest_config
from emperor.base.enums import ActivationOptions, LastLayerBiasOptions
from emperor.linears.options import LinearLayerOptions
from emperor.base.layer import (
    Layer,
    LayerConfig,
    LayerStack,
    LayerStackConfig,
    LayerState,
)
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
        layer_norm_position: LayerNormPositionOptions = LayerNormPositionOptions.DISABLED,
        generator_depth: DynamicDepthOptions = DynamicDepthOptions.DISABLED,
        stack_num_layers: int = 2,
        stack_activation: ActivationOptions = ActivationOptions.RELU,
        stack_residual_flag: bool = True,
        stack_dropout_probability: float = 0.2,
        shared_halting_flag: bool = False,
        last_layer_bias_option: LastLayerBiasOptions = LastLayerBiasOptions.DEFAULT,
        apply_output_pipeline_flag: bool = True,
    ) -> "LayerStackConfig":

        gate_config = LayerStackConfig(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=stack_num_layers,
            last_layer_bias_option=last_layer_bias_option,
            apply_output_pipeline_flag=apply_output_pipeline_flag,
            layer_config=LayerConfig(
                activation=stack_activation,
                layer_norm_position=layer_norm_position,
                residual_flag=stack_residual_flag,
                dropout_probability=stack_dropout_probability,
                halting_config=None,
                shared_halting_flag=False,
                gate_config=None,
                model_config=LinearLayerConfig(
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

        # halting_config = HaltingConfig(
        #     halting_option=HaltingOptions.SOFT_HALTING,
        #     input_dim=output_dim,
        #     threshold=0.99,
        #     halting_dropout=0.0,
        #     hidden_state_mode=HaltingHiddenStateModeOptions.RAW,
        #     model_config=LayerStackConfig(
        #     input_dim=input_dim,
        #     hidden_dim=hidden_dim,
        #     output_dim=output_dim,
        #     num_layers=stack_num_layers,
        #     layer_config=LayerConfig(
        #         activation=stack_activation,
        #         layer_norm_position=layer_norm_position,
        #         residual_flag=stack_residual_flag,
        #         dropout_probability=stack_dropout_probability,
        #         halting_config=None,
        #         shared_halting_flag=False,
        #         gate_config=gate_config,
        #         model_config=LinearLayerConfig(
        #             input_dim=input_dim,
        #             output_dim=output_dim,
        #             bias_flag=bias_flag,
        #             data_monitor=None,
        #             parameter_monitor=None,
        #             override_config=AdaptiveParameterAugmentationConfig(
        #                 generator_depth=generator_depth,
        #             ),
        #         ),
        #     ),
        # )

        return LayerStackConfig(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=stack_num_layers,
            last_layer_bias_option=last_layer_bias_option,
            apply_output_pipeline_flag=apply_output_pipeline_flag,
            layer_config=LayerConfig(
                input_dim=input_dim,
                output_dim=output_dim,
                activation=stack_activation,
                layer_norm_position=layer_norm_position,
                residual_flag=stack_residual_flag,
                dropout_probability=stack_dropout_probability,
                gate_config=gate_config,
                halting_config=None,
                shared_halting_flag=shared_halting_flag,
                model_config=LinearLayerConfig(
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

    def test_init_stores_all_config_attributes(self):
        cfg = self.preset()
        stack = LayerStack(cfg)

        self.assertIsInstance(stack, LayerStack)
        self.assertEqual(stack.input_dim, cfg.input_dim)
        self.assertEqual(stack.hidden_dim, cfg.hidden_dim)
        self.assertEqual(stack.output_dim, cfg.output_dim)
        self.assertEqual(stack.num_layers, cfg.num_layers)
        self.assertEqual(
            stack.apply_output_pipeline_flag, cfg.apply_output_pipeline_flag
        )
        self.assertEqual(stack.layer_config, cfg.layer_config)

        model = stack.build()
        layers = [model] if isinstance(model, Layer) else list(model)

        for i, layer in enumerate(layers):
            is_last_layer = i == len(layers) - 1
            with self.subTest(layer_index=i, is_last_layer=is_last_layer):
                self.assertIsInstance(layer, Layer)
                self.assertIsNotNone(layer.model)
                self.assertEqual(
                    layer.output_dim,
                    cfg.output_dim if is_last_layer else cfg.hidden_dim,
                )

                if is_last_layer and not cfg.apply_output_pipeline_flag:
                    self.assertEqual(
                        layer.activation_function, ActivationOptions.DISABLED
                    )
                    self.assertEqual(layer.dropout_probability, 0.0)
                    self.assertFalse(layer.residual_flag)
                else:
                    self.assertEqual(
                        layer.activation_function, cfg.layer_config.activation
                    )
                    self.assertEqual(
                        layer.dropout_probability, cfg.layer_config.dropout_probability
                    )

                if layer.gate_model is not None:
                    gate = layer.gate_model
                    gate_layers = [gate] if isinstance(gate, Layer) else list(gate)
                    for j, gate_layer in enumerate(gate_layers):
                        with self.subTest(gate_layer_index=j):
                            self.assertIsInstance(gate_layer, Layer)
                            self.assertIsNotNone(gate_layer.model)
                            self.assertIsNone(gate_layer.gate_config)
                            self.assertIsNone(gate_layer.halting_config)
                            self.assertFalse(gate_layer.shared_halting_flag)

    def test_build_returns_correct_type_for_num_layers(self):
        num_layers_options = [1, 2, 3, 4]
        for num_layers in num_layers_options:
            with self.subTest(num_layers=num_layers):
                cfg = self.preset(stack_num_layers=num_layers)
                model = LayerStack(cfg).build()

                if num_layers == 1:
                    self.assertIsInstance(model, Layer)
                    self.assertIsInstance(model.gate_model, Layer)
                else:
                    self.assertIsInstance(model, Sequential)
                    for layer in model:
                        self.assertIsInstance(layer.gate_model, Sequential)

    def test_layer_overrides_apply_correctly(self):
        cfg = self.preset(
            input_dim=8, output_dim=16, stack_dropout_probability=0.5
        ).layer_config
        overrides = LayerConfig(input_dim=12, output_dim=24)
        layer = Layer(cfg=cfg, overrides=overrides)

        self.assertEqual(layer.input_dim, 12)
        self.assertEqual(layer.output_dim, 24)
        self.assertEqual(layer.activation_function, ActivationOptions.RELU)
        self.assertEqual(layer.dropout_probability, 0.5)

    def test_stack_overrides_apply_correctly(self):
        cfg = self.preset(input_dim=8, hidden_dim=16, output_dim=4, stack_num_layers=3)
        overrides = LayerStackConfig(input_dim=12, hidden_dim=24, output_dim=6)
        stack = LayerStack(cfg, overrides)

        self.assertEqual(stack.input_dim, 12)
        self.assertEqual(stack.hidden_dim, 24)
        self.assertEqual(stack.output_dim, 6)
        self.assertEqual(stack.num_layers, 3)

    def test_last_layer_bias_option_applies_correctly(self):
        num_layers_options = [1, 2, 3]
        bias_options = [
            LastLayerBiasOptions.DEFAULT,
            LastLayerBiasOptions.DISABLED,
            LastLayerBiasOptions.ENABLED,
        ]
        bias_flags = [True, False]

        for num_layers in num_layers_options:
            for bias_option in bias_options:
                for bias_flag in bias_flags:
                    message = (
                        f"num_layers={num_layers}, "
                        f"bias_option={bias_option}, "
                        f"bias_flag={bias_flag}"
                    )
                    with self.subTest(msg=message):
                        cfg = self.preset(
                            stack_num_layers=num_layers,
                            bias_flag=bias_flag,
                            last_layer_bias_option=bias_option,
                        )
                        model = LayerStack(cfg).build()
                        layers = [model] if isinstance(model, Layer) else list(model)
                        last_layer = layers[-1]

                        match bias_option:
                            case LastLayerBiasOptions.DEFAULT:
                                if bias_flag:
                                    self.assertIsNotNone(last_layer.model.bias_params)
                                else:
                                    self.assertIsNone(last_layer.model.bias_params)
                            case LastLayerBiasOptions.DISABLED:
                                self.assertIsNone(last_layer.model.bias_params)
                            case LastLayerBiasOptions.ENABLED:
                                self.assertIsNotNone(last_layer.model.bias_params)

    def test_build_forward_pass_output_shape(self):
        batch_size = 4
        num_layers_options = [1, 2, 3]
        input_dims = [6, 12]
        output_dims = [6, 8]
        activations = [ActivationOptions.RELU, ActivationOptions.DISABLED]
        residual_flags = [True, False]
        dropout_probabilities = [0.0, 0.2]
        layer_norm_positions = [
            LayerNormPositionOptions.DISABLED,
            LayerNormPositionOptions.DEFAULT,
            LayerNormPositionOptions.BEFORE,
            LayerNormPositionOptions.AFTER,
        ]

        for num_layers in num_layers_options:
            for input_dim in input_dims:
                for output_dim in output_dims:
                    for activation in activations:
                        for residual_flag in residual_flags:
                            for dropout in dropout_probabilities:
                                for layer_norm in layer_norm_positions:
                                    message = (
                                        f"num_layers={num_layers}, "
                                        f"input_dim={input_dim}, "
                                        f"output_dim={output_dim}, "
                                        f"activation={activation}, "
                                        f"residual_flag={residual_flag}, "
                                        f"dropout={dropout}, "
                                        f"layer_norm={layer_norm}"
                                    )
                                    with self.subTest(msg=message):
                                        cfg = self.preset(
                                            stack_num_layers=num_layers,
                                            input_dim=input_dim,
                                            output_dim=output_dim,
                                            stack_activation=activation,
                                            stack_residual_flag=residual_flag,
                                            stack_dropout_probability=dropout,
                                            layer_norm_position=layer_norm,
                                        )
                                        model = LayerStack(cfg).build()
                                        x = torch.randn(batch_size, input_dim)
                                        state = LayerState(hidden=x)
                                        output_state = model(state)
                                        expected_shape = (batch_size, output_dim)

                                        self.assertEqual(
                                            output_state.hidden.shape,
                                            expected_shape,
                                        )

                                        layers = (
                                            [model]
                                            if isinstance(model, Layer)
                                            else list(model)
                                        )
                                        for layer in layers:
                                            if layer.input_dim != layer.output_dim:
                                                self.assertFalse(layer.residual_flag)

    # def test_no_layer_is_added_when_num_layers_is_one(self):
    #     types = (LinearLayerOptions, ParameterGeneratorTypes)
    #     for type in types:
    #         for layer_type in type:
    #             config = LayerStackConfig(
    #                 input_dim=8,
    #                 output_dim=16,
    #                 num_layers=1,
    #                 model_type=layer_type,
    #             )
    #             self.rebuild_presets(config)
    #
    #             model_list = []
    #             adjustment = self.model._LayerStack__add_initial_layer(model_list)
    #
    #             self.assertEqual(len(model_list), 0)
    #             self.assertEqual(
    #                 adjustment, LayerStackAdjustments.SHARED_INPUT_OUTPUT_DIM
    #             )

    # def test_no_layer_is_added_when_num_layers_is_three_with_same_input_output_dim(
    #     self,
    # ):
    #     types = (LinearLayerOptions, ParameterGeneratorTypes)
    #     for type in types:
    #         for layer_type in type:
    #             config = LayerStackConfig(
    #                 input_dim=16,
    #                 output_dim=16,
    #                 num_layers=1,
    #                 model_type=layer_type,
    #             )
    #             self.rebuild_presets(config)
    #
    #             model_list = []
    #             adjustment = self.model._LayerStack__add_initial_layer(model_list)
    #
    #             self.assertEqual(len(model_list), 0)
    #             self.assertEqual(
    #                 adjustment, LayerStackAdjustments.SHARED_INPUT_OUTPUT_DIM
    #             )


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
