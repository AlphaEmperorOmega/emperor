import torch
import unittest

from torch.nn import Sequential
from emperor.base.layer import Layer
from emperor.linears.utils.presets import LinearPresets
from emperor.linears.utils.stack import AdaptiveLinearLayerStack, LinearLayerStack
from emperor.augmentations.adaptive_parameters.options import (
    DynamicBiasOptions,
    DynamicDepthOptions,
    DynamicDiagonalOptions,
)
from emperor.linears.utils.config import LinearLayerConfig
from emperor.linears.utils.layers import (
    AdaptiveLinearLayer,
    LinearLayer,
)

from emperor.base.layer import LayerStackConfig
from emperor.augmentations.adaptive_parameters.config import (
    AdaptiveParameterAugmentationConfig,
)
from emperor.base.enums import ActivationOptions, LayerNormPositionOptions
from emperor.augmentations.adaptive_parameters.options import (
    DynamicWeightOptions,
    LinearMemoryOptions,
    LinearMemoryPositionOptions,
    LinearMemorySizeOptions,
    RowMaskOptions,
    WeightNormalizationOptions,
)


class TestLinearLayer(unittest.TestCase):
    def preset(
        self,
        input_dim: int = 12,
        output_dim: int = 6,
        bias_flag: bool = True,
        data_monitor=None,
        parameter_monitor=None,
    ) -> LinearLayerConfig:

        return LinearLayerConfig(
            input_dim=input_dim,
            output_dim=output_dim,
            bias_flag=bias_flag,
            data_monitor=data_monitor,
            parameter_monitor=parameter_monitor,
        )

    def test_init_with_different_configation_options(self):
        bias_options = [True, False]
        for bias_flag in bias_options:
            message = f"Test failed for the inputs: {bias_flag}"
            with self.subTest(i=message):
                c = self.preset(bias_flag=bias_flag)
                overrides = LinearLayerConfig(bias_flag=bias_flag)
                m = LinearLayer(c, overrides)

                self.assertEqual(m.input_dim, c.input_dim)
                self.assertEqual(m.output_dim, c.output_dim)
                self.assertIsInstance(m.weight_params, torch.Tensor)
                if bias_flag:
                    self.assertIsInstance(m.bias_params, torch.Tensor)
                else:
                    self.assertIsNone(m.bias_params)

    def test_forward(self):
        batch_size = 5
        bias_options = [True, False]
        input_params = output_params = [4, 8, 16]

        for input_dim in input_params:
            for output_dim in output_params:
                for bias_flag in bias_options:
                    message = f"Test failed for the options: {input_dim}, {output_dim}, {bias_flag}"
                    with self.subTest(i=message):
                        c = self.preset()
                        overrides = LinearLayerConfig(
                            input_dim=input_dim,
                            output_dim=output_dim,
                            bias_flag=bias_flag,
                        )
                        m = LinearLayer(c, overrides)

                        input_batch = torch.randn(batch_size, overrides.input_dim)
                        output = m.forward(input_batch)
                        expected_output_shape = (batch_size, overrides.output_dim)
                        self.assertEqual(output.shape, expected_output_shape)

    def test_gradients_flow_through_linear_layer(self):
        batch_size = 5
        input_params = [4, 8]
        output_params = [3, 6]
        for input_dim in input_params:
            for output_dim in output_params:
                for bias_flag in [True, False]:
                    with self.subTest(
                        input_dim=input_dim, output_dim=output_dim, bias_flag=bias_flag
                    ):
                        c = self.preset()
                        overrides = LinearLayerConfig(
                            input_dim=input_dim,
                            output_dim=output_dim,
                            bias_flag=bias_flag,
                        )
                        m = LinearLayer(c, overrides)

                        input_batch = torch.randn(
                            batch_size, overrides.input_dim, requires_grad=True
                        )
                        output = m.forward(input_batch)
                        output.sum().backward()

                        self.assertIsNotNone(m.weight_params.grad)
                        self.assertEqual(
                            m.weight_params.grad.shape, m.weight_params.shape
                        )

                        if bias_flag:
                            self.assertIsNotNone(m.bias_params.grad)
                            self.assertEqual(
                                m.bias_params.grad.shape, m.bias_params.shape
                            )
                        else:
                            self.assertIsNone(m.bias_params)


# class TestAdaptiveLinearLayer(unittest.TestCase):
#     def preset(
#         self,
#         input_dim: int = 12,
#         output_dim: int = 6,
#         bias_flag: bool = True,
#         layer_norm_position: LayerNormPositionOptions = LayerNormPositionOptions.DISABLED,
#         weight_option: DynamicWeightOptions = DynamicWeightOptions.DUAL_MODEL,
#         weight_normalization: WeightNormalizationOptions = WeightNormalizationOptions.CLAMP,
#         generator_depth: DynamicDepthOptions = DynamicDepthOptions.DISABLED,
#         diagonal_option: DynamicDiagonalOptions = DynamicDiagonalOptions.DISABLED,
#         bias_option: DynamicBiasOptions = DynamicBiasOptions.DISABLED,
#         row_mask_option: RowMaskOptions = RowMaskOptions.DISABLED,
#         memory_option: LinearMemoryOptions = LinearMemoryOptions.DISABLED,
#         memory_size_option: LinearMemorySizeOptions = LinearMemorySizeOptions.DISABLED,
#         memory_position_option: LinearMemoryPositionOptions = LinearMemoryPositionOptions.BEFORE_AFFINE,
#         weight_bank_size: int = 4,
#         stack_num_layers: int = 2,
#         stack_hidden_dim: int = 0,
#         stack_activation: ActivationOptions = ActivationOptions.RELU,
#         stack_residual_flag: bool = False,
#         stack_dropout_probability: float = 0.0,
#     ) -> LinearLayerConfig:
#
#         return LinearLayerConfig(
#             input_dim=input_dim,
#             output_dim=output_dim,
#             bias_flag=bias_flag,
#             data_monitor=None,
#             parameter_monitor=None,
#             override_config=AdaptiveParameterAugmentationConfig(
#                 input_dim=input_dim,
#                 output_dim=output_dim,
#                 weight_option=weight_option,
#                 weight_normalization=weight_normalization,
#                 generator_depth=generator_depth,
#                 diagonal_option=diagonal_option,
#                 bias_flag=bias_flag,
#                 bias_option=bias_option,
#                 weight_bank_size=weight_bank_size,
#                 row_mask_option=row_mask_option,
#                 memory_option=memory_option,
#                 memory_size_option=memory_size_option,
#                 memory_position_option=memory_position_option,
#                 override_config=LayerStackConfig(
#                     input_dim=input_dim,
#                     hidden_dim=stack_hidden_dim,
#                     output_dim=output_dim,
#                     num_layers=stack_num_layers,
#                     activation=stack_activation,
#                     layer_norm_position=layer_norm_position,
#                     residual_flag=stack_residual_flag,
#                     adaptive_computation_flag=False,
#                     dropout_probability=stack_dropout_probability,
#                     override_config=LinearLayerConfig(
#                         input_dim=input_dim,
#                         output_dim=output_dim,
#                         bias_flag=bias_flag,
#                         data_monitor=None,
#                         parameter_monitor=None,
#                         override_config=AdaptiveParameterAugmentationConfig(
#                             generator_depth=generator_depth,
#                         ),
#                     ),
#                 ),
#             ),
#         )
#
#     def test_init_with_different_configation_options(self):
#         bias_options = [True, False]
#
#         for bias_flag in bias_options:
#             for generators_depth in DynamicDepthOptions:
#                 for diagonal_option in DynamicDiagonalOptions:
#                     for bias_option in DynamicBiasOptions:
#                         message = f"Test failed for the options: {bias_flag}, {generators_depth}, {diagonal_option}, {bias_option}"
#                         with self.subTest(message=message):
#                             cfg = LinearPresets.adaptive_linear_layer_preset(
#                                 return_model_config_flag=True,
#                                 bias_flag=bias_flag,
#                                 generator_depth=generators_depth,
#                                 diagonal_option=diagonal_option,
#                                 bias_option=bias_option,
#                             )
#                             m = AdaptiveLinearLayer(cfg)
#
#                             self.assertEqual(m.input_dim, cfg.input_dim)
#                             self.assertEqual(m.output_dim, cfg.output_dim)
#                             self.assertIsInstance(m.weight_params, torch.Tensor)
#                             if bias_flag:
#                                 self.assertIsInstance(m.bias_params, torch.Tensor)
#                             else:
#                                 self.assertIsNone(m.bias_params)
#
#     def test_forward(self):
#         bias_options = [True, False]
#         input_params = output_params = [8, 16]
#
#         for bias_flag in bias_options:
#             for input_dim in input_params:
#                 for output_dim in output_params:
#                     for generators_depth in DynamicDepthOptions:
#                         for diagonal_option in DynamicDiagonalOptions:
#                             for bias_option in DynamicBiasOptions:
#                                 message = f"Test failed for options - Bias flag: {bias_flag}, Generator depth: {generators_depth}, Diagonal option: {diagonal_option}, Bias option: {bias_option}, Input dimension: {input_dim}, Output dimension: {output_dim}."
#                                 with self.subTest(message=message):
#                                     batch_size = 2
#                                     cfg = LinearPresets.adaptive_linear_layer_preset(
#                                         return_model_config_flag=True,
#                                         stack_num_layers=3,
#                                         batch_size=batch_size,
#                                         input_dim=input_dim,
#                                         output_dim=output_dim,
#                                         bias_flag=bias_flag,
#                                         generator_depth=generators_depth,
#                                         diagonal_option=diagonal_option,
#                                         bias_option=bias_option,
#                                     )
#
#                                     m = AdaptiveLinearLayer(cfg)
#                                     input_batch = torch.randn(batch_size, input_dim)
#                                     output = m.forward(input_batch)
#                                     expected_output_shape = (
#                                         batch_size,
#                                         output_dim,
#                                     )
#                                     self.assertEqual(
#                                         output.shape,
#                                         expected_output_shape,
#                                     )
#
#
# class TestLinearLayerBaseStack(unittest.TestCase):
#     def test_init_with_different_configation_options(self):
#         num_layer_options = [1, 2, 3]
#
#         for num_layers in num_layer_options:
#             message = f"Test failed for the inputs: {num_layers}"
#             with self.subTest(i=message):
#                 cfg = LinearPresets.base_linear_layer_stack_preset(
#                     return_model_config_flag=True,
#                     stack_num_layers=num_layers,
#                 )
#                 m = LinearLayerStack(cfg).build_model()
#
#                 if num_layers == 1:
#                     self.assertIsInstance(m, Layer)
#                 else:
#                     self.assertIsInstance(m, Sequential)
#
#     def test_gradients_flow_through_linear_layer_stack(self):
#         num_layer_options = [1, 2, 3]
#         for num_layers in num_layer_options:
#             with self.subTest(num_layers=num_layers):
#                 batch_size = 2
#                 input_dim = 8
#                 output_dim = 4
#                 cfg = LinearPresets.base_linear_layer_stack_preset(
#                     return_model_config_flag=True,
#                     stack_num_layers=num_layers,
#                     batch_size=batch_size,
#                     input_dim=input_dim,
#                     output_dim=output_dim,
#                 )
#                 m = LinearLayerStack(cfg).build_model()
#
#                 input_batch = torch.randn(batch_size, input_dim, requires_grad=True)
#                 output = m.forward(input_batch)
#                 output.sum().backward()
#
#                 grads = [p.grad for p in m.parameters() if p.requires_grad]
#                 non_none_grads = [g for g in grads if g is not None]
#                 self.assertTrue(len(non_none_grads) > 0)
#
#
# class TestLinearLayerAdaptiveStack(unittest.TestCase):
#     def test_init_with_different_configation_options(self):
#         num_layer_options = [1, 2, 3]
#
#         for num_layers in num_layer_options:
#             message = f"Test failed for the inputs: {num_layers}"
#             with self.subTest(i=message):
#                 cfg = LinearPresets.adaptive_linear_layer_stack_preset(
#                     return_model_config_flag=True,
#                     stack_num_layers=num_layers,
#                 )
#                 m = AdaptiveLinearLayerStack(cfg).build_model()
#
#                 if num_layers == 1:
#                     self.assertIsInstance(m, Layer)
#                 else:
#                     self.assertIsInstance(m, Sequential)
#
#     def test_gradients_flow_through_adaptive_linear_layer_stack(self):
#         num_layer_options = [1, 2, 3]
#         for num_layers in num_layer_options:
#             with self.subTest(num_layers=num_layers):
#                 batch_size = 2
#                 input_dim = 8
#                 output_dim = 4
#                 cfg = LinearPresets.adaptive_linear_layer_stack_preset(
#                     return_model_config_flag=True,
#                     stack_num_layers=num_layers,
#                     batch_size=batch_size,
#                     input_dim=input_dim,
#                     output_dim=output_dim,
#                 )
#
#                 m = AdaptiveLinearLayerStack(cfg).build_model()
#
#                 input_batch = torch.randn(batch_size, input_dim, requires_grad=True)
#                 output = m.forward(input_batch)
#                 output.sum().backward()
#
#                 grads = [p.grad for p in m.parameters() if p.requires_grad]
#                 non_none_grads = [g for g in grads if g is not None]
#                 self.assertTrue(len(non_none_grads) > 0)
