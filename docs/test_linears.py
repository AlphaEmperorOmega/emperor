from typing import ParamSpec
from emperor.base.layer.config import LayerConfig
import torch
import unittest

from torch.nn import Sequential
from emperor.base.layer import Layer
from emperor.halting.config import StickBreakingConfig
from emperor.halting.options import HaltingHiddenStateModeOptions
from emperor.base.layer import LayerStack
from emperor.augmentations.adaptive_parameters.options import (
    DynamicBiasOptions,
    DynamicDepthOptions,
    DynamicDiagonalOptions,
    MaskDimensionOptions,
)
from emperor.linears.core.config import AdaptiveLinearLayerConfig, LinearLayerConfig
from emperor.linears.core.layers import (
    AdaptiveLinearLayer,
    LinearLayer,
)

from emperor.base.layer import LayerStackConfig
from emperor.augmentations.adaptive_parameters.config import (
    AdaptiveParameterAugmentationConfig,
)
from emperor.base.enums import (
    ActivationOptions,
    LastLayerBiasOptions,
    LayerNormPositionOptions,
)
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
    ) -> LinearLayerConfig:

        return LinearLayerConfig(
            input_dim=input_dim,
            output_dim=output_dim,
            bias_flag=bias_flag,
        )

    def test_init_with_different_configation_options(self):
        bias_options = [True, False]
        for bias_flag in bias_options:
            message = f"Test failed for the inputs: {bias_flag}"
            with self.subTest(i=message):
                c = self.preset(bias_flag=bias_flag)
                m = LinearLayer(c)

                expected_weight_shape = (c.input_dim, c.output_dim)
                expected_bias_shape = (c.output_dim,)
                self.assertEqual(m.input_dim, c.input_dim)
                self.assertEqual(m.output_dim, c.output_dim)
                self.assertEqual(m.bias_flag, bias_flag)
                self.assertIsInstance(m.weight_params, torch.Tensor)
                self.assertEqual(m.weight_params.shape, expected_weight_shape)
                if bias_flag:
                    self.assertIsInstance(m.bias_params, torch.Tensor)
                    self.assertEqual(m.bias_params.shape, expected_bias_shape)
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

    def test_output_matches_torch_linear(self):
        for bias_flag in [True, False]:
            with self.subTest(bias_flag=bias_flag):
                c = self.preset(input_dim=4, output_dim=3, bias_flag=bias_flag)
                m = LinearLayer(c)

                ref = torch.nn.Linear(4, 3, bias=bias_flag)
                with torch.no_grad():
                    ref.weight.copy_(m.weight_params.T)
                    if bias_flag:
                        ref.bias.copy_(m.bias_params)

                input_batch = torch.randn(2, 4)
                torch.testing.assert_close(m.forward(input_batch), ref(input_batch))

    def test_deterministic_output(self):
        c = self.preset(input_dim=4, output_dim=3, bias_flag=True)
        m = LinearLayer(c)

        input_batch = torch.randn(2, 4)
        output_1 = m.forward(input_batch)
        output_2 = m.forward(input_batch)
        torch.testing.assert_close(output_1, output_2)


class TestLinearLayerStack(unittest.TestCase):
    def preset(
        self,
        input_dim: int = 12,
        hidden_dim: int = 24,
        output_dim: int = 6,
        bias_flag: bool = True,
        layer_norm_position: LayerNormPositionOptions = LayerNormPositionOptions.DISABLED,
        stack_num_layers: int = 2,
        stack_activation: ActivationOptions = ActivationOptions.RELU,
        stack_residual_flag: bool = True,
        stack_dropout_probability: float = 0.2,
        shared_halting_flag: bool = False,
        last_layer_bias_option: LastLayerBiasOptions = LastLayerBiasOptions.DEFAULT,
        apply_output_pipeline_flag: bool = True,
        gate_config: "LayerStackConfig | None" = None,
    ) -> LayerStackConfig:

        if gate_config is None:
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
                    ),
                ),
            )

        halting_config = None
        if stack_num_layers > 1 and input_dim == hidden_dim == output_dim:
            halting_config = StickBreakingConfig(
                input_dim=output_dim,
                threshold=0.99,
                halting_dropout=0.0,
                hidden_state_mode=HaltingHiddenStateModeOptions.RAW,
                halting_gate_config=LayerStackConfig(
                    input_dim=output_dim,
                    hidden_dim=output_dim,
                    output_dim=2,
                    num_layers=stack_num_layers,
                    last_layer_bias_option=LastLayerBiasOptions.DISABLED,
                    apply_output_pipeline_flag=False,
                    layer_config=LayerConfig(
                        activation=ActivationOptions.DISABLED,
                        layer_norm_position=LayerNormPositionOptions.DISABLED,
                        residual_flag=stack_residual_flag,
                        dropout_probability=stack_dropout_probability,
                        halting_config=None,
                        shared_halting_flag=False,
                        gate_config=None,
                        model_config=LinearLayerConfig(
                            input_dim=output_dim,
                            output_dim=output_dim,
                            bias_flag=True,
                        ),
                    ),
                ),
            )

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
                halting_config=halting_config,
                shared_halting_flag=shared_halting_flag,
                model_config=LinearLayerConfig(
                    input_dim=input_dim,
                    output_dim=output_dim,
                    bias_flag=bias_flag,
                ),
            ),
        )

    def test_stack_layers_contain_linear_layer(self):
        num_layer_options = [1, 2, 3]

        for num_layers in num_layer_options:
            with self.subTest(num_layers=num_layers):
                cfg = self.preset(stack_num_layers=num_layers)
                m = LayerStack(cfg).build()

                layers = [m] if isinstance(m, Layer) else list(m)
                for i, layer in enumerate(layers):
                    with self.subTest(layer_index=i):
                        self.assertIsInstance(layer.model, LinearLayer)

    def test_gradients_flow_through_linear_layer_stack(self):
        num_layer_options = [1, 2, 3]
        for num_layers in num_layer_options:
            with self.subTest(num_layers=num_layers):
                batch_size = 2
                input_dim = 8
                output_dim = 4
                cfg = self.preset(
                    stack_num_layers=num_layers,
                    input_dim=input_dim,
                    output_dim=output_dim,
                )
                m = LayerStack(cfg).build()

                input_batch = torch.randn(batch_size, input_dim, requires_grad=True)
                output = Layer.forward_with_state(m, input_batch)
                output.sum().backward()

                grads = [p.grad for p in m.parameters() if p.requires_grad]
                non_none_grads = [g for g in grads if g is not None]
                self.assertTrue(len(non_none_grads) > 0)


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
#         weight_bank_expansion_factor: int = 4,
#         bias_bank_expansion_factor: int = 8,
#         mask_dimension_option: MaskDimensionOptions = MaskDimensionOptions.COLUMN,
#         last_layer_bias_option: LastLayerBiasOptions = LastLayerBiasOptions.DEFAULT,
#         stack_num_layers: int = 2,
#         stack_hidden_dim: int = 0,
#         stack_activation: ActivationOptions = ActivationOptions.RELU,
#         stack_residual_flag: bool = False,
#         stack_dropout_probability: float = 0.0,
#     ) -> LinearLayerConfig:
#
#         return AdaptiveLinearLayerConfig(
#             input_dim=input_dim,
#             output_dim=output_dim,
#             bias_flag=bias_flag,
#             data_monitor=None,
#             parameter_monitor=None,
#             adaptive_augmentation_config=AdaptiveParameterAugmentationConfig(
#                 input_dim=input_dim,
#                 output_dim=output_dim,
#                 weight_option=weight_option,
#                 weight_normalization=weight_normalization,
#                 generator_depth=generator_depth,
#                 diagonal_option=diagonal_option,
#                 bias_flag=bias_flag,
#                 bias_option=bias_option,
#                 weight_bank_expansion_factor=weight_bank_expansion_factor,
#                 bias_bank_expansion_factor=bias_bank_expansion_factor,
#                 row_mask_option=row_mask_option,
#                 mask_dimension_option=mask_dimension_option,
#                 memory_option=memory_option,
#                 memory_size_option=memory_size_option,
#                 memory_position_option=memory_position_option,
#                 model_config=LayerStackConfig(
#                     input_dim=input_dim,
#                     hidden_dim=stack_hidden_dim,
#                     output_dim=output_dim,
#                     num_layers=stack_num_layers,
#                     last_layer_bias_option=last_layer_bias_option,
#                     apply_output_pipeline_flag=False,
#                     layer_config=LayerConfig(
#                         input_dim=input_dim,
#                         output_dim=output_dim,
#                         activation=stack_activation,
#                         layer_norm_position=layer_norm_position,
#                         residual_flag=stack_residual_flag,
#                         dropout_probability=stack_dropout_probability,
#                         gate_config=None,
#                         halting_config=None,
#                         shared_halting_flag=False,
#                         model_config=LinearLayerConfig(
#                             input_dim=input_dim,
#                             output_dim=output_dim,
#                             bias_flag=bias_flag,
#                             data_monitor=None,
#                             parameter_monitor=None,
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
#                 m = AdaptiveLayerStack(cfg).build()
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
#                 m = AdaptiveLayerStack(cfg).build()
#
#                 input_batch = torch.randn(batch_size, input_dim, requires_grad=True)
#                 output = m.forward(input_batch)
#                 output.sum().backward()
#
#                 grads = [p.grad for p in m.parameters() if p.requires_grad]
#                 non_none_grads = [g for g in grads if g is not None]
#                 self.assertTrue(len(non_none_grads) > 0)
