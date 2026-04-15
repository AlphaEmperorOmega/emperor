import torch
import unittest

from emperor.base.utils import Module
from emperor.base.layer.config import LayerConfig, LayerStackConfig
from emperor.base.enums import (
    ActivationOptions,
    LastLayerBiasOptions,
    LayerNormPositionOptions,
)
from emperor.linears.core.config import LinearLayerConfig
from emperor.augmentations.adaptive_parameters.options import (
    DynamicDepthOptions,
    DynamicWeightOptions,
    WeightNormalizationOptions,
    WeightNormalizationPositionOptions,
)
from emperor.augmentations.adaptive_parameters.core.handlers.depth_mapper import (
    DepthMappingHandlerConfig,
)
from emperor.augmentations.adaptive_parameters.core.factory import DynamicWeightFactory
from emperor.augmentations.adaptive_parameters.core.handlers.weight import (
    DualModelWeightHandler,
    HypernetworkWeightHandler,
    LowRankWeightHandler,
    SingleModelWeightHandler,
    WeightedBankWeightHandler,
    WeightHandlerAbstract,
    WeightMaskHandler,
)


class TestWeightHandlerForward(unittest.TestCase):
    def preset(
        self,
        dim: int = 12,
        bias_flag: bool = True,
        generator_depth: DynamicDepthOptions = DynamicDepthOptions.DEPTH_OF_TWO,
        stack_num_layers: int = 1,
        stack_activation: ActivationOptions = ActivationOptions.RELU,
        stack_residual_flag: bool = False,
        stack_dropout_probability: float = 0.0,
        last_layer_bias_option: LastLayerBiasOptions = LastLayerBiasOptions.DEFAULT,
        apply_output_pipeline_flag: bool = True,
        weight_normalization: WeightNormalizationOptions = WeightNormalizationOptions.L2_SCALE,
        weight_normalization_position: WeightNormalizationPositionOptions = WeightNormalizationPositionOptions.BEFORE_OUTER_PRODUCT,
        generator_output_dim: int | None = None,
        bank_expansion_factor: int = 2,
    ) -> DepthMappingHandlerConfig:
        if generator_output_dim is None:
            generator_output_dim = dim

        cfg = DepthMappingHandlerConfig(
            input_dim=dim,
            output_dim=dim,
            generator_depth=generator_depth,
            model_config=LayerStackConfig(
                input_dim=dim,
                hidden_dim=dim,
                output_dim=generator_output_dim,
                num_layers=stack_num_layers,
                last_layer_bias_option=last_layer_bias_option,
                apply_output_pipeline_flag=apply_output_pipeline_flag,
                layer_config=LayerConfig(
                    input_dim=dim,
                    output_dim=generator_output_dim,
                    activation=stack_activation,
                    layer_norm_position=LayerNormPositionOptions.DISABLED,
                    residual_flag=stack_residual_flag,
                    dropout_probability=stack_dropout_probability,
                    gate_config=None,
                    halting_config=None,
                    shared_halting_flag=False,
                    layer_model_config=LinearLayerConfig(
                        input_dim=dim,
                        output_dim=generator_output_dim,
                        bias_flag=bias_flag,
                    ),
                ),
            ),
        )
        cfg.weight_normalization = weight_normalization
        cfg.weight_normalization_position = weight_normalization_position
        cfg.weight_bank_expansion_factor = bank_expansion_factor
        return cfg

    def test_single_model_handler_forward(self):
        batch_size = 2
        dim = 12
        cfg = self.preset(dim=dim)
        weight_params = Module()._init_parameter_bank((dim, dim))
        input_tensor = torch.randn(batch_size, dim)
        model = SingleModelWeightHandler(cfg)
        output = model(weight_params, input_tensor)
        self.assertEqual(output.shape, (batch_size, dim, dim))
        self.assertIsInstance(output, torch.Tensor)
        self.assertFalse(torch.all(output == 0))


#     def test_dual_model_handler_forward(self):
#         batch_size = 2
#         dim = 12
#         cfg = self.preset(dim=dim)
#         weight_params = Module()._init_parameter_bank((dim, dim))
#         input_tensor = torch.randn(batch_size, dim)
#         model = DualModelWeightHandler(cfg)
#         output = model(weight_params, input_tensor)
#         self.assertEqual(output.shape, (batch_size, dim, dim))
#         self.assertIsInstance(output, torch.Tensor)
#         self.assertFalse(torch.all(output == 0))
# #
#     def test_low_rank_handler_forward(self):
#         batch_size = 2
#         dim = 12
#         cfg = self.preset(dim=dim)
#         weight_params = Module()._init_parameter_bank((dim, dim))
#         input_tensor = torch.randn(batch_size, dim)
#         model = LowRankWeightHandler(cfg)
#         output = model(weight_params, input_tensor)
#         self.assertEqual(output.shape, (batch_size, dim, dim))
#         self.assertIsInstance(output, torch.Tensor)
#         self.assertFalse(torch.all(output == 0))
#
#     def test_weight_mask_handler_forward(self):
#         batch_size = 2
#         dim = 12
#         cfg = self.preset(dim=dim)
#         weight_params = Module()._init_parameter_bank((dim, dim))
#         input_tensor = torch.randn(batch_size, dim)
#         model = WeightMaskHandler(cfg)
#         output = model(weight_params, input_tensor)
#         self.assertEqual(output.shape, (batch_size, dim, dim))
#         self.assertIsInstance(output, torch.Tensor)
#         self.assertFalse(torch.all(output == 0))
#
#     def test_hypernetwork_handler_forward(self):
#         batch_size = 2
#         dim = 6
#         cfg = self.preset(dim=dim, generator_output_dim=dim * dim)
#         weight_params = Module()._init_parameter_bank((dim, dim))
#         input_tensor = torch.randn(batch_size, dim)
#         model = HypernetworkWeightHandler(cfg)
#         output = model(weight_params, input_tensor)
#         self.assertEqual(output.shape, (batch_size, dim, dim))
#         self.assertIsInstance(output, torch.Tensor)
#         self.assertFalse(torch.all(output == 0))
#
#     def test_weighted_bank_handler_forward(self):
#         batch_size = 2
#         dim = 12
#         factor = 2
#         cfg = self.preset(
#             dim=dim,
#             generator_depth=DynamicDepthOptions.DEPTH_OF_ONE,
#             generator_output_dim=dim,
#             bank_expansion_factor=factor,
#         )
#         weight_params = Module()._init_parameter_bank((dim, dim))
#         input_tensor = torch.randn(batch_size, dim)
#         model = WeightedBankWeightHandler(cfg)
#         output = model(weight_params, input_tensor)
#         self.assertEqual(output.shape, (batch_size, dim, dim))
#         self.assertIsInstance(output, torch.Tensor)
#         self.assertFalse(torch.all(output == 0))
#
#     def test_normalization_position_after_outer_product(self):
#         batch_size = 2
#         dim = 12
#         valid_normalizations = [
#             WeightNormalizationOptions.L2_SCALE,
#             WeightNormalizationOptions.RMS,
#             WeightNormalizationOptions.CLAMP,
#             WeightNormalizationOptions.SOFT_CLAMP,
#             WeightNormalizationOptions.SIGMOID_SCALE,
#         ]
#         for normalization in valid_normalizations:
#             with self.subTest(normalization=normalization):
#                 cfg = self.preset(
#                     dim=dim,
#                     weight_normalization=normalization,
#                     weight_normalization_position=WeightNormalizationPositionOptions.AFTER_OUTER_PRODUCT,
#                 )
#                 weight_params = Module()._init_parameter_bank((dim, dim))
#                 input_tensor = torch.randn(batch_size, dim)
#                 model = DualModelWeightHandler(cfg)
#                 output = model(weight_params, input_tensor)
#                 self.assertEqual(output.shape, (batch_size, dim, dim))
#                 self.assertIsInstance(output, torch.Tensor)
#
#     def test_normalization_position_before_outer_product(self):
#         batch_size = 2
#         dim = 12
#         valid_normalizations = [
#             WeightNormalizationOptions.L2_SCALE,
#             WeightNormalizationOptions.RMS,
#             WeightNormalizationOptions.CLAMP,
#             WeightNormalizationOptions.SOFT_CLAMP,
#             WeightNormalizationOptions.SIGMOID_SCALE,
#         ]
#         for normalization in valid_normalizations:
#             with self.subTest(normalization=normalization):
#                 cfg = self.preset(
#                     dim=dim,
#                     weight_normalization=normalization,
#                     weight_normalization_position=WeightNormalizationPositionOptions.BEFORE_OUTER_PRODUCT,
#                 )
#                 weight_params = Module()._init_parameter_bank((dim, dim))
#                 input_tensor = torch.randn(batch_size, dim)
#                 model = DualModelWeightHandler(cfg)
#                 output = model(weight_params, input_tensor)
#                 self.assertEqual(output.shape, (batch_size, dim, dim))
#                 self.assertIsInstance(output, torch.Tensor)
#
#     def test_gradients_flow(self):
#         batch_size = 2
#         dim = 12
#         depth = DynamicDepthOptions.DEPTH_OF_TWO
#         cfg = self.preset(dim=dim, generator_depth=depth)
#         weight_params = Module()._init_parameter_bank((dim, dim))
#         input_tensor = torch.randn(batch_size, dim, requires_grad=True)
#         model = DualModelWeightHandler(cfg)
#         output = model(weight_params, input_tensor)
#         output.sum().backward()
#         grads = [p.grad for p in model.parameters() if p.requires_grad]
#         non_none_grads = [g for g in grads if g is not None]
#         self.assertTrue(len(non_none_grads) > 0)
#
#
# class TestDynamicWeightFactory(TestWeightHandlerForward):
#     def test_build(self):
#         batch_size = 2
#         dim = 12
#         weight_params = Module()._init_parameter_bank((dim, dim))
#         input_tensor = torch.randn(batch_size, dim)
#
#         for option in DynamicWeightOptions:
#             with self.subTest(f"weight_option={option}"):
#                 if option == DynamicWeightOptions.DISABLED:
#                     cfg = self.preset(dim=dim)
#                     cfg.weight_option = option
#                     with self.assertRaises(ValueError):
#                         DynamicWeightFactory(cfg).build()
#                 elif option == DynamicWeightOptions.HYPERNETWORK:
#                     cfg = self.preset(dim=dim, generator_output_dim=dim * dim)
#                     cfg.weight_option = option
#                     handler = DynamicWeightFactory(cfg).build()
#                     output = handler(weight_params, input_tensor)
#                     self.assertIsInstance(handler, WeightHandlerAbstract)
#                     self.assertIsInstance(output, torch.Tensor)
#                 elif option == DynamicWeightOptions.WEIGHTED_BANK:
#                     cfg = self.preset(
#                         dim=dim,
#                         generator_depth=DynamicDepthOptions.DEPTH_OF_ONE,
#                         generator_output_dim=dim,
#                         bank_expansion_factor=2,
#                     )
#                     cfg.weight_option = option
#                     handler = DynamicWeightFactory(cfg).build()
#                     output = handler(weight_params, input_tensor)
#                     self.assertIsInstance(handler, WeightHandlerAbstract)
#                     self.assertIsInstance(output, torch.Tensor)
#                 else:
#                     cfg = self.preset(dim=dim)
#                     cfg.weight_option = option
#                     handler = DynamicWeightFactory(cfg).build()
#                     output = handler(weight_params, input_tensor)
#                     self.assertIsInstance(handler, WeightHandlerAbstract)
#                     self.assertIsInstance(output, torch.Tensor)
