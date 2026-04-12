from emperor.augmentations.adaptive_parameters.config import (
    AdaptiveParameterAugmentationConfig,
)
import torch
import unittest
import torch.nn as nn

from emperor.base.utils import Module
from emperor.linears.core.config import LinearLayerConfig
from emperor.augmentations.adaptive_parameters.utils.handlers.weight import (
    DualModelWeightHandler,
)
from emperor.augmentations.adaptive_parameters.options import DynamicDepthOptions
from emperor.augmentations.adaptive_parameters.utils.handlers.depth_mapper import (
    DepthMappingLayer,
    DepthMappingLayerConfig,
    DepthMappingLayerStack,
)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from emperor.config import ModelConfig


class TestDepthMappingAugmentation(unittest.TestCase):
    def preset(
        self,
        input_dim: int = 12,
        output_dim: int = 6,
        bias_flag: bool = True,
        generator_depth: DynamicDepthOptions = DynamicDepthOptions.DISABLED,
    ) -> DepthMappingLayerConfig:

        return DepthMappingLayerConfig(
            input_dim=input_dim,
            output_dim=output_dim,
            bias_flag=bias_flag,
            generator_depth=generator_depth,
        )

    def test_forward_across_depths(self):
        batch_size = 2
        input_dim = 12
        output_dim = 6
        bias_options = [True, False]
        valid_depths = [
            DynamicDepthOptions.DEPTH_OF_ONE,
            DynamicDepthOptions.DEPTH_OF_TWO,
            DynamicDepthOptions.DEPTH_OF_THREE,
        ]
        for depth in valid_depths:
            for bias_flag in bias_options:
                with self.subTest(depth=depth, bias_flag=bias_flag):
                    cfg = self.preset(
                        input_dim=input_dim,
                        output_dim=output_dim,
                        bias_flag=bias_flag,
                        generator_depth=depth,
                    )
                    model = DepthMappingLayer(cfg)

                    input_tensor = torch.randn(batch_size, depth.value, input_dim)
                    output = model(input_tensor)
                    expected_shape = (batch_size, depth.value, output_dim)
                    self.assertEqual(output.shape, expected_shape)

                    if bias_flag:
                        self.assertIsNotNone(model.bias_params)
                    else:
                        self.assertIsNone(model.bias_params)

                    for i in range(batch_size):
                        for j in range(depth.value):
                            weight_slice = model.weight_params[j]
                            expected_output = torch.matmul(
                                input_tensor[i, j], weight_slice
                            )
                            if bias_flag:
                                expected_output = expected_output + model.bias_params[j]
                            torch.testing.assert_close(output[i, j], expected_output)

    def test_build_creates_depth_mapping_layer(self):
        cfg = self.preset(generator_depth=DynamicDepthOptions.DEPTH_OF_ONE)
        model = cfg.build()
        self.assertIsInstance(model, DepthMappingLayer)

    def test_build_with_overrides(self):
        cfg = self.preset(generator_depth=DynamicDepthOptions.DEPTH_OF_ONE)
        overrides = self.preset(
            input_dim=8,
            output_dim=4,
            bias_flag=False,
            generator_depth=DynamicDepthOptions.DEPTH_OF_TWO,
        )
        model = cfg.build(overrides)
        self.assertIsInstance(model, DepthMappingLayer)
        self.assertEqual(model.input_dim, 8)
        self.assertEqual(model.output_dim, 4)
        self.assertFalse(model.bias_flag)
        self.assertEqual(model.generator_depth, DynamicDepthOptions.DEPTH_OF_TWO.value)

    def test_disabled_depth_raises_error(self):
        cfg = self.preset(generator_depth=DynamicDepthOptions.DISABLED)
        with self.assertRaises(ValueError):
            DepthMappingLayer(cfg)


class TestDepthMappingLayerStack(TestDepthMappingAugmentation):
    def preset(
        self,
        input_dim: int = 12,
        output_dim: int = 6,
        bias_flag: bool = True,
        generator_depth: DynamicDepthOptions = DynamicDepthOptions.DISABLED,
    ) -> AdaptiveParameterAugmentationConfig:

        return AdaptiveParameterAugmentationConfig(
            input_dim=input_dim,
            output_dim=output_dim,
            diagonal_option=diagonal_option,
            weight_config=None,
            bias_config=None,
            mask_config=None,
            memory_config=None,
            model_config=LinearLayerConfig(
                input_dim=input_dim,
                output_dim=output_dim,
                bias_flag=bias_flag,
                generator_depth=generator_depth,
            ),
        )

    def test_initial_layer_computation(self):
        input_tensor = torch.randn(self.batch_size, self.input_dim)
        cfg = self.preset(generator_depth=DynamicDepthOptions.DEPTH_OF_TWO)
        model = DepthMappingLayerStack(cfg)

        output = model(input_tensor)
        expected_shape = (self.batch_size, self.generator_depth, self.output_dim)
        self.assertEqual(output.shape, expected_shape)


# class TestDualModelWeightHandler(TestDepthMappingAugmentation):
#     def test_initial_layer_computation(self):
#         input_dims = [4, 8]
#         for input_dim in input_dims:
#             for output_dim in input_dims:
#                 message = f"Test failed for input_dim={input_dim} and output_dim={output_dim}."
#                 with self.subTest(message=message):
#                     batch_size = 2
#                     input_tensor = torch.randn(batch_size, input_dim)
#                     weight_shape = (input_dim, output_dim)
#                     self.weight_params = Module()._init_parameter_bank(
#                         weight_shape, nn.init.zeros_
#                     )
#                     generators_depth = DynamicDepthOptions.DEPTH_OF_TWO
#                     cfg = AdaptiveParameterAugmentationConfig(
#                         input_dim=input_dim,
#                         output_dim=output_dim,
#                         generator_depth=generators_depth,
#                     )
#                     model = DualModelWeightHandler(cfg)
#                     output = model(self.weight_params, input_tensor)
#                     expected_shape = (batch_size, input_dim, output_dim)
#                     self.assertEqual(output.shape, expected_shape)
