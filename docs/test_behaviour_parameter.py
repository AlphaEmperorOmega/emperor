import torch
import unittest
import torch.nn as nn

from emperor.base.utils import Module
from emperor.augmentations.utils.handlers.weight import DualModelWeightHandler
from emperor.linears.utils.presets import LinearPresets
from emperor.augmentations.options import DynamicDepthOptions
from emperor.augmentations.utils.handlers.parameter import (
    DepthMappingLayer,
    DepthMappingLayerStack,
)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from emperor.config import ModelConfig


class TestDepthMappingBehaviour(unittest.TestCase):
    def setUp(self):
        self.rebuild_presets()

    def tearDown(self):
        self.cfg = None
        self.config = None
        self.model = None
        self.batch_size = None
        self.num_heads = None
        self.input_dim = None
        self.output_dim = None

    def rebuild_presets(self, config: "ModelConfig | None" = None):
        self.cfg = (
            LinearPresets.adaptive_linear_layer_preset(
                return_model_config_flag=True,
            )
            if config is None
            else config
        )

        self.batch_size = self.cfg.batch_size
        self.generator_depth = DynamicDepthOptions.DEPTH_OF_TWO.value
        self.input_dim = self.cfg.input_dim
        self.output_dim = self.cfg.output_dim

        weight_shape = (self.input_dim, self.output_dim)
        self.weight_params = Module()._init_parameter_bank(weight_shape, nn.init.zeros_)


class TestDepthMappingLayer(TestDepthMappingBehaviour):
    def test_initial_layer_computation(self):
        input_tensor = torch.randn(
            self.batch_size, self.generator_depth, self.input_dim
        )
        cfg = LinearPresets.adaptive_linear_layer_preset(
            generator_depth=DynamicDepthOptions.DEPTH_OF_TWO
        )
        cfg = cfg.override_config.override_config.override_config
        model = DepthMappingLayer(cfg)
        output = model(input_tensor)
        expected_shape = (self.batch_size, self.generator_depth, self.output_dim)
        self.assertEqual(output.shape, expected_shape)
        for i in range(self.batch_size):
            for j in range(self.generator_depth):
                weight_slice = model.weight_params[j]
                bias_slice = model.bias_params[j]
                expected_output = (
                    torch.matmul(input_tensor[i, j], weight_slice) + bias_slice
                )
                self.assertTrue(
                    torch.allclose(
                        output[i, j].round(decimals=4),
                        expected_output.round(decimals=4),
                        atol=1e-6,
                        rtol=1e-5,
                    )
                )

    def test_error_is_thrown_for_zero_depth(self):
        cfg = LinearPresets.adaptive_linear_layer_preset(
            generator_depth=DynamicDepthOptions.DISABLED
        )
        cfg = cfg.override_config.override_config.override_config
        with self.assertRaises(ValueError) as context:
            model = DepthMappingLayer(cfg)


class TestDepthMappingLayerStack(TestDepthMappingBehaviour):
    def test_initial_layer_computation(self):
        input_tensor = torch.randn(self.batch_size, self.input_dim)
        cfg = LinearPresets.adaptive_linear_layer_preset(
            generator_depth=DynamicDepthOptions.DEPTH_OF_TWO
        )
        cfg = cfg.override_config
        model = DepthMappingLayerStack(cfg)
        output = model(input_tensor)
        expected_shape = (self.batch_size, self.generator_depth, self.output_dim)
        self.assertEqual(output.shape, expected_shape)


class TestDualModelWeightHandler(TestDepthMappingBehaviour):
    def test_initial_layer_computation(self):
        input_dims = [4, 8]
        for input_dim in input_dims:
            for output_dim in input_dims:
                message = f"Test failed for input_dim={input_dim} and output_dim={output_dim}."
                with self.subTest(message=message):
                    batch_size = 2
                    input_tensor = torch.randn(batch_size, input_dim)
                    weight_shape = (input_dim, output_dim)
                    self.weight_params = Module()._init_parameter_bank(
                        weight_shape, nn.init.zeros_
                    )
                    generators_depth = DynamicDepthOptions.DEPTH_OF_TWO
                    cfg = LinearPresets.adaptive_linear_layer_preset(
                        batch_size=2,
                        input_dim=input_dim,
                        output_dim=output_dim,
                        generator_depth=generators_depth,
                    )
                    cfg = cfg.override_config
                    model = DualModelWeightHandler(cfg)
                    output = model(self.weight_params, input_tensor)
                    expected_shape = (batch_size, input_dim, output_dim)
                    self.assertEqual(output.shape, expected_shape)
