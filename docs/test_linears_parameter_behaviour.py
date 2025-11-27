import torch
import unittest
import torch.nn as nn

from Emperor.base.utils import Module
from Emperor.linears.utils.behaviours import DynamicParametersBehaviour
from Emperor.linears.utils.config import LinearsConfigs
from Emperor.linears.utils.enums import DynamicDepthOptions
from Emperor.linears.utils.handlers.parameter import (
    DepthMappingLayer,
    DepthMappingLayerStack,
)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from Emperor.config import ModelConfig


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
        generators_depth = DynamicDepthOptions.DEPTH_OF_TWO
        self.cfg = (
            LinearsConfigs.dynamic_preset(generator_depth=generators_depth)
            if config is None
            else config
        )

        self.batch_size = self.cfg.batch_size
        self.generator_depth = self.cfg.linear_layer_config.generator_depth.value
        self.input_dim = self.cfg.input_dim
        self.output_dim = self.cfg.output_dim

        weight_shape = (self.input_dim, self.output_dim)
        self.weight_params = Module()._init_parameter_bank(weight_shape, nn.init.zeros_)


class TestDepthMappingLayer(TestDepthMappingBehaviour):
    def test_initial_layer_computation(self):
        input_tensor = torch.randn(
            self.batch_size, self.generator_depth, self.input_dim
        )
        cfg = LinearsConfigs.dynamic_preset(
            generator_depth=DynamicDepthOptions.DEPTH_OF_TWO
        )
        cfg = cfg.linear_layer_config
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
                    torch.allclose(output[i, j], expected_output, atol=1e-6)
                )

    def test_error_is_thrown_for_zero_depth(self):
        cfg = LinearsConfigs.dynamic_preset(
            generator_depth=DynamicDepthOptions.DISABLED
        )
        cfg = cfg.linear_layer_config
        with self.assertRaises(ValueError) as context:
            model = DepthMappingLayer(cfg)


class TestDepthMappingLayerStack(TestDepthMappingBehaviour):
    def test_initial_layer_computation(self):
        input_tensor = torch.randn(self.batch_size, self.input_dim)
        cfg = LinearsConfigs.dynamic_preset(
            generator_depth=DynamicDepthOptions.DEPTH_OF_TWO
        )
        cfg = cfg.linear_layer_config
        model = DepthMappingLayerStack(cfg)
        output = model(input_tensor)
        expected_shape = (self.batch_size, self.generator_depth, self.output_dim)
        self.assertEqual(output.shape, expected_shape)


class TestDynamicParametersBehaviour(TestDepthMappingBehaviour):
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
                    cfg = LinearsConfigs.dynamic_preset(
                        batch_size=2,
                        input_dim=input_dim,
                        output_dim=output_dim,
                        generator_depth=generators_depth,
                    )
                    cfg = cfg.linear_layer_config
                    model = DynamicParametersBehaviour(cfg)
                    output = model(self.weight_params, input_tensor)
                    expected_shape = (batch_size, input_dim, output_dim)
                    self.assertEqual(output.shape, expected_shape)
