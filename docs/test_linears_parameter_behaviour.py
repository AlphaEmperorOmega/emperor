import torch
import unittest
import torch.nn as nn

from dataclasses import asdict
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
        self.config = self.cfg.linear_layer_config
        if config is not None:
            for k in asdict(config):
                if hasattr(self.config, k) and getattr(config, k) is not None:
                    setattr(self.config, k, getattr(config, k))

        self.batch_size = self.cfg.batch_size
        self.generator_depth = self.config.generator_depth.value
        self.input_dim = self.cfg.input_dim
        self.output_dim = self.cfg.output_dim

        weight_shape = (self.input_dim, self.output_dim)
        self.weight_params = Module()._init_parameter_bank(weight_shape, nn.init.zeros_)


class TestDepthMappingLayer(TestDepthMappingBehaviour):
    def test_initial_layer_computation(self):
        input_tensor = torch.randn(
            self.batch_size, self.generator_depth, self.input_dim
        )
        model = DepthMappingLayer(self.config)
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
        config = LinearsConfigs.dynamic_preset(
            generator_depth=DynamicDepthOptions.DISABLED
        )
        with self.assertRaises(ValueError) as context:
            model = DepthMappingLayer(config)


class TestDepthMappingLayerStack(TestDepthMappingBehaviour):
    def test_initial_layer_computation(self):
        input_tensor = torch.randn(self.batch_size, self.input_dim)
        model = DepthMappingLayerStack(self.cfg)
        output = model(input_tensor)
        expected_shape = (self.batch_size, self.generator_depth, self.output_dim)
        self.assertEqual(output.shape, expected_shape)


class TestDynamicParametersBehaviour(TestDepthMappingBehaviour):
    def test_initial_layer_computation(self):
        input_tensor = torch.randn(self.batch_size, self.input_dim)
        model = DynamicParametersBehaviour(self.cfg)
        output = model(self.weight_params, input_tensor)
        expected_shape = (self.batch_size, self.input_dim, self.output_dim)
        self.assertEqual(output.shape, expected_shape)
