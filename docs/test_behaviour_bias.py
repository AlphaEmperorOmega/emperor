import torch
import unittest
import torch.nn as nn

from Emperor.base.utils import Module
from Emperor.config import ModelConfig
from Emperor.behaviours.utils.behaviours import DynamicBiasSelector
from Emperor.linears.utils.config import LinearPresets
from Emperor.behaviours.utils.enums import DynamicBiasOptions
from Emperor.behaviours.utils.handlers.bias import (
    AffineBiasTransformHandler,
    BiasGeneratorHandler,
    BiasHandlerAbstract,
    ElementwiseBiasHandler,
)


class TestLinearsBiasBehaviour(unittest.TestCase):
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

    def rebuild_presets(self, config: ModelConfig | None = None):
        self.cfg = LinearPresets.adaptive_linear_layer_preset() if config is None else config

        self.batch_size = self.cfg.batch_size
        self.input_dim = self.cfg.input_dim
        self.output_dim = self.cfg.output_dim

        bias_shape = (self.output_dim,)
        self.bias_params = Module()._init_parameter_bank(bias_shape, nn.init.zeros_)


class TestAffineBiasTransformHandler(TestLinearsBiasBehaviour):
    def test_forward(self):
        input_tensor = torch.randn(self.batch_size, self.input_dim)
        cfg = LinearPresets.adaptive_linear_layer_preset()
        cfg = cfg.linear_layer_config
        model = AffineBiasTransformHandler(cfg)
        output = model(self.bias_params, input_tensor)
        bias_shape = (self.batch_size, self.output_dim)
        self.assertEqual(output.shape, bias_shape)
        self.assertIsInstance(output, torch.Tensor)
        self.assertFalse(torch.all(output == 0))


class TestElementwiseBiasHandler(TestLinearsBiasBehaviour):
    def test_forward(self):
        input_tensor = torch.randn(self.batch_size, self.input_dim)
        cfg = LinearPresets.adaptive_linear_layer_preset()
        cfg = cfg.linear_layer_config
        model = ElementwiseBiasHandler(cfg)
        output = model(self.bias_params, input_tensor)
        bias_shape = (self.batch_size, self.output_dim)
        self.assertEqual(output.shape, bias_shape)
        self.assertIsInstance(output, torch.Tensor)
        self.assertFalse(torch.all(output == 0))


class TestBiasGeneratorHandler(TestLinearsBiasBehaviour):
    def test_forward(self):
        input_tensor = torch.randn(self.batch_size, self.input_dim)
        cfg = LinearPresets.adaptive_linear_layer_preset()
        cfg = cfg.linear_layer_config
        model = BiasGeneratorHandler(cfg)
        output = model(self.bias_params, input_tensor)
        bias_shape = (self.batch_size, self.output_dim)
        self.assertEqual(output.shape, bias_shape)
        self.assertIsInstance(output, torch.Tensor)
        self.assertFalse(torch.all(output == 0))


class TestDynamicBiasSelector(TestLinearsBiasBehaviour):
    def test_forward(self):
        for option in DynamicBiasOptions:
            message = f"Test failed for bias option: {option}"
            with self.subTest(message):
                cfg = LinearPresets.adaptive_linear_layer_preset(bias_option=option)
                cfg = cfg.linear_layer_config
                input_tensor = torch.randn(self.batch_size, self.input_dim)
                if option == DynamicBiasOptions.DISABLED:
                    with self.assertRaises(ValueError):
                        model = DynamicBiasSelector(cfg)
                else:
                    model = DynamicBiasSelector(cfg)
                    output = model(self.bias_params, input_tensor)
                    self.assertIsInstance(model.model, BiasHandlerAbstract)
                    self.assertIsInstance(output, torch.Tensor)
