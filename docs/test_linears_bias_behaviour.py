import torch
import unittest
import torch.nn as nn

from dataclasses import asdict
from Emperor.base.utils import Module
from Emperor.config import ModelConfig
from Emperor.linears.utils.behaviours import DynamicBiasSelector
from Emperor.linears.utils.config import LinearsConfigs
from Emperor.linears.utils.enums import DynamicBiasOptions
from Emperor.linears.utils.handlers.bias import (
    AffineBiasTransformHandler,
    BiasGeneratorHandler,
    BiasHandlerAbstract,
    DefaultBiasHandler,
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
        self.cfg = LinearsConfigs.dynamic_preset() if config is None else config
        self.config = self.cfg.transformer_layer_config
        if config is not None:
            for k in asdict(config):
                if hasattr(self.config, k) and getattr(config, k) is not None:
                    setattr(self.config, k, getattr(config, k))

        self.batch_size = self.cfg.batch_size
        self.input_dim = self.cfg.input_dim
        self.output_dim = self.cfg.output_dim

        bias_shape = (self.output_dim,)
        self.bias_params = Module()._init_parameter_bank(bias_shape, nn.init.zeros_)
        self.model = self.get_model()

    def get_model(self):
        raise NotImplementedError("Subclasses must implement the 'get_model' method.")


class TestDefaultBiasHandler(TestLinearsBiasBehaviour):
    def get_model(self):
        return DefaultBiasHandler(self.cfg)

    def test_forward(self):
        input_tensor = torch.randn(self.batch_size, self.output_dim)
        output = self.model(self.bias_params, input_tensor)
        bias_shape = (self.output_dim,)
        self.assertEqual(output.shape, bias_shape)
        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(output.numel(), self.output_dim)
        self.assertTrue(torch.all(output == 0))


class TestAffineBiasTransformHandler(TestLinearsBiasBehaviour):
    def get_model(self):
        return AffineBiasTransformHandler(self.cfg)

    def test_forward(self):
        input_tensor = torch.randn(self.batch_size, self.input_dim)
        output = self.model(self.bias_params, input_tensor)
        bias_shape = (self.batch_size, self.output_dim)
        self.assertEqual(output.shape, bias_shape)
        self.assertIsInstance(output, torch.Tensor)
        self.assertFalse(torch.all(output == 0))


class TestElementwiseBiasHandler(TestLinearsBiasBehaviour):
    def get_model(self):
        return ElementwiseBiasHandler(self.cfg)

    def test_forward(self):
        input_tensor = torch.randn(self.batch_size, self.input_dim)
        output = self.model(self.bias_params, input_tensor)
        bias_shape = (self.batch_size, self.output_dim)
        self.assertEqual(output.shape, bias_shape)
        self.assertIsInstance(output, torch.Tensor)
        self.assertFalse(torch.all(output == 0))


class TestBiasGeneratorHandler(TestLinearsBiasBehaviour):
    def get_model(self):
        return BiasGeneratorHandler(self.cfg)

    def test_forward(self):
        input_tensor = torch.randn(self.batch_size, self.input_dim)
        output = self.model(self.bias_params, input_tensor)
        bias_shape = (self.batch_size, self.output_dim)
        self.assertEqual(output.shape, bias_shape)
        self.assertIsInstance(output, torch.Tensor)
        self.assertFalse(torch.all(output == 0))


class TestDynamicBiasSelector(TestLinearsBiasBehaviour):
    def get_model(self):
        return DynamicBiasSelector(self.cfg)

    def test_forward(self):
        for option in DynamicBiasOptions:
            message = f"Test failed for bias option: {option}"
            with self.subTest(message):
                overrides = LinearsConfigs.dynamic_preset(bias_option=option)
                self.rebuild_presets(overrides)
                input_tensor = torch.randn(self.batch_size, self.input_dim)
                output = self.model(self.bias_params, input_tensor)
                self.assertIsInstance(self.model.model, BiasHandlerAbstract)
                self.assertIsInstance(output, torch.Tensor)
