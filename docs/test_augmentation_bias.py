import torch
import unittest
import torch.nn as nn

from emperor.base.utils import Module
from emperor.config import ModelConfig
from emperor.augmentations.adaptive_parameters.core.factory import DynamicBiasFactory
from emperor.linears.core.presets import LinearPresets
from emperor.augmentations.adaptive_parameters.options import DynamicBiasOptions
from emperor.augmentations.adaptive_parameters.core.handlers.bias import (
    AffineBiasTransformHandler,
    BiasGeneratorHandler,
    BiasHandlerAbstract,
    ElementwiseBiasHandler,
)


class TestLinearsAugmentationBias(unittest.TestCase):
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
        self.config = (
            LinearPresets.adaptive_linear_layer_preset(return_model_config_flag=True)
            if config is None
            else config
        )

        self.batch_size = self.config.batch_size
        self.input_dim = self.config.input_dim
        self.output_dim = self.config.output_dim

        bias_shape = (self.output_dim,)
        self.bias_params = Module()._init_parameter_bank(bias_shape, nn.init.zeros_)


class TestAffineBiasTransformHandler(TestLinearsAugmentationBias):
    def test_forward(self):
        input_tensor = torch.randn(self.batch_size, self.input_dim)
        cfg = LinearPresets.adaptive_linear_layer_preset()
        cfg = cfg.override_config
        model = AffineBiasTransformHandler(cfg)
        output = model(self.bias_params, input_tensor)
        bias_shape = (self.batch_size, self.output_dim)
        self.assertEqual(output.shape, bias_shape)
        self.assertIsInstance(output, torch.Tensor)
        self.assertFalse(torch.all(output == 0))


class TestElementwiseBiasHandler(TestLinearsAugmentationBias):
    def test_forward(self):
        input_tensor = torch.randn(self.batch_size, self.input_dim)
        cfg = LinearPresets.adaptive_linear_layer_preset()
        cfg = cfg.override_config
        model = ElementwiseBiasHandler(cfg)
        output = model(self.bias_params, input_tensor)
        bias_shape = (self.batch_size, self.output_dim)
        self.assertEqual(output.shape, bias_shape)
        self.assertIsInstance(output, torch.Tensor)
        self.assertFalse(torch.all(output == 0))


class TestBiasGeneratorHandler(TestLinearsAugmentationBias):
    def test_forward(self):
        input_tensor = torch.randn(self.batch_size, self.input_dim)
        cfg = LinearPresets.adaptive_linear_layer_preset()
        cfg = cfg.override_config
        model = BiasGeneratorHandler(cfg)
        output = model(self.bias_params, input_tensor)
        bias_shape = (self.batch_size, self.output_dim)
        self.assertEqual(output.shape, bias_shape)
        self.assertIsInstance(output, torch.Tensor)
        self.assertFalse(torch.all(output == 0))


class TestDynamicBiasFactory(TestLinearsAugmentationBias):
    def test_build(self):
        for option in DynamicBiasOptions:
            message = f"Test failed for bias option: {option}"
            with self.subTest(message):
                cfg = LinearPresets.adaptive_linear_layer_preset(bias_option=option)
                cfg = cfg.override_config
                input_tensor = torch.randn(self.batch_size, self.input_dim)
                if option == DynamicBiasOptions.DISABLED:
                    with self.assertRaises(ValueError):
                        DynamicBiasFactory(cfg).build()
                else:
                    handler = DynamicBiasFactory(cfg).build()
                    output = handler(self.bias_params, input_tensor)
                    self.assertIsInstance(handler, BiasHandlerAbstract)
                    self.assertIsInstance(output, torch.Tensor)
