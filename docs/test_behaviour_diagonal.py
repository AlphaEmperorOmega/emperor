import torch
import torch.nn as nn
import unittest

from Emperor.base.utils import Module
from Emperor.config import ModelConfig
from Emperor.behaviours.utils.behaviours import DynamicDiagonalSelector
from Emperor.linears.utils.presets import LinearPresets
from Emperor.behaviours.utils.enums import DynamicDiagonalOptions
from Emperor.behaviours.utils.handlers.diagonal import (
    DiagonalHandlerAbstract,
    DiagonalHandler,
    AntiDiagonalHandler,
    DiagonalAndAntiDiagonalHandler,
)


class TestLinearsDiagonalBehaviour(unittest.TestCase):
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
        self.cfg = (
            LinearPresets.adaptive_linear_layer_preset(return_model_config_flag=True)
            if config is None
            else config
        )
        self.batch_size = self.cfg.batch_size
        self.input_dim = self.cfg.input_dim
        self.output_dim = self.cfg.output_dim

        self.weight_shape = (self.input_dim, self.output_dim)
        self.weight_params = Module()._init_parameter_bank(
            self.weight_shape, nn.init.zeros_
        )


class TestDiagonalHandlerHandler(TestLinearsDiagonalBehaviour):
    def test_forward(self):
        input_tensor = torch.randn(self.batch_size, self.input_dim)
        cfg = LinearPresets.adaptive_linear_layer_preset()
        cfg = cfg.override_config
        model = DiagonalHandler(cfg)
        output = model(self.weight_params, input_tensor)
        expected_weight_shape = (self.batch_size, self.input_dim, self.output_dim)
        self.assertEqual(output.shape, expected_weight_shape)
        self.assertIsInstance(output, torch.Tensor)


class TestAntiDiagonalHandler(TestLinearsDiagonalBehaviour):
    def test_forward(self):
        input_tensor = torch.randn(self.batch_size, self.input_dim)
        cfg = LinearPresets.adaptive_linear_layer_preset()
        cfg = cfg.override_config
        model = AntiDiagonalHandler(cfg)
        output = model(self.weight_params, input_tensor)
        expected_weight_shape = (self.batch_size, self.input_dim, self.output_dim)
        self.assertEqual(output.shape, expected_weight_shape)
        self.assertIsInstance(output, torch.Tensor)


class TestDiagonalAndAntiDiagonalHandler(TestLinearsDiagonalBehaviour):
    def test_forward(self):
        input_tensor = torch.randn(self.batch_size, self.input_dim)
        cfg = LinearPresets.adaptive_linear_layer_preset()
        cfg = cfg.override_config
        model = DiagonalAndAntiDiagonalHandler(cfg)
        output = model(self.weight_params, input_tensor)
        expected_weight_shape = (self.batch_size, self.input_dim, self.output_dim)
        self.assertEqual(output.shape, expected_weight_shape)
        self.assertIsInstance(output, torch.Tensor)


class TestDynamicDiagonalSelector(TestLinearsDiagonalBehaviour):
    def test_forward(self):
        for option in DynamicDiagonalOptions:
            message = f"Test failed for diagonal option: {option}"
            with self.subTest(message):
                cfg = LinearPresets.adaptive_linear_layer_preset(diagonal_option=option)
                cfg = cfg.override_config
                input_tensor = torch.randn(self.batch_size, self.input_dim)
                if option == DynamicDiagonalOptions.DISABLED:
                    with self.assertRaises(ValueError):
                        model = DynamicDiagonalSelector(cfg)
                else:
                    model = DynamicDiagonalSelector(cfg)
                    output = model(self.weight_params, input_tensor)
                    self.assertIsInstance(model.model, DiagonalHandlerAbstract)
                    self.assertIsInstance(output, torch.Tensor)
