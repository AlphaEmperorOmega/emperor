import torch
import torch.nn as nn
import unittest

from dataclasses import asdict
from Emperor.base.utils import Module
from Emperor.config import ModelConfig
from Emperor.linears.utils.behaviours import DynamicDiagonalSelector
from Emperor.linears.utils.config import LinearsConfigs
from Emperor.linears.utils.enums import DynamicDiagonalOptions
from Emperor.linears.utils.handlers.diagonal import (
    DiagonalHandler,
    AntiDiagonalHandler,
    DiagonalHandlerAbstract,
    DefaultDiagonalHandler,
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
        self.cfg = LinearsConfigs.dynamic_preset() if config is None else config
        self.config = self.cfg.transformer_layer_config
        if config is not None:
            for k in asdict(config):
                if hasattr(self.config, k) and getattr(config, k) is not None:
                    setattr(self.config, k, getattr(config, k))

        self.batch_size = self.cfg.batch_size
        self.input_dim = self.cfg.input_dim
        self.output_dim = self.cfg.output_dim

        self.weight_shape = (self.input_dim, self.output_dim)
        self.weight_params = Module()._init_parameter_bank(
            self.weight_shape, nn.init.zeros_
        )
        self.model = self.get_model()

    def get_model(self):
        raise NotImplementedError("Subclasses must implement the 'get_model' method.")


class TestDefaultDiagonalHandler(TestLinearsDiagonalBehaviour):
    def get_model(self):
        return DefaultDiagonalHandler(self.cfg)

    def test_forward(self):
        input_tensor = torch.randn(self.batch_size, self.input_dim)
        output = self.model(self.weight_params, input_tensor)
        print(output.shape)
        self.assertEqual(output.shape, self.weight_shape)
        self.assertIsInstance(output, torch.Tensor)


class TestDiagonalHandlerHandler(TestLinearsDiagonalBehaviour):
    def get_model(self):
        return DiagonalHandler(self.cfg)

    def test_forward(self):
        input_tensor = torch.randn(self.batch_size, self.input_dim)
        output = self.model(self.weight_params, input_tensor)
        expected_weight_shape = (self.batch_size, self.input_dim, self.output_dim)
        self.assertEqual(output.shape, expected_weight_shape)
        self.assertIsInstance(output, torch.Tensor)


class TestAntiDiagonalHandler(TestLinearsDiagonalBehaviour):
    def get_model(self):
        return AntiDiagonalHandler(self.cfg)

    def test_forward(self):
        input_tensor = torch.randn(self.batch_size, self.input_dim)
        output = self.model(self.weight_params, input_tensor)
        expected_weight_shape = (self.batch_size, self.input_dim, self.output_dim)
        self.assertEqual(output.shape, expected_weight_shape)
        self.assertIsInstance(output, torch.Tensor)


class TestDiagonalAndAntiDiagonalHandler(TestLinearsDiagonalBehaviour):
    def get_model(self):
        return DiagonalAndAntiDiagonalHandler(self.cfg)

    def test_forward(self):
        input_tensor = torch.randn(self.batch_size, self.input_dim)
        output = self.model(self.weight_params, input_tensor)
        expected_weight_shape = (self.batch_size, self.input_dim, self.output_dim)
        self.assertEqual(output.shape, expected_weight_shape)
        self.assertIsInstance(output, torch.Tensor)


class TestDynamicBiasSelector(TestLinearsDiagonalBehaviour):
    def get_model(self):
        return DynamicDiagonalSelector(self.cfg)

    def test_forward(self):
        for option in DynamicDiagonalOptions:
            message = f"Test failed for diagonal option: {option}"
            with self.subTest(message):
                overrides = LinearsConfigs.dynamic_preset(diagonal_option=option)
                self.rebuild_presets(overrides)
                input_tensor = torch.randn(self.batch_size, self.input_dim)
                output = self.model(self.weight_params, input_tensor)
                self.assertIsInstance(self.model.model, DiagonalHandlerAbstract)
                self.assertIsInstance(output, torch.Tensor)
