import torch
import unittest

from dataclasses import asdict

from Emperor.layers.utils.enums import (
    LayerTypes,
    LinearLayerTypes,
    ParameterGeneratorTypes,
)
from Emperor.neuron.neuron import Nucleus, NucleusConfig
from docs.config import default_unittest_config


class TestNeuronNucleus(unittest.TestCase):
    def setUp(self):
        self.rebuild_presets()

    def tearDown(self):
        self.cfg = None
        self.config = None
        self.model = None
        self.batch_size = None
        self.embedding_dim = None
        self.model_type = None

    def rebuild_presets(self, config: NucleusConfig | None = None):
        self.cfg = default_unittest_config()
        self.config = self.cfg.neuron_nucleus_config
        if config is not None:
            for k in asdict(config):
                if hasattr(self.config, k) and getattr(config, k) is not None:
                    setattr(self.config, k, getattr(config, k))

        self.model = Nucleus(self.cfg)

        self.batch_size: int = self.cfg.batch_size
        self.input_dim: int = self.cfg.input_dim
        self.hidden_dim: int = self.cfg.hidden_dim
        self.output_dim: int = self.cfg.output_dim


class Test___init(TestNeuronNucleus):
    def test___init(self):
        for layer_type in LayerTypes:
            message = f"Testing Nucleus with model_type={layer_type}"
            with self.subTest(msg=message):
                config = NucleusConfig(model_type=layer_type)
                self.rebuild_presets(config=config)
                self.assertEqual(self.model.model_type, self.config.model_type)
                self.assertIsInstance(
                    self.model.processing_unit, self.config.model_type.value
                )


class Test___create_model(TestNeuronNucleus):
    def test___create_model(self):
        for layer_type in LayerTypes:
            message = f"__create_model method with model_type={layer_type}"
            with self.subTest(msg=message):
                config = NucleusConfig(model_type=layer_type)
                self.rebuild_presets(config=config)
                created_model = self.model._Nucleus__create_model(self.cfg)
                self.assertIsInstance(created_model, layer_type.value)


class Test_forward(TestNeuronNucleus):
    def test_forward_with_linear_layer_types(self):
        for layer_type in LinearLayerTypes:
            message = f"forward method with model_type={layer_type}"
            with self.subTest(msg=message):
                config = NucleusConfig(model_type=layer_type)
                self.rebuild_presets(config=config)
                input_tensor = torch.randn(
                    self.batch_size,
                    self.input_dim,
                )
                output = self.model.forward(input_tensor)
                if isinstance(output, tuple):
                    output, _, _ = output
                expected_output_dim = (self.batch_size, self.output_dim)
                self.assertEqual(output.shape, expected_output_dim)

    def test_forward_with_parameter_generators(self):
        for layer_type in ParameterGeneratorTypes:
            message = f"forward method with model_type={layer_type}"
            with self.subTest(msg=message):
                config = NucleusConfig(model_type=layer_type)
                self.rebuild_presets(config=config)
                input_tensor = torch.randn(
                    self.batch_size,
                    self.hidden_dim,
                )
                output = self.model.forward(input_tensor)
                if isinstance(output, tuple):
                    output, _, _ = output
                expected_output_dim = (self.batch_size, self.output_dim)
                self.assertEqual(output.shape, expected_output_dim)
