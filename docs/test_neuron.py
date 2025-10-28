import torch
import unittest

from dataclasses import asdict

from torch.types import Tensor
from Emperor.neuron.neuron import Axons, Neuron, NeuronConfig, Nucleus, Terminal
from docs.config import default_unittest_config


class TestNeuron(unittest.TestCase):
    def setUp(self):
        self.rebuild_presets()

    def tearDown(self):
        self.cfg = None
        self.config = None
        self.model = None
        self.batch_size = None
        self.embedding_dim = None

    def rebuild_presets(self, config: NeuronConfig | None = None):
        self.cfg = default_unittest_config()
        self.cfg.router_model_config.input_dim = self.cfg.output_dim
        self.config = self.cfg.neuron_terminal_config
        if config is not None:
            for k in asdict(config):
                if hasattr(self.config, k) and getattr(config, k) is not None:
                    setattr(self.config, k, getattr(config, k))

        self.model = Neuron(self.cfg)

        self.batch_size: int = self.cfg.batch_size
        self.input_dim: int = self.cfg.input_dim
        self.hidden_dim: int = self.cfg.hidden_dim
        self.output_dim: int = self.cfg.output_dim


class Test___init(TestNeuron):
    def test_initialization(self):
        self.assertIsInstance(self.model.nucleus, Nucleus)
        self.assertIsInstance(self.model.axons, Axons)
        self.assertIsInstance(self.model.terminal, Terminal)


class Test_forward(TestNeuron):
    def test_method(self):
        input_batch = torch.randn(self.batch_size, self.input_dim)
        output, probabilities, selected_neurons = self.model(input_batch)

        self.assertIsInstance(output, Tensor)
        self.assertIsInstance(probabilities, Tensor)
        self.assertIsInstance(selected_neurons, Tensor)
