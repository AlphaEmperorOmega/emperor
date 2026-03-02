import torch
import unittest

from dataclasses import asdict

from emperor.neuron.neuron import Axons, AxonsConfig
from docs.config import default_unittest_config


class TestNeuronAxon(unittest.TestCase):
    def setUp(self):
        self.rebuild_presets()

    def tearDown(self):
        self.cfg = None
        self.config = None
        self.model = None
        self.batch_size = None
        self.embedding_dim = None
        self.model_type = None

    def rebuild_presets(self, config: AxonsConfig | None = None):
        self.cfg = default_unittest_config()
        self.config = self.cfg.neuron_nucleus_config
        if config is not None:
            for k in asdict(config):
                if hasattr(self.config, k) and getattr(config, k) is not None:
                    setattr(self.config, k, getattr(config, k))

        self.model = Axons(self.cfg)

        self.batch_size: int = self.cfg.batch_size
        self.input_dim: int = self.cfg.input_dim
        self.hidden_dim: int = self.cfg.hidden_dim
        self.output_dim: int = self.cfg.output_dim


class Test___init(TestNeuronAxon):
    def test___init(self):
        self.assertIsInstance(self.model, Axons)
