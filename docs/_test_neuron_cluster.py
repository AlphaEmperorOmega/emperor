import torch
import unittest
import torch.nn as nn

from dataclasses import asdict
from emperor.neuron.neuron import NeuronCluster, NeuronClusterConfig
from docs.config import default_unittest_config


class TestNeuronCluster(unittest.TestCase):
    def setUp(self):
        self.rebuild_presets()

    def tearDown(self):
        self.cfg = None
        self.config = None
        self.model = None
        self.batch_size = None
        self.embedding_dim = None

    def rebuild_presets(self, config: NeuronClusterConfig | None = None):
        self.cfg = default_unittest_config()
        self.cfg.router_model_config.input_dim = self.cfg.output_dim
        self.config = self.cfg.neuron_terminal_config
        if config is not None:
            for k in asdict(config):
                if hasattr(self.config, k) and getattr(config, k) is not None:
                    setattr(self.config, k, getattr(config, k))

        self.model = NeuronCluster(self.cfg)

        self.batch_size: int = self.cfg.batch_size
        self.input_dim: int = self.cfg.input_dim
        self.hidden_dim: int = self.cfg.hidden_dim
        self.output_dim: int = self.cfg.output_dim


class Test___init(TestNeuronCluster):
    def test_initialization(self):
        self.assertIsInstance(self.model.cluster, nn.ModuleDict)

        # TODO: finish unit tests once you finished the `SparseUniversalTransformer` implementation
