import torch
import unittest

from dataclasses import asdict
from torch.nn import LayerNorm, ModuleList
from Emperor.transformer.layer import TransformerConfig, TransformerEncoder
from docs.utils import default_unittest_config


class TestTransformerEncoder(unittest.TestCase):
    def setUp(self):
        self.rebuild_presets()

    def tearDown(self):
        self.cfg = None
        self.config = None
        self.model = None
        self.batch_size = None
        self.input_dim = None
        self.output_dim = None
        self.embedding_dim = None
        self.layer_norm_position = None
        self.layer_norm_dim = None
        self.causalattention = None

    def rebuild_presets(self, config: TransformerConfig | None = None):
        self.cfg = default_unittest_config()
        self.config = self.cfg.transformer_config
        if config is not None:
            for k in asdict(config):
                if hasattr(self.config, k) and getattr(config, k) is not None:
                    setattr(self.config, k, getattr(config, k))

        self.model = TransformerEncoder(self.cfg)

        self.batch_size = self.cfg.batch_size
        self.input_dim = self.cfg.input_dim
        self.embedding_dim = self.cfg.multi_head_attention_model_config.embedding_dim

        # Model attributes
        self.num_layers = self.config.num_layers
        self.source_sequence_length = self.config.source_sequence_length
        self.target_sequence_length = self.config.target_sequence_length
        self.layer_norm_dim = self.config.layer_norm_dim


class Test___init(TestTransformerEncoder):
    def test___init(self):
        self.assertEqual(self.model.num_layers, self.num_layers)
        self.assertEqual(self.model.source_sequence_length, self.source_sequence_length)
        self.assertEqual(self.model.target_sequence_length, self.target_sequence_length)
        self.assertEqual(self.model.layer_norm_dim, self.layer_norm_dim)
        self.assertIsInstance(self.model.layer_norm_module, LayerNorm)
        self.assertIsInstance(self.model.layers, ModuleList)
        self.assertEqual(len(self.model.layers), self.num_layers)


class Test_forward(TestTransformerEncoder):
    def test_ensure_input_passes_through_the_encoder(self):
        soruce_token_embeddings = torch.randn(
            self.source_sequence_length,
            self.batch_size,
            self.embedding_dim,
        )
        attention_mask = None
        source_key_padding_mask = None

        output = self.model(
            soruce_token_embeddings,
            attention_mask,
            source_key_padding_mask,
        )

        expected_output = (
            self.source_sequence_length,
            self.batch_size,
            self.embedding_dim,
        )

        output, loss = output

        self.assertEqual(output.shape, expected_output)
        self.assertIsInstance(loss, torch.Tensor)
