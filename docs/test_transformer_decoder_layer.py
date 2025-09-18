import torch
import unittest

from dataclasses import asdict

from Emperor.attention.attention import MultiHeadAttention
from Emperor.feedForward.feed_forward import FeedForward
from Emperor.layers.utils.base import LayerBlock
from Emperor.transformer.layer import TransformerDecoderLayer, TransformerLayerConfig
from docs.utils import default_unittest_config


class TestTransformerDecoderLayer(unittest.TestCase):
    def setUp(self):
        self.rebuild_presets()

    def tearDown(self):
        self.cfg = None
        self.config = None
        self.model = None
        self.batch_size = None
        self.input_dim = None
        self.output_dim = None

    def rebuild_presets(self, config: TransformerLayerConfig | None = None):
        self.cfg = default_unittest_config()
        self.config = self.cfg.transformer_layer_config
        if config is not None:
            for k in asdict(config):
                if hasattr(self.config, k) and getattr(config, k) is not None:
                    setattr(self.config, k, getattr(config, k))

        self.model = TransformerDecoderLayer(self.cfg)

        self.batch_size = self.cfg.batch_size
        self.input_dim = self.cfg.input_dim
        self.target_sequence_length = (
            self.cfg.multi_head_attention_model_config.target_sequence_length
        )
        # self.cfg.multi_head_attention_model_config.embedding_dim
        self.embedding_dim = self.cfg.multi_head_attention_model_config.embedding_dim

        self.layer_norm_position = self.config.layer_norm_position
        self.layer_norm_dim = self.config.layer_norm_dim
        self.dropout_probability = self.config.dropout_probability


class Test___init(TestTransformerDecoderLayer):
    def test___init(self):
        self.assertEqual(self.model.layer_norm_position, self.layer_norm_position)
        self.assertEqual(self.model.layer_norm_dim, self.layer_norm_dim)
        self.assertEqual(self.model.dropout_probability, self.dropout_probability)
        self.assertIsInstance(self.model.self_attention_model, LayerBlock)
        self.assertIsInstance(self.model.cross_attention_model, LayerBlock)
        self.assertIsInstance(self.model.feed_forward_model, LayerBlock)
        self.assertIsInstance(self.model.self_attention_model.model, MultiHeadAttention)
        self.assertIsInstance(
            self.model.cross_attention_model.model, MultiHeadAttention
        )
        self.assertIsInstance(self.model.feed_forward_model.model, FeedForward)


class Test_forward(TestTransformerDecoderLayer):
    def test_ensure_input_passes_through_the_encoder(self):
        config = TransformerLayerConfig(
            layer_norm_dim=self.embedding_dim,
        )
        self.rebuild_presets(config)
        input = torch.randn(
            self.target_sequence_length,
            self.batch_size,
            self.embedding_dim,
        )
        memory_tensor = torch.randn(
            self.target_sequence_length,
            self.batch_size,
            self.embedding_dim,
        )
        output = self.model(
            input,
            memory_tensor,
        )
        expected_output = (
            self.target_sequence_length,
            self.batch_size,
            self.embedding_dim,
        )

        if isinstance(output, tuple):
            output, _ = output

        self.assertEqual(output.shape, expected_output)
