import torch
import unittest

from dataclasses import asdict
from Emperor.attention.attention import MultiHeadAttention
from Emperor.feedForward.feed_forward import FeedForward
from Emperor.layers.utils.base import LayerBlock
from Emperor.layers.utils.enums import LinearLayerTypes
from Emperor.transformer.layer import TransformerEncoderLayer, TransformerLayerConfig
from docs.utils import default_unittest_config


class TestTransformerLayer(unittest.TestCase):
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

        self.model = TransformerEncoderLayer(self.cfg)

        self.model_type = self.config.model_type
        self.layer_norm_position = self.config.layer_norm_position
        self.layer_norm_dim = self.config.layer_norm_dim
        self.dropout_probability = self.config.dropout_probability


class Test___init(TestTransformerLayer):
    def test___init(self):
        self.assertEqual(self.model.model_type, self.model_type)
        self.assertEqual(self.model.layer_norm_position, self.layer_norm_position)
        self.assertEqual(self.model.layer_norm_dim, self.layer_norm_position)
        self.assertEqual(self.model.dropout_probability, self.dropout_probability)
        self.assertIsInstance(self.model.attention_model, LayerBlock)
        self.assertIsInstance(self.model.feed_forward_model, LayerBlock)
        self.assertIsInstance(self.model.attention_model.model, MultiHeadAttention)
        self.assertIsInstance(self.model.feed_forward_model.model, FeedForward)


# class Test_forward(TestTransformerLayer):
#     def test_ensure_the_feed_forward_model_processes_2D_input_batch(self):
#         input = torch.randn(self.batch_size, self.input_dim)
#         output = self.model(input)
#
#         expected_output = (self.batch_size, self.output_dim)
#
#         if isinstance(output, tuple):
#             output, _ = output
#
#         self.assertIsInstance(self.model.model[0].model, layer_type.value)
#         self.assertEqual(output.shape, expected_output)
