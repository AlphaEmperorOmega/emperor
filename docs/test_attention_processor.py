from dataclasses import asdict
import unittest
import torch
from Emperor.attention.utils.utils import (
    AttentionProcessor,
    AttentionProcessorDefault,
    AttentionProcessorWithReturnedWeights,
    AttentionValidator,
)
from Emperor.attention.attention import MultiHeadAttention, MultiHeadAttentionConfig
from docs.utils import default_unittest_config


class TestAttentionProcessor(unittest.TestCase):
    def setUp(self):
        self.rebuild_presets()

    def tearDown(self):
        self.cfg = None
        self.config = None
        self.model = None
        self.batch_size = None
        self.embedding_dim = None
        self.target_sequence_length = None
        self.num_heads = None
        self.head_dim = None

    def rebuild_presets(self, config: MultiHeadAttentionConfig | None = None):
        self.cfg = default_unittest_config()
        self.config = self.cfg.multi_head_attention_model_config
        if config is not None:
            for k in asdict(config):
                if hasattr(self.config, k) and getattr(config, k) is not None:
                    setattr(self.config, k, getattr(config, k))

        model = MultiHeadAttention(self.cfg)
        output_model = model.output_model

        validator = AttentionValidator(self.config)
        self.model = AttentionProcessor(self.config, validator, output_model)

        self.batch_size = self.config.batch_size
        self.embedding_dim = self.config.embedding_dim
        self.target_sequence_length = self.config.target_sequence_length
        self.source_sequence_length = self.config.source_sequence_length
        self.num_heads = self.config.num_heads
        self.head_dim = self.embedding_dim // self.num_heads


class Test__init(TestAttentionProcessor):
    def test__init(self):
        self.assertIsInstance(self.model, AttentionProcessor)
        self.assertIsInstance(
            self.model.processor,
            (AttentionProcessorDefault, AttentionProcessorWithReturnedWeights),
        )


class Test____create_processor(TestAttentionProcessor):
    def test__return_attention_weights_flag__False(self):
        config = MultiHeadAttentionConfig(
            return_attention_weights_flag=False,
        )
        self.rebuild_presets(config)
        self.assertIsInstance(self.model.processor, AttentionProcessorDefault)

    def test__return_attention_weights_flag__True(self):
        config = MultiHeadAttentionConfig(
            return_attention_weights_flag=True,
        )
        self.rebuild_presets(config)
        self.assertIsInstance(
            self.model.processor, AttentionProcessorWithReturnedWeights
        )


class Test____compute_weighted_values_default(TestAttentionProcessor):
    def test__return_attention_weights_flag__True(self):
        config = MultiHeadAttentionConfig(
            source_sequence_length=32,
            target_sequence_length=32,
            return_attention_weights_flag=True,
        )
        self.rebuild_presets(config)

        query = torch.randn(
            self.batch_size * self.num_heads, self.source_sequence_length, self.head_dim
        )
        key = torch.randn(
            self.batch_size * self.num_heads, self.target_sequence_length, self.head_dim
        )
        value = torch.randn(
            self.batch_size * self.num_heads, self.target_sequence_length, self.head_dim
        )
        attention_mask = torch.randn(
            1, self.target_sequence_length, self.source_sequence_length
        )
        attention_mask = torch.where(
            attention_mask > 0, torch.tensor(float("-inf")), torch.tensor(0.0)
        )
        attention_mask = attention_mask.repeat(self.batch_size * self.num_heads, 1, 1)

        output_attention_output, output_attention_weights = (
            self.model.compute_attention(query, key, value, attention_mask)
        )

        self.assertIsInstance(output_attention_output, torch.Tensor)
        self.assertIsInstance(output_attention_weights, torch.Tensor)
        self.assertEqual(
            output_attention_output.shape,
            (self.target_sequence_length, self.batch_size, self.embedding_dim),
        )
        self.assertEqual(
            output_attention_weights.shape,
            (
                self.batch_size,
                self.num_heads,
                self.target_sequence_length,
                self.source_sequence_length,
            ),
        )

    def test__return_attention_weights_flag__True__average_attention_weights_flag__True(
        self,
    ):
        config = MultiHeadAttentionConfig(
            source_sequence_length=32,
            target_sequence_length=32,
            return_attention_weights_flag=True,
            average_attention_weights_flag=True,
        )
        self.rebuild_presets(config)

        query = torch.randn(
            self.batch_size * self.num_heads, self.source_sequence_length, self.head_dim
        )
        key = torch.randn(
            self.batch_size * self.num_heads, self.target_sequence_length, self.head_dim
        )
        value = torch.randn(
            self.batch_size * self.num_heads, self.target_sequence_length, self.head_dim
        )
        attention_mask = torch.randn(
            1, self.target_sequence_length, self.source_sequence_length
        )
        attention_mask = torch.where(
            attention_mask > 0, torch.tensor(float("-inf")), torch.tensor(0.0)
        )
        attention_mask = attention_mask.repeat(self.batch_size * self.num_heads, 1, 1)

        output_attention_output, output_attention_weights = (
            self.model.compute_attention(query, key, value, attention_mask)
        )

        self.assertIsInstance(output_attention_output, torch.Tensor)
        self.assertIsInstance(output_attention_weights, torch.Tensor)
        self.assertEqual(
            output_attention_output.shape,
            (self.target_sequence_length, self.batch_size, self.embedding_dim),
        )
        self.assertEqual(
            output_attention_weights.shape,
            (
                self.batch_size,
                self.target_sequence_length,
                self.source_sequence_length,
            ),
        )

    def test__return_attention_weights_flag__False(self):
        config = MultiHeadAttentionConfig(
            source_sequence_length=32,
            target_sequence_length=32,
            return_attention_weights_flag=False,
        )
        self.rebuild_presets(config)

        query = torch.randn(
            self.batch_size * self.num_heads, self.source_sequence_length, self.head_dim
        )
        key = torch.randn(
            self.batch_size * self.num_heads, self.target_sequence_length, self.head_dim
        )
        value = torch.randn(
            self.batch_size * self.num_heads, self.target_sequence_length, self.head_dim
        )
        attention_mask = torch.randn(
            1, self.target_sequence_length, self.source_sequence_length
        )
        attention_mask = torch.where(
            attention_mask > 0, torch.tensor(float("-inf")), torch.tensor(0.0)
        )
        attention_mask = attention_mask.repeat(self.batch_size * self.num_heads, 1, 1)

        output_attention_output, output_attention_weights = (
            self.model.compute_attention(query, key, value, attention_mask)
        )

        self.assertIsInstance(output_attention_output, torch.Tensor)
        self.assertIsNone(output_attention_weights)
        self.assertEqual(
            output_attention_output.shape,
            (self.target_sequence_length, self.batch_size, self.embedding_dim),
        )
