import torch
import unittest

from dataclasses import asdict
from Emperor.attention.utils.layer import MultiHeadAttentionConfig
from Emperor.attention.utils.presets import MultiHeadAttentionPresets
from Emperor.attention.utils.handlers.processor import ProcessorDefault
from Emperor.attention.utils.handlers.projector import ProjectorBuilder
from Emperor.attention.utils._validator import MultiHeadAttentionValidator


class TestProcessorDefault(unittest.TestCase):
    def setUp(self):
        self.rebuild_presets()

    def tearDown(self):
        self.config = None
        self.model = None
        self.batch_size = None
        self.embedding_dim = None
        self.target_sequence_length = None
        self.num_heads = None
        self.head_dim = None

    def rebuild_presets(self, config: MultiHeadAttentionConfig | None = None):
        self.config = MultiHeadAttentionPresets.multi_head_attention_preset(
            embedding_dim=12,
            query_key_projection_dim=12,
            value_projection_dim=12,
        )
        if config is not None:
            for k in asdict(config):
                if hasattr(self.config, k) and getattr(config, k) is not None:
                    setattr(self.config, k, getattr(config, k))

        validator = MultiHeadAttentionValidator(self.config)
        projector = ProjectorBuilder(self.config).build_model()
        self.model = ProcessorDefault(self.config, validator, projector)

        self.batch_size = self.config.batch_size
        self.embedding_dim = self.config.embedding_dim
        self.query_key_projection_dim = self.config.query_key_projection_dim
        self.value_projection_dim = self.config.query_key_projection_dim
        self.target_sequence_length = self.config.target_sequence_length
        self.source_sequence_length = self.config.source_sequence_length
        self.num_heads = self.config.num_heads
        self.head_dim = self.embedding_dim // self.num_heads
        self.qk_head_dim = self.model.qk_head_dim
        self.v_head_dim = self.model.v_head_dim


class Test____prepare_attnetion_mask(TestProcessorDefault):
    def test__attention_mask__None(self):
        attention_mask = None
        output_attention_mask = self.model._ProcessorDefault__prepare_attnetion_mask(
            attention_mask
        )

        self.assertIsNone(output_attention_mask)

    def test__batched_attention_mask(self):
        attention_mask = torch.randn(
            1, self.target_sequence_length, self.source_sequence_length
        )
        attention_mask = torch.where(
            attention_mask > 0, torch.tensor(float("-inf")), torch.tensor(0.0)
        )
        attention_mask = attention_mask.repeat(self.batch_size * self.num_heads, 1, 1)
        output_attention_mask = self.model._ProcessorDefault__prepare_attnetion_mask(
            attention_mask
        )

        self.assertIsInstance(output_attention_mask, torch.Tensor)
        self.assertEqual(
            output_attention_mask.shape,
            (
                self.batch_size,
                self.num_heads,
                self.target_sequence_length,
                self.source_sequence_length,
            ),
        )

    def test__unbatched_attention_mask(self):
        attention_mask = None
        attention_mask = torch.randn(
            1, self.target_sequence_length, self.source_sequence_length
        )
        attention_mask = torch.where(
            attention_mask > 0, torch.tensor(float("-inf")), torch.tensor(0.0)
        )

        output_attention_mask = self.model._ProcessorDefault__prepare_attnetion_mask(
            attention_mask
        )

        self.assertIsInstance(output_attention_mask, torch.Tensor)
        self.assertEqual(
            output_attention_mask.shape,
            (1, 1, self.target_sequence_length, self.source_sequence_length),
        )


class Test____reshape_qkv_for_attention(TestProcessorDefault):
    def test__method(self):
        query = torch.randn(
            self.batch_size * self.num_heads,
            self.target_sequence_length,
            self.qk_head_dim,
        )
        key = torch.randn(
            self.batch_size * self.num_heads,
            self.source_sequence_length,
            self.qk_head_dim,
        )
        value = torch.randn(
            self.batch_size * self.num_heads,
            self.source_sequence_length,
            self.v_head_dim,
        )

        query, key, value = self.model._ProcessorDefault__reshape_qkv_for_attention(
            query, key, value
        )

        self.assertIsInstance(query, torch.Tensor)
        self.assertIsInstance(key, torch.Tensor)
        self.assertIsInstance(value, torch.Tensor)
        self.assertEqual(
            query.shape,
            (
                self.batch_size,
                self.num_heads,
                self.target_sequence_length,
                self.qk_head_dim,
            ),
        )
        self.assertEqual(
            key.shape,
            (
                self.batch_size,
                self.num_heads,
                self.source_sequence_length,
                self.qk_head_dim,
            ),
        )
        self.assertEqual(
            value.shape,
            (
                self.batch_size,
                self.num_heads,
                self.source_sequence_length,
                self.v_head_dim,
            ),
        )


class Test____compute_weighted_values(TestProcessorDefault):
    def test__method(self):
        config = MultiHeadAttentionConfig(
            source_sequence_length=32,
            target_sequence_length=32,
        )
        self.rebuild_presets(config)
        query = torch.randn(
            self.batch_size, self.num_heads, self.target_sequence_length, self.head_dim
        )
        key = torch.randn(
            self.batch_size, self.num_heads, self.source_sequence_length, self.head_dim
        )
        value = torch.randn(
            self.batch_size, self.num_heads, self.source_sequence_length, self.head_dim
        )
        attention_mask = torch.randn(
            self.batch_size,
            self.num_heads,
            self.target_sequence_length,
            self.source_sequence_length,
        )
        attention_mask = torch.where(
            attention_mask > 0, torch.tensor(float("-inf")), torch.tensor(0.0)
        )

        weighted_values = self.model._ProcessorDefault__compute_weighted_values(
            query, key, value, attention_mask
        )

        self.assertIsInstance(weighted_values, torch.Tensor)
        self.assertEqual(
            weighted_values.shape,
            (
                self.batch_size * self.target_sequence_length,
                self.embedding_dim,
            ),
        )


class Test___compute_attention_output(TestProcessorDefault):
    def test__method(self):
        config = MultiHeadAttentionConfig(
            source_sequence_length=32,
            target_sequence_length=32,
        )
        self.rebuild_presets(config)

        weighted_values = torch.randn(
            self.batch_size * self.target_sequence_length, self.embedding_dim
        )

        weighted_values = self.model._compute_attention_output(weighted_values)

        self.assertIsInstance(weighted_values, torch.Tensor)
        self.assertEqual(
            weighted_values.shape,
            (self.target_sequence_length, self.batch_size, self.embedding_dim),
        )


class Test__compute_attention(TestProcessorDefault):
    def test__method(self):
        config = MultiHeadAttentionConfig(
            source_sequence_length=32,
            target_sequence_length=32,
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

        output_attention_output, _ = self.model.compute_attention(
            query, key, value, attention_mask
        )

        self.assertIsInstance(output_attention_output, torch.Tensor)
        self.assertIsNone(_, torch.Tensor)
        self.assertEqual(
            output_attention_output.shape,
            (self.target_sequence_length, self.batch_size, self.embedding_dim),
        )

    def test__attention_mask__None(self):
        config = MultiHeadAttentionConfig(
            source_sequence_length=32,
            target_sequence_length=32,
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
        attention_mask = None

        output_attention_output, _ = self.model.compute_attention(
            query, key, value, attention_mask
        )

        self.assertIsInstance(output_attention_output, torch.Tensor)
        self.assertIsNone(_)
        self.assertEqual(
            output_attention_output.shape,
            (self.target_sequence_length, self.batch_size, self.embedding_dim),
        )
