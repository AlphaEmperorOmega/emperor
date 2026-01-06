from dataclasses import asdict
import unittest
import torch

from Emperor.attention.utils.layer import MultiHeadAttentionConfig
from Emperor.attention.utils.presets import MultiHeadAttentionPresets
from Emperor.attention.utils.handlers.projector import ProjectorSelector
from Emperor.attention.utils._validator import MultiHeadAttentionConfigValidator
from Emperor.attention.utils.handlers.processor import (
    Processor,
    ProcessorDefault,
    ProcessorWithReturnedWeights,
)


class TestProcessor(unittest.TestCase):
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
        self.config = MultiHeadAttentionPresets.multi_head_attention_preset(
            embedding_dim=12,
            query_key_projection_dim=12,
            value_projection_dim=12,
        )
        if config is not None:
            for k in asdict(config):
                if hasattr(self.config, k) and getattr(config, k) is not None:
                    setattr(self.config, k, getattr(config, k))

        validator = MultiHeadAttentionConfigValidator(self.config)
        projector = ProjectorSelector(self.config).build_model()
        self.model = Processor(self.config, validator, projector)

        self.batch_size = self.config.batch_size
        self.embedding_dim = self.config.embedding_dim
        self.target_sequence_length = self.config.target_sequence_length
        self.source_sequence_length = self.config.source_sequence_length
        self.num_heads = self.config.num_heads
        self.head_dim = self.embedding_dim // self.num_heads


class Test__init(TestProcessor):
    def test__init(self):
        self.assertIsInstance(self.model, Processor)
        self.assertIsInstance(
            self.model.processor,
            (ProcessorDefault, ProcessorWithReturnedWeights),
        )


class Test____create_processor(TestProcessor):
    def test__return_attention_weights_flag__False(self):
        config = MultiHeadAttentionConfig(
            return_attention_weights_flag=False,
        )
        self.rebuild_presets(config)
        self.assertIsInstance(self.model.processor, ProcessorDefault)

    def test__return_attention_weights_flag__True(self):
        config = MultiHeadAttentionConfig(
            return_attention_weights_flag=True,
        )
        self.rebuild_presets(config)
        self.assertIsInstance(self.model.processor, ProcessorWithReturnedWeights)


class Test____compute_weighted_values_default(TestProcessor):
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
