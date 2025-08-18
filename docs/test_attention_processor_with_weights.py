from dataclasses import asdict
import math
import unittest
from unittest.mock import MagicMock
import torch
import torch.nn.functional as F
from Emperor.attention.utils.utils import (
    AttentionProcessorWithReturnedWeights,
    AttentionValidator,
)
from Emperor.attention.attention import MultiHeadAttention, MultiHeadAttentionConfig
from docs.utils import default_unittest_config


class TestAttentionProcessorWithReturnedWeights(unittest.TestCase):
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
        self.model = AttentionProcessorWithReturnedWeights(
            self.config, validator, output_model
        )

        self.batch_size = self.config.batch_size
        self.embedding_dim = self.config.embedding_dim
        self.target_sequence_length = self.config.target_sequence_length
        self.source_sequence_length = self.config.source_sequence_length
        self.num_heads = self.config.num_heads
        self.head_dim = self.embedding_dim // self.num_heads


class Test____scale_query(TestAttentionProcessorWithReturnedWeights):
    def test__method(self):
        query = torch.randn(
            self.batch_size * self.num_heads, self.target_sequence_length, self.head_dim
        )
        scaled_query_tensor = (
            self.model._AttentionProcessorWithReturnedWeights__scale_query(query)
        )

        expected_result = query * math.sqrt(1.0 / float(self.head_dim))
        self.assertIsInstance(scaled_query_tensor, torch.Tensor)
        self.assertEqual(query.shape, scaled_query_tensor.shape)
        self.assertTrue(torch.equal(scaled_query_tensor, expected_result))


class Test____compute_raw_masked_attention_weights(
    TestAttentionProcessorWithReturnedWeights
):
    def test__attention_mask__None(self):
        query = torch.randn(
            self.batch_size * self.num_heads, self.target_sequence_length, self.head_dim
        )
        key = torch.randn(
            self.batch_size * self.num_heads, self.source_sequence_length, self.head_dim
        )
        attention_mask = None

        raw_unmasked_weights = self.model._AttentionProcessorWithReturnedWeights__compute_raw_masked_attention_weights(
            query, key, attention_mask
        )

        self.assertIsInstance(raw_unmasked_weights, torch.Tensor)
        self.assertEqual(
            raw_unmasked_weights.shape,
            (
                self.batch_size * self.num_heads,
                self.target_sequence_length,
                self.source_sequence_length,
            ),
        )
        transposed_keys = key.transpose(-2, -1)
        for idx in range(key.size(0)):
            q = query[idx]
            k = transposed_keys[idx]
            single_head_qk_attention_weights = torch.mm(q, k)
            self.assertTrue(
                torch.equal(raw_unmasked_weights[idx], single_head_qk_attention_weights)
            )

    def test__all_inputs(self):
        query = torch.randn(
            self.batch_size * self.num_heads, self.target_sequence_length, self.head_dim
        )
        key = torch.randn(
            self.batch_size * self.num_heads, self.source_sequence_length, self.head_dim
        )
        attention_mask = torch.randn(
            1, self.target_sequence_length, self.source_sequence_length
        )
        attention_mask = torch.where(
            attention_mask > 0, torch.tensor(float("-inf")), torch.tensor(0.0)
        )
        attention_mask = attention_mask.repeat(self.batch_size * self.num_heads, 1, 1)

        raw_masked_weights = self.model._AttentionProcessorWithReturnedWeights__compute_raw_masked_attention_weights(
            query, key, attention_mask
        )

        self.assertIsInstance(raw_masked_weights, torch.Tensor)
        self.assertEqual(
            raw_masked_weights.shape,
            (
                self.batch_size * self.num_heads,
                self.target_sequence_length,
                self.source_sequence_length,
            ),
        )

        transposed_keys = key.transpose(-2, -1)
        for idx in range(key.size(0)):
            q = query[idx]
            k = transposed_keys[idx]
            single_head_qk_attention_weights = torch.mm(q, k).masked_fill(
                (attention_mask[idx] == float("-inf")), torch.tensor(float("-inf"))
            )
            self.assertTrue(
                torch.equal(raw_masked_weights[idx], single_head_qk_attention_weights)
            )


class Test____compute_masked_attention_weights(
    TestAttentionProcessorWithReturnedWeights
):
    def test__method_using_mock(self):
        mock_scaled_query = torch.randn(
            self.batch_size * self.num_heads, self.target_sequence_length, self.head_dim
        )
        mock_raw_weights = torch.randn(
            self.batch_size * self.num_heads,
            self.target_sequence_length,
            self.source_sequence_length,
        )

        self.model._AttentionProcessorWithReturnedWeights__scale_query = MagicMock(
            return_value=mock_scaled_query
        )
        self.model._AttentionProcessorWithReturnedWeights__compute_raw_masked_attention_weights = MagicMock(
            return_value=mock_raw_weights
        )

        query = torch.randn(
            self.batch_size * self.num_heads, self.target_sequence_length, self.head_dim
        )
        key = torch.randn(
            self.batch_size * self.num_heads, self.source_sequence_length, self.head_dim
        )
        attention_mask = None

        result = self.model._AttentionProcessorWithReturnedWeights__compute_masked_attention_weights(
            query, key, attention_mask
        )

        self.model._AttentionProcessorWithReturnedWeights__scale_query.assert_called_once_with(
            query
        )
        self.model._AttentionProcessorWithReturnedWeights__compute_raw_masked_attention_weights.assert_called_once_with(
            mock_scaled_query, key, attention_mask
        )

        expected = F.softmax(mock_raw_weights, dim=-1)
        self.assertTrue(torch.equal(result, expected))

    def test__attention_mask__None(self):
        query = torch.randn(
            self.batch_size * self.num_heads, self.target_sequence_length, self.head_dim
        )
        key = torch.randn(
            self.batch_size * self.num_heads, self.source_sequence_length, self.head_dim
        )
        attention_mask = None

        masked_softmax_weights = self.model._AttentionProcessorWithReturnedWeights__compute_masked_attention_weights(
            query, key, attention_mask
        )
        self.assertIsInstance(masked_softmax_weights, torch.Tensor)
        self.assertFalse((masked_softmax_weights == 0.0).any())
        self.assertEqual(
            masked_softmax_weights.shape,
            (
                self.batch_size * self.num_heads,
                self.target_sequence_length,
                self.source_sequence_length,
            ),
        )

    def test__attention_mask__None__dropout_probability__50percent(self):
        config = MultiHeadAttentionConfig(
            dropout_probability=0.5,
        )
        self.rebuild_presets(config)
        query = torch.randn(
            self.batch_size * self.num_heads, self.target_sequence_length, self.head_dim
        )
        key = torch.randn(
            self.batch_size * self.num_heads, self.source_sequence_length, self.head_dim
        )
        attention_mask = None

        masked_softmax_weights = self.model._AttentionProcessorWithReturnedWeights__compute_masked_attention_weights(
            query, key, attention_mask
        )
        self.assertIsInstance(masked_softmax_weights, torch.Tensor)
        self.assertTrue((masked_softmax_weights == 0.0).any())
        self.assertEqual(
            masked_softmax_weights.shape,
            (
                self.batch_size * self.num_heads,
                self.target_sequence_length,
                self.source_sequence_length,
            ),
        )

    def test__all_inputs(self):
        query = torch.randn(
            self.batch_size * self.num_heads, self.target_sequence_length, self.head_dim
        )
        key = torch.randn(
            self.batch_size * self.num_heads, self.source_sequence_length, self.head_dim
        )
        attention_mask = torch.randn(
            1, self.target_sequence_length, self.source_sequence_length
        )
        attention_mask = torch.where(
            attention_mask > 0, torch.tensor(float("-inf")), torch.tensor(0.0)
        )
        attention_mask = attention_mask.repeat(self.batch_size * self.num_heads, 1, 1)

        masked_softmax_weights = self.model._AttentionProcessorWithReturnedWeights__compute_masked_attention_weights(
            query, key, attention_mask
        )

        self.assertIsInstance(masked_softmax_weights, torch.Tensor)
        self.assertEqual(
            masked_softmax_weights.shape,
            (
                self.batch_size * self.num_heads,
                self.target_sequence_length,
                self.source_sequence_length,
            ),
        )
        self.assertTrue((masked_softmax_weights == 0.0).any())
        self.assertTrue(
            torch.all((masked_softmax_weights >= 0) & (masked_softmax_weights <= 1))
        )


class Test____compute_weighted_values(TestAttentionProcessorWithReturnedWeights):
    def test__method(self):
        config = MultiHeadAttentionConfig(
            source_sequence_length=32,
            target_sequence_length=32,
        )
        self.rebuild_presets(config)
        attention_weights = torch.randn(
            self.batch_size * self.num_heads,
            self.target_sequence_length,
            self.source_sequence_length,
        )
        values = torch.randn(
            self.batch_size * self.num_heads, self.source_sequence_length, self.head_dim
        )

        weighted_values = (
            self.model._AttentionProcessorWithReturnedWeights__compute_weighted_values(
                attention_weights, values
            )
        )

        self.assertIsInstance(weighted_values, torch.Tensor)
        self.assertEqual(
            weighted_values.shape,
            (self.batch_size * self.target_sequence_length, self.embedding_dim),
        )


class Test____compute_attention_output(TestAttentionProcessorWithReturnedWeights):
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


class Test____maybe_average_attention_weights(
    TestAttentionProcessorWithReturnedWeights
):
    def test__average_attention_weights_flag__False(self):
        config = MultiHeadAttentionConfig(average_attention_weights_flag=False)
        self.rebuild_presets(config)
        attention_weights = torch.randn(
            self.batch_size,
            self.num_heads,
            self.target_sequence_length,
            self.source_sequence_length,
        )

        output_attention_weights = self.model._AttentionProcessorWithReturnedWeights__maybe_average_attention_weights(
            attention_weights
        )

        self.assertIsInstance(output_attention_weights, torch.Tensor)
        self.assertTrue(torch.equal(output_attention_weights, attention_weights))
        self.assertEqual(
            output_attention_weights.shape,
            (
                self.batch_size,
                self.num_heads,
                self.target_sequence_length,
                self.source_sequence_length,
            ),
        )

    def test__average_attention_weights_flag__True(self):
        config = MultiHeadAttentionConfig(
            average_attention_weights_flag=True,
        )
        self.rebuild_presets(config)
        attention_weights = torch.randn(
            self.batch_size,
            self.num_heads,
            self.target_sequence_length,
            self.source_sequence_length,
        )

        output_attention_weights = self.model._AttentionProcessorWithReturnedWeights__maybe_average_attention_weights(
            attention_weights
        )

        expected_output = attention_weights.mean(dim=1)
        self.assertIsInstance(output_attention_weights, torch.Tensor)
        self.assertTrue(torch.equal(output_attention_weights, expected_output))
        self.assertEqual(
            output_attention_weights.shape,
            (self.batch_size, self.target_sequence_length, self.source_sequence_length),
        )


class Test____handle_batched_input(TestAttentionProcessorWithReturnedWeights):
    def test__batched_input_flag__False(self):
        config = MultiHeadAttentionConfig(
            batch_size=1,
        )
        self.rebuild_presets(config)

        attention_output = torch.randn(
            self.target_sequence_length, self.batch_size, self.embedding_dim
        )
        attention_weights = torch.randn(
            self.batch_size,
            self.num_heads,
            self.target_sequence_length,
            self.source_sequence_length,
        )

        output_attention_output, output_attention_weights = (
            self.model._AttentionProcessorWithReturnedWeights__handle_batched_input(
                attention_output, attention_weights
            )
        )

        self.assertIsInstance(output_attention_output, torch.Tensor)
        self.assertIsInstance(output_attention_weights, torch.Tensor)
        self.assertEqual(
            output_attention_output.shape,
            (self.target_sequence_length, self.embedding_dim),
        )
        self.assertEqual(
            output_attention_weights.shape,
            (self.num_heads, self.target_sequence_length, self.source_sequence_length),
        )

    def test__batched_input_flag__True(self):
        attention_output = torch.randn(
            self.target_sequence_length, self.batch_size, self.embedding_dim
        )
        attention_weights = torch.randn(
            self.batch_size,
            self.num_heads,
            self.target_sequence_length,
            self.source_sequence_length,
        )

        output_attention_output, output_attention_weights = (
            self.model._AttentionProcessorWithReturnedWeights__handle_batched_input(
                attention_output, attention_weights
            )
        )

        self.assertIsInstance(output_attention_output, torch.Tensor)
        self.assertIsInstance(output_attention_weights, torch.Tensor)
        self.assertTrue(torch.equal(output_attention_output, attention_output))
        self.assertTrue(torch.equal(output_attention_weights, attention_weights))


class Test____ensure_correct_shape_output(TestAttentionProcessorWithReturnedWeights):
    def test__average_attention_weights_flag__True(self):
        config = MultiHeadAttentionConfig(
            average_attention_weights_flag=True,
        )
        self.rebuild_presets(config)

        attention_output = torch.randn(
            self.target_sequence_length, self.batch_size, self.embedding_dim
        )
        attention_weights = torch.randn(
            self.batch_size * self.num_heads,
            self.target_sequence_length,
            self.source_sequence_length,
        )

        output_attention_output, output_attention_weights = (
            self.model._AttentionProcessorWithReturnedWeights__ensure_correct_shape_output(
                attention_output, attention_weights
            )
        )

        self.assertIsInstance(output_attention_output, torch.Tensor)
        self.assertIsInstance(output_attention_weights, torch.Tensor)
        self.assertTrue(torch.equal(output_attention_output, attention_output))
        self.assertEqual(
            output_attention_output.shape,
            (self.target_sequence_length, self.batch_size, self.embedding_dim),
        )
        self.assertEqual(
            output_attention_weights.shape,
            (self.batch_size, self.target_sequence_length, self.source_sequence_length),
        )

    def test__average_attention_weights_flag__True__batched_input_flag__False(self):
        config = MultiHeadAttentionConfig(
            batch_size=1,
            average_attention_weights_flag=True,
        )
        self.rebuild_presets(config)

        attention_output = torch.randn(
            self.target_sequence_length, self.batch_size, self.embedding_dim
        )
        attention_weights = torch.randn(
            self.batch_size * self.num_heads,
            self.target_sequence_length,
            self.source_sequence_length,
        )

        output_attention_output, output_attention_weights = (
            self.model._AttentionProcessorWithReturnedWeights__ensure_correct_shape_output(
                attention_output, attention_weights
            )
        )

        self.assertIsInstance(output_attention_output, torch.Tensor)
        self.assertIsInstance(output_attention_weights, torch.Tensor)
        self.assertEqual(
            output_attention_output.shape,
            (self.target_sequence_length, self.embedding_dim),
        )
        self.assertEqual(
            output_attention_weights.shape,
            (self.target_sequence_length, self.source_sequence_length),
        )


class Test__compute_attention(TestAttentionProcessorWithReturnedWeights):
    def test__average_attention_weights_flag__True(self):
        config = MultiHeadAttentionConfig(
            source_sequence_length=32,
            target_sequence_length=32,
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
            (self.batch_size, self.target_sequence_length, self.source_sequence_length),
        )

    def test__average_attention_weights_flag__False(self):
        config = MultiHeadAttentionConfig(
            source_sequence_length=32,
            target_sequence_length=32,
            average_attention_weights_flag=False,
        )
        self.rebuild_presets(config)
        query = torch.randn(
            self.batch_size * self.num_heads, self.source_sequence_length, self.head_dim
        )
        key = torch.randn(
            self.batch_size * self.num_heads, self.source_sequence_length, self.head_dim
        )
        value = torch.randn(
            self.batch_size * self.num_heads, self.source_sequence_length, self.head_dim
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

    def test__attention_mask__None(self):
        config = MultiHeadAttentionConfig(
            source_sequence_length=32,
            target_sequence_length=32,
            average_attention_weights_flag=False,
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
