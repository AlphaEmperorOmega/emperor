import torch
import unittest

from dataclasses import asdict
from Emperor.attention.utils.utils import Utils
from Emperor.attention.utils.layer import MultiHeadAttentionConfig
from Emperor.attention.utils.presets import MultiHeadAttentionPresets
from Emperor.attention.utils._validator import MultiHeadAttentionValidator


class TestUtils(unittest.TestCase):
    def setUp(self):
        self.rebuild_presets()

    def tearDown(self):
        self.cfg = None
        self.config = None
        self.model = None
        self.batch_size = None
        self.embedding_dim = None
        self.query_key_projection_dim = None
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

        self.validator = MultiHeadAttentionValidator(self.config)
        self.model = Utils(self.config, self.validator)

        self.batch_size = self.config.batch_size
        self.embedding_dim = self.config.embedding_dim
        self.target_sequence_length = self.config.target_sequence_length
        self.source_sequence_length = self.config.source_sequence_length
        self.query_key_projection_dim = self.config.query_key_projection_dim
        self.num_heads = self.config.num_heads
        self.head_dim = self.embedding_dim // self.num_heads


class Test_add_batch_dimension_if_missing(TestUtils):
    def test_batched_input_with_correct_padding_and_attention_mask(self):
        query = torch.randn(
            self.target_sequence_length, self.batch_size, self.embedding_dim
        )
        key = torch.randn(
            self.source_sequence_length, self.batch_size, self.embedding_dim
        )
        value = torch.randn(
            self.source_sequence_length, self.batch_size, self.embedding_dim
        )
        key_padding_mask = torch.randint(
            0, 2, (self.batch_size, self.source_sequence_length)
        )
        attention_mask = torch.randn(
            self.batch_size * self.num_heads,
            self.target_sequence_length,
            self.source_sequence_length,
        )

        output_q, output_k, output_v, output_padding, output_attention_mask = (
            self.model.add_batch_dimension_if_missing(
                query, key, value, key_padding_mask, attention_mask
            )
        )

        self.assertEqual(query.shape, output_q.shape)
        self.assertEqual(key.shape, output_k.shape)
        self.assertEqual(value.shape, output_v.shape)
        self.assertEqual(key_padding_mask.shape, output_padding.shape)
        self.assertEqual(attention_mask.shape, output_attention_mask.shape)

    def test_unbatched_input_with_no_masks(self):
        query = torch.randn(
            self.target_sequence_length, self.batch_size, self.embedding_dim
        )
        key = torch.randn(
            self.source_sequence_length, self.batch_size, self.embedding_dim
        )
        value = torch.randn(
            self.source_sequence_length, self.batch_size, self.embedding_dim
        )

        output_q, output_k, output_v, output_padding, output_attention_mask = (
            self.model.add_batch_dimension_if_missing(query, key, value)
        )

        self.assertEqual(query.shape, output_q.shape)
        self.assertEqual(key.shape, output_k.shape)
        self.assertEqual(value.shape, output_v.shape)
        self.assertIsNone(output_padding)
        self.assertIsNone(output_attention_mask)

    def test_unbatched_input_with_correct_padding_and_attention_mask(self):
        query = torch.randn(self.target_sequence_length, self.embedding_dim)
        key = torch.randn(self.source_sequence_length, self.embedding_dim)
        value = torch.randn(self.source_sequence_length, self.embedding_dim)
        key_padding_mask = torch.randint(0, 2, (self.source_sequence_length,))
        attention_mask = torch.randn(
            self.target_sequence_length, self.source_sequence_length
        )

        output_q, output_k, output_v, output_padding, output_attention_mask = (
            self.model.add_batch_dimension_if_missing(
                query, key, value, key_padding_mask, attention_mask
            )
        )

        expected_q_shape = (self.target_sequence_length, 1, self.embedding_dim)
        expected_kv_shape = (self.source_sequence_length, 1, self.embedding_dim)
        self.assertEqual(output_q.shape, expected_q_shape)
        self.assertEqual(output_k.shape, expected_kv_shape)
        self.assertEqual(output_v.shape, expected_kv_shape)
        self.assertEqual(output_padding.shape, (1, self.source_sequence_length))
        self.assertEqual(
            output_attention_mask.shape,
            (1, self.target_sequence_length, self.source_sequence_length),
        )


class Test____concatenate_zeros_tensor(TestUtils):
    def test__method(self):
        tensor = torch.randn(
            self.batch_size * self.num_heads, self.source_sequence_length, self.head_dim
        )
        output = self.model._Utils__concatenate_zeros_tensor(tensor)

        expected_sequence_length = self.source_sequence_length + 1
        expected_shape = (
            self.batch_size * self.num_heads,
            expected_sequence_length,
            self.head_dim,
        )
        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(output.shape, expected_shape)


class Test__add_zero_attention(TestUtils):
    def test__all_inputs__zero_attention_flag__False(self):
        config = MultiHeadAttentionConfig(
            zero_attention_flag=False,
        )
        self.rebuild_presets(config)
        key = torch.randn(
            self.batch_size * self.num_heads, self.source_sequence_length, self.head_dim
        )
        value = torch.randn(
            self.batch_size * self.num_heads, self.source_sequence_length, self.head_dim
        )
        key_padding_mask = torch.randint(
            0, 2, (self.batch_size, self.source_sequence_length)
        )
        attention_mask = torch.randn(
            self.batch_size * self.num_heads,
            self.target_sequence_length,
            self.source_sequence_length,
        )

        padded_key, padded_value, output_key_padding_mask, output_attention_mask = (
            self.model.add_zero_attention(key, value, key_padding_mask, attention_mask)
        )

        self.assertIsInstance(padded_key, torch.Tensor)
        self.assertIsInstance(padded_value, torch.Tensor)
        self.assertTrue(torch.allclose(padded_key, key, atol=1e-6, rtol=1e-5))
        self.assertTrue(torch.allclose(padded_value, value, atol=1e-6, rtol=1e-5))
        self.assertTrue(
            torch.allclose(
                output_key_padding_mask, key_padding_mask, atol=1e-6, rtol=1e-5
            )
        )
        self.assertTrue(
            torch.allclose(
                output_attention_mask, output_attention_mask, atol=1e-6, rtol=1e-5
            )
        )

    def test__kv_inputs_only__zero_attention_flag__True(self):
        config = MultiHeadAttentionConfig(
            zero_attention_flag=True,
        )
        self.rebuild_presets(config)

        key = torch.randn(
            self.batch_size * self.num_heads, self.source_sequence_length, self.head_dim
        )
        value = torch.randn(
            self.batch_size * self.num_heads, self.source_sequence_length, self.head_dim
        )
        padded_key, padded_value, key_padding_mask, attention_mask = (
            self.model.add_zero_attention(key, value)
        )

        expected_sequence_length = self.source_sequence_length + 1
        expected_shape = (
            self.batch_size * self.num_heads,
            expected_sequence_length,
            self.head_dim,
        )
        self.assertIsInstance(padded_key, torch.Tensor)
        self.assertIsInstance(padded_value, torch.Tensor)
        self.assertIsNone(key_padding_mask)
        self.assertIsNone(attention_mask)
        self.assertEqual(padded_key.shape, expected_shape)
        self.assertEqual(padded_value.shape, expected_shape)

    def test__all_inputs__zero_attention_flag__True(self):
        config = MultiHeadAttentionConfig(
            zero_attention_flag=True,
        )
        self.rebuild_presets(config)

        key = torch.randn(
            self.batch_size * self.num_heads, self.source_sequence_length, self.head_dim
        )
        value = torch.randn(
            self.batch_size * self.num_heads, self.source_sequence_length, self.head_dim
        )
        key_padding_mask = torch.randint(
            0, 2, (self.batch_size, self.source_sequence_length)
        )
        attention_mask = torch.randn(
            self.batch_size * self.num_heads,
            self.target_sequence_length,
            self.source_sequence_length,
        )
        padded_key, padded_value, key_padding_mask, attention_mask = (
            self.model.add_zero_attention(key, value, key_padding_mask, attention_mask)
        )

        expected_sequence_length = self.source_sequence_length + 1
        expected_shape = (
            self.batch_size * self.num_heads,
            expected_sequence_length,
            self.head_dim,
        )
        expected_key_padding_shape = (self.batch_size, expected_sequence_length)
        expected_attention_mask_shape = (
            self.batch_size * self.num_heads,
            self.target_sequence_length,
            expected_sequence_length,
        )
        self.assertIsInstance(padded_key, torch.Tensor)
        self.assertIsInstance(padded_value, torch.Tensor)
        self.assertIsInstance(key_padding_mask, torch.Tensor)
        self.assertIsInstance(attention_mask, torch.Tensor)
        self.assertEqual(padded_key.shape, expected_shape)
        self.assertEqual(padded_value.shape, expected_shape)
        self.assertEqual(key_padding_mask.shape, expected_key_padding_shape)
        self.assertEqual(attention_mask.shape, expected_attention_mask_shape)
