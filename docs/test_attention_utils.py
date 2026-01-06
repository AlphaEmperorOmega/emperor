import torch
import unittest

from dataclasses import asdict
from Emperor.attention.utils.utils import Utils
from Emperor.attention.utils.layer import MultiHeadAttentionConfig
from Emperor.attention.utils._validator import MultiHeadAttentionConfigValidator
from Emperor.attention.utils.presets import MultiHeadAttentionPresets


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

        self.validator = MultiHeadAttentionConfigValidator(self.config)
        self.model = Utils(self.config, self.validator)

        self.batch_size = self.config.batch_size
        self.embedding_dim = self.config.embedding_dim
        self.target_sequence_length = self.config.target_sequence_length
        self.source_sequence_length = self.config.source_sequence_length
        self.query_key_projection_dim = self.config.query_key_projection_dim
        self.qk_head_dim = self.model.qk_head_dim
        self.v_head_dim = self.model.v_head_dim
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


class Test___reshape_projection_tesnor(TestUtils):
    def test_with_given_static_tensor(self):
        tensor = torch.randn(
            self.target_sequence_length, self.batch_size, self.embedding_dim
        )
        static_tensor = torch.randn(
            self.batch_size * self.num_heads, self.source_sequence_length, self.head_dim
        )
        output = self.model._Utils__reshape_projection_tesnor(tensor, static_tensor)

        self.assertTrue(torch.equal(output, static_tensor))

    def test_without_static_tensor(self):
        tensor = torch.randn(
            self.target_sequence_length, self.batch_size, self.embedding_dim
        )
        output = self.model._Utils__reshape_projection_tesnor(tensor)
        expected_output_shape = (
            self.batch_size * self.num_heads,
            self.target_sequence_length,
            self.head_dim,
        )
        self.assertEqual(output.shape, expected_output_shape)

    def test_without_static_tensor_and_custom_projection_dim(self):
        config = MultiHeadAttentionConfig(
            query_key_projection_dim=64,
        )
        self.rebuild_presets(config)
        tensor = torch.randn(
            self.target_sequence_length, self.batch_size, self.query_key_projection_dim
        )
        static_tensor = None
        head_dim = self.model.qk_head_dim
        output = self.model._Utils__reshape_projection_tesnor(
            tensor, static_tensor, head_dim
        )
        expected_output_shape = (
            self.batch_size * self.num_heads,
            self.target_sequence_length,
            head_dim,
        )
        self.assertEqual(output.shape, expected_output_shape)


class Test__reshape_qkv_for_attention(TestUtils):
    def test__qkv_inputs_only(self):
        query = torch.randn(
            self.target_sequence_length, self.batch_size, self.embedding_dim
        )
        key = torch.randn(
            self.source_sequence_length, self.batch_size, self.embedding_dim
        )
        value = torch.randn(
            self.source_sequence_length, self.batch_size, self.embedding_dim
        )
        static_key = None
        static_value = None

        query, key, value = self.model.reshape_qkv_for_attention(
            query, key, value, static_key, static_value
        )

        expected_q_shape = (
            self.batch_size * self.num_heads,
            self.target_sequence_length,
            self.head_dim,
        )
        expected_kv_shape = (
            self.batch_size * self.num_heads,
            self.source_sequence_length,
            self.head_dim,
        )
        self.assertIsInstance(query, torch.Tensor)
        self.assertIsInstance(key, torch.Tensor)
        self.assertIsInstance(value, torch.Tensor)
        self.assertEqual(query.shape, expected_q_shape)
        self.assertEqual(key.shape, expected_kv_shape)
        self.assertEqual(value.shape, expected_kv_shape)

    def test__all_inputs(self):
        query = torch.randn(
            self.target_sequence_length, self.batch_size, self.embedding_dim
        )
        key = torch.randn(
            self.source_sequence_length, self.batch_size, self.embedding_dim
        )
        value = torch.randn(
            self.source_sequence_length, self.batch_size, self.embedding_dim
        )
        static_key = torch.randn(
            self.batch_size * self.num_heads, self.source_sequence_length, self.head_dim
        )
        static_value = torch.randn(
            self.batch_size * self.num_heads, self.source_sequence_length, self.head_dim
        )

        query, key, value = self.model.reshape_qkv_for_attention(
            query, key, value, static_key, static_value
        )

        expected_q_shape = (
            self.batch_size * self.num_heads,
            self.target_sequence_length,
            self.qk_head_dim,
        )
        expected_kv_shape = (
            self.batch_size * self.num_heads,
            self.source_sequence_length,
            self.v_head_dim,
        )
        self.assertIsInstance(query, torch.Tensor)
        self.assertIsInstance(key, torch.Tensor)
        self.assertIsInstance(value, torch.Tensor)
        self.assertEqual(query.shape, expected_q_shape)
        self.assertEqual(key.shape, expected_kv_shape)
        self.assertEqual(value.shape, expected_kv_shape)
        self.assertTrue(torch.equal(key, static_key))
        self.assertTrue(torch.equal(value, static_value))


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
        self.assertTrue(torch.equal(padded_key, key))
        self.assertTrue(torch.equal(padded_value, value))
        self.assertTrue(torch.equal(output_key_padding_mask, key_padding_mask))
        self.assertTrue(torch.equal(output_attention_mask, output_attention_mask))

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


class Test____merge_attention_and_padding_mask(TestUtils):
    def test__inputs_as_None(self):
        key_padding_mask = None
        attention_mask = None

        output = self.model._Utils__merge_attention_and_padding_mask(
            key_padding_mask, attention_mask
        )
        self.assertIsNone(output)

    def test__attention_mask__None(self):
        key_padding_mask = torch.randint(
            0, 1, (self.batch_size * self.num_heads, 1, self.source_sequence_length)
        )
        attention_mask = None

        output = self.model._Utils__merge_attention_and_padding_mask(
            key_padding_mask, attention_mask
        )

        self.assertIsInstance(output, torch.Tensor)
        self.assertTrue(torch.equal(output, key_padding_mask))

    def test__all_inputs(self):
        key_padding_mask = torch.randint(
            0, 2, (self.batch_size * self.num_heads, 1, self.source_sequence_length)
        )
        attention_mask = torch.randn(
            self.batch_size * self.num_heads,
            self.target_sequence_length,
            self.source_sequence_length,
        )

        output = self.model._Utils__merge_attention_and_padding_mask(
            key_padding_mask, attention_mask
        )

        self.assertIsInstance(output, torch.Tensor)
        expected_output = attention_mask + key_padding_mask
        self.assertTrue(torch.equal(output, expected_output))


class Test__merge_padding_and_attention_mask(TestUtils):
    def test__inputs_as_None(self):
        key = torch.randn(
            self.batch_size * self.num_heads, self.source_sequence_length, self.head_dim
        )
        key_padding_mask = None
        attention_mask = None

        output = self.model.merge_padding_and_attention_mask(
            key, key_padding_mask, attention_mask
        )

        self.assertIsNone(output)

    def test__attention_mask__None(self):
        key = torch.randn(
            self.batch_size * self.num_heads, self.source_sequence_length, self.head_dim
        )
        key_padding_mask = torch.randint(
            0, 2, (self.batch_size, self.source_sequence_length)
        )
        attention_mask = None

        output = self.model.merge_padding_and_attention_mask(
            key, key_padding_mask, attention_mask
        )

        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(
            output.shape,
            (self.batch_size * self.num_heads, 1, self.source_sequence_length),
        )

    def test__key_padding_mask__None(self):
        key = torch.randn(
            self.batch_size * self.num_heads, self.source_sequence_length, self.head_dim
        )
        key_padding_mask = None
        attention_mask = torch.randn(
            self.batch_size * self.num_heads,
            self.target_sequence_length,
            self.source_sequence_length,
        )
        output = self.model.merge_padding_and_attention_mask(
            key, key_padding_mask, attention_mask
        )

        self.assertIsInstance(output, torch.Tensor)
        self.assertTrue(torch.equal(output, attention_mask))

    def test__all_inputs(self):
        key = torch.randn(
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

        output = self.model.merge_padding_and_attention_mask(
            key, key_padding_mask, attention_mask
        )

        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(
            output.shape,
            (
                self.batch_size * self.num_heads,
                self.target_sequence_length,
                self.source_sequence_length,
            ),
        )
