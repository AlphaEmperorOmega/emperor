import copy
import unittest
from unittest.mock import MagicMock, Mock, patch
import torch
import torch.nn as nn
import torch.nn.functional as F
from Emperor.attention.utils.utils import (
    AttentionMask,
    AttentionProcessor,
    AttentionProjector,
    AttentionUtils,
    AttentionValidator,
)
from Emperor.attention.attention import MultiHeadAttention
from docs.utils import default_unittest_config


class TestAttentionUtils(unittest.TestCase):
    def setUp(self):
        self.cfg = default_unittest_config()


class Test____transpose_shared_qkv(TestAttentionUtils):
    def setUp(self):
        super().setUp()
        c = copy.deepcopy(self.cfg)
        config = c.multi_head_attention_model_config
        batch_size = config.batch_size
        target_sequence_length = config.target_sequence_length
        source_sequence_length = config.source_sequence_length
        embedding_dim = config.embedding_dim

        self.q_input_shape = (target_sequence_length, batch_size, embedding_dim)
        self.kv_input_shape = (source_sequence_length, batch_size, embedding_dim)
        self.q_output_shape = (batch_size, target_sequence_length, embedding_dim)
        self.kv_output_shape = (batch_size, source_sequence_length, embedding_dim)

    def test__same_tensor_for_kv(self):
        c = copy.deepcopy(self.cfg)
        config = c.multi_head_attention_model_config
        validator = AttentionValidator(config)
        m = AttentionUtils(config, validator)

        query = key = torch.randn(self.q_input_shape)
        output_q, output_k, output_v = m._AttentionUtils__transpose_shared_qkv(
            query, key
        )

        self.assertEqual(output_q.shape, self.q_output_shape)
        self.assertEqual(output_k.shape, self.q_output_shape)
        self.assertEqual(output_v.shape, self.q_output_shape)
        self.assertTrue(torch.equal(output_q, output_k))
        self.assertTrue(torch.equal(output_k, output_v))

    def test__different_tensors_for_kv(self):
        c = copy.deepcopy(self.cfg)
        config = c.multi_head_attention_model_config
        validator = AttentionValidator(config)
        m = AttentionUtils(config, validator)

        query = torch.randn(self.q_input_shape)
        key = torch.randn(self.kv_input_shape)

        output_q, output_k, output_v = m._AttentionUtils__transpose_shared_qkv(
            query, key
        )

        self.assertEqual(output_q.shape, self.q_output_shape)
        self.assertEqual(output_k.shape, self.kv_output_shape)
        self.assertEqual(output_v.shape, self.kv_output_shape)
        self.assertFalse(torch.equal(output_q, output_k))
        self.assertFalse(torch.equal(output_q, output_v))
        self.assertTrue(torch.equal(output_k, output_v))


class Test__maybe_transpose_qkv(TestAttentionUtils):
    def test__batch_first__False(self):
        c = copy.deepcopy(self.cfg)
        config = c.multi_head_attention_model_config
        config.batch_first_flag = False
        validator = AttentionValidator(config)
        m = AttentionUtils(config, validator)

        batch_size = config.batch_size
        target_sequence_length = config.target_sequence_length
        embedding_dim = config.embedding_dim

        query = key = value = torch.randn(
            target_sequence_length, batch_size, embedding_dim
        )

        output_q, output_k, output_v = m.maybe_transpose_qkv(query, key, value)

        self.assertTrue(torch.equal(output_q, query))
        self.assertTrue(torch.equal(output_k, key))
        self.assertTrue(torch.equal(output_v, value))

    def test__batch_first__True(self):
        c = copy.deepcopy(self.cfg)
        config = c.multi_head_attention_model_config
        config.batch_first_flag = True
        validator = AttentionValidator(config)
        m = AttentionUtils(config, validator)

        batch_size = config.batch_size
        target_sequence_length = config.target_sequence_length
        source_sequence_length = config.target_sequence_length
        embedding_dim = config.embedding_dim

        query = key = value = torch.randn(
            target_sequence_length, batch_size, embedding_dim
        )
        output_q, output_k, output_v = m.maybe_transpose_qkv(query, key, value)

        self.assertEqual(
            output_q.shape, (batch_size, target_sequence_length, embedding_dim)
        )
        self.assertEqual(
            output_k.shape, (batch_size, target_sequence_length, embedding_dim)
        )
        self.assertEqual(
            output_v.shape, (batch_size, target_sequence_length, embedding_dim)
        )
        self.assertTrue(torch.equal(output_q, output_k))
        self.assertTrue(torch.equal(output_k, output_v))
        self.assertTrue(torch.equal(output_q, output_v))

    def test__batch_first__True__same_kv(self):
        c = copy.deepcopy(self.cfg)
        config = c.multi_head_attention_model_config
        config.batch_first_flag = True
        validator = AttentionValidator(config)
        m = AttentionUtils(config, validator)

        batch_size = config.batch_size
        target_sequence_length = config.target_sequence_length
        source_sequence_length = config.target_sequence_length
        embedding_dim = config.embedding_dim

        query = torch.randn(target_sequence_length, batch_size, embedding_dim)
        key = value = torch.randn(source_sequence_length, batch_size, embedding_dim)
        output_q, output_k, output_v = m.maybe_transpose_qkv(query, key, value)

        self.assertEqual(
            output_q.shape, (batch_size, target_sequence_length, embedding_dim)
        )
        self.assertEqual(
            output_k.shape, (batch_size, source_sequence_length, embedding_dim)
        )
        self.assertEqual(
            output_v.shape, (batch_size, source_sequence_length, embedding_dim)
        )
        self.assertTrue(torch.equal(output_k, output_v))
        self.assertTrue(torch.all(output_q != output_k))
        self.assertTrue(torch.all(output_q != output_v))

    def test__batch_first__True__different__qkv(self):
        c = copy.deepcopy(self.cfg)
        config = c.multi_head_attention_model_config
        config.batch_first_flag = True
        validator = AttentionValidator(config)
        m = AttentionUtils(config, validator)

        batch_size = config.batch_size
        source_sequence_length = config.source_sequence_length
        target_sequence_length = config.source_sequence_length
        embedding_dim = config.embedding_dim

        query = torch.randn(target_sequence_length, batch_size, embedding_dim)
        key = torch.randn(source_sequence_length, batch_size, embedding_dim)
        value = torch.randn(source_sequence_length, batch_size, embedding_dim)
        output_q, output_k, output_v = m.maybe_transpose_qkv(query, key, value)

        self.assertEqual(
            output_q.shape, (batch_size, target_sequence_length, embedding_dim)
        )
        self.assertEqual(
            output_k.shape, (batch_size, source_sequence_length, embedding_dim)
        )
        self.assertEqual(
            output_v.shape, (batch_size, source_sequence_length, embedding_dim)
        )
        self.assertTrue(torch.all(output_k != output_v))
        self.assertTrue(torch.all(output_q != output_k))
        self.assertTrue(torch.all(output_q != output_v))


class Test__add_batch_dimension_if_missing(TestAttentionUtils):
    def test__batched_input(self):
        c = copy.deepcopy(self.cfg)
        config = c.multi_head_attention_model_config
        config.batch_first_flag = True
        validator = AttentionValidator(config)
        m = AttentionUtils(config, validator)

        batch_size = config.batch_size
        num_heads = config.num_heads
        source_sequence_length = config.source_sequence_length
        target_sequence_length = config.source_sequence_length
        embedding_dim = config.embedding_dim

        query = torch.randn(target_sequence_length, batch_size, embedding_dim)
        key = torch.randn(source_sequence_length, batch_size, embedding_dim)
        value = torch.randn(source_sequence_length, batch_size, embedding_dim)
        key_padding_mask = torch.randint(0, 2, (batch_size, source_sequence_length))
        attention_mask = torch.randn(
            batch_size * num_heads, target_sequence_length, source_sequence_length
        )

        output_q, output_k, output_v, output_padding, output_attention_mask = (
            m.add_batch_dimension_if_missing(
                query, key, value, key_padding_mask, attention_mask
            )
        )

        self.assertEqual(query.shape, output_q.shape)
        self.assertEqual(key.shape, output_k.shape)
        self.assertEqual(value.shape, output_v.shape)
        self.assertEqual(key_padding_mask.shape, output_padding.shape)
        self.assertEqual(attention_mask.shape, output_attention_mask.shape)

    def test__non_batched_input(self):
        c = copy.deepcopy(self.cfg)
        config = c.multi_head_attention_model_config
        config.batch_first_flag = True
        validator = AttentionValidator(config)
        m = AttentionUtils(config, validator)

        source_sequence_length = config.source_sequence_length
        target_sequence_length = config.source_sequence_length
        embedding_dim = config.embedding_dim

        query = torch.randn(target_sequence_length, embedding_dim)
        key = torch.randn(source_sequence_length, embedding_dim)
        value = torch.randn(source_sequence_length, embedding_dim)
        key_padding_mask = torch.randint(0, 2, (source_sequence_length,))
        attention_mask = torch.randn(target_sequence_length, source_sequence_length)

        output_q, output_k, output_v, output_padding, output_attention_mask = (
            m.add_batch_dimension_if_missing(
                query, key, value, key_padding_mask, attention_mask
            )
        )

        self.assertEqual(output_q.shape, (target_sequence_length, 1, embedding_dim))
        self.assertEqual(output_k.shape, (source_sequence_length, 1, embedding_dim))
        self.assertEqual(output_v.shape, (source_sequence_length, 1, embedding_dim))
        self.assertEqual(output_padding.shape, (1, source_sequence_length))
        self.assertEqual(
            output_attention_mask.shape,
            (1, target_sequence_length, source_sequence_length),
        )


class Test__add_learnable_bias_vectors(TestAttentionUtils):
    def test__kv_input_tensor_only__no_kv_biases(self):
        c = copy.deepcopy(self.cfg)
        config = c.multi_head_attention_model_config
        config.batch_first_flag = True
        validator = AttentionValidator(config)
        m = AttentionUtils(config, validator)

        batch_size = config.batch_size
        source_sequence_length = config.source_sequence_length
        embedding_dim = config.embedding_dim

        key_projections = torch.randn(source_sequence_length, batch_size, embedding_dim)
        value_projections = torch.randn(
            source_sequence_length, batch_size, embedding_dim
        )

        (
            out_key_projections,
            out_value_projections,
            out_key_padding_mask,
            out_attention_mask,
        ) = m.add_learnable_bias_vectors(key_projections, value_projections)
        self.assertEqual(out_key_projections.shape, key_projections.shape)
        self.assertEqual(out_value_projections.shape, out_value_projections.shape)
        self.assertIsNone(out_key_padding_mask)
        self.assertIsNone(out_attention_mask)

    def test__all_inputs__no_kv_biases(self):
        c = copy.deepcopy(self.cfg)
        config = c.multi_head_attention_model_config
        config.batch_first_flag = True
        validator = AttentionValidator(config)
        m = AttentionUtils(config, validator)

        num_heads = config.num_heads
        batch_size = config.batch_size
        source_sequence_length = config.source_sequence_length
        target_sequence_length = config.source_sequence_length
        embedding_dim = config.embedding_dim

        key_projections = torch.randn(source_sequence_length, batch_size, embedding_dim)
        value_projections = torch.randn(
            source_sequence_length, batch_size, embedding_dim
        )
        key_padding_mask = torch.randint(0, 2, (batch_size, source_sequence_length))
        attention_mask = torch.randn(
            batch_size * num_heads, target_sequence_length, source_sequence_length
        )

        (
            out_key_projections,
            out_value_projections,
            out_key_padding_mask,
            out_attention_mask,
        ) = m.add_learnable_bias_vectors(
            key_projections, value_projections, key_padding_mask, attention_mask
        )
        self.assertEqual(out_key_projections.shape, key_projections.shape)
        self.assertEqual(out_value_projections.shape, out_value_projections.shape)
        self.assertEqual(out_key_padding_mask.shape, key_padding_mask.shape)
        self.assertEqual(out_attention_mask.shape, attention_mask.shape)
        self.assertTrue(torch.equal(out_key_projections, key_projections))
        self.assertTrue(torch.equal(out_value_projections, out_value_projections))
        self.assertTrue(torch.equal(out_key_padding_mask, key_padding_mask))
        self.assertTrue(torch.equal(out_attention_mask, attention_mask))

    def test__all_inputs__add_key_value_bias_flag__True(self):
        c = copy.deepcopy(self.cfg)
        c.multi_head_attention_model_config.add_key_value_bias_flag = True
        config = c.multi_head_attention_model_config
        config.batch_first_flag = True
        model = MultiHeadAttention(c)
        validator = AttentionValidator(config)
        m = AttentionUtils(
            config, validator, model.key_bias_vector, model.value_bias_vector
        )

        num_heads = config.num_heads
        batch_size = config.batch_size
        source_sequence_length = config.source_sequence_length
        target_sequence_length = config.source_sequence_length
        embedding_dim = config.embedding_dim

        key_projections = torch.randn(source_sequence_length, batch_size, embedding_dim)
        value_projections = torch.randn(
            source_sequence_length, batch_size, embedding_dim
        )
        key_padding_mask = torch.randint(0, 2, (batch_size, source_sequence_length))
        attention_mask = torch.randn(
            batch_size * num_heads, target_sequence_length, source_sequence_length
        )

        (
            out_key_projections,
            out_value_projections,
            out_key_padding_mask,
            out_attention_mask,
        ) = m.add_learnable_bias_vectors(
            key_projections, value_projections, key_padding_mask, attention_mask
        )
        source_sequence_length_updated = source_sequence_length + 1
        self.assertEqual(
            out_key_projections.shape,
            (source_sequence_length_updated, batch_size, embedding_dim),
        )
        self.assertEqual(
            out_value_projections.shape,
            (source_sequence_length_updated, batch_size, embedding_dim),
        )
        self.assertEqual(
            out_key_padding_mask.shape, (batch_size, source_sequence_length_updated)
        )
        self.assertEqual(
            out_attention_mask.shape,
            (
                batch_size * num_heads,
                target_sequence_length,
                source_sequence_length_updated,
            ),
        )


class Test____reshape_projection_tesnor(TestAttentionUtils):
    def test__input_as_tensor_and_static_tensor(self):
        c = copy.deepcopy(self.cfg)
        c.multi_head_attention_model_config.add_key_value_bias_flag = True
        config = c.multi_head_attention_model_config
        config.batch_first_flag = True
        model = MultiHeadAttention(c)
        validator = AttentionValidator(config)
        m = AttentionUtils(
            config, validator, model.key_bias_vector, model.value_bias_vector
        )

        batch_size = config.batch_size
        num_heads = config.num_heads
        embedding_dim = config.embedding_dim
        head_dim = embedding_dim // num_heads
        sequence_length = config.source_sequence_length
        target_sequence_length = config.source_sequence_length

        tensor = torch.randn(target_sequence_length, batch_size, embedding_dim)
        static_tensor = torch.randn(batch_size * num_heads, sequence_length, head_dim)
        output = m._AttentionUtils__reshape_projection_tesnor(tensor, static_tensor)

        self.assertTrue(torch.equal(output, static_tensor))

    def test__input_as_tensor_and_static_vensor__None(self):
        c = copy.deepcopy(self.cfg)
        c.multi_head_attention_model_config.add_key_value_bias_flag = True
        config = c.multi_head_attention_model_config
        config.batch_first_flag = True
        model = MultiHeadAttention(c)
        validator = AttentionValidator(config)
        m = AttentionUtils(
            config, validator, model.key_bias_vector, model.value_bias_vector
        )

        batch_size = config.batch_size
        embedding_dim = config.embedding_dim
        sequence_length = config.source_sequence_length
        num_heads = config.num_heads
        head_dim = embedding_dim // num_heads

        tensor = torch.randn(sequence_length, batch_size, embedding_dim)
        static_tensor = None
        output = m._AttentionUtils__reshape_projection_tesnor(tensor, static_tensor)

        self.assertEqual(
            output.shape,
            (batch_size * num_heads, sequence_length, head_dim),
        )


class Test__prepare_qkv_projection_for_attention(TestAttentionUtils):
    def test__qkv_inputs_only(self):
        c = copy.deepcopy(self.cfg)
        c.multi_head_attention_model_config.add_key_value_bias_flag = True
        config = c.multi_head_attention_model_config
        config.batch_first_flag = True
        model = MultiHeadAttention(c)
        validator = AttentionValidator(config)
        m = AttentionUtils(
            config, validator, model.key_bias_vector, model.value_bias_vector
        )

        batch_size = config.batch_size
        embedding_dim = config.embedding_dim
        target_sequence_length = config.source_sequence_length
        source_sequence_length = config.source_sequence_length
        num_heads = config.num_heads
        head_dim = embedding_dim // num_heads

        query = torch.randn(target_sequence_length, batch_size, embedding_dim)
        key = torch.randn(source_sequence_length, batch_size, embedding_dim)
        value = torch.randn(source_sequence_length, batch_size, embedding_dim)
        static_key = None
        static_value = None

        query, key, value = m.prepare_qkv_projection_for_attention(
            query, key, value, static_key, static_value
        )

        expected_q_shape = (batch_size * num_heads, target_sequence_length, head_dim)
        expected_kv_shape = (batch_size * num_heads, source_sequence_length, head_dim)
        self.assertIsInstance(query, torch.Tensor)
        self.assertIsInstance(key, torch.Tensor)
        self.assertIsInstance(value, torch.Tensor)
        self.assertEqual(query.shape, expected_q_shape)
        self.assertEqual(key.shape, expected_kv_shape)
        self.assertEqual(value.shape, expected_kv_shape)

    def test__all_inputs(self):
        c = copy.deepcopy(self.cfg)
        c.multi_head_attention_model_config.add_key_value_bias_flag = True
        config = c.multi_head_attention_model_config
        config.batch_first_flag = True
        model = MultiHeadAttention(c)
        validator = AttentionValidator(config)
        m = AttentionUtils(
            config, validator, model.key_bias_vector, model.value_bias_vector
        )

        batch_size = config.batch_size
        embedding_dim = config.embedding_dim
        target_sequence_length = config.source_sequence_length
        source_sequence_length = config.source_sequence_length
        num_heads = config.num_heads
        head_dim = embedding_dim // num_heads

        query = torch.randn(target_sequence_length, batch_size, embedding_dim)
        key = torch.randn(source_sequence_length, batch_size, embedding_dim)
        value = torch.randn(source_sequence_length, batch_size, embedding_dim)
        static_key = torch.randn(
            batch_size * num_heads, source_sequence_length, head_dim
        )
        static_value = torch.randn(
            batch_size * num_heads, source_sequence_length, head_dim
        )

        query, key, value = m.prepare_qkv_projection_for_attention(
            query, key, value, static_key, static_value
        )

        expected_q_shape = (batch_size * num_heads, source_sequence_length, head_dim)
        expected_kv_shape = (batch_size * num_heads, target_sequence_length, head_dim)
        self.assertIsInstance(query, torch.Tensor)
        self.assertIsInstance(key, torch.Tensor)
        self.assertIsInstance(value, torch.Tensor)
        self.assertEqual(query.shape, expected_q_shape)
        self.assertEqual(key.shape, expected_kv_shape)
        self.assertEqual(value.shape, expected_kv_shape)
        self.assertTrue(torch.equal(key, static_key))
        self.assertTrue(torch.equal(value, static_value))


class Test____concatenate_zeros_tensor(TestAttentionUtils):
    def test__method(self):
        c = copy.deepcopy(self.cfg)
        c.multi_head_attention_model_config.add_key_value_bias_flag = True
        config = c.multi_head_attention_model_config
        config.batch_first_flag = True
        validator = AttentionValidator(config)

        m = AttentionUtils(config, validator)

        batch_size = config.batch_size
        num_heads = config.num_heads
        embedding_dim = config.embedding_dim
        sequence_length = config.source_sequence_length
        head_dim = embedding_dim // num_heads

        tensor = torch.randn(batch_size * num_heads, sequence_length, head_dim)
        output = m._AttentionUtils__concatenate_zeros_tensor(tensor)

        expected_sequence_length = sequence_length + 1
        expected_shape = (batch_size * num_heads, expected_sequence_length, head_dim)
        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(output.shape, expected_shape)


class Test__add_zero_attention(TestAttentionUtils):
    def test__all_inputs__zero_attention_flag__False(self):
        c = copy.deepcopy(self.cfg)
        c.multi_head_attention_model_config.add_key_value_bias_flag = True
        config = c.multi_head_attention_model_config
        config.zero_attention_flag = False
        validator = AttentionValidator(config)
        m = AttentionUtils(config, validator)

        batch_size = config.batch_size
        num_heads = config.num_heads
        embedding_dim = config.embedding_dim
        source_sequence_length = config.source_sequence_length
        target_sequence_length = config.target_sequence_length
        head_dim = embedding_dim // num_heads

        key = torch.randn(batch_size * num_heads, source_sequence_length, head_dim)
        value = torch.randn(batch_size * num_heads, source_sequence_length, head_dim)
        key_padding_mask = torch.randint(0, 2, (batch_size, source_sequence_length))
        attention_mask = torch.randn(
            batch_size * num_heads, target_sequence_length, source_sequence_length
        )

        padded_key, padded_value, output_key_padding_mask, output_attention_mask = (
            m.add_zero_attention(key, value, key_padding_mask, attention_mask)
        )

        self.assertIsInstance(padded_key, torch.Tensor)
        self.assertIsInstance(padded_value, torch.Tensor)
        self.assertTrue(torch.equal(padded_key, key))
        self.assertTrue(torch.equal(padded_value, value))
        self.assertTrue(torch.equal(output_key_padding_mask, key_padding_mask))
        self.assertTrue(torch.equal(output_attention_mask, output_attention_mask))

    def test__kv_inputs_only__zero_attention_flag__True(self):
        c = copy.deepcopy(self.cfg)
        c.multi_head_attention_model_config.add_key_value_bias_flag = True
        config = c.multi_head_attention_model_config
        config.zero_attention_flag = True
        validator = AttentionValidator(config)
        m = AttentionUtils(config, validator)

        batch_size = config.batch_size
        num_heads = config.num_heads
        embedding_dim = config.embedding_dim
        source_sequence_length = config.source_sequence_length
        head_dim = embedding_dim // num_heads

        key = torch.randn(batch_size * num_heads, source_sequence_length, head_dim)
        value = torch.randn(batch_size * num_heads, source_sequence_length, head_dim)
        padded_key, padded_value, key_padding_mask, attention_mask = (
            m.add_zero_attention(key, value)
        )

        expected_sequence_length = source_sequence_length + 1
        expected_shape = (batch_size * num_heads, expected_sequence_length, head_dim)
        self.assertIsInstance(padded_key, torch.Tensor)
        self.assertIsInstance(padded_value, torch.Tensor)
        self.assertIsNone(key_padding_mask)
        self.assertIsNone(attention_mask)
        self.assertEqual(padded_key.shape, expected_shape)
        self.assertEqual(padded_value.shape, expected_shape)

    def test__all_inputs__zero_attention_flag__True(self):
        c = copy.deepcopy(self.cfg)
        c.multi_head_attention_model_config.add_key_value_bias_flag = True
        config = c.multi_head_attention_model_config
        config.zero_attention_flag = True
        validator = AttentionValidator(config)
        m = AttentionUtils(config, validator)

        batch_size = config.batch_size
        num_heads = config.num_heads
        embedding_dim = config.embedding_dim
        source_sequence_length = config.source_sequence_length
        target_sequence_length = config.source_sequence_length
        head_dim = embedding_dim // num_heads

        key = torch.randn(batch_size * num_heads, source_sequence_length, head_dim)
        value = torch.randn(batch_size * num_heads, source_sequence_length, head_dim)
        key_padding_mask = torch.randint(0, 2, (batch_size, source_sequence_length))
        attention_mask = torch.randn(
            batch_size * num_heads, target_sequence_length, source_sequence_length
        )
        padded_key, padded_value, key_padding_mask, attention_mask = (
            m.add_zero_attention(key, value, key_padding_mask, attention_mask)
        )

        expected_sequence_length = source_sequence_length + 1
        expected_shape = (batch_size * num_heads, expected_sequence_length, head_dim)
        expected_key_padding_shape = (batch_size, expected_sequence_length)
        expected_attention_mask_shape = (
            batch_size * num_heads,
            target_sequence_length,
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


class Test____merge_attention_and_padding_mask(TestAttentionUtils):
    def test__inputs_as_None(self):
        c = copy.deepcopy(self.cfg)
        config = c.multi_head_attention_model_config
        validator = AttentionValidator(config)
        m = AttentionUtils(config, validator)

        key_padding_mask = None
        attention_mask = None

        output = m._AttentionUtils__merge_attention_and_padding_mask(
            key_padding_mask, attention_mask
        )
        self.assertIsNone(output)

    def test__attention_mask__None(self):
        c = copy.deepcopy(self.cfg)
        config = c.multi_head_attention_model_config
        validator = AttentionValidator(config)
        m = AttentionUtils(config, validator)

        batch_size = config.batch_size
        num_heads = config.num_heads
        source_sequence_length = config.source_sequence_length

        key_padding_mask = torch.randint(
            0, 1, (batch_size * num_heads, 1, source_sequence_length)
        )
        attention_mask = None

        output = m._AttentionUtils__merge_attention_and_padding_mask(
            key_padding_mask, attention_mask
        )

        self.assertIsInstance(output, torch.Tensor)
        self.assertTrue(torch.equal(output, key_padding_mask))

    def test__all_inputs(self):
        c = copy.deepcopy(self.cfg)
        config = c.multi_head_attention_model_config
        validator = AttentionValidator(config)
        m = AttentionUtils(config, validator)

        batch_size = config.batch_size
        num_heads = config.num_heads
        source_sequence_length = config.source_sequence_length
        target_sequence_length = config.source_sequence_length

        key_padding_mask = torch.randint(
            0, 2, (batch_size * num_heads, 1, source_sequence_length)
        )
        attention_mask = torch.randn(
            batch_size * num_heads, target_sequence_length, source_sequence_length
        )

        output = m._AttentionUtils__merge_attention_and_padding_mask(
            key_padding_mask, attention_mask
        )

        self.assertIsInstance(output, torch.Tensor)
        expected_output = attention_mask + key_padding_mask
        self.assertTrue(torch.equal(output, expected_output))


class Test__merge_masks(TestAttentionUtils):
    def test__inputs_as_None(self):
        c = copy.deepcopy(self.cfg)
        config = c.multi_head_attention_model_config
        validator = AttentionValidator(config)
        m = AttentionUtils(config, validator)

        batch_size = config.batch_size
        num_heads = config.num_heads
        embedding_dim = config.embedding_dim
        source_sequence_length = config.source_sequence_length
        head_dim = embedding_dim // num_heads

        key = torch.randn(batch_size * num_heads, source_sequence_length, head_dim)
        key_padding_mask = None
        attention_mask = None

        output = m.merge_masks(key, key_padding_mask, attention_mask)

        self.assertIsNone(output)

    def test__attention_mask__None(self):
        c = copy.deepcopy(self.cfg)
        config = c.multi_head_attention_model_config
        validator = AttentionValidator(config)
        m = AttentionUtils(config, validator)

        batch_size = config.batch_size
        num_heads = config.num_heads
        embedding_dim = config.embedding_dim
        source_sequence_length = config.source_sequence_length
        head_dim = embedding_dim // num_heads

        key = torch.randn(batch_size * num_heads, source_sequence_length, head_dim)
        key_padding_mask = torch.randint(0, 2, (batch_size, source_sequence_length))
        attention_mask = None

        output = m.merge_masks(key, key_padding_mask, attention_mask)

        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(
            output.shape, (batch_size * num_heads, 1, source_sequence_length)
        )

    def test__key_padding_mask__None(self):
        c = copy.deepcopy(self.cfg)
        config = c.multi_head_attention_model_config
        validator = AttentionValidator(config)
        m = AttentionUtils(config, validator)

        batch_size = config.batch_size
        num_heads = config.num_heads
        embedding_dim = config.embedding_dim
        source_sequence_length = config.source_sequence_length
        target_sequence_length = config.source_sequence_length
        head_dim = embedding_dim // num_heads

        key = torch.randn(batch_size * num_heads, source_sequence_length, head_dim)
        key_padding_mask = None
        attention_mask = torch.randn(
            batch_size * num_heads, target_sequence_length, source_sequence_length
        )
        output = m.merge_masks(key, key_padding_mask, attention_mask)

        self.assertIsInstance(output, torch.Tensor)
        self.assertTrue(torch.equal(output, attention_mask))

    def test__all_inputs(self):
        c = copy.deepcopy(self.cfg)
        config = c.multi_head_attention_model_config
        validator = AttentionValidator(config)
        m = AttentionUtils(config, validator)

        batch_size = config.batch_size
        num_heads = config.num_heads
        embedding_dim = config.embedding_dim
        source_sequence_length = config.source_sequence_length
        target_sequence_length = config.source_sequence_length
        head_dim = embedding_dim // num_heads

        key = torch.randn(batch_size * num_heads, source_sequence_length, head_dim)
        key_padding_mask = torch.randint(0, 2, (batch_size, source_sequence_length))
        attention_mask = torch.randn(
            batch_size * num_heads, target_sequence_length, source_sequence_length
        )

        output = m.merge_masks(key, key_padding_mask, attention_mask)

        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(
            output.shape,
            (batch_size * num_heads, target_sequence_length, source_sequence_length),
        )
