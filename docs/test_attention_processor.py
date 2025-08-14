import math
import copy
import unittest
from unittest.mock import MagicMock, Mock, patch
import torch
import torch.nn as nn
import torch.nn.functional as F
from Emperor.attention.utils.utils import (
    AttentionProcessor,
    AttentionValidator,
)
from Emperor.layers.utils.base import LayerBlock
from Emperor.attention.attention import MultiHeadAttention
from docs.utils import default_unittest_config


class TestAttentionProcessor(unittest.TestCase):
    def setUp(self):
        self.cfg = default_unittest_config()
        self.config = self.cfg.multi_head_attention_model_config

        model = MultiHeadAttention(self.cfg)
        output_model = model.output_model

        validator = AttentionValidator(self.config)
        self.model = AttentionProcessor(self.config, validator, output_model)

        self.batch_size = self.config.batch_size
        self.embedding_dim = self.config.embedding_dim
        self.target_sequence_length = self.config.source_sequence_length
        self.num_heads = self.config.num_heads
        self.head_dim = self.embedding_dim // self.num_heads

    def tearDown(self):
        self.cfg = None
        self.config = None
        self.model = None
        self.batch_size = None
        self.embedding_dim = None
        self.target_sequence_length = None
        self.num_heads = None
        self.head_dim = None


class Test__init(TestAttentionProcessor):
    def test__init(self):
        self.assertIsInstance(self.model, AttentionProcessor)
        self.assertIsInstance(self.model.output_model, LayerBlock)


class Test____scale_query(TestAttentionProcessor):
    def test__method(self):
        query = torch.randn(
            self.batch_size * self.num_heads, self.target_sequence_length, self.head_dim
        )
        scaled_query_tensor = self.model._AttentionProcessor__scale_query(query)

        expected_result = query * math.sqrt(1.0 / float(self.head_dim))
        self.assertIsInstance(scaled_query_tensor, torch.Tensor)
        self.assertEqual(query.shape, scaled_query_tensor.shape)
        self.assertTrue(torch.equal(scaled_query_tensor, expected_result))


class Test____compute_raw_masked_attention_weights(TestAttentionProcessor):
    def test__attention_mask__None(self):
        c = copy.deepcopy(self.cfg)
        config = c.multi_head_attention_model_config
        model = MultiHeadAttention(c)
        output_model = model.output_model

        validator = AttentionValidator(config)
        m = AttentionProcessor(config, validator, output_model)

        batch_size = config.batch_size
        embedding_dim = config.embedding_dim
        target_sequence_length = config.target_sequence_length
        source_sequence_length = config.source_sequence_length
        num_heads = config.num_heads
        head_dim = embedding_dim // num_heads

        query = torch.randn(batch_size * num_heads, target_sequence_length, head_dim)
        key = torch.randn(batch_size * num_heads, source_sequence_length, head_dim)
        attention_mask = None

        raw_unmasked_weights = (
            m._AttentionProcessor__compute_raw_masked_attention_weights(
                query, key, attention_mask
            )
        )

        self.assertIsInstance(raw_unmasked_weights, torch.Tensor)
        self.assertEqual(
            raw_unmasked_weights.shape,
            (batch_size * num_heads, target_sequence_length, source_sequence_length),
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
        c = copy.deepcopy(self.cfg)
        config = c.multi_head_attention_model_config
        model = MultiHeadAttention(c)
        output_model = model.output_model

        validator = AttentionValidator(config)
        m = AttentionProcessor(config, validator, output_model)

        batch_size = config.batch_size
        embedding_dim = config.embedding_dim
        target_sequence_length = config.target_sequence_length
        source_sequence_length = config.source_sequence_length
        num_heads = config.num_heads
        head_dim = embedding_dim // num_heads

        query = torch.randn(batch_size * num_heads, target_sequence_length, head_dim)
        key = torch.randn(batch_size * num_heads, source_sequence_length, head_dim)
        attention_mask = torch.randn(1, target_sequence_length, source_sequence_length)
        attention_mask = torch.where(
            attention_mask > 0, torch.tensor(float("-inf")), torch.tensor(0.0)
        )
        attention_mask = attention_mask.repeat(batch_size * num_heads, 1, 1)

        raw_masked_weights = (
            m._AttentionProcessor__compute_raw_masked_attention_weights(
                query, key, attention_mask
            )
        )

        self.assertIsInstance(raw_masked_weights, torch.Tensor)
        self.assertEqual(
            raw_masked_weights.shape,
            (batch_size * num_heads, target_sequence_length, source_sequence_length),
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


class Test____compute_masked_attention_weights(TestAttentionProcessor):
    def test__method_using_mock(self):
        c = copy.deepcopy(self.cfg)
        config = c.multi_head_attention_model_config
        model = MultiHeadAttention(c)
        output_model = model.output_model

        validator = AttentionValidator(config)
        m = AttentionProcessor(config, validator, output_model)

        batch_size = config.batch_size
        embedding_dim = config.embedding_dim
        target_sequence_length = config.target_sequence_length
        source_sequence_length = config.source_sequence_length
        num_heads = config.num_heads
        head_dim = embedding_dim // num_heads

        mock_scaled_query = torch.randn(
            batch_size * num_heads, target_sequence_length, head_dim
        )
        mock_raw_weights = torch.randn(
            batch_size * num_heads, target_sequence_length, source_sequence_length
        )

        m._AttentionProcessor__scale_query = MagicMock(return_value=mock_scaled_query)
        m._AttentionProcessor__compute_raw_masked_attention_weights = MagicMock(
            return_value=mock_raw_weights
        )

        query = torch.randn(batch_size * num_heads, target_sequence_length, head_dim)
        key = torch.randn(batch_size * num_heads, source_sequence_length, head_dim)
        attention_mask = None

        result = m._AttentionProcessor__compute_masked_attention_weights(
            query, key, attention_mask
        )

        m._AttentionProcessor__scale_query.assert_called_once_with(query)
        m._AttentionProcessor__compute_raw_masked_attention_weights.assert_called_once_with(
            mock_scaled_query, key, attention_mask
        )

        expected = F.softmax(mock_raw_weights, dim=-1)
        self.assertTrue(torch.equal(result, expected))

    def test__attention_mask__None(self):
        c = copy.deepcopy(self.cfg)
        config = c.multi_head_attention_model_config
        model = MultiHeadAttention(c)
        output_model = model.output_model

        validator = AttentionValidator(config)
        m = AttentionProcessor(config, validator, output_model)

        batch_size = config.batch_size
        embedding_dim = config.embedding_dim
        target_sequence_length = config.target_sequence_length
        source_sequence_length = config.source_sequence_length
        num_heads = config.num_heads
        head_dim = embedding_dim // num_heads

        query = torch.randn(batch_size * num_heads, target_sequence_length, head_dim)
        key = torch.randn(batch_size * num_heads, source_sequence_length, head_dim)
        attention_mask = None

        masked_softmax_weights = (
            m._AttentionProcessor__compute_masked_attention_weights(
                query, key, attention_mask
            )
        )
        self.assertIsInstance(masked_softmax_weights, torch.Tensor)
        self.assertFalse((masked_softmax_weights == 0.0).any())
        self.assertEqual(
            masked_softmax_weights.shape,
            (batch_size * num_heads, target_sequence_length, source_sequence_length),
        )

    def test__attention_mask__None__dropout_probability__50percent(self):
        c = copy.deepcopy(self.cfg)
        config = c.multi_head_attention_model_config
        config.dropout_probability = 0.5
        model = MultiHeadAttention(c)
        output_model = model.output_model

        validator = AttentionValidator(config)
        m = AttentionProcessor(config, validator, output_model)

        batch_size = config.batch_size
        embedding_dim = config.embedding_dim
        target_sequence_length = config.target_sequence_length
        source_sequence_length = config.source_sequence_length
        num_heads = config.num_heads
        head_dim = embedding_dim // num_heads

        query = torch.randn(batch_size * num_heads, target_sequence_length, head_dim)
        key = torch.randn(batch_size * num_heads, source_sequence_length, head_dim)
        attention_mask = None

        masked_softmax_weights = (
            m._AttentionProcessor__compute_masked_attention_weights(
                query, key, attention_mask
            )
        )
        self.assertIsInstance(masked_softmax_weights, torch.Tensor)
        self.assertTrue((masked_softmax_weights == 0.0).any())
        self.assertEqual(
            masked_softmax_weights.shape,
            (batch_size * num_heads, target_sequence_length, source_sequence_length),
        )

    def test__all_inputs(self):
        c = copy.deepcopy(self.cfg)
        config = c.multi_head_attention_model_config
        model = MultiHeadAttention(c)
        output_model = model.output_model

        validator = AttentionValidator(config)
        m = AttentionProcessor(config, validator, output_model)

        batch_size = config.batch_size
        embedding_dim = config.embedding_dim
        target_sequence_length = config.target_sequence_length
        source_sequence_length = config.source_sequence_length
        num_heads = config.num_heads
        head_dim = embedding_dim // num_heads

        query = torch.randn(batch_size * num_heads, target_sequence_length, head_dim)
        key = torch.randn(batch_size * num_heads, source_sequence_length, head_dim)
        attention_mask = torch.randn(1, target_sequence_length, source_sequence_length)
        attention_mask = torch.where(
            attention_mask > 0, torch.tensor(float("-inf")), torch.tensor(0.0)
        )
        attention_mask = attention_mask.repeat(batch_size * num_heads, 1, 1)

        masked_softmax_weights = (
            m._AttentionProcessor__compute_masked_attention_weights(
                query, key, attention_mask
            )
        )

        self.assertIsInstance(masked_softmax_weights, torch.Tensor)
        self.assertEqual(
            masked_softmax_weights.shape,
            (batch_size * num_heads, target_sequence_length, source_sequence_length),
        )
        self.assertTrue((masked_softmax_weights == 0.0).any())
        self.assertTrue(
            torch.all((masked_softmax_weights >= 0) & (masked_softmax_weights <= 1))
        )


class Test____compute_weighted_values(TestAttentionProcessor):
    def test__method(self):
        c = copy.deepcopy(self.cfg)
        config = c.multi_head_attention_model_config
        config.target_sequence_length = 32
        config.source_sequence_length = 32

        model = MultiHeadAttention(c)
        output_model = model.output_model

        validator = AttentionValidator(config)
        m = AttentionProcessor(config, validator, output_model)

        batch_size = config.batch_size
        embedding_dim = config.embedding_dim
        target_sequence_length = config.target_sequence_length
        source_sequence_length = config.source_sequence_length
        num_heads = config.num_heads
        head_dim = embedding_dim // num_heads

        attention_weights = torch.randn(
            batch_size * num_heads, target_sequence_length, source_sequence_length
        )
        values = torch.randn(batch_size * num_heads, source_sequence_length, head_dim)

        weighted_values = m._AttentionProcessor__compute_weighted_values(
            attention_weights, values
        )

        self.assertIsInstance(weighted_values, torch.Tensor)
        self.assertEqual(
            weighted_values.shape,
            (batch_size * target_sequence_length, embedding_dim),
        )


class Test____compute_attention_output(TestAttentionProcessor):
    def test__method(self):
        c = copy.deepcopy(self.cfg)
        config = c.multi_head_attention_model_config
        config.target_sequence_length = 32
        config.source_sequence_length = 32
        model = MultiHeadAttention(c)
        output_model = model.output_model

        validator = AttentionValidator(config)
        m = AttentionProcessor(config, validator, output_model)

        batch_size = config.batch_size
        embedding_dim = config.embedding_dim
        target_sequence_length = config.source_sequence_length

        weighted_values = torch.randn(
            batch_size * target_sequence_length, embedding_dim
        )

        weighted_values = m._AttentionProcessor__compute_attention_output(
            weighted_values
        )

        self.assertIsInstance(weighted_values, torch.Tensor)
        self.assertEqual(
            weighted_values.shape, (target_sequence_length, batch_size, embedding_dim)
        )


class Test____maybe_average_attention_weights(TestAttentionProcessor):
    def test__average_attention_weights_flag__False(self):
        c = copy.deepcopy(self.cfg)
        config = c.multi_head_attention_model_config
        config.average_attention_weights = False
        model = MultiHeadAttention(c)
        output_model = model.output_model

        validator = AttentionValidator(config)
        m = AttentionProcessor(config, validator, output_model)

        batch_size = config.batch_size
        target_sequence_length = config.target_sequence_length
        source_sequence_length = config.source_sequence_length
        num_heads = config.num_heads

        attention_weights = torch.randn(
            batch_size, num_heads, target_sequence_length, source_sequence_length
        )

        output_attention_weights = (
            m._AttentionProcessor__maybe_average_attention_weights(attention_weights)
        )

        self.assertIsInstance(output_attention_weights, torch.Tensor)
        self.assertTrue(torch.equal(output_attention_weights, attention_weights))
        self.assertEqual(
            output_attention_weights.shape,
            (batch_size, num_heads, target_sequence_length, source_sequence_length),
        )

    def test__average_attention_weights_flag__True(self):
        c = copy.deepcopy(self.cfg)
        config = c.multi_head_attention_model_config
        config.average_attention_weights_flag = True
        model = MultiHeadAttention(c)
        output_model = model.output_model

        validator = AttentionValidator(config)
        m = AttentionProcessor(config, validator, output_model)

        batch_size = config.batch_size
        target_sequence_length = config.target_sequence_length
        source_sequence_length = config.source_sequence_length
        num_heads = config.num_heads

        attention_weights = torch.randn(
            batch_size, num_heads, target_sequence_length, source_sequence_length
        )

        output_attention_weights = (
            m._AttentionProcessor__maybe_average_attention_weights(attention_weights)
        )

        expected_output = attention_weights.mean(dim=1)
        self.assertIsInstance(output_attention_weights, torch.Tensor)
        self.assertTrue(torch.equal(output_attention_weights, expected_output))
        self.assertEqual(
            output_attention_weights.shape,
            (batch_size, target_sequence_length, source_sequence_length),
        )


class Test____handle_batched_input(TestAttentionProcessor):
    def test__batched_input_flag__False(self):
        c = copy.deepcopy(self.cfg)
        config = c.multi_head_attention_model_config
        model = MultiHeadAttention(c)
        output_model = model.output_model

        validator = AttentionValidator(config)
        validator.batched_input_flag = False
        m = AttentionProcessor(config, validator, output_model)

        batch_size = 1
        embedding_dim = config.embedding_dim
        target_sequence_length = config.target_sequence_length
        source_sequence_length = config.source_sequence_length
        num_heads = config.num_heads

        attention_output = torch.randn(
            target_sequence_length, batch_size, embedding_dim
        )
        attention_weights = torch.randn(
            batch_size, num_heads, target_sequence_length, source_sequence_length
        )

        output_attention_output, output_attention_weights = (
            m._AttentionProcessor__handle_batched_input(
                attention_output, attention_weights
            )
        )

        self.assertIsInstance(output_attention_output, torch.Tensor)
        self.assertIsInstance(output_attention_weights, torch.Tensor)
        self.assertEqual(
            output_attention_output.shape,
            (target_sequence_length, embedding_dim),
        )
        self.assertEqual(
            output_attention_weights.shape,
            (num_heads, target_sequence_length, source_sequence_length),
        )

    def test__batched_input_flag__True(self):
        c = copy.deepcopy(self.cfg)
        config = c.multi_head_attention_model_config
        model = MultiHeadAttention(c)
        output_model = model.output_model

        validator = AttentionValidator(config)
        validator.batched_input_flag = True
        m = AttentionProcessor(config, validator, output_model)

        batch_size = config.batch_size
        embedding_dim = config.embedding_dim
        target_sequence_length = config.target_sequence_length
        source_sequence_length = config.source_sequence_length
        num_heads = config.num_heads

        attention_output = torch.randn(
            target_sequence_length, batch_size, embedding_dim
        )
        attention_weights = torch.randn(
            batch_size, num_heads, target_sequence_length, source_sequence_length
        )

        output_attention_output, output_attention_weights = (
            m._AttentionProcessor__handle_batched_input(
                attention_output, attention_weights
            )
        )

        self.assertIsInstance(output_attention_output, torch.Tensor)
        self.assertIsInstance(output_attention_weights, torch.Tensor)
        self.assertTrue(torch.equal(output_attention_output, attention_output))
        self.assertTrue(torch.equal(output_attention_weights, attention_weights))


class Test____prepare_output(TestAttentionProcessor):
    def test__average_attention_weights_flag__True(self):
        c = copy.deepcopy(self.cfg)
        config = c.multi_head_attention_model_config
        config.average_attention_weights_flag = True
        model = MultiHeadAttention(c)
        output_model = model.output_model

        validator = AttentionValidator(config)
        validator.batched_input_flag = True
        m = AttentionProcessor(config, validator, output_model)

        batch_size = config.batch_size
        embedding_dim = config.embedding_dim
        target_sequence_length = config.target_sequence_length
        source_sequence_length = config.source_sequence_length
        num_heads = config.num_heads

        attention_output = torch.randn(
            target_sequence_length, batch_size, embedding_dim
        )
        attention_weights = torch.randn(
            batch_size * num_heads, target_sequence_length, source_sequence_length
        )

        output_attention_output, output_attention_weights = (
            m._AttentionProcessor__prepare_output(attention_output, attention_weights)
        )

        self.assertIsInstance(output_attention_output, torch.Tensor)
        self.assertIsInstance(output_attention_weights, torch.Tensor)
        self.assertTrue(torch.equal(output_attention_output, attention_output))
        self.assertEqual(
            output_attention_output.shape,
            (target_sequence_length, batch_size, embedding_dim),
        )
        self.assertEqual(
            output_attention_weights.shape,
            (batch_size, target_sequence_length, source_sequence_length),
        )

    def test__average_attention_weights_flag__True__batched_input_flag__False(self):
        c = copy.deepcopy(self.cfg)
        config = c.multi_head_attention_model_config
        config.batch_size = 1
        config.average_attention_weights_flag = True
        model = MultiHeadAttention(c)
        output_model = model.output_model

        validator = AttentionValidator(config)
        validator.batched_input_flag = False
        m = AttentionProcessor(config, validator, output_model)

        batch_size = 1
        embedding_dim = config.embedding_dim
        target_sequence_length = config.target_sequence_length
        source_sequence_length = config.source_sequence_length
        num_heads = config.num_heads

        attention_output = torch.randn(
            target_sequence_length, batch_size, embedding_dim
        )
        attention_weights = torch.randn(
            batch_size * num_heads, target_sequence_length, source_sequence_length
        )

        output_attention_output, output_attention_weights = (
            m._AttentionProcessor__prepare_output(attention_output, attention_weights)
        )

        self.assertIsInstance(output_attention_output, torch.Tensor)
        self.assertIsInstance(output_attention_weights, torch.Tensor)
        self.assertEqual(
            output_attention_output.shape,
            (target_sequence_length, embedding_dim),
        )
        self.assertEqual(
            output_attention_weights.shape,
            (target_sequence_length, source_sequence_length),
        )


class Test____compute_attention_with_weights(TestAttentionProcessor):
    def test__average_attention_weights_flag__True(self):
        c = copy.deepcopy(self.cfg)
        config = c.multi_head_attention_model_config
        config.average_attention_weights_flag = True
        config.target_sequence_length = 32
        config.source_sequence_length = 32

        model = MultiHeadAttention(c)
        output_model = model.output_model

        validator = AttentionValidator(config)
        validator.batched_input_flag = True
        m = AttentionProcessor(config, validator, output_model)

        batch_size = config.batch_size
        embedding_dim = config.embedding_dim
        target_sequence_length = config.target_sequence_length
        source_sequence_length = config.source_sequence_length
        num_heads = config.num_heads
        head_dim = embedding_dim // num_heads

        query = torch.randn(batch_size * num_heads, source_sequence_length, head_dim)
        key = torch.randn(batch_size * num_heads, source_sequence_length, head_dim)
        value = torch.randn(batch_size * num_heads, source_sequence_length, head_dim)
        attention_mask = torch.randn(1, target_sequence_length, source_sequence_length)
        attention_mask = torch.where(
            attention_mask > 0, torch.tensor(float("-inf")), torch.tensor(0.0)
        )
        attention_mask = attention_mask.repeat(batch_size * num_heads, 1, 1)

        output_attention_output, output_attention_weights = (
            m._AttentionProcessor__compute_attention_with_weights(
                query, key, value, attention_mask
            )
        )

        self.assertIsInstance(output_attention_output, torch.Tensor)
        self.assertIsInstance(output_attention_weights, torch.Tensor)
        self.assertEqual(
            output_attention_output.shape,
            (target_sequence_length, batch_size, embedding_dim),
        )
        self.assertEqual(
            output_attention_weights.shape,
            (batch_size, target_sequence_length, source_sequence_length),
        )

    def test__average_attention_weights_flag__False(self):
        c = copy.deepcopy(self.cfg)
        config = c.multi_head_attention_model_config
        config.average_attention_weights_flag = False
        config.target_sequence_length = 32
        config.source_sequence_length = 32

        model = MultiHeadAttention(c)
        output_model = model.output_model

        validator = AttentionValidator(config)
        validator.batched_input_flag = True
        m = AttentionProcessor(config, validator, output_model)

        batch_size = config.batch_size
        embedding_dim = config.embedding_dim
        target_sequence_length = config.target_sequence_length
        source_sequence_length = config.source_sequence_length
        num_heads = config.num_heads
        head_dim = embedding_dim // num_heads

        query = torch.randn(batch_size * num_heads, source_sequence_length, head_dim)
        key = torch.randn(batch_size * num_heads, source_sequence_length, head_dim)
        value = torch.randn(batch_size * num_heads, source_sequence_length, head_dim)
        attention_mask = torch.randn(1, target_sequence_length, source_sequence_length)
        attention_mask = torch.where(
            attention_mask > 0, torch.tensor(float("-inf")), torch.tensor(0.0)
        )
        attention_mask = attention_mask.repeat(batch_size * num_heads, 1, 1)

        output_attention_output, output_attention_weights = (
            m._AttentionProcessor__compute_attention_with_weights(
                query, key, value, attention_mask
            )
        )

        self.assertIsInstance(output_attention_output, torch.Tensor)
        self.assertIsInstance(output_attention_weights, torch.Tensor)
        self.assertEqual(
            output_attention_output.shape,
            (target_sequence_length, batch_size, embedding_dim),
        )
        self.assertEqual(
            output_attention_weights.shape,
            (batch_size, num_heads, target_sequence_length, source_sequence_length),
        )

    def test__attention_mask__None(self):
        c = copy.deepcopy(self.cfg)
        config = c.multi_head_attention_model_config
        config.average_attention_weights_flag = False
        config.target_sequence_length = 32
        config.source_sequence_length = 32

        model = MultiHeadAttention(c)
        output_model = model.output_model

        validator = AttentionValidator(config)
        validator.batched_input_flag = True
        m = AttentionProcessor(config, validator, output_model)

        batch_size = config.batch_size
        embedding_dim = config.embedding_dim
        target_sequence_length = config.target_sequence_length
        source_sequence_length = config.source_sequence_length
        num_heads = config.num_heads
        head_dim = embedding_dim // num_heads

        query = torch.randn(batch_size * num_heads, source_sequence_length, head_dim)
        key = torch.randn(batch_size * num_heads, source_sequence_length, head_dim)
        value = torch.randn(batch_size * num_heads, source_sequence_length, head_dim)
        attention_mask = None

        output_attention_output, output_attention_weights = (
            m._AttentionProcessor__compute_attention_with_weights(
                query, key, value, attention_mask
            )
        )

        self.assertIsInstance(output_attention_output, torch.Tensor)
        self.assertIsInstance(output_attention_weights, torch.Tensor)
        self.assertEqual(
            output_attention_output.shape,
            (target_sequence_length, batch_size, embedding_dim),
        )
        self.assertEqual(
            output_attention_weights.shape,
            (batch_size, num_heads, target_sequence_length, source_sequence_length),
        )


class Test____prepare_attnetion_mask(TestAttentionProcessor):
    def test__attention_mask__None(self):
        c = copy.deepcopy(self.cfg)
        config = c.multi_head_attention_model_config
        model = MultiHeadAttention(c)
        output_model = model.output_model

        validator = AttentionValidator(config)
        validator.batched_input_flag = True
        m = AttentionProcessor(config, validator, output_model)

        attention_mask = None

        output_attention_mask = m._AttentionProcessor__prepare_attnetion_mask(
            attention_mask
        )

        self.assertIsNone(output_attention_mask)

    def test__batched_attention_mask(self):
        c = copy.deepcopy(self.cfg)
        config = c.multi_head_attention_model_config
        config.average_attention_weights_flag = False
        config.target_sequence_length = 32
        config.source_sequence_length = 32

        model = MultiHeadAttention(c)
        output_model = model.output_model

        validator = AttentionValidator(config)
        validator.batched_input_flag = True
        m = AttentionProcessor(config, validator, output_model)

        batch_size = config.batch_size
        num_heads = config.num_heads
        source_sequence_length = config.source_sequence_length
        target_sequence_length = config.target_sequence_length

        attention_mask = None

        attention_mask = torch.randn(1, target_sequence_length, source_sequence_length)
        attention_mask = torch.where(
            attention_mask > 0, torch.tensor(float("-inf")), torch.tensor(0.0)
        )
        attention_mask = attention_mask.repeat(batch_size * num_heads, 1, 1)

        output_attention_mask = m._AttentionProcessor__prepare_attnetion_mask(
            attention_mask
        )

        self.assertIsInstance(output_attention_mask, torch.Tensor)
        self.assertEqual(
            output_attention_mask.shape,
            (batch_size, num_heads, target_sequence_length, source_sequence_length),
        )

    def test__unbatched_attention_mask(self):
        c = copy.deepcopy(self.cfg)
        config = c.multi_head_attention_model_config
        config.average_attention_weights_flag = False
        config.target_sequence_length = 32
        config.source_sequence_length = 32

        model = MultiHeadAttention(c)
        output_model = model.output_model

        validator = AttentionValidator(config)
        validator.batched_input_flag = True
        m = AttentionProcessor(config, validator, output_model)

        source_sequence_length = config.source_sequence_length
        target_sequence_length = config.target_sequence_length

        attention_mask = None

        attention_mask = torch.randn(1, target_sequence_length, source_sequence_length)
        attention_mask = torch.where(
            attention_mask > 0, torch.tensor(float("-inf")), torch.tensor(0.0)
        )

        output_attention_mask = m._AttentionProcessor__prepare_attnetion_mask(
            attention_mask
        )

        self.assertIsInstance(output_attention_mask, torch.Tensor)
        self.assertEqual(
            output_attention_mask.shape,
            (1, 1, target_sequence_length, source_sequence_length),
        )
