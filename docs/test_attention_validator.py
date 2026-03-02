import copy
import torch
import unittest

from dataclasses import asdict
from docs.config import default_unittest_config
from emperor.attention.utils.layer import MultiHeadAttentionConfig
from emperor.attention.utils._validator import MultiHeadAttentionValidator


class TestValidator(unittest.TestCase):
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
        self.query_model = None
        self.key_model = None
        self.value_model = None
        self.qkv_model = None

    def rebuild_presets(self, config: MultiHeadAttentionConfig | None = None):
        self.cfg = default_unittest_config()
        self.config = self.cfg.multi_head_attention_model_config
        if config is not None:
            for k in asdict(config):
                if hasattr(self.config, k) and getattr(config, k) is not None:
                    setattr(self.config, k, getattr(config, k))

        self.model = MultiHeadAttentionValidator(self.config)

        self.batch_size = self.config.batch_size
        self.embedding_dim = self.config.embedding_dim
        self.target_sequence_length = self.config.target_sequence_length
        self.source_sequence_length = self.config.source_sequence_length
        self.num_heads = self.config.num_heads
        self.head_dim = self.embedding_dim // self.num_heads


class Test___check_query_dims(TestValidator):
    def test_incorrect_1D_tensor(self):
        query = torch.randn(self.embedding_dim)
        with self.assertRaises(RuntimeError) as context:
            self.model._MultiHeadAttentionValidator__check_query_dims(query)

    def test_correct_2D_tensor(self):
        query = torch.randn(self.target_sequence_length, self.embedding_dim)

        output = self.model._MultiHeadAttentionValidator__check_query_dims(query)
        self.assertIsNone(output)

    def test_correct_3D_tensor(self):
        query = torch.randn(
            self.target_sequence_length, self.batch_size, self.embedding_dim
        )

        output = self.model._MultiHeadAttentionValidator__check_query_dims(query)
        self.assertIsNone(output)

    def test_incorrect_4D_tensor(self):
        query = torch.randn(
            self.target_sequence_length,
            self.batch_size,
            self.embedding_dim,
            self.embedding_dim,
        )
        with self.assertRaises(RuntimeError) as context:
            self.model._MultiHeadAttentionValidator__check_query_dims(query)


class Test___check_query_key_value_dimension_count(TestValidator):
    def test_check_incorrect_input_dim_count_with_batched_input_flag_set_to_False(
        self,
    ):
        self.model.batched_input_flag = False

        key = torch.randn(
            self.source_sequence_length, self.batch_size, self.embedding_dim
        )
        value = torch.randn(
            self.source_sequence_length * self.batch_size, self.embedding_dim
        )
        with self.assertRaises(RuntimeError) as context:
            self.model._MultiHeadAttentionValidator__check_query_key_value_dimension_count(
                key, value
            )

    def test_check_incorrect_input_dim_count_with_batched_input_flag_True(
        self,
    ):
        self.model.batched_input_flag = True
        key = torch.randn(self.source_sequence_length, self.embedding_dim)
        value = torch.randn(self.source_sequence_length, self.embedding_dim)
        with self.assertRaises(RuntimeError) as context:
            self.model._MultiHeadAttentionValidator__check_query_key_value_dimension_count(
                key, value
            )

    def test_check_correct_input_dim_count_with_batched_input_flag_set_to_False(
        self,
    ):
        self.model.batched_input_flag = False

        key = torch.randn(self.source_sequence_length, self.embedding_dim)
        value = torch.randn(self.source_sequence_length, self.embedding_dim)
        output = self.model._MultiHeadAttentionValidator__check_query_key_value_dimension_count(
            key, value
        )
        self.assertIsNone(output)

    def test_check_correct_input_dim_count_with_batched_input_flag_set_to_True(
        self,
    ):
        self.model.batched_input_flag = True

        key = torch.randn(
            self.source_sequence_length, self.batch_size, self.embedding_dim
        )
        value = torch.randn(
            self.source_sequence_length, self.batch_size, self.embedding_dim
        )
        output = self.model._MultiHeadAttentionValidator__check_query_key_value_dimension_count(
            key, value
        )
        self.assertIsNone(output)


class Test___check_key_padding_mask_dimension_count(TestValidator):
    def test_no_padding_mask_input(self):
        output = self.model._MultiHeadAttentionValidator__check_key_padding_mask_dimension_count()
        self.assertIsNone(output)

    def test_incorrect_3D_input_padding_mask(self):
        key_padding_mask = torch.randn(
            self.source_sequence_length, self.batch_size, self.embedding_dim
        )
        with self.assertRaises(RuntimeError) as context:
            self.model._MultiHeadAttentionValidator__check_key_padding_mask_dimension_count(
                key_padding_mask
            )

    def test_correct_2D_padding_mask_input_with_batched_input_flag_set_to_True(self):
        self.model.batched_input_flag = True
        key_padding_mask = torch.randn(self.batch_size, self.source_sequence_length)
        output = self.model._MultiHeadAttentionValidator__check_key_padding_mask_dimension_count(
            key_padding_mask
        )
        self.assertIsNone(output)

    def test_correct_1D_padding_mask_input_with_batched_input_flag_set_to_False(self):
        self.model.batched_input_flag = False
        key_padding_mask = torch.randn(self.source_sequence_length)
        output = self.model._MultiHeadAttentionValidator__check_key_padding_mask_dimension_count(
            key_padding_mask
        )
        self.assertIsNone(output)


class Test___check_attention_mask_dim_count_and_shape(TestValidator):
    def test_no_attention_mask_input(self):
        output = self.model._MultiHeadAttentionValidator__check_attention_mask_dim_count_and_shape()
        self.assertIsNone(output)

    def test_incorrect_1D_input(self):
        attention_mask = torch.randn(self.source_sequence_length)
        with self.assertRaises(RuntimeError) as context:
            self.model._MultiHeadAttentionValidator__check_attention_mask_dim_count_and_shape(
                attention_mask
            )

    def test_input_with_correct_2D_dim_count_correct_shape(self):
        attention_mask = torch.randn(
            self.target_sequence_length, self.source_sequence_length
        )

        output = self.model._MultiHeadAttentionValidator__check_attention_mask_dim_count_and_shape(
            attention_mask
        )
        self.assertIsNone(output)

    def test_input_with_correct_3D_dim_count_but_correct_shape(self):
        attention_mask = torch.randn(
            self.batch_size * self.num_heads,
            self.target_sequence_length,
            self.source_sequence_length,
        )

        output = self.model._MultiHeadAttentionValidator__check_attention_mask_dim_count_and_shape(
            attention_mask
        )
        self.assertIsNone(output)

    def test_incorrect_4D_input(self):
        attention_mask = torch.randn(
            self.num_heads,
            self.batch_size,
            self.source_sequence_length,
            self.target_sequence_length,
        )

        with self.assertRaises(RuntimeError) as context:
            self.model._MultiHeadAttentionValidator__check_attention_mask_dim_count_and_shape(
                attention_mask
            )


class Test_check_attention_input_shapes(TestValidator):
    def test_all_inputs_batched(self):
        q_input_shape = (
            self.target_sequence_length,
            self.batch_size,
            self.embedding_dim,
        )
        kv_input_shape = (
            self.source_sequence_length,
            self.batch_size,
            self.embedding_dim,
        )
        query = torch.randn(q_input_shape)
        key = torch.randn(kv_input_shape)
        value = torch.randn(kv_input_shape)

        key_padding_mask = torch.randint(
            0, 2, (self.batch_size, self.source_sequence_length)
        )
        attention_mask = torch.randn(
            self.batch_size * self.num_heads,
            self.target_sequence_length,
            self.source_sequence_length,
        )

        output = self.model.check_attention_input_shapes(
            query, key, value, key_padding_mask, attention_mask
        )

        self.assertTrue(output)

    def test_all_inputs_not_batched(self):
        q_input_shape = (
            self.target_sequence_length,
            self.embedding_dim,
        )
        kv_input_shape = (
            self.source_sequence_length,
            self.embedding_dim,
        )
        query = torch.randn(q_input_shape)
        key = torch.randn(kv_input_shape)
        value = torch.randn(kv_input_shape)
        key_padding_mask = torch.randint(0, 2, (self.source_sequence_length,))
        attention_mask = torch.randn(
            self.target_sequence_length,
            self.source_sequence_length,
        )

        output = self.model.check_attention_input_shapes(
            query, key, value, key_padding_mask, attention_mask
        )

        self.assertFalse(output)

    def test_all_inputs_no_attention_mask(self):
        q_input_shape = (
            self.target_sequence_length,
            self.batch_size,
            self.embedding_dim,
        )
        kv_input_shape = (
            self.source_sequence_length,
            self.batch_size,
            self.embedding_dim,
        )
        query = torch.randn(q_input_shape)
        key = torch.randn(kv_input_shape)
        value = torch.randn(kv_input_shape)

        key_padding_mask = torch.randint(
            0, 2, (self.batch_size, self.source_sequence_length)
        )

        output = self.model.check_attention_input_shapes(
            query, key, value, key_padding_mask
        )

        self.assertTrue(output)

    def test_no_key_and_attention_masks(self):
        q_input_shape = (
            self.target_sequence_length,
            self.batch_size,
            self.embedding_dim,
        )
        kv_input_shape = (
            self.source_sequence_length,
            self.batch_size,
            self.embedding_dim,
        )
        query = torch.randn(q_input_shape)
        key = torch.randn(kv_input_shape)
        value = torch.randn(kv_input_shape)

        output = self.model.check_attention_input_shapes(query, key, value)

        self.assertTrue(output)


class Test_check_self_attention_projection_inputs(TestValidator):
    def test__method(self):
        c = copy.deepcopy(self.cfg)
        config = c.multi_head_attention_model_config
        m = MultiHeadAttentionValidator(config)

        batch_size = config.batch_size
        source_sequence_length = config.source_sequence_length
        embedding_dim = config.embedding_dim

        key = torch.randn(source_sequence_length, batch_size, embedding_dim)
        value = torch.randn(source_sequence_length, batch_size, embedding_dim)

        output = m.check_self_attention_projection_inputs(key, value)

        self.assertIsNone(output)

    def test__is_error_raised(self):
        c = copy.deepcopy(self.cfg)
        config = c.multi_head_attention_model_config
        m = MultiHeadAttentionValidator(config)

        batch_size = config.batch_size
        source_sequence_length = config.source_sequence_length
        embedding_dim = config.embedding_dim

        changed_sequence_length = source_sequence_length + 1
        key = torch.randn(changed_sequence_length, batch_size, embedding_dim)
        value = torch.randn(source_sequence_length, batch_size, embedding_dim)

        with self.assertRaises(RuntimeError) as context:
            m.check_self_attention_projection_inputs(key, value)


class Test_check_indepentent_projections_inputs(TestValidator):
    def test__method(self):
        c = copy.deepcopy(self.cfg)
        config = c.multi_head_attention_model_config
        m = MultiHeadAttentionValidator(config)

        batch_size = config.batch_size
        source_sequence_length = config.source_sequence_length
        embedding_dim = config.embedding_dim

        key = torch.randn(source_sequence_length, batch_size, embedding_dim)
        value = torch.randn(source_sequence_length, batch_size, embedding_dim)

        output = m.check_self_attention_projection_inputs(key, value)

        self.assertIsNone(output)

    def test__is_error_raised(self):
        c = copy.deepcopy(self.cfg)
        config = c.multi_head_attention_model_config
        m = MultiHeadAttentionValidator(config)

        batch_size = config.batch_size
        source_sequence_length = config.source_sequence_length
        embedding_dim = config.embedding_dim

        changed_sequence_length = source_sequence_length + 1
        key = torch.randn(changed_sequence_length, batch_size, embedding_dim)
        value = torch.randn(source_sequence_length, batch_size, embedding_dim)

        with self.assertRaises(RuntimeError) as context:
            m.check_self_attention_projection_inputs(key, value)


class Test___resolve_static_projection_type(TestValidator):
    def test__value_tensor_flag__False(self):
        c = copy.deepcopy(self.cfg)
        config = c.multi_head_attention_model_config
        m = MultiHeadAttentionValidator(config)

        value_tensor_flag = False

        output = m._MultiHeadAttentionValidator__resolve_static_projection_type(
            value_tensor_flag
        )

        self.assertEqual(output, "static_keys")

    def test__value_tensor_flag__True(self):
        c = copy.deepcopy(self.cfg)
        config = c.multi_head_attention_model_config
        m = MultiHeadAttentionValidator(config)

        value_tensor_flag = True

        output = m._MultiHeadAttentionValidator__resolve_static_projection_type(
            value_tensor_flag
        )
        self.assertEqual(output, "static_values")


class Test___resolve_static_projection_shape(TestValidator):
    def test_no_input_tensor(self):
        static_tensor = None
        output = (
            self.model._MultiHeadAttentionValidator__resolve_static_projection_shape(
                static_tensor
            )
        )

        self.assertIsNone(output)

    def test_correct_input_tensor(self):
        static_tensor = torch.randn(
            self.batch_size * self.num_heads, self.source_sequence_length, self.head_dim
        )
        output = (
            self.model._MultiHeadAttentionValidator__resolve_static_projection_shape(
                static_tensor
            )
        )

        self.assertIsNone(output)

    def test_input_tensor_with_incorrect_shape(self):
        wrong_static_tensor = torch.randn(
            self.batch_size, self.source_sequence_length, self.head_dim
        )
        with self.assertRaises(AssertionError) as context:
            self.model._MultiHeadAttentionValidator__resolve_static_projection_shape(
                wrong_static_tensor
            )


class Test_check_static_projection_shapes(TestValidator):
    def test_no_inputs(self):
        static_keys = None
        static_values = None

        output = self.model.check_static_projection_shapes(static_keys, static_values)

        self.assertIsNone(output)

    def test_check_correct_static_projection_inputs(self):
        static_keys = torch.randn(
            self.batch_size * self.num_heads,
            self.source_sequence_length,
            self.head_dim,
            self.embedding_dim,
        )
        static_values = torch.randn(
            self.batch_size * self.num_heads,
            self.source_sequence_length,
            self.head_dim,
            self.embedding_dim,
        )

        output = self.model.check_static_projection_shapes(static_keys, static_values)
        self.assertIsNone(output)

    def test_check_wrong_static_projection_inputs(self):
        static_keys = torch.randn(
            self.batch_size,
            self.source_sequence_length,
            self.head_dim,
            self.embedding_dim,
        )
        static_values = torch.randn(
            self.batch_size * self.num_heads,
            self.source_sequence_length,
            self.head_dim,
            self.embedding_dim,
        )

        with self.assertRaises(AssertionError) as context:
            self.model.check_static_projection_shapes(static_keys, static_values)
