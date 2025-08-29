import copy
import torch
import unittest
from dataclasses import asdict
from Emperor.attention.utils.utils import (
    AttentionProjector,
    AttentionValidator,
)
from Emperor.attention.attention import MultiHeadAttention, MultiHeadAttentionConfig
from docs.utils import default_unittest_config


class TestAttentionValidator(unittest.TestCase):
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

        self.model = AttentionProjector(
            self.config,
            self.cfg,
        )

        self.batch_size = self.config.batch_size
        self.embedding_dim = self.config.embedding_dim
        self.target_sequence_length = self.config.target_sequence_length
        self.source_sequence_length = self.config.source_sequence_length
        self.num_heads = self.config.num_heads
        self.head_dim = self.embedding_dim // self.num_heads


class Test___check_query_dims(TestAttentionValidator):
    def test__incorrect_1D_tensor(self):
        c = copy.deepcopy(self.cfg)
        config = c.multi_head_attention_model_config
        m = AttentionValidator(config)

        test_dim = config.embedding_dim

        query = torch.randn(test_dim)
        with self.assertRaises(RuntimeError) as context:
            m._AttentionValidator__check_query_dims(query)

    def test__correct_2D_tensor(self):
        c = copy.deepcopy(self.cfg)
        config = c.multi_head_attention_model_config
        m = AttentionValidator(config)

        target_sequence_length = m.target_sequence_length
        embedding_dim = config.embedding_dim

        query = torch.randn(target_sequence_length, embedding_dim)

        output = m._AttentionValidator__check_query_dims(query)
        self.assertIsNone(output)

    def test__correct_3D_tensor(self):
        c = copy.deepcopy(self.cfg)
        config = c.multi_head_attention_model_config
        m = AttentionValidator(config)

        batch_size = m.batch_size
        target_sequence_length = m.target_sequence_length
        embedding_dim = config.embedding_dim

        query = torch.randn(target_sequence_length, batch_size, embedding_dim)

        output = m._AttentionValidator__check_query_dims(query)
        self.assertIsNone(output)


class Test___check_query_key_value_dimensions(TestAttentionValidator):
    def test__batched_input_flag__False__incorrect_input_dim(
        self,
    ):
        c = copy.deepcopy(self.cfg)
        config = c.multi_head_attention_model_config
        m = AttentionValidator(config)
        m.batched_input_flag = False

        batch_size = m.batch_size
        source_sequence_length = m.source_sequence_length
        embedding_dim = config.embedding_dim

        key = torch.randn(source_sequence_length, batch_size, embedding_dim)
        value = torch.randn(source_sequence_length * batch_size, embedding_dim)
        with self.assertRaises(RuntimeError) as context:
            m._AttentionValidator__check_query_key_value_dimensions(key, value)

    def test__batched_input_flag__True__incorrect_input_dim(
        self,
    ):
        c = copy.deepcopy(self.cfg)
        config = c.multi_head_attention_model_config
        m = AttentionValidator(config)
        m.batched_input_flag = True

        batch_size = m.batch_size
        source_sequence_length = m.source_sequence_length
        embedding_dim = config.embedding_dim

        key = torch.randn(source_sequence_length, batch_size, embedding_dim)
        value = torch.randn(source_sequence_length * batch_size, embedding_dim)
        with self.assertRaises(RuntimeError) as context:
            m._AttentionValidator__check_query_key_value_dimensions(key, value)

    def test__batched_input_flag__False__correct_input_dim(
        self,
    ):
        c = copy.deepcopy(self.cfg)
        config = c.multi_head_attention_model_config
        m = AttentionValidator(config)
        m.batched_input_flag = False

        batch_size = m.batch_size
        source_sequence_length = m.source_sequence_length
        embedding_dim = config.embedding_dim

        key = torch.randn(source_sequence_length * batch_size, embedding_dim)
        value = torch.randn(source_sequence_length * batch_size, embedding_dim)
        output = m._AttentionValidator__check_query_key_value_dimensions(key, value)
        self.assertIsNone(output)

    def test__batched_input_flag__True__correct_input_dim(
        self,
    ):
        c = copy.deepcopy(self.cfg)
        config = c.multi_head_attention_model_config
        m = AttentionValidator(config)
        m.batched_input_flag = True

        batch_size = m.batch_size
        source_sequence_length = m.source_sequence_length
        embedding_dim = config.embedding_dim

        key = torch.randn(source_sequence_length, batch_size, embedding_dim)
        value = torch.randn(source_sequence_length, batch_size, embedding_dim)
        output = m._AttentionValidator__check_query_key_value_dimensions(key, value)
        self.assertIsNone(output)


class Test___check_key_padding_mask_dimensions(TestAttentionValidator):
    def test__None_as_input(self):
        c = copy.deepcopy(self.cfg)
        config = c.multi_head_attention_model_config
        m = AttentionValidator(config)

        output = m._AttentionValidator__check_key_padding_mask_dimensions(None)
        self.assertIsNone(output)

    def test__incorrect_3D_key_padding_mask_dims(self):
        c = copy.deepcopy(self.cfg)
        config = c.multi_head_attention_model_config
        m = AttentionValidator(config)

        batch_size = m.batch_size
        source_sequence_length = m.source_sequence_length
        embedding_dim = config.embedding_dim

        key_padding_mask = torch.randn(
            source_sequence_length, batch_size, embedding_dim
        )
        with self.assertRaises(RuntimeError) as context:
            m._AttentionValidator__check_key_padding_mask_dimensions(key_padding_mask)

    def test__batched_input_flag__True__with__2D_key_padding_mask_shape(self):
        c = copy.deepcopy(self.cfg)
        config = c.multi_head_attention_model_config
        m = AttentionValidator(config)
        m.batched_input_flag = True

        batch_size = m.batch_size
        source_sequence_length = m.source_sequence_length

        key_padding_mask = torch.randn(batch_size, source_sequence_length)
        output = m._AttentionValidator__check_key_padding_mask_dimensions(
            key_padding_mask
        )
        self.assertIsNone(output)

    def test__batched_input_flag__False__with__1D_key_padding_mask(self):
        c = copy.deepcopy(self.cfg)
        config = c.multi_head_attention_model_config
        m = AttentionValidator(config)
        m.batched_input_flag = False

        source_sequence_length = m.source_sequence_length

        key_padding_mask = torch.randn(source_sequence_length)
        output = m._AttentionValidator__check_key_padding_mask_dimensions(
            key_padding_mask
        )
        self.assertIsNone(output)


class Test___check_attention_mask(TestAttentionValidator):
    def test__None_as_input(self):
        c = copy.deepcopy(self.cfg)
        config = c.multi_head_attention_model_config
        m = AttentionValidator(config)
        m.batched_input_flag = True

        output = m._AttentionValidator__check_attention_mask(None)
        self.assertIsNone(output)

    def test__1D_incorrect_input_dims(self):
        c = copy.deepcopy(self.cfg)
        config = c.multi_head_attention_model_config
        m = AttentionValidator(config)
        m.batched_input_flag = True

        source_sequence_length = m.source_sequence_length

        attention_mask = torch.randn(source_sequence_length)
        with self.assertRaises(RuntimeError) as context:
            m._AttentionValidator__check_attention_mask(attention_mask)

    def test__2D__incorrect_mask_shape__correct_input_dims(self):
        c = copy.deepcopy(self.cfg)
        config = c.multi_head_attention_model_config
        m = AttentionValidator(config)
        m.batched_input_flag = True

        source_sequence_length = 12
        target_sequence_length = m.target_sequence_length

        attention_mask = torch.randn(source_sequence_length, target_sequence_length)

        with self.assertRaises(RuntimeError) as context:
            m._AttentionValidator__check_attention_mask(attention_mask)

    def test__2D__correct_mask_shape__correct_input_dims(self):
        c = copy.deepcopy(self.cfg)
        config = c.multi_head_attention_model_config
        m = AttentionValidator(config)
        m.batched_input_flag = True

        source_sequence_length = m.source_sequence_length
        target_sequence_length = m.target_sequence_length

        attention_mask = torch.randn(target_sequence_length, source_sequence_length)

        output = m._AttentionValidator__check_attention_mask(attention_mask)
        self.assertIsNone(output)

    def test__3D__incorrect_mask_shape__correct_input_dims(self):
        c = copy.deepcopy(self.cfg)
        config = c.multi_head_attention_model_config
        m = AttentionValidator(config)
        m.batched_input_flag = True

        batch_size = 2
        num_heads = 3
        source_sequence_length = 12
        target_sequence_length = m.target_sequence_length

        attention_mask = torch.randn(
            batch_size * num_heads, source_sequence_length, target_sequence_length
        )

        with self.assertRaises(RuntimeError) as context:
            m._AttentionValidator__check_attention_mask(attention_mask)

    def test__3D__correct_mask_shape__correct_input_dims(
        self,
    ):
        c = copy.deepcopy(self.cfg)
        config = c.multi_head_attention_model_config
        m = AttentionValidator(config)
        m.batched_input_flag = True

        batch_size = m.batch_size
        num_heads = m.num_heads
        source_sequence_length = m.source_sequence_length
        target_sequence_length = m.target_sequence_length

        attention_mask = torch.randn(
            batch_size * num_heads,
            target_sequence_length,
            source_sequence_length,
        )

        output = m._AttentionValidator__check_attention_mask(attention_mask)
        self.assertIsNone(output)

    def test__4D_incorrect_input_dims(self):
        c = copy.deepcopy(self.cfg)
        config = c.multi_head_attention_model_config
        m = AttentionValidator(config)
        m.batched_input_flag = True

        batch_size = 2
        num_heads = 3
        source_sequence_length = 12
        target_sequence_length = m.target_sequence_length

        attention_mask = torch.randn(
            num_heads,
            batch_size,
            source_sequence_length,
            target_sequence_length,
        )

        with self.assertRaises(RuntimeError) as context:
            m._AttentionValidator__check_attention_mask(attention_mask)


class Test___resolve_attention_mask_shape(TestAttentionValidator):
    def test__ensure_correct_shape_for_2D_mask(self):
        c = copy.deepcopy(self.cfg)
        config = c.multi_head_attention_model_config
        m = AttentionValidator(config)
        m.batched_input_flag = True

        source_sequence_length = m.source_sequence_length
        target_sequence_length = m.target_sequence_length

        expected_attention_mask_shape = (
            target_sequence_length,
            source_sequence_length,
        )

        attention_mask = torch.randn(
            target_sequence_length,
            source_sequence_length,
        )

        output = m._AttentionValidator__resolve_attention_mask_shape(attention_mask)
        self.assertEqual(output, expected_attention_mask_shape)

    def test__ensure_correct_shape_for_3D_mask(self):
        c = copy.deepcopy(self.cfg)
        config = c.multi_head_attention_model_config
        m = AttentionValidator(config)
        m.batched_input_flag = True

        batch_size = m.batch_size
        num_heads = m.num_heads
        source_sequence_length = m.source_sequence_length
        target_sequence_length = m.target_sequence_length

        expected_attention_mask_shape = (
            batch_size * num_heads,
            target_sequence_length,
            source_sequence_length,
        )

        attention_mask = torch.randn(
            batch_size * num_heads,
            target_sequence_length,
            source_sequence_length,
        )

        output = m._AttentionValidator__resolve_attention_mask_shape(attention_mask)
        self.assertEqual(output, expected_attention_mask_shape)


class Test___ensure_attention_mask_if_causal(TestAttentionValidator):
    def test__None_as_input(self):
        c = copy.deepcopy(self.cfg)
        config = c.multi_head_attention_model_config
        m = AttentionValidator(config)
        m.causal_attention_mask_flag = True

        with self.assertRaises(RuntimeError) as context:
            m._AttentionValidator__ensure_attention_mask_if_causal(None)

    def test__causal_attention_mask_flag__True__and__None_as_input(self):
        c = copy.deepcopy(self.cfg)
        config = c.multi_head_attention_model_config
        m = AttentionValidator(config)
        m.causal_attention_mask_flag = True

        with self.assertRaises(RuntimeError) as context:
            m._AttentionValidator__ensure_attention_mask_if_causal(None)

    def test__causal_attention_mask_flag__False__and__attention_mask(self):
        c = copy.deepcopy(self.cfg)
        config = c.multi_head_attention_model_config
        m = AttentionValidator(config)
        m.causal_attention_mask_flag = False

        batch_size = m.batch_size
        num_heads = m.num_heads
        source_sequence_length = m.source_sequence_length
        target_sequence_length = m.target_sequence_length

        attention_mask = torch.randn(
            batch_size * num_heads,
            source_sequence_length,
            target_sequence_length,
        )

        output = m._AttentionValidator__ensure_attention_mask_if_causal(attention_mask)
        self.assertIsNone(output)


class Test_multi_head_attention_input_shapes(TestAttentionValidator):
    def test__all_inputs_batched(self):
        c = copy.deepcopy(self.cfg)
        config = c.multi_head_attention_model_config
        m = AttentionValidator(config)

        batch_size = m.batch_size
        num_heads = m.num_heads
        source_sequence_length = m.source_sequence_length
        target_sequence_length = m.target_sequence_length
        embedding_dim = config.embedding_dim

        query = torch.randn(target_sequence_length, batch_size, embedding_dim)
        key = torch.randn(source_sequence_length, batch_size, embedding_dim)
        value = torch.randn(source_sequence_length, batch_size, embedding_dim)
        key_padding_mask = torch.randint(0, 2, (batch_size, source_sequence_length))
        attention_mask = torch.randn(
            batch_size * num_heads,
            target_sequence_length,
            source_sequence_length,
        )

        output = m.multi_head_attention_input_shapes(
            query, key, value, key_padding_mask, attention_mask
        )

        self.assertTrue(output)

    def test__all_inputs_not_batched(self):
        c = copy.deepcopy(self.cfg)
        config = c.multi_head_attention_model_config
        m = AttentionValidator(config)

        source_sequence_length = m.source_sequence_length
        target_sequence_length = m.target_sequence_length
        embedding_dim = config.embedding_dim

        query = torch.randn(target_sequence_length, embedding_dim)
        key = torch.randn(source_sequence_length, embedding_dim)
        value = torch.randn(source_sequence_length, embedding_dim)
        key_padding_mask = torch.randint(0, 2, (source_sequence_length,))
        attention_mask = torch.randn(target_sequence_length, source_sequence_length)

        output = m.multi_head_attention_input_shapes(
            query, key, value, key_padding_mask, attention_mask
        )

        self.assertFalse(output)

    def test__no_key_and_attention_masks(self):
        c = copy.deepcopy(self.cfg)
        config = c.multi_head_attention_model_config
        m = AttentionValidator(config)

        batch_size = m.batch_size
        source_sequence_length = m.source_sequence_length
        target_sequence_length = m.target_sequence_length
        embedding_dim = config.embedding_dim

        query = torch.randn(target_sequence_length, batch_size, embedding_dim)
        key = torch.randn(source_sequence_length, batch_size, embedding_dim)
        value = torch.randn(source_sequence_length, batch_size, embedding_dim)

        output = m.multi_head_attention_input_shapes(query, key, value)

        self.assertTrue(output)


class Test_is_mask_float_or_bool(TestAttentionValidator):
    def test__incorect_int_input(self):
        c = copy.deepcopy(self.cfg)
        config = c.multi_head_attention_model_config
        m = AttentionValidator(config)
        m.batched_input_flag = False

        mask = torch.randint(0, 20, (10, 10))
        maks_name = "test_mask"

        with self.assertRaises(RuntimeError) as context:
            m.is_mask_float_or_bool(mask, maks_name)

    def test__correct_float_input(self):
        c = copy.deepcopy(self.cfg)
        config = c.multi_head_attention_model_config
        m = AttentionValidator(config)
        m.batched_input_flag = False

        mask = torch.randn(10, 10)
        maks_name = "test_mask"

        output = m.is_mask_float_or_bool(mask, maks_name)
        self.assertIsNone(output)

    def test__correct_boolean_input(self):
        c = copy.deepcopy(self.cfg)
        config = c.multi_head_attention_model_config
        m = AttentionValidator(config)
        m.batched_input_flag = False

        mask = torch.randn(10, 10) > 0
        maks_name = "test_mask"

        output = m.is_mask_float_or_bool(mask, maks_name)
        self.assertIsNone(output)


class Test_is_mask_correct_dtype(TestAttentionValidator):
    def test__incorrect__other_dtype__check_other__True(self):
        c = copy.deepcopy(self.cfg)
        config = c.multi_head_attention_model_config
        m = AttentionValidator(config)
        m.batched_input_flag = False

        mask = torch.randn(10, 10, dtype=torch.float64)
        maks_name = "test_mask"
        other_type = torch.float32
        other_name = "real_maks_dtype"
        check_other = True

        with self.assertRaises(RuntimeError) as context:
            m.is_mask_correct_dtype(
                mask, maks_name, other_type, other_name, check_other
            )

    def test__incorrect__other_dtype__check_other__False(self):
        c = copy.deepcopy(self.cfg)
        config = c.multi_head_attention_model_config
        m = AttentionValidator(config)
        m.batched_input_flag = False

        mask = torch.randn(10, 10, dtype=torch.float64)
        maks_name = "test_mask"
        other_type = torch.float32
        other_name = "real_maks_dtype"
        check_other = False

        output = m.is_mask_correct_dtype(
            mask, maks_name, other_type, other_name, check_other
        )
        self.assertIsNone(output)

    def test__mask__and__other_type__same_dtype(self):
        c = copy.deepcopy(self.cfg)
        config = c.multi_head_attention_model_config
        m = AttentionValidator(config)
        m.batched_input_flag = False

        mask = torch.randn(10, 10, dtype=torch.float32)
        maks_name = "test_mask"
        other_type = torch.float32
        other_name = "real_maks_dtype"
        check_other = True

        output = m.is_mask_correct_dtype(
            mask, maks_name, other_type, other_name, check_other
        )
        self.assertIsNone(output)

    def test__other_type__None__check_other__True(self):
        c = copy.deepcopy(self.cfg)
        config = c.multi_head_attention_model_config
        m = AttentionValidator(config)
        m.batched_input_flag = False

        mask = torch.randn(10, 10, dtype=torch.float32)
        maks_name = "test_mask"
        other_type = None
        other_name = "real_maks_dtype"
        check_other = True

        output = m.is_mask_correct_dtype(
            mask, maks_name, other_type, other_name, check_other
        )
        self.assertIsNone(output)


class Test_check_self_attention_projection_inputs(TestAttentionValidator):
    def test__method(self):
        c = copy.deepcopy(self.cfg)
        config = c.multi_head_attention_model_config
        m = AttentionValidator(config)

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
        m = AttentionValidator(config)

        batch_size = config.batch_size
        source_sequence_length = config.source_sequence_length
        embedding_dim = config.embedding_dim

        changed_sequence_length = source_sequence_length + 1
        key = torch.randn(changed_sequence_length, batch_size, embedding_dim)
        value = torch.randn(source_sequence_length, batch_size, embedding_dim)

        with self.assertRaises(RuntimeError) as context:
            m.check_self_attention_projection_inputs(key, value)


class Test_check_indepentent_projections_inputs(TestAttentionValidator):
    def test__method(self):
        c = copy.deepcopy(self.cfg)
        config = c.multi_head_attention_model_config
        m = AttentionValidator(config)

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
        m = AttentionValidator(config)

        batch_size = config.batch_size
        source_sequence_length = config.source_sequence_length
        embedding_dim = config.embedding_dim

        changed_sequence_length = source_sequence_length + 1
        key = torch.randn(changed_sequence_length, batch_size, embedding_dim)
        value = torch.randn(source_sequence_length, batch_size, embedding_dim)

        with self.assertRaises(RuntimeError) as context:
            m.check_self_attention_projection_inputs(key, value)


class Test___resolve_static_projection_type(TestAttentionValidator):
    def test__value_tensor_flag__False(self):
        c = copy.deepcopy(self.cfg)
        config = c.multi_head_attention_model_config
        m = AttentionValidator(config)

        value_tensor_flag = False

        output = m._AttentionValidator__resolve_static_projection_type(
            value_tensor_flag
        )

        self.assertEqual(output, "static_keys")

    def test__value_tensor_flag__True(self):
        c = copy.deepcopy(self.cfg)
        config = c.multi_head_attention_model_config
        m = AttentionValidator(config)

        value_tensor_flag = True

        output = m._AttentionValidator__resolve_static_projection_type(
            value_tensor_flag
        )
        self.assertEqual(output, "static_values")


class Test___resolve_static_projection_shape(TestAttentionValidator):
    def test__static_tensor__None__value_tensor_flag__False(self):
        c = copy.deepcopy(self.cfg)
        config = c.multi_head_attention_model_config
        m = AttentionValidator(config)

        static_tensor = None
        value_tensor_flag = False
        output = m._AttentionValidator__resolve_static_projection_shape(
            static_tensor, value_tensor_flag
        )

        self.assertIsNone(output)

    def test__value_tensor_flag__False(self):
        c = copy.deepcopy(self.cfg)
        config = c.multi_head_attention_model_config
        m = AttentionValidator(config)

        batch_size = m.batch_size
        num_heads = m.num_heads
        embedding_dim = config.embedding_dim
        head_dim = embedding_dim // num_heads
        sequence_length = m.source_sequence_length

        static_tensor = torch.randn(batch_size * num_heads, sequence_length, head_dim)
        value_tensor_flag = False
        output = m._AttentionValidator__resolve_static_projection_shape(
            static_tensor, value_tensor_flag
        )

        self.assertIsNone(output)

    def test__is_assertion_raised(self):
        c = copy.deepcopy(self.cfg)
        config = c.multi_head_attention_model_config
        m = AttentionValidator(config)

        batch_size = m.batch_size
        num_heads = m.num_heads
        embedding_dim = config.embedding_dim
        head_dim = embedding_dim // num_heads
        sequence_length = m.source_sequence_length

        wrong_static_tensor = torch.randn(batch_size, sequence_length, head_dim)
        value_tensor_flag = False
        with self.assertRaises(AssertionError) as context:
            m._AttentionValidator__resolve_static_projection_shape(
                wrong_static_tensor, value_tensor_flag
            )


class Test_check_static_projection_shapes(TestAttentionValidator):
    def test__no_inputs(self):
        c = copy.deepcopy(self.cfg)
        config = c.multi_head_attention_model_config
        m = AttentionValidator(config)

        static_keys = None
        static_values = None

        output = m.check_static_projection_shapes(static_keys, static_values)

        self.assertIsNone(output)

    def test__check_static_projection_shapes(self):
        c = copy.deepcopy(self.cfg)
        config = c.multi_head_attention_model_config
        m = AttentionValidator(config)

        batch_size = m.batch_size
        num_heads = m.num_heads
        embedding_dim = config.embedding_dim
        head_dim = embedding_dim // num_heads
        sequence_length = m.source_sequence_length

        static_keys = torch.randn(
            batch_size * num_heads, sequence_length, head_dim, embedding_dim
        )
        static_values = torch.randn(
            batch_size * num_heads, sequence_length, head_dim, embedding_dim
        )

        output = m.check_static_projection_shapes(static_keys, static_values)
        self.assertIsNone(output)

    def test__is_assertion_raised(self):
        c = copy.deepcopy(self.cfg)
        config = c.multi_head_attention_model_config
        m = AttentionValidator(config)

        batch_size = m.batch_size
        num_heads = m.num_heads
        embedding_dim = config.embedding_dim
        head_dim = embedding_dim // num_heads
        sequence_length = m.source_sequence_length

        static_keys = torch.randn(batch_size, sequence_length, head_dim, embedding_dim)
        static_values = torch.randn(
            batch_size * num_heads, sequence_length, head_dim, embedding_dim
        )

        with self.assertRaises(AssertionError) as context:
            m.check_static_projection_shapes(static_keys, static_values)
