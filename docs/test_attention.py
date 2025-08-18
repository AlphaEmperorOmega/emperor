import copy
import unittest
from dataclasses import asdict
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
from Emperor.layers.utils.enums import LayerTypes
from Emperor.layers.utils.base import LayerBlock
from Emperor.attention.attention import MultiHeadAttention, MultiHeadAttentionConfig
from docs.utils import default_unittest_config


class TestAttention(unittest.TestCase):
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

        self.model = MultiHeadAttention(self.cfg)

        self.batch_size = self.config.batch_size
        self.embedding_dim = self.config.embedding_dim
        self.target_sequence_length = self.config.target_sequence_length
        self.source_sequence_length = self.config.source_sequence_length
        self.num_heads = self.config.num_heads
        self.head_dim = self.embedding_dim // self.num_heads


class TestMultiHeadAttention__init(TestAttention):
    def test__init_input_layer_with_default_config(self):
        self.assertIsInstance(self.model, MultiHeadAttention)
        self.assertEqual(self.model.batch_size, self.config.batch_size)
        self.assertEqual(self.model.model_type, self.config.model_type)
        self.assertEqual(self.model.num_heads, self.config.num_heads)
        self.assertEqual(self.model.embedding_dim, self.config.embedding_dim)
        self.assertEqual(self.model.target_dtype, self.config.target_dtype)
        self.assertEqual(
            self.model.target_sequence_length, self.config.target_sequence_length
        )
        self.assertEqual(
            self.model.source_sequence_length, self.config.source_sequence_length
        )
        self.assertEqual(
            self.model.use_separate_projection_weight,
            self.config.use_separate_projection_weight,
        )
        self.assertEqual(
            self.model.dropout_probability, self.config.dropout_probability
        )
        self.assertEqual(
            self.model.key_value_bias_flag, self.config.key_value_bias_flag
        )
        self.assertEqual(
            self.model.zero_attention_flag, self.config.zero_attention_flag
        )
        self.assertEqual(self.model.batch_first_flag, self.config.batch_first_flag)
        self.assertEqual(self.model.key_dim, self.config.embedding_dim)
        self.assertEqual(self.model.value_dim, self.config.embedding_dim)


class TestMultIHeadAttention____resolve_kv_dimensions(TestAttention):
    def test__qkv_zero(self):
        config = MultiHeadAttentionConfig(
            key_dim=0,
            value_dim=0,
        )
        self.rebuild_presets(config)

        self.assertEqual(self.model.key_dim, self.config.embedding_dim)
        self.assertEqual(self.model.value_dim, self.config.embedding_dim)

    def test__kv_nonzero(self):
        config = MultiHeadAttentionConfig(
            key_dim=128,
            value_dim=256,
        )
        self.rebuild_presets(config)

        self.model._MultiHeadAttention__resolve_head_dim()
        self.assertEqual(self.model.key_dim, self.config.key_dim)
        self.assertEqual(self.model.value_dim, self.config.value_dim)


class TestMultIHeadAttention____resolve_head_dim(TestAttention):
    def test__computed_head_dim(self):
        head_dim = self.model._MultiHeadAttention__resolve_head_dim()
        expected_head_dim = self.config.embedding_dim // self.config.num_heads
        self.assertEqual(head_dim, expected_head_dim)

    def test__if_assertion_is_raised(self):
        self.model.num_heads = 3

        with self.assertRaises(AssertionError) as context:
            _ = self.model._MultiHeadAttention__resolve_head_dim()


class TestMultIHeadAttention____initialize_attention_components(TestAttention):
    def test__ensure_componets_are_initialzied(self):
        c = copy.deepcopy(self.cfg)
        m = MultiHeadAttention(c)

        self.assertIsInstance(m.validator, AttentionValidator)
        self.assertIsInstance(m.masks, AttentionMask)
        self.assertIsInstance(m.projector, AttentionProjector)
        self.assertIsInstance(m.processor, AttentionProcessor)
        self.assertIsInstance(m.utils, AttentionUtils)


class TestMultIHeadAttention____are_qkv_dimensions_equal(TestAttention):
    def test__different_embedding_dim(self):
        c = copy.deepcopy(self.cfg)
        c.multi_head_attention_model_config.embedding_dim = 64
        c.multi_head_attention_model_config.key_dim = 32
        c.multi_head_attention_model_config.value_dim = 32
        m = MultiHeadAttention(c)

        output = m._MultiHeadAttention__are_qkv_dimensions_equal()
        self.assertFalse(output)

    def test__different_key_dim(self):
        c = copy.deepcopy(self.cfg)
        c.multi_head_attention_model_config.embedding_dim = 32
        c.multi_head_attention_model_config.key_dim = 64
        c.multi_head_attention_model_config.value_dim = 32
        m = MultiHeadAttention(c)

        output = m._MultiHeadAttention__are_qkv_dimensions_equal()
        self.assertFalse(output)

    def test__different_value_dim(self):
        c = copy.deepcopy(self.cfg)
        # This exists here to ensure because of `register_parameter` in the
        # this method is allready called when `MultiHeadAttention` is initialized
        c.multi_head_attention_model_config.embedding_dim = 32
        c.multi_head_attention_model_config.key_dim = 32
        c.multi_head_attention_model_config.value_dim = 64
        m = MultiHeadAttention(c)

        output = m._MultiHeadAttention__are_qkv_dimensions_equal()
        self.assertFalse(output)

    def test__embd_key_value_same_dim(self):
        c = copy.deepcopy(self.cfg)
        # This exists here to ensure because of `register_parameter` in the
        # this method is allready called when `MultiHeadAttention` is initialized
        c.multi_head_attention_model_config.embedding_dim = 32
        c.multi_head_attention_model_config.key_dim = 32
        c.multi_head_attention_model_config.value_dim = 32
        m = MultiHeadAttention(c)

        output = m._MultiHeadAttention__are_qkv_dimensions_equal()
        self.assertTrue(output)


class TestMultIHeadAttention____build_shared_projection_models(TestAttention):
    def test__shared_model_inizialization(self):
        c = copy.deepcopy(self.cfg)
        # This exists here to ensure because of `register_parameter` in the
        # this method is allready called when `MultiHeadAttention` is initialized
        c.multi_head_attention_model_config.embedding_dim = 32
        c.multi_head_attention_model_config.key_dim = 32
        c.multi_head_attention_model_config.value_dim = 32
        m = MultiHeadAttention(c)
        qkv_model, output_model = (
            m._MultiHeadAttention__build_shared_projection_models()
        )

        self.assertIsNone(m.query_model)
        self.assertIsNone(m.key_model)
        self.assertIsNone(m.value_model)
        self.assertIsInstance(qkv_model, LayerBlock)
        self.assertIsInstance(output_model, LayerBlock)


class TestMultIHeadAttention____build_separate_projection_models(TestAttention):
    def test__separate_models_initializations(self):
        c = copy.deepcopy(self.cfg)
        c.multi_head_attention_model_config.embedding_dim = 32
        c.multi_head_attention_model_config.key_dim = 64
        c.multi_head_attention_model_config.value_dim = 32
        m = MultiHeadAttention(c)
        query_model, key_model, value_model, output_model = (
            m._MultiHeadAttention__build_separate_projection_models()
        )

        self.assertIsInstance(query_model, LayerBlock)
        self.assertIsInstance(key_model, LayerBlock)
        self.assertIsInstance(value_model, LayerBlock)
        self.assertIsInstance(output_model, LayerBlock)
        self.assertIsNone(m.qkv_model)


class TestMultIHeadAttention____build_projection_models(TestAttention):
    def test__same_qkv_dim__use_separate_projection_weight__False(self):
        c = copy.deepcopy(self.cfg)
        c.multi_head_attention_model_config.embedding_dim = 32
        c.multi_head_attention_model_config.key_dim = 32
        c.multi_head_attention_model_config.value_dim = 32
        c.multi_head_attention_model_config.use_separate_projection_weight = False
        m = MultiHeadAttention(c)

        self.assertIsInstance(m.qkv_model, LayerBlock)
        self.assertIsInstance(m.output_model, LayerBlock)
        self.assertIsNone(m.query_model)
        self.assertIsNone(m.key_model)
        self.assertIsNone(m.value_model)

    def test__same_qkv_dim__use_separate_projection_weight__True(self):
        c = copy.deepcopy(self.cfg)
        c.multi_head_attention_model_config.embedding_dim = 32
        c.multi_head_attention_model_config.key_dim = 32
        c.multi_head_attention_model_config.value_dim = 32
        c.multi_head_attention_model_config.use_separate_projection_weight = True
        m = MultiHeadAttention(c)

        self.assertIsInstance(m.query_model, LayerBlock)
        self.assertIsInstance(m.key_model, LayerBlock)
        self.assertIsInstance(m.value_model, LayerBlock)
        self.assertIsInstance(m.output_model, LayerBlock)
        self.assertIsNone(m.qkv_model)

    def test__different_qkv_dim(self):
        c = copy.deepcopy(self.cfg)
        c.multi_head_attention_model_config.embedding_dim = 32
        c.multi_head_attention_model_config.key_dim = 64
        c.multi_head_attention_model_config.value_dim = 32
        m = MultiHeadAttention(c)

        self.assertIsInstance(m.query_model, LayerBlock)
        self.assertIsInstance(m.key_model, LayerBlock)
        self.assertIsInstance(m.value_model, LayerBlock)
        self.assertIsInstance(m.output_model, LayerBlock)
        self.assertIsNone(m.qkv_model)


class TestAttentionValidator____check_query_dims(TestAttention):
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


class TestAttentionValidator____check_query_key_value_dimensions(TestAttention):
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


class TestAttentionValidator____check_key_padding_mask_dimensions(TestAttention):
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


class TestAttentionValidator____check_attention_mask(TestAttention):
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

        attention_mask = torch.randn(source_sequence_length, target_sequence_length)

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
            batch_size * num_heads, source_sequence_length, target_sequence_length
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


class TestAttentionValidator____resolve_attention_mask_shape(TestAttention):
    def test__ensure_correct_shape_for_2D_mask(self):
        c = copy.deepcopy(self.cfg)
        config = c.multi_head_attention_model_config
        m = AttentionValidator(config)
        m.batched_input_flag = True

        source_sequence_length = m.source_sequence_length
        target_sequence_length = m.target_sequence_length

        expected_attention_mask_shape = (
            source_sequence_length,
            target_sequence_length,
        )

        attention_mask = torch.randn(
            source_sequence_length,
            target_sequence_length,
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
            source_sequence_length,
            target_sequence_length,
        )

        attention_mask = torch.randn(
            batch_size * num_heads,
            source_sequence_length,
            target_sequence_length,
        )

        output = m._AttentionValidator__resolve_attention_mask_shape(attention_mask)
        self.assertEqual(output, expected_attention_mask_shape)


class TestAttentionValidator____ensure_attention_mask_if_causal(TestAttention):
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


class TestAttentionValidator__multi_head_attention_input_shapes(TestAttention):
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
            batch_size * num_heads, source_sequence_length, target_sequence_length
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
        attention_mask = torch.randn(source_sequence_length, target_sequence_length)

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


class TestAttentionValidator__is_mask_float_or_bool(TestAttention):
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


class TestAttentionValidator__is_mask_correct_dtype(TestAttention):
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


class TestAttentionValidator____canonical_mask(TestAttention):
    def test__input_as_None(self):
        c = copy.deepcopy(self.cfg)
        config = c.multi_head_attention_model_config
        validator = AttentionValidator(config)
        m = AttentionMask(config, validator)

        mask = None
        mask_name = ""
        other_type = None
        other_name = ""
        target_type = torch.float32
        check_other = False

        output = m._AttentionMask__canonical_mask(
            mask, mask_name, other_type, other_name, target_type, check_other
        )
        self.assertIsNone(output)

    def test__boolean_mask_input(self):
        c = copy.deepcopy(self.cfg)
        config = c.multi_head_attention_model_config
        validator = AttentionValidator(config)
        m = AttentionMask(config, validator)

        mask = torch.randn(10, 10, dtype=torch.float32) > 0
        mask_name = "maks_to_test"
        other_type = torch.bool
        other_name = "required_mask_dtype"
        target_type = torch.float32
        check_other = True

        output = m._AttentionMask__canonical_mask(
            mask, mask_name, other_type, other_name, target_type, check_other
        )

        self.assertTrue(output.dtype == torch.float32)
        self.assertFalse(output.dtype == mask.dtype)

    def test__float_mask_input(self):
        c = copy.deepcopy(self.cfg)
        config = c.multi_head_attention_model_config
        validator = AttentionValidator(config)
        m = AttentionMask(config, validator)

        mask = torch.randn(10, 10, dtype=torch.float32)
        mask_name = "maks_to_test"
        other_type = torch.float32
        other_name = "required_mask_dtype"
        target_type = torch.float32
        check_other = True

        output = m._AttentionMask__canonical_mask(
            mask, mask_name, other_type, other_name, target_type, check_other
        )

        self.assertTrue(torch.equal(output, mask))


class TestAttentionValidator__check_self_attention_projection_inputs(TestAttention):
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


class TestAttentionValidator__check_indepentent_projections_inputs(TestAttention):
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


class TestAttentionValidator____resolve_static_projection_type(TestAttention):
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


class TestAttentionValidator____resolve_static_projection_shape(TestAttention):
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


class TestAttentionValidator__check_static_projection_shapes(TestAttention):
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


class TestAttentionMask__validate_attention_mask(TestAttention):
    def test__key_padding_mask__None(self):
        c = copy.deepcopy(self.cfg)
        config = c.multi_head_attention_model_config
        validator = AttentionValidator(config)
        m = AttentionMask(config, validator)
        m.causal_attention_mask_flag = True

        batch_size = config.batch_size
        num_heads = config.num_heads
        source_sequence_length = config.source_sequence_length
        target_sequence_length = config.target_sequence_length

        key_padding_mask = None
        attention_mask = torch.randn(
            batch_size * num_heads, source_sequence_length, target_sequence_length
        )
        attention_mask = attention_mask > 0
        need_weights = False

        output = m._AttentionMask__validate_attention_mask(
            key_padding_mask,
            attention_mask,
            need_weights,
        )

        self.assertIsNone(output)

    def test__boolean_attention_mask(self):
        c = copy.deepcopy(self.cfg)
        config = c.multi_head_attention_model_config
        validator = AttentionValidator(config)
        m = AttentionMask(config, validator)
        m.causal_attention_mask_flag = True

        batch_size = config.batch_size
        num_heads = config.num_heads
        source_sequence_length = config.source_sequence_length
        target_sequence_length = config.target_sequence_length

        key_padding_mask = torch.randint(0, 2, (batch_size, source_sequence_length))
        attention_mask = torch.randn(
            batch_size * num_heads, source_sequence_length, target_sequence_length
        )
        attention_mask = attention_mask > 0
        need_weights = True

        output = m._AttentionMask__validate_attention_mask(
            key_padding_mask,
            attention_mask,
            need_weights,
        )

        self.assertIsInstance(output, torch.Tensor)
        self.assertNotEqual(output.dtype, torch.bool)
        self.assertEqual(output.dtype, config.target_dtype)
        self.assertFalse(m.causal_attention_mask_flag)

    def test__float_attention_mask(self):
        c = copy.deepcopy(self.cfg)
        config = c.multi_head_attention_model_config
        validator = AttentionValidator(config)
        m = AttentionMask(config, validator)
        m.causal_attention_mask_flag = True

        batch_size = config.batch_size
        num_heads = config.num_heads
        source_sequence_length = config.source_sequence_length
        target_sequence_length = config.target_sequence_length

        key_padding_mask = torch.randint(0, 2, (batch_size, source_sequence_length))
        attention_mask = torch.randn(
            batch_size * num_heads, source_sequence_length, target_sequence_length
        )
        need_weights = True

        output = m._AttentionMask__validate_attention_mask(
            key_padding_mask,
            attention_mask,
            need_weights,
        )

        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(output.dtype, attention_mask.dtype)
        self.assertTrue(torch.equal(output, attention_mask))
        self.assertFalse(m.causal_attention_mask_flag)


class TestAttentionMask__validate_padding_and_attention_masks(TestAttention):
    def test__inputs_as_None(self):
        c = copy.deepcopy(self.cfg)
        config = c.multi_head_attention_model_config
        validator = AttentionValidator(config)
        m = AttentionMask(config, validator)
        m.causal_attention_mask_flag = True

        key_padding_mask = None
        attention_mask = None
        need_weights = True

        key_padding_mask, attention_mask = m.validate_padding_and_attention_masks(
            key_padding_mask,
            attention_mask,
            need_weights,
        )

        self.assertIsNone(key_padding_mask)
        self.assertIsNone(attention_mask)

    def test__only_key_padding_mask_input(self):
        c = copy.deepcopy(self.cfg)
        config = c.multi_head_attention_model_config
        validator = AttentionValidator(config)
        m = AttentionMask(config, validator)
        m.causal_attention_mask_flag = True

        batch_size = config.batch_size
        source_sequence_length = config.source_sequence_length

        key_padding_mask = torch.randint(0, 2, (batch_size, source_sequence_length)) > 0
        attention_mask = None
        need_weights = True

        output_key_padding_mask, output_attention_mask = (
            m.validate_padding_and_attention_masks(
                key_padding_mask,
                attention_mask,
                need_weights,
            )
        )

        self.assertIsInstance(output_key_padding_mask, torch.Tensor)
        self.assertEqual(output_key_padding_mask.dtype, torch.float32)
        self.assertIsNone(output_attention_mask)

    def test__only_attention_mask_input(self):
        c = copy.deepcopy(self.cfg)
        config = c.multi_head_attention_model_config
        validator = AttentionValidator(config)
        m = AttentionMask(config, validator)
        m.causal_attention_mask_flag = True

        batch_size = config.batch_size
        num_heads = config.num_heads
        source_sequence_length = config.source_sequence_length
        target_sequence_length = config.target_sequence_length

        key_padding_mask = None
        attention_mask = (
            torch.randn(
                batch_size * num_heads, source_sequence_length, target_sequence_length
            )
            > 0
        )
        need_weights = True

        key_padding_mask, attention_mask = m.validate_padding_and_attention_masks(
            key_padding_mask,
            attention_mask,
            need_weights,
        )

        self.assertIsNone(key_padding_mask)
        self.assertIsInstance(attention_mask, torch.Tensor)
        self.assertEqual(attention_mask.dtype, config.target_dtype)

    def test__key_padding_mask__and__attention_mask_input(self):
        c = copy.deepcopy(self.cfg)
        config = c.multi_head_attention_model_config
        validator = AttentionValidator(config)
        m = AttentionMask(config, validator)
        m.causal_attention_mask_flag = True

        batch_size = config.batch_size
        num_heads = config.num_heads
        source_sequence_length = config.source_sequence_length
        target_sequence_length = config.target_sequence_length

        key_padding_mask = torch.randint(0, 2, (batch_size, source_sequence_length)) > 0
        attention_mask = (
            torch.randn(
                batch_size * num_heads, source_sequence_length, target_sequence_length
            )
            > 0
        )
        need_weights = True

        key_padding_mask, attention_mask = m.validate_padding_and_attention_masks(
            key_padding_mask,
            attention_mask,
            need_weights,
        )

        self.assertIsInstance(key_padding_mask, torch.Tensor)
        self.assertEqual(key_padding_mask.dtype, torch.float32)
        self.assertIsInstance(attention_mask, torch.Tensor)
        self.assertEqual(attention_mask.dtype, torch.float32)

    def test__key_padding_mask__and__attention_mask__as__float_inputs(self):
        c = copy.deepcopy(self.cfg)
        config = c.multi_head_attention_model_config
        validator = AttentionValidator(config)
        m = AttentionMask(config, validator)
        m.causal_attention_mask_flag = True

        batch_size = config.batch_size
        num_heads = config.num_heads
        source_sequence_length = config.source_sequence_length
        target_sequence_length = config.target_sequence_length

        key_padding_mask = torch.randn(batch_size, source_sequence_length)
        attention_mask = torch.randn(
            batch_size * num_heads, source_sequence_length, target_sequence_length
        )
        need_weights = True

        output_key_padding_mask, output_attention_mask = (
            m.validate_padding_and_attention_masks(
                key_padding_mask,
                attention_mask,
                need_weights,
            )
        )

        self.assertTrue(torch.equal(output_key_padding_mask, key_padding_mask))
        self.assertTrue(torch.equal(output_attention_mask, attention_mask))
