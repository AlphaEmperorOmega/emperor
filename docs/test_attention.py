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
        self.assertIsInstance(self.model.validator, AttentionValidator)
        self.assertIsInstance(self.model.masks, AttentionMask)
        self.assertIsInstance(self.model.projector, AttentionProjector)
        self.assertIsInstance(self.model.processor, AttentionProcessor)
        self.assertIsInstance(self.model.utils, AttentionUtils)


class TestMultIHeadAttention____are_qkv_dimensions_equal(TestAttention):
    def test__different_embedding_dim(self):
        config = MultiHeadAttentionConfig(
            embedding_dim=64,
            key_dim=32,
            value_dim=32,
        )
        self.rebuild_presets(config)

        output = self.model._MultiHeadAttention__are_qkv_dimensions_equal()
        self.assertFalse(output)

    def test__different_key_dim(self):
        config = MultiHeadAttentionConfig(
            embedding_dim=32,
            key_dim=64,
            value_dim=32,
        )
        self.rebuild_presets(config)

        output = self.model._MultiHeadAttention__are_qkv_dimensions_equal()
        self.assertFalse(output)

    def test__different_value_dim(self):
        config = MultiHeadAttentionConfig(
            embedding_dim=32,
            key_dim=32,
            value_dim=64,
        )
        self.rebuild_presets(config)

        output = self.model._MultiHeadAttention__are_qkv_dimensions_equal()
        self.assertFalse(output)

    def test__embd_key_value_same_dim(self):
        config = MultiHeadAttentionConfig(
            embedding_dim=32,
            key_dim=32,
            value_dim=32,
        )
        self.rebuild_presets(config)

        output = self.model._MultiHeadAttention__are_qkv_dimensions_equal()
        self.assertTrue(output)


class TestMultIHeadAttention____build_shared_projection_models(TestAttention):
    def test__shared_model_inizialization(self):
        config = MultiHeadAttentionConfig(
            embedding_dim=32,
            key_dim=32,
            value_dim=32,
        )
        self.rebuild_presets(config)
        del self.model.query_model  # Ensure model is not initialized
        del self.model.key_model  # Ensure model is not initialized
        del self.model.value_model  # Ensure model is not initialized
        qkv_model, output_model = (
            self.model._MultiHeadAttention__build_shared_projection_models()
        )

        self.assertIsNone(self.model.query_model)
        self.assertIsNone(self.model.key_model)
        self.assertIsNone(self.model.value_model)
        self.assertIsInstance(qkv_model, LayerBlock)
        self.assertIsInstance(output_model, LayerBlock)


class TestMultIHeadAttention____build_separate_projection_models(TestAttention):
    def test__separate_models_initializations(self):
        config = MultiHeadAttentionConfig(
            embedding_dim=32,
            key_dim=64,
            value_dim=32,
        )
        self.rebuild_presets(config)
        del self.model.qkv_model  # Ensure model is not initialized
        query_model, key_model, value_model, output_model = (
            self.model._MultiHeadAttention__build_separate_projection_models()
        )

        self.assertIsInstance(query_model, LayerBlock)
        self.assertIsInstance(key_model, LayerBlock)
        self.assertIsInstance(value_model, LayerBlock)
        self.assertIsInstance(output_model, LayerBlock)
        self.assertIsNone(self.model.qkv_model)


class TestMultIHeadAttention____build_projection_models(TestAttention):
    def test__same_qkv_dim__use_separate_projection_weight__False(self):
        config = MultiHeadAttentionConfig(
            embedding_dim=32,
            key_dim=32,
            value_dim=32,
            use_separate_projection_weight=False,
        )
        self.rebuild_presets(config)

        self.assertIsInstance(self.model.qkv_model, LayerBlock)
        self.assertIsInstance(self.model.output_model, LayerBlock)
        self.assertIsNone(self.model.query_model)
        self.assertIsNone(self.model.key_model)
        self.assertIsNone(self.model.value_model)

    def test__same_qkv_dim__use_separate_projection_weight__True(self):
        config = MultiHeadAttentionConfig(
            embedding_dim=32,
            key_dim=32,
            value_dim=32,
            use_separate_projection_weight=True,
        )
        self.rebuild_presets(config)

        self.assertIsInstance(self.model.query_model, LayerBlock)
        self.assertIsInstance(self.model.key_model, LayerBlock)
        self.assertIsInstance(self.model.value_model, LayerBlock)
        self.assertIsInstance(self.model.output_model, LayerBlock)
        self.assertIsNone(self.model.qkv_model)

    def test__different_qkv_dim(self):
        config = MultiHeadAttentionConfig(
            embedding_dim=32,
            key_dim=64,
            value_dim=32,
            use_separate_projection_weight=True,
        )
        self.rebuild_presets(config)

        self.assertIsInstance(self.model.query_model, LayerBlock)
        self.assertIsInstance(self.model.key_model, LayerBlock)
        self.assertIsInstance(self.model.value_model, LayerBlock)
        self.assertIsInstance(self.model.output_model, LayerBlock)
        self.assertIsNone(self.model.qkv_model)
