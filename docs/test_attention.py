import unittest
import itertools
from dataclasses import asdict
import torch
from Emperor.attention.utils.utils import (
    AttentionMask,
    AttentionProcessor,
    AttentionProjector,
    AttentionUtils,
    AttentionValidator,
)
from Emperor.layers.utils.base import LayerBlock
from Emperor.attention.attention import MultiHeadAttention, MultiHeadAttentionConfig
from Emperor.layers.utils.enums import LayerTypes
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
            self.model.use_separate_projection_weight_flag,
            self.config.use_separate_projection_weight_flag,
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
        self.assertEqual(self.model.query_key_projection_dim, self.config.embedding_dim)
        self.assertEqual(self.model.value_projection_dim, self.config.embedding_dim)


class TestMultIHeadAttention____resolve_kv_dimensions(TestAttention):
    def test__qkv_zero(self):
        config = MultiHeadAttentionConfig(
            query_key_projection_dim=0,
            value_projection_dim=0,
        )
        self.rebuild_presets(config)

        self.assertEqual(self.model.query_key_projection_dim, self.config.embedding_dim)
        self.assertEqual(self.model.value_projection_dim, self.config.embedding_dim)

    def test__kv_nonzero(self):
        config = MultiHeadAttentionConfig(
            query_key_projection_dim=128,
            value_projection_dim=256,
        )
        self.rebuild_presets(config)

        self.model._MultiHeadAttention__resolve_head_dim()
        self.assertEqual(
            self.model.query_key_projection_dim, self.config.query_key_projection_dim
        )
        self.assertEqual(
            self.model.value_projection_dim, self.config.value_projection_dim
        )


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
            query_key_projection_dim=32,
            value_projection_dim=32,
        )
        self.rebuild_presets(config)

        output = self.model._MultiHeadAttention__are_qkv_dimensions_equal()
        self.assertFalse(output)

    def test__different_query_key_projection_dim(self):
        config = MultiHeadAttentionConfig(
            embedding_dim=32,
            query_key_projection_dim=64,
            value_projection_dim=32,
        )
        self.rebuild_presets(config)

        output = self.model._MultiHeadAttention__are_qkv_dimensions_equal()
        self.assertFalse(output)

    def test__different_value_projection_dim(self):
        config = MultiHeadAttentionConfig(
            embedding_dim=32,
            query_key_projection_dim=32,
            value_projection_dim=64,
        )
        self.rebuild_presets(config)

        output = self.model._MultiHeadAttention__are_qkv_dimensions_equal()
        self.assertFalse(output)

    def test__embd_key_value_same_dim(self):
        config = MultiHeadAttentionConfig(
            embedding_dim=32,
            query_key_projection_dim=32,
            value_projection_dim=32,
        )
        self.rebuild_presets(config)

        output = self.model._MultiHeadAttention__are_qkv_dimensions_equal()
        self.assertTrue(output)


class TestMultIHeadAttention____build_shared_projection_models(TestAttention):
    def test__shared_model_inizialization(self):
        config = MultiHeadAttentionConfig(
            embedding_dim=32,
            query_key_projection_dim=32,
            value_projection_dim=32,
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
            query_key_projection_dim=64,
            value_projection_dim=32,
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
    def test__same_qkv_dim__use_separate_projection_weight_flag__False(self):
        config = MultiHeadAttentionConfig(
            embedding_dim=32,
            query_key_projection_dim=32,
            value_projection_dim=32,
            use_separate_projection_weight_flag=False,
        )
        self.rebuild_presets(config)

        self.assertIsInstance(self.model.qkv_model, LayerBlock)
        self.assertIsInstance(self.model.output_model, LayerBlock)
        self.assertIsNone(self.model.query_model)
        self.assertIsNone(self.model.key_model)
        self.assertIsNone(self.model.value_model)

    def test__same_qkv_dim__use_separate_projection_weight_flag__True(self):
        config = MultiHeadAttentionConfig(
            embedding_dim=32,
            query_key_projection_dim=32,
            value_projection_dim=32,
            use_separate_projection_weight_flag=True,
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
            query_key_projection_dim=64,
            value_projection_dim=32,
            use_separate_projection_weight_flag=True,
        )
        self.rebuild_presets(config)

        self.assertIsInstance(self.model.query_model, LayerBlock)
        self.assertIsInstance(self.model.key_model, LayerBlock)
        self.assertIsInstance(self.model.value_model, LayerBlock)
        self.assertIsInstance(self.model.output_model, LayerBlock)
        self.assertIsNone(self.model.qkv_model)


class TestMultIHeadAttention_forward(TestAttention):
    def test__use_separate_projection_weight_flag_with_same_qkv_tensors(self):
        tests = [
            {"use_separate_projection_weight_flag": False},
            {"use_separate_projection_weight_flag": True},
        ]
        for test in tests:
            config = MultiHeadAttentionConfig(
                target_sequence_length=32,
                source_sequence_length=32,
                **test,
            )
            self.rebuild_presets(config)

            query = key = value = torch.randn(
                self.target_sequence_length, self.batch_size, self.embedding_dim
            )

            key_padding_mask = None
            attention_mask = None
            static_key = None
            static_values = None

            attention_output, attention_weights = self.model.forward(
                query,
                key,
                value,
                key_padding_mask,
                attention_mask,
                static_key,
                static_values,
            )

            self.assertIsInstance(attention_output, torch.Tensor)
            self.assertIsNone(attention_weights)
            self.assertEqual(
                attention_output.shape,
                (self.target_sequence_length, self.batch_size, self.embedding_dim),
            )

    def test__use_separate_projection_weight_flag_with_different_qkv_tensors(self):
        config = MultiHeadAttentionConfig(
            target_sequence_length=32,
            source_sequence_length=32,
            use_separate_projection_weight_flag=True,
        )
        self.rebuild_presets(config)

        query = torch.randn(
            self.target_sequence_length, self.batch_size, self.embedding_dim
        )
        key = torch.randn(
            self.source_sequence_length, self.batch_size, self.embedding_dim
        )
        value = torch.randn(
            self.source_sequence_length, self.batch_size, self.embedding_dim
        )

        key_padding_mask = None
        attention_mask = None
        static_key = None
        static_values = None

        attention_output, attention_weights = self.model.forward(
            query,
            key,
            value,
            key_padding_mask,
            attention_mask,
            static_key,
            static_values,
        )

        self.assertIsInstance(attention_output, torch.Tensor)
        self.assertIsNone(attention_weights)
        self.assertEqual(
            attention_output.shape,
            (self.target_sequence_length, self.batch_size, self.embedding_dim),
        )

    def test__qkv_tensors_and_key_padding_mask(self):
        config = MultiHeadAttentionConfig(
            target_sequence_length=32,
            source_sequence_length=32,
            use_separate_projection_weight_flag=True,
        )
        self.rebuild_presets(config)

        query = torch.randn(
            self.target_sequence_length, self.batch_size, self.embedding_dim
        )
        key = torch.randn(
            self.target_sequence_length, self.batch_size, self.embedding_dim
        )
        value = torch.randn(
            self.target_sequence_length, self.batch_size, self.embedding_dim
        )

        key_padding_mask = torch.randn(self.batch_size, self.source_sequence_length)
        attention_mask = None
        static_key = None
        static_values = None

        attention_output, attention_weights = self.model.forward(
            query,
            key,
            value,
            key_padding_mask,
            attention_mask,
            static_key,
            static_values,
        )

        self.assertIsInstance(attention_output, torch.Tensor)
        self.assertIsNone(attention_weights)

    def test__qkv_tensors_and_attention_mask(self):
        config = MultiHeadAttentionConfig(
            target_sequence_length=32,
            source_sequence_length=32,
            use_separate_projection_weight_flag=True,
        )
        self.rebuild_presets(config)

        query = torch.randn(
            self.target_sequence_length, self.batch_size, self.embedding_dim
        )
        key = torch.randn(
            self.target_sequence_length, self.batch_size, self.embedding_dim
        )
        value = torch.randn(
            self.target_sequence_length, self.batch_size, self.embedding_dim
        )

        key_padding_mask = None
        attention_mask = torch.randn(
            1, self.target_sequence_length, self.source_sequence_length
        )
        attention_mask = torch.where(
            attention_mask > 0, torch.tensor(float("-inf")), torch.tensor(0.0)
        )
        attention_mask = attention_mask.repeat(self.batch_size * self.num_heads, 1, 1)
        static_key = None
        static_values = None

        attention_output, attention_weights = self.model.forward(
            query,
            key,
            value,
            key_padding_mask,
            attention_mask,
            static_key,
            static_values,
        )

        self.assertIsInstance(attention_output, torch.Tensor)
        self.assertIsNone(attention_weights)

    def test__qkv_tensors_and_key_padding_mask_and_attention_mask(self):
        config = MultiHeadAttentionConfig(
            target_sequence_length=32,
            source_sequence_length=32,
            use_separate_projection_weight_flag=True,
        )
        self.rebuild_presets(config)

        query = torch.randn(
            self.target_sequence_length, self.batch_size, self.embedding_dim
        )
        key = torch.randn(
            self.target_sequence_length, self.batch_size, self.embedding_dim
        )
        value = torch.randn(
            self.target_sequence_length, self.batch_size, self.embedding_dim
        )
        key_padding_mask = torch.randn(self.batch_size, self.source_sequence_length)
        attention_mask = torch.randn(
            1, self.target_sequence_length, self.source_sequence_length
        )
        attention_mask = torch.where(
            attention_mask > 0, torch.tensor(float("-inf")), torch.tensor(0.0)
        )
        attention_mask = attention_mask.repeat(self.batch_size * self.num_heads, 1, 1)
        static_key = None
        static_values = None

        attention_output, attention_weights = self.model.forward(
            query,
            key,
            value,
            key_padding_mask,
            attention_mask,
            static_key,
            static_values,
        )

        self.assertIsInstance(attention_output, torch.Tensor)
        self.assertIsNone(attention_weights)

    def test__return_attention_weights_flag(self):
        tests = [
            {"return_attention_weights_flag": False},
            {"return_attention_weights_flag": True},
        ]

        for test in tests:
            config = MultiHeadAttentionConfig(
                target_sequence_length=32,
                source_sequence_length=32,
                use_separate_projection_weight_flag=False,
                **test,
            )
            self.rebuild_presets(config)

            query = key = value = torch.randn(
                self.target_sequence_length, self.batch_size, self.embedding_dim
            )
            key_padding_mask = torch.randn(self.batch_size, self.source_sequence_length)
            attention_mask = torch.randn(
                1, self.target_sequence_length, self.source_sequence_length
            )
            attention_mask = torch.where(
                attention_mask > 0, torch.tensor(float("-inf")), torch.tensor(0.0)
            )
            attention_mask = attention_mask.repeat(
                self.batch_size * self.num_heads, 1, 1
            )
            static_key = None
            static_values = None

            attention_output, attention_weights = self.model.forward(
                query,
                key,
                value,
                key_padding_mask,
                attention_mask,
                static_key,
                static_values,
            )

            self.assertIsInstance(attention_output, torch.Tensor)
            if test["return_attention_weights_flag"]:
                self.assertIsInstance(attention_weights, torch.Tensor)
            else:
                self.assertIsNone(attention_weights)

    def test__use_separate_projection_weight_flag(self):
        tests = [
            {"use_separate_projection_weight_flag": False},
            {"use_separate_projection_weight_flag": True},
        ]

        for test in tests:
            config = MultiHeadAttentionConfig(
                target_sequence_length=32,
                source_sequence_length=32,
                **test,
            )
            self.rebuild_presets(config)

            query = key = value = torch.randn(
                self.target_sequence_length, self.batch_size, self.embedding_dim
            )
            key_padding_mask = torch.randn(self.batch_size, self.source_sequence_length)
            attention_mask = torch.randn(
                1, self.target_sequence_length, self.source_sequence_length
            )
            attention_mask = torch.where(
                attention_mask > 0, torch.tensor(float("-inf")), torch.tensor(0.0)
            )
            attention_mask = attention_mask.repeat(
                self.batch_size * self.num_heads, 1, 1
            )
            static_key = None
            static_value = None

            attention_output, attention_weights = self.model.forward(
                query,
                key,
                value,
                key_padding_mask,
                attention_mask,
                static_key,
                static_value,
            )

            self.assertIsInstance(attention_output, torch.Tensor)
            self.assertIsNone(attention_weights)

    def test__zero_attention_flag(self):
        tests = [
            {"zero_attention_flag": False},
            {"zero_attention_flag": True},
        ]

        for test in tests:
            config = MultiHeadAttentionConfig(
                target_sequence_length=32,
                source_sequence_length=32,
                use_separate_projection_weight_flag=True,
                **test,
            )
            self.rebuild_presets(config)

            query = torch.randn(
                self.target_sequence_length, self.batch_size, self.embedding_dim
            )
            key = torch.randn(
                self.target_sequence_length, self.batch_size, self.embedding_dim
            )
            value = torch.randn(
                self.target_sequence_length, self.batch_size, self.embedding_dim
            )

            key_padding_mask = torch.randn(self.batch_size, self.source_sequence_length)
            attention_mask = torch.randn(
                1, self.target_sequence_length, self.source_sequence_length
            )
            attention_mask = torch.where(
                attention_mask > 0, torch.tensor(float("-inf")), torch.tensor(0.0)
            )
            attention_mask = attention_mask.repeat(
                self.batch_size * self.num_heads, 1, 1
            )
            static_key = None
            static_value = None

            attention_output, attention_weights = self.model.forward(
                query,
                key,
                value,
                key_padding_mask,
                attention_mask,
                static_key,
                static_value,
            )

            self.assertIsInstance(attention_output, torch.Tensor)
            self.assertIsNone(attention_weights)

    def test__causal_attention_mask_flag(self):
        tests = [
            {"causal_attention_mask_flag": False},
            {"causal_attention_mask_flag": True},
        ]

        for test in tests:
            config = MultiHeadAttentionConfig(
                target_sequence_length=32,
                source_sequence_length=32,
                use_separate_projection_weight_flag=True,
                **test,
            )
            self.rebuild_presets(config)

            query = torch.randn(
                self.target_sequence_length, self.batch_size, self.embedding_dim
            )
            key = torch.randn(
                self.target_sequence_length, self.batch_size, self.embedding_dim
            )
            value = torch.randn(
                self.target_sequence_length, self.batch_size, self.embedding_dim
            )
            key_padding_mask = torch.randn(self.batch_size, self.source_sequence_length)
            attention_mask = torch.randn(
                1, self.target_sequence_length, self.source_sequence_length
            )
            attention_mask = torch.where(
                attention_mask > 0, torch.tensor(float("-inf")), torch.tensor(0.0)
            )
            attention_mask = attention_mask.repeat(
                self.batch_size * self.num_heads, 1, 1
            )
            static_key = None
            static_value = None

            attention_output, attention_weights = self.model.forward(
                query,
                key,
                value,
                key_padding_mask,
                attention_mask,
                static_key,
                static_value,
            )

            self.assertIsInstance(attention_output, torch.Tensor)
            self.assertIsNone(attention_weights)

    def test__add_key_value_bias_flag(self):
        tests = [
            {"add_key_value_bias_flag": False},
            {"add_key_value_bias_flag": True},
        ]

        for test in tests:
            config = MultiHeadAttentionConfig(
                target_sequence_length=32,
                source_sequence_length=32,
                use_separate_projection_weight_flag=True,
                **test,
            )
            self.rebuild_presets(config)

            query = torch.randn(
                self.target_sequence_length, self.batch_size, self.embedding_dim
            )
            key = torch.randn(
                self.target_sequence_length, self.batch_size, self.embedding_dim
            )
            value = torch.randn(
                self.target_sequence_length, self.batch_size, self.embedding_dim
            )
            key_padding_mask = torch.randn(self.batch_size, self.source_sequence_length)
            attention_mask = torch.randn(
                1, self.target_sequence_length, self.source_sequence_length
            )
            attention_mask = torch.where(
                attention_mask > 0, torch.tensor(float("-inf")), torch.tensor(0.0)
            )
            attention_mask = attention_mask.repeat(
                self.batch_size * self.num_heads, 1, 1
            )
            static_key = None
            static_value = None

            attention_output, attention_weights = self.model.forward(
                query,
                key,
                value,
                key_padding_mask,
                attention_mask,
                static_key,
                static_value,
            )
            self.assertIsInstance(attention_output, torch.Tensor)
            self.assertIsNone(attention_weights)

    def test__average_attention_weights_flag(self):
        tests = [
            {"average_attention_weights_flag": False},
            {"average_attention_weights_flag": True},
        ]

        for test in tests:
            config = MultiHeadAttentionConfig(
                target_sequence_length=32,
                source_sequence_length=32,
                return_attention_weights_flag=True,
                use_separate_projection_weight_flag=False,
                **test,
            )
            self.rebuild_presets(config)

            query = key = value = torch.randn(
                self.target_sequence_length, self.batch_size, self.embedding_dim
            )
            key_padding_mask = torch.randn(self.batch_size, self.source_sequence_length)
            attention_mask = torch.randn(
                1, self.target_sequence_length, self.source_sequence_length
            )
            attention_mask = torch.where(
                attention_mask > 0, torch.tensor(float("-inf")), torch.tensor(0.0)
            )
            attention_mask = attention_mask.repeat(
                self.batch_size * self.num_heads, 1, 1
            )
            static_key = None
            static_value = None

            attention_output, attention_weights = self.model.forward(
                query,
                key,
                value,
                key_padding_mask,
                attention_mask,
                static_key,
                static_value,
            )
            self.assertIsInstance(attention_output, torch.Tensor)
            self.assertIsInstance(attention_weights, torch.Tensor)
            if test["average_attention_weights_flag"]:
                self.assertEqual(attention_weights.dim(), 3)
            else:
                self.assertEqual(attention_weights.dim(), 4)

    def test__batch_first_flag(self):
        tests = [
            {"batch_first_flag": False},
            {"batch_first_flag": True},
        ]

        for test in tests:
            config = MultiHeadAttentionConfig(
                target_sequence_length=32,
                source_sequence_length=32,
                use_separate_projection_weight_flag=True,
                **test,
            )
            self.rebuild_presets(config)

            if test["batch_first_flag"]:
                q_shape = (
                    self.batch_size,
                    self.source_sequence_length,
                    self.embedding_dim,
                )
                kv_shape = (
                    self.batch_size,
                    self.target_sequence_length,
                    self.embedding_dim,
                )
            else:
                q_shape = (
                    self.source_sequence_length,
                    self.batch_size,
                    self.embedding_dim,
                )
                kv_shape = (
                    self.target_sequence_length,
                    self.batch_size,
                    self.embedding_dim,
                )

            query = torch.randn(q_shape)
            key = torch.randn(kv_shape)
            value = torch.randn(kv_shape)
            key_padding_mask = torch.randn(self.batch_size, self.source_sequence_length)
            attention_mask = torch.randn(
                1, self.target_sequence_length, self.source_sequence_length
            )
            attention_mask = torch.where(
                attention_mask > 0, torch.tensor(float("-inf")), torch.tensor(0.0)
            )
            attention_mask = attention_mask.repeat(
                self.batch_size * self.num_heads, 1, 1
            )
            static_key = None
            static_value = None

            attention_output, attention_weights = self.model.forward(
                query,
                key,
                value,
                key_padding_mask,
                attention_mask,
                static_key,
                static_value,
            )

            self.assertIsInstance(attention_output, torch.Tensor)
            self.assertEqual(attention_output.shape, kv_shape)
            self.assertIsNone(attention_weights)

    def test__add_key_value_bias_flag__and__zero_attention_flag(self):
        tests = [
            {"add_key_value_bias_flag": False, "zero_attention_flag": False},
            {"add_key_value_bias_flag": True, "zero_attention_flag": False},
            {"add_key_value_bias_flag": False, "zero_attention_flag": True},
            {"add_key_value_bias_flag": True, "zero_attention_flag": True},
        ]

        for test in tests:
            config = MultiHeadAttentionConfig(
                target_sequence_length=32,
                source_sequence_length=32,
                use_separate_projection_weight_flag=True,
                **test,
            )
            self.rebuild_presets(config)

            query = torch.randn(
                self.target_sequence_length, self.batch_size, self.embedding_dim
            )
            key = torch.randn(
                self.target_sequence_length, self.batch_size, self.embedding_dim
            )
            value = torch.randn(
                self.target_sequence_length, self.batch_size, self.embedding_dim
            )
            key_padding_mask = torch.randn(self.batch_size, self.source_sequence_length)
            attention_mask = torch.randn(
                1, self.target_sequence_length, self.source_sequence_length
            )
            attention_mask = torch.where(
                attention_mask > 0, torch.tensor(float("-inf")), torch.tensor(0.0)
            )
            attention_mask = attention_mask.repeat(
                self.batch_size * self.num_heads, 1, 1
            )
            static_key = None
            static_value = None

            attention_output, attention_weights = self.model.forward(
                query,
                key,
                value,
                key_padding_mask,
                attention_mask,
                static_key,
                static_value,
            )
            self.assertIsInstance(attention_output, torch.Tensor)
            self.assertIsNone(attention_weights)

    def test__self_attention__possible_flag_combinations(self):
        flags = [
            "add_key_value_bias_flag",
            "zero_attention_flag",
            # "return_attention_weights_flag",
            "average_attention_weights_flag",
            "causal_attention_mask_flag",
            "use_separate_projection_weight_flag",
        ]
        combinations = list(itertools.product([False, True], repeat=len(flags)))
        tests = [dict(zip(flags, combo)) for combo in combinations]

        for test in tests:
            config = MultiHeadAttentionConfig(
                target_sequence_length=32,
                source_sequence_length=32,
                **test,
            )
            self.rebuild_presets(config)

            query = key = value = torch.randn(
                self.target_sequence_length, self.batch_size, self.embedding_dim
            )
            key_padding_mask = torch.randn(self.batch_size, self.source_sequence_length)
            attention_mask = torch.randn(
                1, self.target_sequence_length, self.source_sequence_length
            )
            attention_mask = torch.where(
                attention_mask > 0, torch.tensor(float("-inf")), torch.tensor(0.0)
            )
            attention_mask = attention_mask.repeat(
                self.batch_size * self.num_heads, 1, 1
            )
            static_key = None
            static_value = None

            attention_output, attention_weights = self.model.forward(
                query,
                key,
                value,
                key_padding_mask,
                attention_mask,
                static_key,
                static_value,
            )
            self.assertIsInstance(attention_output, torch.Tensor)
            # self.assertIsNone(attention_weights)

    def test__differetn_input_tensors__possible_flag_combinations(self):
        flags = [
            "add_key_value_bias_flag",
            "zero_attention_flag",
            # "return_attention_weights_flag",
            "average_attention_weights_flag",
            "causal_attention_mask_flag",
            "use_separate_projection_weight_flag",
        ]
        combinations = list(itertools.product([False, True], repeat=len(flags)))
        tests = [dict(zip(flags, combo)) for combo in combinations]

        for test in tests:
            config = MultiHeadAttentionConfig(
                target_sequence_length=32,
                source_sequence_length=32,
                **test,
            )
            self.rebuild_presets(config)

            query = key = value = torch.randn(
                self.target_sequence_length, self.batch_size, self.embedding_dim
            )

            key_padding_mask = torch.randn(self.batch_size, self.source_sequence_length)
            attention_mask = torch.randn(
                1, self.target_sequence_length, self.source_sequence_length
            )
            attention_mask = torch.where(
                attention_mask > 0, torch.tensor(float("-inf")), torch.tensor(0.0)
            )
            attention_mask = attention_mask.repeat(
                self.batch_size * self.num_heads, 1, 1
            )
            static_key = None
            static_value = None

            attention_output, attention_weights = self.model.forward(
                query,
                key,
                value,
                key_padding_mask,
                attention_mask,
                static_key,
                static_value,
            )
            self.assertIsInstance(attention_output, torch.Tensor)

    def test__all_layer_types_and_flags_and_different_batch_sizes(self):
        flags = [
            "add_key_value_bias_flag",
            "zero_attention_flag",
            # "return_attention_weights_flag",
            "average_attention_weights_flag",
            "causal_attention_mask_flag",
            "use_separate_projection_weight_flag",
        ]
        combinations = list(itertools.product([False, True], repeat=len(flags)))
        tests = [dict(zip(flags, combo)) for combo in combinations]
        batch_sizes = [1, 2, 4, 8]  # Pass
        for test in tests:
            for batch_size in batch_sizes:
                for model_type in LayerTypes:
                    config = MultiHeadAttentionConfig(
                        model_type=model_type,
                        batch_size=batch_size,
                        target_sequence_length=32,
                        source_sequence_length=32,
                        **test,
                    )
                    self.rebuild_presets(config)
                    query = key = value = torch.randn(
                        self.target_sequence_length,
                        self.batch_size,
                        self.embedding_dim,
                    )
                    # query = torch.randn(
                    #     self.target_sequence_length, self.batch_size, self.embedding_dim
                    # )
                    # key = torch.randn(
                    #     self.target_sequence_length, self.batch_size, self.embedding_dim
                    # )
                    # value = torch.randn(
                    #     self.target_sequence_length, self.batch_size, self.embedding_dim
                    # )
                    key_padding_mask = torch.randn(
                        self.batch_size, self.source_sequence_length
                    )
                    attention_mask = torch.randn(
                        1, self.target_sequence_length, self.source_sequence_length
                    )
                    attention_mask = torch.where(
                        attention_mask > 0,
                        torch.tensor(float("-inf")),
                        torch.tensor(0.0),
                    )
                    attention_mask = attention_mask.repeat(
                        self.batch_size * self.num_heads, 1, 1
                    )
                    static_key = None
                    static_value = None

                    attention_output, attention_weights = self.model.forward(
                        query,
                        key,
                        value,
                        key_padding_mask,
                        attention_mask,
                        static_key,
                        static_value,
                    )
                    self.assertIsInstance(attention_output, torch.Tensor)

    def test__all_layer_types_and_flags_and_different_embedding_dims(self):
        flags = [
            "add_key_value_bias_flag",
            "zero_attention_flag",
            # "return_attention_weights_flag",
            "average_attention_weights_flag",
            "causal_attention_mask_flag",
            "use_separate_projection_weight_flag",
        ]
        combinations = list(itertools.product([False, True], repeat=len(flags)))
        tests = [dict(zip(flags, combo)) for combo in combinations]
        batch_sizes = [1, 2, 4, 8]  # Pass
        embedding_dims = [16, 32, 64]
        query_key_projection_dims = [16, 32, 64]
        value_projection_dims = [16, 32, 64]
        target_sequence_lengths = [16, 32, 64]
        for test in tests:
            for batch_size in batch_sizes:
                for model_type in LayerTypes:
                    config = MultiHeadAttentionConfig(
                        model_type=model_type,
                        batch_size=batch_size,
                        # embedding_dim=embedding_dim,
                        # query_key_projection_dim=query_key_projection_dims,
                        # value_projection_dim=value_projection_dim,
                        # target_sequence_length=target_sequence_length,
                        target_sequence_length=32,
                        source_sequence_length=32,
                        # use_separate_projection_weight_flag=True,
                        **test,
                    )
                    self.rebuild_presets(config)

                    query = key = value = torch.randn(
                        self.target_sequence_length,
                        self.batch_size,
                        self.embedding_dim,
                    )
                    # query = torch.randn(
                    #     self.target_sequence_length, self.batch_size, self.embedding_dim
                    # )
                    # key = torch.randn(
                    #     self.target_sequence_length, self.batch_size, self.embedding_dim
                    # )
                    # value = torch.randn(
                    #     self.target_sequence_length, self.batch_size, self.embedding_dim
                    # )
                    key_padding_mask = torch.randn(
                        self.batch_size, self.source_sequence_length
                    )
                    attention_mask = torch.randn(
                        1, self.target_sequence_length, self.source_sequence_length
                    )
                    attention_mask = torch.where(
                        attention_mask > 0,
                        torch.tensor(float("-inf")),
                        torch.tensor(0.0),
                    )
                    attention_mask = attention_mask.repeat(
                        self.batch_size * self.num_heads, 1, 1
                    )
                    static_key = None
                    static_value = None

                    attention_output, attention_weights = self.model.forward(
                        query,
                        key,
                        value,
                        key_padding_mask,
                        attention_mask,
                        static_key,
                        static_value,
                    )
                    self.assertIsInstance(attention_output, torch.Tensor)
                # self.assertIsNone(attention_weights)
