import torch
import unittest

from dataclasses import asdict
from Emperor.attention.utils.projection_handler import Projector
from Emperor.attention.attention import MultiHeadAttentionConfig
from Emperor.layers.utils.enums import LayerTypes
from Emperor.layers.utils.base import LayerBlock
from docs.utils import default_unittest_config


class TestProjector(unittest.TestCase):
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

        self.model = Projector(self.config, self.cfg)

        self.batch_size = self.config.batch_size
        self.embedding_dim = self.config.embedding_dim
        self.target_sequence_length = self.config.target_sequence_length
        self.source_sequence_length = self.config.source_sequence_length
        self.num_heads = self.config.num_heads
        self.head_dim = self.embedding_dim // self.num_heads
        self.query_model = self.model.query_model
        self.key_model = self.model.key_model
        self.value_model = self.model.value_model
        self.qkv_model = self.model.qkv_model


class Test___are_qkv_dimensions_equal(TestProjector):
    def test__different_embedding_dim(self):
        config = MultiHeadAttentionConfig(
            embedding_dim=64,
            query_key_projection_dim=32,
            value_projection_dim=32,
        )
        self.rebuild_presets(config)

        output = self.model._Projector__are_qkv_dimensions_equal()
        self.assertFalse(output)

    def test__different_query_key_projection_dim(self):
        config = MultiHeadAttentionConfig(
            embedding_dim=32,
            query_key_projection_dim=64,
            value_projection_dim=32,
        )
        self.rebuild_presets(config)

        output = self.model._Projector__are_qkv_dimensions_equal()
        self.assertFalse(output)

    def test__different_value_projection_dim(self):
        config = MultiHeadAttentionConfig(
            embedding_dim=32,
            query_key_projection_dim=32,
            value_projection_dim=64,
        )
        self.rebuild_presets(config)

        output = self.model._Projector__are_qkv_dimensions_equal()
        self.assertFalse(output)

    def test__embd_key_value_same_dim(self):
        config = MultiHeadAttentionConfig(
            embedding_dim=32,
            query_key_projection_dim=32,
            value_projection_dim=32,
        )
        self.rebuild_presets(config)

        output = self.model._Projector__are_qkv_dimensions_equal()
        self.assertTrue(output)


class Test___build_shared_projection_models(TestProjector):
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
        qkv_model = self.model._Projector__build_shared_projection_models()

        self.assertIsNone(self.model.query_model)
        self.assertIsNone(self.model.key_model)
        self.assertIsNone(self.model.value_model)
        self.assertIsInstance(qkv_model, LayerBlock)


class Test___build_separate_projection_models(TestProjector):
    def test__separate_models_initializations(self):
        config = MultiHeadAttentionConfig(
            embedding_dim=32,
            query_key_projection_dim=64,
            value_projection_dim=32,
        )
        self.rebuild_presets(config)
        del self.model.qkv_model  # Ensure model is not initialized
        query_model, key_model, value_model = (
            self.model._Projector__build_separate_projection_models()
        )

        self.assertIsInstance(query_model, LayerBlock)
        self.assertIsInstance(key_model, LayerBlock)
        self.assertIsInstance(value_model, LayerBlock)
        self.assertIsNone(self.model.qkv_model)


class Test___build_projection_models(TestProjector):
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


class Test___compute_projection(TestProjector):
    def test_method(self):
        config = MultiHeadAttentionConfig(
            embedding_dim=4,
            use_separate_projection_weight_flag=True,
        )
        self.rebuild_presets(config)

        tensor = torch.randn(
            self.target_sequence_length, self.batch_size, self.embedding_dim
        )

        projected_tensor = self.model._Projector__compute_projection(
            tensor, self.query_model
        )

        self.assertIsInstance(projected_tensor, torch.Tensor)

        self.assertEqual(
            projected_tensor.shape,
            (self.target_sequence_length, self.batch_size, self.embedding_dim),
        )


class Test___compute_indepentet_projections(TestProjector):
    def test_method(self):
        config = MultiHeadAttentionConfig(
            embedding_dim=12,
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

        query_projections, key_projections, value_projections = (
            self.model._Projector__compute_indepentet_projections(query, key, value)
        )

        self.assertIsInstance(query_projections, torch.Tensor)
        self.assertIsInstance(key_projections, torch.Tensor)
        self.assertIsInstance(value_projections, torch.Tensor)

        self.assertEqual(
            query_projections.shape,
            (self.target_sequence_length, self.batch_size, self.embedding_dim),
        )
        self.assertEqual(
            key_projections.shape,
            (self.source_sequence_length, self.batch_size, self.embedding_dim),
        )
        self.assertEqual(
            value_projections.shape,
            (self.source_sequence_length, self.batch_size, self.embedding_dim),
        )


class Test___split_self_attention_projection(TestProjector):
    def test_method(self):
        embeding_dim = 12
        shared_projection_embeding_dim = embeding_dim * 3
        shared_projections = torch.randn(
            self.target_sequence_length,
            self.batch_size,
            shared_projection_embeding_dim,
        )

        query_projections, key_projections, value_projections = (
            self.model._Projector__split_self_attention_projection(shared_projections)
        )
        self.assertEqual(
            query_projections.shape,
            (self.target_sequence_length, self.batch_size, embeding_dim),
        )
        self.assertEqual(
            key_projections.shape,
            (self.target_sequence_length, self.batch_size, embeding_dim),
        )
        self.assertEqual(
            value_projections.shape,
            (self.target_sequence_length, self.batch_size, embeding_dim),
        )


class Test___compute_self_attention_projections(TestProjector):
    def test_method(self):
        config = MultiHeadAttentionConfig(
            embedding_dim=12,
            use_separate_projection_weight_flag=False,
        )
        self.rebuild_presets(config)

        query = torch.randn(
            self.config.target_sequence_length,
            self.config.batch_size,
            self.embedding_dim,
        )

        query_projections, key_projections, value_projections = (
            self.model._Projector__compute_self_attention_projections(query)
        )
        self.assertEqual(
            query_projections.shape,
            (
                self.config.target_sequence_length,
                self.config.batch_size,
                self.embedding_dim,
            ),
        )
        self.assertEqual(
            key_projections.shape,
            (
                self.config.target_sequence_length,
                self.config.batch_size,
                self.embedding_dim,
            ),
        )
        self.assertEqual(
            value_projections.shape,
            (
                self.config.target_sequence_length,
                self.config.batch_size,
                self.embedding_dim,
            ),
        )


class Test_compute_qkv_projections(TestProjector):
    def test__indepented_projections__model_type__base(self):
        config = MultiHeadAttentionConfig(
            target_sequence_length=16,
            source_sequence_length=16,
            use_separate_projection_weight_flag=True,
            query_key_projection_dim=32,
            value_projection_dim=64,
            model_type=LayerTypes.BASE,
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

        query_projections, key_projections, value_projections = (
            self.model.compute_qkv_projections(query, key, value)
        )

        expected_qk_output_shape = (
            self.target_sequence_length,
            self.batch_size,
            config.query_key_projection_dim,
        )
        expected_v_output_shape = (
            self.source_sequence_length,
            self.batch_size,
            config.value_projection_dim,
        )
        self.assertEqual(query_projections.shape, expected_qk_output_shape)
        self.assertEqual(key_projections.shape, expected_qk_output_shape)
        self.assertEqual(value_projections.shape, expected_v_output_shape)

    def test__self_attention_projections__model_type__base(self):
        config = MultiHeadAttentionConfig(
            target_sequence_length=16,
            source_sequence_length=16,
            use_separate_projection_weight_flag=False,
            query_key_projection_dim=self.config.embedding_dim,
            value_projection_dim=self.config.embedding_dim,
            model_type=LayerTypes.BASE,
        )
        self.rebuild_presets(config)

        tensor = torch.randn(
            self.target_sequence_length, self.batch_size, self.embedding_dim
        )
        query = key = value = tensor

        query_projections, key_projections, value_projections = (
            self.model.compute_qkv_projections(query, key, value)
        )

        expected_qk_output_shape = (
            self.target_sequence_length,
            self.batch_size,
            config.query_key_projection_dim,
        )
        expected_v_output_shape = (
            self.source_sequence_length,
            self.batch_size,
            config.value_projection_dim,
        )
        self.assertEqual(query_projections.shape, expected_qk_output_shape)
        self.assertEqual(key_projections.shape, expected_qk_output_shape)
        self.assertEqual(value_projections.shape, expected_v_output_shape)

    def test__indepented_projections__model_type__dynamic_base(self):
        config = MultiHeadAttentionConfig(
            target_sequence_length=16,
            source_sequence_length=16,
            use_separate_projection_weight_flag=True,
            query_key_projection_dim=32,
            value_projection_dim=64,
            model_type=LayerTypes.DYNAMIC_BASE,
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

        query_projections, key_projections, value_projections = (
            self.model.compute_qkv_projections(query, key, value)
        )

        expected_qk_output_shape = (
            self.target_sequence_length,
            self.batch_size,
            config.query_key_projection_dim,
        )
        expected_v_output_shape = (
            self.source_sequence_length,
            self.batch_size,
            config.value_projection_dim,
        )
        self.assertEqual(query_projections.shape, expected_qk_output_shape)
        self.assertEqual(key_projections.shape, expected_qk_output_shape)
        self.assertEqual(value_projections.shape, expected_v_output_shape)

    def test__self_attention_projections__model_type__dynamic_base(self):
        config = MultiHeadAttentionConfig(
            target_sequence_length=16,
            source_sequence_length=16,
            use_separate_projection_weight_flag=False,
            query_key_projection_dim=self.config.embedding_dim,
            value_projection_dim=self.config.embedding_dim,
            model_type=LayerTypes.DYNAMIC_BASE,
        )
        self.rebuild_presets(config)
        tensor = torch.randn(
            self.target_sequence_length, self.batch_size, self.embedding_dim
        )
        query = key = value = tensor

        query_projections, key_projections, value_projections = (
            self.model.compute_qkv_projections(query, key, value)
        )

        expected_qk_output_shape = (
            self.target_sequence_length,
            self.batch_size,
            config.query_key_projection_dim,
        )
        expected_v_output_shape = (
            self.source_sequence_length,
            self.batch_size,
            config.value_projection_dim,
        )
        self.assertEqual(query_projections.shape, expected_qk_output_shape)
        self.assertEqual(key_projections.shape, expected_qk_output_shape)
        self.assertEqual(value_projections.shape, expected_v_output_shape)

    def test__indepented_projections__model_type__vector(self):
        config = MultiHeadAttentionConfig(
            target_sequence_length=16,
            source_sequence_length=16,
            use_separate_projection_weight_flag=False,
            query_key_projection_dim=32,
            value_projection_dim=64,
            model_type=LayerTypes.VECTOR,
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
        query_projections, key_projections, value_projections = (
            self.model.compute_qkv_projections(query, key, value)
        )

        expected_qk_output_shape = (
            self.target_sequence_length,
            self.batch_size,
            config.query_key_projection_dim,
        )
        expected_v_output_shape = (
            self.source_sequence_length,
            self.batch_size,
            config.value_projection_dim,
        )
        self.assertEqual(query_projections.shape, expected_qk_output_shape)
        self.assertEqual(key_projections.shape, expected_qk_output_shape)
        self.assertEqual(value_projections.shape, expected_v_output_shape)

    def test__self_attention_projections__model_type__vector(self):
        config = MultiHeadAttentionConfig(
            target_sequence_length=16,
            source_sequence_length=16,
            use_separate_projection_weight_flag=False,
            query_key_projection_dim=self.config.embedding_dim,
            value_projection_dim=self.config.embedding_dim,
            model_type=LayerTypes.VECTOR,
        )
        self.rebuild_presets(config)

        tensor = torch.randn(
            self.target_sequence_length, self.batch_size, self.embedding_dim
        )
        query = key = value = tensor

        query_projections, key_projections, value_projections = (
            self.model.compute_qkv_projections(query, key, value)
        )

        expected_qk_output_shape = (
            self.target_sequence_length,
            self.batch_size,
            config.query_key_projection_dim,
        )
        expected_v_output_shape = (
            self.source_sequence_length,
            self.batch_size,
            config.value_projection_dim,
        )
        self.assertEqual(query_projections.shape, expected_qk_output_shape)
        self.assertEqual(key_projections.shape, expected_qk_output_shape)
        self.assertEqual(value_projections.shape, expected_v_output_shape)

    def test__indepented_projections__model_type__matrix(self):
        config = MultiHeadAttentionConfig(
            target_sequence_length=16,
            source_sequence_length=16,
            use_separate_projection_weight_flag=True,
            query_key_projection_dim=32,
            value_projection_dim=64,
            model_type=LayerTypes.MATRIX,
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

        query_projections, key_projections, value_projections = (
            self.model.compute_qkv_projections(query, key, value)
        )

        expected_qk_output_shape = (
            self.target_sequence_length,
            self.batch_size,
            config.query_key_projection_dim,
        )
        expected_v_output_shape = (
            self.source_sequence_length,
            self.batch_size,
            config.value_projection_dim,
        )
        self.assertEqual(query_projections.shape, expected_qk_output_shape)
        self.assertEqual(key_projections.shape, expected_qk_output_shape)
        self.assertEqual(value_projections.shape, expected_v_output_shape)

    def test__self_attention_projections__model_type__matrix(self):
        config = MultiHeadAttentionConfig(
            target_sequence_length=16,
            source_sequence_length=16,
            use_separate_projection_weight_flag=False,
            query_key_projection_dim=self.config.embedding_dim,
            value_projection_dim=self.config.embedding_dim,
            model_type=LayerTypes.MATRIX,
        )
        self.rebuild_presets(config)
        tensor = torch.randn(
            self.target_sequence_length, self.batch_size, self.embedding_dim
        )
        query = key = value = tensor

        query_projections, key_projections, value_projections = (
            self.model.compute_qkv_projections(query, key, value)
        )
        expected_qk_output_shape = (
            self.target_sequence_length,
            self.batch_size,
            config.query_key_projection_dim,
        )
        expected_v_output_shape = (
            self.source_sequence_length,
            self.batch_size,
            config.value_projection_dim,
        )
        self.assertEqual(query_projections.shape, expected_qk_output_shape)
        self.assertEqual(key_projections.shape, expected_qk_output_shape)
        self.assertEqual(value_projections.shape, expected_v_output_shape)

    def test__indepented_projections__model_type__generator(self):
        config = MultiHeadAttentionConfig(
            target_sequence_length=16,
            source_sequence_length=16,
            use_separate_projection_weight_flag=True,
            query_key_projection_dim=32,
            value_projection_dim=64,
            model_type=LayerTypes.GENERATOR,
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

        query_projections, key_projections, value_projections = (
            self.model.compute_qkv_projections(query, key, value)
        )

        expected_qk_output_shape = (
            self.target_sequence_length,
            self.batch_size,
            config.query_key_projection_dim,
        )
        expected_v_output_shape = (
            self.source_sequence_length,
            self.batch_size,
            config.value_projection_dim,
        )
        self.assertEqual(query_projections.shape, expected_qk_output_shape)
        self.assertEqual(key_projections.shape, expected_qk_output_shape)
        self.assertEqual(value_projections.shape, expected_v_output_shape)

    def test__self_attention_projections__model_type__generator(self):
        config = MultiHeadAttentionConfig(
            target_sequence_length=16,
            source_sequence_length=16,
            use_separate_projection_weight_flag=False,
            query_key_projection_dim=self.config.embedding_dim,
            value_projection_dim=self.config.embedding_dim,
            model_type=LayerTypes.GENERATOR,
        )
        self.rebuild_presets(config)

        tensor = torch.randn(
            self.target_sequence_length, self.batch_size, self.embedding_dim
        )
        query = key = value = tensor

        query_projections, key_projections, value_projections = (
            self.model.compute_qkv_projections(query, key, value)
        )

        expected_qk_output_shape = (
            self.target_sequence_length,
            self.batch_size,
            config.query_key_projection_dim,
        )
        expected_v_output_shape = (
            self.source_sequence_length,
            self.batch_size,
            config.value_projection_dim,
        )
        self.assertEqual(query_projections.shape, expected_qk_output_shape)
        self.assertEqual(key_projections.shape, expected_qk_output_shape)
        self.assertEqual(value_projections.shape, expected_v_output_shape)
