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


class TestAttentionProjector(unittest.TestCase):
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

        validator = AttentionValidator(self.config)
        query_model = self.config.model_type.value(self.cfg)
        key_model = self.config.model_type.value(self.cfg)
        value_model = self.config.model_type.value(self.cfg)
        qkv_model = None

        self.model = AttentionProjector(
            self.config,
            validator,
            qkv_model,
            query_model,
            key_model,
            value_model,
        )

        self.batch_size = self.config.batch_size
        self.embedding_dim = self.config.embedding_dim
        self.target_sequence_length = self.config.target_sequence_length
        self.source_sequence_length = self.config.source_sequence_length
        self.num_heads = self.config.num_heads
        self.head_dim = self.embedding_dim // self.num_heads


class TestAttentionProjectorProjector____compute_projection(TestAttentionProjector):
    def test_method(self):
        c = copy.deepcopy(self.cfg)
        config = c.multi_head_attention_model_config
        config.embedding_dim = 4
        validator = AttentionValidator(config)
        query_model = config.model_type.value(c)
        key_model = config.model_type.value(c)
        value_model = config.model_type.value(c)
        qkv_model = None
        m = AttentionProjector(
            config, validator, qkv_model, query_model, key_model, value_model
        )

        batch_size = config.batch_size
        target_sequence_length = config.target_sequence_length
        embedding_dim = config.embedding_dim

        tensor = torch.randn(target_sequence_length, batch_size, embedding_dim)

        projected_tensor = m._AttentionProjector__compute_projection(
            tensor, query_model
        )

        self.assertIsInstance(projected_tensor, torch.Tensor)

        output_embedding = c.linear_layer_model_config.output_dim
        self.assertEqual(
            projected_tensor.shape,
            (target_sequence_length, batch_size, output_embedding),
        )


class Test____compute_indepentet_projections(TestAttentionProjector):
    def test_method(self):
        c = copy.deepcopy(self.cfg)
        config = c.multi_head_attention_model_config
        config.embedding_dim = 4
        validator = AttentionValidator(config)
        query_model = config.model_type.value(c)
        key_model = config.model_type.value(c)
        value_model = config.model_type.value(c)
        qkv_model = None
        m = AttentionProjector(
            config, validator, qkv_model, query_model, key_model, value_model
        )

        batch_size = config.batch_size
        target_sequence_length = config.target_sequence_length
        source_sequence_length = config.source_sequence_length
        embedding_dim = config.embedding_dim

        query = torch.randn(target_sequence_length, batch_size, embedding_dim)
        key = torch.randn(source_sequence_length, batch_size, embedding_dim)
        value = torch.randn(source_sequence_length, batch_size, embedding_dim)

        query_projections, key_projections, value_projections = (
            m._AttentionProjector__compute_indepentet_projections(query, key, value)
        )

        self.assertIsInstance(query_projections, torch.Tensor)
        self.assertIsInstance(key_projections, torch.Tensor)
        self.assertIsInstance(value_projections, torch.Tensor)

        output_embedding = c.linear_layer_model_config.output_dim
        self.assertEqual(
            query_projections.shape,
            (target_sequence_length, batch_size, output_embedding),
        )
        self.assertEqual(
            key_projections.shape,
            (source_sequence_length, batch_size, output_embedding),
        )
        self.assertEqual(
            value_projections.shape,
            (source_sequence_length, batch_size, output_embedding),
        )


class TestAttentionProjectorProjector____split_self_attention_projection(
    TestAttentionProjector
):
    def test_method(self):
        c = copy.deepcopy(self.cfg)
        config = c.multi_head_attention_model_config
        validator = AttentionValidator(config)
        query_model = None
        key_model = None
        value_model = None
        qkv_model = None

        m = AttentionProjector(
            config, validator, qkv_model, query_model, key_model, value_model
        )

        batch_size = config.batch_size
        target_and_source_sequence_length = config.target_sequence_length

        embeding_dim = 12
        shared_projection_embeding_dim = embeding_dim * 3
        shared_projections = torch.randn(
            target_and_source_sequence_length,
            batch_size,
            shared_projection_embeding_dim,
        )

        query_projections, key_projections, value_projections = (
            m._AttentionProjector__split_self_attention_projection(shared_projections)
        )
        self.assertEqual(
            query_projections.shape,
            (target_and_source_sequence_length, batch_size, embeding_dim),
        )
        self.assertEqual(
            key_projections.shape,
            (target_and_source_sequence_length, batch_size, embeding_dim),
        )
        self.assertEqual(
            value_projections.shape,
            (target_and_source_sequence_length, batch_size, embeding_dim),
        )


class TestAttentionProjectorProjector____compute_self_attention_projections(
    TestAttentionProjector
):
    def test_method(self):
        c = copy.deepcopy(self.cfg)
        old_output_dim = c.linear_layer_model_config.output_dim
        # When shared `__compute_self_projections` is used ensure that
        # ensure that `output_dim` is 3 * expected projection dimension
        c.linear_layer_model_config.output_dim = old_output_dim * 3
        config = c.multi_head_attention_model_config
        layer_config = c.linear_layer_model_config
        validator = AttentionValidator(config)
        query_model = None
        key_model = None
        value_model = None
        qkv_model = config.model_type.value(c)

        m = AttentionProjector(
            config, validator, qkv_model, query_model, key_model, value_model
        )

        batch_size = config.batch_size
        target_and_source_sequence_length = config.target_sequence_length

        embeding_dim = layer_config.input_dim
        query = torch.randn(
            target_and_source_sequence_length,
            batch_size,
            embeding_dim,
        )

        output_embedding_dim = layer_config.output_dim // 3
        query_projections, key_projections, value_projections = (
            m._AttentionProjector__compute_self_attention_projections(query)
        )
        self.assertEqual(
            query_projections.shape,
            (target_and_source_sequence_length, batch_size, output_embedding_dim),
        )
        self.assertEqual(
            key_projections.shape,
            (target_and_source_sequence_length, batch_size, output_embedding_dim),
        )
        self.assertEqual(
            value_projections.shape,
            (target_and_source_sequence_length, batch_size, output_embedding_dim),
        )


class TestAttentionProjectorProjector__compute_qkv_projections(TestAttentionProjector):
    def test__indepented_projections__model_type__base(self):
        c = copy.deepcopy(self.cfg)
        config = c.multi_head_attention_model_config
        config.model_type = LayerTypes.BASE
        layer_config = c.linear_layer_model_config
        validator = AttentionValidator(config)
        query_model = config.model_type.value(c)
        key_model = config.model_type.value(c)
        value_model = config.model_type.value(c)
        qkv_model = None

        m = AttentionProjector(
            config, validator, qkv_model, query_model, key_model, value_model
        )

        batch_size = config.batch_size
        target_sequence_length = source_sequence_length = config.target_sequence_length

        embeding_dim = layer_config.input_dim
        query = torch.randn(target_sequence_length, batch_size, embeding_dim)
        key = torch.randn(source_sequence_length, batch_size, embeding_dim)
        value = torch.randn(source_sequence_length, batch_size, embeding_dim)

        query_projections, key_projections, value_projections = (
            m.compute_qkv_projections(query, key, value)
        )
        expected_output_embedding_dim = c.linear_layer_model_config.output_dim
        self.assertEqual(
            query_projections.shape,
            (target_sequence_length, batch_size, expected_output_embedding_dim),
        )
        self.assertEqual(
            key_projections.shape,
            (source_sequence_length, batch_size, expected_output_embedding_dim),
        )
        self.assertEqual(
            value_projections.shape,
            (source_sequence_length, batch_size, expected_output_embedding_dim),
        )

    def test__self_attention_projections__model_type__base(self):
        c = copy.deepcopy(self.cfg)
        old_output_dim = c.linear_layer_model_config.output_dim
        # When shared `__compute_self_projections` is used ensure that
        # ensure that `output_dim` is 3 * expected projection dimension
        c.linear_layer_model_config.output_dim = old_output_dim * 3
        config = c.multi_head_attention_model_config
        config.model_type = LayerTypes.BASE
        layer_config = c.linear_layer_model_config
        validator = AttentionValidator(config)
        query_model = None
        key_model = None
        value_model = None
        qkv_model = config.model_type.value(c)

        m = AttentionProjector(
            config, validator, qkv_model, query_model, key_model, value_model
        )

        batch_size = config.batch_size
        target_sequence_length = source_sequence_length = config.target_sequence_length

        embeding_dim = layer_config.input_dim
        tensor = torch.randn(target_sequence_length, batch_size, embeding_dim)
        query = key = value = tensor

        query_projections, key_projections, value_projections = (
            m.compute_qkv_projections(query, key, value)
        )
        expected_output_embedding_dim = c.linear_layer_model_config.output_dim // 3
        self.assertEqual(
            query_projections.shape,
            (target_sequence_length, batch_size, expected_output_embedding_dim),
        )
        self.assertEqual(
            key_projections.shape,
            (source_sequence_length, batch_size, expected_output_embedding_dim),
        )
        self.assertEqual(
            value_projections.shape,
            (source_sequence_length, batch_size, expected_output_embedding_dim),
        )

    def test__indepented_projections__model_type__dynamic_base(self):
        c = copy.deepcopy(self.cfg)
        config = c.multi_head_attention_model_config
        config.model_type = LayerTypes.DYNAMIC_BASE
        layer_config = c.linear_layer_model_config
        validator = AttentionValidator(config)
        query_model = config.model_type.value(c)
        key_model = config.model_type.value(c)
        value_model = config.model_type.value(c)
        qkv_model = None

        m = AttentionProjector(
            config, validator, qkv_model, query_model, key_model, value_model
        )

        batch_size = config.batch_size
        target_sequence_length = source_sequence_length = config.target_sequence_length

        embeding_dim = layer_config.input_dim
        query = torch.randn(target_sequence_length, batch_size, embeding_dim)
        key = torch.randn(source_sequence_length, batch_size, embeding_dim)
        value = torch.randn(source_sequence_length, batch_size, embeding_dim)

        query_projections, key_projections, value_projections = (
            m.compute_qkv_projections(query, key, value)
        )
        expected_output_embedding_dim = c.linear_layer_model_config.output_dim
        self.assertEqual(
            query_projections.shape,
            (target_sequence_length, batch_size, expected_output_embedding_dim),
        )
        self.assertEqual(
            key_projections.shape,
            (source_sequence_length, batch_size, expected_output_embedding_dim),
        )
        self.assertEqual(
            value_projections.shape,
            (source_sequence_length, batch_size, expected_output_embedding_dim),
        )

    def test__self_attention_projections__model_type__dynamic_base(self):
        c = copy.deepcopy(self.cfg)
        old_output_dim = c.linear_layer_model_config.output_dim
        # When shared `__compute_self_projections` is used ensure that
        # ensure that `output_dim` is 3 * expected projection dimension
        c.linear_layer_model_config.output_dim = old_output_dim * 3
        config = c.multi_head_attention_model_config
        config.model_type = LayerTypes.DYNAMIC_BASE
        layer_config = c.linear_layer_model_config
        validator = AttentionValidator(config)
        query_model = None
        key_model = None
        value_model = None
        qkv_model = config.model_type.value(c)

        m = AttentionProjector(
            config, validator, qkv_model, query_model, key_model, value_model
        )

        batch_size = config.batch_size
        target_sequence_length = source_sequence_length = config.target_sequence_length

        embeding_dim = layer_config.input_dim
        tensor = torch.randn(target_sequence_length, batch_size, embeding_dim)
        query = key = value = tensor

        query_projections, key_projections, value_projections = (
            m.compute_qkv_projections(query, key, value)
        )
        expected_output_embedding_dim = c.linear_layer_model_config.output_dim // 3
        self.assertEqual(
            query_projections.shape,
            (target_sequence_length, batch_size, expected_output_embedding_dim),
        )
        self.assertEqual(
            key_projections.shape,
            (source_sequence_length, batch_size, expected_output_embedding_dim),
        )
        self.assertEqual(
            value_projections.shape,
            (source_sequence_length, batch_size, expected_output_embedding_dim),
        )

    def test__indepented_projections__model_type__vector(self):
        c = copy.deepcopy(self.cfg)
        config = c.multi_head_attention_model_config
        config.use_separate_projection_weight = True
        config.key_dim = 32
        config.value_dim = 64
        config.model_type = LayerTypes.VECTOR

        model = MultiHeadAttention(c)
        validator = AttentionValidator(config)
        qkv_model = None
        query_model = model.query_model
        key_model = model.key_model
        value_model = model.value_model

        m = AttentionProjector(
            config, validator, qkv_model, query_model, key_model, value_model
        )

        batch_size = config.batch_size
        target_sequence_length = source_sequence_length = config.target_sequence_length

        embeding_dim = config.embedding_dim
        query = torch.randn(target_sequence_length, batch_size, embeding_dim)
        key = torch.randn(source_sequence_length, batch_size, embeding_dim)
        value = torch.randn(source_sequence_length, batch_size, embeding_dim)

        query_projections, key_projections, value_projections = (
            m.compute_qkv_projections(query, key, value)
        )
        self.assertEqual(
            query_projections.shape,
            (target_sequence_length, batch_size, config.key_dim),
        )
        self.assertEqual(
            key_projections.shape,
            (source_sequence_length, batch_size, config.key_dim),
        )
        self.assertEqual(
            value_projections.shape,
            (source_sequence_length, batch_size, config.value_dim),
        )

    def test__self_attention_projections__model_type__vector(self):
        c = copy.deepcopy(self.cfg)
        config = c.multi_head_attention_model_config
        config.use_separate_projection_weight = False
        config.key_dim = config.embedding_dim
        config.value_dim = config.embedding_dim
        config.model_type = LayerTypes.VECTOR

        model = MultiHeadAttention(c)
        validator = AttentionValidator(config)
        query_model = None
        key_model = None
        value_model = None
        qkv_model = model.qkv_model

        m = AttentionProjector(
            config, validator, qkv_model, query_model, key_model, value_model
        )

        batch_size = config.batch_size
        target_sequence_length = source_sequence_length = config.target_sequence_length

        embedding_dim = config.embedding_dim
        tensor = torch.randn(target_sequence_length, batch_size, embedding_dim)
        query = key = value = tensor

        query_projections, key_projections, value_projections = (
            m.compute_qkv_projections(query, key, value)
        )
        self.assertEqual(
            query_projections.shape,
            (target_sequence_length, batch_size, config.embedding_dim),
        )
        self.assertEqual(
            key_projections.shape,
            (source_sequence_length, batch_size, config.embedding_dim),
        )
        self.assertEqual(
            value_projections.shape,
            (source_sequence_length, batch_size, config.embedding_dim),
        )

    def test__indepented_projections__model_type__matrix(self):
        c = copy.deepcopy(self.cfg)
        config = c.multi_head_attention_model_config
        config.use_separate_projection_weight = True
        config.key_dim = 32
        config.value_dim = 64
        config.model_type = LayerTypes.MATRIX

        model = MultiHeadAttention(c)
        validator = AttentionValidator(config)
        qkv_model = None
        query_model = model.query_model
        key_model = model.key_model
        value_model = model.value_model

        m = AttentionProjector(
            config, validator, qkv_model, query_model, key_model, value_model
        )

        batch_size = config.batch_size
        target_sequence_length = source_sequence_length = config.target_sequence_length

        embeding_dim = config.embedding_dim
        query = torch.randn(target_sequence_length, batch_size, embeding_dim)
        key = torch.randn(source_sequence_length, batch_size, embeding_dim)
        value = torch.randn(source_sequence_length, batch_size, embeding_dim)

        query_projections, key_projections, value_projections = (
            m.compute_qkv_projections(query, key, value)
        )
        self.assertEqual(
            query_projections.shape,
            (target_sequence_length, batch_size, config.key_dim),
        )
        self.assertEqual(
            key_projections.shape,
            (source_sequence_length, batch_size, config.key_dim),
        )
        self.assertEqual(
            value_projections.shape,
            (source_sequence_length, batch_size, config.value_dim),
        )

    def test__self_attention_projections__model_type__matrix(self):
        c = copy.deepcopy(self.cfg)
        config = c.multi_head_attention_model_config
        config.use_separate_projection_weight = False
        config.key_dim = config.embedding_dim
        config.value_dim = config.embedding_dim
        config.model_type = LayerTypes.MATRIX

        model = MultiHeadAttention(c)
        validator = AttentionValidator(config)
        query_model = None
        key_model = None
        value_model = None
        qkv_model = model.qkv_model

        m = AttentionProjector(
            config, validator, qkv_model, query_model, key_model, value_model
        )

        batch_size = config.batch_size
        target_sequence_length = source_sequence_length = config.target_sequence_length

        embedding_dim = config.embedding_dim
        tensor = torch.randn(target_sequence_length, batch_size, embedding_dim)
        query = key = value = tensor

        query_projections, key_projections, value_projections = (
            m.compute_qkv_projections(query, key, value)
        )
        self.assertEqual(
            query_projections.shape,
            (target_sequence_length, batch_size, config.embedding_dim),
        )
        self.assertEqual(
            key_projections.shape,
            (source_sequence_length, batch_size, config.embedding_dim),
        )
        self.assertEqual(
            value_projections.shape,
            (source_sequence_length, batch_size, config.embedding_dim),
        )

    def test__indepented_projections__model_type__generator(self):
        c = copy.deepcopy(self.cfg)
        config = c.multi_head_attention_model_config
        config.use_separate_projection_weight = True
        config.key_dim = 32
        config.value_dim = 64
        config.model_type = LayerTypes.GENERATOR

        model = MultiHeadAttention(c)
        validator = AttentionValidator(config)
        qkv_model = None
        query_model = model.query_model
        key_model = model.key_model
        value_model = model.value_model

        m = AttentionProjector(
            config, validator, qkv_model, query_model, key_model, value_model
        )

        batch_size = config.batch_size
        target_sequence_length = source_sequence_length = config.target_sequence_length

        embeding_dim = config.embedding_dim
        query = torch.randn(target_sequence_length, batch_size, embeding_dim)
        key = torch.randn(source_sequence_length, batch_size, embeding_dim)
        value = torch.randn(source_sequence_length, batch_size, embeding_dim)

        query_projections, key_projections, value_projections = (
            m.compute_qkv_projections(query, key, value)
        )
        self.assertEqual(
            query_projections.shape,
            (target_sequence_length, batch_size, config.key_dim),
        )
        self.assertEqual(
            key_projections.shape,
            (source_sequence_length, batch_size, config.key_dim),
        )
        self.assertEqual(
            value_projections.shape,
            (source_sequence_length, batch_size, config.value_dim),
        )

    def test__self_attention_projections__model_type__generator(self):
        c = copy.deepcopy(self.cfg)
        config = c.multi_head_attention_model_config
        config.use_separate_projection_weight = False
        config.key_dim = config.embedding_dim
        config.value_dim = config.embedding_dim
        config.model_type = LayerTypes.GENERATOR

        model = MultiHeadAttention(c)
        validator = AttentionValidator(config)
        query_model = None
        key_model = None
        value_model = None
        qkv_model = model.qkv_model

        m = AttentionProjector(
            config, validator, qkv_model, query_model, key_model, value_model
        )

        batch_size = config.batch_size
        target_sequence_length = source_sequence_length = config.target_sequence_length

        embedding_dim = config.embedding_dim
        tensor = torch.randn(target_sequence_length, batch_size, embedding_dim)
        query = key = value = tensor

        query_projections, key_projections, value_projections = (
            m.compute_qkv_projections(query, key, value)
        )
        self.assertEqual(
            query_projections.shape,
            (target_sequence_length, batch_size, config.embedding_dim),
        )
        self.assertEqual(
            key_projections.shape,
            (source_sequence_length, batch_size, config.embedding_dim),
        )
        self.assertEqual(
            value_projections.shape,
            (source_sequence_length, batch_size, config.embedding_dim),
        )
