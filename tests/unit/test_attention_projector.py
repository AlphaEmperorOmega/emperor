import unittest

import torch
from emperor.attention import (
    IndependentAttentionConfig,
    MixtureOfAttentionHeadsConfig,
    SelfAttentionConfig,
    SelfAttentionProjectionStrategy,
)
from emperor.attention.core.variants.independent_attention.projector import (
    IndependentProjector,
)
from emperor.attention.core.variants.mixture_of_attention_heads.projector import (
    MixtureOfAttentionHeadsProjector,
)
from emperor.attention.core.variants.self_attention.projector import (
    SelfAttentionProjector,
)
from emperor.base.layer import Layer, LayerStack, RecurrentLayer
from emperor.experts.core.layers import MixtureOfExperts

from support.attention import build_attention_config

PROJECTION_KINDS = ["base", "adaptive"]


class TestSelfAttentionProjector(unittest.TestCase):
    def test_init(self):
        for projection_kind in PROJECTION_KINDS:
            with self.subTest(projection_kind=projection_kind):
                embedding_dim = 12
                c = build_attention_config(
                    config_class=SelfAttentionConfig,
                    projection_kind=projection_kind,
                    embedding_dim=embedding_dim,
                    query_key_projection_dim=embedding_dim,
                    value_projection_dim=embedding_dim,
                )
                m = SelfAttentionProjector(c)
                self.assertEqual(m.embedding_dim, embedding_dim)
                self.assertIsInstance(m.qkv_model, (Layer, LayerStack))
                self.assertIsNone(m.query_model)
                self.assertIsNone(m.key_model)
                self.assertIsNone(m.value_model)

    def test_separate_strategy_builds_named_projection_models(self):
        cfg = build_attention_config(
            config_class=SelfAttentionConfig,
            embedding_dim=12,
            query_key_projection_dim=12,
            value_projection_dim=12,
            self_attention_projection_strategy=(
                SelfAttentionProjectionStrategy.SEPARATE
            ),
        )

        model = SelfAttentionProjector(cfg)

        self.assertIsNone(model.qkv_model)
        self.assertIsInstance(model.query_model, (Layer, LayerStack))
        self.assertIsInstance(model.key_model, (Layer, LayerStack))
        self.assertIsInstance(model.value_model, (Layer, LayerStack))

    def test_separate_recurrent_strategy_forwards_and_backpropagates(self):
        torch.manual_seed(0)
        cfg = build_attention_config(
            config_class=SelfAttentionConfig,
            batch_size=2,
            num_heads=2,
            embedding_dim=8,
            query_key_projection_dim=8,
            value_projection_dim=8,
            target_sequence_length=4,
            source_sequence_length=4,
            projection_kind="recurrent",
            self_attention_projection_strategy=(
                SelfAttentionProjectionStrategy.SEPARATE
            ),
        )
        attention = cfg.build()
        inputs = torch.randn(4, 2, 8, requires_grad=True)

        output, attention_weights, auxiliary_loss = attention(
            inputs,
            inputs,
            inputs,
        )

        self.assertEqual(output.shape, inputs.shape)
        self.assertIsNone(attention_weights)
        self.assertIsNone(auxiliary_loss)
        self.assertIsInstance(attention.projector.output_model, RecurrentLayer)
        self.assertIsInstance(attention.projector.query_model, RecurrentLayer)
        self.assertIsInstance(attention.projector.key_model, RecurrentLayer)
        self.assertIsInstance(attention.projector.value_model, RecurrentLayer)

        output.square().mean().backward()
        self.assertIsNotNone(inputs.grad)
        self.assertTrue(torch.any(inputs.grad.abs() > 0))
        projection_models = {
            "query": attention.projector.query_model,
            "key": attention.projector.key_model,
            "value": attention.projector.value_model,
            "output": attention.projector.output_model,
        }
        for label, projection_model in projection_models.items():
            with self.subTest(projection_model=label):
                gradients = [
                    parameter.grad
                    for parameter in projection_model.parameters()
                    if parameter.requires_grad
                ]
                self.assertTrue(
                    any(
                        gradient is not None and torch.any(gradient.abs() > 0)
                        for gradient in gradients
                    ),
                    f"Expected non-zero gradients through the {label} projection.",
                )

    def test__split_self_attention_projection(self):
        embedding_dim = 12
        c = build_attention_config(
            config_class=SelfAttentionConfig,
            embedding_dim=embedding_dim,
            query_key_projection_dim=embedding_dim,
            value_projection_dim=embedding_dim,
        )
        m = SelfAttentionProjector(c)
        shared_projection_embeding_dim = embedding_dim * 3
        shared_projections = torch.randn(
            c.target_sequence_length,
            c.batch_size,
            shared_projection_embeding_dim,
        )

        query_projections, key_projections, value_projections = (
            m._SelfAttentionProjector__split_self_attention_projection(
                shared_projections
            )
        )
        expected_shape = (c.target_sequence_length, c.batch_size, embedding_dim)
        self.assertEqual(query_projections.shape, expected_shape)
        self.assertEqual(key_projections.shape, expected_shape)
        self.assertEqual(value_projections.shape, expected_shape)

    def test__compute_projection(self):
        for projection_kind in PROJECTION_KINDS:
            with self.subTest(projection_kind=projection_kind):
                embedding_dim = 18
                c = build_attention_config(
                    config_class=SelfAttentionConfig,
                    projection_kind=projection_kind,
                    embedding_dim=embedding_dim,
                    query_key_projection_dim=embedding_dim,
                    value_projection_dim=embedding_dim,
                )
                m = SelfAttentionProjector(c)

                tensor = torch.randn(
                    c.target_sequence_length, c.batch_size, c.embedding_dim
                )

                projected_tensor = m._compute_projection(tensor, m.qkv_model)

                expected_shape = (
                    c.target_sequence_length,
                    c.batch_size,
                    c.embedding_dim * 3,
                )
                self.assertIsInstance(projected_tensor, torch.Tensor)
                self.assertEqual(projected_tensor.shape, expected_shape)

    def test_compute_qkv_projections(self):
        for projection_kind in PROJECTION_KINDS:
            with self.subTest(projection_kind=projection_kind):
                embedding_dim = 12
                c = build_attention_config(
                    config_class=SelfAttentionConfig,
                    projection_kind=projection_kind,
                    embedding_dim=embedding_dim,
                    query_key_projection_dim=embedding_dim,
                    value_projection_dim=embedding_dim,
                )
                m = SelfAttentionProjector(c)

                tensor = torch.randn(
                    c.target_sequence_length, c.batch_size, c.embedding_dim
                )

                q_projections, k_projections, v_projections = m.compute_qkv_projections(
                    tensor, tensor, tensor
                )

                expected_shape = (
                    c.target_sequence_length,
                    c.batch_size,
                    c.embedding_dim,
                )
                self.assertEqual(q_projections.shape, expected_shape)
                self.assertEqual(k_projections.shape, expected_shape)
                self.assertEqual(v_projections.shape, expected_shape)

    def test_compute_output_projection(self):
        for projection_kind in PROJECTION_KINDS:
            with self.subTest(projection_kind=projection_kind):
                embedding_dim = 18
                c = build_attention_config(
                    config_class=SelfAttentionConfig,
                    projection_kind=projection_kind,
                    embedding_dim=embedding_dim,
                    query_key_projection_dim=embedding_dim,
                    value_projection_dim=embedding_dim,
                )
                m = SelfAttentionProjector(c)

                tensor = torch.randn(
                    c.target_sequence_length, c.batch_size, c.embedding_dim
                )

                attention_output = m.compute_output_projection(tensor)

                expected_shape = (
                    c.target_sequence_length,
                    c.batch_size,
                    c.embedding_dim,
                )
                self.assertIsInstance(attention_output, torch.Tensor)
                self.assertEqual(attention_output.shape, expected_shape)


class TestIndependentProjector(unittest.TestCase):
    def test_init(self):
        for projection_kind in PROJECTION_KINDS:
            with self.subTest(projection_kind=projection_kind):
                c = build_attention_config(
                    config_class=IndependentAttentionConfig,
                    projection_kind=projection_kind,
                    query_key_projection_dim=16,
                    value_projection_dim=20,
                )
                m = IndependentProjector(c)
                self.assertIsInstance(m.query_model, (Layer, LayerStack))
                self.assertIsInstance(m.key_model, (Layer, LayerStack))
                self.assertIsInstance(m.value_model, (Layer, LayerStack))

    def test__compute_projection(self):
        for projection_kind in PROJECTION_KINDS:
            with self.subTest(projection_kind=projection_kind):
                c = build_attention_config(
                    config_class=IndependentAttentionConfig,
                    projection_kind=projection_kind,
                    query_key_projection_dim=16,
                    value_projection_dim=20,
                )
                m = IndependentProjector(c)

                tensor = torch.randn(
                    c.target_sequence_length, c.batch_size, c.embedding_dim
                )

                projected_tensor = m._compute_projection(tensor, m.query_model)

                expected_shape = (
                    c.target_sequence_length,
                    c.batch_size,
                    m.query_key_projection_dim,
                )
                self.assertIsInstance(projected_tensor, torch.Tensor)
                self.assertEqual(projected_tensor.shape, expected_shape)

    def test_compute_qkv_projections(self):
        for projection_kind in PROJECTION_KINDS:
            with self.subTest(projection_kind=projection_kind):
                c = build_attention_config(
                    config_class=IndependentAttentionConfig,
                    projection_kind=projection_kind,
                    query_key_projection_dim=16,
                    value_projection_dim=20,
                )
                m = IndependentProjector(c)

                tensor = torch.randn(
                    c.target_sequence_length, c.batch_size, c.embedding_dim
                )

                q_projections, k_projections, v_projections = m.compute_qkv_projections(
                    tensor, tensor, tensor
                )

                expected_shape = (
                    c.target_sequence_length,
                    c.batch_size,
                    m.query_key_projection_dim,
                )
                expected_value_shape = (
                    c.target_sequence_length,
                    c.batch_size,
                    m.value_projection_dim,
                )
                self.assertEqual(q_projections.shape, expected_shape)
                self.assertEqual(k_projections.shape, expected_shape)
                self.assertEqual(v_projections.shape, expected_value_shape)


class TestMixtureOfAttentionHeadsProjector(unittest.TestCase):
    def test_init(self):
        for use_kv_expert_models_flag in (True, False):
            with self.subTest(use_kv_expert_models_flag=use_kv_expert_models_flag):
                c = build_attention_config(
                    config_class=MixtureOfAttentionHeadsConfig,
                    query_key_projection_dim=12,
                    value_projection_dim=12,
                    embedding_dim=12,
                    use_kv_expert_models_flag=use_kv_expert_models_flag,
                )
                m = MixtureOfAttentionHeadsProjector(c)

                self.assertIsInstance(m.query_model, MixtureOfExperts)
                self.assertIsInstance(m.output_model, MixtureOfExperts)
                if use_kv_expert_models_flag:
                    self.assertIsInstance(m.key_model, MixtureOfExperts)
                    self.assertIsInstance(m.value_model, MixtureOfExperts)
                else:
                    self.assertIsInstance(m.key_model, (Layer, LayerStack))
                    self.assertIsInstance(m.value_model, (Layer, LayerStack))

    def test__compute_kv_projection(self):
        c = build_attention_config(
            config_class=MixtureOfAttentionHeadsConfig,
            query_key_projection_dim=12,
            value_projection_dim=12,
            embedding_dim=12,
            experts_compute_expert_mixture_flag=False,
            experts_stack_num_layers=1,
        )
        m = MixtureOfAttentionHeadsProjector(c)

        tensor = torch.randn(c.target_sequence_length, c.batch_size, c.embedding_dim)

        projected_tensor = m._compute_kv_projection(tensor, m.key_model)

        expected_shape = (
            c.target_sequence_length,
            c.batch_size,
            m.query_key_projection_dim,
        )
        self.assertIsInstance(projected_tensor, torch.Tensor)
        self.assertEqual(projected_tensor.shape, expected_shape)

    def test_compute_qkv_projections(self):
        for use_kv_expert_models_flag in (True, False):
            with self.subTest(use_kv_expert_models_flag=use_kv_expert_models_flag):
                c = build_attention_config(
                    config_class=MixtureOfAttentionHeadsConfig,
                    query_key_projection_dim=14,
                    value_projection_dim=14,
                    embedding_dim=14,
                    target_sequence_length=7,
                    source_sequence_length=7,
                    use_kv_expert_models_flag=use_kv_expert_models_flag,
                    experts_compute_expert_mixture_flag=False,
                    experts_stack_num_layers=1,
                )
                m = MixtureOfAttentionHeadsProjector(c)

                tensor = torch.randn(
                    c.target_sequence_length, c.batch_size, c.embedding_dim
                )

                q_projections, k_projections, v_projections = m.compute_qkv_projections(
                    tensor, tensor, tensor
                )

                expected_top_k_shape = (
                    c.target_sequence_length,
                    c.batch_size,
                    m.top_k,
                    m.query_key_projection_dim,
                )
                expected_shape = (
                    c.target_sequence_length,
                    c.batch_size,
                    m.query_key_projection_dim,
                )

                self.assertEqual(q_projections.shape, expected_top_k_shape)
                if use_kv_expert_models_flag:
                    self.assertEqual(k_projections.shape, expected_top_k_shape)
                    self.assertEqual(v_projections.shape, expected_top_k_shape)
                else:
                    self.assertEqual(k_projections.shape, expected_shape)
                    self.assertEqual(v_projections.shape, expected_shape)

    def test_compute_output_projection(self):
        c = build_attention_config(
            config_class=MixtureOfAttentionHeadsConfig,
            embedding_dim=12,
            query_key_projection_dim=12,
            value_projection_dim=12,
        )
        m = MixtureOfAttentionHeadsProjector(c)

        tensor = torch.randn(c.target_sequence_length, c.batch_size, c.embedding_dim)
        m.compute_qkv_projections(tensor, tensor, tensor)

        weighted_values = torch.randn(
            c.target_sequence_length,
            c.batch_size,
            m.top_k,
            c.embedding_dim,
        )

        attention_output = m.compute_output_projection(weighted_values)

        expected_shape = (
            c.target_sequence_length * c.batch_size,
            c.embedding_dim,
        )
        self.assertIsInstance(attention_output, torch.Tensor)
        self.assertEqual(attention_output.shape, expected_shape)


class TestProjectorDispatch(unittest.TestCase):
    def test_leaf_runtime_builds_expected_projector(self):
        expected_map = {
            SelfAttentionConfig: SelfAttentionProjector,
            IndependentAttentionConfig: IndependentProjector,
            MixtureOfAttentionHeadsConfig: (MixtureOfAttentionHeadsProjector),
        }
        for config_class, expected_cls in expected_map.items():
            for projection_kind in PROJECTION_KINDS:
                with self.subTest(
                    config_class=config_class,
                    projection_kind=projection_kind,
                ):
                    qkv_dim = 12 if config_class == SelfAttentionConfig else 12
                    c = build_attention_config(
                        config_class=config_class,
                        projection_kind=projection_kind,
                        embedding_dim=12,
                        query_key_projection_dim=qkv_dim,
                        value_projection_dim=qkv_dim,
                    )
                    m = c.build()
                    self.assertIsInstance(m.projector, expected_cls)
