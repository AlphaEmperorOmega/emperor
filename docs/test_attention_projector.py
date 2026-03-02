import torch
import unittest

from torch.nn import Sequential
from emperor.experts.utils.layers import MixtureOfExperts
from emperor.attention.utils.enums import AttentionOptions
from emperor.experts.utils.enums import InitSamplerOptions
from emperor.linears.options import LinearLayerStackOptions
from emperor.parametric.options import AdaptiveLayerStackOptions
from emperor.attention.utils.presets import MultiHeadAttentionPresets
from emperor.parametric.utils.mixtures.options import AdaptiveWeightOptions
from emperor.attention.utils.handlers.projector import (
    IndependentProjector,
    MixtureOfAttentionHeadsProjector,
    ProjectorBuilder,
    SelfAttentionProjector,
)


class TestSelfAttentionProjector(unittest.TestCase):
    def test_init(self):
        attention_options = [LinearLayerStackOptions, AdaptiveLayerStackOptions]

        for attention_option in attention_options:
            for model_type in attention_option:
                message = f"Testing configuration: model_type: {model_type}"
                with self.subTest(i=message):
                    embedding_dim = 12
                    c = MultiHeadAttentionPresets.multi_head_attention_preset(
                        attention_option=AttentionOptions.SELF_ATTENTION,
                        model_type=model_type,
                        embedding_dim=embedding_dim,
                        query_key_projection_dim=embedding_dim,
                        value_projection_dim=embedding_dim,
                    )
                    m = SelfAttentionProjector(c)
                    self.assertEqual(model_type.value, m.model_type)

    def test__split_self_attention_projection(self):
        embedding_dim = 12
        c = MultiHeadAttentionPresets.multi_head_attention_preset(
            embedding_dim=embedding_dim,
            query_key_projection_dim=embedding_dim,
            value_projection_dim=embedding_dim,
        )
        m = SelfAttentionProjector(c)
        embeding_dim = 12
        shared_projection_embeding_dim = embeding_dim * 3
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
        expected_shape = (c.target_sequence_length, c.batch_size, embeding_dim)
        self.assertEqual(query_projections.shape, expected_shape)
        self.assertEqual(key_projections.shape, expected_shape)
        self.assertEqual(value_projections.shape, expected_shape)

    def test__compute_projection(self):
        attention_options = [LinearLayerStackOptions, AdaptiveLayerStackOptions]

        for attention_option in attention_options:
            for model_type in attention_option:
                message = f"Testing configuration: model_type: {model_type}"
                with self.subTest(i=message):
                    embedding_dim = 18
                    c = MultiHeadAttentionPresets.multi_head_attention_preset(
                        attention_option=AttentionOptions.SELF_ATTENTION,
                        model_type=model_type,
                        embedding_dim=embedding_dim,
                        query_key_projection_dim=embedding_dim,
                        value_projection_dim=embedding_dim,
                    )
                    m = SelfAttentionProjector(c)

                    tensor = torch.randn(
                        c.target_sequence_length,
                        c.batch_size,
                        c.embedding_dim,
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
        attention_options = [LinearLayerStackOptions, AdaptiveLayerStackOptions]

        for attention_option in attention_options:
            for model_type in attention_option:
                for adaptive_type in AdaptiveWeightOptions:
                    message = f"Testing configuration: model_type: {model_type}, adaptive_type: {adaptive_type}"
                    with self.subTest(i=message):
                        if adaptive_type == AdaptiveWeightOptions.VECTOR:
                            continue
                        embedding_dim = 12
                        c = MultiHeadAttentionPresets.multi_head_attention_preset(
                            attention_option=AttentionOptions.SELF_ATTENTION,
                            model_type=model_type,
                            embedding_dim=embedding_dim,
                            query_key_projection_dim=embedding_dim,
                            value_projection_dim=embedding_dim,
                            projector_adaptive_weight_option=adaptive_type,
                        )
                        m = SelfAttentionProjector(c)

                        tensor = torch.randn(
                            c.target_sequence_length,
                            c.batch_size,
                            c.embedding_dim,
                        )

                        q_projections, k_projections, v_projections = (
                            m.compute_qkv_projections(tensor, tensor, tensor)
                        )

                        expected_shape = (
                            c.target_sequence_length,
                            c.batch_size,
                            c.embedding_dim,
                        )
                        self.assertIsInstance(q_projections, torch.Tensor)
                        self.assertIsInstance(k_projections, torch.Tensor)
                        self.assertIsInstance(v_projections, torch.Tensor)
                        self.assertEqual(q_projections.shape, expected_shape)
                        self.assertEqual(k_projections.shape, expected_shape)
                        self.assertEqual(v_projections.shape, expected_shape)

    def test_compute_output_projection(self):
        attention_options = [LinearLayerStackOptions, AdaptiveLayerStackOptions]

        for attention_option in attention_options:
            for model_type in attention_option:
                for adaptive_type in AdaptiveWeightOptions:
                    message = f"Testing configuration: model_type: {model_type}, adaptive_type: {adaptive_type}"
                    with self.subTest(i=message):
                        if adaptive_type == AdaptiveWeightOptions.VECTOR:
                            continue
                        embedding_dim = 18
                        c = MultiHeadAttentionPresets.multi_head_attention_preset(
                            model_type=model_type,
                            embedding_dim=embedding_dim,
                            query_key_projection_dim=embedding_dim,
                            value_projection_dim=embedding_dim,
                            projector_adaptive_weight_option=adaptive_type,
                        )
                        m = SelfAttentionProjector(c)

                        tensor = torch.randn(
                            c.target_sequence_length,
                            c.batch_size,
                            c.embedding_dim,
                        )

                        attentiion_output = m.compute_output_projection(tensor)

                        expected_shape = (
                            c.target_sequence_length,
                            c.batch_size,
                            c.embedding_dim,
                        )
                        self.assertIsInstance(attentiion_output, torch.Tensor)
                        self.assertEqual(attentiion_output.shape, expected_shape)


class TestIndependentProjector(unittest.TestCase):
    def test_init(self):
        attention_options = [LinearLayerStackOptions, AdaptiveLayerStackOptions]

        for attention_option in attention_options:
            for model_type in attention_option:
                message = f"Testing configuration: model_type: {model_type}"
                with self.subTest(i=message):
                    c = MultiHeadAttentionPresets.multi_head_attention_preset(
                        attention_option=AttentionOptions.INDEPENDENT,
                        model_type=model_type,
                    )
                    m = IndependentProjector(c)
                    self.assertEqual(model_type.value, m.model_type)

    def test__compute_projection(self):
        attention_options = [LinearLayerStackOptions, AdaptiveLayerStackOptions]

        for attention_option in attention_options:
            for model_type in attention_option:
                message = f"Testing configuration: model_type: {model_type}"
                with self.subTest(i=message):
                    c = MultiHeadAttentionPresets.multi_head_attention_preset(
                        attention_option=AttentionOptions.INDEPENDENT,
                        model_type=model_type,
                    )
                    m = IndependentProjector(c)

                    tensor = torch.randn(
                        c.target_sequence_length,
                        c.batch_size,
                        c.embedding_dim,
                    )

                    projected_tensor = m._compute_projection(tensor, m.query_model)

                    expected_shape = (
                        c.target_sequence_length,
                        c.batch_size,
                        c.query_key_projection_dim,
                    )
                    self.assertIsInstance(projected_tensor, torch.Tensor)
                    self.assertEqual(projected_tensor.shape, expected_shape)

    def test_compute_qkv_projections(self):
        attention_options = [LinearLayerStackOptions, AdaptiveLayerStackOptions]

        for attention_option in attention_options:
            for model_type in attention_option:
                for adaptive_type in AdaptiveWeightOptions:
                    message = f"Testing configuration: model_type: {model_type}, adaptive_type: {adaptive_type}"
                    with self.subTest(i=message):
                        if adaptive_type == AdaptiveWeightOptions.VECTOR:
                            continue
                        c = MultiHeadAttentionPresets.multi_head_attention_preset(
                            model_type=model_type,
                            projector_adaptive_weight_option=adaptive_type,
                        )
                        m = IndependentProjector(c)

                        tensor = torch.randn(
                            c.target_sequence_length,
                            c.batch_size,
                            c.embedding_dim,
                        )

                        q_projections, k_projections, v_projections = (
                            m.compute_qkv_projections(tensor, tensor, tensor)
                        )

                        expected_shape = (
                            c.target_sequence_length,
                            c.batch_size,
                            c.query_key_projection_dim,
                        )
                        expected_value_shape = (
                            c.target_sequence_length,
                            c.batch_size,
                            c.value_projection_dim,
                        )
                        self.assertIsInstance(q_projections, torch.Tensor)
                        self.assertIsInstance(k_projections, torch.Tensor)
                        self.assertIsInstance(v_projections, torch.Tensor)
                        self.assertEqual(q_projections.shape, expected_shape)
                        self.assertEqual(k_projections.shape, expected_shape)
                        self.assertEqual(v_projections.shape, expected_value_shape)


class TestMixtureOfAttentionHeadsProjector(unittest.TestCase):
    def test_init(self):
        boolean_options = [True, False]
        attention_options = [LinearLayerStackOptions]

        for use_kv_expert_models_flag in boolean_options:
            for attention_option in attention_options:
                for model_type in attention_option:
                    message = f"Testing configuration: model_type: {model_type}"
                    with self.subTest(i=message):
                        if model_type == AdaptiveWeightOptions.VECTOR:
                            continue
                        c = MultiHeadAttentionPresets.multi_head_attention_preset(
                            attention_option=AttentionOptions.MIXTURE_OF_ATTENTION_HEADS,
                            model_type=model_type,
                            use_kv_expert_models_flag=use_kv_expert_models_flag,
                        )
                        m = MixtureOfAttentionHeadsProjector(c)

                        self.assertIsInstance(m.query_model, MixtureOfExperts)
                        self.assertIsInstance(m.output_model, MixtureOfExperts)
                        if use_kv_expert_models_flag:
                            self.assertIsInstance(m.key_model, MixtureOfExperts)
                            self.assertIsInstance(m.value_model, MixtureOfExperts)
                        else:
                            self.assertIsInstance(m.key_model, Sequential)
                            self.assertIsInstance(m.value_model, Sequential)

    def test__compute_projection(self):
        attention_options = [LinearLayerStackOptions]

        for attention_option in attention_options:
            for model_type in attention_option:
                message = f"Testing configuration: model_type: {model_type}"
                with self.subTest(i=message):
                    if model_type == AdaptiveWeightOptions.VECTOR:
                        continue

                    c = MultiHeadAttentionPresets.multi_head_attention_preset(
                        attention_option=AttentionOptions.MIXTURE_OF_ATTENTION_HEADS,
                        model_type=model_type,
                        projector_experts_compute_expert_mixture_flag=False,
                        projector_experts_layer_stack_option=model_type,
                        projector_experts_stack_num_layers=1,
                    )
                    m = MixtureOfAttentionHeadsProjector(c)

                    tensor = torch.randn(
                        c.target_sequence_length,
                        c.batch_size,
                        c.embedding_dim,
                    )

                    projected_tensor = m._compute_kv_projection(tensor, m.key_model)

                    expected_shape = (
                        c.target_sequence_length,
                        c.batch_size,
                        c.query_key_projection_dim,
                    )
                    self.assertIsInstance(projected_tensor, torch.Tensor)
                    self.assertEqual(projected_tensor.shape, expected_shape)

    def test_compute_qkv_projections(self):
        boolean_options = [True, False]
        attention_options = [LinearLayerStackOptions]

        for use_kv_expert_models_flag in boolean_options:
            for attention_option in attention_options:
                for model_type in attention_option:
                    for adaptive_type in AdaptiveWeightOptions:
                        message = f"Testing configuration: model_type: {model_type}, adaptive_type: {adaptive_type}, use_kv_expert_models_flag: {use_kv_expert_models_flag}, attention_option: {attention_option}"
                        with self.subTest(i=message):
                            if adaptive_type == AdaptiveWeightOptions.VECTOR:
                                continue
                            c = MultiHeadAttentionPresets.multi_head_attention_preset(
                                model_type=model_type,
                                query_key_projection_dim=14,
                                value_projection_dim=14,
                                embedding_dim=14,
                                target_sequence_length=7,
                                source_sequence_length=7,
                                projector_adaptive_weight_option=adaptive_type,
                                use_kv_expert_models_flag=use_kv_expert_models_flag,
                                projector_experts_compute_expert_mixture_flag=False,
                                projector_experts_layer_stack_option=model_type,
                                projector_experts_stack_num_layers=1,
                                projector_experts_init_sampler_option=InitSamplerOptions.LAYER,
                            )
                            m = MixtureOfAttentionHeadsProjector(c)

                            tensor = torch.randn(
                                c.target_sequence_length,
                                c.batch_size,
                                c.embedding_dim,
                            )

                            q_projections, k_projections, v_projections = (
                                m.compute_qkv_projections(tensor, tensor, tensor)
                            )

                            expected_top_k_shape = (
                                c.target_sequence_length,
                                c.batch_size,
                                m.top_k,
                                c.query_key_projection_dim,
                            )

                            expected_shape = (
                                c.target_sequence_length,
                                c.batch_size,
                                c.query_key_projection_dim,
                            )

                            self.assertIsInstance(q_projections, torch.Tensor)
                            self.assertIsInstance(k_projections, torch.Tensor)
                            self.assertIsInstance(v_projections, torch.Tensor)

                            if use_kv_expert_models_flag:
                                self.assertEqual(
                                    q_projections.shape, expected_top_k_shape
                                )
                                self.assertEqual(
                                    k_projections.shape, expected_top_k_shape
                                )
                                self.assertEqual(
                                    v_projections.shape, expected_top_k_shape
                                )
                                continue

                            self.assertEqual(q_projections.shape, expected_top_k_shape)
                            self.assertEqual(k_projections.shape, expected_shape)
                            self.assertEqual(v_projections.shape, expected_shape)

    def test_compute_output_projection(self):
        attention_options = [LinearLayerStackOptions, AdaptiveLayerStackOptions]

        for attention_option in attention_options:
            for model_type in attention_option:
                for adaptive_type in AdaptiveWeightOptions:
                    message = f"Testing configuration: model_type: {model_type}, adaptive_type: {adaptive_type}"
                    with self.subTest(i=message):
                        if adaptive_type == AdaptiveWeightOptions.VECTOR:
                            continue
                        c = MultiHeadAttentionPresets.multi_head_attention_preset(
                            embedding_dim=12,
                            query_key_projection_dim=12,
                            value_projection_dim=12,
                            model_type=model_type,
                            projector_adaptive_weight_option=adaptive_type,
                        )
                        m = MixtureOfAttentionHeadsProjector(c)

                        tensor = torch.randn(
                            c.target_sequence_length,
                            c.batch_size,
                            c.embedding_dim,
                        )

                        q_projections, k_projections, v_projections = (
                            m.compute_qkv_projections(tensor, tensor, tensor)
                        )

                        weighted_values = torch.randn(
                            c.target_sequence_length,
                            c.batch_size,
                            m.top_k,
                            c.embedding_dim,
                        )

                        attentiion_output = m.compute_output_projection(weighted_values)

                        expected_shape = (
                            c.target_sequence_length * c.batch_size,
                            c.embedding_dim,
                        )
                        self.assertIsInstance(attentiion_output, torch.Tensor)
                        self.assertEqual(attentiion_output.shape, expected_shape)


class TestProjectorBuilder(unittest.TestCase):
    def test_init(self):
        model_types = [
            LinearLayerStackOptions,
            AdaptiveLayerStackOptions,
        ]
        boolean_options = [True, False]

        for attention_option in AttentionOptions:
            for type_options in model_types:
                for model_type in type_options:
                    for adaptive_type in AdaptiveWeightOptions:
                        for use_attention_flag in boolean_options:
                            message = f"Testing configuration: attention_option: {attention_option}, model_type: {model_type}, adaptive_type: {adaptive_type}, use_attention_flag: {use_attention_flag}"
                            with self.subTest(i=message):
                                if adaptive_type == AdaptiveWeightOptions.VECTOR:
                                    continue

                                c = MultiHeadAttentionPresets.multi_head_attention_preset(
                                    attention_option=attention_option,
                                    model_type=model_type,
                                    projector_adaptive_weight_option=adaptive_type,
                                    embedding_dim=12,
                                    query_key_projection_dim=12,
                                    value_projection_dim=12,
                                )
                                model = ProjectorBuilder(c).build()

                                expected_map = {
                                    AttentionOptions.SELF_ATTENTION: SelfAttentionProjector,
                                    AttentionOptions.INDEPENDENT: IndependentProjector,
                                    AttentionOptions.MIXTURE_OF_ATTENTION_HEADS: (
                                        MixtureOfAttentionHeadsProjector
                                    ),
                                }

                                expected_cls = expected_map.get(attention_option)
                                if expected_cls is None:
                                    raise ValueError(
                                        f"Attention option not tested or unknown option is given: {attention_option}"
                                    )

                                self.assertIsInstance(model, expected_cls)
