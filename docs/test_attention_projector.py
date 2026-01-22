import torch
import unittest

from torch.nn import Sequential

from Emperor.attention.utils.enums import ProjectorOptions
from Emperor.experts.utils.model import MixtureOfExpertsModel
from Emperor.linears.options import LinearLayerStackOptions
from Emperor.adaptive.options import AdaptiveLayerStackOptions
from Emperor.attention.utils.presets import MultiHeadAttentionPresets
from Emperor.adaptive.utils.mixtures.options import AdaptiveWeightOptions
from Emperor.attention.utils.handlers.projector import (
    IndependentProjector,
    MixtureOfAttentionHeadsProjector,
    ProjectorBuilder,
    SelfAttentionProjector,
)


class TestSelfAttentionProjector(unittest.TestCase):
    def setUp(self):
        self.cfg = MultiHeadAttentionPresets.multi_head_attention_preset()

    def test_init(self):
        projector_options = [LinearLayerStackOptions, AdaptiveLayerStackOptions]

        for projector_option in projector_options:
            for model_type in projector_option:
                message = f"Testing configuration: model_type: {model_type}"
                with self.subTest(i=message):
                    c = MultiHeadAttentionPresets.multi_head_attention_preset(
                        model_type=model_type
                    )
                    m = SelfAttentionProjector(c)
                    self.assertEqual(model_type.value, m.model_type)

    def test__split_self_attention_projection(self):
        c = MultiHeadAttentionPresets.multi_head_attention_preset()
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
        projector_options = [LinearLayerStackOptions, AdaptiveLayerStackOptions]

        for projector_option in projector_options:
            for model_type in projector_option:
                message = f"Testing configuration: model_type: {model_type}"
                with self.subTest(i=message):
                    c = MultiHeadAttentionPresets.multi_head_attention_preset(
                        model_type=model_type
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
        projector_options = [LinearLayerStackOptions, AdaptiveLayerStackOptions]

        for projector_option in projector_options:
            for model_type in projector_option:
                for adaptive_type in AdaptiveWeightOptions:
                    message = f"Testing configuration: model_type: {model_type}, adaptive_type: {adaptive_type}"
                    with self.subTest(i=message):
                        if adaptive_type == AdaptiveWeightOptions.VECTOR:
                            continue
                        c = MultiHeadAttentionPresets.multi_head_attention_preset(
                            model_type=model_type,
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
        projector_options = [LinearLayerStackOptions, AdaptiveLayerStackOptions]

        for projector_option in projector_options:
            for model_type in projector_option:
                for adaptive_type in AdaptiveWeightOptions:
                    message = f"Testing configuration: model_type: {model_type}, adaptive_type: {adaptive_type}"
                    with self.subTest(i=message):
                        if adaptive_type == AdaptiveWeightOptions.VECTOR:
                            continue
                        c = MultiHeadAttentionPresets.multi_head_attention_preset(
                            model_type=model_type,
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
    def setUp(self):
        self.cfg = MultiHeadAttentionPresets.multi_head_attention_preset()

    def test_init(self):
        projector_options = [LinearLayerStackOptions, AdaptiveLayerStackOptions]

        for projector_option in projector_options:
            for model_type in projector_option:
                message = f"Testing configuration: model_type: {model_type}"
                with self.subTest(i=message):
                    c = MultiHeadAttentionPresets.multi_head_attention_preset(
                        model_type=model_type
                    )
                    m = IndependentProjector(c)
                    self.assertEqual(model_type.value, m.model_type)

    def test__compute_projection(self):
        projector_options = [LinearLayerStackOptions, AdaptiveLayerStackOptions]

        for projector_option in projector_options:
            for model_type in projector_option:
                message = f"Testing configuration: model_type: {model_type}"
                with self.subTest(i=message):
                    c = MultiHeadAttentionPresets.multi_head_attention_preset(
                        model_type=model_type
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
        projector_options = [LinearLayerStackOptions, AdaptiveLayerStackOptions]

        for projector_option in projector_options:
            for model_type in projector_option:
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
    def setUp(self):
        self.cfg = MultiHeadAttentionPresets.multi_head_attention_preset()

    def test_init(self):
        boolean_options = [True, False]
        projector_options = [LinearLayerStackOptions, AdaptiveLayerStackOptions]

        for use_kv_expert_models_flag in boolean_options:
            for projector_option in projector_options:
                for model_type in projector_option:
                    message = f"Testing configuration: model_type: {model_type}"
                    with self.subTest(i=message):
                        c = MultiHeadAttentionPresets.multi_head_attention_preset(
                            model_type=model_type,
                            use_kv_expert_models_flag=use_kv_expert_models_flag,
                        )
                        m = MixtureOfAttentionHeadsProjector(c)

                        self.assertIsInstance(m.query_model, MixtureOfExpertsModel)
                        self.assertIsInstance(m.output_model, MixtureOfExpertsModel)
                        if use_kv_expert_models_flag:
                            self.assertIsInstance(m.key_model, MixtureOfExpertsModel)
                            self.assertIsInstance(m.value_model, MixtureOfExpertsModel)
                        else:
                            self.assertIsInstance(m.key_model, Sequential)
                            self.assertIsInstance(m.value_model, Sequential)

    def test__compute_projection(self):
        boolean_options = [True, False]
        projector_options = [LinearLayerStackOptions]

        for use_kv_expert_models_flag in boolean_options:
            for projector_option in projector_options:
                for model_type in projector_option:
                    message = f"Testing configuration: model_type: {model_type}"
                    with self.subTest(i=message):
                        c = MultiHeadAttentionPresets.multi_head_attention_preset(
                            model_type=model_type,
                            use_kv_expert_models_flag=use_kv_expert_models_flag,
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

                        if use_kv_expert_models_flag:
                            expected_shape = (
                                c.target_sequence_length,
                                c.batch_size,
                                m.top_k,
                                c.query_key_projection_dim,
                            )
                            self.assertIsInstance(projected_tensor, torch.Tensor)
                            self.assertEqual(projected_tensor.shape, expected_shape)
                            continue

                        expected_shape = (
                            c.target_sequence_length,
                            c.batch_size,
                            c.query_key_projection_dim,
                        )
                        self.assertIsInstance(projected_tensor, torch.Tensor)
                        self.assertEqual(projected_tensor.shape, expected_shape)

    def test_compute_qkv_projections(self):
        boolean_options = [True, False]
        projector_options = [LinearLayerStackOptions]

        for use_kv_expert_models_flag in boolean_options:
            for projector_option in projector_options:
                for model_type in projector_option:
                    for adaptive_type in AdaptiveWeightOptions:
                        message = f"Testing configuration: model_type: {model_type}, adaptive_type: {adaptive_type}"
                        with self.subTest(i=message):
                            if adaptive_type == AdaptiveWeightOptions.VECTOR:
                                continue
                            c = MultiHeadAttentionPresets.multi_head_attention_preset(
                                model_type=model_type,
                                query_key_projection_dim=14,
                                value_projection_dim=14,
                                projector_adaptive_weight_option=adaptive_type,
                                use_kv_expert_models_flag=use_kv_expert_models_flag,
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


class TestProjectorBuilder(unittest.TestCase):
    def setUp(self):
        self.cfg = MultiHeadAttentionPresets.multi_head_attention_preset()

    def test_init(self):
        projector_options = [LinearLayerStackOptions, AdaptiveLayerStackOptions]
        boolean_options = [True, False]

        for projector_option in projector_options:
            for model_type in projector_option:
                for adaptive_type in AdaptiveWeightOptions:
                    for projector_option in boolean_options:
                        message = f"Testing configuration: model_type: {model_type}, adaptive_type: {adaptive_type}, projector_option: {projector_option}"
                        with self.subTest(i=message):
                            if adaptive_type == AdaptiveWeightOptions.VECTOR:
                                continue

                            projector_options = {
                                "projector_option": ProjectorOptions.INDEPENDENT,
                                "embedding_dim": 15,
                                "query_key_projection_dim": 10,
                                "value_projection_dim": 12,
                            }
                            if projector_option:
                                projector_options = {
                                    "projector_option": ProjectorOptions.SELF_ATTENTION,
                                    "embedding_dim": 12,
                                    "query_key_projection_dim": 12,
                                    "value_projection_dim": 12,
                                }

                            c = MultiHeadAttentionPresets.multi_head_attention_preset(
                                model_type=model_type,
                                projector_adaptive_weight_option=adaptive_type,
                                **projector_options,
                            )
                            model = ProjectorBuilder(c).build_model()

                            if projector_option:
                                self.assertIsInstance(model, SelfAttentionProjector)
                            else:
                                self.assertIsInstance(model, IndependentProjector)
