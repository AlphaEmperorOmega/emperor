import math
import unittest
from unittest.mock import patch

import torch
from emperor.attention import (
    IndependentAttentionConfig,
    MixtureOfAttentionHeadsConfig,
    SelfAttentionConfig,
)
from emperor.attention.core.config import MultiHeadAttentionConfig
from emperor.attention.core.handlers.processor import ProcessorBase
from emperor.attention.core.handlers.reshaper import AttentionReshaper
from emperor.attention.core.runtime import QKV
from emperor.attention.core.variants.independent_attention.processor import (
    IndependentProcessor,
)
from emperor.attention.core.variants.independent_attention.projector import (
    IndependentProjector,
)
from emperor.attention.core.variants.mixture_of_attention_heads.processor import (
    MixtureOfAttentionHeadsProcessor,
)
from emperor.attention.core.variants.mixture_of_attention_heads.projector import (
    MixtureOfAttentionHeadsProjector,
)
from emperor.attention.core.variants.mixture_of_attention_heads.reshaper import (
    MixtureOfAttentionHeadsReshaper,
)
from emperor.attention.core.variants.self_attention.processor import (
    SelfAttentionProcessor,
)
from emperor.attention.core.variants.self_attention.projector import (
    SelfAttentionProjector,
)
from torch.types import Tensor

from support.attention import (
    RELATIVE_POSITIONAL_EMBEDDING_CASES,
    build_attention_config,
)

PROJECTION_KINDS = ["base", "adaptive"]


class IdentityOutputProjector:
    def compute_output_projection(self, tensor):
        return tensor


class CapturingRelativeEmbedding(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.received_query = None
        self.received_source_length = None

    def forward(self, query, sequence_length, last=False):
        self.received_query = query
        self.received_source_length = sequence_length
        return query.new_zeros(*query.shape[:-1], sequence_length)


def create_attention_mask(
    c: "MultiHeadAttentionConfig", is_four_dinemsional=False
) -> Tensor:
    attn_mask_shape = (c.target_sequence_length, c.source_sequence_length)
    attention_mask = torch.zeros(*attn_mask_shape)
    bool_attention_mask = torch.ones(*attn_mask_shape, dtype=torch.bool)
    bool_attention_mask = bool_attention_mask.tril(diagonal=0)
    attention_mask = attention_mask.masked_fill(
        bool_attention_mask.logical_not(), float("-inf")
    )
    attention_mask = attention_mask.repeat(c.batch_size * c.num_heads, 1, 1)
    if is_four_dinemsional:
        return attention_mask.reshape(
            c.batch_size,
            c.num_heads,
            c.target_sequence_length,
            c.source_sequence_length,
        )
    return attention_mask


class TestProcessorBase(unittest.TestCase):
    def model(self):
        cfg = build_attention_config(
            config_class=IndependentAttentionConfig,
            batch_size=2,
            num_heads=2,
            embedding_dim=4,
            target_sequence_length=2,
            source_sequence_length=2,
        )
        projector = IndependentProjector(cfg)
        return ProcessorBase(cfg, projector, AttentionReshaper(cfg))

    def test_single_batch_detection_uses_leading_dimension(self):
        model = self.model()

        self.assertTrue(model._is_single_batch(torch.empty(1, 2, 4)))
        self.assertFalse(model._is_single_batch(torch.empty(2, 2, 4)))

    def test_compute_attention_is_abstract(self):
        model = self.model()
        tensor = torch.randn(4, 2, 2)

        with self.assertRaises(NotImplementedError) as caught:
            model.compute_attention(QKV(query=tensor, key=tensor, value=tensor))
        self.assertEqual(
            str(caught.exception),
            "compute_attention must be implemented by subclass.",
        )

    def test_effective_projection_and_head_widths_use_integer_contracts(self):
        cfg = build_attention_config(
            config_class=IndependentAttentionConfig,
            num_heads=2,
            embedding_dim=8,
            query_key_projection_dim=0,
            value_projection_dim=0,
        )
        model = ProcessorBase(
            cfg,
            IdentityOutputProjector(),
            AttentionReshaper(cfg),
        )

        self.assertEqual(model.query_key_projection_dim, 8)
        self.assertEqual(model.value_projection_dim, 8)
        self.assertEqual(model.qk_head_dim, 4)
        self.assertEqual(model.v_head_dim, 4)
        self.assertIs(type(model.qk_head_dim), int)
        self.assertIs(type(model.v_head_dim), int)

    def test_relative_logits_scale_by_head_width_and_pad_synthetic_sources(self):
        cfg = build_attention_config(
            config_class=IndependentAttentionConfig,
            batch_size=1,
            num_heads=1,
            embedding_dim=2,
            query_key_projection_dim=2,
            value_projection_dim=2,
            target_sequence_length=3,
            source_sequence_length=5,
        )
        model = ProcessorBase(
            cfg,
            IdentityOutputProjector(),
            AttentionReshaper(cfg),
        )
        relative = CapturingRelativeEmbedding()
        model.relative_positional_embedding = relative
        query = torch.arange(6.0).view(1, 3, 2)

        logits = model._compute_relative_position_logits(query, 5)

        expected_scaled_query = query.view(1, 1, 3, 2) * (2**-0.5)
        torch.testing.assert_close(relative.received_query, expected_scaled_query)
        self.assertEqual(relative.received_source_length, 5)
        self.assertEqual(logits.shape, (1, 3, 5))

    def test_attention_output_uses_last_axis_as_embedding_width(self):
        cfg = build_attention_config(
            config_class=IndependentAttentionConfig,
            batch_size=2,
            num_heads=1,
            embedding_dim=5,
            query_key_projection_dim=5,
            value_projection_dim=5,
            target_sequence_length=3,
            source_sequence_length=4,
        )
        model = ProcessorBase(
            cfg,
            IdentityOutputProjector(),
            AttentionReshaper(cfg),
        )
        weighted_values = torch.arange(30.0).view(6, 5)

        output = model._compute_attention_output(weighted_values)

        self.assertEqual(output.shape, (3, 2, 5))
        torch.testing.assert_close(output, weighted_values.view(3, 2, 5))


class TestSelfAttentionProcessor(unittest.TestCase):
    def test_compute_attention_applies_the_supplied_mask(self):
        cfg = build_attention_config(
            config_class=SelfAttentionConfig,
            batch_size=1,
            num_heads=1,
            embedding_dim=2,
            target_sequence_length=2,
            source_sequence_length=2,
            dropout_probability=0.0,
            return_attention_weights_flag=True,
            average_attention_weights_flag=False,
        )
        model = SelfAttentionProcessor(
            cfg,
            SelfAttentionProjector(cfg),
            AttentionReshaper(cfg),
        ).eval()
        query = torch.zeros(1, 2, 2)
        key = torch.zeros(1, 2, 2)
        value = torch.tensor([[[2.0, 0.0], [0.0, 4.0]]])
        mask = torch.tensor([[[0.0, -torch.inf], [0.0, -torch.inf]]])

        _, weights = model.compute_attention(
            QKV(query=query, key=key, value=value),
            mask,
        )

        expected = torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]])
        torch.testing.assert_close(weights, expected)

    def test_training_dropout_uses_the_configured_probability(self):
        cfg = build_attention_config(
            config_class=SelfAttentionConfig,
            batch_size=1,
            num_heads=1,
            embedding_dim=2,
            target_sequence_length=2,
            source_sequence_length=2,
            dropout_probability=0.25,
        )
        model = SelfAttentionProcessor(
            cfg,
            SelfAttentionProjector(cfg),
            AttentionReshaper(cfg),
        ).train()
        query = torch.zeros(1, 2, 2)
        key = torch.zeros(1, 2, 2)

        torch.manual_seed(37)
        actual = model._SelfAttentionProcessor__compute_masked_attention_weights(
            query,
            key,
            None,
        )
        torch.manual_seed(37)
        expected = torch.nn.functional.dropout(
            torch.full((1, 2, 2), 0.5),
            p=0.25,
            training=True,
        )

        torch.testing.assert_close(actual, expected)

    def test_relative_logits_receive_scaled_query_contract_and_are_added(self):
        cfg = build_attention_config(
            config_class=SelfAttentionConfig,
            batch_size=1,
            num_heads=1,
            embedding_dim=2,
            target_sequence_length=2,
            source_sequence_length=3,
        )
        model = SelfAttentionProcessor(
            cfg,
            SelfAttentionProjector(cfg),
            AttentionReshaper(cfg),
        )
        query = torch.arange(4.0).view(1, 2, 2)
        weights = torch.arange(6.0).view(1, 2, 3)
        positional = torch.full_like(weights, 1.25)
        runtime_shape = object()

        with patch.object(
            model,
            "_compute_relative_position_logits",
            return_value=positional,
        ) as compute_relative:
            actual = (
                model._SelfAttentionProcessor__maybe_add_relative_positional_embedding(
                    query,
                    weights,
                    runtime_shape,
                )
            )

        args = compute_relative.call_args.args
        self.assertIs(args[0], query)
        self.assertEqual(args[1], 3)
        self.assertIs(args[2], runtime_shape)
        self.assertIs(compute_relative.call_args.kwargs["query_is_scaled"], True)
        torch.testing.assert_close(actual, weights + positional)

    def test_legacy_shape_helper_does_not_squeeze_rank_two_output(self):
        cfg = build_attention_config(
            config_class=SelfAttentionConfig,
            batch_size=1,
            num_heads=1,
            embedding_dim=2,
        )
        model = SelfAttentionProcessor(
            cfg,
            SelfAttentionProjector(cfg),
            AttentionReshaper(cfg),
        )
        output = torch.arange(2.0).view(2, 1)
        weights = torch.arange(4.0).view(1, 2, 2)

        actual_output, actual_weights = (
            model._SelfAttentionProcessor__handle_batched_input(output, weights)
        )

        self.assertIs(actual_output, output)
        self.assertIs(actual_weights, weights)

    def test_seeded_dropout_changes_training_weights_only(self):
        cfg = build_attention_config(
            config_class=SelfAttentionConfig,
            batch_size=1,
            num_heads=1,
            embedding_dim=2,
            target_sequence_length=3,
            source_sequence_length=3,
            dropout_probability=0.5,
        )
        model = SelfAttentionProcessor(
            cfg,
            SelfAttentionProjector(cfg),
            AttentionReshaper(cfg),
        )
        query = torch.zeros(1, 3, 2)
        key = torch.zeros(1, 3, 2)

        model.train()
        torch.manual_seed(7)
        first = model._SelfAttentionProcessor__compute_masked_attention_weights(
            query,
            key,
            None,
        )
        torch.manual_seed(7)
        second = model._SelfAttentionProcessor__compute_masked_attention_weights(
            query,
            key,
            None,
        )
        model.eval()
        evaluation = model._SelfAttentionProcessor__compute_masked_attention_weights(
            query,
            key,
            None,
        )

        torch.testing.assert_close(first, second)
        self.assertFalse(torch.equal(first, evaluation))
        torch.testing.assert_close(
            evaluation,
            torch.full_like(evaluation, 1 / 3),
        )

    def test_legacy_shape_helper_omits_weights_when_disabled(self):
        cfg = build_attention_config(
            config_class=SelfAttentionConfig,
            batch_size=1,
            num_heads=1,
            embedding_dim=2,
            target_sequence_length=2,
            source_sequence_length=2,
            return_attention_weights_flag=False,
        )
        model = SelfAttentionProcessor(
            cfg,
            SelfAttentionProjector(cfg),
            AttentionReshaper(cfg),
        )
        output = torch.randn(2, 1, 2)
        weights = torch.randn(1, 2, 2)

        actual_output, actual_weights = (
            model._SelfAttentionProcessor__ensure_correct_shape_output(
                output,
                weights,
            )
        )

        self.assertIs(actual_output, output)
        self.assertIsNone(actual_weights)

    def test_init(self):
        attention_options = [PROJECTION_KINDS]

        for attention_option in attention_options:
            for model_type in attention_option:
                message = f"Testing configuration: model_type: {model_type}"
                with self.subTest(i=message):
                    c = build_attention_config(
                        config_class=SelfAttentionConfig,
                        projection_kind=model_type,
                        embedding_dim=12,
                        query_key_projection_dim=12,
                        value_projection_dim=12,
                        return_attention_weights_flag=True,
                    )
                    projector = SelfAttentionProjector(c)
                    m = SelfAttentionProcessor(c, projector, AttentionReshaper(c))
                    self.assertIsInstance(m, SelfAttentionProcessor)
                    self.assertIsInstance(m.projector, SelfAttentionProjector)
                    self.assertIsNone(m.relative_positional_embedding)

    def test__scale_query(self):
        c = build_attention_config(
            config_class=SelfAttentionConfig,
            embedding_dim=12,
            query_key_projection_dim=12,
            value_projection_dim=12,
            return_attention_weights_flag=True,
        )
        head_dim = c.embedding_dim // c.num_heads
        query = torch.randn(
            c.batch_size * c.num_heads, c.target_sequence_length, head_dim
        )

        projector = SelfAttentionProjector(c)
        m = SelfAttentionProcessor(c, projector, AttentionReshaper(c))
        scaled_query_tensor = m._SelfAttentionProcessor__scale_query(query)

        expected_result = query * math.sqrt(1.0 / float(head_dim))
        self.assertIsInstance(scaled_query_tensor, torch.Tensor)
        self.assertEqual(query.shape, scaled_query_tensor.shape)
        self.assertTrue(
            torch.allclose(scaled_query_tensor, expected_result, atol=1e-6, rtol=1e-5)
        )

    def test__compute_raw_masked_attention_weights(self):
        source_sequence_length = 8
        target_sequence_length = 8
        embedding_dim = 12
        query_key_projection_dim = 12
        value_projection_dim = 12
        return_attention_weights_flag = True
        c = build_attention_config(
            config_class=SelfAttentionConfig,
            source_sequence_length=source_sequence_length,
            target_sequence_length=target_sequence_length,
            embedding_dim=embedding_dim,
            query_key_projection_dim=query_key_projection_dim,
            value_projection_dim=value_projection_dim,
            return_attention_weights_flag=return_attention_weights_flag,
        )
        head_dim = c.embedding_dim // c.num_heads
        query = torch.randn(
            c.batch_size * c.num_heads, c.target_sequence_length, head_dim
        )
        key = torch.randn(
            c.batch_size * c.num_heads, c.source_sequence_length, head_dim
        )

        attention_mask = create_attention_mask(c)
        attention_mask_options = [None, attention_mask]

        for relative_case, relative_config_cls in RELATIVE_POSITIONAL_EMBEDDING_CASES:
            for attention_mask_option in attention_mask_options:
                message = (
                    "Testing configuration: "
                    f"relative_positional_embedding: {relative_case}, "
                    f"attention_mask_option: {type(attention_mask_option)}"
                )
                with self.subTest(i=message):
                    c = build_attention_config(
                        config_class=SelfAttentionConfig,
                        source_sequence_length=source_sequence_length,
                        target_sequence_length=target_sequence_length,
                        embedding_dim=embedding_dim,
                        query_key_projection_dim=query_key_projection_dim,
                        value_projection_dim=value_projection_dim,
                        return_attention_weights_flag=return_attention_weights_flag,
                        relative_positional_embedding_config_cls=relative_config_cls,
                    )
                    projector = SelfAttentionProjector(c)
                    m = SelfAttentionProcessor(c, projector, AttentionReshaper(c))
                    raw_masked_weights = (
                        m._SelfAttentionProcessor__compute_raw_masked_attention_weights(
                            query, key, attention_mask_option
                        )
                    )

                    expected_shape = (
                        c.batch_size * c.num_heads,
                        c.target_sequence_length,
                        c.source_sequence_length,
                    )
                    self.assertIsInstance(raw_masked_weights, torch.Tensor)
                    self.assertEqual(raw_masked_weights.shape, expected_shape)

                    if (
                        attention_mask_option is not None
                        and relative_config_cls is None
                    ):
                        transposed_keys = key.transpose(-2, -1)
                        for idx in range(key.size(0)):
                            q = query[idx]
                            k = transposed_keys[idx]
                            single_head_qk_attention_weights = torch.mm(
                                q, k
                            ).masked_fill(
                                (attention_mask[idx] == float("-inf")),
                                torch.tensor(float("-inf")),
                            )
                            self.assertTrue(
                                torch.allclose(
                                    raw_masked_weights[idx],
                                    single_head_qk_attention_weights,
                                    atol=1e-6,
                                    rtol=1e-5,
                                )
                            )

    def test__compute_masked_attention_weights(self):
        c = build_attention_config(
            config_class=SelfAttentionConfig,
            source_sequence_length=8,
            target_sequence_length=8,
            embedding_dim=12,
            query_key_projection_dim=12,
            value_projection_dim=12,
            return_attention_weights_flag=True,
        )
        projector = SelfAttentionProjector(c)
        m = SelfAttentionProcessor(c, projector, AttentionReshaper(c))
        head_dim = c.embedding_dim // c.num_heads

        query = torch.randn(
            c.batch_size * c.num_heads, c.target_sequence_length, head_dim
        )
        key = torch.randn(
            c.batch_size * c.num_heads, c.source_sequence_length, head_dim
        )
        attention_mask_options = [None, create_attention_mask(c)]

        for attention_mask_option in attention_mask_options:
            with self.subTest(
                attention_mask_type=type(attention_mask_option).__name__,
            ):
                result = m._SelfAttentionProcessor__compute_masked_attention_weights(
                    query, key, attention_mask_option
                )

                expected_shape = (
                    c.batch_size * c.num_heads,
                    c.target_sequence_length,
                    c.source_sequence_length,
                )
                self.assertEqual(result.shape, expected_shape)

    def test__compute_weighted_values(self):
        c = build_attention_config(
            config_class=SelfAttentionConfig,
            source_sequence_length=8,
            target_sequence_length=8,
            embedding_dim=12,
            query_key_projection_dim=12,
            value_projection_dim=12,
            return_attention_weights_flag=True,
        )
        projector = SelfAttentionProjector(c)
        m = SelfAttentionProcessor(c, projector, AttentionReshaper(c))

        head_dim = c.embedding_dim // c.num_heads

        attention_weights = torch.randn(
            c.batch_size * c.num_heads,
            c.target_sequence_length,
            c.source_sequence_length,
        )
        values = torch.randn(
            c.batch_size * c.num_heads, c.source_sequence_length, head_dim
        )

        weighted_values = m._SelfAttentionProcessor__compute_weighted_values(
            attention_weights, values
        )

        expected_shape = (c.batch_size * c.target_sequence_length, c.embedding_dim)
        self.assertEqual(weighted_values.shape, expected_shape)

    def test__compute_attention_output(self):
        c = build_attention_config(
            config_class=SelfAttentionConfig,
            source_sequence_length=8,
            target_sequence_length=8,
            embedding_dim=12,
            query_key_projection_dim=12,
            value_projection_dim=12,
            return_attention_weights_flag=True,
        )
        projector = SelfAttentionProjector(c)
        m = SelfAttentionProcessor(c, projector, AttentionReshaper(c))

        weighted_values = torch.randn(
            c.batch_size * c.target_sequence_length, c.embedding_dim
        )

        weighted_values = m._compute_attention_output(weighted_values)

        expected_shape = (c.target_sequence_length, c.batch_size, c.embedding_dim)
        self.assertEqual(weighted_values.shape, expected_shape)

    def test__maybe_average_attention_weights(self):
        boolean_options = [True, False]

        for average_attention_weights_flag in boolean_options:
            with self.subTest(
                average_attention_weights_flag=average_attention_weights_flag,
            ):
                c = build_attention_config(
                    config_class=SelfAttentionConfig,
                    source_sequence_length=8,
                    target_sequence_length=8,
                    embedding_dim=12,
                    query_key_projection_dim=12,
                    value_projection_dim=12,
                    return_attention_weights_flag=True,
                    average_attention_weights_flag=average_attention_weights_flag,
                )
                projector = SelfAttentionProjector(c)
                m = SelfAttentionProcessor(c, projector, AttentionReshaper(c))

                attention_weights = torch.randn(
                    c.batch_size,
                    c.num_heads,
                    c.target_sequence_length,
                    c.source_sequence_length,
                )

                output_attention_weights = (
                    m._SelfAttentionProcessor__maybe_average_attention_weights(
                        attention_weights
                    )
                )

                if average_attention_weights_flag:
                    expected_shape = (
                        c.batch_size,
                        c.target_sequence_length,
                        c.source_sequence_length,
                    )
                    self.assertEqual(output_attention_weights.shape, expected_shape)
                else:
                    expected_shape = (
                        c.batch_size,
                        c.num_heads,
                        c.target_sequence_length,
                        c.source_sequence_length,
                    )
                    self.assertEqual(output_attention_weights.shape, expected_shape)
                    self.assertTrue(
                        torch.allclose(
                            output_attention_weights,
                            attention_weights,
                            atol=1e-6,
                            rtol=1e-5,
                        )
                    )

    def test__handle_batched_input(self):
        batch_size_options = [1, 4]

        for batch_size_option in batch_size_options:
            message = f"Testing configuration: batch_size_option: {batch_size_option}"
            with self.subTest(i=message):
                c = build_attention_config(
                    config_class=SelfAttentionConfig,
                    source_sequence_length=8,
                    target_sequence_length=8,
                    embedding_dim=12,
                    query_key_projection_dim=12,
                    value_projection_dim=12,
                    return_attention_weights_flag=True,
                    batch_size=batch_size_option,
                )
                projector = SelfAttentionProjector(c)

                m = SelfAttentionProcessor(c, projector, AttentionReshaper(c))

                attention_output = torch.randn(
                    c.target_sequence_length, c.batch_size, c.embedding_dim
                )
                attention_weights = torch.randn(
                    c.batch_size,
                    c.num_heads,
                    c.target_sequence_length,
                    c.source_sequence_length,
                )

                output_attention_output, output_attention_weights = (
                    m._SelfAttentionProcessor__handle_batched_input(
                        attention_output, attention_weights
                    )
                )

                if batch_size_option == 1:
                    expected_output_shape = (c.target_sequence_length, c.embedding_dim)
                    epxected_weights_shape = (
                        c.num_heads,
                        c.target_sequence_length,
                        c.source_sequence_length,
                    )
                    self.assertEqual(
                        output_attention_output.shape, expected_output_shape
                    )
                    self.assertEqual(
                        output_attention_weights.shape, epxected_weights_shape
                    )
                else:
                    self.assertTrue(
                        torch.allclose(
                            output_attention_output,
                            attention_output,
                            atol=1e-6,
                            rtol=1e-5,
                        )
                    )
                    self.assertTrue(
                        torch.allclose(
                            output_attention_weights,
                            attention_weights,
                            atol=1e-6,
                            rtol=1e-5,
                        )
                    )

    def test__ensure_correct_shape_output(self):
        c = build_attention_config(
            config_class=SelfAttentionConfig,
            source_sequence_length=8,
            target_sequence_length=8,
            embedding_dim=12,
            query_key_projection_dim=12,
            value_projection_dim=12,
            return_attention_weights_flag=True,
        )
        projector = SelfAttentionProjector(c)
        m = SelfAttentionProcessor(c, projector, AttentionReshaper(c))

        attention_output = torch.randn(
            c.target_sequence_length, c.batch_size, c.embedding_dim
        )
        attention_weights = torch.randn(
            c.batch_size * c.num_heads,
            c.target_sequence_length,
            c.source_sequence_length,
        )

        output_attention_output, output_attention_weights = (
            m._SelfAttentionProcessor__ensure_correct_shape_output(
                attention_output, attention_weights
            )
        )

        expected_output_shape = (
            c.target_sequence_length,
            c.batch_size,
            c.embedding_dim,
        )
        expected_weights_shape = (
            c.batch_size,
            c.num_heads,
            c.target_sequence_length,
            c.source_sequence_length,
        )
        self.assertEqual(output_attention_output.shape, expected_output_shape)
        self.assertEqual(output_attention_weights.shape, expected_weights_shape)

    def test__average_attention_weights_flag__True(self):
        boolean_options = [True, False]
        batch_size_options = [1, 4]

        c = build_attention_config(
            config_class=SelfAttentionConfig,
            source_sequence_length=8,
            target_sequence_length=8,
            embedding_dim=12,
            query_key_projection_dim=12,
            value_projection_dim=12,
            return_attention_weights_flag=True,
        )
        head_dim = c.embedding_dim // c.num_heads

        query = torch.randn(
            c.batch_size * c.num_heads, c.source_sequence_length, head_dim
        )
        key = torch.randn(
            c.batch_size * c.num_heads, c.target_sequence_length, head_dim
        )
        value = torch.randn(
            c.batch_size * c.num_heads, c.target_sequence_length, head_dim
        )
        attention_mask = create_attention_mask(c)

        for average_attention_weights_flag in boolean_options:
            for batch_size_option in batch_size_options:
                with self.subTest(
                    batch_size_option=batch_size_option,
                    average_attention_weights_flag=average_attention_weights_flag,
                ):
                    c.average_attention_weights_flag = average_attention_weights_flag
                    projector = SelfAttentionProjector(c)
                    m = SelfAttentionProcessor(c, projector, AttentionReshaper(c))
                    output_attention_output, output_attention_weights = (
                        m.compute_attention(
                            QKV(query=query, key=key, value=value),
                            attention_mask,
                        )
                    )

                    if average_attention_weights_flag:
                        expected_output_shape = (
                            c.target_sequence_length,
                            c.batch_size,
                            c.embedding_dim,
                        )
                        expected_weights_shape = (
                            c.batch_size,
                            c.target_sequence_length,
                            c.source_sequence_length,
                        )
                    else:
                        expected_output_shape = (
                            c.target_sequence_length,
                            c.batch_size,
                            c.embedding_dim,
                        )
                        expected_weights_shape = (
                            c.batch_size,
                            c.num_heads,
                            c.target_sequence_length,
                            c.source_sequence_length,
                        )

                    self.assertEqual(
                        output_attention_output.shape, expected_output_shape
                    )
                    self.assertEqual(
                        output_attention_weights.shape, expected_weights_shape
                    )


class TestIndependentProcessor(unittest.TestCase):
    def test_compute_attention_prepares_the_supplied_mask(self):
        cfg = build_attention_config(
            config_class=IndependentAttentionConfig,
            batch_size=1,
            num_heads=1,
            embedding_dim=2,
            query_key_projection_dim=2,
            value_projection_dim=2,
            target_sequence_length=2,
            source_sequence_length=2,
        )
        model = IndependentProcessor(
            cfg,
            IndependentProjector(cfg),
            AttentionReshaper(cfg),
        ).eval()
        tensor = torch.zeros(1, 2, 2)
        mask = torch.tensor([[[0.0, -torch.inf], [0.0, -torch.inf]]])
        method_name = "_IndependentProcessor__prepare_attention_mask"

        with patch.object(
            model,
            method_name,
            wraps=getattr(model, method_name),
        ) as prepare_mask:
            model.compute_attention(
                QKV(query=tensor, key=tensor, value=tensor),
                mask,
            )

        self.assertIs(prepare_mask.call_args.args[0], mask)

    def test_weighted_values_match_sdpa_dropout_in_train_and_eval_modes(self):
        cfg = build_attention_config(
            config_class=IndependentAttentionConfig,
            batch_size=1,
            num_heads=1,
            embedding_dim=2,
            query_key_projection_dim=2,
            value_projection_dim=2,
            target_sequence_length=2,
            source_sequence_length=2,
            dropout_probability=0.25,
        )
        model = IndependentProcessor(
            cfg,
            IndependentProjector(cfg),
            AttentionReshaper(cfg),
        )
        query = torch.zeros(1, 1, 2, 2)
        key = torch.zeros(1, 1, 2, 2)
        value = torch.tensor([[[[2.0, 0.0], [0.0, 4.0]]]])
        compute = model._IndependentProcessor__compute_weighted_values

        model.train()
        torch.manual_seed(43)
        actual_training = compute(query, key, value, None)
        torch.manual_seed(43)
        expected_training = torch.nn.functional.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=0.25,
        )
        expected_training = expected_training.permute(2, 0, 1, 3).reshape(2, 2)

        model.eval()
        actual_evaluation = compute(query, key, value, None)
        expected_evaluation = torch.nn.functional.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=0.0,
        )
        expected_evaluation = expected_evaluation.permute(2, 0, 1, 3).reshape(2, 2)

        torch.testing.assert_close(actual_training, expected_training)
        torch.testing.assert_close(actual_evaluation, expected_evaluation)

    def test_init(self):
        attention_options = [PROJECTION_KINDS]

        for attention_option in attention_options:
            for model_type in attention_option:
                message = f"Testing configuration: model_type: {model_type}"
                with self.subTest(i=message):
                    c = build_attention_config(
                        config_class=IndependentAttentionConfig,
                        projection_kind=model_type,
                        embedding_dim=12,
                        query_key_projection_dim=20,
                        value_projection_dim=16,
                    )
                    projector = IndependentProjector(c)
                    m = IndependentProcessor(c, projector, AttentionReshaper(c))
                    self.assertIsInstance(m, IndependentProcessor)
                    self.assertIsInstance(m.projector, IndependentProjector)
                    self.assertIsNone(m.relative_positional_embedding)

    def test__prepare_attnetion_mask(self):
        c = build_attention_config(
            config_class=IndependentAttentionConfig,
            embedding_dim=12,
            query_key_projection_dim=20,
            value_projection_dim=24,
            target_sequence_length=3,
            source_sequence_length=5,
        )

        attention_mask = create_attention_mask(c)
        unbatched_attention_mask = attention_mask[0].unsqueeze(0)
        batched_attention_mask = attention_mask

        attention_mask_options = {
            "none": None,
            "batched": batched_attention_mask,
            "unbatched": unbatched_attention_mask,
        }

        for mask_name, mask_option in attention_mask_options.items():
            message = f"Testing configuration: mask_option: {mask_name}"
            with self.subTest(i=message):
                projector = IndependentProjector(c)
                m = IndependentProcessor(c, projector, AttentionReshaper(c))
                output_attention_mask = m._IndependentProcessor__prepare_attention_mask(
                    mask_option
                )

                if mask_name == "none":
                    self.assertIsNone(output_attention_mask)
                elif mask_name == "batched":
                    expected_shape = (
                        m.batch_size,
                        m.num_heads,
                        m.target_sequence_length,
                        c.source_sequence_length,
                    )
                    self.assertEqual(output_attention_mask.shape, expected_shape)
                elif mask_name == "unbatched":
                    expected_shape = (
                        1,
                        1,
                        m.target_sequence_length,
                        c.source_sequence_length,
                    )
                    self.assertEqual(output_attention_mask.shape, expected_shape)

    def test__compute_weighted_values(self):
        c = build_attention_config(
            config_class=IndependentAttentionConfig,
            embedding_dim=12,
            query_key_projection_dim=20,
            value_projection_dim=16,
            source_sequence_length=24,
            target_sequence_length=16,
        )
        attention_mask_options = [None, create_attention_mask(c, True)]
        boolean_options = [True, False]

        for causal_attention_mask_flag in boolean_options:
            for attention_mask in attention_mask_options:
                mask_name = "none" if attention_mask is None else "batched"
                with self.subTest(
                    causal_attention_mask_flag=causal_attention_mask_flag,
                    mask_option=mask_name,
                ):
                    c.causal_attention_mask_flag = causal_attention_mask_flag
                    projector = IndependentProjector(c)
                    m = IndependentProcessor(c, projector, AttentionReshaper(c))

                    query = torch.randn(
                        c.batch_size,
                        c.num_heads,
                        c.target_sequence_length,
                        m.query_key_projection_dim // c.num_heads,
                    )
                    key = value = torch.randn(
                        c.batch_size,
                        c.num_heads,
                        c.source_sequence_length,
                        m.query_key_projection_dim // c.num_heads,
                    )
                    value = torch.randn(
                        c.batch_size,
                        c.num_heads,
                        c.source_sequence_length,
                        m.value_projection_dim // c.num_heads,
                    )

                    if causal_attention_mask_flag:
                        attention_mask = None

                    weighted_values = m._IndependentProcessor__compute_weighted_values(
                        query, key, value, attention_mask
                    )

                    expected_shape = (
                        m.batch_size * m.target_sequence_length,
                        m.value_projection_dim,
                    )
                    self.assertIsInstance(weighted_values, torch.Tensor)
                    self.assertEqual(weighted_values.shape, expected_shape)

    def test_compute_attention(self):
        c = build_attention_config(
            config_class=IndependentAttentionConfig,
            embedding_dim=12,
            query_key_projection_dim=20,
            value_projection_dim=16,
            source_sequence_length=16,
            target_sequence_length=24,
        )
        head_dim = c.embedding_dim // c.num_heads
        query = torch.randn(
            c.batch_size * c.num_heads, c.source_sequence_length, head_dim
        )
        key = value = torch.randn(
            c.batch_size * c.num_heads, c.target_sequence_length, head_dim
        )

        attention_mask_options = [None, create_attention_mask(c)]

        boolean_options = [True, False]

        for causal_attention_mask_flag in boolean_options:
            for attention_mask in attention_mask_options:
                mask_name = "none" if attention_mask is None else "batched"
                with self.subTest(
                    causal_attention_mask_flag=causal_attention_mask_flag,
                    mask_option=mask_name,
                ):
                    c.causal_attention_mask_flag = causal_attention_mask_flag
                    projector = IndependentProjector(c)
                    m = IndependentProcessor(c, projector, AttentionReshaper(c))

                    query = torch.randn(
                        c.batch_size * c.num_heads,
                        c.target_sequence_length,
                        m.query_key_projection_dim // c.num_heads,
                    )
                    key = value = torch.randn(
                        c.batch_size * c.num_heads,
                        c.source_sequence_length,
                        m.query_key_projection_dim // c.num_heads,
                    )
                    value = torch.randn(
                        c.batch_size * c.num_heads,
                        c.source_sequence_length,
                        m.value_projection_dim // c.num_heads,
                    )

                    output_attention_output, _ = m.compute_attention(
                        QKV(query=query, key=key, value=value),
                        attention_mask,
                    )

                    expected_shape = (
                        m.target_sequence_length,
                        m.batch_size,
                        m.embedding_dim,
                    )

                    self.assertIsInstance(output_attention_output, torch.Tensor)
                    self.assertIsNone(_)
                    self.assertEqual(output_attention_output.shape, expected_shape)


class TestMixtureOfAttentionHeadsProcessor(unittest.TestCase):
    def test_compute_attention_forwards_the_supplied_mask(self):
        cfg = build_attention_config(
            config_class=MixtureOfAttentionHeadsConfig,
            batch_size=1,
            num_heads=1,
            embedding_dim=2,
            query_key_projection_dim=2,
            value_projection_dim=2,
            target_sequence_length=2,
            source_sequence_length=2,
            experts_top_k=1,
            experts_num_experts=2,
        )
        model = MixtureOfAttentionHeadsProcessor(
            cfg,
            MixtureOfAttentionHeadsProjector(cfg),
            MixtureOfAttentionHeadsReshaper(cfg),
        ).eval()
        tensor = torch.zeros(1, 2, 2)
        mask = torch.tensor([[[0.0, -torch.inf], [0.0, -torch.inf]]])
        method_name = (
            "_MixtureOfAttentionHeadsProcessor__compute_masked_attention_weights"
        )

        with (
            patch.object(
                model,
                method_name,
                wraps=getattr(model, method_name),
            ) as compute_weights,
            patch.object(
                model,
                "_compute_attention_output",
                return_value=torch.zeros(2, 1, 2),
            ),
        ):
            model.compute_attention(
                QKV(query=tensor, key=tensor, value=tensor),
                mask,
            )

        self.assertIs(compute_weights.call_args.args[2], mask)

    def test_mask_is_applied_before_softmax(self):
        cfg = build_attention_config(
            config_class=MixtureOfAttentionHeadsConfig,
            batch_size=1,
            num_heads=1,
            embedding_dim=2,
            query_key_projection_dim=2,
            value_projection_dim=2,
            target_sequence_length=2,
            source_sequence_length=2,
            dropout_probability=0.0,
            experts_top_k=1,
            experts_num_experts=2,
        )
        model = MixtureOfAttentionHeadsProcessor(
            cfg,
            MixtureOfAttentionHeadsProjector(cfg),
            MixtureOfAttentionHeadsReshaper(cfg),
        ).eval()
        query = torch.zeros(1, 1, 1, 2, 2)
        key = torch.zeros(1, 1, 2, 2)
        mask = torch.tensor([[[0.0, -torch.inf], [0.0, -torch.inf]]])

        actual = (
            model._MixtureOfAttentionHeadsProcessor__compute_masked_attention_weights(
                query,
                key,
                mask,
            )
        )

        expected = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])
        torch.testing.assert_close(actual, expected)

    def test_training_dropout_uses_the_configured_probability(self):
        cfg = build_attention_config(
            config_class=MixtureOfAttentionHeadsConfig,
            batch_size=1,
            num_heads=1,
            embedding_dim=2,
            query_key_projection_dim=2,
            value_projection_dim=2,
            target_sequence_length=2,
            source_sequence_length=2,
            dropout_probability=0.25,
            experts_top_k=1,
            experts_num_experts=2,
        )
        model = MixtureOfAttentionHeadsProcessor(
            cfg,
            MixtureOfAttentionHeadsProjector(cfg),
            MixtureOfAttentionHeadsReshaper(cfg),
        ).train()
        query = torch.zeros(1, 1, 1, 2, 2)
        key = torch.zeros(1, 1, 2, 2)

        torch.manual_seed(47)
        actual = (
            model._MixtureOfAttentionHeadsProcessor__compute_masked_attention_weights(
                query,
                key,
                None,
            )
        )
        torch.manual_seed(47)
        expected = torch.nn.functional.dropout(
            torch.full((1, 2, 2), 0.5),
            p=0.25,
            training=True,
        )

        torch.testing.assert_close(actual, expected)

    def test_relative_logits_receive_scaled_query_contract_and_are_added(self):
        cfg = build_attention_config(
            config_class=MixtureOfAttentionHeadsConfig,
            batch_size=1,
            num_heads=1,
            embedding_dim=2,
            query_key_projection_dim=2,
            value_projection_dim=2,
            target_sequence_length=2,
            source_sequence_length=3,
            experts_top_k=1,
            experts_num_experts=2,
        )
        model = MixtureOfAttentionHeadsProcessor(
            cfg,
            MixtureOfAttentionHeadsProjector(cfg),
            MixtureOfAttentionHeadsReshaper(cfg),
        )
        query = torch.arange(4.0).view(1, 1, 1, 2, 2)
        weights = torch.arange(6.0).view(1, 1, 1, 2, 3)
        positional = torch.full_like(weights, 1.25)
        runtime_shape = object()
        method_name = (
            "_MixtureOfAttentionHeadsProcessor__maybe_add_relative_positional_embedding"
        )
        add_relative = getattr(model, method_name)

        with patch.object(
            model,
            "_compute_relative_position_logits",
            return_value=positional,
        ) as compute_relative:
            actual = add_relative(
                query,
                weights,
                runtime_shape,
            )

        args = compute_relative.call_args.args
        self.assertIs(args[0], query)
        self.assertEqual(args[1], 3)
        self.assertIs(args[2], runtime_shape)
        self.assertIs(compute_relative.call_args.kwargs["query_is_scaled"], True)
        torch.testing.assert_close(actual, weights + positional)

    def test_seeded_dropout_changes_training_weights_only(self):
        cfg = build_attention_config(
            config_class=MixtureOfAttentionHeadsConfig,
            batch_size=1,
            num_heads=1,
            embedding_dim=2,
            target_sequence_length=3,
            source_sequence_length=3,
            dropout_probability=0.5,
            experts_top_k=1,
            experts_num_experts=2,
        )
        projector = MixtureOfAttentionHeadsProjector(cfg)
        model = MixtureOfAttentionHeadsProcessor(
            cfg,
            projector,
            MixtureOfAttentionHeadsReshaper(cfg),
        )
        query = torch.zeros(1, 1, 1, 3, 2)
        key = torch.zeros(1, 1, 3, 2)
        compute_weights = (
            model._MixtureOfAttentionHeadsProcessor__compute_masked_attention_weights
        )

        model.train()
        torch.manual_seed(11)
        first = compute_weights(
            query,
            key,
            None,
        )
        torch.manual_seed(11)
        second = compute_weights(
            query,
            key,
            None,
        )
        model.eval()
        evaluation = compute_weights(
            query,
            key,
            None,
        )

        torch.testing.assert_close(first, second)
        self.assertFalse(torch.equal(first, evaluation))
        torch.testing.assert_close(
            evaluation,
            torch.full_like(evaluation, 1 / 3),
        )

    def test_init(self):
        attention_options = [PROJECTION_KINDS]

        for attention_option in attention_options:
            for model_type in attention_option:
                message = f"Testing configuration: model_type: {model_type}"
                with self.subTest(i=message):
                    c = build_attention_config(
                        config_class=MixtureOfAttentionHeadsConfig,
                        projection_kind=model_type,
                        embedding_dim=12,
                        query_key_projection_dim=20,
                        value_projection_dim=16,
                        source_sequence_length=8,
                        target_sequence_length=12,
                    )
                    projector = MixtureOfAttentionHeadsProjector(c)
                    m = MixtureOfAttentionHeadsProcessor(
                        c, projector, MixtureOfAttentionHeadsReshaper(c)
                    )
                    self.assertIsInstance(m, MixtureOfAttentionHeadsProcessor)
                    self.assertIsInstance(m.projector, MixtureOfAttentionHeadsProjector)
                    self.assertIsNone(m.relative_positional_embedding)

    def test__scale_query(self):
        c = build_attention_config(
            config_class=MixtureOfAttentionHeadsConfig,
            embedding_dim=12,
            query_key_projection_dim=20,
            value_projection_dim=16,
            source_sequence_length=8,
            target_sequence_length=12,
        )
        head_dim = c.embedding_dim // c.num_heads
        query = torch.randn(
            c.batch_size * c.num_heads, c.target_sequence_length, head_dim
        )

        projector = MixtureOfAttentionHeadsProjector(c)
        m = MixtureOfAttentionHeadsProcessor(
            c, projector, MixtureOfAttentionHeadsReshaper(c)
        )
        scaled_query_tensor = m._MixtureOfAttentionHeadsProcessor__scale_query(query)

        expected_result = query * math.sqrt(1.0 / float(head_dim))
        self.assertIsInstance(scaled_query_tensor, torch.Tensor)
        self.assertEqual(query.shape, scaled_query_tensor.shape)
        self.assertTrue(
            torch.allclose(scaled_query_tensor, expected_result, atol=1e-6, rtol=1e-5)
        )

    def test__compute_raw_masked_attention_weights(self):
        top_k = 3
        source_sequence_length = 8
        target_sequence_length = 8
        embedding_dim = 12
        query_key_projection_dim = 12
        value_projection_dim = 12
        return_attention_weights_flag = True
        c = build_attention_config(
            config_class=MixtureOfAttentionHeadsConfig,
            source_sequence_length=source_sequence_length,
            target_sequence_length=target_sequence_length,
            embedding_dim=embedding_dim,
            query_key_projection_dim=query_key_projection_dim,
            value_projection_dim=value_projection_dim,
            return_attention_weights_flag=return_attention_weights_flag,
        )
        head_dim = c.embedding_dim // c.num_heads
        query = torch.randn(
            c.batch_size,
            top_k,
            c.num_heads,
            c.target_sequence_length,
            head_dim,
        )

        attention_mask = create_attention_mask(c).repeat(top_k, 1, 1)
        attention_mask_options = [None, attention_mask]
        boolean_options = [True, False]

        for relative_case, relative_config_cls in RELATIVE_POSITIONAL_EMBEDDING_CASES:
            for use_kv_expert_models_flag in boolean_options:
                for attention_mask_option in attention_mask_options:
                    message = (
                        "Testing configuration: "
                        f"relative_positional_embedding: {relative_case}, "
                        f"attention_mask_option: {type(attention_mask_option)}, "
                        f"use_kv_expert_models_flag: {use_kv_expert_models_flag}"
                    )
                    with self.subTest(i=message):
                        c = build_attention_config(
                            config_class=MixtureOfAttentionHeadsConfig,
                            source_sequence_length=source_sequence_length,
                            target_sequence_length=target_sequence_length,
                            embedding_dim=embedding_dim,
                            query_key_projection_dim=query_key_projection_dim,
                            value_projection_dim=value_projection_dim,
                            return_attention_weights_flag=return_attention_weights_flag,
                            relative_positional_embedding_config_cls=relative_config_cls,
                            use_kv_expert_models_flag=use_kv_expert_models_flag,
                        )
                        key_shape = (
                            c.batch_size,
                            c.num_heads,
                            c.source_sequence_length,
                            head_dim,
                        )
                        if use_kv_expert_models_flag:
                            key_shape = (
                                c.batch_size,
                                top_k,
                                c.num_heads,
                                c.source_sequence_length,
                                head_dim,
                            )
                        key = torch.randn(key_shape)

                        projector = MixtureOfAttentionHeadsProjector(c)
                        m = MixtureOfAttentionHeadsProcessor(
                            c, projector, MixtureOfAttentionHeadsReshaper(c)
                        )
                        method_name = (
                            "_MixtureOfAttentionHeadsProcessor"
                            "__compute_raw_masked_attention_weights"
                        )
                        compute_raw_masked_weights = getattr(m, method_name)
                        raw_masked_weights = compute_raw_masked_weights(
                            query, key, attention_mask_option
                        )

                        expected_shape = (
                            c.batch_size * c.num_heads * top_k,
                            c.target_sequence_length,
                            c.source_sequence_length,
                        )
                        self.assertIsInstance(raw_masked_weights, torch.Tensor)
                        self.assertEqual(raw_masked_weights.shape, expected_shape)

    def test__compute_masked_attention_weights(self):
        top_k = 3
        c = build_attention_config(
            config_class=MixtureOfAttentionHeadsConfig,
            embedding_dim=12,
            query_key_projection_dim=20,
            value_projection_dim=16,
            source_sequence_length=8,
            target_sequence_length=12,
            experts_top_k=top_k,
        )
        projector = MixtureOfAttentionHeadsProjector(c)
        m = MixtureOfAttentionHeadsProcessor(
            c, projector, MixtureOfAttentionHeadsReshaper(c)
        )
        head_dim = c.embedding_dim // c.num_heads

        query = torch.randn(
            c.batch_size, top_k, c.num_heads, c.target_sequence_length, head_dim
        )
        attention_mask_options = [None, create_attention_mask(c).repeat(top_k, 1, 1)]

        boolean_options = [True, False]
        for use_kv_expert_models_flag in boolean_options:
            for attention_mask_option in attention_mask_options:
                with self.subTest(
                    attention_mask_type=type(attention_mask_option).__name__,
                    use_kv_expert_models_flag=use_kv_expert_models_flag,
                ):
                    c.use_kv_expert_models_flag = use_kv_expert_models_flag
                    projector = MixtureOfAttentionHeadsProjector(c)
                    m = MixtureOfAttentionHeadsProcessor(
                        c, projector, MixtureOfAttentionHeadsReshaper(c)
                    )
                    key_shape = (
                        c.batch_size,
                        c.num_heads,
                        c.source_sequence_length,
                        head_dim,
                    )
                    if use_kv_expert_models_flag:
                        key_shape = (
                            c.batch_size,
                            top_k,
                            c.num_heads,
                            c.source_sequence_length,
                            head_dim,
                        )

                    key = torch.randn(key_shape)
                    method_name = (
                        "_MixtureOfAttentionHeadsProcessor"
                        "__compute_masked_attention_weights"
                    )
                    compute_masked_weights = getattr(m, method_name)
                    result = compute_masked_weights(query, key, attention_mask_option)

                    expected_shape = (
                        c.batch_size * top_k * c.num_heads,
                        c.target_sequence_length,
                        c.source_sequence_length,
                    )
                    self.assertEqual(result.shape, expected_shape)

    def test__compute_weighted_values(self):
        top_k = 3
        boolean_options = [True, False]
        for use_kv_expert_models_flag in boolean_options:
            with self.subTest(
                use_kv_expert_models_flag=use_kv_expert_models_flag,
            ):
                c = build_attention_config(
                    config_class=MixtureOfAttentionHeadsConfig,
                    embedding_dim=12,
                    query_key_projection_dim=20,
                    value_projection_dim=16,
                    source_sequence_length=8,
                    target_sequence_length=12,
                    experts_top_k=top_k,
                    use_kv_expert_models_flag=use_kv_expert_models_flag,
                )

                projector = MixtureOfAttentionHeadsProjector(c)
                m = MixtureOfAttentionHeadsProcessor(
                    c, projector, MixtureOfAttentionHeadsReshaper(c)
                )

                attention_weights = torch.randn(
                    c.batch_size,
                    top_k,
                    c.num_heads,
                    c.target_sequence_length,
                    c.source_sequence_length,
                )
                value_shape = (
                    c.batch_size,
                    c.num_heads,
                    c.source_sequence_length,
                    m.value_projection_dim // c.num_heads,
                )
                if use_kv_expert_models_flag:
                    value_shape = (
                        c.batch_size,
                        top_k,
                        c.num_heads,
                        c.source_sequence_length,
                        m.value_projection_dim // c.num_heads,
                    )

                values = torch.randn(value_shape)
                weighted_values = (
                    m._MixtureOfAttentionHeadsProcessor__compute_weighted_values(
                        attention_weights, values
                    )
                )

                expected_shape = (
                    c.target_sequence_length,
                    c.batch_size,
                    top_k,
                    c.value_projection_dim,
                )
                self.assertEqual(weighted_values.shape, expected_shape)

    def test__compute_attention_output(self):
        top_k = 3
        c = build_attention_config(
            config_class=MixtureOfAttentionHeadsConfig,
            embedding_dim=12,
            query_key_projection_dim=20,
            value_projection_dim=16,
            source_sequence_length=8,
            target_sequence_length=12,
            experts_top_k=top_k,
        )
        projector = MixtureOfAttentionHeadsProjector(c)
        reshaper = MixtureOfAttentionHeadsReshaper(c)
        m = MixtureOfAttentionHeadsProcessor(
            c, projector, MixtureOfAttentionHeadsReshaper(c)
        )

        tensor = torch.randn(
            c.target_sequence_length,
            c.batch_size,
            c.embedding_dim,
        )

        projections = projector.compute_qkv_projections(
            QKV(query=tensor, key=tensor, value=tensor)
        )
        projections = reshaper.reshape_qkv_for_attention(projections)

        weighted_values, _ = m.compute_attention(projections)

        expected_shape = (c.target_sequence_length, c.batch_size, c.embedding_dim)
        self.assertEqual(weighted_values.shape, expected_shape)


class TestProcessorDispatch(unittest.TestCase):
    def test_leaf_runtime_builds_expected_processor(self):
        expected_map = {
            SelfAttentionConfig: SelfAttentionProcessor,
            IndependentAttentionConfig: IndependentProcessor,
            MixtureOfAttentionHeadsConfig: (MixtureOfAttentionHeadsProcessor),
        }
        for config_class, processor_cls in expected_map.items():
            with self.subTest(config_class=config_class):
                c = build_attention_config(
                    config_class=config_class,
                    source_sequence_length=8,
                    target_sequence_length=8,
                    embedding_dim=12,
                    query_key_projection_dim=12,
                    value_projection_dim=12,
                )
                m = c.build()
                self.assertIsInstance(m.processor, processor_cls)
                self.assertIsNone(m.processor.relative_positional_embedding)

    def test_compute_attention__self_attention(self):
        c = build_attention_config(
            config_class=SelfAttentionConfig,
            source_sequence_length=32,
            target_sequence_length=32,
            embedding_dim=12,
            query_key_projection_dim=12,
            value_projection_dim=12,
            return_attention_weights_flag=True,
        )

        head_dim = c.embedding_dim // c.num_heads
        projector = SelfAttentionProjector(c)
        m = SelfAttentionProcessor(c, projector, AttentionReshaper(c))

        query = torch.randn(
            c.batch_size * c.num_heads, c.source_sequence_length, head_dim
        )
        key = torch.randn(
            c.batch_size * c.num_heads, c.target_sequence_length, head_dim
        )
        value = torch.randn(
            c.batch_size * c.num_heads, c.target_sequence_length, head_dim
        )
        attention_mask = create_attention_mask(c)

        output_attention_output, output_attention_weights = m.compute_attention(
            QKV(query=query, key=key, value=value),
            attention_mask,
        )

        expected_output_shape = (
            c.target_sequence_length,
            c.batch_size,
            c.embedding_dim,
        )
        expected_weights_shape = (
            c.batch_size,
            c.num_heads,
            c.target_sequence_length,
            c.source_sequence_length,
        )
        self.assertIsInstance(output_attention_output, torch.Tensor)
        self.assertIsInstance(output_attention_weights, torch.Tensor)
        self.assertEqual(output_attention_output.shape, expected_output_shape)
        self.assertEqual(output_attention_weights.shape, expected_weights_shape)

    def test_compute_attention__independent(self):
        c = build_attention_config(
            config_class=IndependentAttentionConfig,
            embedding_dim=12,
            query_key_projection_dim=0,
            value_projection_dim=0,
            source_sequence_length=32,
            target_sequence_length=32,
        )
        head_dim = c.embedding_dim // c.num_heads
        projector = IndependentProjector(c)
        m = IndependentProcessor(c, projector, AttentionReshaper(c))

        query = torch.randn(
            c.batch_size * c.num_heads, c.source_sequence_length, head_dim
        )
        key = value = torch.randn(
            c.batch_size * c.num_heads, c.target_sequence_length, head_dim
        )
        attention_mask = create_attention_mask(c)

        output_attention_output, output_attention_weights = m.compute_attention(
            QKV(query=query, key=key, value=value),
            attention_mask,
        )

        expected_output_shape = (
            c.target_sequence_length,
            c.batch_size,
            c.embedding_dim,
        )
        self.assertIsInstance(output_attention_output, torch.Tensor)
        self.assertIsNone(output_attention_weights)
        self.assertEqual(output_attention_output.shape, expected_output_shape)

    def test_compute_attention__mixture_of_attention_heads(self):
        top_k = 3
        c = build_attention_config(
            config_class=MixtureOfAttentionHeadsConfig,
            source_sequence_length=12,
            target_sequence_length=12,
            embedding_dim=16,
            query_key_projection_dim=16,
            value_projection_dim=16,
            experts_top_k=top_k,
        )
        projector = MixtureOfAttentionHeadsProjector(c)
        m = MixtureOfAttentionHeadsProcessor(
            c, projector, MixtureOfAttentionHeadsReshaper(c)
        )

        tensor = torch.randn(
            c.target_sequence_length,
            c.batch_size,
            c.embedding_dim,
        )

        projections = projector.compute_qkv_projections(
            QKV(query=tensor, key=tensor, value=tensor)
        )

        output_attention_output, output_attention_weights = m.compute_attention(
            projections
        )

        expected_output_shape = (
            c.target_sequence_length,
            c.batch_size,
            c.embedding_dim,
        )
        self.assertIsInstance(output_attention_output, torch.Tensor)
        self.assertIsNone(output_attention_weights)
        self.assertEqual(output_attention_output.shape, expected_output_shape)
