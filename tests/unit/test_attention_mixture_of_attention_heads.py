import unittest
from pathlib import Path

import torch
from emperor.attention import MixtureOfAttentionHeadsConfig
from emperor.attention.core.runtime import QKV, AttentionMasks, AttentionRuntimeShape
from emperor.attention.core.variants.mixture_of_attention_heads.bias import (
    MixtureOfAttentionHeadsKeyValueBias,
)
from emperor.attention.core.variants.mixture_of_attention_heads.reshaper import (
    MixtureOfAttentionHeadsReshaper,
)
from emperor.attention.core.variants.mixture_of_attention_heads.zero_attention import (
    MixtureOfAttentionHeadsZeroAttention,
)
from emperor.embedding.relative.core.config import DynamicPositionalBiasConfig

from support.attention import build_attention_config


class SummingRelativeEmbedding(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.received_query = None
        self.received_source_length = None

    def forward(self, query, sequence_length, last=False):
        self.received_query = query
        self.received_source_length = sequence_length
        return query.sum(dim=-1, keepdim=True).expand(
            *query.shape[:-1],
            sequence_length,
        )


class TestMixtureOfAttentionHeadsExpertKeyValue(unittest.TestCase):
    def preset(self, **overrides):
        values = {
            "config_class": MixtureOfAttentionHeadsConfig,
            "batch_size": 2,
            "num_heads": 2,
            "embedding_dim": 8,
            "query_key_projection_dim": 8,
            "value_projection_dim": 8,
            "target_sequence_length": 4,
            "source_sequence_length": 4,
            "use_kv_expert_models_flag": True,
            "experts_top_k": 2,
            "experts_num_experts": 4,
            "experts_compute_expert_mixture_flag": False,
            "experts_stack_num_layers": 1,
        }
        values.update(overrides)
        return build_attention_config(**values)

    def assert_has_nonzero_parameter_gradient(self, model, label):
        gradients = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.requires_grad
        ]
        self.assertTrue(
            any(
                gradient is not None and torch.any(gradient.abs() > 0)
                for gradient in gradients
            ),
            f"Expected a non-zero gradient through {label}.",
        )

    def input_tensor(self, cfg, *, requires_grad=False):
        return torch.randn(
            cfg.target_sequence_length,
            cfg.batch_size,
            cfg.embedding_dim,
            requires_grad=requires_grad,
        )

    def runtime_shape(self, cfg):
        return AttentionRuntimeShape(
            batch_size=cfg.batch_size,
            target_sequence_length=cfg.target_sequence_length,
            source_sequence_length=cfg.source_sequence_length,
        )

    def test_self_attention_forward(self):
        torch.manual_seed(0)
        cfg = self.preset()
        model = cfg.build()
        inputs = torch.randn(
            cfg.target_sequence_length,
            cfg.batch_size,
            cfg.embedding_dim,
        )

        output, attention_weights, auxiliary_loss = model(inputs, inputs, inputs)

        self.assertEqual(
            output.shape,
            (
                cfg.target_sequence_length,
                cfg.batch_size,
                cfg.embedding_dim,
            ),
        )
        self.assertIsNone(attention_weights)
        self.assertIsInstance(auxiliary_loss, torch.Tensor)

    def test_build_selects_mixture_specific_handlers(self):
        model = self.preset().build()

        self.assertIsInstance(model.bias, MixtureOfAttentionHeadsKeyValueBias)
        self.assertIsInstance(
            model.zero_attention,
            MixtureOfAttentionHeadsZeroAttention,
        )

    def test_key_value_bias_supports_shared_and_expert_projection_ranks(self):
        for use_kv_expert_models_flag in (False, True):
            for top_k in (1, 2):
                with self.subTest(
                    use_kv_expert_models_flag=use_kv_expert_models_flag,
                    top_k=top_k,
                ):
                    cfg = self.preset(
                        batch_size=3,
                        value_projection_dim=12,
                        add_key_value_bias_flag=True,
                        use_kv_expert_models_flag=use_kv_expert_models_flag,
                        experts_top_k=top_k,
                    )
                    model = MixtureOfAttentionHeadsKeyValueBias(cfg)
                    key_head_dim = cfg.query_key_projection_dim // cfg.num_heads
                    value_head_dim = cfg.value_projection_dim // cfg.num_heads
                    runtime_batch_size = 2
                    branch_multiplier = top_k if use_kv_expert_models_flag else 1
                    branch_count = (
                        runtime_batch_size * branch_multiplier * cfg.num_heads
                    )
                    key = torch.randn(branch_count, 4, key_head_dim)
                    value = torch.randn(branch_count, 4, value_head_dim)
                    expected_key_bias = (
                        model.key_bias_vector.view(
                            1,
                            1,
                            cfg.num_heads,
                            key_head_dim,
                        )
                        .expand(
                            runtime_batch_size,
                            branch_multiplier,
                            -1,
                            -1,
                        )
                        .reshape(branch_count, key_head_dim)
                    )
                    expected_value_bias = (
                        model.value_bias_vector.view(
                            1,
                            1,
                            cfg.num_heads,
                            value_head_dim,
                        )
                        .expand(
                            runtime_batch_size,
                            branch_multiplier,
                            -1,
                            -1,
                        )
                        .reshape(branch_count, value_head_dim)
                    )
                    attention_branch_count = (
                        runtime_batch_size * top_k * cfg.num_heads
                    )
                    key_padding_mask = torch.zeros(
                        runtime_batch_size,
                        4,
                        dtype=torch.bool,
                    )
                    attention_mask = torch.zeros(
                        attention_branch_count,
                        4,
                        4,
                        dtype=torch.bool,
                    )

                    runtime_shape = AttentionRuntimeShape(
                        batch_size=runtime_batch_size,
                        target_sequence_length=4,
                        source_sequence_length=4,
                    )
                    output_qkv, output_masks, output_runtime_shape = (
                        model.add_kv_learnable_bias_vectors(
                            QKV(query=key, key=key, value=value),
                            AttentionMasks(
                                key_padding_mask=key_padding_mask,
                                attention_mask=attention_mask,
                            ),
                            runtime_shape,
                        )
                    )

                    self.assertEqual(
                        output_qkv.key.shape,
                        (branch_count, 5, key_head_dim),
                    )
                    self.assertEqual(
                        output_qkv.value.shape,
                        (branch_count, 5, value_head_dim),
                    )
                    self.assertEqual(
                        output_masks.key_padding_mask.shape,
                        (runtime_batch_size, 5),
                    )
                    self.assertEqual(
                        output_masks.attention_mask.shape,
                        (attention_branch_count, 4, 5),
                    )
                    torch.testing.assert_close(
                        output_qkv.key[:, -1],
                        expected_key_bias,
                    )
                    torch.testing.assert_close(
                        output_qkv.value[:, -1],
                        expected_value_bias,
                    )
                    self.assertEqual(
                        output_runtime_shape.source_sequence_length,
                        5,
                    )
                    self.assertEqual(output_runtime_shape.source_extension_count, 1)

                    bias_loss = (
                        output_qkv.key[:, -1].sum()
                        + output_qkv.value[:, -1].sum()
                    )
                    bias_loss.backward()
                    self.assertTrue(torch.any(model.key_bias_vector.grad != 0))
                    self.assertTrue(torch.any(model.value_bias_vector.grad != 0))

    def test_key_value_bias_rejects_nonmatching_standard_and_expert_branches(self):
        runtime_shape = AttentionRuntimeShape(
            batch_size=2,
            target_sequence_length=4,
            source_sequence_length=4,
        )
        for use_kv_expert_models_flag, branch_count, expected_message in (
            (
                False,
                8,
                "Attention-ready key/value projections must have a leading "
                "dimension equal to batch_size * num_heads (4), got 8.",
            ),
            (
                True,
                4,
                "Mixture attention-ready key/value projections must have a leading "
                "dimension equal to batch_size * top_k * num_heads (8), got 4.",
            ),
            (
                True,
                16,
                "Mixture attention-ready key/value projections must have a leading "
                "dimension equal to batch_size * top_k * num_heads (8), got 16.",
            ),
        ):
            with self.subTest(
                use_kv_expert_models_flag=use_kv_expert_models_flag,
                branch_count=branch_count,
            ):
                cfg = self.preset(
                    add_key_value_bias_flag=True,
                    use_kv_expert_models_flag=use_kv_expert_models_flag,
                )
                model = MixtureOfAttentionHeadsKeyValueBias(cfg)
                projection = torch.zeros(branch_count, 4, 4)

                with self.assertRaises(RuntimeError) as caught:
                    model.add_kv_learnable_bias_vectors(
                        QKV(query=projection, key=projection, value=projection),
                        AttentionMasks(),
                        runtime_shape,
                    )

                self.assertEqual(str(caught.exception), expected_message)

    def test_zero_attention_supports_shared_and_expert_projection_branches(self):
        for use_kv_expert_models_flag in (False, True):
            with self.subTest(
                use_kv_expert_models_flag=use_kv_expert_models_flag,
            ):
                cfg = self.preset(
                    value_projection_dim=12,
                    zero_attention_flag=True,
                    use_kv_expert_models_flag=use_kv_expert_models_flag,
                )
                model = MixtureOfAttentionHeadsZeroAttention(cfg)
                branch_count = cfg.batch_size * cfg.num_heads
                if use_kv_expert_models_flag:
                    branch_count *= cfg.experts_config.top_k
                attention_branch_count = (
                    cfg.batch_size * cfg.experts_config.top_k * cfg.num_heads
                )
                key = torch.randn(branch_count, 4, 4)
                value = torch.randn(branch_count, 4, 6)
                key_padding_mask = torch.zeros(2, 4, dtype=torch.bool)
                attention_mask = torch.zeros(
                    attention_branch_count,
                    4,
                    4,
                    dtype=torch.bool,
                )

                runtime_shape = AttentionRuntimeShape(
                    batch_size=cfg.batch_size,
                    target_sequence_length=4,
                    source_sequence_length=4,
                )
                output_qkv, output_masks, output_runtime_shape = (
                    model.add_zero_attention(
                        QKV(query=key, key=key, value=value),
                        AttentionMasks(
                            key_padding_mask=key_padding_mask,
                            attention_mask=attention_mask,
                        ),
                        runtime_shape,
                    )
                )

                self.assertEqual(output_qkv.key.shape, (branch_count, 5, 4))
                self.assertEqual(output_qkv.value.shape, (branch_count, 5, 6))
                self.assertEqual(output_masks.key_padding_mask.shape, (2, 5))
                self.assertEqual(
                    output_masks.attention_mask.shape,
                    (attention_branch_count, 4, 5),
                )
                torch.testing.assert_close(
                    output_qkv.key[:, -1],
                    torch.zeros_like(key[:, 0]),
                )
                torch.testing.assert_close(
                    output_qkv.value[:, -1],
                    torch.zeros_like(value[:, 0]),
                )
                self.assertEqual(output_runtime_shape.source_sequence_length, 5)
                self.assertEqual(output_runtime_shape.source_extension_count, 1)

    def test_shared_and_expert_key_value_top_k_matrix_forward_backward(self):
        for use_kv_expert_models_flag in (False, True):
            for top_k in (1, 2):
                with self.subTest(
                    use_kv_expert_models_flag=use_kv_expert_models_flag,
                    top_k=top_k,
                ):
                    torch.manual_seed(100 + top_k + use_kv_expert_models_flag)
                    cfg = self.preset(
                        use_kv_expert_models_flag=use_kv_expert_models_flag,
                        experts_top_k=top_k,
                        experts_num_experts=4,
                    )
                    model = cfg.build()
                    inputs = self.input_tensor(cfg, requires_grad=True)

                    output, attention_weights, auxiliary_loss = model(
                        inputs,
                        inputs,
                        inputs,
                    )

                    self.assertEqual(output.shape, inputs.shape)
                    self.assertTrue(torch.isfinite(output).all())
                    self.assertIsNone(attention_weights)
                    self.assertIsInstance(auxiliary_loss, torch.Tensor)
                    self.assertTrue(torch.isfinite(auxiliary_loss).all())
                    self.assertIsNone(model.projector.auxiliary_loss)

                    (output.square().mean() + auxiliary_loss).backward()

                    self.assertIsNotNone(inputs.grad)
                    self.assertTrue(torch.any(inputs.grad.abs() > 0))
                    self.assert_has_nonzero_parameter_gradient(
                        model.projector.sampler.router,
                        "router",
                    )
                    self.assert_has_nonzero_parameter_gradient(
                        model.projector.query_model,
                        "query_model",
                    )
                    self.assert_has_nonzero_parameter_gradient(
                        model.projector.output_model,
                        "output_model",
                    )
                    for label in ("key_model", "value_model"):
                        self.assert_has_nonzero_parameter_gradient(
                            getattr(model.projector, label),
                            label,
                        )

    def test_relative_positional_embedding_with_shared_and_expert_key_value(self):
        for use_kv_expert_models_flag in (False, True):
            with self.subTest(
                use_kv_expert_models_flag=use_kv_expert_models_flag,
            ):
                torch.manual_seed(200 + use_kv_expert_models_flag)
                cfg = self.preset(
                    use_kv_expert_models_flag=use_kv_expert_models_flag,
                    add_key_value_bias_flag=True,
                    zero_attention_flag=True,
                    relative_positional_embedding_config_cls=(
                        DynamicPositionalBiasConfig
                    ),
                )
                model = cfg.build()
                inputs = self.input_tensor(cfg, requires_grad=True)

                output, _, auxiliary_loss = model(inputs, inputs, inputs)

                self.assertEqual(output.shape, inputs.shape)
                self.assertTrue(torch.isfinite(output).all())
                self.assertIsNotNone(model.processor.relative_positional_embedding)
                (output.square().mean() + auxiliary_loss).backward()
                self.assert_has_nonzero_parameter_gradient(
                    model.processor.relative_positional_embedding,
                    "relative_positional_embedding",
                )

    def test_relative_position_adapter_preserves_expert_order_padding_and_gradients(
        self,
    ):
        cfg = self.preset(
            batch_size=3,
            target_sequence_length=6,
            source_sequence_length=6,
        )
        processor = cfg.build().processor
        relative = SummingRelativeEmbedding()
        processor.relative_positional_embedding = relative
        runtime_shape = AttentionRuntimeShape(
            batch_size=2,
            target_sequence_length=3,
            source_sequence_length=6,
            source_extension_count=2,
        )
        query = torch.arange(
            2 * cfg.experts_config.top_k * cfg.num_heads * 3 * 4,
            dtype=torch.float32,
        ).view(2, cfg.experts_config.top_k, cfg.num_heads, 3, 4)
        query.requires_grad_()

        logits = processor._compute_relative_position_logits(
            query,
            runtime_shape.source_sequence_length,
            runtime_shape,
        )

        expected_prepared_query = (query * (4**-0.5)).reshape(
            2 * cfg.experts_config.top_k,
            cfg.num_heads,
            3,
            4,
        )
        torch.testing.assert_close(relative.received_query, expected_prepared_query)
        self.assertEqual(relative.received_source_length, 4)
        expected_real_logits = expected_prepared_query.sum(
            dim=-1,
            keepdim=True,
        ).expand(-1, -1, -1, 4)
        expected_logits = torch.nn.functional.pad(
            expected_real_logits.view(
                2,
                cfg.experts_config.top_k,
                cfg.num_heads,
                3,
                4,
            ),
            (0, 2),
        )
        torch.testing.assert_close(logits, expected_logits)
        self.assertEqual(
            logits.shape,
            (2, cfg.experts_config.top_k, cfg.num_heads, 3, 6),
        )
        torch.testing.assert_close(logits[..., -2:], torch.zeros_like(logits[..., -2:]))

        logits.sum().backward()
        self.assertIsNotNone(query.grad)
        self.assertTrue(torch.all(query.grad != 0))

    def test_relative_position_adapter_rejects_non_mixture_layouts(self):
        cfg = self.preset()
        processor = cfg.build().processor
        processor.relative_positional_embedding = SummingRelativeEmbedding()
        invalid_cases = (
            (
                torch.zeros(cfg.batch_size, cfg.num_heads, 4, 4),
                "mixture relative-position query must be rank 5 with layout "
                "[batch, selected_expert, head, target, head_width], got rank 4.",
            ),
            (
                torch.zeros(1, cfg.experts_config.top_k, cfg.num_heads, 4, 4),
                "mixture relative-position query must have leading dimensions "
                "(batch_size, top_k, num_heads) (2, 2, 2), got (1, 2, 2).",
            ),
        )

        for query, message in invalid_cases:
            with self.subTest(shape=tuple(query.shape)):
                with self.assertRaises(RuntimeError) as caught:
                    processor._compute_relative_position_logits(query, 4)
                self.assertEqual(str(caught.exception), message)

    def test_mask_formats_normalize_in_batch_expert_head_order(self):
        cfg = self.preset()
        model = cfg.build()
        runtime_shape = self.runtime_shape(cfg)
        branch_count = cfg.batch_size * cfg.experts_config.top_k * cfg.num_heads
        key = torch.randn(
            branch_count,
            cfg.source_sequence_length,
            cfg.query_key_projection_dim // cfg.num_heads,
        )
        sequence_shape = (
            cfg.target_sequence_length,
            cfg.source_sequence_length,
        )

        mask_2d = torch.arange(
            cfg.target_sequence_length * cfg.source_sequence_length,
            dtype=cfg.target_dtype,
        ).view(sequence_shape)
        normalized_2d = model.masks.merge_padding_and_attention_mask(
            key,
            AttentionMasks(attention_mask=mask_2d),
            runtime_shape,
        )
        expected_2d = mask_2d.view(1, *sequence_shape).expand(
            branch_count,
            -1,
            -1,
        )
        torch.testing.assert_close(normalized_2d, expected_2d)

        standard_mask = torch.arange(
            cfg.batch_size
            * cfg.num_heads
            * cfg.target_sequence_length
            * cfg.source_sequence_length,
            dtype=cfg.target_dtype,
        ).view(cfg.batch_size, cfg.num_heads, *sequence_shape)
        normalized_standard = model.masks.merge_padding_and_attention_mask(
            key,
            AttentionMasks(
                attention_mask=standard_mask.reshape(-1, *sequence_shape),
            ),
            runtime_shape,
        )
        expected_standard = (
            standard_mask.unsqueeze(1)
            .expand(
                -1,
                cfg.experts_config.top_k,
                -1,
                -1,
                -1,
            )
            .reshape(branch_count, *sequence_shape)
        )
        torch.testing.assert_close(normalized_standard, expected_standard)

        expanded_mask = torch.arange(
            branch_count * cfg.target_sequence_length * cfg.source_sequence_length,
            dtype=cfg.target_dtype,
        ).view(
            cfg.batch_size,
            cfg.experts_config.top_k,
            cfg.num_heads,
            *sequence_shape,
        )
        normalized_expanded = model.masks.merge_padding_and_attention_mask(
            key,
            AttentionMasks(
                attention_mask=expanded_mask.reshape(-1, *sequence_shape),
            ),
            runtime_shape,
        )
        torch.testing.assert_close(
            normalized_expanded,
            expanded_mask.reshape(branch_count, *sequence_shape),
        )

    def test_mask_preparation_canonicalizes_boolean_and_preserves_floating_forms(self):
        cfg = self.preset()
        model = cfg.build()
        runtime_shape = self.runtime_shape(cfg)
        sequence_shape = (
            cfg.target_sequence_length,
            cfg.source_sequence_length,
        )
        standard_branch_count = cfg.batch_size * cfg.num_heads
        expert_branch_count = standard_branch_count * cfg.experts_config.top_k

        for leading_dimension in (1, standard_branch_count, expert_branch_count):
            with self.subTest(leading_dimension=leading_dimension, dtype="bool"):
                boolean_mask = torch.zeros(
                    leading_dimension,
                    *sequence_shape,
                    dtype=torch.bool,
                )
                boolean_mask[:, 0, -1] = True
                prepared = model.masks.prepare_attention_masks(
                    self.input_tensor(cfg),
                    AttentionMasks(attention_mask=boolean_mask),
                    runtime_shape,
                )
                expected = torch.zeros_like(boolean_mask, dtype=cfg.target_dtype)
                expected.masked_fill_(boolean_mask, -torch.inf)
                torch.testing.assert_close(prepared.attention_mask, expected)

            with self.subTest(leading_dimension=leading_dimension, dtype="float"):
                floating_mask = torch.randn(
                    leading_dimension,
                    *sequence_shape,
                    dtype=cfg.target_dtype,
                )
                prepared = model.masks.prepare_attention_masks(
                    self.input_tensor(cfg),
                    AttentionMasks(attention_mask=floating_mask),
                    runtime_shape,
                )
                self.assertIs(prepared.attention_mask, floating_mask)

    def test_causal_padding_bias_and_zero_masks_compose_in_mixture_branch_order(self):
        for use_kv_expert_models_flag in (False, True):
            with self.subTest(
                use_kv_expert_models_flag=use_kv_expert_models_flag,
            ):
                cfg = self.preset(
                    use_kv_expert_models_flag=use_kv_expert_models_flag,
                    causal_attention_mask_flag=True,
                    add_key_value_bias_flag=True,
                    zero_attention_flag=True,
                )
                model = cfg.build()
                runtime_shape = self.runtime_shape(cfg)
                padding_mask = torch.tensor(
                    [
                        [False, True, False, False],
                        [True, False, False, True],
                    ]
                )
                masks = model.masks.prepare_attention_masks(
                    self.input_tensor(cfg),
                    AttentionMasks(key_padding_mask=padding_mask),
                    runtime_shape,
                )
                key_value_branch_count = cfg.batch_size * cfg.num_heads
                if use_kv_expert_models_flag:
                    key_value_branch_count *= cfg.experts_config.top_k
                key = torch.zeros(key_value_branch_count, 4, 4)
                value = torch.zeros(key_value_branch_count, 4, 4)
                qkv = QKV(query=key, key=key, value=value)
                qkv, masks, runtime_shape = model.bias.add_kv_learnable_bias_vectors(
                    qkv,
                    masks,
                    runtime_shape,
                )
                qkv, masks, runtime_shape = model.zero_attention.add_zero_attention(
                    qkv,
                    masks,
                    runtime_shape,
                )

                merged = model.masks.merge_padding_and_attention_mask(
                    qkv.key,
                    masks,
                    runtime_shape,
                )

                branch_count = (
                    cfg.batch_size * cfg.experts_config.top_k * cfg.num_heads
                )
                causal = torch.triu(
                    torch.full((4, 4), -torch.inf, dtype=cfg.target_dtype),
                    diagonal=1,
                )
                causal = torch.nn.functional.pad(causal, (0, 2))
                canonical_padding = torch.zeros_like(
                    padding_mask,
                    dtype=cfg.target_dtype,
                ).masked_fill_(padding_mask, -torch.inf)
                canonical_padding = torch.nn.functional.pad(
                    canonical_padding,
                    (0, 2),
                )
                expected = causal.view(1, 1, 1, 4, 6).expand(
                    cfg.batch_size,
                    cfg.experts_config.top_k,
                    cfg.num_heads,
                    -1,
                    -1,
                )
                expected = expected + canonical_padding.view(
                    cfg.batch_size,
                    1,
                    1,
                    1,
                    6,
                )
                torch.testing.assert_close(
                    merged,
                    expected.reshape(branch_count, 4, 6),
                )

    def test_direct_mask_normalization_rejects_every_invalid_contract(self):
        cfg = self.preset()
        masks = cfg.build().masks
        runtime_shape = self.runtime_shape(cfg)
        branch_count = cfg.batch_size * cfg.experts_config.top_k * cfg.num_heads
        key = torch.randn(branch_count, cfg.source_sequence_length, 4)
        invalid_cases = (
            (
                "sequence_dimensions",
                AttentionMasks(
                    attention_mask=torch.zeros(
                        1,
                        cfg.target_sequence_length + 1,
                        cfg.source_sequence_length,
                    )
                ),
                "attention_mask must have target/source dimensions (4, 4), got (5, 4).",
            ),
            (
                "rank",
                AttentionMasks(
                    attention_mask=torch.zeros(
                        1,
                        1,
                        cfg.target_sequence_length,
                        cfg.source_sequence_length,
                    )
                ),
                "attention_mask must be 2-D or 3-D for mixture of attention heads, "
                "got 4-D.",
            ),
            (
                "leading_dimension",
                AttentionMasks(
                    attention_mask=torch.zeros(
                        3,
                        cfg.target_sequence_length,
                        cfg.source_sequence_length,
                    )
                ),
                "3-D attention_mask leading dimension must be batch_size * num_heads "
                "or batch_size * top_k * num_heads (4 or 8), got 3.",
            ),
            (
                "key_padding_mask",
                AttentionMasks(
                    key_padding_mask=torch.zeros(
                        cfg.batch_size,
                        cfg.source_sequence_length + 1,
                    )
                ),
                "key_padding_mask must have shape (2, 4), got (2, 5).",
            ),
        )

        for name, invalid_masks, message in invalid_cases:
            with self.subTest(name=name):
                with self.assertRaises(RuntimeError) as caught:
                    masks.merge_padding_and_attention_mask(
                        key,
                        invalid_masks,
                        runtime_shape,
                    )
                self.assertEqual(str(caught.exception), message)

    def test_mask_shape_validation_has_exact_rank_and_leading_contracts(self):
        cfg = self.preset()
        masks = cfg.build().masks
        runtime_shape = self.runtime_shape(cfg)
        sequence_shape = (
            cfg.target_sequence_length,
            cfg.source_sequence_length,
        )
        standard_branches = cfg.batch_size * cfg.num_heads
        expert_branches = standard_branches * cfg.experts_config.top_k

        for leading_dimension in (1, standard_branches, expert_branches):
            with self.subTest(leading_dimension=leading_dimension):
                attention_mask = torch.zeros(leading_dimension, *sequence_shape)
                prepared_masks = masks.prepare_attention_masks(
                    self.input_tensor(cfg),
                    AttentionMasks(attention_mask=attention_mask),
                    runtime_shape,
                )
                self.assertIs(prepared_masks.attention_mask, attention_mask)

        invalid_cases = (
            (
                "sequence_dimensions",
                torch.zeros(
                    1,
                    cfg.target_sequence_length + 1,
                    cfg.source_sequence_length,
                ),
                "attention_mask must have target/source dimensions (4, 4), got (5, 4).",
            ),
            (
                "leading_dimension",
                torch.zeros(3, *sequence_shape),
                "3-D attention_mask leading dimension must be 1, batch_size * "
                "num_heads, or batch_size * top_k * num_heads (1, 4, or 8), got 3.",
            ),
            (
                "rank",
                torch.zeros(1, 1, *sequence_shape),
                "attention_mask must be 2-D or 3-D for mixture of attention heads, "
                "got 4-D.",
            ),
        )

        for name, attention_mask, message in invalid_cases:
            with self.subTest(name=name):
                with self.assertRaises(RuntimeError) as caught:
                    masks.prepare_attention_masks(
                        self.input_tensor(cfg),
                        AttentionMasks(attention_mask=attention_mask),
                        runtime_shape,
                    )
                self.assertEqual(str(caught.exception), message)

    def test_singleton_leading_attention_mask_expands_to_every_branch(self):
        cfg = self.preset()
        masks = cfg.build().masks
        runtime_shape = self.runtime_shape(cfg)
        branch_count = cfg.batch_size * cfg.experts_config.top_k * cfg.num_heads
        key = torch.randn(branch_count, cfg.source_sequence_length, 4)
        attention_mask = torch.arange(
            cfg.target_sequence_length * cfg.source_sequence_length,
            dtype=cfg.target_dtype,
        ).view(1, cfg.target_sequence_length, cfg.source_sequence_length)

        normalized = masks.merge_padding_and_attention_mask(
            key,
            AttentionMasks(attention_mask=attention_mask),
            runtime_shape,
        )

        expected = attention_mask.expand(branch_count, -1, -1)
        torch.testing.assert_close(normalized, expected)

    def test_reshaper_directly_rejects_static_expert_key_value(self):
        cfg = self.preset(use_kv_expert_models_flag=True)
        reshaper = MixtureOfAttentionHeadsReshaper(cfg)
        tensor = torch.randn(1, 1, cfg.embedding_dim)

        with self.assertRaises(ValueError) as caught:
            reshaper.reshape_qkv_for_attention(
                QKV(query=tensor, key=tensor, value=tensor),
                static_keys=tensor,
            )
        self.assertEqual(
            str(caught.exception),
            "static key/value projections are not supported when "
            "use_kv_expert_models_flag is True.",
        )

    def test_standard_and_equivalent_expanded_masks_produce_same_output(self):
        for use_kv_expert_models_flag in (False, True):
            with self.subTest(
                use_kv_expert_models_flag=use_kv_expert_models_flag,
            ):
                torch.manual_seed(300 + use_kv_expert_models_flag)
                cfg = self.preset(
                    use_kv_expert_models_flag=use_kv_expert_models_flag,
                )
                model = cfg.build().eval()
                inputs = self.input_tensor(cfg)
                sequence_shape = (
                    cfg.target_sequence_length,
                    cfg.source_sequence_length,
                )
                standard_mask = torch.zeros(
                    cfg.batch_size,
                    cfg.num_heads,
                    *sequence_shape,
                    dtype=cfg.target_dtype,
                )
                for batch_index in range(cfg.batch_size):
                    for head_index in range(cfg.num_heads):
                        source_index = (
                            batch_index * cfg.num_heads + head_index
                        ) % cfg.source_sequence_length
                        standard_mask[batch_index, head_index, :, source_index] = float(
                            "-inf"
                        )
                expanded_mask = standard_mask.unsqueeze(1).expand(
                    -1,
                    cfg.experts_config.top_k,
                    -1,
                    -1,
                    -1,
                )

                standard_output, _, _ = model(
                    inputs,
                    inputs,
                    inputs,
                    attention_mask=standard_mask.reshape(-1, *sequence_shape),
                )
                expanded_output, _, _ = model(
                    inputs,
                    inputs,
                    inputs,
                    attention_mask=expanded_mask.reshape(-1, *sequence_shape),
                )

                torch.testing.assert_close(standard_output, expanded_output)

    def test_padding_mask_expands_across_experts_and_heads(self):
        cfg = self.preset()
        model = cfg.build()
        runtime_shape = self.runtime_shape(cfg)
        branch_count = cfg.batch_size * cfg.experts_config.top_k * cfg.num_heads
        key = torch.randn(
            branch_count,
            cfg.source_sequence_length,
            cfg.query_key_projection_dim // cfg.num_heads,
        )
        padding_mask = torch.arange(
            cfg.batch_size * cfg.source_sequence_length,
            dtype=cfg.target_dtype,
        ).view(cfg.batch_size, cfg.source_sequence_length)

        normalized = model.masks.merge_padding_and_attention_mask(
            key,
            AttentionMasks(key_padding_mask=padding_mask),
            runtime_shape,
        )

        expected = (
            padding_mask.view(
                cfg.batch_size,
                1,
                1,
                1,
                cfg.source_sequence_length,
            )
            .expand(
                -1,
                cfg.experts_config.top_k,
                cfg.num_heads,
                -1,
                -1,
            )
            .reshape(branch_count, 1, cfg.source_sequence_length)
        )
        torch.testing.assert_close(normalized, expected)

    def test_invalid_mask_dimensions_raise_clear_errors_and_allow_retry(self):
        cfg = self.preset()
        model = cfg.build()
        inputs = self.input_tensor(cfg)
        invalid_cases = [
            (
                "leading dimension",
                None,
                torch.zeros(
                    3,
                    cfg.target_sequence_length,
                    cfg.source_sequence_length,
                ),
            ),
            (
                "target/source dimensions",
                None,
                torch.zeros(
                    cfg.target_sequence_length + 1,
                    cfg.source_sequence_length,
                ),
            ),
            (
                "target/source dimensions",
                None,
                torch.zeros(
                    cfg.target_sequence_length,
                    cfg.source_sequence_length + 1,
                ),
            ),
            (
                "key_padding_mask must have shape",
                torch.zeros(
                    cfg.batch_size,
                    cfg.source_sequence_length + 1,
                    dtype=torch.bool,
                ),
                None,
            ),
        ]

        for error_pattern, padding_mask, attention_mask in invalid_cases:
            with self.subTest(error_pattern=error_pattern):
                with self.assertRaisesRegex(RuntimeError, error_pattern):
                    model(
                        inputs,
                        inputs,
                        inputs,
                        padding_mask,
                        attention_mask,
                    )
                output, _, auxiliary_loss = model(inputs, inputs, inputs)
                self.assertEqual(output.shape, inputs.shape)
                self.assertIsInstance(auxiliary_loss, torch.Tensor)
                self.assertIsNone(model.projector.auxiliary_loss)

    def test_shared_static_key_value_remains_supported(self):
        cfg = self.preset(
            use_kv_expert_models_flag=False,
            target_sequence_length=3,
            source_sequence_length=5,
        )
        model = cfg.build()
        query = self.input_tensor(cfg)
        key = value = torch.randn(
            4,
            cfg.batch_size,
            cfg.embedding_dim,
        )
        static_source_sequence_length = 5
        head_dim = cfg.embedding_dim // cfg.num_heads
        static_key = torch.randn(
            cfg.batch_size * cfg.num_heads,
            static_source_sequence_length,
            head_dim,
        )
        static_value = torch.randn_like(static_key)

        output, attention_weights, auxiliary_loss = model(
            query,
            key,
            value,
            static_k=static_key,
            static_v=static_value,
        )

        self.assertEqual(output.shape, query.shape)
        self.assertTrue(torch.isfinite(output).all())
        self.assertIsNone(attention_weights)
        self.assertIsInstance(auxiliary_loss, torch.Tensor)

    def test_expert_static_key_value_is_rejected_before_routing(self):
        cfg = self.preset()
        model = cfg.build()
        inputs = self.input_tensor(cfg)
        head_dim = cfg.embedding_dim // cfg.num_heads
        static_key = torch.randn(
            cfg.batch_size * cfg.num_heads,
            cfg.source_sequence_length,
            head_dim,
        )

        with self.assertRaisesRegex(
            ValueError,
            "static key/value projections are not supported",
        ):
            model(
                inputs,
                inputs,
                inputs,
                static_k=static_key,
                static_v=static_key,
            )

        self.assertIsNone(model.projector.auxiliary_loss)
        output, _, auxiliary_loss = model(inputs, inputs, inputs)
        self.assertEqual(output.shape, inputs.shape)
        self.assertIsInstance(auxiliary_loss, torch.Tensor)
        self.assertIsNone(model.projector.auxiliary_loss)

    def test_routing_is_sampled_once_reused_and_loss_is_cleared(self):
        for use_kv_expert_models_flag in (False, True):
            with self.subTest(
                use_kv_expert_models_flag=use_kv_expert_models_flag,
            ):
                torch.manual_seed(400 + use_kv_expert_models_flag)
                cfg = self.preset(
                    use_kv_expert_models_flag=use_kv_expert_models_flag,
                )
                model = cfg.build()
                inputs = self.input_tensor(cfg)
                sampler_calls = []
                routing_inputs = {}

                original_sample = (
                    model.projector.sampler.sample_probabilities_and_indices
                )

                def sample_with_loss(
                    input_matrix,
                    skip_mask=None,
                    *,
                    _original_sample=original_sample,
                    _sampler_calls=sampler_calls,
                ):
                    result = _original_sample(input_matrix, skip_mask)
                    _sampler_calls.append(input_matrix)
                    probabilities, indices, updated_skip_mask, loss = result
                    return (
                        probabilities,
                        indices,
                        updated_skip_mask,
                        loss + input_matrix.new_tensor(1.0),
                    )

                model.projector.sampler.sample_probabilities_and_indices = (
                    sample_with_loss
                )

                expert_models = {
                    "query": model.projector.query_model,
                    "output": model.projector.output_model,
                }
                if use_kv_expert_models_flag:
                    expert_models.update(
                        key=model.projector.key_model,
                        value=model.projector.value_model,
                    )

                for label, expert_model in expert_models.items():
                    original_forward = expert_model.forward
                    routing_inputs[label] = []

                    def forward_with_loss(
                        input_batch,
                        probabilities=None,
                        indices=None,
                        *,
                        _label=label,
                        _original_forward=original_forward,
                        _routing_inputs=routing_inputs,
                    ):
                        _routing_inputs[_label].append((probabilities, indices))
                        output, loss = _original_forward(
                            input_batch,
                            probabilities,
                            indices,
                        )
                        return output, loss + input_batch.new_tensor(1.0)

                    expert_model.forward = forward_with_loss

                expected_loss = 5.0 if use_kv_expert_models_flag else 3.0
                for forward_index in range(2):
                    output, _, auxiliary_loss = model(inputs, inputs, inputs)

                    self.assertEqual(output.shape, inputs.shape)
                    self.assertEqual(auxiliary_loss.item(), expected_loss)
                    self.assertIsNone(model.projector.auxiliary_loss)
                    routed_probabilities = [
                        records[forward_index][0] for records in routing_inputs.values()
                    ]
                    routed_indices = [
                        records[forward_index][1] for records in routing_inputs.values()
                    ]
                    self.assertTrue(
                        all(
                            probabilities is routed_probabilities[0]
                            for probabilities in routed_probabilities[1:]
                        )
                    )
                    self.assertTrue(
                        all(
                            indices is routed_indices[0]
                            for indices in routed_indices[1:]
                        )
                    )

                self.assertEqual(len(sampler_calls), 2)
                for records in routing_inputs.values():
                    self.assertEqual(len(records), 2)

    def test_returning_attention_weights_is_unsupported(self):
        cfg = self.preset(return_attention_weights_flag=True)
        model = cfg.build()
        inputs = self.input_tensor(cfg)

        with self.assertRaisesRegex(RuntimeError, "attention_weights"):
            model(inputs, inputs, inputs)

    def test_expert_projections_are_reshaped_for_processor(self):
        torch.manual_seed(0)
        cfg = self.preset()
        model = cfg.build()
        inputs = torch.randn(
            cfg.target_sequence_length,
            cfg.batch_size,
            cfg.embedding_dim,
        )
        projections = model.projector.compute_qkv_projections(
            QKV(query=inputs, key=inputs, value=inputs),
        )

        projections = model.reshaper.reshape_qkv_for_attention(projections)
        projections = model.reshaper.reshape_before_attention(projections)

        head_dim = cfg.query_key_projection_dim // cfg.num_heads
        value_head_dim = cfg.value_projection_dim // cfg.num_heads
        self.assertEqual(
            projections.query.shape,
            (
                cfg.batch_size,
                cfg.experts_config.top_k,
                cfg.num_heads,
                cfg.target_sequence_length,
                head_dim,
            ),
        )
        self.assertEqual(
            projections.key.shape,
            (
                cfg.batch_size,
                cfg.experts_config.top_k,
                cfg.num_heads,
                cfg.source_sequence_length,
                head_dim,
            ),
        )
        self.assertEqual(
            projections.value.shape,
            (
                cfg.batch_size,
                cfg.experts_config.top_k,
                cfg.num_heads,
                cfg.source_sequence_length,
                value_head_dim,
            ),
        )

    def test_expert_key_value_requires_equal_configured_sequence_lengths(self):
        cfg = self.preset(
            target_sequence_length=3,
            source_sequence_length=4,
        )

        with self.assertRaisesRegex(
            ValueError,
            "target_sequence_length and source_sequence_length must be equal",
        ):
            cfg.build()

    def test_expert_sampler_and_router_dimensions_must_match(self):
        cfg = self.preset()
        cfg.experts_config.sampler_config.top_k = 1
        with self.assertRaisesRegex(ValueError, "top_k must match"):
            cfg.build()

        cfg = self.preset()
        cfg.experts_config.sampler_config.num_experts = 3
        with self.assertRaisesRegex(ValueError, "num_experts must match"):
            cfg.build()

        cfg = self.preset()
        cfg.experts_config.sampler_config.router_config.num_experts = 3
        with self.assertRaisesRegex(ValueError, "router_config.num_experts"):
            cfg.build()

    def test_dense_expert_routing_is_rejected_clearly(self):
        cfg = self.preset(
            experts_top_k=4,
            experts_num_experts=4,
        )

        with self.assertRaisesRegex(ValueError, "dense routing is not supported"):
            cfg.build()

    def test_expert_key_value_requires_shared_query_key_value_tensor(self):
        cfg = self.preset()
        model = cfg.build()
        query = torch.randn(4, 2, 8)
        key = query.clone()
        value = query.clone()

        with self.assertRaisesRegex(
            ValueError,
            "query, key, and value must be the same tensor",
        ):
            model(query, key, value)

        self.assertIsNone(model.projector.auxiliary_loss)

    def test_flag_matrix_propagates_auxiliary_loss_and_gradients(self):
        cases = [
            ("plain", False, False, False),
            ("masks", False, False, True),
            ("bias", True, False, False),
            ("zero_attention", False, True, False),
            ("combined", True, True, True),
        ]
        for case_index, (
            case_name,
            add_key_value_bias_flag,
            zero_attention_flag,
            use_masks,
        ) in enumerate(cases):
            with self.subTest(case_name=case_name):
                torch.manual_seed(case_index)
                cfg = self.preset(
                    add_key_value_bias_flag=add_key_value_bias_flag,
                    zero_attention_flag=zero_attention_flag,
                )
                model = cfg.build()
                inputs = torch.randn(
                    cfg.target_sequence_length,
                    cfg.batch_size,
                    cfg.embedding_dim,
                    requires_grad=True,
                )
                key_padding_mask = None
                attention_mask = None
                if use_masks:
                    key_padding_mask = torch.tensor(
                        [
                            [False, True, False, False],
                            [False, False, True, False],
                        ]
                    )
                    total_batch_size = (
                        cfg.batch_size * cfg.experts_config.top_k * cfg.num_heads
                    )
                    attention_mask = torch.zeros(
                        total_batch_size,
                        cfg.target_sequence_length,
                        cfg.source_sequence_length,
                        dtype=torch.bool,
                    )
                    attention_mask[:, 0, -1] = True

                output, attention_weights, auxiliary_loss = model(
                    inputs,
                    inputs,
                    inputs,
                    key_padding_mask,
                    attention_mask,
                )

                self.assertEqual(
                    output.shape,
                    (
                        cfg.target_sequence_length,
                        cfg.batch_size,
                        cfg.embedding_dim,
                    ),
                )
                self.assertIsNone(attention_weights)
                self.assertIsInstance(auxiliary_loss, torch.Tensor)

                (output.square().sum() + auxiliary_loss).backward()

                self.assertIsNotNone(inputs.grad)
                self.assertTrue(torch.any(inputs.grad.abs() > 0))
                for label in (
                    "query_model",
                    "key_model",
                    "value_model",
                    "output_model",
                ):
                    self.assert_has_nonzero_parameter_gradient(
                        getattr(model.projector, label),
                        label,
                    )
                if add_key_value_bias_flag:
                    self.assertIsNotNone(model.bias.key_bias_vector.grad)
                    self.assertIsNotNone(model.bias.value_bias_vector.grad)
                    self.assertTrue(
                        torch.any(model.bias.key_bias_vector.grad.abs() > 0)
                    )
                    self.assertTrue(
                        torch.any(model.bias.value_bias_vector.grad.abs() > 0)
                    )


class TestMixtureOfAttentionHeadsLocality(unittest.TestCase):
    def test_shared_attention_modules_do_not_contain_mixture_layout_knowledge(self):
        root = Path(__file__).parents[2]
        shared_paths = (
            *sorted((root / "emperor/attention/core/handlers").glob("*.py")),
            root / "emperor/attention/core/_validator.py",
            root / "emperor/attention/core/runtime.py",
            root / "emperor/attention/core/monitor.py",
        )
        forbidden_tokens = (
            "top_k",
            "expert_branch_count",
            "supports_expert_branches",
            "MixtureOfAttentionHeads",
            "Mixture attention",
            "mixture of attention heads",
            "selected_expert",
            "multiplier",
        )

        for path in shared_paths:
            source = path.read_text()
            for token in forbidden_tokens:
                with self.subTest(path=path.relative_to(root), token=token):
                    self.assertNotIn(token, source)

        shared_processor = (
            root / "emperor/attention/core/handlers/processor.py"
        ).read_text()
        standard_monitor = (root / "emperor/attention/core/monitor.py").read_text()
        self.assertNotIn("query.dim() == 5", shared_processor)
        self.assertNotIn("detached_weights.dim() == 5", standard_monitor)


if __name__ == "__main__":
    unittest.main()
