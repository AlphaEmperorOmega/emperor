import unittest
from types import SimpleNamespace

import torch

from emperor.attention import (
    MixtureOfAttentionHeadsConfig,
    SelfAttentionConfig,
)
from emperor.attention._ops.reshaping import AttentionReshaper
from emperor.attention._runtime import QKV, AttentionRuntimeLayout
from emperor.attention._validation import (
    AttentionValidatorBase,
    MultiHeadAttentionValidator,
)
from emperor.attention._variants.mixture.zero_attention import (
    MixtureOfAttentionHeadsZeroAttention,
)
from support.attention import build_attention_config


class ExactErrorMixin:
    def assert_exact_error(self, error_type, message, callable_, *args, **kwargs):
        with self.assertRaises(error_type) as raised:
            callable_(*args, **kwargs)
        self.assertEqual(str(raised.exception), message)


class TestAttentionValidatorMutationContracts(ExactErrorMixin, unittest.TestCase):
    def test_relative_position_head_shape_error_reports_actual_head_axis(self):
        query = torch.empty(2, 3, 4, 5)

        self.assert_exact_error(
            RuntimeError,
            "relative-position rank-4 query head dimension must equal "
            "num_heads (2), got 3.",
            AttentionValidatorBase.validate_standard_relative_position_query_shape,
            query,
            2,
        )

    def test_head_divisibility_reports_each_effective_width_exactly(self):
        cases = (
            (
                SimpleNamespace(
                    embedding_dim=5,
                    num_heads=2,
                    query_key_projection_dim=0,
                    value_projection_dim=0,
                ),
                "embedding_dim (5) must be perfectly divisible by num_heads (2).",
            ),
            (
                SimpleNamespace(
                    embedding_dim=4,
                    num_heads=2,
                    query_key_projection_dim=3,
                    value_projection_dim=0,
                ),
                "query_key_projection_dim (3) must be perfectly divisible by "
                "num_heads (2).",
            ),
            (
                SimpleNamespace(
                    embedding_dim=4,
                    num_heads=2,
                    query_key_projection_dim=4,
                    value_projection_dim=3,
                ),
                "value_projection_dim (3) must be perfectly divisible by "
                "num_heads (2).",
            ),
        )

        for model, message in cases:
            with self.subTest(message=message):
                self.assert_exact_error(
                    ValueError,
                    message,
                    AttentionValidatorBase.validate_head_divisibility,
                    model,
                )

    def test_non_self_attention_weight_error_is_stable(self):
        self.assert_exact_error(
            RuntimeError,
            "attention_weights can be returned only when self attention is "
            "computed; ensure the query, key, and value tensors are the same "
            "tensor.",
            AttentionValidatorBase.validate_attention_weights_returned_for_self_attention_only,
            SimpleNamespace(return_attention_weights_flag=True),
        )

    def test_input_rank_errors_distinguish_every_operand_and_layout(self):
        batched = torch.empty(2, 1, 4)
        unbatched = torch.empty(2, 4)
        cases = (
            (
                (torch.empty(4), unbatched, unbatched, None, None),
                "Query should be an unbatched 2D or batched 3D tensor but received "
                "a 1-D tensor.",
            ),
            (
                (batched, unbatched, batched, None, None),
                "For a batched (3-D) query, expected key and value to be 3-D but "
                "found 2-D and 3-D tensors respectively.",
            ),
            (
                (unbatched, unbatched, batched, None, None),
                "For a unbatched (2-D) query, expected key and value to be 2-D but "
                "found 2-D and 3-D tensors respectively.",
            ),
            (
                (batched, batched, batched, torch.empty(2), None),
                "For a batched (3-D) query, expected key_padding_mask to be None or "
                "2-D but found a 1-D tensor instead.",
            ),
            (
                (unbatched, unbatched, unbatched, torch.empty(1, 2), None),
                "For a unbatched (2-D) query, expected key_padding_mask to be None "
                "or 1-D but found a 2-D tensor instead.",
            ),
            (
                (batched, batched, batched, None, torch.empty(1)),
                "Expected attention_mask to be None, 2-D, or 3-D but found a 1-D "
                "tensor instead.",
            ),
        )

        for args, message in cases:
            with self.subTest(message=message):
                self.assert_exact_error(
                    RuntimeError,
                    message,
                    AttentionValidatorBase.validate_input_shapes,
                    *args,
                )

    def test_key_value_projection_shape_error_lists_both_shapes(self):
        key = torch.empty(2, 3, 4)
        value = torch.empty(2, 5, 6)
        self.assert_exact_error(
            RuntimeError,
            "key shape (2, 3, 4) does not match value shape (2, 5, 6) on the "
            "sequence and batch dimensions.",
            AttentionValidatorBase.validate_key_value_projection_shapes,
            key,
            value,
        )

    def test_static_projection_contract_uses_runtime_batch_and_distinct_widths(self):
        model = SimpleNamespace(
            batch_size=7,
            num_heads=2,
            embedding_dim=8,
            query_key_projection_dim=6,
            value_projection_dim=8,
        )
        runtime_layout = AttentionRuntimeLayout(2, 1, 5)
        static_keys = torch.empty(4, 5, 3)
        static_values = torch.empty(4, 5, 4)

        self.assertIsNone(
            AttentionValidatorBase.validate_static_projection_shapes(
                model,
                static_keys,
                static_values,
                runtime_layout,
            )
        )

        cases = (
            (
                torch.empty(4, 3),
                "static_keys",
                "static_keys must be rank 3 with shape [batch * heads, source, "
                "head_width], got rank 2.",
            ),
            (
                torch.empty(5, 7, 3),
                "static_keys",
                "expecting static_keys.size(0) of 4, but got 5.",
            ),
            (
                torch.empty(4, 5, 2),
                "static_keys",
                "expecting static_keys.size(2) of 3, but got 2.",
            ),
            (
                torch.empty(4, 5, 3),
                "static_values",
                "expecting static_values.size(2) of 4, but got 3.",
            ),
        )
        for tensor, name, message in cases:
            with self.subTest(name=name, message=message):
                self.assert_exact_error(
                    RuntimeError,
                    message,
                    AttentionValidatorBase._validate_static_projection_shape,
                    model,
                    tensor,
                    name,
                    runtime_layout,
                )

    def test_mask_relationship_and_required_causal_errors_are_stable(self):
        self.assert_exact_error(
            RuntimeError,
            "Support for mismatched attention_mask and key_padding_mask is "
            "deprecated. Use the same type for both instead.",
            AttentionValidatorBase.validate_mask_dtype_matches,
            torch.zeros(1, dtype=torch.bool),
            "attention_mask",
            torch.float32,
            "key_padding_mask",
        )
        self.assertIsNone(
            AttentionValidatorBase.validate_mask_dtype_matches(
                torch.zeros(1, dtype=torch.bool),
                "attention_mask",
                torch.float32,
                "key_padding_mask",
                check_other=False,
            )
        )
        self.assert_exact_error(
            RuntimeError,
            "Causal attention requires a prepared attention_mask. The attention "
            "masking module generates this mask from the runtime query and key "
            "lengths before validation.",
            AttentionValidatorBase.validate_attention_mask_for_required_causal_mask,
            None,
            True,
        )


class TestAttentionOperationMutationContracts(unittest.TestCase):
    def test_mixture_zero_attention_uses_runtime_and_configured_batch_sizes(self):
        cfg = build_attention_config(
            config_class=MixtureOfAttentionHeadsConfig,
            batch_size=7,
            num_heads=2,
            experts_top_k=3,
            use_kv_expert_models_flag=True,
        )
        model = MixtureOfAttentionHeadsZeroAttention(cfg)

        self.assertEqual(
            model._get_branch_count(AttentionRuntimeLayout(2, 3, 5)),
            12,
        )
        self.assertEqual(model._get_branch_count(), 42)


class TestMultiHeadAttentionValidatorMutationContracts(
    ExactErrorMixin,
    unittest.TestCase,
):
    def test_scalar_configuration_errors_are_exact(self):
        model = SimpleNamespace(
            batch_size=2.0,
            num_heads=2,
            embedding_dim=4,
            query_key_projection_dim=4,
            value_projection_dim=4,
            target_sequence_length=3,
            source_sequence_length=3,
        )
        cases = (
            (
                TypeError,
                "batch_first_flag must be True, False, or None, received 1.",
                MultiHeadAttentionValidator.validate_batch_first_flag,
                (1,),
            ),
            (
                TypeError,
                "batch_size must be int, received float.",
                MultiHeadAttentionValidator.validate_integer_dimension_types,
                (model,),
            ),
            (
                ValueError,
                "dropout probability is invalid: dropout_probability must be between "
                "0 and 1 inclusive, received -0.25.",
                MultiHeadAttentionValidator.validate_dropout_probability,
                (-0.25,),
            ),
            (
                TypeError,
                "target_dtype must be a torch dtype, received str.",
                MultiHeadAttentionValidator.validate_target_dtype,
                ("float32",),
            ),
            (
                ValueError,
                "target_dtype must be a floating torch dtype, received torch.int64.",
                MultiHeadAttentionValidator.validate_target_dtype,
                (torch.int64,),
            ),
        )
        for error_type, message, callable_, args in cases:
            with self.subTest(message=message):
                self.assert_exact_error(error_type, message, callable_, *args)
        self.assertIsNone(MultiHeadAttentionValidator.validate_dropout_probability(1.0))

    def test_static_input_validator_checks_each_projection_shape(self):
        cfg = build_attention_config(
            batch_size=7,
            num_heads=2,
            embedding_dim=12,
            query_key_projection_dim=8,
            value_projection_dim=6,
        )
        model = cfg.build()
        runtime_layout = AttentionRuntimeLayout(2, 3, 5)
        qkv = QKV(
            query=torch.empty(3, 2, 12),
            key=torch.empty(5, 2, 12),
            value=torch.empty(5, 2, 12),
        )
        cases = (
            (
                torch.empty(5, 5, 4),
                None,
                "expecting static_keys.size(0) of 4, but got 5.",
            ),
            (
                None,
                torch.empty(4, 5, 4),
                "expecting static_values.size(2) of 3, but got 4.",
            ),
        )

        for static_keys, static_values, message in cases:
            with self.subTest(message=message):
                self.assert_exact_error(
                    RuntimeError,
                    message,
                    MultiHeadAttentionValidator.validate_static_key_value_inputs,
                    model,
                    qkv,
                    static_keys,
                    static_values,
                    runtime_layout,
                )

    def test_attention_reshaper_validates_each_static_projection_shape(self):
        cfg = build_attention_config(
            batch_size=7,
            num_heads=2,
            embedding_dim=12,
            query_key_projection_dim=8,
            value_projection_dim=6,
        )
        reshaper = AttentionReshaper(cfg)
        runtime_layout = AttentionRuntimeLayout(2, 3, 5)
        qkv = QKV(
            query=torch.arange(48.0).view(3, 2, 8),
            key=torch.arange(80.0).view(5, 2, 8),
            value=torch.arange(60.0).view(5, 2, 6),
        )
        cases = (
            (
                torch.empty(5, 5, 4),
                None,
                "expecting static_keys.size(0) of 4, but got 5.",
            ),
            (
                None,
                torch.empty(4, 5, 4),
                "expecting static_values.size(2) of 3, but got 4.",
            ),
        )

        for static_keys, static_values, message in cases:
            with self.subTest(message=message):
                self.assert_exact_error(
                    RuntimeError,
                    message,
                    reshaper.reshape_qkv_for_attention,
                    qkv,
                    static_keys,
                    static_values,
                    runtime_layout,
                )

    def test_attention_reshaper_uses_runtime_sized_static_projections(self):
        cfg = build_attention_config(
            batch_size=7,
            num_heads=2,
            embedding_dim=12,
            query_key_projection_dim=8,
            value_projection_dim=6,
        )
        reshaper = AttentionReshaper(cfg)
        runtime_layout = AttentionRuntimeLayout(2, 3, 5)
        query = torch.arange(48.0).view(3, 2, 8)
        static_keys = torch.arange(80.0).view(4, 5, 4)
        static_values = torch.arange(60.0).view(4, 5, 3)

        output = reshaper.reshape_qkv_for_attention(
            QKV(
                query=query,
                key=torch.full((5, 2, 8), -1.0),
                value=torch.full((5, 2, 6), -2.0),
            ),
            static_keys,
            static_values,
            runtime_layout,
        )

        torch.testing.assert_close(
            output.query,
            query.view(3, 4, 4).transpose(0, 1),
        )
        self.assertIs(output.key, static_keys)
        self.assertIs(output.value, static_values)

    def test_attention_reshaper_uses_distinct_qk_and_value_head_widths(self):
        cfg = build_attention_config(
            batch_size=7,
            num_heads=2,
            embedding_dim=12,
            query_key_projection_dim=8,
            value_projection_dim=6,
        )
        reshaper = AttentionReshaper(cfg)
        runtime_layout = AttentionRuntimeLayout(2, 3, 5)
        query = torch.arange(48.0).view(3, 2, 8)
        key = torch.arange(80.0).view(5, 2, 8)
        value = torch.arange(60.0).view(5, 2, 6)

        output = reshaper.reshape_qkv_for_attention(
            QKV(query=query, key=key, value=value),
            runtime_layout=runtime_layout,
        )

        torch.testing.assert_close(
            output.query,
            query.view(3, 4, 4).transpose(0, 1),
        )
        torch.testing.assert_close(
            output.key,
            key.view(5, 4, 4).transpose(0, 1),
        )
        torch.testing.assert_close(
            output.value,
            value.view(5, 4, 3).transpose(0, 1),
        )

    def test_relative_head_count_error_is_exact(self):
        relative_config = SimpleNamespace(num_heads=3, embedding_dim=4)
        model = SimpleNamespace(
            cfg=SimpleNamespace(
                relative_positional_embedding_config=relative_config,
            ),
            num_heads=2,
            query_key_projection_dim=4,
            embedding_dim=4,
        )
        self.assert_exact_error(
            ValueError,
            "relative positional embedding num_heads must match attention "
            "num_heads, got 3 and 2.",
            MultiHeadAttentionValidator.validate_relative_configuration,
            model,
        )

    def test_nested_configuration_type_errors_are_exact(self):
        cases = (
            (
                SimpleNamespace(
                    projection_model_config=object(),
                    relative_positional_embedding_config=None,
                ),
                "projection model configuration must be a LayerStackConfig or "
                "RecurrentLayerConfig, received object.",
            ),
            (
                SimpleNamespace(
                    projection_model_config=build_attention_config(
                        SelfAttentionConfig
                    ).projection_model_config,
                    relative_positional_embedding_config=object(),
                ),
                "relative positional embedding configuration must be a "
                "RelativePositionalEmbeddingConfig or None, received object.",
            ),
        )
        for cfg, message in cases:
            with self.subTest(message=message):
                self.assert_exact_error(
                    TypeError,
                    message,
                    MultiHeadAttentionValidator.validate_nested_configurations,
                    cfg,
                )

    def test_runtime_maximum_errors_name_the_dimension_and_values(self):
        model = SimpleNamespace(
            batch_size=2,
            target_sequence_length=3,
            source_sequence_length=4,
        )
        cases = (
            (
                AttentionRuntimeLayout(3, 3, 4),
                "Runtime batch_size (3) exceeds configured maximum (2).",
            ),
            (
                AttentionRuntimeLayout(2, 4, 4),
                "Runtime target_sequence_length (4) exceeds configured maximum (3).",
            ),
            (
                AttentionRuntimeLayout(2, 3, 5),
                "Runtime source_sequence_length (5) exceeds configured maximum (4).",
            ),
        )
        for runtime_layout, message in cases:
            with self.subTest(message=message):
                self.assert_exact_error(
                    ValueError,
                    message,
                    MultiHeadAttentionValidator.validate_runtime_layout,
                    model,
                    runtime_layout,
                )

    def test_runtime_tensor_contract_distinguishes_each_source(self):
        model = SimpleNamespace(embedding_dim=4)
        valid = torch.empty(2, 1, 4)
        cases = (
            (
                QKV(query=torch.empty(0, 1, 4), key=valid, value=valid),
                "query sequence length must be greater than 0.",
            ),
            (
                QKV(query=valid, key=torch.empty(0, 1, 4), value=valid),
                "key and value sequence lengths must be greater than 0.",
            ),
            (
                QKV(query=valid, key=valid, value=torch.empty(0, 1, 4)),
                "key and value sequence lengths must be greater than 0.",
            ),
            (
                QKV(
                    query=torch.empty(2, 1, 4),
                    key=torch.empty(2, 2, 4),
                    value=torch.empty(2, 3, 4),
                ),
                "query, key, and value batch sizes must match, got 1, 2, and 3.",
            ),
            (
                QKV(
                    query=torch.empty(2, 1, 5),
                    key=torch.empty(3, 1, 4),
                    value=torch.empty(3, 1, 4),
                ),
                "query embedding width must be 4, got 5.",
            ),
            (
                QKV(
                    query=torch.empty(2, 1, 4, dtype=torch.float16),
                    key=torch.empty(2, 1, 4, dtype=torch.float32),
                    value=torch.empty(2, 1, 4, dtype=torch.float64),
                ),
                "query, key, and value dtypes must match, got torch.float16, "
                "torch.float32, and torch.float64.",
            ),
            (
                QKV(
                    query=torch.empty(2, 1, 4, dtype=torch.int64),
                    key=torch.empty(2, 1, 4, dtype=torch.int64),
                    value=torch.empty(2, 1, 4, dtype=torch.int64),
                ),
                "query, key, and value must be floating point tensors.",
            ),
            (
                QKV(
                    query=torch.empty(2, 1, 4, device="meta"),
                    key=torch.empty(2, 1, 4),
                    value=torch.empty(2, 1, 4, device="meta"),
                ),
                "query, key, and value devices must match, got meta, cpu, and meta.",
            ),
        )
        for qkv, message in cases:
            with self.subTest(message=message):
                self.assert_exact_error(
                    RuntimeError,
                    message,
                    MultiHeadAttentionValidator.validate_runtime_tensors,
                    model,
                    qkv,
                )

    def test_static_input_validation_checks_each_selected_source(self):
        qkv = QKV(
            query=torch.empty(2, 1, 4),
            key=torch.empty(3, 1, 4),
            value=torch.empty(3, 1, 4),
        )
        model = SimpleNamespace(
            embedding_dim=4,
            num_heads=1,
            query_key_projection_dim=4,
            value_projection_dim=4,
        )
        runtime_layout = AttentionRuntimeLayout(1, 2, 3)
        self.assert_exact_error(
            RuntimeError,
            "static_values dtype must match query dtype, got torch.float64 and "
            "torch.float32.",
            MultiHeadAttentionValidator.validate_static_key_value_inputs,
            model,
            qkv,
            None,
            torch.empty(1, 3, 4, dtype=torch.float64),
            runtime_layout,
        )
        self.assert_exact_error(
            RuntimeError,
            "static_keys device must match query device, got meta and cpu.",
            MultiHeadAttentionValidator.validate_static_key_value_inputs,
            model,
            qkv,
            torch.empty(1, 3, 4, device="meta"),
            None,
            runtime_layout,
        )
        self.assert_exact_error(
            RuntimeError,
            "Selected key and value sources must have equal sequence lengths, got "
            "2 and 4.",
            MultiHeadAttentionValidator.validate_static_key_value_inputs,
            model,
            qkv,
            torch.empty(1, 2, 4),
            torch.empty(1, 4, 4),
            runtime_layout,
        )

    def test_static_input_validation_is_read_only(self):
        model = SimpleNamespace(
            embedding_dim=4,
            num_heads=1,
            query_key_projection_dim=4,
            value_projection_dim=4,
        )
        qkv = QKV(
            query=torch.randn(2, 1, 4),
            key=torch.randn(3, 1, 4),
            value=torch.randn(3, 1, 4),
        )
        static_keys = torch.randn(1, 3, 4)
        static_values = torch.randn(1, 3, 4)
        runtime_layout = AttentionRuntimeLayout(1, 2, 3)
        original_tensors = (
            qkv.query.clone(),
            qkv.key.clone(),
            qkv.value.clone(),
            static_keys.clone(),
            static_values.clone(),
        )

        result = MultiHeadAttentionValidator.validate_static_key_value_inputs(
            model,
            qkv,
            static_keys,
            static_values,
            runtime_layout,
        )

        self.assertIsNone(result)
        self.assertEqual(runtime_layout.source_sequence_length, 3)
        for actual, expected in zip(
            (qkv.query, qkv.key, qkv.value, static_keys, static_values),
            original_tensors,
            strict=True,
        ):
            torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
    unittest.main()
