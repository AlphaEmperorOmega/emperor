import unittest
from types import SimpleNamespace

import torch

from emperor.attention import (
    IndependentAttentionConfig,
    MixtureOfAttentionHeadsConfig,
    SelfAttentionConfig,
    SelfAttentionProjectionStrategy,
)
from emperor.attention._runtime import QKV, AttentionMasks, AttentionRuntimeLayout
from emperor.attention._validation import (
    AttentionValidatorBase,
    MultiHeadAttentionValidator,
)
from emperor.attention._variants.independent.validation import (
    IndependentAttentionValidator,
)
from emperor.attention._variants.mixture.validation import (
    MixtureOfAttentionHeadsValidator,
)
from emperor.attention._variants.self_attention.validation import (
    SelfAttentionValidator,
)
from support.attention import build_attention_config

BATCH_SIZE = 4
NUM_HEADS = 4
EMBEDDING_DIM = 12
TARGET_SEQUENCE_LENGTH = 8
SOURCE_SEQUENCE_LENGTH = 8
HEAD_DIM = EMBEDDING_DIM // NUM_HEADS


class TestValidateInputShapes(unittest.TestCase):
    def _batched_qkv(self):
        query = torch.randn(TARGET_SEQUENCE_LENGTH, BATCH_SIZE, EMBEDDING_DIM)
        key = torch.randn(SOURCE_SEQUENCE_LENGTH, BATCH_SIZE, EMBEDDING_DIM)
        value = torch.randn(SOURCE_SEQUENCE_LENGTH, BATCH_SIZE, EMBEDDING_DIM)
        return query, key, value

    def test_raises_on_1d_query(self):
        with self.assertRaises(RuntimeError):
            AttentionValidatorBase.validate_input_shapes(
                torch.randn(EMBEDDING_DIM),
                torch.randn(SOURCE_SEQUENCE_LENGTH, EMBEDDING_DIM),
                torch.randn(SOURCE_SEQUENCE_LENGTH, EMBEDDING_DIM),
            )

    def test_raises_on_4d_query(self):
        with self.assertRaises(RuntimeError):
            AttentionValidatorBase.validate_input_shapes(
                torch.randn(2, 2, 2, 2),
                torch.randn(SOURCE_SEQUENCE_LENGTH, BATCH_SIZE, EMBEDDING_DIM),
                torch.randn(SOURCE_SEQUENCE_LENGTH, BATCH_SIZE, EMBEDDING_DIM),
            )

    def test_passes_on_unbatched_2d(self):
        query = torch.randn(TARGET_SEQUENCE_LENGTH, EMBEDDING_DIM)
        key = torch.randn(SOURCE_SEQUENCE_LENGTH, EMBEDDING_DIM)
        value = torch.randn(SOURCE_SEQUENCE_LENGTH, EMBEDDING_DIM)
        self.assertIsNone(
            AttentionValidatorBase.validate_input_shapes(query, key, value)
        )

    def test_passes_on_batched_3d(self):
        query, key, value = self._batched_qkv()
        self.assertIsNone(
            AttentionValidatorBase.validate_input_shapes(query, key, value)
        )

    def test_raises_on_kv_dim_mismatch(self):
        query, _, _ = self._batched_qkv()
        with self.assertRaises(RuntimeError):
            AttentionValidatorBase.validate_input_shapes(
                query,
                torch.randn(SOURCE_SEQUENCE_LENGTH, EMBEDDING_DIM),
                torch.randn(SOURCE_SEQUENCE_LENGTH, EMBEDDING_DIM),
            )

    def test_raises_on_bad_key_padding_mask_dims(self):
        query, key, value = self._batched_qkv()
        bad_key_padding_mask = torch.randn(
            SOURCE_SEQUENCE_LENGTH, BATCH_SIZE, EMBEDDING_DIM
        )
        with self.assertRaises(RuntimeError):
            AttentionValidatorBase.validate_input_shapes(
                query, key, value, bad_key_padding_mask
            )

    def test_raises_on_4d_attention_mask(self):
        query, key, value = self._batched_qkv()
        bad_attention_mask = torch.randn(
            NUM_HEADS, BATCH_SIZE, TARGET_SEQUENCE_LENGTH, SOURCE_SEQUENCE_LENGTH
        )
        with self.assertRaises(RuntimeError):
            AttentionValidatorBase.validate_input_shapes(
                query, key, value, None, bad_attention_mask
            )

    def test_passes_on_2d_and_3d_attention_mask(self):
        query, key, value = self._batched_qkv()
        two_d = torch.randn(TARGET_SEQUENCE_LENGTH, SOURCE_SEQUENCE_LENGTH)
        three_d = torch.randn(
            BATCH_SIZE * NUM_HEADS, TARGET_SEQUENCE_LENGTH, SOURCE_SEQUENCE_LENGTH
        )
        self.assertIsNone(
            AttentionValidatorBase.validate_input_shapes(query, key, value, None, two_d)
        )
        self.assertIsNone(
            AttentionValidatorBase.validate_input_shapes(
                query, key, value, None, three_d
            )
        )


class TestValidateKeyValueProjectionShapes(unittest.TestCase):
    def test_passes_when_shapes_match(self):
        key = torch.randn(SOURCE_SEQUENCE_LENGTH, BATCH_SIZE, EMBEDDING_DIM)
        value = torch.randn(SOURCE_SEQUENCE_LENGTH, BATCH_SIZE, EMBEDDING_DIM)
        self.assertIsNone(
            AttentionValidatorBase.validate_key_value_projection_shapes(key, value)
        )

    def test_raises_when_sequence_length_mismatch(self):
        key = torch.randn(SOURCE_SEQUENCE_LENGTH + 1, BATCH_SIZE, EMBEDDING_DIM)
        value = torch.randn(SOURCE_SEQUENCE_LENGTH, BATCH_SIZE, EMBEDDING_DIM)
        with self.assertRaises(RuntimeError):
            AttentionValidatorBase.validate_key_value_projection_shapes(key, value)


class TestValidateStaticProjectionShapes(unittest.TestCase):
    def setUp(self):
        self.model = SimpleNamespace(
            batch_size=BATCH_SIZE,
            num_heads=NUM_HEADS,
            head_dim=HEAD_DIM,
            embedding_dim=EMBEDDING_DIM,
            query_key_projection_dim=0,
            value_projection_dim=0,
        )

    def test_none_inputs(self):
        self.assertIsNone(
            AttentionValidatorBase.validate_static_projection_shapes(
                self.model, None, None
            )
        )

    def test_correct_static_shapes(self):
        static = torch.randn(BATCH_SIZE * NUM_HEADS, SOURCE_SEQUENCE_LENGTH, HEAD_DIM)
        self.assertIsNone(
            AttentionValidatorBase.validate_static_projection_shapes(
                self.model, static, static
            )
        )

    def test_wrong_static_first_dim(self):
        wrong = torch.randn(BATCH_SIZE, SOURCE_SEQUENCE_LENGTH, HEAD_DIM)
        with self.assertRaises(RuntimeError):
            AttentionValidatorBase.validate_static_projection_shapes(
                self.model, wrong, None
            )

    def test_wrong_static_head_width(self):
        wrong = torch.randn(
            BATCH_SIZE * NUM_HEADS,
            SOURCE_SEQUENCE_LENGTH,
            HEAD_DIM + 1,
        )
        with self.assertRaisesRegex(RuntimeError, r"static_values.size\(2\)"):
            AttentionValidatorBase.validate_static_projection_shapes(
                self.model,
                None,
                wrong,
            )


class TestValidateHeadDivisibility(unittest.TestCase):
    def test_passes_when_divisible(self):
        model = SimpleNamespace(
            embedding_dim=12,
            num_heads=4,
            query_key_projection_dim=0,
            value_projection_dim=0,
        )
        self.assertIsNone(AttentionValidatorBase.validate_head_divisibility(model))

    def test_raises_when_embedding_not_divisible(self):
        model = SimpleNamespace(
            embedding_dim=13,
            num_heads=4,
            query_key_projection_dim=0,
            value_projection_dim=0,
        )
        with self.assertRaises(ValueError):
            AttentionValidatorBase.validate_head_divisibility(model)

    def test_raises_when_projection_not_divisible(self):
        model = SimpleNamespace(
            embedding_dim=12,
            num_heads=4,
            query_key_projection_dim=14,
            value_projection_dim=0,
        )
        with self.assertRaises(ValueError):
            AttentionValidatorBase.validate_head_divisibility(model)

    def test_raises_when_value_projection_not_divisible(self):
        model = SimpleNamespace(
            embedding_dim=12,
            num_heads=4,
            query_key_projection_dim=8,
            value_projection_dim=10,
        )

        with self.assertRaisesRegex(ValueError, "value_projection_dim"):
            AttentionValidatorBase.validate_head_divisibility(model)


class TestRuntimeDeviceValidation(unittest.TestCase):
    def test_runtime_qkv_devices_must_match(self):
        model = SimpleNamespace(embedding_dim=4)
        query = torch.empty(2, 1, 4)
        key = torch.empty(2, 1, 4, device="meta")
        value = torch.empty(2, 1, 4, device="meta")

        with self.assertRaisesRegex(RuntimeError, "devices must match"):
            MultiHeadAttentionValidator.validate_runtime_tensors(
                model,
                QKV(query=query, key=key, value=value),
            )

    def test_static_projection_device_must_match_query(self):
        model = SimpleNamespace(
            embedding_dim=4,
            num_heads=1,
            query_key_projection_dim=4,
            value_projection_dim=4,
        )
        query = torch.empty(2, 1, 4)
        qkv = QKV(query=query, key=query, value=query)
        static_key = torch.empty(1, 2, 4, device="meta")
        runtime_layout = AttentionRuntimeLayout(1, 2, 2)

        with self.assertRaisesRegex(RuntimeError, "device must match query device"):
            MultiHeadAttentionValidator.validate_static_key_value_inputs(
                model,
                qkv,
                static_key,
                None,
                runtime_layout,
            )


class TestAttentionWeightsForSelfAttentionOnly(unittest.TestCase):
    def test_passes_when_flag_false(self):
        model = SimpleNamespace(return_attention_weights_flag=False)
        self.assertIsNone(
            AttentionValidatorBase.validate_attention_weights_returned_for_self_attention_only(
                model
            )
        )

    def test_raises_when_flag_true(self):
        model = SimpleNamespace(return_attention_weights_flag=True)
        with self.assertRaises(RuntimeError):
            AttentionValidatorBase.validate_attention_weights_returned_for_self_attention_only(
                model
            )


class TestSelfAttentionValidator(unittest.TestCase):
    def test_validate_passes_for_equal_dims(self):
        model = build_attention_config(
            SelfAttentionConfig,
            embedding_dim=EMBEDDING_DIM,
            query_key_projection_dim=EMBEDDING_DIM,
            value_projection_dim=EMBEDDING_DIM,
        ).build()
        self.assertIsNone(SelfAttentionValidator.validate(model))

    def test_fused_strategies_reject_recurrent_projection_model(self):
        cases = (
            (SelfAttentionProjectionStrategy.FUSED, 3),
            (SelfAttentionProjectionStrategy.FUSED_KEY_VALUE, 2),
        )
        for projection_strategy, output_multiplier in cases:
            with self.subTest(projection_strategy=projection_strategy):
                cfg = build_attention_config(
                    SelfAttentionConfig,
                    embedding_dim=EMBEDDING_DIM,
                    query_key_projection_dim=EMBEDDING_DIM,
                    value_projection_dim=EMBEDDING_DIM,
                    projection_kind="recurrent",
                    self_attention_projection_strategy=projection_strategy,
                )

                with self.assertRaises(ValueError) as caught:
                    cfg.build()
                self.assertEqual(
                    str(caught.exception),
                    "Self-attention with RecurrentLayerConfig requires "
                    "projection_strategy=SelfAttentionProjectionStrategy.SEPARATE; "
                    f"the {projection_strategy.name} strategy changes embedding_dim "
                    f"to {output_multiplier} * embedding_dim.",
                )

    def test_dimensions_equal_raises_for_unequal(self):
        model = SimpleNamespace(
            embedding_dim=12, query_key_projection_dim=16, value_projection_dim=12
        )
        with self.assertRaises(RuntimeError) as caught:
            SelfAttentionValidator.validate_self_attention_dimensions_equal(model)
        self.assertEqual(
            str(caught.exception),
            "Self attention requires query_key_projection_dim, "
            "value_projection_dim, and embedding_dim to be equal, but got "
            "query_key_projection_dim=16, value_projection_dim=12, "
            "embedding_dim=12.",
        )

    def test_query_key_value_same_tensor_passes(self):
        tensor = torch.randn(TARGET_SEQUENCE_LENGTH, BATCH_SIZE, EMBEDDING_DIM)
        self.assertIsNone(
            SelfAttentionValidator.validate_query_key_value_are_same_tensor(
                tensor, tensor, tensor
            )
        )

    def test_query_may_differ_when_key_and_value_share_context(self):
        query = torch.randn(TARGET_SEQUENCE_LENGTH, BATCH_SIZE, EMBEDDING_DIM)
        key = torch.randn(TARGET_SEQUENCE_LENGTH, BATCH_SIZE, EMBEDDING_DIM)

        self.assertIsNone(
            SelfAttentionValidator.validate_query_key_value_are_same_tensor(
                query, key, key
            )
        )

    def test_fused_strategy_rejects_a_distinct_query_and_context(self):
        model = build_attention_config(
            SelfAttentionConfig,
            embedding_dim=EMBEDDING_DIM,
            query_key_projection_dim=EMBEDDING_DIM,
            value_projection_dim=EMBEDDING_DIM,
            self_attention_projection_strategy=SelfAttentionProjectionStrategy.FUSED,
        ).build()
        query = torch.randn(TARGET_SEQUENCE_LENGTH, BATCH_SIZE, EMBEDDING_DIM)
        context = torch.randn(TARGET_SEQUENCE_LENGTH, BATCH_SIZE, EMBEDDING_DIM)

        with self.assertRaises(RuntimeError) as caught:
            model(query, context, context)

        self.assertEqual(
            str(caught.exception),
            "SelfAttentionProjectionStrategy.FUSED requires query, key, and value "
            "to be the same tensor; use FUSED_KEY_VALUE when query and context "
            "differ.",
        )

    def test_fused_key_value_strategy_accepts_a_distinct_query_and_context(self):
        model = build_attention_config(
            SelfAttentionConfig,
            batch_size=BATCH_SIZE,
            embedding_dim=EMBEDDING_DIM,
            query_key_projection_dim=EMBEDDING_DIM,
            value_projection_dim=EMBEDDING_DIM,
            target_sequence_length=TARGET_SEQUENCE_LENGTH,
            source_sequence_length=SOURCE_SEQUENCE_LENGTH,
            self_attention_projection_strategy=(
                SelfAttentionProjectionStrategy.FUSED_KEY_VALUE
            ),
        ).build()
        query = torch.randn(TARGET_SEQUENCE_LENGTH, BATCH_SIZE, EMBEDDING_DIM)
        context = torch.randn(SOURCE_SEQUENCE_LENGTH, BATCH_SIZE, EMBEDDING_DIM)

        output, attention_weights, auxiliary_loss = model(query, context, context)

        self.assertEqual(output.shape, query.shape)
        self.assertIsNone(attention_weights)
        self.assertIsNone(auxiliary_loss)

    def test_distinct_key_and_value_raise(self):
        query = torch.randn(TARGET_SEQUENCE_LENGTH, BATCH_SIZE, EMBEDDING_DIM)
        key = torch.randn(TARGET_SEQUENCE_LENGTH, BATCH_SIZE, EMBEDDING_DIM)
        value = torch.randn(TARGET_SEQUENCE_LENGTH, BATCH_SIZE, EMBEDDING_DIM)
        with self.assertRaises(RuntimeError) as caught:
            SelfAttentionValidator.validate_query_key_value_are_same_tensor(
                query, key, value
            )
        self.assertEqual(
            str(caught.exception),
            "Self attention requires the key and value to be the same tensor.",
        )


class TestIndependentAttentionValidator(unittest.TestCase):
    def test_forward_inputs_raise_when_returning_weights(self):
        model = build_attention_config(
            IndependentAttentionConfig,
            return_attention_weights_flag=True,
        ).build()
        query = torch.randn(TARGET_SEQUENCE_LENGTH, BATCH_SIZE, EMBEDDING_DIM)
        key = torch.randn(SOURCE_SEQUENCE_LENGTH, BATCH_SIZE, EMBEDDING_DIM)
        value = torch.randn(SOURCE_SEQUENCE_LENGTH, BATCH_SIZE, EMBEDDING_DIM)
        with self.assertRaises(RuntimeError):
            IndependentAttentionValidator.validate_forward_inputs(
                model,
                QKV(query=query, key=key, value=value),
                AttentionMasks(),
            )


class TestMixtureOfAttentionHeadsValidator(unittest.TestCase):
    def config(self):
        return build_attention_config(
            MixtureOfAttentionHeadsConfig,
            embedding_dim=EMBEDDING_DIM,
            query_key_projection_dim=EMBEDDING_DIM,
            value_projection_dim=EMBEDDING_DIM,
        )

    def test_validate_passes_with_experts_config(self):
        model = build_attention_config(
            MixtureOfAttentionHeadsConfig,
            embedding_dim=EMBEDDING_DIM,
            query_key_projection_dim=EMBEDDING_DIM,
            value_projection_dim=EMBEDDING_DIM,
        ).build()
        self.assertIsNone(MixtureOfAttentionHeadsValidator.validate(model))

    def test_validate_experts_configuration_raises_when_missing(self):
        model = SimpleNamespace(
            cfg=SimpleNamespace(experts_config=None, use_kv_expert_models_flag=False)
        )
        with self.assertRaises(ValueError) as caught:
            MixtureOfAttentionHeadsValidator.validate_experts_configuration(model)
        self.assertEqual(
            str(caught.exception),
            "experts_config is required for mixture of attention heads.",
        )

    def test_validate_experts_configuration_rejects_wrong_nested_types(self):
        cases = (
            (
                "experts_config",
                lambda cfg: setattr(cfg, "experts_config", object()),
                TypeError,
                "experts_config must be a MixtureOfExpertsConfig, received object.",
            ),
            (
                "missing_kv_flag",
                lambda cfg: setattr(cfg, "use_kv_expert_models_flag", None),
                ValueError,
                "use_kv_expert_models_flag is required for mixture of attention heads.",
            ),
            (
                "wrong_kv_flag",
                lambda cfg: setattr(cfg, "use_kv_expert_models_flag", "yes"),
                TypeError,
                "use_kv_expert_models_flag must be a bool, received str.",
            ),
            (
                "sampler_config",
                lambda cfg: setattr(cfg.experts_config, "sampler_config", object()),
                TypeError,
                "experts_config.sampler_config must be a SamplerConfig, received "
                "object.",
            ),
            (
                "router_config",
                lambda cfg: setattr(
                    cfg.experts_config.sampler_config,
                    "router_config",
                    object(),
                ),
                TypeError,
                "experts_config.sampler_config.router_config must be a RouterConfig, "
                "received object.",
            ),
        )

        for name, mutate, error_type, message in cases:
            with self.subTest(name=name):
                cfg = self.config()
                mutate(cfg)
                with self.assertRaises(error_type) as caught:
                    MixtureOfAttentionHeadsValidator.validate_experts_configuration(
                        SimpleNamespace(cfg=cfg)
                    )
                self.assertEqual(str(caught.exception), message)

    def test_validate_experts_configuration_rejects_exact_dimension_mismatches(self):
        cases = (
            (
                "top_k",
                lambda cfg: setattr(cfg.experts_config.sampler_config, "top_k", 2),
                "experts_config.top_k must match "
                "experts_config.sampler_config.top_k, got 3 and 2.",
            ),
            (
                "sampler_num_experts",
                lambda cfg: setattr(
                    cfg.experts_config.sampler_config,
                    "num_experts",
                    5,
                ),
                "experts_config.num_experts must match "
                "experts_config.sampler_config.num_experts, got 6 and 5.",
            ),
            (
                "router_num_experts",
                lambda cfg: setattr(
                    cfg.experts_config.sampler_config.router_config,
                    "num_experts",
                    5,
                ),
                "experts_config.num_experts must match experts_config.sampler_config."
                "router_config.num_experts, got 6 and 5.",
            ),
        )

        for name, mutate, message in cases:
            with self.subTest(name=name):
                cfg = self.config()
                mutate(cfg)
                with self.assertRaises(ValueError) as caught:
                    MixtureOfAttentionHeadsValidator.validate_experts_configuration(
                        SimpleNamespace(cfg=cfg)
                    )
                self.assertEqual(str(caught.exception), message)

    def test_validate_experts_configuration_rejects_dense_routing_exactly(self):
        cfg = build_attention_config(
            MixtureOfAttentionHeadsConfig,
            experts_top_k=6,
            experts_num_experts=6,
        )

        with self.assertRaises(ValueError) as caught:
            MixtureOfAttentionHeadsValidator.validate_experts_configuration(
                SimpleNamespace(cfg=cfg)
            )

        self.assertEqual(
            str(caught.exception),
            "MixtureOfAttentionHeads requires sparse indexed routing, so "
            "experts_config.top_k must be less than "
            "experts_config.num_experts; dense routing is not supported.",
        )

    def test_expert_sequence_length_error_is_exact(self):
        model = SimpleNamespace(
            cfg=SimpleNamespace(use_kv_expert_models_flag=True),
            target_sequence_length=3,
            source_sequence_length=4,
        )

        with self.assertRaises(ValueError) as caught:
            MixtureOfAttentionHeadsValidator.validate_expert_key_value_sequence_lengths(
                model
            )

        self.assertEqual(
            str(caught.exception),
            "target_sequence_length and source_sequence_length must be equal "
            "when use_kv_expert_models_flag is True, got "
            "target_sequence_length=3 and source_sequence_length=4.",
        )

    def test_attention_weights_error_is_exact(self):
        model = SimpleNamespace(return_attention_weights_flag=True)

        with self.assertRaises(RuntimeError) as caught:
            MixtureOfAttentionHeadsValidator.validate_attention_weights_are_not_requested(
                model
            )

        self.assertEqual(
            str(caught.exception),
            "MixtureOfAttentionHeads does not support returning attention_weights; "
            "set return_attention_weights_flag to False.",
        )

    def test_expert_key_value_inputs_require_each_identity_relation(self):
        model = SimpleNamespace(cfg=SimpleNamespace(use_kv_expert_models_flag=True))
        shared = torch.empty(1)
        distinct = torch.empty(1)

        for name, query, key, value in (
            ("value", shared, shared, distinct),
            ("query", distinct, shared, shared),
        ):
            with self.subTest(name=name):
                with self.assertRaises(ValueError) as caught:
                    MixtureOfAttentionHeadsValidator.validate_expert_key_value_inputs(
                        model,
                        query,
                        key,
                        value,
                    )
                self.assertEqual(
                    str(caught.exception),
                    "query, key, and value must be the same tensor when "
                    "use_kv_expert_models_flag is True.",
                )

    def test_static_inputs_are_independently_rejected_for_expert_key_values(self):
        model = SimpleNamespace(cfg=SimpleNamespace(use_kv_expert_models_flag=True))
        static = torch.empty(1, 1, 1)

        for name, static_keys, static_values in (
            ("keys", static, None),
            ("values", None, static),
        ):
            with self.subTest(name=name):
                with self.assertRaises(ValueError) as caught:
                    MixtureOfAttentionHeadsValidator.validate_static_key_value_inputs(
                        model,
                        object(),
                        static_keys,
                        static_values,
                    )
                self.assertEqual(
                    str(caught.exception),
                    "static key/value projections are not supported when "
                    "use_kv_expert_models_flag is True.",
                )
