import torch
import unittest

from types import SimpleNamespace
from emperor.attention import (
    SelfAttentionConfig,
    SelfAttentionProjectionStrategy,
    IndependentAttentionConfig,
    MixtureOfAttentionHeadsConfig,
)
from emperor.attention.core._validator import AttentionValidatorBase
from emperor.attention.core.variants.self_attention.validator import SelfAttentionValidator
from emperor.attention.core.variants.independent_attention.validator import (
    IndependentAttentionValidator,
)
from emperor.attention.core.variants.mixture_of_attention_heads.validator import (
    MixtureOfAttentionHeadsValidator,
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
            AttentionValidatorBase.validate_input_shapes(
                query, key, value, None, two_d
            )
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
            batch_size=BATCH_SIZE, num_heads=NUM_HEADS, head_dim=HEAD_DIM
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
        with self.assertRaises(ValueError):
            AttentionValidatorBase.validate_static_projection_shapes(
                self.model, wrong, None
            )


class TestValidateHeadDivisibility(unittest.TestCase):
    def test_passes_when_divisible(self):
        model = SimpleNamespace(
            embedding_dim=12,
            num_heads=4,
            query_key_projection_dim=0,
            value_projection_dim=0,
        )
        self.assertIsNone(
            AttentionValidatorBase.validate_head_divisibility(model)
        )

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

    def test_fused_strategy_rejects_recurrent_projection_model(self):
        cfg = build_attention_config(
            SelfAttentionConfig,
            embedding_dim=EMBEDDING_DIM,
            query_key_projection_dim=EMBEDDING_DIM,
            value_projection_dim=EMBEDDING_DIM,
            projection_kind="recurrent",
            self_attention_projection_strategy=(
                SelfAttentionProjectionStrategy.FUSED
            ),
        )

        with self.assertRaisesRegex(
            ValueError,
            "requires projection_strategy=.*SEPARATE",
        ):
            cfg.build()

    def test_dimensions_equal_raises_for_unequal(self):
        model = SimpleNamespace(
            embedding_dim=12, query_key_projection_dim=16, value_projection_dim=12
        )
        with self.assertRaises(RuntimeError):
            SelfAttentionValidator.validate_self_attention_dimensions_equal(model)

    def test_query_key_value_same_tensor_passes(self):
        tensor = torch.randn(TARGET_SEQUENCE_LENGTH, BATCH_SIZE, EMBEDDING_DIM)
        self.assertIsNone(
            SelfAttentionValidator.validate_query_key_value_are_same_tensor(
                tensor, tensor, tensor
            )
        )

    def test_query_key_value_distinct_raises(self):
        query = torch.randn(TARGET_SEQUENCE_LENGTH, BATCH_SIZE, EMBEDDING_DIM)
        key = torch.randn(TARGET_SEQUENCE_LENGTH, BATCH_SIZE, EMBEDDING_DIM)
        with self.assertRaises(RuntimeError):
            SelfAttentionValidator.validate_query_key_value_are_same_tensor(
                query, key, key
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
                model, query, key, value
            )


class TestMixtureOfAttentionHeadsValidator(unittest.TestCase):
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
            cfg=SimpleNamespace(
                experts_config=None, use_kv_expert_models_flag=False
            )
        )
        with self.assertRaises(ValueError):
            MixtureOfAttentionHeadsValidator.validate_experts_configuration(model)
