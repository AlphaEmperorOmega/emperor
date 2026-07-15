import unittest

import torch
from emperor.attention import SelfAttentionConfig
from emperor.attention.core.runtime import QKV, AttentionMasks
from emperor.attention.core.variants.self_attention.layer import SelfAttention
from emperor.attention.core.variants.self_attention.validator import (
    SelfAttentionValidator,
)

from support.attention import build_attention_config


class TestSelfAttentionValidatorAdapter(unittest.TestCase):
    def test_module_declares_its_variant_validator_adapter(self):
        self.assertIs(SelfAttention.VALIDATOR, SelfAttentionValidator)

    def test_construction_orchestration_dispatches_through_subclass(self):
        class TrackingValidator(SelfAttentionValidator):
            @staticmethod
            def validate_projection_strategy(model):
                raise RuntimeError("substituted construction validator was called")

        class TrackingAttention(SelfAttention):
            VALIDATOR = TrackingValidator

        with self.assertRaisesRegex(
            RuntimeError,
            "substituted construction validator was called",
        ):
            TrackingAttention(build_attention_config(SelfAttentionConfig))

    def test_runtime_orchestration_dispatches_through_subclass(self):
        class RejectingValidator(SelfAttentionValidator):
            @staticmethod
            def validate_query_key_value_are_same_tensor(query, key, value):
                raise RuntimeError("substituted runtime validator was called")

        model = SelfAttention(build_attention_config(SelfAttentionConfig))
        tensor = torch.ones(2, 1, model.embedding_dim)

        with self.assertRaisesRegex(
            RuntimeError,
            "substituted runtime validator was called",
        ):
            RejectingValidator.validate_forward_inputs(
                model,
                QKV(query=tensor, key=tensor, value=tensor),
                AttentionMasks(),
            )


if __name__ == "__main__":
    unittest.main()
