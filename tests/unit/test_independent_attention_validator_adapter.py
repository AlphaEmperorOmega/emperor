import unittest

import torch

from emperor.attention import IndependentAttentionConfig
from emperor.attention._runtime import QKV, AttentionMasks
from emperor.attention._variants.independent.layer import IndependentAttention
from emperor.attention._variants.independent.validation import (
    IndependentAttentionValidator,
)
from support.attention import build_attention_config


class TestIndependentAttentionValidatorAdapter(unittest.TestCase):
    def test_module_declares_its_variant_validator_adapter(self):
        self.assertIs(
            IndependentAttention.VALIDATOR,
            IndependentAttentionValidator,
        )

    def test_runtime_orchestration_dispatches_through_subclass(self):
        class RejectingValidator(IndependentAttentionValidator):
            @staticmethod
            def validate_attention_weights_returned_for_self_attention_only(model):
                raise RuntimeError("substituted runtime validator was called")

        model = IndependentAttention(build_attention_config(IndependentAttentionConfig))
        query = torch.ones(2, 1, model.embedding_dim)
        key = torch.ones(3, 1, model.embedding_dim)
        value = torch.ones(3, 1, model.embedding_dim)

        with self.assertRaisesRegex(
            RuntimeError,
            "substituted runtime validator was called",
        ):
            RejectingValidator.validate_forward_inputs(
                model,
                QKV(query=query, key=key, value=value),
                AttentionMasks(),
            )


if __name__ == "__main__":
    unittest.main()
