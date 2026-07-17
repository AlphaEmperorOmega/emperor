import unittest

import torch

from emperor.attention import MixtureOfAttentionHeadsConfig
from emperor.attention._runtime import QKV, AttentionMasks
from emperor.attention._variants.mixture.layer import MixtureOfAttentionHeads
from emperor.attention._variants.mixture.processing import (
    MixtureOfAttentionHeadsProcessor,
)
from emperor.attention._variants.mixture.validation import (
    MixtureOfAttentionHeadsValidator,
)
from support.attention import build_attention_config


class TestMixtureOfAttentionHeadsValidatorAdapter(unittest.TestCase):
    def test_module_declares_its_variant_validator_adapter(self):
        self.assertIs(
            MixtureOfAttentionHeads.VALIDATOR,
            MixtureOfAttentionHeadsValidator,
        )
        self.assertIs(
            MixtureOfAttentionHeadsProcessor.VALIDATOR,
            MixtureOfAttentionHeadsValidator,
        )

    def test_construction_orchestration_dispatches_through_subclass(self):
        class TrackingValidator(MixtureOfAttentionHeadsValidator):
            @staticmethod
            def validate_experts_configuration(model):
                raise RuntimeError("substituted construction validator was called")

        class TrackingAttention(MixtureOfAttentionHeads):
            VALIDATOR = TrackingValidator

        with self.assertRaisesRegex(
            RuntimeError,
            "substituted construction validator was called",
        ):
            TrackingAttention(build_attention_config(MixtureOfAttentionHeadsConfig))

    def test_runtime_orchestration_dispatches_through_subclass(self):
        class RejectingValidator(MixtureOfAttentionHeadsValidator):
            @staticmethod
            def validate_attention_weights_are_not_requested(model):
                raise RuntimeError("substituted runtime validator was called")

        model = MixtureOfAttentionHeads(
            build_attention_config(MixtureOfAttentionHeadsConfig)
        )
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
