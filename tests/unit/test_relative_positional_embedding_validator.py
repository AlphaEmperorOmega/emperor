import unittest

import torch
from emperor.embedding.relative.core._validator import (
    RelativePositionalEmbeddingValidator,
)
from emperor.embedding.relative.core.config import DynamicPositionalBiasConfig
from emperor.embedding.relative.core.layers import DynamicPositionalBias


def make_config(**overrides) -> DynamicPositionalBiasConfig:
    values = {
        "text_processing_flag": True,
        "num_heads": 2,
        "num_embeddings": 8,
        "embedding_dim": 4,
        "init_size": 8,
        "padding_idx": None,
        "auto_expand_flag": False,
        "max_positions": 4,
    }
    values.update(overrides)
    return DynamicPositionalBiasConfig(**values)


class TestRelativePositionalEmbeddingValidatorAdapter(unittest.TestCase):
    def test_module_exposes_validator_adapter(self):
        self.assertIs(
            DynamicPositionalBias.VALIDATOR,
            RelativePositionalEmbeddingValidator,
        )

    def test_construction_dispatches_through_substituted_validator(self):
        validated_padding_indices = []

        class TrackingValidator(RelativePositionalEmbeddingValidator):
            @staticmethod
            def _validate_padding_idx(padding_idx):
                validated_padding_indices.append(padding_idx)

        class TrackingDynamicPositionalBias(DynamicPositionalBias):
            VALIDATOR = TrackingValidator

        model = TrackingDynamicPositionalBias(make_config(padding_idx=0))

        self.assertEqual(validated_padding_indices, [0])
        self.assertIs(model.VALIDATOR, TrackingValidator)

    def test_forward_dispatches_through_substituted_validator(self):
        class RejectingValidator(RelativePositionalEmbeddingValidator):
            @staticmethod
            def validate_forward_inputs(query, sequence_length):
                raise RuntimeError("substituted runtime validator was called")

        class RejectingDynamicPositionalBias(DynamicPositionalBias):
            VALIDATOR = RejectingValidator

        model = RejectingDynamicPositionalBias(make_config())

        with self.assertRaisesRegex(
            RuntimeError, "substituted runtime validator was called"
        ):
            model(torch.randn(1, 2, 3, 2), 3)

    def test_divisibility_error_contract_is_preserved(self):
        with self.assertRaisesRegex(
            ValueError,
            "embedding_dim must be divisible by num_heads, received "
            "embedding_dim=5, num_heads=2",
        ):
            DynamicPositionalBias(make_config(embedding_dim=5))


if __name__ == "__main__":
    unittest.main()
