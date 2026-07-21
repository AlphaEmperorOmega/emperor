from __future__ import annotations

import unittest

import torch

from emperor.embedding.relative import (
    DynamicPositionalBiasConfig,
)
from emperor.embedding.relative._bias import DynamicPositionalBias
from emperor.embedding.relative._validation import (
    RelativePositionalEmbeddingValidator,
)


def make_config(**overrides: object) -> DynamicPositionalBiasConfig:
    values: dict[str, object] = {
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


class RelativePositionalEmbeddingValidationTests(unittest.TestCase):
    def test_module_uses_the_public_validator(self) -> None:
        self.assertIs(
            DynamicPositionalBias.VALIDATOR,
            RelativePositionalEmbeddingValidator,
        )

    def test_every_non_optional_configuration_field_is_required(self) -> None:
        for field_name in (
            "text_processing_flag",
            "num_heads",
            "num_embeddings",
            "embedding_dim",
            "init_size",
            "auto_expand_flag",
            "max_positions",
        ):
            with self.subTest(field_name=field_name):
                with self.assertRaises(ValueError) as error:
                    DynamicPositionalBias(make_config(**{field_name: None}))
                self.assertEqual(
                    str(error.exception),
                    f"{field_name} is required for "
                    "DynamicPositionalBiasConfig, received None",
                )

        model = DynamicPositionalBias(make_config(padding_idx=None))
        self.assertIsNone(model.padding_idx)
        self.assertEqual(model.num_embeddings, 8)

    def test_configuration_rejects_exact_wrong_types(self) -> None:
        invalid_values = {
            "text_processing_flag": 1,
            "num_heads": False,
            "num_embeddings": 8.0,
            "embedding_dim": "4",
            "init_size": 8.0,
            "padding_idx": 0.0,
            "auto_expand_flag": 0,
            "max_positions": 4.0,
        }
        expected_types = {
            "text_processing_flag": "bool",
            "num_heads": "int",
            "num_embeddings": "int",
            "embedding_dim": "int",
            "init_size": "int",
            "padding_idx": "int",
            "auto_expand_flag": "bool",
            "max_positions": "int",
        }

        for field_name, value in invalid_values.items():
            with self.subTest(field_name=field_name):
                with self.assertRaises(TypeError) as error:
                    DynamicPositionalBias(make_config(**{field_name: value}))
                self.assertEqual(
                    str(error.exception),
                    f"{field_name} must be {expected_types[field_name]} for "
                    "DynamicPositionalBiasConfig, "
                    f"got {type(value).__name__}",
                )

    def test_numeric_boundaries_padding_and_head_divisibility(self) -> None:
        valid = DynamicPositionalBias(
            make_config(
                num_heads=1,
                num_embeddings=1,
                embedding_dim=1,
                init_size=1,
                padding_idx=0,
                max_positions=1,
            )
        )
        self.assertEqual(
            valid.relative_positional_embeddings.shape,
            (1, 1, 3),
        )
        self.assertEqual(valid.num_embeddings, 2)

        invalid_cases = (
            (
                "num_heads",
                0,
                "num_heads must be greater than 0, received 0",
            ),
            (
                "num_embeddings",
                -1,
                "num_embeddings must be greater than 0, received -1",
            ),
            (
                "embedding_dim",
                0,
                "embedding_dim must be greater than 0, received 0",
            ),
            (
                "init_size",
                0,
                "init_size must be greater than 0, received 0",
            ),
            (
                "max_positions",
                0,
                "max_positions must be greater than 0, received 0",
            ),
            (
                "padding_idx",
                -1,
                "padding_idx must be >= 0 when provided, received -1",
            ),
        )
        for field_name, value, message in invalid_cases:
            with self.subTest(field_name=field_name):
                with self.assertRaises(ValueError) as error:
                    DynamicPositionalBias(make_config(**{field_name: value}))
                self.assertEqual(str(error.exception), message)

        with self.assertRaises(ValueError) as divisibility_error:
            DynamicPositionalBias(make_config(num_heads=3, embedding_dim=4))
        self.assertEqual(
            str(divisibility_error.exception),
            "embedding_dim must be divisible by num_heads, received "
            "embedding_dim=4, num_heads=3",
        )

    def test_forward_rejects_non_tensor_rank_and_nonpositive_source_length(
        self,
    ) -> None:
        model = DynamicPositionalBias(make_config())

        with self.assertRaises(TypeError) as type_error:
            model([1.0], sequence_length=1)
        self.assertEqual(
            str(type_error.exception),
            "query must be a Tensor, got list",
        )

        for invalid in (
            torch.ones(2, 3, 2),
            torch.ones(1, 2, 3, 2, 1),
        ):
            with self.subTest(shape=tuple(invalid.shape)):
                with self.assertRaises(ValueError) as rank_error:
                    model(invalid, sequence_length=3)
                self.assertEqual(
                    str(rank_error.exception),
                    "query must be a 4D tensor "
                    "(batch, heads, sequence, head_dim), "
                    f"got shape {tuple(invalid.shape)}",
                )

        with self.assertRaises(ValueError) as length_error:
            model(torch.ones(1, 2, 3, 2), sequence_length=0)
        self.assertEqual(
            str(length_error.exception),
            "sequence_length must be greater than 0, received 0",
        )

    def test_forward_validator_default_is_full_sequence_mode(self) -> None:
        result = RelativePositionalEmbeddingValidator.validate_forward_inputs(
            torch.ones(1, 2, 3, 2),
            3,
            num_heads=2,
            head_dim=2,
        )

        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
