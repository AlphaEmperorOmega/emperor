from __future__ import annotations

import unittest

import torch

from emperor.embedding.absolute import (
    AbsolutePositionalEmbeddingConfig,
    ImageLearnedPositionalEmbeddingConfig,
    ImageSinusoidalPositionalEmbeddingConfig,
    TextLearnedPositionalEmbeddingConfig,
)
from emperor.embedding.absolute._base import AbsolutePositionalEmbeddingBase
from emperor.embedding.absolute._learned import (
    ImageLearnedPositionalEmbedding,
    LearnedPositionalEmbedding,
    TextLearnedPositionalEmbedding,
)
from emperor.embedding.absolute._sinusoidal import (
    ImageSinusoidalPositionalEmbedding,
    SinusoidalPositionalEmbedding,
    TextSinusoidalPositionalEmbedding,
)
from emperor.embedding.absolute._validation import (
    AbsolutePositionalEmbeddingValidator,
)


def make_text_config(**overrides: object) -> TextLearnedPositionalEmbeddingConfig:
    values: dict[str, object] = {
        "num_embeddings": 8,
        "embedding_dim": 4,
        "init_size": 8,
        "padding_idx": None,
        "auto_expand_flag": False,
    }
    values.update(overrides)
    return TextLearnedPositionalEmbeddingConfig(**values)


def make_image_config(
    config_type: type[
        ImageLearnedPositionalEmbeddingConfig | ImageSinusoidalPositionalEmbeddingConfig
    ],
    **overrides: object,
) -> ImageLearnedPositionalEmbeddingConfig | ImageSinusoidalPositionalEmbeddingConfig:
    values: dict[str, object] = {
        "num_embeddings": 4,
        "embedding_dim": 4,
        "init_size": 4,
        "padding_idx": (
            None if config_type is ImageSinusoidalPositionalEmbeddingConfig else 0
        ),
        "auto_expand_flag": False,
        "class_token_flag": True,
    }
    values.update(overrides)
    return config_type(**values)


class AbsolutePositionalEmbeddingValidationTests(unittest.TestCase):
    def test_all_embedding_modules_use_the_public_validator(self) -> None:
        module_types = (
            AbsolutePositionalEmbeddingBase,
            LearnedPositionalEmbedding,
            TextLearnedPositionalEmbedding,
            ImageLearnedPositionalEmbedding,
            SinusoidalPositionalEmbedding,
            TextSinusoidalPositionalEmbedding,
            ImageSinusoidalPositionalEmbedding,
        )

        for module_type in module_types:
            with self.subTest(module_type=module_type.__name__):
                self.assertIs(
                    module_type.VALIDATOR,
                    AbsolutePositionalEmbeddingValidator,
                )

    def test_every_non_optional_text_configuration_field_is_required(
        self,
    ) -> None:
        for field_name in (
            "num_embeddings",
            "embedding_dim",
            "init_size",
            "auto_expand_flag",
        ):
            with self.subTest(field_name=field_name):
                with self.assertRaises(ValueError) as error:
                    TextLearnedPositionalEmbedding(
                        make_text_config(**{field_name: None})
                    )
                self.assertEqual(
                    str(error.exception),
                    f"{field_name} is required for "
                    "TextLearnedPositionalEmbeddingConfig, received None",
                )

        model = TextLearnedPositionalEmbedding(make_text_config(padding_idx=None))
        self.assertIsNone(model.padding_idx)

    def test_text_configuration_rejects_exact_wrong_types(self) -> None:
        invalid_values = {
            "num_embeddings": False,
            "embedding_dim": 4.0,
            "init_size": "8",
            "padding_idx": 0.0,
            "auto_expand_flag": 1,
        }
        expected_types = {
            "num_embeddings": "int",
            "embedding_dim": "int",
            "init_size": "int",
            "padding_idx": "int",
            "auto_expand_flag": "bool",
        }

        for field_name, value in invalid_values.items():
            with self.subTest(field_name=field_name):
                with self.assertRaises(TypeError) as error:
                    TextLearnedPositionalEmbedding(
                        make_text_config(**{field_name: value})
                    )
                self.assertEqual(
                    str(error.exception),
                    f"{field_name} must be {expected_types[field_name]} for "
                    "TextLearnedPositionalEmbeddingConfig, "
                    f"got {type(value).__name__}",
                )

    def test_numeric_boundaries_and_padding_validation(self) -> None:
        valid = TextLearnedPositionalEmbedding(
            make_text_config(
                num_embeddings=1,
                embedding_dim=1,
                init_size=1,
                padding_idx=0,
            )
        )
        self.assertEqual(valid.embedding_model.weight.shape, (2, 1))

        invalid_cases = (
            (
                "num_embeddings",
                0,
                "num_embeddings must be greater than 0, received 0",
            ),
            (
                "embedding_dim",
                -1,
                "embedding_dim must be greater than 0, received -1",
            ),
            (
                "init_size",
                0,
                "init_size must be greater than 0, received 0",
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
                    TextLearnedPositionalEmbedding(
                        make_text_config(**{field_name: value})
                    )
                self.assertEqual(str(error.exception), message)

    def test_fixed_learned_tables_reject_out_of_range_padding_indices(
        self,
    ) -> None:
        invalid_cases = (
            (
                lambda: LearnedPositionalEmbedding(
                    AbsolutePositionalEmbeddingConfig(
                        num_embeddings=3,
                        embedding_dim=2,
                        init_size=3,
                        padding_idx=3,
                        auto_expand_flag=False,
                    )
                ),
                "padding_idx must be in [0, 3) for "
                "LearnedPositionalEmbedding, received 3",
            ),
            (
                lambda: ImageLearnedPositionalEmbedding(
                    make_image_config(
                        ImageLearnedPositionalEmbeddingConfig,
                        num_embeddings=3,
                        class_token_flag=False,
                        padding_idx=3,
                    )
                ),
                "padding_idx must be in [0, 3) for "
                "ImageLearnedPositionalEmbedding, received 3",
            ),
            (
                lambda: ImageLearnedPositionalEmbedding(
                    make_image_config(
                        ImageLearnedPositionalEmbeddingConfig,
                        num_embeddings=3,
                        class_token_flag=True,
                        padding_idx=4,
                    )
                ),
                "padding_idx must be in [0, 4) for "
                "ImageLearnedPositionalEmbedding, received 4",
            ),
        )

        for build, message in invalid_cases:
            with self.subTest(message=message):
                with self.assertRaises(ValueError) as error:
                    build()
                self.assertEqual(str(error.exception), message)

    def test_image_class_token_field_is_required_and_boolean(self) -> None:
        for config_type, model_type in (
            (
                ImageLearnedPositionalEmbeddingConfig,
                ImageLearnedPositionalEmbedding,
            ),
            (
                ImageSinusoidalPositionalEmbeddingConfig,
                ImageSinusoidalPositionalEmbedding,
            ),
        ):
            with self.subTest(config_type=config_type.__name__, value=None):
                with self.assertRaises(ValueError) as missing_error:
                    model_type(
                        make_image_config(
                            config_type,
                            class_token_flag=None,
                        )
                    )
                self.assertEqual(
                    str(missing_error.exception),
                    f"class_token_flag is required for "
                    f"{config_type.__name__}, received None",
                )

            with self.subTest(config_type=config_type.__name__, value=1):
                with self.assertRaises(TypeError) as type_error:
                    model_type(
                        make_image_config(
                            config_type,
                            class_token_flag=1,
                        )
                    )
                self.assertEqual(
                    str(type_error.exception),
                    f"class_token_flag must be bool for "
                    f"{config_type.__name__}, got int",
                )

    def test_image_sinusoidal_rejects_numeric_padding_indices(self) -> None:
        for padding_idx in (0, 2):
            with self.subTest(padding_idx=padding_idx):
                with self.assertRaises(ValueError) as error:
                    ImageSinusoidalPositionalEmbedding(
                        make_image_config(
                            ImageSinusoidalPositionalEmbeddingConfig,
                            padding_idx=padding_idx,
                        )
                    )
                self.assertEqual(
                    str(error.exception),
                    "padding_idx must be None for "
                    "ImageSinusoidalPositionalEmbeddingConfig because image patch "
                    "sequences do not contain padding tokens.",
                )

    def test_text_token_validation_rejects_type_rank_and_fractional_values(
        self,
    ) -> None:
        model = TextLearnedPositionalEmbedding(make_text_config())

        with self.assertRaises(TypeError) as type_error:
            model([[1, 2]])
        self.assertEqual(
            str(type_error.exception),
            "input_tokens must be a Tensor, got list",
        )

        for invalid in (
            torch.ones(3),
            torch.ones(1, 2, 3),
        ):
            with self.subTest(shape=tuple(invalid.shape)):
                with self.assertRaises(ValueError) as rank_error:
                    model(invalid)
                self.assertEqual(
                    str(rank_error.exception),
                    "input_tokens must be a 2D tensor, got shape "
                    f"{tuple(invalid.shape)}",
                )

        with self.assertRaises(ValueError) as fractional_error:
            model(torch.tensor([[1.0, 2.5]]))
        self.assertEqual(
            str(fractional_error.exception),
            "input_tokens must contain integer-valued positions/tokens.",
        )

    def test_patch_validation_rejects_type_and_rank(self) -> None:
        model = ImageLearnedPositionalEmbedding(
            make_image_config(
                ImageLearnedPositionalEmbeddingConfig,
                class_token_flag=False,
            )
        )

        with self.assertRaises(TypeError) as type_error:
            model([[1.0]])
        self.assertEqual(
            str(type_error.exception),
            "patch_embeddings must be a Tensor, got list",
        )

        for invalid in (
            torch.ones(4, 4),
            torch.ones(1, 2, 3, 4),
        ):
            with self.subTest(shape=tuple(invalid.shape)):
                with self.assertRaises(ValueError) as rank_error:
                    model(invalid)
                self.assertEqual(
                    str(rank_error.exception),
                    "patch_embeddings must be a 3D tensor, got shape "
                    f"{tuple(invalid.shape)}",
                )


if __name__ == "__main__":
    unittest.main()
