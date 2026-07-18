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


def make_text_config(**overrides) -> TextLearnedPositionalEmbeddingConfig:
    values = {
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


class TestAbsolutePositionalEmbeddingValidatorAdapter(unittest.TestCase):
    def test_embedding_modules_share_the_base_owner_adapter(self):
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

    def test_construction_dispatches_through_substituted_validator(self):
        validated_padding_indices = []

        class TrackingValidator(AbsolutePositionalEmbeddingValidator):
            @staticmethod
            def _validate_padding_idx(padding_idx):
                validated_padding_indices.append(padding_idx)

        class TrackingTextEmbedding(TextLearnedPositionalEmbedding):
            VALIDATOR = TrackingValidator

        model = TrackingTextEmbedding(make_text_config(padding_idx=0))

        self.assertEqual(validated_padding_indices, [0])
        self.assertIs(model.VALIDATOR, TrackingValidator)

    def test_forward_dispatches_through_substituted_validator(self):
        class RejectingValidator(AbsolutePositionalEmbeddingValidator):
            @staticmethod
            def validate_text_tokens(input_tokens):
                raise RuntimeError("substituted runtime validator was called")

        class RejectingTextEmbedding(TextLearnedPositionalEmbedding):
            VALIDATOR = RejectingValidator

        model = RejectingTextEmbedding(make_text_config())

        with self.assertRaisesRegex(
            RuntimeError, "substituted runtime validator was called"
        ):
            model(torch.ones(1, 2))

    def test_image_class_token_error_contract_is_preserved(self):
        cfg = ImageLearnedPositionalEmbeddingConfig(
            num_embeddings=8,
            embedding_dim=4,
            init_size=8,
            padding_idx=None,
            auto_expand_flag=False,
            class_token_flag=None,
        )

        with self.assertRaisesRegex(
            ValueError,
            "class_token_flag is required for "
            "ImageLearnedPositionalEmbeddingConfig, received None",
        ):
            ImageLearnedPositionalEmbedding(cfg)

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


if __name__ == "__main__":
    unittest.main()
