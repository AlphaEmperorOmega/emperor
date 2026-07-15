import unittest

import torch
from emperor.embedding.absolute.core._validator import (
    AbsolutePositionalEmbeddingValidator,
)
from emperor.embedding.absolute.core.config import (
    ImageLearnedPositionalEmbeddingConfig,
    TextLearnedPositionalEmbeddingConfig,
)
from emperor.embedding.absolute.core.layers import (
    AbsolutePositionalEmbeddingBase,
    ImageLearnedPositionalEmbedding,
    ImageSinusoidalPositionalEmbedding,
    LearnedPositionalEmbedding,
    SinusoidalPositionalEmbedding,
    TextLearnedPositionalEmbedding,
    TextSinusoidalPositionalEmbedding,
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


if __name__ == "__main__":
    unittest.main()
