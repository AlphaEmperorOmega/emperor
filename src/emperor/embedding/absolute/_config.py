from dataclasses import dataclass

from emperor.config import ConfigBase, optional_field


@dataclass
class AbsolutePositionalEmbeddingConfig(ConfigBase):
    num_embeddings: int | None = optional_field(
        "Number of positional embeddings available to the module."
    )
    embedding_dim: int | None = optional_field("Embedding dimension for each position.")


@dataclass
class TextLearnedPositionalEmbeddingConfig(AbsolutePositionalEmbeddingConfig):
    padding_idx: int | None = optional_field(
        "Optional padding index whose embedding should stay zero."
    )

    def _registry_owner(self) -> type:
        from emperor.embedding.absolute._variants.learned import (
            TextLearnedPositionalEmbedding,
        )

        return TextLearnedPositionalEmbedding


@dataclass
class ImageLearnedPositionalEmbeddingConfig(AbsolutePositionalEmbeddingConfig):
    padding_idx: int | None = optional_field(
        "Optional positional index whose learned embedding should stay zero."
    )
    class_token_flag: bool | None = optional_field(
        "Whether the image patch sequence includes a class token."
    )

    def _registry_owner(self) -> type:
        from emperor.embedding.absolute._variants.learned import (
            ImageLearnedPositionalEmbedding,
        )

        return ImageLearnedPositionalEmbedding


@dataclass
class TextSinusoidalPositionalEmbeddingConfig(AbsolutePositionalEmbeddingConfig):
    padding_idx: int | None = optional_field(
        "Optional padding index whose positional values should stay zero."
    )
    auto_expand_flag: bool | None = optional_field(
        "Allow the positional table to expand when the input sequence grows."
    )

    def _registry_owner(self) -> type:
        from emperor.embedding.absolute._variants.sinusoidal import (
            TextSinusoidalPositionalEmbedding,
        )

        return TextSinusoidalPositionalEmbedding


@dataclass
class ImageSinusoidalPositionalEmbeddingConfig(AbsolutePositionalEmbeddingConfig):
    class_token_flag: bool | None = optional_field(
        "Whether the image patch sequence includes a class token."
    )

    def _registry_owner(self) -> type:
        from emperor.embedding.absolute._variants.sinusoidal import (
            ImageSinusoidalPositionalEmbedding,
        )

        return ImageSinusoidalPositionalEmbedding
