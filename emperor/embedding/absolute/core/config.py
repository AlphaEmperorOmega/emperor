from dataclasses import dataclass
from emperor.base.config import ConfigBase, optional_field


@dataclass
class AbsolutePositionalEmbeddingConfig(ConfigBase):
    num_embeddings: int | None = optional_field(
        "Number of positional embeddings available to the module."
    )
    embedding_dim: int | None = optional_field(
        "Embedding dimension for each position."
    )
    init_size: int | None = optional_field(
        "Initial size used when constructing positional tables."
    )
    padding_idx: int | None = optional_field(
        "Optional padding index whose embedding should stay zero."
    )
    auto_expand_flag: bool | None = optional_field(
        "Allow sinusoidal tables to expand when the input sequence grows."
    )


@dataclass
class TextLearnedPositionalEmbeddingConfig(AbsolutePositionalEmbeddingConfig):
    def _registry_owner(self) -> type:
        from emperor.embedding.absolute.core.layers import (
            TextLearnedPositionalEmbedding,
        )

        return TextLearnedPositionalEmbedding


@dataclass
class ImageLearnedPositionalEmbeddingConfig(AbsolutePositionalEmbeddingConfig):
    class_token_flag: bool | None = optional_field(
        "Whether the image patch sequence includes a class token."
    )

    def _registry_owner(self) -> type:
        from emperor.embedding.absolute.core.layers import (
            ImageLearnedPositionalEmbedding,
        )

        return ImageLearnedPositionalEmbedding


@dataclass
class TextSinusoidalPositionalEmbeddingConfig(AbsolutePositionalEmbeddingConfig):
    def _registry_owner(self) -> type:
        from emperor.embedding.absolute.core.layers import (
            TextSinusoidalPositionalEmbedding,
        )

        return TextSinusoidalPositionalEmbedding


@dataclass
class ImageSinusoidalPositionalEmbeddingConfig(AbsolutePositionalEmbeddingConfig):
    class_token_flag: bool | None = optional_field(
        "Whether the image patch sequence includes a class token."
    )

    def _registry_owner(self) -> type:
        from emperor.embedding.absolute.core.layers import (
            ImageSinusoidalPositionalEmbedding,
        )

        return ImageSinusoidalPositionalEmbedding
