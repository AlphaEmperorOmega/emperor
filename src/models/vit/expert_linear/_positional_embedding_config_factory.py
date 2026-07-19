from dataclasses import dataclass

import models.vit.expert_linear.config as config
from models.vit.expert_linear.runtime_options import (
    TransformerPositionalEmbeddingOptions,
)


@dataclass(frozen=True)
class PositionalEmbeddingConfigDependencies:
    hidden_dim: int
    sequence_length: int
    positional_embedding_options: TransformerPositionalEmbeddingOptions | None


class PositionalEmbeddingConfigFactory:
    def __init__(
        self,
        dependencies: PositionalEmbeddingConfigDependencies,
    ) -> None:
        self.hidden_dim = dependencies.hidden_dim
        self.sequence_length = dependencies.sequence_length
        self.positional_embedding_options = self.__default_positional_embedding_options(
            dependencies.positional_embedding_options
        )

    def __default_positional_embedding_options(
        self,
        positional_embedding_options: TransformerPositionalEmbeddingOptions | None,
    ) -> TransformerPositionalEmbeddingOptions:
        if positional_embedding_options is not None:
            return positional_embedding_options
        return TransformerPositionalEmbeddingOptions(
            option=config.POSITIONAL_EMBEDDING_OPTION,
            padding_idx=config.POSITIONAL_EMBEDDING_PADDING_IDX,
            auto_expand_flag=config.POSITIONAL_EMBEDDING_AUTO_EXPAND_FLAG,
        )

    def build_positional_embedding_config(self):
        options = self.positional_embedding_options
        positional_embedding_config = options.option
        return positional_embedding_config(
            num_embeddings=self.sequence_length - 1,
            embedding_dim=self.hidden_dim,
            init_size=self.sequence_length,
            padding_idx=options.padding_idx,
            auto_expand_flag=options.auto_expand_flag,
            class_token_flag=True,
        )
