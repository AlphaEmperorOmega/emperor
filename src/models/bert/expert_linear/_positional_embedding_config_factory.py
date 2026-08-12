from dataclasses import dataclass, fields

import models.bert.expert_linear.config as config
from models.bert.expert_linear._config_defaults import (
    bert_positional_embedding_options,
)
from models.bert.expert_linear.runtime_options import (
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
        self.positional_embedding_options = (
            bert_positional_embedding_options(config)
            if dependencies.positional_embedding_options is None
            else dependencies.positional_embedding_options
        )

    def build_positional_embedding_config(self):
        options = self.positional_embedding_options
        positional_embedding_config = options.option
        available_values = {
            "num_embeddings": self.sequence_length,
            "embedding_dim": self.hidden_dim,
            "padding_idx": options.padding_idx,
            "auto_expand_flag": options.auto_expand_flag,
        }
        active_fields = {field.name for field in fields(positional_embedding_config)}
        return positional_embedding_config(
            **{
                name: value
                for name, value in available_values.items()
                if name in active_fields
            }
        )
