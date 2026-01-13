from torch import Tensor
from dataclasses import dataclass, field
from Emperor.base.enums import BaseOptions
from Emperor.base.utils import ConfigBase, Module
from Emperor.transformer.utils.embedding.options.learned_embedding import (
    LearnedPositionalEmbedding,
)
from Emperor.transformer.utils.embedding.options.sinusoidal_embedding import (
    SinusoidalPositionalEmbedding,
)


class PositionalEmbeddingOptions(BaseOptions):
    DISABLED = 0
    SINUSOIDAL = 1
    LEARNED = 2


@dataclass
class PositionalEmbeddingConfig(ConfigBase):
    positional_embedding_option: PositionalEmbeddingOptions | None = field(
        default=None,
        metadata={"help": ""},
    )
    num_embeddings: int | None = field(
        default=None,
        metadata={"help": ""},
    )
    embedding_dim: int | None = field(
        default=None,
        metadata={"help": ""},
    )
    init_size: int | None = field(
        default=None,
        metadata={"help": ""},
    )
    padding_idx: int | None = field(
        default=None,
        metadata={"help": ""},
    )
    auto_expand_flag: bool | None = field(
        default=None,
        metadata={"help": ""},
    )


class PositionalEmbedding(Module):
    def __init__(
        self,
        cfg: "PositionalEmbeddingConfig",
    ):
        super().__init__()
        self.cfg = cfg
        self.positional_embedding_option = self.cfg.positional_embedding_option
        self.embedding_model = self.__init_positiona_embedding_model()

    def __init_positiona_embedding_model(self) -> Module | None:
        match self.positional_embedding_option:
            case PositionalEmbeddingOptions.SINUSOIDAL:
                return SinusoidalPositionalEmbedding(self.cfg)
            case PositionalEmbeddingOptions.LEARNED:
                return LearnedPositionalEmbedding(self.cfg)
            case _:
                raise ValueError(
                    "If the `positional_embedding_option` is set to `DISABLED`, this class should not be initialized"
                )

    def forward(
        self,
        token_embeddings: Tensor,
        incremental_state: dict[str, dict[str, Tensor | None]] | None = None,
        time_step: Tensor | None = None,
        positions: Tensor | None = None,
    ) -> Tensor:
        if self.embedding_model is None:
            return token_embeddings
        return self.embedding_model(
            token_embeddings,
            incremental_state=incremental_state,
            time_step=time_step,
            positions=positions,
        )
