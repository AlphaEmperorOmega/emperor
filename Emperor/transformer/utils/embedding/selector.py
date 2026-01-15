from Emperor.base.utils import Module
from Emperor.base.enums import BaseOptions
from Emperor.transformer.utils.embedding.options.base import PositionalEmbeddingConfig
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


class PositionalEmbeddingSelector(Module):
    def __init__(
        self,
        cfg: "PositionalEmbeddingConfig",
    ):
        super().__init__()
        self.cfg = cfg
        self.positional_embedding_option = self.cfg.positional_embedding_option

    def build(self) -> Module:
        match self.positional_embedding_option:
            case PositionalEmbeddingOptions.SINUSOIDAL:
                return SinusoidalPositionalEmbedding(self.cfg)
            case PositionalEmbeddingOptions.LEARNED:
                return LearnedPositionalEmbedding(self.cfg)
            case _:
                raise ValueError(
                    "If the `positional_embedding_option` is set to `DISABLED`, this class should not be initialized"
                )
