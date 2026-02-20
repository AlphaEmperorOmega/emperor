from Emperor.base.utils import Module
from Emperor.base.enums import BaseOptions
from Emperor.transformer.utils.embedding.options.base import PositionalEmbeddingConfig
from Emperor.embedding.relative.options.learned_embedding import (
    LearnedPositionalEmbedding,
)


class RelativePositionalEmbeddingOptions(BaseOptions):
    DISABLED = 0
    LEARNED = 1


class RelativePositionalEmbeddingSelector(Module):
    def __init__(
        self,
        cfg: "PositionalEmbeddingConfig",
    ):
        super().__init__()
        self.cfg = cfg
        self.positional_embedding_option = self.cfg.positional_embedding_option

    def build(self) -> Module:
        match self.positional_embedding_option:
            case RelativePositionalEmbeddingOptions.LEARNED:
                return LearnedPositionalEmbedding(self.cfg)
            case _:
                raise ValueError(
                    "If the `relative_positional_embedding_option` is set to `DISABLED`, this class should not be initialized"
                )
