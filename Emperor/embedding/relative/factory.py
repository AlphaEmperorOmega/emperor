from Emperor.base.utils import Module
from Emperor.embedding.options import RelativePositionalEmbeddingOptions
from Emperor.embedding.relative.options.learned_embedding import (
    LearnedPositionalBias,
)


class RelativePositionalEmbeddingFactory(Module):
    def __init__(
        self,
        cfg: "RelativePositionalEmbeddingOptions",
    ):
        super().__init__()
        self.cfg = cfg
        self.positional_embedding_option = self.cfg.positional_embedding_option

    def build(self) -> Module:
        match self.positional_embedding_option:
            case RelativePositionalEmbeddingOptions.LEARNED:
                return LearnedPositionalBias(self.cfg)
            case _:
                raise ValueError(
                    "If the `relative_positional_embedding_option` is set to `DISABLED`, this class should not be initialized"
                )
