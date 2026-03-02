from emperor.base.utils import Module
from emperor.embedding.options import RelativePositionalEmbeddingOptions
from emperor.embedding.relative.config import RelativePositionalEmbeddingConfig
from emperor.embedding.relative.options.dynamic_positional_bias import (
    DynamicPostionalBias,
)


class RelativePositionalEmbeddingFactory(Module):
    def __init__(
        self,
        cfg: "RelativePositionalEmbeddingConfig",
    ):
        super().__init__()
        self.cfg = cfg
        self.positional_embedding_option = self.cfg.positional_embedding_option

    def build(self) -> Module:
        match self.positional_embedding_option:
            case RelativePositionalEmbeddingOptions.LEARNED:
                return DynamicPostionalBias(self.cfg)
            case _:
                raise ValueError(
                    "If the `positional_embedding_option` is set to `DISABLED`, this class should not be initialized"
                )
