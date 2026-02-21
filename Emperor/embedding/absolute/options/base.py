from Emperor.base.utils import Module

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from Emperor.embedding.options import AbsolutePositionalEmbeddingOptions
    from Emperor.embedding.absolute.options.config import (
        AbsolutePositionalEmbeddingConfig,
    )


class AbsolutePositionalEmbeddingBase(Module):

    def __init__(
        self,
        cfg: "AbsolutePositionalEmbeddingConfig",
    ):
        super().__init__()
        self.cfg = cfg
        self.positional_embedding_option: "AbsolutePositionalEmbeddingOptions" = (
            self.cfg.positional_embedding_option
        )
        self.embedding_dim: int = self.cfg.embedding_dim
        self.padding_idx: int = self.cfg.padding_idx
        self.num_embeddings: int = self.cfg.num_embeddings
        self.init_size: int = self.cfg.init_size
        self.auto_expand_flag: bool = self.cfg.auto_expand_flag
