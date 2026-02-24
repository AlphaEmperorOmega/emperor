from Emperor.base.utils import Module
from Emperor.embedding.options import AbsolutePositionalEmbeddingOptions
from Emperor.embedding.absolute.utils.options.sinusoidal_embedding import (
    TextSinusoidalPositionalEmbedding,
    ImageSinusoidalPositionalEmbedding,
)
from Emperor.embedding.absolute.utils.options.learned_embedding import (
    TextLearnedPositionalEmbedding,
    ImageLearnedPositionalEmbedding,
)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from Emperor.embedding.absolute.utils.config import (
        AbsolutePositionalEmbeddingConfig,
    )


class AbsolutePositionalEmbeddingFactory(Module):
    def __init__(
        self,
        cfg: "AbsolutePositionalEmbeddingConfig",
    ):
        super().__init__()
        self.cfg = cfg
        self.positional_embedding_option = self.cfg.positional_embedding_option
        self.text_processing_flag = self.cfg.text_processing_flag

    def build(self) -> Module:
        match self.positional_embedding_option:
            case AbsolutePositionalEmbeddingOptions.SINUSOIDAL:
                return self.__build_sinusoidal_embeddings()
            case AbsolutePositionalEmbeddingOptions.LEARNED:
                return self.__build_learned_embeddings()
            case _:
                raise ValueError(
                    "If the `positional_embedding_option` is set to `DISABLED`, this class should not be initialized"
                )

    def __build_sinusoidal_embeddings(self) -> Module:
        if self.text_processing_flag:
            return TextSinusoidalPositionalEmbedding(self.cfg)
        return ImageSinusoidalPositionalEmbedding(self.cfg)

    def __build_learned_embeddings(self) -> Module:
        if self.text_processing_flag:
            return TextLearnedPositionalEmbedding(self.cfg)
        return ImageLearnedPositionalEmbedding(self.cfg)
