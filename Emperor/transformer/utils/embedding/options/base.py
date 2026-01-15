from dataclasses import dataclass, field
from Emperor.base.utils import ConfigBase

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from Emperor.transformer.utils.embedding.selector import PositionalEmbeddingOptions


@dataclass
class PositionalEmbeddingConfig(ConfigBase):
    positional_embedding_option: "PositionalEmbeddingOptions | None" = field(
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
