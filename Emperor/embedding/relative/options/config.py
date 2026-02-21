from dataclasses import dataclass, field
from Emperor.base.utils import ConfigBase

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from Emperor.embedding.options import RelativePositionalEmbeddingOptions


@dataclass
class RelativePositionalEmbeddingConfig(ConfigBase):
    text_processing_flag: bool = field(
        default=False,
        metadata={"help": ""},
    )
    positional_embedding_option: "RelativePositionalEmbeddingOptions | None" = field(
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
