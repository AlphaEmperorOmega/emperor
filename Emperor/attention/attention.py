import torch
import torch.nn as nn
from dataclasses import dataclass, field
from Emperor.base.utils import DataClassBase, Module, device

from Emperor.layers.utils.enums import (
    LayerTypes,
)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from Emperor.config import ModelConfig


@dataclass
class MultiHeadAttentionConfig(DataClassBase):
    model_type: LayerTypes | None = field(
        default=None,
        metadata={
            "help": "Type of layer used for to generate query, key, value projections."
        },
    )
    embedding_dim: int | None = field(
        default=None,
        metadata={"help": "Expert input dimension"},
    )
    num_heads: int | None = field(
        default=None,
        metadata={"help": "Expert input dimension"},
    )
    dropout_probability: int | None = field(
        default=None,
        metadata={"help": "Expert input dimension"},
    )
    key_value_bias_flag: bool | None = field(
        default=None,
        metadata={"help": "Expert input dimension"},
    )
    zero_attention_flag: bool | None = field(
        default=None,
        metadata={"help": "Expert input dimension"},
    )
    key_dim: int | None = field(
        default=None,
        metadata={"help": "Expert input dimension"},
    )
    value_dim: int | None = field(
        default=None,
        metadata={"help": "Expert input dimension"},
    )
    batch_first_flag: bool | None = field(
        default=None,
        metadata={"help": "Expert input dimension"},
    )
    dtype: torch.dtype | None = field(
        default=None,
        metadata={"help": "Expert input dimension"},
    )


class MultiHeadAttention(Module):
    def __init__(
        self,
        cfg: "MultiHeadAttentionConfig | ModelConfig",
        overrides: "MultiHeadAttentionConfig | None" = None,
    ):
        super().__init__()
        config = getattr(cfg, "multi_head_attention_model_config", cfg)
        self.cfg: "MultiHeadAttentionConfig" = self._overwrite_config(config, overrides)

        self.model_type = self.cfg.model_type
        self.embedding_dim = self.cfg.embedding_dim
        self.num_heads = self.cfg.num_heads
        self.dropout_probability = self.cfg.dropout_probability
        self.key_value_bias_flag = self.cfg.key_value_bias_flag
        self.zero_attention_flag = self.cfg.zero_attention_flag
        self.batch_first_flag = self.cfg.batch_first_flag
        self.model_type = self.cfg.model_type
        self.dtype = self.cfg.dtype
        self.key_dim = self.cfg.key_dim
        self.value_dim = self.cfg.value_dim
        self.query_dim = self.embedding_dim
        self.head_dim = self.embedding_dim // self.num_heads

        if self.__are_key_query_value_dims_equal():
            self.query_key_value_module = nn.Linear(
                self.embedding_dim, self.embedding_dim * 3
            )
            self.register_parameter("query_module", None)
            self.register_parameter("key_module", None)
            self.register_parameter("query_module", None)
        else:
            self.query_module = nn.Linear(self.embedding_dim, self.query_dim)
            self.key_module = nn.Linear(self.embedding_dim, self.key_dim)
            self.value_module = nn.Linear(self.embedding_dim, self.value_dim)
            self.register_parameter("query_key_value_module", None)

        assert (self.head_dim * self.num_heads) == self.embedding_dim, (
            "`embedding_dim` must be perfectly divisible by `number_of_heads`."
        )

    def __create_projection_models(self):
        if self.__are_key_query_value_dims_equal():
            query_key_value_module = nn.Linear(
                self.embedding_dim, self.embedding_dim * 3
            )
            return query_key_value_module
        query_module = nn.Linear(self.embedding_dim, self.query_dim)
        key_module = nn.Linear(self.embedding_dim, self.key_dim)
        value_module = nn.Linear(self.embedding_dim, self.value_dim)
        return query_module, key_module, value_module

    def __are_key_query_value_dims_equal(self) -> bool:
        are_keys_querys_same = self.key_dim == self.embed_dim
        are_values_querys_same = self.value_dim == self.embed_dim
        return are_keys_querys_same and are_values_querys_same
