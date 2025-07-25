import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from dataclasses import dataclass, field
from Emperor.base.utils import DataClassBase, Module, device

from Emperor.layers.utils.enums import (
    LayerTypes,
)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch.types import _dtype as DType
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

        temp = nn.MultiheadAttention

        self.query_key_value_module = None
        self.query_module = None
        self.key_module = None
        self.value_module = None
        self.input_tesnor_3D_flag = None
        self.__create_projection_models(cfg)
        self.__assert_input_requirements()

    def __assert_input_requirements(self):
        assert (self.head_dim * self.num_heads) == self.embedding_dim, (
            "`embedding_dim` must be perfectly divisible by `number_of_heads`."
        )

    def __are_key_query_value_dims_equal(self) -> bool:
        are_keys_querys_same = self.key_dim == self.embed_dim
        are_values_querys_same = self.value_dim == self.embed_dim
        return are_keys_querys_same and are_values_querys_same

    def __create_projection_models(self, cfg: "ModelConfig") -> None:
        self.output_dim = self.model_type.value(cfg)
        if self.__are_key_query_value_dims_equal():
            self.query_key_value_module = self.model_type.value(cfg)
            return
        self.query_module = self.model_type.value(cfg)
        self.key_module = self.model_type.value(cfg)
        self.value_module = self.model_type.value(cfg)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Tensor | None = None,
        need_weights: bool = False,
        attention_mask: Tensor | None = None,
        average_attention_weights: bool = False,
        causal_attention_mask: bool = False,
    ) -> tuple[Tensor, Tensor | None]:
        self.input_tesnor_3D_flag = query.dim() == 3
        key_padding_mask, attention_mask = self.__update_masks(
            key_padding_mask, attention_mask, query.dtype
        )
        query, key, value = self.__resolve_query_key_value_shapes(query, key, value)

        return ()

    def is_input_tensor_3D(self) -> bool:
        if self.input_tesnor_3D_flag is None:
            AssertionError(
                "Input matrix flag is not set. Call `is_input_matrix` first."
            )
        return self.input_tesnor_3D_flag

    def __update_masks(
        self,
        key_padding_mask: Tensor | None,
        attention_mask: Tensor | None,
        target_type: DType,
    ) -> tuple[Tensor | None, Tensor | None]:
        key_padding_mask = F._canonical_mask(
            mask=key_padding_mask,
            mask_name="key_padding_mask",
            other_type=F._none_or_dtype(key_padding_mask, self.dtype),
            other_name="attention_mask",
            target_type=target_type,
        )
        attention_mask = F._canonical_mask(
            mask=attention_mask,
            mask_name="attention_mask",
            other_type=None,
            other_name="",
            target_type=target_type,
            check_other=False,
        )
        return key_padding_mask, attention_mask

    def __resolve_query_key_value_shapes(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
    ):
        if self.batch_first_flag and self.is_input_tensor_3D():
            if key is value:
                if query is key:
                    key = query = value = query.transpose(0, 1)
                else:
                    query, key = (x.transpose(0, 1) for x in (query, key))
                    value = key
            else:
                query, key, value = (x.transpose(0, 1) for x in (query, key, value))
        return query, key, value
