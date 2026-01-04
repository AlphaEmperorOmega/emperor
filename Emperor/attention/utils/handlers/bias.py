import torch
import torch.nn.functional as F

from torch import Tensor
from Emperor.base.utils import Module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from Emperor.attention.utils.layer import MultiHeadAttentionConfig


class KeyValueBias(Module):
    def __init__(self, cfg: "MultiHeadAttentionConfig"):
        super().__init__()
        self.cfg = cfg
        self.batch_size = self.cfg.batch_size
        self.embedding_dim = self.cfg.embedding_dim
        self.add_key_value_bias_flag = self.cfg.add_key_value_bias_flag
        self.key_bias_vector, self.value_bias_vector = self.__build_kv_bias_vectors()

    def __build_kv_bias_vectors(self):
        if not self.add_key_value_bias_flag:
            return None, None
        bias_k = self._init_parameter_bank((1, 1, self.embedding_dim))
        bias_v = self._init_parameter_bank((1, 1, self.embedding_dim))
        return bias_k, bias_v

    def add_kv_learnable_bias_vectors(
        self,
        key_projections: Tensor,
        value_projections: Tensor,
        key_padding_mask: Tensor | None = None,
        attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor | None, Tensor | None]:
        if not self.add_key_value_bias_flag:
            return (
                key_projections,
                value_projections,
                key_padding_mask,
                attention_mask,
            )
        repeated_key_bias = self.key_bias_vector.repeat(1, self.batch_size, 1)
        key_projections_with_bias_vector = torch.cat(
            [key_projections, repeated_key_bias]
        )
        repeated_value_bias = self.value_bias_vector.repeat(1, self.batch_size, 1)
        value_projections_with_bias_vector = torch.cat(
            [value_projections, repeated_value_bias]
        )
        if key_padding_mask is not None:
            key_padding_mask = F.pad(key_padding_mask, (0, 1))
        if attention_mask is not None:
            attention_mask = F.pad(attention_mask, (0, 1))

        return (
            key_projections_with_bias_vector,
            value_projections_with_bias_vector,
            key_padding_mask,
            attention_mask,
        )
