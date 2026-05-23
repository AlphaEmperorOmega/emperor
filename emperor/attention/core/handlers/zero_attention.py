import torch
import torch.nn.functional as F

from torch import Tensor

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from emperor.attention.core.config import MultiHeadAttentionConfig


class ZeroAttention:
    def __init__(self, cfg: "MultiHeadAttentionConfig"):
        self.cfg = cfg
        self.batch_size = self.cfg.batch_size
        self.num_heads = self.cfg.num_heads
        self.zero_attention_flag = self.cfg.zero_attention_flag

    def add_zero_attention(
        self,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Tensor | None = None,
        attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor | None, Tensor | None]:
        if not self.zero_attention_flag:
            return key, value, key_padding_mask, attention_mask

        padded_key = self.__concatenate_zeros_tensor(key)
        padded_value = self.__concatenate_zeros_tensor(value)
        if key_padding_mask is not None:
            key_padding_mask = F.pad(key_padding_mask, (0, 1))
        if attention_mask is not None:
            attention_mask = F.pad(attention_mask, (0, 1))

        return padded_key, padded_value, key_padding_mask, attention_mask

    def __concatenate_zeros_tensor(self, tensor: Tensor) -> Tensor:
        head_dim = tensor.size(-1)
        zero_attention_shape = (self.batch_size * self.num_heads, 1, head_dim)
        zeros_tensor = tensor.new_zeros(zero_attention_shape)
        return torch.cat([tensor, zeros_tensor], dim=1)
