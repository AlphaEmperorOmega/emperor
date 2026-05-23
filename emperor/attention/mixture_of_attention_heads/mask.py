from torch import Tensor
from emperor.attention.core.handlers.mask import Mask

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from emperor.attention.mixture_of_attention_heads.config import (
        MixtureOfAttentionHeadsConfig,
    )


class MixtureOfAttentionHeadsMask(Mask):
    def __init__(self, cfg: "MixtureOfAttentionHeadsConfig"):
        super().__init__(cfg)
        self.top_k = self.cfg.experts_config.top_k

    def merge_padding_and_attention_mask(
        self,
        key: Tensor,
        key_padding_mask: Tensor | None = None,
        attention_mask: Tensor | None = None,
    ) -> Tensor | None:
        if key_padding_mask is None:
            return attention_mask

        source_sequence_length = key.size(1)

        shape_view = (self.batch_size, 1, 1, source_sequence_length)
        key_padding_mask = key_padding_mask.view(shape_view)

        shape_expand = (-1, self.num_heads * self.top_k, -1, -1)
        key_padding_mask = key_padding_mask.expand(shape_expand)

        batch_size = self.batch_size * self.num_heads * self.top_k
        shape_reshape = (batch_size, 1, source_sequence_length)
        key_padding_mask = key_padding_mask.reshape(shape_reshape)

        if attention_mask is None:
            return key_padding_mask
        return attention_mask + key_padding_mask
