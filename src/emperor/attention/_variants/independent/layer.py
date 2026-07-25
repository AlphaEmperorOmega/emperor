"""Private independent-attention layer implementation."""

from torch import Tensor

from emperor.attention._base import MultiHeadAttentionAbstract
from emperor.attention._ops.masking import Mask
from emperor.attention._ops.reshaping import AttentionReshaper
from emperor.attention._runtime import MultiHeadAttentionInputs
from emperor.attention._variants.independent.processing import IndependentProcessor
from emperor.attention._variants.independent.projection import IndependentProjector
from emperor.attention._variants.independent.validation import (
    IndependentAttentionValidator,
)


class IndependentAttention(MultiHeadAttentionAbstract):
    VALIDATOR = IndependentAttentionValidator

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        k_padding_mask: Tensor | None = None,
        attention_mask: Tensor | None = None,
        static_k: Tensor | None = None,
        static_v: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None, Tensor | None]:
        attention_inputs = MultiHeadAttentionInputs(
            query=q,
            key=k,
            value=v,
            key_padding_mask=k_padding_mask,
            attention_mask=attention_mask,
            static_key=static_k,
            static_value=static_v,
        )
        return self._run_attention(attention_inputs)

    def _build_attention_components(self) -> None:
        self.projector = IndependentProjector(self.cfg)
        self.reshaper = AttentionReshaper(self.cfg)
        self.processor = IndependentProcessor(self.cfg, self.projector, self.reshaper)
        self.masks = Mask(self.cfg)
