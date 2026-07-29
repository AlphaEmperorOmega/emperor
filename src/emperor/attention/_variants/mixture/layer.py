"""Private mixture-of-attention-heads layer implementation."""

from torch import Tensor

from emperor.attention._base import MultiHeadAttentionAbstract
from emperor.attention._runtime import MultiHeadAttentionInputs
from emperor.attention._variants.mixture.bias import (
    MixtureOfAttentionHeadsKeyValueBias,
)
from emperor.attention._variants.mixture.masking import (
    MixtureOfAttentionHeadsMask,
)
from emperor.attention._variants.mixture.processing import (
    MixtureOfAttentionHeadsProcessor,
)
from emperor.attention._variants.mixture.projection import (
    MixtureOfAttentionHeadsProjector,
)
from emperor.attention._variants.mixture.reshaping import (
    MixtureOfAttentionHeadsReshaper,
)
from emperor.attention._variants.mixture.validation import (
    MixtureOfAttentionHeadsValidator,
)
from emperor.attention._variants.mixture.zero_attention import (
    MixtureOfAttentionHeadsZeroAttention,
)


class MixtureOfAttentionHeads(MultiHeadAttentionAbstract):
    VALIDATOR = MixtureOfAttentionHeadsValidator
    BIAS_HANDLER = MixtureOfAttentionHeadsKeyValueBias
    ZERO_ATTENTION_HANDLER = MixtureOfAttentionHeadsZeroAttention

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
        self.projector = MixtureOfAttentionHeadsProjector(self.cfg)
        self.reshaper = MixtureOfAttentionHeadsReshaper(self.cfg)
        self.processor = MixtureOfAttentionHeadsProcessor(
            self.cfg, self.projector, self.reshaper
        )
        self.masks = MixtureOfAttentionHeadsMask(self.cfg)
