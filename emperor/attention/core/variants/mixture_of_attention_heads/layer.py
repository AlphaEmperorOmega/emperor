from emperor.attention.core.layers import MultiHeadAttentionAbstract
from emperor.attention.core.variants.mixture_of_attention_heads.bias import (
    MixtureOfAttentionHeadsKeyValueBias,
)
from emperor.attention.core.variants.mixture_of_attention_heads.mask import (
    MixtureOfAttentionHeadsMask,
)
from emperor.attention.core.variants.mixture_of_attention_heads.monitor import (
    _MixtureOfAttentionHeadsMonitorAdapter,
)
from emperor.attention.core.variants.mixture_of_attention_heads.processor import (
    MixtureOfAttentionHeadsProcessor,
)
from emperor.attention.core.variants.mixture_of_attention_heads.projector import (
    MixtureOfAttentionHeadsProjector,
)
from emperor.attention.core.variants.mixture_of_attention_heads.reshaper import (
    MixtureOfAttentionHeadsReshaper,
)
from emperor.attention.core.variants.mixture_of_attention_heads.validator import (
    MixtureOfAttentionHeadsValidator,
)
from emperor.attention.core.variants.mixture_of_attention_heads.zero_attention import (
    MixtureOfAttentionHeadsZeroAttention,
)


class MixtureOfAttentionHeads(MultiHeadAttentionAbstract):
    VALIDATOR = MixtureOfAttentionHeadsValidator
    BIAS_HANDLER = MixtureOfAttentionHeadsKeyValueBias
    ZERO_ATTENTION_HANDLER = MixtureOfAttentionHeadsZeroAttention
    _MONITOR_ADAPTER = _MixtureOfAttentionHeadsMonitorAdapter()

    def _build_attention_components(self) -> None:
        self.projector = MixtureOfAttentionHeadsProjector(self.cfg)
        self.reshaper = MixtureOfAttentionHeadsReshaper(self.cfg)
        self.processor = MixtureOfAttentionHeadsProcessor(
            self.cfg, self.projector, self.reshaper
        )
        self.masks = MixtureOfAttentionHeadsMask(self.cfg)
