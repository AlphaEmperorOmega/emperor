"""Private mixture-of-attention-heads layer implementation."""

from emperor.attention._base import MultiHeadAttentionAbstract
from emperor.attention._variants.mixture.bias import (
    MixtureOfAttentionHeadsKeyValueBias,
)
from emperor.attention._variants.mixture.masking import (
    MixtureOfAttentionHeadsMask,
)
from emperor.attention._variants.mixture.monitoring import (
    _MixtureOfAttentionHeadsMonitorAdapter,
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
    _MONITOR_ADAPTER = _MixtureOfAttentionHeadsMonitorAdapter()

    def _build_attention_components(self) -> None:
        self.projector = MixtureOfAttentionHeadsProjector(self.cfg)
        self.reshaper = MixtureOfAttentionHeadsReshaper(self.cfg)
        self.processor = MixtureOfAttentionHeadsProcessor(
            self.cfg, self.projector, self.reshaper
        )
        self.masks = MixtureOfAttentionHeadsMask(self.cfg)
