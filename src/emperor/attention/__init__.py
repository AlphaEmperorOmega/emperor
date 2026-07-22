"""Public Interface for attention configuration and state."""

from emperor.attention._config import MultiHeadAttentionConfig
from emperor.attention._monitoring.callback import AttentionMonitorCallback
from emperor.attention._state import AttentionLayerState
from emperor.attention._variants.independent.config import (
    IndependentAttentionConfig,
)
from emperor.attention._variants.mixer.config import MixerAttentionConfig
from emperor.attention._variants.mixture.config import (
    MixtureOfAttentionHeadsConfig,
)
from emperor.attention._variants.self_attention.config import (
    SelfAttentionConfig,
    SelfAttentionProjectionStrategy,
)

__all__ = (
    "MultiHeadAttentionConfig",
    "SelfAttentionConfig",
    "SelfAttentionProjectionStrategy",
    "IndependentAttentionConfig",
    "MixtureOfAttentionHeadsConfig",
    "MixerAttentionConfig",
    "AttentionLayerState",
    "AttentionMonitorCallback",
)
