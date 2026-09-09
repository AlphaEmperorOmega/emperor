from dataclasses import dataclass
from typing import TYPE_CHECKING

from emperor.config import ConfigBase, optional_field

if TYPE_CHECKING:
    from emperor.layers import LayerStackConfig
    from emperor.linears import LinearLayerConfig


@dataclass
class ResidualConfig(ConfigBase):
    residual_dim: int | None = optional_field(
        "Residual feature dimension. Layer owners override this value from their "
        "output dimension."
    )

    def _registry_owner(self) -> type:
        raise ValueError(
            "ResidualConfig is abstract and has no registered residual connection; "
            "instantiate a concrete residual config instead."
        )


@dataclass
class AdditiveResidualConfig(ResidualConfig):
    def _registry_owner(self) -> type:
        from emperor.layers._composition.residual.variants.additive import (
            AdditiveResidual,
        )

        return AdditiveResidual


@dataclass
class WeightedResidualConfig(ResidualConfig):
    model_config: "LayerStackConfig | LinearLayerConfig | None" = optional_field(
        "Optional data-dependent coefficient model. When provided, the model "
        "receives concatenated current and previous values and produces one raw "
        "mixing coefficient per feature. When omitted, a learned scalar parameter "
        "is used. The coefficient model retains its own parameter initialization."
    )

    def _registry_owner(self) -> type:
        from emperor.layers._composition.residual.variants.weighted import (
            WeightedResidual,
        )

        return WeightedResidual


@dataclass
class WeightedBlendResidualConfig(ResidualConfig):
    model_config: "LayerStackConfig | LinearLayerConfig | None" = optional_field(
        "Optional data-dependent coefficient model. When provided, the model "
        "receives concatenated current and previous values and produces one raw "
        "blend coefficient per feature. When omitted, a learned scalar parameter "
        "is used. The coefficient model retains its own parameter initialization."
    )

    def _registry_owner(self) -> type:
        from emperor.layers._composition.residual.variants.weighted_blend import (
            WeightedBlendResidual,
        )

        return WeightedBlendResidual


@dataclass
class AttentionResidualConfig(ResidualConfig):
    block_size: int | None = optional_field(
        "Required number of consecutive raw transformation outputs combined into one depth "
        "source. Use 1 for Full Attention Residuals and values greater than 1 for "
        "Block Attention Residuals. Suggested starting value: 1. Must be supplied "
        "explicitly; no default is applied."
    )
    rms_norm_epsilon: float | None = optional_field(
        "Required finite positive floating-point epsilon used to RMS-normalize "
        "routing keys. Suggested starting value: 1e-6. Must be supplied explicitly; "
        "no default is applied."
    )
    model_config: "LayerStackConfig | LinearLayerConfig | None" = optional_field(
        "Optional input-dependent query model. Receives the current raw "
        "transformation output and returns a query with the same shape; input "
        "and output feature dimensions are set to residual_dim. When omitted, "
        "a zero-initialized learned query vector is used. The query model retains "
        "its own parameter initialization."
    )

    def _registry_owner(self) -> type:
        from emperor.layers._composition.residual.variants.attention import (
            AttentionResidual,
        )

        return AttentionResidual
