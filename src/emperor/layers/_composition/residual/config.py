from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from emperor.config import ConfigBase, optional_field

if TYPE_CHECKING:
    from emperor.layers._options import ResidualConnectionOptions
    from emperor.linears import LinearLayerConfig


@dataclass(init=False)
class ResidualConfig(ConfigBase):
    residual_dim: int | None = optional_field(
        "Residual feature dimension. Layer owners override this value from their "
        "output dimension."
    )
    def __init__(
        self,
        residual_dim: int | None = None,
        *,
        option: ResidualConnectionOptions | None = None,
        model_config: LinearLayerConfig | None = None,
        attention_config: ConfigBase | None = None,
    ) -> None:
        self.residual_dim = residual_dim
        self.option = option
        self.model_config = model_config
        self.attention_config = attention_config
        self.__post_init__()

    def __post_init__(self) -> None:
        super().__post_init__()
        if type(self) is not ResidualConfig:
            return
        for field_name in ("option", "model_config", "attention_config"):
            value = getattr(self, field_name)
            if value is not None:
                self._passed_args[field_name] = value
        if not isinstance(
            self.attention_config,
            AttentionResidualConfig,
        ):
            return
        from emperor.layers._config import (
            AttentionResidualConfig as LegacyAttentionResidualConfig,
        )

        self.attention_config = LegacyAttentionResidualConfig(
            residual_dim=self.attention_config.residual_dim,
            block_size=self.attention_config.block_size,
            rms_norm_epsilon=self.attention_config.rms_norm_epsilon,
        )

    def _registry_owner(self) -> type:
        if getattr(self, "option", None) is not None:
            from emperor.layers._composition.residual.legacy import (
                ResidualConnection,
            )

            return ResidualConnection
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
    model_config: LinearLayerConfig | None = optional_field(
        "Optional data-dependent coefficient model. When provided, the model "
        "receives concatenated current and previous values and produces one raw "
        "mixing coefficient per feature. When omitted, a learned scalar parameter "
        "is used."
    )

    def _registry_owner(self) -> type:
        from emperor.layers._composition.residual.variants.weighted import (
            WeightedResidual,
        )

        return WeightedResidual


@dataclass
class WeightedBlendResidualConfig(ResidualConfig):
    model_config: LinearLayerConfig | None = optional_field(
        "Optional data-dependent coefficient model. When provided, the model "
        "receives concatenated current and previous values and produces one raw "
        "blend coefficient per feature. When omitted, a learned scalar parameter "
        "is used."
    )

    def _registry_owner(self) -> type:
        from emperor.layers._composition.residual.variants.weighted_blend import (
            WeightedBlendResidual,
        )

        return WeightedBlendResidual


@dataclass
class AttentionResidualConfig(ResidualConfig):
    block_size: int | None = optional_field(
        "Number of consecutive raw transformation outputs combined into one depth "
        "source. Use 1 for Full Attention Residuals and values greater than 1 for "
        "Block Attention Residuals. Defaults to 1."
    )
    rms_norm_epsilon: float | None = optional_field(
        "Numerical stability epsilon used to RMS-normalize routing keys. Defaults "
        "to 1e-6."
    )

    def _registry_owner(self) -> type:
        from emperor.layers._composition.residual.variants.attention import (
            AttentionResidual,
        )

        return AttentionResidual
