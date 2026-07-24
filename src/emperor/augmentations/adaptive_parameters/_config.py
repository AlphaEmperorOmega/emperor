"""Private adaptive-parameter configuration implementation."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from emperor.augmentations.adaptive_parameters._options import (
    AdaptiveParameterGroupingScopeOptions,
)
from emperor.config import ConfigBase, optional_field
from emperor.linears import LinearLayerConfig

if TYPE_CHECKING:
    from emperor.augmentations.adaptive_parameters._biases.config import (
        DynamicBiasConfig,
    )
    from emperor.augmentations.adaptive_parameters._diagonals.config import (
        DynamicDiagonalConfig,
    )
    from emperor.augmentations.adaptive_parameters._masks.config import AxisMaskConfig
    from emperor.augmentations.adaptive_parameters._weights.config import (
        DynamicWeightConfig,
    )
    from emperor.layers import LayerStackConfig


@dataclass
class AdaptiveParameterAugmentationConfig(ConfigBase):
    input_dim: int | None = optional_field("Input feature dimension.")
    output_dim: int | None = optional_field("Output feature dimension.")
    diagonal_config: "DynamicDiagonalConfig | None" = optional_field(
        "Optional dynamic diagonal adjustment."
    )
    weight_config: "DynamicWeightConfig | None" = optional_field(
        "Optional dynamic weight adjustment."
    )
    bias_config: "DynamicBiasConfig | None" = optional_field(
        "Optional dynamic bias adjustment."
    )
    mask_config: "AxisMaskConfig | None" = optional_field(
        "Optional dynamic weight mask."
    )
    model_config: "LayerStackConfig | None" = optional_field(
        "Internal generator network config."
    )
    grouping_scope: AdaptiveParameterGroupingScopeOptions | None = optional_field(
        "ROWS groups one declared row axis; SEQUENCE groups positions separately "
        "inside each batch member; DISABLED preserves per-row generation. None is "
        "reserved for unresolved partial overrides and means inherit."
    )
    group_count: int | None = optional_field(
        "Number of generated adaptive parameter sets per row set or sequence sample."
    )

    def _registry_owner(self) -> type:
        from emperor.augmentations.adaptive_parameters._augmentation import (
            AdaptiveParameterAugmentation,
        )

        return AdaptiveParameterAugmentation


@dataclass
class AdaptiveLinearLayerConfig(LinearLayerConfig):
    adaptive_augmentation_config: AdaptiveParameterAugmentationConfig | None = (
        optional_field("Optional input-dependent parameter augmentation.")
    )

    def _registry_owner(self) -> type:
        from emperor.augmentations.adaptive_parameters._linear_adapter import (
            AdaptiveLinearLayer,
        )

        return AdaptiveLinearLayer
