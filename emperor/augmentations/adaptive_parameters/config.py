from dataclasses import dataclass
from emperor.augmentations.adaptive_parameters.core.diagonal.config import (
    DynamicDiagonalConfig,
)
from emperor.base.layer.config import LayerStackConfig
from emperor.base.config import ConfigBase, optional_field
from emperor.augmentations.adaptive_parameters.core.bias.config import (
    DynamicBiasConfig,
)
from emperor.augmentations.adaptive_parameters.core.weight.config import (
    DynamicWeightConfig,
)
from emperor.augmentations.adaptive_parameters.core.mask.config import (
    AxisMaskConfig,
)


@dataclass
class AdaptiveParameterAugmentationConfig(ConfigBase):
    input_dim: int | None = optional_field("Input feature dimension.")
    output_dim: int | None = optional_field("Output feature dimension.")
    diagonal_config: DynamicDiagonalConfig | None = optional_field(
        "Optional dynamic diagonal adjustment."
    )
    weight_config: DynamicWeightConfig | None = optional_field(
        "Optional dynamic weight adjustment."
    )
    bias_config: DynamicBiasConfig | None = optional_field(
        "Optional dynamic bias adjustment."
    )
    mask_config: AxisMaskConfig | None = optional_field("Optional dynamic weight mask.")
    model_config: LayerStackConfig | None = optional_field(
        "Internal generator network config."
    )

    def _registry_owner(self) -> type:
        from emperor.augmentations.adaptive_parameters.model import (
            AdaptiveParameterAugmentation,
        )

        return AdaptiveParameterAugmentation
