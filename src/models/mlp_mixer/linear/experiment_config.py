from dataclasses import dataclass, field

from emperor.config import ConfigBase, optional_field
from emperor.layers import (
    LayerConfig,
    LayerStackConfig,
    NormalizationOptions,
    RecurrentLayerConfig,
)
from emperor.patch import PatchConfig


@dataclass
class ExperimentConfig(ConfigBase):
    encoder_output_normalization: NormalizationOptions = field(
        default=NormalizationOptions.LAYER_NORM, kw_only=True
    )
    patch_config: PatchConfig | None = optional_field(
        "Config for non-overlapping image patch projection without a class token."
    )
    encoder_config: LayerStackConfig | RecurrentLayerConfig | None = optional_field(
        "Config for the generic Transformer encoder stack of Mixer blocks."
    )
    output_config: LayerConfig | None = optional_field(
        "Config for the mean-pooled classification head."
    )
