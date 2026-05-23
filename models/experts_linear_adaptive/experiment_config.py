from dataclasses import dataclass

from emperor.base.utils import ConfigBase, optional_field
from emperor.base.layer.config import LayerConfig, LayerStackConfig


@dataclass
class ExperimentConfig(ConfigBase):
    input_model_config: "LayerConfig | None" = optional_field(
        "Config for the input adaptive layer that maps input_dim to hidden_dim."
    )
    model_config: "LayerStackConfig | None" = optional_field(
        "Config for the main stack of MixtureOfExpertsLayer modules (each holding "
        "a MixtureOfExpertsConfig with adaptive experts and sampler)."
    )
    output_model_config: "LayerConfig | None" = optional_field(
        "Config for the output adaptive layer that maps hidden_dim to output_dim."
    )
