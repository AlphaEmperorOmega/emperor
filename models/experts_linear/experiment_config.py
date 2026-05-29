from dataclasses import dataclass

from emperor.base.utils import ConfigBase, optional_field
from emperor.base.layer.config import LayerConfig
from emperor.experts.config import MixtureOfExpertsModelConfig


@dataclass
class ExperimentConfig(ConfigBase):
    input_model_config: "LayerConfig | None" = optional_field(
        "Config for the input layer that maps input_dim to hidden_dim."
    )
    model_config: "MixtureOfExpertsModelConfig | None" = optional_field(
        "Config for the MixtureOfExpertsModel that orchestrates the expert stack."
    )
    output_model_config: "LayerConfig | None" = optional_field(
        "Config for the output layer that maps hidden_dim to output_dim."
    )
