from dataclasses import dataclass

from emperor.config import ConfigBase, optional_field
from emperor.experts import MixtureOfExpertsModelConfig
from emperor.layers import LayerConfig, RecurrentLayerConfig


@dataclass
class ExperimentConfig(ConfigBase):
    input_model_config: LayerConfig | None = optional_field(
        "Config for the input layer that maps input_dim to hidden_dim."
    )
    model_config: MixtureOfExpertsModelConfig | RecurrentLayerConfig | None = (
        optional_field(
            "Config for the MixtureOfExpertsModel, optionally wrapped in a "
            "recurrent layer."
        )
    )
    output_model_config: LayerConfig | None = optional_field(
        "Config for the output layer that maps hidden_dim to output_dim."
    )
