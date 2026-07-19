from dataclasses import dataclass

from emperor.config import ConfigBase, optional_field
from emperor.layers import LayerConfig, LayerStackConfig, RecurrentLayerConfig


@dataclass
class ExperimentConfig(ConfigBase):
    input_model_config: LayerConfig | None = optional_field(
        "Config for the input layer that maps input_dim to hidden_dim."
    )
    model_config: LayerStackConfig | RecurrentLayerConfig | None = optional_field(
        "Config for the main hidden-dim layer stack, optionally wrapped in a "
        "recurrent layer."
    )
    output_model_config: LayerConfig | None = optional_field(
        "Config for the output layer that maps hidden_dim to output_dim."
    )
