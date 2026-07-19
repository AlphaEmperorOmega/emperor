from dataclasses import dataclass

from emperor.config import ConfigBase, optional_field
from emperor.layers import LayerConfig, LayerStackConfig


@dataclass
class ExperimentConfig(ConfigBase):
    input_model_config: LayerConfig | None = optional_field(
        "Config for the input layer that maps input_dim to hidden_dim."
    )
    model_config: LayerStackConfig | None = optional_field(
        "Parametric vector hidden stack."
    )
    output_model_config: LayerConfig | None = optional_field(
        "Config for the output layer that maps hidden_dim to output_dim."
    )
