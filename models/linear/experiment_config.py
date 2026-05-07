from dataclasses import dataclass

from emperor.base.utils import ConfigBase, optional_field
from emperor.base.layer import LayerStackConfig
from emperor.base.layer.config import LayerConfig


@dataclass
class ExperimentConfig(ConfigBase):
    input_model_config: "LayerConfig | None" = optional_field(
        "Config for the input layer that maps input_dim to hidden_dim."
    )
    model_config: "LayerStackConfig | None" = optional_field(
        "Config for the main hidden-dim layer stack."
    )
    output_model_config: "LayerConfig | None" = optional_field(
        "Config for the output layer that maps hidden_dim to output_dim."
    )
