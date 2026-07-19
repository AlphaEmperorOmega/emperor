from dataclasses import dataclass

from emperor.base.layer.config import LayerConfig, RecurrentLayerConfig
from emperor.base.config import ConfigBase, optional_field
from emperor.experts.config import MixtureOfExpertsModelConfig


@dataclass
class ExperimentConfig(ConfigBase):
    input_model_config: LayerConfig | None = optional_field(
        "Config for the input adaptive layer that maps input_dim to hidden_dim."
    )
    model_config: MixtureOfExpertsModelConfig | RecurrentLayerConfig | None = (
        optional_field(
            "Config for the MixtureOfExpertsModel, optionally wrapped in a "
            "recurrent layer."
        )
    )
    output_model_config: LayerConfig | None = optional_field(
        "Config for the output adaptive layer that maps hidden_dim to output_dim."
    )
