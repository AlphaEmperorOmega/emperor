from dataclasses import dataclass

from emperor.base.utils import ConfigBase, optional_field
from emperor.neuron.core.config import NeuronClusterConfig


@dataclass
class ExperimentConfig(ConfigBase):
    input_model_config: ConfigBase | None = optional_field(
        "Package-local input boundary projection config."
    )
    neuron_cluster_config: NeuronClusterConfig | None = optional_field(
        "Neuron cluster that wraps the package-local hidden block."
    )
    output_model_config: ConfigBase | None = optional_field(
        "Package-local output boundary projection config."
    )
