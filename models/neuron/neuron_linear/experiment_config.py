from dataclasses import dataclass

from emperor.base.utils import ConfigBase, optional_field
from emperor.neuron.core.config import NeuronClusterConfig


@dataclass
class HiddenBlockConfig(ConfigBase):
    input_dim: int | None = optional_field(
        "Input feature dimension for the source hidden block adapter."
    )
    output_dim: int | None = optional_field(
        "Output feature dimension for the source hidden block adapter."
    )
    model_config: ConfigBase | None = optional_field(
        "Source hidden block config wrapped by the neuron nucleus."
    )

    def _registry_owner(self) -> type:
        from models.neuron.neuron_linear.model import HiddenBlockAdapter

        return HiddenBlockAdapter


@dataclass
class ExperimentConfig(ConfigBase):
    input_model_config: ConfigBase | None = optional_field(
        "Source input boundary projection config."
    )
    neuron_cluster_config: NeuronClusterConfig | None = optional_field(
        "Neuron cluster that wraps the source hidden block."
    )
    output_model_config: ConfigBase | None = optional_field(
        "Source output boundary projection config."
    )
