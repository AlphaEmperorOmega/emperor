from __future__ import annotations

from emperor.config import ModelConfig
from models.neuron.linear._hidden._hidden_model_config_factory import (
    HiddenModelConfigFactory,
)
from models.neuron.linear._hidden._projection_config_factory import (
    ProjectionConfigFactory,
)
from models.neuron.linear._neuron_control_config_factory import (
    NeuronControlConfigDependencies,
    NeuronControlConfigFactory,
)
from models.neuron.linear.experiment_config import ExperimentConfig
from models.neuron.linear.runtime_options import (
    RuntimeOptions,
)


class NeuronConfigBuilder:
    def __init__(
        self,
        *,
        runtime: RuntimeOptions,
    ) -> None:
        self.runtime = runtime

    def build(self) -> ModelConfig:
        hidden_runtime = self.runtime.hidden_runtime
        projection_factory = ProjectionConfigFactory(hidden_runtime)
        hidden_factory = HiddenModelConfigFactory(hidden_runtime)
        neuron_dependencies = self.__neuron_control_config_dependencies()
        neuron_control_factory = NeuronControlConfigFactory(neuron_dependencies)
        neuron_cluster_config = neuron_control_factory.build(
            hidden_factory.build_hidden_model_config(),
            hidden_runtime.hidden_dim,
        )

        return ModelConfig(
            learning_rate=hidden_runtime.learning_rate,
            batch_size=hidden_runtime.batch_size,
            input_dim=hidden_runtime.input_dim,
            hidden_dim=hidden_runtime.hidden_dim,
            output_dim=hidden_runtime.output_dim,
            experiment_config=ExperimentConfig(
                input_model_config=projection_factory.build_input_model_config(),
                neuron_cluster_config=neuron_cluster_config,
                output_model_config=projection_factory.build_output_model_config(),
            ),
        )

    def __neuron_control_config_dependencies(
        self,
    ) -> NeuronControlConfigDependencies:
        return NeuronControlConfigDependencies(
            cluster_capacity_options=self.runtime.cluster_capacity_options,
            terminal_options=self.runtime.terminal_options,
            terminal_router_options=self.runtime.terminal_router_options,
            terminal_sampler_options=self.runtime.terminal_sampler_options,
            cluster_halting_options=self.runtime.cluster_halting_options,
        )
