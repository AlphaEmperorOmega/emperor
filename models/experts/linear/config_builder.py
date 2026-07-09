from typing import TYPE_CHECKING

import models.experts.linear.config as config
from models.experts._builder_options import (
    ExpertsDynamicMemoryOptions,
    ExpertsLayerControllerOptions,
    ExpertsMixtureOptions,
    ExpertsRecurrentControllerOptions,
    ExpertsRouterOptions,
    ExpertsSamplerOptions,
    ExpertsStackOptions,
    ExpertsSubmoduleStackOptions,
)
from models.experts.linear._boundary_config_factory import (
    BoundaryConfigDependencies,
    BoundaryConfigFactory,
)
from models.experts.linear._control_config_factory import (
    ControlConfigDependencies,
    ControlConfigFactory,
)
from models.experts.linear.experiment_config import ExperimentConfig

if TYPE_CHECKING:
    from emperor.config import ModelConfig


class LinearConfigBuilder:
    def __init__(
        self,
        batch_size: int = config.BATCH_SIZE,
        learning_rate: float = config.LEARNING_RATE,
        input_dim: int = config.INPUT_DIM,
        hidden_dim: int = config.HIDDEN_DIM,
        output_dim: int = config.OUTPUT_DIM,
        stack_options: ExpertsStackOptions | None = None,
        submodule_stack_options: ExpertsSubmoduleStackOptions | None = None,
        mixture_options: ExpertsMixtureOptions | None = None,
        expert_stack_options: ExpertsSubmoduleStackOptions | None = None,
        sampler_options: ExpertsSamplerOptions | None = None,
        router_options: ExpertsRouterOptions | None = None,
        router_stack_options: ExpertsSubmoduleStackOptions | None = None,
        layer_controller_options: ExpertsLayerControllerOptions | None = None,
        dynamic_memory_options: ExpertsDynamicMemoryOptions | None = None,
        recurrent_controller_options: ExpertsRecurrentControllerOptions | None = None,
        expert_layer_controller_options: ExpertsLayerControllerOptions | None = None,
        expert_dynamic_memory_options: ExpertsDynamicMemoryOptions | None = None,
        expert_recurrent_controller_options: (
            ExpertsRecurrentControllerOptions | None
        ) = None,
    ) -> None:
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.stack_options = stack_options
        self.submodule_stack_options = submodule_stack_options
        self.mixture_options = mixture_options
        self.expert_stack_options = expert_stack_options
        self.sampler_options = sampler_options
        self.router_options = router_options
        self.router_stack_options = router_stack_options
        self.layer_controller_options = layer_controller_options
        self.dynamic_memory_options = dynamic_memory_options
        self.recurrent_controller_options = recurrent_controller_options
        self.expert_layer_controller_options = expert_layer_controller_options
        self.expert_dynamic_memory_options = expert_dynamic_memory_options
        self.expert_recurrent_controller_options = expert_recurrent_controller_options

    def build(self) -> "ModelConfig":
        from emperor.config import ModelConfig

        return ModelConfig(
            learning_rate=self.learning_rate,
            batch_size=self.batch_size,
            input_dim=self.input_dim,
            hidden_dim=self.__model_hidden_dim(),
            output_dim=self.output_dim,
            experiment_config=ExperimentConfig(
                input_model_config=self.__input_model_config(),
                model_config=self.__model_config(),
                output_model_config=self.__output_model_config(),
            ),
        )

    def __input_model_config(self):
        boundary_dependencies = self.__boundary_config_dependencies()
        boundary_factory = BoundaryConfigFactory(boundary_dependencies)
        return boundary_factory.build_input_model_config()

    def __model_config(self):
        control_dependencies = self.__control_config_dependencies()
        control_factory = ControlConfigFactory(control_dependencies)
        return control_factory.build()

    def __output_model_config(self):
        boundary_dependencies = self.__boundary_config_dependencies()
        boundary_factory = BoundaryConfigFactory(boundary_dependencies)
        return boundary_factory.build_output_model_config()

    def __model_hidden_dim(self) -> int:
        control_dependencies = self.__control_config_dependencies()
        control_factory = ControlConfigFactory(control_dependencies)
        return control_factory.stack_options.hidden_dim

    def __boundary_config_dependencies(self) -> BoundaryConfigDependencies:
        return BoundaryConfigDependencies(
            hidden_dim=self.hidden_dim,
            stack_options=self.stack_options,
        )

    def __control_config_dependencies(self) -> ControlConfigDependencies:
        return ControlConfigDependencies(
            hidden_dim=self.hidden_dim,
            stack_options=self.stack_options,
            submodule_stack_options=self.submodule_stack_options,
            mixture_options=self.mixture_options,
            expert_stack_options=self.expert_stack_options,
            sampler_options=self.sampler_options,
            router_options=self.router_options,
            router_stack_options=self.router_stack_options,
            layer_controller_options=self.layer_controller_options,
            dynamic_memory_options=self.dynamic_memory_options,
            recurrent_controller_options=self.recurrent_controller_options,
            expert_layer_controller_options=self.expert_layer_controller_options,
            expert_dynamic_memory_options=self.expert_dynamic_memory_options,
            expert_recurrent_controller_options=(
                self.expert_recurrent_controller_options
            ),
            output_dim=self.output_dim,
        )
