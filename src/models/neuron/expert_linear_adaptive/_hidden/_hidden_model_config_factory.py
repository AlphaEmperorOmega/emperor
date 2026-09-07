from __future__ import annotations

from emperor.experts import MixtureOfExpertsModelConfig
from emperor.layers import LayerConfig, RecurrentLayerConfig
from models.neuron.expert_linear_adaptive._hidden._control_config_factory import (
    ControlConfigDependencies,
    ControlConfigFactory,
)
from models.neuron.expert_linear_adaptive._hidden._projection_config_factory import (
    BoundaryModelConfigDependencies,
    BoundaryModelConfigFactory,
)
from models.neuron.expert_linear_adaptive._hidden.runtime_options import RuntimeOptions


class HiddenModelConfigFactory:
    """Build package-local boundaries and hidden expert-adaptive configuration."""

    def __init__(self, runtime: RuntimeOptions) -> None:
        if not isinstance(runtime, RuntimeOptions):
            raise TypeError("runtime must be a RuntimeOptions value")
        self.runtime = runtime
        self.hidden_dim = runtime.stack_options.hidden_dim

    def build_input_model_config(self) -> LayerConfig:
        return self.__boundary_factory().build_input_model_config()

    def build_hidden_model_config(
        self,
    ) -> MixtureOfExpertsModelConfig | RecurrentLayerConfig:
        return self.__control_factory().build()

    def build_output_model_config(self) -> LayerConfig:
        return self.__boundary_factory().build_output_model_config()

    def __boundary_factory(self) -> BoundaryModelConfigFactory:
        runtime = self.runtime
        return BoundaryModelConfigFactory(
            BoundaryModelConfigDependencies(
                stack_options=runtime.stack_options,
                input_boundary_options=runtime.input_boundary_options,
                output_boundary_options=runtime.output_boundary_options,
                adaptive_generator_stack_options=(
                    runtime.adaptive_generator_stack_options
                ),
            )
        )

    def __control_factory(self) -> ControlConfigFactory:
        runtime = self.runtime
        return ControlConfigFactory(
            ControlConfigDependencies(
                stack_options=runtime.stack_options,
                submodule_stack_options=runtime.submodule_stack_options,
                mixture_options=runtime.mixture_options,
                expert_stack_options=runtime.expert_stack_options,
                sampler_options=runtime.sampler_options,
                router_options=runtime.router_options,
                router_stack_options=runtime.router_stack_options,
                router_layer_controller_options=(
                    runtime.router_layer_controller_options
                ),
                router_dynamic_memory_options=runtime.router_dynamic_memory_options,
                router_recurrent_controller_options=(
                    runtime.router_recurrent_controller_options
                ),
                layer_controller_options=runtime.layer_controller_options,
                dynamic_memory_options=runtime.dynamic_memory_options,
                recurrent_controller_options=runtime.recurrent_controller_options,
                expert_layer_controller_options=(
                    runtime.expert_layer_controller_options
                ),
                expert_dynamic_memory_options=runtime.expert_dynamic_memory_options,
                expert_recurrent_controller_options=(
                    runtime.expert_recurrent_controller_options
                ),
                adaptive_generator_stack_options=(
                    runtime.adaptive_generator_stack_options
                ),
                grouping_config=runtime.grouping_config,
                hidden_adaptive_weight_options=(runtime.hidden_adaptive_weight_options),
                hidden_adaptive_bias_options=runtime.hidden_adaptive_bias_options,
                hidden_adaptive_diagonal_options=(
                    runtime.hidden_adaptive_diagonal_options
                ),
                hidden_adaptive_mask_options=runtime.hidden_adaptive_mask_options,
                router_grouping_config=runtime.router_grouping_config,
                router_adaptive_weight_options=(runtime.router_adaptive_weight_options),
                router_adaptive_bias_options=runtime.router_adaptive_bias_options,
                router_adaptive_diagonal_options=(
                    runtime.router_adaptive_diagonal_options
                ),
                router_adaptive_mask_options=runtime.router_adaptive_mask_options,
                hidden_dim=runtime.stack_options.hidden_dim,
                output_dim=runtime.output_dim,
            )
        )


__all__ = ["HiddenModelConfigFactory"]
