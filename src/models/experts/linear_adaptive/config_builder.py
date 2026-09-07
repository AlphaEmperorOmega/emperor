from __future__ import annotations

from emperor.config import ModelConfig
from models.experts.linear_adaptive._control_config_factory import (
    ControlConfigDependencies,
    ControlConfigFactory,
)
from models.experts.linear_adaptive._projection_config_factory import (
    BoundaryModelConfigDependencies,
    BoundaryModelConfigFactory,
)
from models.experts.linear_adaptive.experiment_config import ExperimentConfig
from models.experts.linear_adaptive.runtime_defaults import DEFAULT_RUNTIME
from models.experts.linear_adaptive.runtime_options import RuntimeOptions


class LinearAdaptiveConfigBuilder:
    """Build adaptive-expert configs from one immutable runtime value."""

    def __init__(
        self,
        *,
        runtime: RuntimeOptions = DEFAULT_RUNTIME,
    ) -> None:
        if not isinstance(runtime, RuntimeOptions):
            raise TypeError("runtime must be a RuntimeOptions value")
        self.runtime = runtime

    def build(self) -> ModelConfig:
        runtime = self.runtime
        return ModelConfig(
            learning_rate=runtime.learning_rate,
            batch_size=runtime.batch_size,
            input_dim=runtime.input_dim,
            hidden_dim=runtime.stack_options.hidden_dim,
            output_dim=runtime.output_dim,
            experiment_config=ExperimentConfig(
                input_model_config=self.__boundary_factory().build_input_model_config(),
                model_config=self.__control_factory().build(),
                output_model_config=(
                    self.__boundary_factory().build_output_model_config()
                ),
            ),
        )

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
                grouping_config=runtime.grouping_config,
                router_grouping_config=runtime.router_grouping_config,
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
                hidden_adaptive_weight_options=(runtime.hidden_adaptive_weight_options),
                hidden_adaptive_bias_options=runtime.hidden_adaptive_bias_options,
                hidden_adaptive_diagonal_options=(
                    runtime.hidden_adaptive_diagonal_options
                ),
                hidden_adaptive_mask_options=runtime.hidden_adaptive_mask_options,
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


__all__ = ["LinearAdaptiveConfigBuilder"]
