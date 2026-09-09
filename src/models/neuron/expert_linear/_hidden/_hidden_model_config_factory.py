from dataclasses import dataclass, replace

from emperor.config import ConfigBase
from emperor.experts import (
    MixtureOfExpertsConfig,
    MixtureOfExpertsLayerConfig,
    MixtureOfExpertsModelConfig,
)
from emperor.halting import HaltingConfig
from emperor.layers import (
    GateConfig,
    LastLayerBiasOptions,
    LayerConfig,
    LayerStackConfig,
    RecurrentLayerConfig,
)
from emperor.linears import LinearLayerConfig
from emperor.memory import DynamicMemoryConfig
from emperor.sampler import RouterConfig, SamplerConfig
from models.neuron.expert_linear._hidden.runtime_options import (
    ExpertsAdaptiveGeneratorStackOptions,
    ExpertsDynamicMemoryOptions,
    ExpertsLayerControllerOptions,
    ExpertsMixtureOptions,
    ExpertsRecurrentControllerOptions,
    ExpertsRouterOptions,
    ExpertsSamplerOptions,
    ExpertsStackOptions,
    ExpertsSubmoduleStackOptions,
    resolve_experts_controller_stack_options,
)

from .._residual import build_residual_config


def build_linear_controller_stack(
    options: ExpertsSubmoduleStackOptions,
    *,
    hidden_dim: int | None = None,
    output_dim: int | None = None,
) -> LayerStackConfig:
    return build_controller_stack(
        options,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        layer_model_config=LinearLayerConfig(bias_flag=options.bias_flag),
    )


def build_controller_stack(
    options: ExpertsSubmoduleStackOptions | ExpertsAdaptiveGeneratorStackOptions,
    *,
    layer_model_config: ConfigBase,
    hidden_dim: int | None = None,
    output_dim: int | None = None,
) -> LayerStackConfig:
    return LayerStackConfig(
        hidden_dim=options.hidden_dim if hidden_dim is None else hidden_dim,
        output_dim=output_dim,
        num_layers=options.num_layers,
        last_layer_bias_option=options.last_layer_bias_option,
        apply_output_postprocessing_flag=options.apply_output_postprocessing_flag,
        layer_config=LayerConfig(
            activation=options.activation,
            layer_norm_position=options.layer_norm_position,
            normalization=options.normalization,
            residual_config=build_residual_config(
                options.residual_connection_option,
                options.residual_model_flag,
                options.residual_stack_options,
            ),
            dropout_probability=options.dropout_probability,
            gate_config=None,
            halting_config=None,
            memory_config=None,
            layer_model_config=layer_model_config,
        ),
    )


class ExpertsGateConfigFactory:
    def __init__(
        self,
        *,
        layer_controller_options: ExpertsLayerControllerOptions,
        recurrent_controller_options: ExpertsRecurrentControllerOptions,
        submodule_stack_options: ExpertsSubmoduleStackOptions,
        recurrent_stack_inherits_gate_stack: bool = True,
    ) -> None:
        self.layer_controller_options = layer_controller_options
        self.recurrent_controller_options = recurrent_controller_options
        self.submodule_stack_options = submodule_stack_options
        self.recurrent_stack_inherits_gate_stack = recurrent_stack_inherits_gate_stack

    def build_gate_config(self) -> GateConfig | None:
        if not self.layer_controller_options.stack_gate_flag:
            return None
        return GateConfig(
            model_config=self.__build_gate_model_config(),
            option=self.layer_controller_options.gate_option,
            activation=self.layer_controller_options.gate_activation,
        )

    def build_recurrent_gate_config(self) -> GateConfig | None:
        if not self.recurrent_controller_options.recurrent_stack_gate_flag:
            return None
        options = resolve_experts_controller_stack_options(
            self.recurrent_controller_options.recurrent_gate_stack_source,
            self.__recurrent_gate_stack_defaults(),
        )
        return GateConfig(
            model_config=build_linear_controller_stack(options),
            option=self.recurrent_controller_options.recurrent_gate_option,
            activation=self.recurrent_controller_options.recurrent_gate_activation,
        )

    def __build_gate_model_config(self) -> LayerStackConfig:
        options = resolve_experts_controller_stack_options(
            self.layer_controller_options.gate_stack_source,
            self.submodule_stack_options,
        )
        return build_linear_controller_stack(options)

    def __recurrent_gate_stack_defaults(self) -> ExpertsSubmoduleStackOptions:
        if not self.recurrent_stack_inherits_gate_stack:
            return self.submodule_stack_options
        return resolve_experts_controller_stack_options(
            self.layer_controller_options.gate_stack_source,
            self.submodule_stack_options,
        )


class ExpertsHaltingConfigFactory:
    def __init__(
        self,
        *,
        layer_controller_options: ExpertsLayerControllerOptions,
        recurrent_controller_options: ExpertsRecurrentControllerOptions,
        submodule_stack_options: ExpertsSubmoduleStackOptions,
        output_dim: int,
        halting_stack_defaults: ExpertsSubmoduleStackOptions | None = None,
        recurrent_stack_inherits_halting_stack: bool = True,
    ) -> None:
        self.layer_controller_options = layer_controller_options
        self.recurrent_controller_options = recurrent_controller_options
        self.submodule_stack_options = submodule_stack_options
        self.output_dim = output_dim
        self.halting_stack_defaults = halting_stack_defaults
        self.recurrent_stack_inherits_halting_stack = (
            recurrent_stack_inherits_halting_stack
        )

    def build_halting_config(self) -> HaltingConfig | None:
        if not self.layer_controller_options.stack_halting_flag:
            return None
        controller = self.layer_controller_options
        options = resolve_experts_controller_stack_options(
            controller.halting_stack_source, self.__halting_stack_defaults()
        )
        return controller.halting_option(
            threshold=controller.halting_threshold,
            min_steps=1,
            ponder_cost_weight=1.0,
            dropout_probability=controller.halting_dropout,
            hidden_state_mode=controller.halting_hidden_state_mode,
            halting_gate_config=self.__build_halting_gate_stack(options),
        )

    def build_recurrent_halting_config(self) -> HaltingConfig | None:
        if not self.recurrent_controller_options.recurrent_stack_halting_flag:
            return None
        controller = self.recurrent_controller_options
        options = resolve_experts_controller_stack_options(
            controller.recurrent_halting_stack_source,
            self.__recurrent_halting_stack_defaults(),
        )
        return controller.recurrent_halting_option(
            threshold=controller.recurrent_halting_threshold,
            min_steps=1,
            ponder_cost_weight=1.0,
            dropout_probability=controller.recurrent_halting_dropout,
            hidden_state_mode=controller.recurrent_halting_hidden_state_mode,
            halting_gate_config=self.__build_halting_gate_stack(options),
        )

    def __build_halting_gate_stack(
        self, options: ExpertsSubmoduleStackOptions
    ) -> LayerStackConfig:
        return build_linear_controller_stack(
            options,
            hidden_dim=options.hidden_dim or self.output_dim,
            output_dim=self.layer_controller_options.halting_output_dim,
        )

    def __halting_stack_defaults(self) -> ExpertsSubmoduleStackOptions:
        if self.halting_stack_defaults is not None:
            return self.halting_stack_defaults
        return replace(
            self.submodule_stack_options,
            last_layer_bias_option=LastLayerBiasOptions.DISABLED,
        )

    def __recurrent_halting_stack_defaults(self) -> ExpertsSubmoduleStackOptions:
        if not self.recurrent_stack_inherits_halting_stack:
            return self.__halting_stack_defaults()
        return resolve_experts_controller_stack_options(
            self.layer_controller_options.halting_stack_source,
            self.__halting_stack_defaults(),
        )


class ExpertsMemoryConfigFactory:
    def __init__(
        self,
        *,
        stack_options: ExpertsStackOptions | ExpertsSubmoduleStackOptions,
        dynamic_memory_options: ExpertsDynamicMemoryOptions,
        submodule_stack_options: ExpertsSubmoduleStackOptions,
    ) -> None:
        self.stack_options = stack_options
        self.dynamic_memory_options = dynamic_memory_options
        self.submodule_stack_options = submodule_stack_options

    def build_memory_config(self) -> DynamicMemoryConfig | None:
        if not self.dynamic_memory_options.memory_flag:
            return None
        options = resolve_experts_controller_stack_options(
            self.dynamic_memory_options.memory_stack_source,
            self.submodule_stack_options,
        )
        return self.dynamic_memory_options.memory_option(
            input_dim=self.stack_options.hidden_dim,
            output_dim=self.stack_options.hidden_dim,
            memory_position_option=self.dynamic_memory_options.memory_position_option,
            test_time_training_learning_rate=self.dynamic_memory_options.memory_test_time_training_learning_rate,
            test_time_training_num_inner_steps=self.dynamic_memory_options.memory_test_time_training_num_inner_steps,
            model_config=build_linear_controller_stack(options),
        )


class ExpertsRecurrentConfigFactory:
    def __init__(
        self,
        *,
        recurrent_controller_options: ExpertsRecurrentControllerOptions,
        gate_config_factory: ExpertsGateConfigFactory,
        halting_config_factory: ExpertsHaltingConfigFactory,
    ) -> None:
        self.recurrent_controller_options = recurrent_controller_options
        self.gate_config_factory = gate_config_factory
        self.halting_config_factory = halting_config_factory

    def build_config(
        self, block_config: ConfigBase
    ) -> ConfigBase | RecurrentLayerConfig:
        if not self.recurrent_controller_options.recurrent_flag:
            return block_config
        return RecurrentLayerConfig(
            max_steps=self.recurrent_controller_options.recurrent_max_steps,
            gradient_transition_count=(
                self.recurrent_controller_options.recurrent_gradient_transition_count
            ),
            no_gradient_transition_count=(
                self.recurrent_controller_options.recurrent_no_gradient_transition_count
            ),
            initial_iterations=self.recurrent_controller_options.recurrent_initial_iterations,
            iteration_increment=self.recurrent_controller_options.recurrent_iteration_increment,
            forward_calls_before_iteration_increment=(
                self.recurrent_controller_options.recurrent_forward_calls_before_iteration_increment
            ),
            smooth_iteration_growth_flag=(
                self.recurrent_controller_options.recurrent_smooth_iteration_growth_flag
            ),
            recurrent_layer_norm_position=self.recurrent_controller_options.recurrent_layer_norm_position,
            recurrent_normalization=self.recurrent_controller_options.recurrent_normalization,
            block_config=block_config,
            gate_config=self.gate_config_factory.build_recurrent_gate_config(),
            residual_config=None,
            halting_config=self.halting_config_factory.build_recurrent_halting_config(),
        )


@dataclass(frozen=True)
class HiddenModelConfigDependencies:
    stack_options: ExpertsStackOptions
    submodule_stack_options: ExpertsSubmoduleStackOptions
    mixture_options: ExpertsMixtureOptions
    expert_stack_options: ExpertsSubmoduleStackOptions
    sampler_options: ExpertsSamplerOptions
    router_options: ExpertsRouterOptions
    router_stack_options: ExpertsSubmoduleStackOptions
    layer_controller_options: ExpertsLayerControllerOptions
    dynamic_memory_options: ExpertsDynamicMemoryOptions
    recurrent_controller_options: ExpertsRecurrentControllerOptions
    expert_layer_controller_options: ExpertsLayerControllerOptions
    expert_dynamic_memory_options: ExpertsDynamicMemoryOptions
    expert_recurrent_controller_options: ExpertsRecurrentControllerOptions
    output_dim: int


class HiddenModelConfigFactory:
    def __init__(self, dependencies: HiddenModelConfigDependencies) -> None:
        self.stack_options = dependencies.stack_options
        self.submodule_stack_options = dependencies.submodule_stack_options
        self.mixture_options = dependencies.mixture_options
        self.expert_stack_options = dependencies.expert_stack_options
        self.sampler_options = dependencies.sampler_options
        self.router_options = dependencies.router_options
        self.router_stack_options = dependencies.router_stack_options
        self.layer_controller_options = dependencies.layer_controller_options
        self.dynamic_memory_options = dependencies.dynamic_memory_options
        self.recurrent_controller_options = dependencies.recurrent_controller_options
        self.expert_layer_controller_options = (
            dependencies.expert_layer_controller_options
        )
        self.expert_dynamic_memory_options = dependencies.expert_dynamic_memory_options
        self.expert_recurrent_controller_options = (
            dependencies.expert_recurrent_controller_options
        )
        self.hidden_dim = self.stack_options.hidden_dim
        self.output_dim = dependencies.output_dim
        self.gate_config_factory = ExpertsGateConfigFactory(
            layer_controller_options=self.layer_controller_options,
            recurrent_controller_options=self.recurrent_controller_options,
            submodule_stack_options=self.submodule_stack_options,
        )
        self.halting_config_factory = ExpertsHaltingConfigFactory(
            layer_controller_options=self.layer_controller_options,
            recurrent_controller_options=self.recurrent_controller_options,
            submodule_stack_options=self.submodule_stack_options,
            output_dim=self.output_dim,
        )
        self.memory_config_factory = ExpertsMemoryConfigFactory(
            stack_options=self.stack_options,
            dynamic_memory_options=self.dynamic_memory_options,
            submodule_stack_options=self.submodule_stack_options,
        )
        self.recurrent_config_factory = ExpertsRecurrentConfigFactory(
            recurrent_controller_options=self.recurrent_controller_options,
            gate_config_factory=self.gate_config_factory,
            halting_config_factory=self.halting_config_factory,
        )
        self.expert_gate_config_factory = ExpertsGateConfigFactory(
            layer_controller_options=self.expert_layer_controller_options,
            recurrent_controller_options=self.expert_recurrent_controller_options,
            submodule_stack_options=self.expert_stack_options,
            recurrent_stack_inherits_gate_stack=False,
        )
        self.expert_halting_config_factory = ExpertsHaltingConfigFactory(
            layer_controller_options=self.expert_layer_controller_options,
            recurrent_controller_options=self.expert_recurrent_controller_options,
            submodule_stack_options=self.expert_stack_options,
            output_dim=self.expert_stack_options.hidden_dim,
            halting_stack_defaults=self.expert_stack_options,
            recurrent_stack_inherits_halting_stack=False,
        )
        self.expert_memory_config_factory = ExpertsMemoryConfigFactory(
            stack_options=self.expert_stack_options,
            dynamic_memory_options=self.expert_dynamic_memory_options,
            submodule_stack_options=self.expert_stack_options,
        )
        self.expert_recurrent_config_factory = ExpertsRecurrentConfigFactory(
            recurrent_controller_options=self.expert_recurrent_controller_options,
            gate_config_factory=self.expert_gate_config_factory,
            halting_config_factory=self.expert_halting_config_factory,
        )

    def build(self) -> MixtureOfExpertsModelConfig | RecurrentLayerConfig:
        return self.recurrent_config_factory.build_config(
            self.__build_main_model_config()
        )

    def __build_main_model_config(self) -> MixtureOfExpertsModelConfig:
        mixture_options = self.mixture_options
        return MixtureOfExpertsModelConfig(
            input_dim=self.hidden_dim,
            output_dim=self.hidden_dim,
            top_k=mixture_options.top_k,
            routing_initialization_mode=mixture_options.routing_initialization_mode,
            sampler_config=self.__build_sampler_config(),
            stack_config=self.__build_main_stack_config(),
        )

    def __build_main_stack_config(self) -> LayerStackConfig:
        stack_options = self.stack_options
        layer_controller = self.layer_controller_options
        gate_config = self.gate_config_factory.build_gate_config()
        halting_config = self.halting_config_factory.build_halting_config()
        memory_config = self.memory_config_factory.build_memory_config()
        layer_config = self.__build_layer_config(gate_config, halting_config)
        return LayerStackConfig(
            input_dim=stack_options.hidden_dim,
            hidden_dim=stack_options.hidden_dim,
            output_dim=stack_options.hidden_dim,
            num_layers=stack_options.num_layers,
            last_layer_bias_option=stack_options.last_layer_bias_option,
            apply_output_postprocessing_flag=stack_options.apply_output_postprocessing_flag,
            shared_gate_config=layer_controller.shared_gate_config,
            shared_memory_config=memory_config,
            layer_config=layer_config,
        )

    def __build_layer_config(
        self, gate_config: GateConfig | None, halting_config: HaltingConfig | None
    ) -> LayerConfig:
        stack_options = self.stack_options
        return MixtureOfExpertsLayerConfig(
            activation=stack_options.activation,
            layer_norm_position=stack_options.layer_norm_position,
            normalization=stack_options.normalization,
            residual_config=build_residual_config(
                stack_options.residual_connection_option,
                stack_options.residual_model_flag,
                stack_options.residual_stack_options,
            ),
            dropout_probability=stack_options.dropout_probability,
            gate_config=gate_config,
            halting_config=halting_config,
            layer_model_config=self.__build_mixture_of_experts_config(),
        )

    def __build_mixture_of_experts_config(self) -> MixtureOfExpertsConfig:
        mixture_options = self.mixture_options
        return MixtureOfExpertsConfig(
            input_dim=self.hidden_dim,
            output_dim=self.hidden_dim,
            top_k=mixture_options.top_k,
            num_experts=mixture_options.num_experts,
            capacity_factor=mixture_options.capacity_factor,
            dropped_token_behavior=mixture_options.dropped_token_behavior,
            compute_expert_mixture_flag=mixture_options.compute_expert_mixture_flag,
            weighted_parameters_flag=mixture_options.weighted_parameters_flag,
            weighting_position_option=mixture_options.weighting_position_option,
            routing_initialization_mode=mixture_options.routing_initialization_mode,
            sampler_config=self.__build_sampler_config(),
            expert_model_config=self.__build_expert_model_config(),
        )

    def __build_expert_model_config(self) -> LayerStackConfig | RecurrentLayerConfig:
        return self.expert_recurrent_config_factory.build_config(
            self.__build_expert_stack_config()
        )

    def __build_expert_stack_config(self) -> LayerStackConfig:
        expert_stack_options = self.expert_stack_options
        layer_controller = self.expert_layer_controller_options
        gate_config = self.expert_gate_config_factory.build_gate_config()
        halting_config = self.expert_halting_config_factory.build_halting_config()
        memory_config = self.expert_memory_config_factory.build_memory_config()
        return LayerStackConfig(
            hidden_dim=expert_stack_options.hidden_dim,
            num_layers=expert_stack_options.num_layers,
            last_layer_bias_option=expert_stack_options.last_layer_bias_option,
            apply_output_postprocessing_flag=expert_stack_options.apply_output_postprocessing_flag,
            shared_gate_config=layer_controller.shared_gate_config,
            shared_memory_config=memory_config,
            layer_config=LayerConfig(
                activation=expert_stack_options.activation,
                layer_norm_position=expert_stack_options.layer_norm_position,
                normalization=expert_stack_options.normalization,
                residual_config=build_residual_config(
                    expert_stack_options.residual_connection_option,
                    expert_stack_options.residual_model_flag,
                    expert_stack_options.residual_stack_options,
                ),
                dropout_probability=expert_stack_options.dropout_probability,
                gate_config=gate_config,
                halting_config=halting_config,
                memory_config=None,
                layer_model_config=LinearLayerConfig(
                    bias_flag=expert_stack_options.bias_flag
                ),
            ),
        )

    def __build_sampler_config(self) -> SamplerConfig:
        mixture_options = self.mixture_options
        sampler_options = self.sampler_options
        router_config = self.__build_router_config()
        return SamplerConfig(
            top_k=mixture_options.top_k,
            threshold=sampler_options.threshold,
            filter_above_threshold=sampler_options.filter_above_threshold,
            num_topk_samples=sampler_options.num_topk_samples,
            normalize_probabilities_flag=sampler_options.normalize_probabilities_flag,
            noisy_topk_flag=sampler_options.noisy_topk_flag,
            num_experts=mixture_options.num_experts,
            coefficient_of_variation_loss_weight=sampler_options.coefficient_of_variation_loss_weight,
            switch_loss_weight=sampler_options.switch_loss_weight,
            zero_centred_loss_weight=sampler_options.zero_centred_loss_weight,
            mutual_information_loss_weight=sampler_options.mutual_information_loss_weight,
            router_config=router_config,
        )

    def __build_router_config(self) -> RouterConfig:
        mixture_options = self.mixture_options
        router_options = self.router_options
        model_config = build_linear_controller_stack(self.router_stack_options)
        return RouterConfig(
            input_dim=self.hidden_dim,
            num_experts=mixture_options.num_experts,
            noisy_topk_flag=router_options.noisy_topk_flag,
            model_config=model_config,
        )
