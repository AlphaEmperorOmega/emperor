from dataclasses import dataclass

import models.bert.expert_linear.config as config
from emperor.experts import (
    MixtureOfExpertsConfig,
    MixtureOfExpertsLayerConfig,
    MixtureOfExpertsModelConfig,
)
from emperor.halting import HaltingConfig
from emperor.layers import (
    GateConfig,
    LayerConfig,
    LayerStackConfig,
    RecurrentLayerConfig,
)
from emperor.linears import LinearLayerConfig
from emperor.sampler import RouterConfig, SamplerConfig
from models.bert.expert_linear import _config_defaults as config_defaults
from models.bert.expert_linear._expert_control_support import (
    ExpertsGateConfigFactory,
    ExpertsHaltingConfigFactory,
    ExpertsMemoryConfigFactory,
    ExpertsRecurrentConfigFactory,
)
from models.bert.expert_linear.runtime_options import (
    ExpertsDynamicMemoryOptions,
    ExpertsLayerControllerOptions,
    ExpertsMixtureOptions,
    ExpertsRecurrentControllerOptions,
    ExpertsRouterOptions,
    ExpertsSamplerOptions,
    ExpertsStackOptions,
    ExpertsSubmoduleStackOptions,
)

from ._controller_stack_config import build_controller_stack_config
from ._residual import build_residual_config


@dataclass(frozen=True)
class ControlConfigDependencies:
    hidden_dim: int
    stack_options: ExpertsStackOptions | None
    submodule_stack_options: ExpertsSubmoduleStackOptions | None
    mixture_options: ExpertsMixtureOptions | None
    expert_stack_options: ExpertsSubmoduleStackOptions | None
    sampler_options: ExpertsSamplerOptions | None
    router_options: ExpertsRouterOptions | None
    router_stack_options: ExpertsSubmoduleStackOptions | None
    layer_controller_options: ExpertsLayerControllerOptions | None
    dynamic_memory_options: ExpertsDynamicMemoryOptions | None
    recurrent_controller_options: ExpertsRecurrentControllerOptions | None
    expert_layer_controller_options: ExpertsLayerControllerOptions | None
    expert_dynamic_memory_options: ExpertsDynamicMemoryOptions | None
    expert_recurrent_controller_options: ExpertsRecurrentControllerOptions | None
    output_dim: int


class ControlConfigFactory:
    def __init__(self, dependencies: ControlConfigDependencies) -> None:
        self.stack_options = (
            dependencies.stack_options
            if dependencies.stack_options is not None
            else config_defaults.experts_stack_options(config, dependencies.hidden_dim)
        )
        self.submodule_stack_options = (
            dependencies.submodule_stack_options
            if dependencies.submodule_stack_options is not None
            else config_defaults.experts_submodule_stack_options(config, "base")
        )
        self.mixture_options = (
            dependencies.mixture_options
            if dependencies.mixture_options is not None
            else config_defaults.experts_mixture_options(config)
        )
        self.expert_stack_options = (
            dependencies.expert_stack_options
            if dependencies.expert_stack_options is not None
            else config_defaults.experts_role_stack_options(
                config,
                "expert",
                self.submodule_stack_options,
            )
        )
        self.sampler_options = (
            dependencies.sampler_options
            if dependencies.sampler_options is not None
            else config_defaults.experts_sampler_options(config)
        )
        self.router_options = (
            dependencies.router_options
            if dependencies.router_options is not None
            else config_defaults.experts_router_options(config)
        )
        self.router_stack_options = (
            dependencies.router_stack_options
            if dependencies.router_stack_options is not None
            else config_defaults.experts_submodule_stack_options(config, "router")
        )
        self.layer_controller_options = (
            dependencies.layer_controller_options
            if dependencies.layer_controller_options is not None
            else config_defaults.experts_layer_controller_options(config, "main")
        )
        self.dynamic_memory_options = (
            dependencies.dynamic_memory_options
            if dependencies.dynamic_memory_options is not None
            else config_defaults.experts_dynamic_memory_options(config, "main")
        )
        self.recurrent_controller_options = (
            dependencies.recurrent_controller_options
            if dependencies.recurrent_controller_options is not None
            else config_defaults.experts_recurrent_controller_options(config, "main")
        )
        self.expert_layer_controller_options = (
            config_defaults.expert_layer_controller_options(
                config,
                dependencies.expert_layer_controller_options,
            )
        )
        self.expert_dynamic_memory_options = (
            config_defaults.expert_dynamic_memory_options(
                config,
                dependencies.expert_dynamic_memory_options,
            )
        )
        self.expert_recurrent_controller_options = (
            config_defaults.expert_recurrent_controller_options(
                config,
                dependencies.expert_recurrent_controller_options,
            )
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
            routing_initialization_mode=(mixture_options.routing_initialization_mode),
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
            apply_output_pipeline_flag=stack_options.apply_output_pipeline_flag,
            shared_gate_config=layer_controller.shared_gate_config,
            shared_memory_config=memory_config,
            layer_config=layer_config,
        )

    def __build_layer_config(
        self,
        gate_config: GateConfig | None,
        halting_config: HaltingConfig | None,
    ) -> LayerConfig:
        stack_options = self.stack_options
        return MixtureOfExpertsLayerConfig(
            activation=stack_options.activation,
            layer_norm_position=stack_options.layer_norm_position,
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
            compute_expert_mixture_flag=(mixture_options.compute_expert_mixture_flag),
            weighted_parameters_flag=mixture_options.weighted_parameters_flag,
            weighting_position_option=mixture_options.weighting_position_option,
            routing_initialization_mode=(mixture_options.routing_initialization_mode),
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
            apply_output_pipeline_flag=(
                expert_stack_options.apply_output_pipeline_flag
            ),
            shared_gate_config=layer_controller.shared_gate_config,
            shared_memory_config=memory_config,
            layer_config=LayerConfig(
                activation=expert_stack_options.activation,
                layer_norm_position=expert_stack_options.layer_norm_position,
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
                    bias_flag=expert_stack_options.bias_flag,
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
            normalize_probabilities_flag=(sampler_options.normalize_probabilities_flag),
            noisy_topk_flag=sampler_options.noisy_topk_flag,
            num_experts=mixture_options.num_experts,
            coefficient_of_variation_loss_weight=(
                sampler_options.coefficient_of_variation_loss_weight
            ),
            switch_loss_weight=sampler_options.switch_loss_weight,
            zero_centred_loss_weight=sampler_options.zero_centred_loss_weight,
            mutual_information_loss_weight=(
                sampler_options.mutual_information_loss_weight
            ),
            router_config=router_config,
        )

    def __build_router_config(self) -> RouterConfig:
        mixture_options = self.mixture_options
        router_options = self.router_options
        model_config = build_controller_stack_config(self.router_stack_options)
        return RouterConfig(
            input_dim=self.hidden_dim,
            num_experts=mixture_options.num_experts,
            noisy_topk_flag=router_options.noisy_topk_flag,
            model_config=model_config,
        )
