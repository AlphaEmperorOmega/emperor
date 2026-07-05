from dataclasses import dataclass

from emperor.base.layer.config import (
    LayerConfig,
    LayerStackConfig,
    RecurrentLayerConfig,
)
from emperor.base.layer.gate import GateConfig
from emperor.experts.config import MixtureOfExpertsModelConfig
from emperor.experts.core.config import (
    MixtureOfExpertsConfig,
    MixtureOfExpertsLayerConfig,
)
from emperor.halting.config import StickBreakingConfig
from emperor.linears.core.config import LinearLayerConfig
from emperor.sampler.core.config import RouterConfig, SamplerConfig

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
    ExpertsSubmoduleStackSource,
    resolve_experts_controller_stack_options,
    resolve_experts_submodule_stack_options,
)
from models.experts._controller_stack import (
    build_linear_controller_stack,
)
from models.experts._gate_config_factory import ExpertsGateConfigFactory
from models.experts._halting_config_factory import ExpertsHaltingConfigFactory
from models.experts._memory_config_factory import ExpertsMemoryConfigFactory
from models.experts._recurrent_config_factory import ExpertsRecurrentConfigFactory


@dataclass(frozen=True)
class ControlConfigDependencies:
    stack_options: ExpertsStackOptions | None
    submodule_stack_options: ExpertsSubmoduleStackOptions | None
    mixture_options: ExpertsMixtureOptions | None
    expert_stack_options: ExpertsSubmoduleStackOptions | None
    sampler_options: ExpertsSamplerOptions | None
    router_options: ExpertsRouterOptions | None
    sampler_stack_options: ExpertsSubmoduleStackOptions | None
    layer_controller_options: ExpertsLayerControllerOptions | None
    dynamic_memory_options: ExpertsDynamicMemoryOptions | None
    recurrent_controller_options: ExpertsRecurrentControllerOptions | None
    expert_layer_controller_options: ExpertsLayerControllerOptions | None
    expert_dynamic_memory_options: ExpertsDynamicMemoryOptions | None
    expert_recurrent_controller_options: ExpertsRecurrentControllerOptions | None
    output_dim: int


class ControlConfigFactory:
    def __init__(self, dependencies: ControlConfigDependencies) -> None:
        self.stack_options = self.__default_stack_options(dependencies.stack_options)
        self.submodule_stack_options = self.__default_submodule_stack_options(
            dependencies.submodule_stack_options
        )
        self.mixture_options = self.__default_mixture_options(
            dependencies.mixture_options
        )
        self.expert_stack_options = self.__default_expert_stack_options(
            dependencies.expert_stack_options
        )
        self.sampler_options = self.__default_sampler_options(
            dependencies.sampler_options
        )
        self.router_options = self.__default_router_options(dependencies.router_options)
        self.sampler_stack_options = self.__default_sampler_stack_options(
            dependencies.sampler_stack_options
        )
        self.layer_controller_options = self.__default_layer_controller_options(
            dependencies.layer_controller_options
        )
        self.dynamic_memory_options = self.__default_dynamic_memory_options(
            dependencies.dynamic_memory_options
        )
        self.recurrent_controller_options = self.__default_recurrent_controller_options(
            dependencies.recurrent_controller_options
        )
        self.expert_layer_controller_options = (
            self.__default_expert_layer_controller_options(
                dependencies.expert_layer_controller_options
            )
        )
        self.expert_dynamic_memory_options = (
            self.__default_expert_dynamic_memory_options(
                dependencies.expert_dynamic_memory_options
            )
        )
        self.expert_recurrent_controller_options = (
            self.__default_expert_recurrent_controller_options(
                dependencies.expert_recurrent_controller_options
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

    def __default_stack_options(
        self,
        stack_options: ExpertsStackOptions | None,
    ) -> ExpertsStackOptions:
        if stack_options is not None:
            return stack_options
        return ExpertsStackOptions(
            hidden_dim=config.STACK_HIDDEN_DIM,
            bias_flag=config.STACK_BIAS_FLAG,
            layer_norm_position=config.STACK_LAYER_NORM_POSITION,
            num_layers=config.STACK_NUM_LAYERS,
            activation=config.STACK_ACTIVATION,
            residual_connection_option=config.STACK_RESIDUAL_CONNECTION_OPTION,
            dropout_probability=config.STACK_DROPOUT_PROBABILITY,
            last_layer_bias_option=config.STACK_LAST_LAYER_BIAS_OPTION,
            apply_output_pipeline_flag=config.STACK_APPLY_OUTPUT_PIPELINE_FLAG,
        )

    def __default_submodule_stack_options(
        self,
        submodule_stack_options: ExpertsSubmoduleStackOptions | None,
    ) -> ExpertsSubmoduleStackOptions:
        if submodule_stack_options is not None:
            return submodule_stack_options
        return ExpertsSubmoduleStackOptions(
            hidden_dim=config.SUBMODULE_STACK_HIDDEN_DIM,
            num_layers=config.SUBMODULE_STACK_NUM_LAYERS,
            last_layer_bias_option=config.SUBMODULE_STACK_LAST_LAYER_BIAS_OPTION,
            apply_output_pipeline_flag=(
                config.SUBMODULE_STACK_APPLY_OUTPUT_PIPELINE_FLAG
            ),
            activation=config.SUBMODULE_STACK_ACTIVATION,
            layer_norm_position=config.SUBMODULE_STACK_LAYER_NORM_POSITION,
            residual_connection_option=(
                config.SUBMODULE_STACK_RESIDUAL_CONNECTION_OPTION
            ),
            dropout_probability=config.SUBMODULE_STACK_DROPOUT_PROBABILITY,
            bias_flag=config.SUBMODULE_STACK_BIAS_FLAG,
        )

    def __default_mixture_options(
        self,
        mixture_options: ExpertsMixtureOptions | None,
    ) -> ExpertsMixtureOptions:
        if mixture_options is not None:
            return mixture_options
        return ExpertsMixtureOptions(
            top_k=config.EXPERT_TOP_K,
            num_experts=config.EXPERT_NUM_EXPERTS,
            capacity_factor=config.EXPERT_CAPACITY_FACTOR,
            dropped_token_behavior=config.EXPERT_DROPPED_TOKEN_BEHAVIOR,
            compute_expert_mixture_flag=(config.EXPERT_COMPUTE_EXPERT_MIXTURE_FLAG),
            weighted_parameters_flag=config.EXPERT_WEIGHTED_PARAMETERS_FLAG,
            weighting_position_option=config.EXPERT_WEIGHTING_POSITION_OPTION,
            routing_initialization_mode=config.EXPERT_ROUTING_INITIALIZATION_MODE,
        )

    def __default_expert_stack_options(
        self,
        expert_stack_options: ExpertsSubmoduleStackOptions | None,
    ) -> ExpertsSubmoduleStackOptions:
        if expert_stack_options is not None:
            return expert_stack_options
        return resolve_experts_submodule_stack_options(
            self.submodule_stack_options,
            layer_norm_position=config.EXPERT_STACK_LAYER_NORM_POSITION,
            apply_output_pipeline_flag=config.EXPERT_STACK_APPLY_OUTPUT_PIPELINE_FLAG,
        )

    def __default_sampler_options(
        self,
        sampler_options: ExpertsSamplerOptions | None,
    ) -> ExpertsSamplerOptions:
        if sampler_options is not None:
            return sampler_options
        return ExpertsSamplerOptions(
            threshold=config.SAMPLER_THRESHOLD,
            filter_above_threshold=config.SAMPLER_FILTER_ABOVE_THRESHOLD,
            num_topk_samples=config.SAMPLER_NUM_TOPK_SAMPLES,
            normalize_probabilities_flag=config.SAMPLER_NORMALIZE_PROBABILITIES_FLAG,
            noisy_topk_flag=config.SAMPLER_NOISY_TOPK_FLAG,
            coefficient_of_variation_loss_weight=(
                config.SAMPLER_COEFFICIENT_OF_VARIATION_LOSS_WEIGHT
            ),
            switch_loss_weight=config.SAMPLER_SWITCH_LOSS_WEIGHT,
            zero_centred_loss_weight=config.SAMPLER_ZERO_CENTRED_LOSS_WEIGHT,
            mutual_information_loss_weight=(
                config.SAMPLER_MUTUAL_INFORMATION_LOSS_WEIGHT
            ),
        )

    def __default_router_options(
        self,
        router_options: ExpertsRouterOptions | None,
    ) -> ExpertsRouterOptions:
        if router_options is not None:
            return router_options
        return ExpertsRouterOptions(
            noisy_topk_flag=config.ROUTER_NOISY_TOPK_FLAG,
        )

    def __default_sampler_stack_options(
        self,
        sampler_stack_options: ExpertsSubmoduleStackOptions | None,
    ) -> ExpertsSubmoduleStackOptions:
        if sampler_stack_options is not None:
            return sampler_stack_options
        return resolve_experts_controller_stack_options(
            ExpertsSubmoduleStackSource(
                independent_flag=config.SAMPLER_STACK_INDEPENDENT_FLAG,
                hidden_dim=config.SAMPLER_STACK_HIDDEN_DIM,
                num_layers=config.SAMPLER_STACK_NUM_LAYERS,
                last_layer_bias_option=config.SAMPLER_STACK_LAST_LAYER_BIAS_OPTION,
                apply_output_pipeline_flag=(
                    config.SAMPLER_STACK_APPLY_OUTPUT_PIPELINE_FLAG
                ),
                activation=config.SAMPLER_STACK_ACTIVATION,
                layer_norm_position=config.SAMPLER_STACK_LAYER_NORM_POSITION,
                residual_connection_option=(
                    config.SAMPLER_STACK_RESIDUAL_CONNECTION_OPTION
                ),
                dropout_probability=config.SAMPLER_STACK_DROPOUT_PROBABILITY,
                bias_flag=config.SAMPLER_BIAS_FLAG,
            ),
            self.submodule_stack_options,
        )

    def __default_layer_controller_options(
        self,
        layer_controller_options: ExpertsLayerControllerOptions | None,
    ) -> ExpertsLayerControllerOptions:
        if layer_controller_options is not None:
            return layer_controller_options
        return ExpertsLayerControllerOptions(
            stack_gate_flag=config.GATE_FLAG,
            gate_option=config.GATE_OPTION,
            gate_activation=config.GATE_ACTIVATION,
            gate_stack_source=self.__default_controller_stack_source("GATE_STACK"),
            stack_halting_flag=config.HALTING_FLAG,
            halting_threshold=config.HALTING_THRESHOLD,
            halting_dropout=config.HALTING_DROPOUT,
            halting_hidden_state_mode=config.HALTING_HIDDEN_STATE_MODE,
            halting_stack_source=self.__default_controller_stack_source(
                "HALTING_STACK"
            ),
            halting_output_dim=config.HALTING_OUTPUT_DIM,
        )

    def __default_dynamic_memory_options(
        self,
        dynamic_memory_options: ExpertsDynamicMemoryOptions | None,
    ) -> ExpertsDynamicMemoryOptions:
        if dynamic_memory_options is not None:
            return dynamic_memory_options
        return ExpertsDynamicMemoryOptions(
            memory_flag=config.MEMORY_FLAG,
            memory_option=config.MEMORY_OPTION,
            memory_position_option=config.MEMORY_POSITION_OPTION,
            memory_test_time_training_learning_rate=(
                config.MEMORY_TEST_TIME_TRAINING_LEARNING_RATE
            ),
            memory_test_time_training_num_inner_steps=(
                config.MEMORY_TEST_TIME_TRAINING_NUM_INNER_STEPS
            ),
            memory_stack_source=self.__default_controller_stack_source("MEMORY_STACK"),
        )

    def __default_recurrent_controller_options(
        self,
        recurrent_controller_options: ExpertsRecurrentControllerOptions | None,
    ) -> ExpertsRecurrentControllerOptions:
        if recurrent_controller_options is not None:
            return recurrent_controller_options
        return ExpertsRecurrentControllerOptions(
            recurrent_flag=config.RECURRENT_FLAG,
            recurrent_max_steps=config.RECURRENT_MAX_STEPS,
            recurrent_layer_norm_position=config.RECURRENT_LAYER_NORM_POSITION,
            recurrent_gate_flag=config.RECURRENT_GATE_FLAG,
            recurrent_gate_option=config.RECURRENT_GATE_OPTION,
            recurrent_gate_activation=config.RECURRENT_GATE_ACTIVATION,
            recurrent_gate_stack_source=self.__default_controller_stack_source(
                "RECURRENT_GATE_STACK"
            ),
            recurrent_halting_flag=config.RECURRENT_HALTING_FLAG,
            recurrent_halting_threshold=config.RECURRENT_HALTING_THRESHOLD,
            recurrent_halting_dropout=config.RECURRENT_HALTING_DROPOUT,
            recurrent_halting_hidden_state_mode=(
                config.RECURRENT_HALTING_HIDDEN_STATE_MODE
            ),
            recurrent_halting_stack_source=self.__default_controller_stack_source(
                "RECURRENT_HALTING_STACK"
            ),
        )

    def __default_expert_layer_controller_options(
        self,
        layer_controller_options: ExpertsLayerControllerOptions | None,
    ) -> ExpertsLayerControllerOptions:
        if layer_controller_options is not None:
            return layer_controller_options
        return ExpertsLayerControllerOptions(
            stack_gate_flag=config.EXPERT_GATE_FLAG,
            gate_option=config.EXPERT_GATE_OPTION,
            gate_activation=config.EXPERT_GATE_ACTIVATION,
            gate_stack_source=self.__default_controller_stack_source(
                "EXPERT_GATE_STACK"
            ),
            stack_halting_flag=config.EXPERT_HALTING_FLAG,
            halting_threshold=config.EXPERT_HALTING_THRESHOLD,
            halting_dropout=config.EXPERT_HALTING_DROPOUT,
            halting_hidden_state_mode=config.EXPERT_HALTING_HIDDEN_STATE_MODE,
            halting_stack_source=self.__default_controller_stack_source(
                "EXPERT_HALTING_STACK"
            ),
            halting_output_dim=config.EXPERT_HALTING_OUTPUT_DIM,
        )

    def __default_expert_dynamic_memory_options(
        self,
        dynamic_memory_options: ExpertsDynamicMemoryOptions | None,
    ) -> ExpertsDynamicMemoryOptions:
        if dynamic_memory_options is not None:
            return dynamic_memory_options
        return ExpertsDynamicMemoryOptions(
            memory_flag=config.EXPERT_MEMORY_FLAG,
            memory_option=config.EXPERT_MEMORY_OPTION,
            memory_position_option=config.EXPERT_MEMORY_POSITION_OPTION,
            memory_test_time_training_learning_rate=(
                config.EXPERT_MEMORY_TEST_TIME_TRAINING_LEARNING_RATE
            ),
            memory_test_time_training_num_inner_steps=(
                config.EXPERT_MEMORY_TEST_TIME_TRAINING_NUM_INNER_STEPS
            ),
            memory_stack_source=self.__default_controller_stack_source(
                "EXPERT_MEMORY_STACK"
            ),
        )

    def __default_expert_recurrent_controller_options(
        self,
        recurrent_controller_options: ExpertsRecurrentControllerOptions | None,
    ) -> ExpertsRecurrentControllerOptions:
        if recurrent_controller_options is not None:
            return recurrent_controller_options
        return ExpertsRecurrentControllerOptions(
            recurrent_flag=config.EXPERT_RECURRENT_FLAG,
            recurrent_max_steps=config.EXPERT_RECURRENT_MAX_STEPS,
            recurrent_layer_norm_position=(
                config.EXPERT_RECURRENT_LAYER_NORM_POSITION
            ),
            recurrent_gate_flag=config.EXPERT_RECURRENT_GATE_FLAG,
            recurrent_gate_option=config.EXPERT_RECURRENT_GATE_OPTION,
            recurrent_gate_activation=config.EXPERT_RECURRENT_GATE_ACTIVATION,
            recurrent_gate_stack_source=self.__default_controller_stack_source(
                "EXPERT_RECURRENT_GATE_STACK"
            ),
            recurrent_halting_flag=config.EXPERT_RECURRENT_HALTING_FLAG,
            recurrent_halting_threshold=config.EXPERT_RECURRENT_HALTING_THRESHOLD,
            recurrent_halting_dropout=config.EXPERT_RECURRENT_HALTING_DROPOUT,
            recurrent_halting_hidden_state_mode=(
                config.EXPERT_RECURRENT_HALTING_HIDDEN_STATE_MODE
            ),
            recurrent_halting_stack_source=self.__default_controller_stack_source(
                "EXPERT_RECURRENT_HALTING_STACK"
            ),
        )

    def __default_controller_stack_source(
        self,
        prefix: str,
    ) -> ExpertsSubmoduleStackSource:
        return ExpertsSubmoduleStackSource(
            independent_flag=getattr(config, f"{prefix}_INDEPENDENT_FLAG"),
            hidden_dim=getattr(config, f"{prefix}_HIDDEN_DIM"),
            num_layers=getattr(config, f"{prefix}_NUM_LAYERS"),
            last_layer_bias_option=getattr(
                config,
                f"{prefix}_LAST_LAYER_BIAS_OPTION",
            ),
            apply_output_pipeline_flag=getattr(
                config,
                f"{prefix}_APPLY_OUTPUT_PIPELINE_FLAG",
            ),
            activation=getattr(config, f"{prefix}_ACTIVATION"),
            layer_norm_position=getattr(config, f"{prefix}_LAYER_NORM_POSITION"),
            residual_connection_option=getattr(
                config,
                f"{prefix}_RESIDUAL_CONNECTION_OPTION",
            ),
            dropout_probability=getattr(config, f"{prefix}_DROPOUT_PROBABILITY"),
            bias_flag=getattr(config, f"{prefix}_BIAS_FLAG"),
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
        halting_config: StickBreakingConfig | None,
    ) -> LayerConfig:
        stack_options = self.stack_options
        return MixtureOfExpertsLayerConfig(
            activation=stack_options.activation,
            layer_norm_position=stack_options.layer_norm_position,
            residual_connection_option=(stack_options.residual_connection_option),
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
                residual_connection_option=(
                    expert_stack_options.residual_connection_option
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
        model_config = self.__build_controller_stack(self.sampler_stack_options)
        return RouterConfig(
            input_dim=self.hidden_dim,
            num_experts=mixture_options.num_experts,
            noisy_topk_flag=router_options.noisy_topk_flag,
            model_config=model_config,
        )

    def __build_controller_stack(
        self,
        options: ExpertsSubmoduleStackOptions,
    ) -> LayerStackConfig:
        return build_linear_controller_stack(options)
