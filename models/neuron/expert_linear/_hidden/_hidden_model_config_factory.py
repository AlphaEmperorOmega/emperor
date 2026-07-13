from dataclasses import dataclass, replace
from typing import Any

from emperor.base.layer.config import (
    LayerConfig,
    LayerStackConfig,
    RecurrentLayerConfig,
)
from emperor.base.layer.gate import GateConfig
from emperor.base.layer.residual import ResidualConnectionOptions
from emperor.base.options import LastLayerBiasOptions
from emperor.base.utils import ConfigBase
from emperor.experts.config import MixtureOfExpertsModelConfig
from emperor.experts.core.config import (
    MixtureOfExpertsConfig,
    MixtureOfExpertsLayerConfig,
)
from emperor.halting.config import StickBreakingConfig
from emperor.linears.core.config import LinearLayerConfig
from emperor.memory.config import DynamicMemoryConfig
from emperor.sampler.core.config import RouterConfig, SamplerConfig

import models.neuron.expert_linear.config as config
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
    ExpertsSubmoduleStackSource,
    resolve_experts_controller_stack_options,
    resolve_experts_submodule_stack_options,
)


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
    layer_model_config: Any,
    hidden_dim: int | None = None,
    output_dim: int | None = None,
) -> LayerStackConfig:
    return LayerStackConfig(
        hidden_dim=options.hidden_dim if hidden_dim is None else hidden_dim,
        output_dim=output_dim,
        num_layers=options.num_layers,
        last_layer_bias_option=options.last_layer_bias_option,
        apply_output_pipeline_flag=options.apply_output_pipeline_flag,
        layer_config=LayerConfig(
            activation=options.activation,
            layer_norm_position=options.layer_norm_position,
            residual_connection_option=options.residual_connection_option,
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
        if not self.recurrent_controller_options.recurrent_gate_flag:
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

    def build_halting_config(self) -> StickBreakingConfig | None:
        if not self.layer_controller_options.stack_halting_flag:
            return None
        controller = self.layer_controller_options
        options = resolve_experts_controller_stack_options(
            controller.halting_stack_source, self.__halting_stack_defaults()
        )
        return StickBreakingConfig(
            threshold=controller.halting_threshold,
            halting_dropout=controller.halting_dropout,
            hidden_state_mode=controller.halting_hidden_state_mode,
            halting_gate_config=self.__build_halting_gate_stack(options),
        )

    def build_recurrent_halting_config(self) -> StickBreakingConfig | None:
        if not self.recurrent_controller_options.recurrent_halting_flag:
            return None
        controller = self.recurrent_controller_options
        options = resolve_experts_controller_stack_options(
            controller.recurrent_halting_stack_source,
            self.__recurrent_halting_stack_defaults(),
        )
        return StickBreakingConfig(
            threshold=controller.recurrent_halting_threshold,
            halting_dropout=controller.recurrent_halting_dropout,
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
            recurrent_layer_norm_position=self.recurrent_controller_options.recurrent_layer_norm_position,
            block_config=block_config,
            gate_config=self.gate_config_factory.build_recurrent_gate_config(),
            residual_connection_option=ResidualConnectionOptions.DISABLED,
            halting_config=self.halting_config_factory.build_recurrent_halting_config(),
        )


@dataclass(frozen=True)
class HiddenModelConfigDependencies:
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


class HiddenModelConfigFactory:
    def __init__(self, dependencies: HiddenModelConfigDependencies) -> None:
        self._hidden_dim = dependencies.hidden_dim
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
        self.router_stack_options = self.__default_router_stack_options(
            dependencies.router_stack_options
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
        self, stack_options: ExpertsStackOptions | None
    ) -> ExpertsStackOptions:
        if stack_options is not None:
            return stack_options
        return ExpertsStackOptions(
            hidden_dim=self._hidden_dim,
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
        self, submodule_stack_options: ExpertsSubmoduleStackOptions | None
    ) -> ExpertsSubmoduleStackOptions:
        if submodule_stack_options is not None:
            return submodule_stack_options
        return ExpertsSubmoduleStackOptions(
            hidden_dim=config.SUBMODULE_STACK_HIDDEN_DIM,
            num_layers=config.SUBMODULE_STACK_NUM_LAYERS,
            last_layer_bias_option=config.SUBMODULE_STACK_LAST_LAYER_BIAS_OPTION,
            apply_output_pipeline_flag=config.SUBMODULE_STACK_APPLY_OUTPUT_PIPELINE_FLAG,
            activation=config.SUBMODULE_STACK_ACTIVATION,
            layer_norm_position=config.SUBMODULE_STACK_LAYER_NORM_POSITION,
            residual_connection_option=config.SUBMODULE_STACK_RESIDUAL_CONNECTION_OPTION,
            dropout_probability=config.SUBMODULE_STACK_DROPOUT_PROBABILITY,
            bias_flag=config.SUBMODULE_STACK_BIAS_FLAG,
        )

    def __default_mixture_options(
        self, mixture_options: ExpertsMixtureOptions | None
    ) -> ExpertsMixtureOptions:
        if mixture_options is not None:
            return mixture_options
        return ExpertsMixtureOptions(
            top_k=config.EXPERT_TOP_K,
            num_experts=config.EXPERT_NUM_EXPERTS,
            capacity_factor=config.EXPERT_CAPACITY_FACTOR,
            dropped_token_behavior=config.EXPERT_DROPPED_TOKEN_BEHAVIOR,
            compute_expert_mixture_flag=config.EXPERT_COMPUTE_EXPERT_MIXTURE_FLAG,
            weighted_parameters_flag=config.EXPERT_WEIGHTED_PARAMETERS_FLAG,
            weighting_position_option=config.EXPERT_WEIGHTING_POSITION_OPTION,
            routing_initialization_mode=config.EXPERT_ROUTING_INITIALIZATION_MODE,
        )

    def __default_expert_stack_options(
        self, expert_stack_options: ExpertsSubmoduleStackOptions | None
    ) -> ExpertsSubmoduleStackOptions:
        if expert_stack_options is not None:
            return expert_stack_options
        return resolve_experts_submodule_stack_options(
            self.submodule_stack_options,
            layer_norm_position=config.EXPERT_STACK_LAYER_NORM_POSITION,
            apply_output_pipeline_flag=config.EXPERT_STACK_APPLY_OUTPUT_PIPELINE_FLAG,
        )

    def __default_sampler_options(
        self, sampler_options: ExpertsSamplerOptions | None
    ) -> ExpertsSamplerOptions:
        if sampler_options is not None:
            return sampler_options
        return ExpertsSamplerOptions(
            threshold=config.SAMPLER_THRESHOLD,
            filter_above_threshold=config.SAMPLER_FILTER_ABOVE_THRESHOLD,
            num_topk_samples=config.SAMPLER_NUM_TOPK_SAMPLES,
            normalize_probabilities_flag=config.SAMPLER_NORMALIZE_PROBABILITIES_FLAG,
            noisy_topk_flag=config.SAMPLER_NOISY_TOPK_FLAG,
            coefficient_of_variation_loss_weight=config.SAMPLER_COEFFICIENT_OF_VARIATION_LOSS_WEIGHT,
            switch_loss_weight=config.SAMPLER_SWITCH_LOSS_WEIGHT,
            zero_centred_loss_weight=config.SAMPLER_ZERO_CENTRED_LOSS_WEIGHT,
            mutual_information_loss_weight=config.SAMPLER_MUTUAL_INFORMATION_LOSS_WEIGHT,
        )

    def __default_router_options(
        self, router_options: ExpertsRouterOptions | None
    ) -> ExpertsRouterOptions:
        if router_options is not None:
            return router_options
        return ExpertsRouterOptions(noisy_topk_flag=config.ROUTER_NOISY_TOPK_FLAG)

    def __default_router_stack_options(
        self, router_stack_options: ExpertsSubmoduleStackOptions | None
    ) -> ExpertsSubmoduleStackOptions:
        if router_stack_options is not None:
            return router_stack_options
        return ExpertsSubmoduleStackOptions(
            hidden_dim=config.ROUTER_STACK_HIDDEN_DIM,
            num_layers=config.ROUTER_STACK_NUM_LAYERS,
            last_layer_bias_option=config.ROUTER_STACK_LAST_LAYER_BIAS_OPTION,
            apply_output_pipeline_flag=config.ROUTER_STACK_APPLY_OUTPUT_PIPELINE_FLAG,
            activation=config.ROUTER_STACK_ACTIVATION,
            layer_norm_position=config.ROUTER_STACK_LAYER_NORM_POSITION,
            residual_connection_option=config.ROUTER_STACK_RESIDUAL_CONNECTION_OPTION,
            dropout_probability=config.ROUTER_STACK_DROPOUT_PROBABILITY,
            bias_flag=config.ROUTER_BIAS_FLAG,
        )

    def __default_layer_controller_options(
        self, layer_controller_options: ExpertsLayerControllerOptions | None
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
        self, dynamic_memory_options: ExpertsDynamicMemoryOptions | None
    ) -> ExpertsDynamicMemoryOptions:
        if dynamic_memory_options is not None:
            return dynamic_memory_options
        return ExpertsDynamicMemoryOptions(
            memory_flag=config.MEMORY_FLAG,
            memory_option=config.MEMORY_OPTION,
            memory_position_option=config.MEMORY_POSITION_OPTION,
            memory_test_time_training_learning_rate=config.MEMORY_TEST_TIME_TRAINING_LEARNING_RATE,
            memory_test_time_training_num_inner_steps=config.MEMORY_TEST_TIME_TRAINING_NUM_INNER_STEPS,
            memory_stack_source=self.__default_controller_stack_source("MEMORY_STACK"),
        )

    def __default_recurrent_controller_options(
        self, recurrent_controller_options: ExpertsRecurrentControllerOptions | None
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
            recurrent_halting_hidden_state_mode=config.RECURRENT_HALTING_HIDDEN_STATE_MODE,
            recurrent_halting_stack_source=self.__default_controller_stack_source(
                "RECURRENT_HALTING_STACK"
            ),
        )

    def __default_expert_layer_controller_options(
        self, layer_controller_options: ExpertsLayerControllerOptions | None
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
        self, dynamic_memory_options: ExpertsDynamicMemoryOptions | None
    ) -> ExpertsDynamicMemoryOptions:
        if dynamic_memory_options is not None:
            return dynamic_memory_options
        return ExpertsDynamicMemoryOptions(
            memory_flag=config.EXPERT_MEMORY_FLAG,
            memory_option=config.EXPERT_MEMORY_OPTION,
            memory_position_option=config.EXPERT_MEMORY_POSITION_OPTION,
            memory_test_time_training_learning_rate=config.EXPERT_MEMORY_TEST_TIME_TRAINING_LEARNING_RATE,
            memory_test_time_training_num_inner_steps=config.EXPERT_MEMORY_TEST_TIME_TRAINING_NUM_INNER_STEPS,
            memory_stack_source=self.__default_controller_stack_source(
                "EXPERT_MEMORY_STACK"
            ),
        )

    def __default_expert_recurrent_controller_options(
        self, recurrent_controller_options: ExpertsRecurrentControllerOptions | None
    ) -> ExpertsRecurrentControllerOptions:
        if recurrent_controller_options is not None:
            return recurrent_controller_options
        return ExpertsRecurrentControllerOptions(
            recurrent_flag=config.EXPERT_RECURRENT_FLAG,
            recurrent_max_steps=config.EXPERT_RECURRENT_MAX_STEPS,
            recurrent_layer_norm_position=config.EXPERT_RECURRENT_LAYER_NORM_POSITION,
            recurrent_gate_flag=config.EXPERT_RECURRENT_GATE_FLAG,
            recurrent_gate_option=config.EXPERT_RECURRENT_GATE_OPTION,
            recurrent_gate_activation=config.EXPERT_RECURRENT_GATE_ACTIVATION,
            recurrent_gate_stack_source=self.__default_controller_stack_source(
                "EXPERT_RECURRENT_GATE_STACK"
            ),
            recurrent_halting_flag=config.EXPERT_RECURRENT_HALTING_FLAG,
            recurrent_halting_threshold=config.EXPERT_RECURRENT_HALTING_THRESHOLD,
            recurrent_halting_dropout=config.EXPERT_RECURRENT_HALTING_DROPOUT,
            recurrent_halting_hidden_state_mode=config.EXPERT_RECURRENT_HALTING_HIDDEN_STATE_MODE,
            recurrent_halting_stack_source=self.__default_controller_stack_source(
                "EXPERT_RECURRENT_HALTING_STACK"
            ),
        )

    def __default_controller_stack_source(
        self, prefix: str
    ) -> ExpertsSubmoduleStackSource:
        return ExpertsSubmoduleStackSource(
            independent_flag=getattr(config, f"{prefix}_INDEPENDENT_FLAG"),
            hidden_dim=getattr(config, f"{prefix}_HIDDEN_DIM"),
            num_layers=getattr(config, f"{prefix}_NUM_LAYERS"),
            last_layer_bias_option=getattr(config, f"{prefix}_LAST_LAYER_BIAS_OPTION"),
            apply_output_pipeline_flag=getattr(
                config, f"{prefix}_APPLY_OUTPUT_PIPELINE_FLAG"
            ),
            activation=getattr(config, f"{prefix}_ACTIVATION"),
            layer_norm_position=getattr(config, f"{prefix}_LAYER_NORM_POSITION"),
            residual_connection_option=getattr(
                config, f"{prefix}_RESIDUAL_CONNECTION_OPTION"
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
            apply_output_pipeline_flag=stack_options.apply_output_pipeline_flag,
            shared_gate_config=layer_controller.shared_gate_config,
            shared_memory_config=memory_config,
            layer_config=layer_config,
        )

    def __build_layer_config(
        self, gate_config: GateConfig | None, halting_config: StickBreakingConfig | None
    ) -> LayerConfig:
        stack_options = self.stack_options
        return MixtureOfExpertsLayerConfig(
            activation=stack_options.activation,
            layer_norm_position=stack_options.layer_norm_position,
            residual_connection_option=stack_options.residual_connection_option,
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
            apply_output_pipeline_flag=expert_stack_options.apply_output_pipeline_flag,
            shared_gate_config=layer_controller.shared_gate_config,
            shared_memory_config=memory_config,
            layer_config=LayerConfig(
                activation=expert_stack_options.activation,
                layer_norm_position=expert_stack_options.layer_norm_position,
                residual_connection_option=expert_stack_options.residual_connection_option,
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
        model_config = self.__build_controller_stack(self.router_stack_options)
        return RouterConfig(
            input_dim=self.hidden_dim,
            num_experts=mixture_options.num_experts,
            noisy_topk_flag=router_options.noisy_topk_flag,
            model_config=model_config,
        )

    def __build_controller_stack(
        self, options: ExpertsSubmoduleStackOptions
    ) -> LayerStackConfig:
        return build_linear_controller_stack(options)
