from dataclasses import replace
from typing import Any

from emperor.config import ConfigBase
from emperor.halting import HaltingConfig
from emperor.layers import (
    GateConfig,
    LastLayerBiasOptions,
    LayerConfig,
    LayerStackConfig,
    RecurrentLayerConfig,
    ResidualConfig,
)
from emperor.linears import LinearLayerConfig
from emperor.memory import DynamicMemoryConfig
from models.vit.expert_linear_adaptive.runtime_options import (
    ExpertsAdaptiveGeneratorStackOptions,
    ExpertsDynamicMemoryOptions,
    ExpertsLayerControllerOptions,
    ExpertsRecurrentControllerOptions,
    ExpertsStackOptions,
    ExpertsSubmoduleStackOptions,
    resolve_experts_controller_stack_options,
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
            residual_config=None
            if options.residual_connection_option is None
            else ResidualConfig(option=options.residual_connection_option),
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

    def build_halting_config(self) -> HaltingConfig | None:
        if not self.layer_controller_options.stack_halting_flag:
            return None
        controller = self.layer_controller_options
        options = resolve_experts_controller_stack_options(
            controller.halting_stack_source,
            self.__halting_stack_defaults(),
        )
        return controller.halting_option(
            threshold=controller.halting_threshold,
            dropout_probability=controller.halting_dropout,
            hidden_state_mode=controller.halting_hidden_state_mode,
            halting_gate_config=self.__build_halting_gate_stack(options),
        )

    def build_recurrent_halting_config(self) -> HaltingConfig | None:
        if not self.recurrent_controller_options.recurrent_halting_flag:
            return None
        controller = self.recurrent_controller_options
        options = resolve_experts_controller_stack_options(
            controller.recurrent_halting_stack_source,
            self.__recurrent_halting_stack_defaults(),
        )
        return controller.recurrent_halting_option(
            threshold=controller.recurrent_halting_threshold,
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
            test_time_training_learning_rate=(
                self.dynamic_memory_options.memory_test_time_training_learning_rate
            ),
            test_time_training_num_inner_steps=(
                self.dynamic_memory_options.memory_test_time_training_num_inner_steps
            ),
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
            recurrent_layer_norm_position=(
                self.recurrent_controller_options.recurrent_layer_norm_position
            ),
            block_config=block_config,
            gate_config=self.gate_config_factory.build_recurrent_gate_config(),
            residual_config=None,
            halting_config=self.halting_config_factory.build_recurrent_halting_config(),
        )
