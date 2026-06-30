from dataclasses import replace
from typing import Any

from emperor.base.layer.config import (
    LayerConfig,
    LayerStackConfig,
    RecurrentLayerConfig,
)
from emperor.base.layer.gate import GateConfig
from emperor.base.layer.residual import ResidualConnectionOptions
from emperor.base.options import LastLayerBiasOptions
from emperor.halting.config import StickBreakingConfig
from emperor.linears.core.config import LinearLayerConfig
from emperor.memory.config import DynamicMemoryConfig

from models.linears._controller_stack import (
    ControllerStackOptions,
    ControllerStackSource,
    resolve_controller_stack_options,
    resolve_enabled,
)
from models.linears.linear._controller_stack import build_linear_controller_stack

STICK_BREAKING_GATE_OUTPUT_DIM = 2


class ControlConfigFactory:
    def __init__(self, builder: Any) -> None:
        self.builder = builder

    def build(self) -> LayerStackConfig | RecurrentLayerConfig:
        gate_config = self.__build_gate_config()
        self.__validate_shared_gate_config(gate_config)
        halting_config = self.__build_halting_config()
        memory_config = self.__build_memory_config()
        layer_stack_config = self.__build_stack_config(
            gate_config=gate_config,
            halting_config=halting_config,
            memory_config=memory_config,
        )
        return self.__maybe_wrap_recurrent(layer_stack_config)

    def __build_stack_config(
        self,
        *,
        gate_config: GateConfig | None,
        halting_config: StickBreakingConfig | None,
        memory_config: DynamicMemoryConfig | None,
    ) -> LayerStackConfig:
        builder = self.builder
        stack_options = builder.stack_options
        return LayerStackConfig(
            hidden_dim=stack_options.hidden_dim,
            num_layers=stack_options.num_layers,
            last_layer_bias_option=stack_options.last_layer_bias_option,
            apply_output_pipeline_flag=stack_options.apply_output_pipeline_flag,
            shared_gate_config=builder.layer_controller_options.shared_gate_config,
            shared_memory_config=memory_config,
            layer_config=LayerConfig(
                activation=stack_options.activation,
                layer_norm_position=stack_options.layer_norm_position,
                residual_connection_option=(
                    stack_options.residual_connection_option
                ),
                dropout_probability=stack_options.dropout_probability,
                gate_config=gate_config,
                halting_config=halting_config,
                layer_model_config=LinearLayerConfig(
                    bias_flag=stack_options.bias_flag,
                ),
            ),
        )

    def __validate_shared_gate_config(self, gate_config: GateConfig | None) -> None:
        if self.__is_active_gate_config(
            self.builder.shared_gate_config
        ) and self.__is_active_gate_config(gate_config):
            raise ValueError(
                "shared_gate_config cannot be provided when stack_gate_flag "
                "enables per-layer gate_config."
            )

    @staticmethod
    def __is_active_gate_config(gate_config: GateConfig | None) -> bool:
        return gate_config is not None

    def __maybe_wrap_recurrent(
        self, block_config: LayerStackConfig
    ) -> LayerStackConfig | RecurrentLayerConfig:
        recurrent_options = self.builder.recurrent_controller_options
        if not recurrent_options.recurrent_flag:
            return block_config
        return RecurrentLayerConfig(
            max_steps=recurrent_options.recurrent_max_steps,
            recurrent_layer_norm_position=(
                recurrent_options.recurrent_layer_norm_position
            ),
            block_config=block_config,
            gate_config=self.__build_recurrent_gate_config(),
            residual_connection_option=ResidualConnectionOptions.DISABLED,
            halting_config=self.__build_recurrent_halting_config(),
        )

    def __build_gate_config(self, enabled: bool | None = None) -> GateConfig | None:
        layer_controller = self.builder.layer_controller_options
        enabled = resolve_enabled(enabled, layer_controller.stack_gate_flag)
        if not enabled:
            return None
        model_config = self.__build_gate_model_config(enabled)
        return GateConfig(
            model_config=model_config,
            option=layer_controller.gate_option,
            activation=layer_controller.gate_activation,
        )

    def __build_gate_model_config(
        self, enabled: bool | None = None
    ) -> LayerStackConfig | None:
        enabled = resolve_enabled(
            enabled,
            self.builder.layer_controller_options.stack_gate_flag,
        )
        if not enabled:
            return None
        options = resolve_controller_stack_options(
            self.__gate_stack_source(),
            self.__submodule_stack_defaults(),
        )
        return build_linear_controller_stack(options)

    def __build_recurrent_gate_config(self) -> GateConfig | None:
        recurrent_options = self.builder.recurrent_controller_options
        if not recurrent_options.recurrent_gate_flag:
            return None
        gate_defaults = resolve_controller_stack_options(
            self.__gate_stack_source(),
            self.__submodule_stack_defaults(),
        )
        options = resolve_controller_stack_options(
            self.__recurrent_gate_stack_source(),
            gate_defaults,
        )
        return GateConfig(
            model_config=build_linear_controller_stack(options),
            option=recurrent_options.recurrent_gate_option,
            activation=recurrent_options.recurrent_gate_activation,
        )

    def __build_halting_config(
        self,
        enabled: bool | None = None,
    ) -> StickBreakingConfig | None:
        layer_controller = self.builder.layer_controller_options
        enabled = resolve_enabled(enabled, layer_controller.stack_halting_flag)
        if not enabled:
            return None
        options = resolve_controller_stack_options(
            self.__halting_stack_source(),
            self.__submodule_stack_defaults(
                last_layer_bias_option=LastLayerBiasOptions.DISABLED
            ),
        )
        return StickBreakingConfig(
            threshold=layer_controller.halting_threshold,
            halting_dropout=layer_controller.halting_dropout,
            hidden_state_mode=layer_controller.halting_hidden_state_mode,
            halting_gate_config=build_linear_controller_stack(
                options,
                hidden_dim=options.hidden_dim or self.builder.output_dim,
                output_dim=STICK_BREAKING_GATE_OUTPUT_DIM,
            ),
        )

    def __build_recurrent_halting_config(self) -> StickBreakingConfig | None:
        recurrent_options = self.builder.recurrent_controller_options
        if not recurrent_options.recurrent_halting_flag:
            return None
        halting_defaults = resolve_controller_stack_options(
            self.__halting_stack_source(),
            self.__submodule_stack_defaults(
                last_layer_bias_option=LastLayerBiasOptions.DISABLED
            ),
        )
        options = resolve_controller_stack_options(
            self.__recurrent_halting_stack_source(),
            halting_defaults,
        )
        return StickBreakingConfig(
            threshold=recurrent_options.recurrent_halting_threshold,
            halting_dropout=recurrent_options.recurrent_halting_dropout,
            hidden_state_mode=recurrent_options.recurrent_halting_hidden_state_mode,
            halting_gate_config=build_linear_controller_stack(
                options,
                hidden_dim=options.hidden_dim or self.builder.output_dim,
                output_dim=STICK_BREAKING_GATE_OUTPUT_DIM,
            ),
        )

    def __build_memory_config(
        self,
        enabled: bool | None = None,
    ) -> DynamicMemoryConfig | None:
        memory_options = self.builder.dynamic_memory_options
        enabled = resolve_enabled(enabled, memory_options.memory_flag)
        if not enabled:
            return None
        options = resolve_controller_stack_options(
            self.__memory_stack_source(),
            self.__submodule_stack_defaults(),
        )
        return memory_options.memory_option(
            input_dim=self.builder.hidden_dim,
            output_dim=self.builder.hidden_dim,
            memory_position_option=memory_options.memory_position_option,
            test_time_training_learning_rate=(
                memory_options.memory_test_time_training_learning_rate
            ),
            test_time_training_num_inner_steps=(
                memory_options.memory_test_time_training_num_inner_steps
            ),
            model_config=build_linear_controller_stack(options),
        )

    def __submodule_stack_defaults(
        self,
        *,
        last_layer_bias_option: LastLayerBiasOptions | None = None,
    ) -> ControllerStackOptions:
        options = self.builder.submodule_stack_options
        if last_layer_bias_option is None:
            return options
        return replace(options, last_layer_bias_option=last_layer_bias_option)

    def __gate_stack_source(self) -> ControllerStackSource:
        return self.builder.layer_controller_options.gate_stack_source

    def __halting_stack_source(self) -> ControllerStackSource:
        return self.builder.layer_controller_options.halting_stack_source

    def __memory_stack_source(self) -> ControllerStackSource:
        return self.builder.dynamic_memory_options.memory_stack_source

    def __recurrent_gate_stack_source(self) -> ControllerStackSource:
        return self.builder.recurrent_controller_options.recurrent_gate_stack_source

    def __recurrent_halting_stack_source(self) -> ControllerStackSource:
        return (
            self.builder.recurrent_controller_options.recurrent_halting_stack_source
        )
