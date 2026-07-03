from dataclasses import dataclass, replace

from emperor.base.layer.config import (
    LayerConfig,
    LayerStackConfig,
    RecurrentLayerConfig,
)
from emperor.base.layer.gate import GateConfig
from emperor.base.options import LastLayerBiasOptions
from emperor.memory.config import DynamicMemoryConfig
from emperor.halting.config import StickBreakingConfig
from emperor.linears.core.config import LinearLayerConfig
from emperor.base.layer.residual import ResidualConnectionOptions
from models.linears.linear._controller_stack import build_linear_controller_stack

from models.linears._controller_stack import (
    ControllerStackOptions,
    ControllerStackSource,
    resolve_controller_stack_options,
    resolve_enabled,
)
from models.linears._builder_options import (
    DynamicMemoryOptions,
    LayerControllerOptions,
    LinearStackOptions,
    RecurrentControllerOptions,
)

import models.linears.linear.config as config

STICK_BREAKING_GATE_OUTPUT_DIM = 2


@dataclass(frozen=True)
class ControlConfigDependencies:
    stack_options: LinearStackOptions | None
    submodule_stack_options: ControllerStackOptions | None
    layer_controller_options: LayerControllerOptions | None
    dynamic_memory_options: DynamicMemoryOptions | None
    recurrent_controller_options: RecurrentControllerOptions | None
    output_dim: int


class ControlConfigFactory:
    def __init__(self, dependencies: ControlConfigDependencies) -> None:
        stack_options = self.__default_stack_options(
            dependencies.stack_options
        )
        submodule_stack_options = self.__default_submodule_stack_options(
            dependencies.submodule_stack_options
        )
        layer_controller_options = self.__default_layer_controller_options(
            dependencies.layer_controller_options
        )
        dynamic_memory_options = self.__default_dynamic_memory_options(
            dependencies.dynamic_memory_options
        )
        recurrent_controller_options = self.__default_recurrent_controller_options(
            dependencies.recurrent_controller_options
        )

        self.stack_options = stack_options
        self.submodule_stack_options = submodule_stack_options
        self.layer_controller_options = layer_controller_options
        self.dynamic_memory_options = dynamic_memory_options
        self.recurrent_controller_options = recurrent_controller_options
        self.output_dim = dependencies.output_dim

    def __default_stack_options(
        self,
        stack_options: LinearStackOptions | None,
    ) -> LinearStackOptions:
        if stack_options is not None:
            return stack_options
        return LinearStackOptions(
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
        submodule_stack_options: ControllerStackOptions | None,
    ) -> ControllerStackOptions:
        if submodule_stack_options is not None:
            return submodule_stack_options
        return ControllerStackOptions(
            hidden_dim=config.SUBMODULE_STACK_HIDDEN_DIM,
            num_layers=config.SUBMODULE_STACK_NUM_LAYERS,
            last_layer_bias_option=(
                config.SUBMODULE_STACK_LAST_LAYER_BIAS_OPTION
            ),
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

    def __default_layer_controller_options(
        self,
        layer_controller_options: LayerControllerOptions | None,
    ) -> LayerControllerOptions:
        if layer_controller_options is not None:
            return layer_controller_options
        gate_stack_source = self.__default_controller_stack_source("gate_stack")
        halting_stack_source = self.__default_controller_stack_source(
            "halting_stack"
        )
        return LayerControllerOptions(
            stack_gate_flag=config.GATE_FLAG,
            gate_option=config.GATE_OPTION,
            gate_activation=config.GATE_ACTIVATION,
            gate_stack_source=gate_stack_source,
            stack_halting_flag=config.HALTING_FLAG,
            halting_threshold=config.HALTING_THRESHOLD,
            halting_dropout=config.HALTING_DROPOUT,
            halting_hidden_state_mode=config.HALTING_HIDDEN_STATE_MODE,
            halting_stack_source=halting_stack_source,
        )

    def __default_dynamic_memory_options(
        self,
        dynamic_memory_options: DynamicMemoryOptions | None,
    ) -> DynamicMemoryOptions:
        if dynamic_memory_options is not None:
            return dynamic_memory_options
        memory_stack_source = self.__default_controller_stack_source("memory_stack")
        return DynamicMemoryOptions(
            memory_flag=config.MEMORY_FLAG,
            memory_option=config.MEMORY_OPTION,
            memory_position_option=config.MEMORY_POSITION_OPTION,
            memory_test_time_training_learning_rate=(
                config.MEMORY_TEST_TIME_TRAINING_LEARNING_RATE
            ),
            memory_test_time_training_num_inner_steps=(
                config.MEMORY_TEST_TIME_TRAINING_NUM_INNER_STEPS
            ),
            memory_stack_source=memory_stack_source,
        )

    def __default_recurrent_controller_options(
        self,
        recurrent_controller_options: RecurrentControllerOptions | None,
    ) -> RecurrentControllerOptions:
        if recurrent_controller_options is not None:
            return recurrent_controller_options
        recurrent_gate_stack_source = self.__default_controller_stack_source(
            "recurrent_gate_stack"
        )
        recurrent_halting_stack_source = self.__default_controller_stack_source(
            "recurrent_halting_stack"
        )
        return RecurrentControllerOptions(
            recurrent_flag=config.RECURRENT_FLAG,
            recurrent_max_steps=config.RECURRENT_MAX_STEPS,
            recurrent_layer_norm_position=config.RECURRENT_LAYER_NORM_POSITION,
            recurrent_gate_flag=config.RECURRENT_GATE_FLAG,
            recurrent_gate_option=config.RECURRENT_GATE_OPTION,
            recurrent_gate_activation=config.RECURRENT_GATE_ACTIVATION,
            recurrent_gate_stack_source=recurrent_gate_stack_source,
            recurrent_halting_flag=config.RECURRENT_HALTING_FLAG,
            recurrent_halting_threshold=config.RECURRENT_HALTING_THRESHOLD,
            recurrent_halting_dropout=config.RECURRENT_HALTING_DROPOUT,
            recurrent_halting_hidden_state_mode=(
                config.RECURRENT_HALTING_HIDDEN_STATE_MODE
            ),
            recurrent_halting_stack_source=recurrent_halting_stack_source,
        )

    def __default_controller_stack_source(
        self,
        prefix: str,
    ) -> ControllerStackSource:
        config_prefix = prefix.upper()
        return ControllerStackSource(
            independent_flag=getattr(
                config,
                f"{config_prefix}_INDEPENDENT_FLAG",
            ),
            hidden_dim=getattr(config, f"{config_prefix}_HIDDEN_DIM"),
            num_layers=getattr(config, f"{config_prefix}_NUM_LAYERS"),
            last_layer_bias_option=getattr(
                config,
                f"{config_prefix}_LAST_LAYER_BIAS_OPTION",
            ),
            apply_output_pipeline_flag=getattr(
                config,
                f"{config_prefix}_APPLY_OUTPUT_PIPELINE_FLAG",
            ),
            activation=getattr(config, f"{config_prefix}_ACTIVATION"),
            layer_norm_position=getattr(
                config,
                f"{config_prefix}_LAYER_NORM_POSITION",
            ),
            residual_connection_option=getattr(
                config,
                f"{config_prefix}_RESIDUAL_CONNECTION_OPTION",
            ),
            dropout_probability=getattr(
                config,
                f"{config_prefix}_DROPOUT_PROBABILITY",
            ),
            bias_flag=getattr(config, f"{config_prefix}_BIAS_FLAG"),
        )

    def build_hidden_model_config(self) -> LayerStackConfig | RecurrentLayerConfig:
        gate_config = self.__build_gate_config()
        halting_config = self.__build_halting_config()
        memory_config = self.__build_memory_config()
        layer_stack_config = self.__build_stack_config(
            gate_config=gate_config,
            halting_config=halting_config,
            memory_config=memory_config,
        )
        model_config = self.__maybe_wrap_recurrent(layer_stack_config)
        return model_config

    def __build_gate_config(
        self,
        enabled_flag: bool | None = None,
    ) -> GateConfig | None:
        layer_controller = self.layer_controller_options

        enabled_flag = resolve_enabled(
            enabled_flag,
            layer_controller.stack_gate_flag,
        )
        if not enabled_flag:
            return None
        model_config = self.__build_gate_model_config(enabled_flag)
        return GateConfig(
            model_config=model_config,
            option=layer_controller.gate_option,
            activation=layer_controller.gate_activation,
        )

    def __build_halting_config(
        self,
        enabled_flag: bool | None = None,
    ) -> StickBreakingConfig | None:
        layer_controller = self.layer_controller_options

        enabled_flag = resolve_enabled(
            enabled_flag,
            layer_controller.stack_halting_flag,
        )
        if not enabled_flag:
            return None
        halting_stack_source = self.__halting_stack_source()
        submodule_stack_defaults = self.__submodule_stack_defaults(
            last_layer_bias_option=LastLayerBiasOptions.DISABLED
        )
        halting_gate_config = self.__build_halting_gate_stack(
            halting_stack_source,
            submodule_stack_defaults,
        )
        return StickBreakingConfig(
            threshold=layer_controller.halting_threshold,
            halting_dropout=layer_controller.halting_dropout,
            hidden_state_mode=layer_controller.halting_hidden_state_mode,
            halting_gate_config=halting_gate_config,
        )

    def __build_memory_config(
        self,
        enabled_flag: bool | None = None,
    ) -> DynamicMemoryConfig | None:
        memory_options = self.dynamic_memory_options
        stack_options = self.stack_options

        enabled_flag = resolve_enabled(
            enabled_flag,
            memory_options.memory_flag,
        )
        if not enabled_flag:
            return None
        memory_stack_source = self.__memory_stack_source()
        submodule_stack_defaults = self.__submodule_stack_defaults()
        model_config = self.__build_controller_stack(
            memory_stack_source,
            submodule_stack_defaults,
        )
        memory_config = memory_options.memory_option
        return memory_config(
            input_dim=stack_options.hidden_dim,
            output_dim=stack_options.hidden_dim,
            memory_position_option=memory_options.memory_position_option,
            test_time_training_learning_rate=(
                memory_options.memory_test_time_training_learning_rate
            ),
            test_time_training_num_inner_steps=(
                memory_options.memory_test_time_training_num_inner_steps
            ),
            model_config=model_config,
        )

    def __build_stack_config(
        self,
        *,
        gate_config: GateConfig | None,
        halting_config: StickBreakingConfig | None,
        memory_config: DynamicMemoryConfig | None,
    ) -> LayerStackConfig:
        stack_options = self.stack_options

        layer_config = self.__build_layer_config(
            gate_config=gate_config,
            halting_config=halting_config,
        )
        return LayerStackConfig(
            hidden_dim=stack_options.hidden_dim,
            num_layers=stack_options.num_layers,
            last_layer_bias_option=stack_options.last_layer_bias_option,
            apply_output_pipeline_flag=(
                stack_options.apply_output_pipeline_flag
            ),
            shared_gate_config=(
                self.layer_controller_options.shared_gate_config
            ),
            shared_memory_config=memory_config,
            layer_config=layer_config,
        )

    def __build_layer_config(
        self,
        *,
        gate_config: GateConfig | None,
        halting_config: StickBreakingConfig | None,
    ) -> LayerConfig:
        stack_options = self.stack_options
        layer_model_config = LinearLayerConfig(
            bias_flag=stack_options.bias_flag,
        )
        return LayerConfig(
            activation=stack_options.activation,
            layer_norm_position=stack_options.layer_norm_position,
            residual_connection_option=stack_options.residual_connection_option,
            dropout_probability=stack_options.dropout_probability,
            gate_config=gate_config,
            halting_config=halting_config,
            layer_model_config=layer_model_config,
        )

    def __maybe_wrap_recurrent(
        self, block_config: LayerStackConfig
    ) -> LayerStackConfig | RecurrentLayerConfig:
        recurrent_options = self.recurrent_controller_options

        if not recurrent_options.recurrent_flag:
            return block_config
        gate_config = self.__build_recurrent_gate_config()
        halting_config = self.__build_recurrent_halting_config()
        return RecurrentLayerConfig(
            max_steps=recurrent_options.recurrent_max_steps,
            recurrent_layer_norm_position=(
                recurrent_options.recurrent_layer_norm_position
            ),
            block_config=block_config,
            gate_config=gate_config,
            residual_connection_option=ResidualConnectionOptions.DISABLED,
            halting_config=halting_config,
        )

    def __build_gate_model_config(
        self, enabled_flag: bool | None = None
    ) -> LayerStackConfig | None:
        layer_controller = self.layer_controller_options

        enabled_flag = resolve_enabled(
            enabled_flag,
            layer_controller.stack_gate_flag,
        )
        if not enabled_flag:
            return None
        gate_stack_source = self.__gate_stack_source()
        submodule_stack_defaults = self.__submodule_stack_defaults()
        model_config = self.__build_controller_stack(
            gate_stack_source,
            submodule_stack_defaults,
        )
        return model_config

    def __build_recurrent_gate_config(self) -> GateConfig | None:
        recurrent_options = self.recurrent_controller_options

        if not recurrent_options.recurrent_gate_flag:
            return None
        gate_stack_source = self.__gate_stack_source()
        submodule_stack_defaults = self.__submodule_stack_defaults()
        gate_defaults = resolve_controller_stack_options(
            gate_stack_source,
            submodule_stack_defaults,
        )
        recurrent_gate_stack_source = self.__recurrent_gate_stack_source()
        model_config = self.__build_controller_stack(
            recurrent_gate_stack_source,
            gate_defaults,
        )
        return GateConfig(
            model_config=model_config,
            option=recurrent_options.recurrent_gate_option,
            activation=recurrent_options.recurrent_gate_activation,
        )

    def __build_recurrent_halting_config(self) -> StickBreakingConfig | None:
        recurrent_options = self.recurrent_controller_options

        if not recurrent_options.recurrent_halting_flag:
            return None
        halting_stack_source = self.__halting_stack_source()
        submodule_stack_defaults = self.__submodule_stack_defaults(
            last_layer_bias_option=LastLayerBiasOptions.DISABLED
        )
        halting_defaults = resolve_controller_stack_options(
            halting_stack_source,
            submodule_stack_defaults,
        )
        recurrent_halting_stack_source = self.__recurrent_halting_stack_source()
        halting_gate_config = self.__build_halting_gate_stack(
            recurrent_halting_stack_source,
            halting_defaults,
        )
        return StickBreakingConfig(
            threshold=recurrent_options.recurrent_halting_threshold,
            halting_dropout=recurrent_options.recurrent_halting_dropout,
            hidden_state_mode=(
                recurrent_options.recurrent_halting_hidden_state_mode
            ),
            halting_gate_config=halting_gate_config,
        )

    def __build_controller_stack(
        self,
        source: ControllerStackSource,
        defaults: ControllerStackOptions,
    ) -> LayerStackConfig:
        options = resolve_controller_stack_options(
            source,
            defaults,
        )
        return build_linear_controller_stack(options)

    def __build_halting_gate_stack(
        self,
        source: ControllerStackSource,
        defaults: ControllerStackOptions,
    ) -> LayerStackConfig:
        options = resolve_controller_stack_options(
            source,
            defaults,
        )
        halting_hidden_dim = options.hidden_dim or self.output_dim
        return build_linear_controller_stack(
            options,
            hidden_dim=halting_hidden_dim,
            output_dim=STICK_BREAKING_GATE_OUTPUT_DIM,
        )

    def __submodule_stack_defaults(
        self,
        *,
        last_layer_bias_option: LastLayerBiasOptions | None = None,
    ) -> ControllerStackOptions:
        if last_layer_bias_option is None:
            return self.submodule_stack_options
        return replace(
            self.submodule_stack_options,
            last_layer_bias_option=last_layer_bias_option,
        )

    def __gate_stack_source(self) -> ControllerStackSource:
        return self.layer_controller_options.gate_stack_source

    def __halting_stack_source(self) -> ControllerStackSource:
        return self.layer_controller_options.halting_stack_source

    def __memory_stack_source(self) -> ControllerStackSource:
        return self.dynamic_memory_options.memory_stack_source

    def __recurrent_gate_stack_source(self) -> ControllerStackSource:
        return self.recurrent_controller_options.recurrent_gate_stack_source

    def __recurrent_halting_stack_source(self) -> ControllerStackSource:
        return self.recurrent_controller_options.recurrent_halting_stack_source
