from dataclasses import replace

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
from models.gpt.linear.runtime_options import (
    DynamicMemoryOptions,
    LayerControllerOptions,
    MainLayerStackOptions,
    RecurrentControllerOptions,
    SubmoduleStackOptions,
    resolve_controller_stack_options,
)


class GateConfigFactory:
    def __init__(
        self,
        *,
        layer_controller_options: LayerControllerOptions,
        recurrent_controller_options: RecurrentControllerOptions,
        submodule_stack_options: SubmoduleStackOptions,
        recurrent_stack_inherits_gate_stack: bool = True,
    ) -> None:
        self.layer_controller_options = layer_controller_options
        self.recurrent_controller_options = recurrent_controller_options
        self.submodule_stack_options = submodule_stack_options
        self.recurrent_stack_inherits_gate_stack = recurrent_stack_inherits_gate_stack

    def build_gate_config(self) -> GateConfig | None:
        if not self.layer_controller_options.stack_gate_flag:
            return None
        model_config = self.__build_gate_model_config()
        return GateConfig(
            model_config=model_config,
            option=self.layer_controller_options.gate_option,
            activation=self.layer_controller_options.gate_activation,
        )

    def build_recurrent_gate_config(self) -> GateConfig | None:
        if not self.recurrent_controller_options.recurrent_gate_flag:
            return None
        resolved_gate_stack_defaults = self.__recurrent_gate_stack_defaults()
        recurrent_gate_stack_source = (
            self.recurrent_controller_options.recurrent_gate_stack_source
        )
        resolved_recurrent_gate_stack_options = resolve_controller_stack_options(
            recurrent_gate_stack_source, resolved_gate_stack_defaults
        )
        model_config = self.__build_controller_stack(
            resolved_recurrent_gate_stack_options
        )
        return GateConfig(
            model_config=model_config,
            option=self.recurrent_controller_options.recurrent_gate_option,
            activation=self.recurrent_controller_options.recurrent_gate_activation,
        )

    def __build_gate_model_config(self) -> LayerStackConfig:
        gate_stack_source = self.layer_controller_options.gate_stack_source
        submodule_stack_defaults = self.submodule_stack_options
        resolved_gate_stack_options = resolve_controller_stack_options(
            gate_stack_source, submodule_stack_defaults
        )
        return self.__build_controller_stack(resolved_gate_stack_options)

    def __recurrent_gate_stack_defaults(self) -> SubmoduleStackOptions:
        if not self.recurrent_stack_inherits_gate_stack:
            return self.submodule_stack_options
        return resolve_controller_stack_options(
            self.layer_controller_options.gate_stack_source,
            self.submodule_stack_options,
        )

    def __build_controller_stack(
        self,
        options: SubmoduleStackOptions,
        *,
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
                halting_config=None,
                gate_config=None,
                memory_config=None,
                layer_model_config=LinearLayerConfig(bias_flag=options.bias_flag),
            ),
        )


STICK_BREAKING_GATE_OUTPUT_DIM = 2


class HaltingConfigFactory:
    def __init__(
        self,
        *,
        layer_controller_options: LayerControllerOptions,
        recurrent_controller_options: RecurrentControllerOptions,
        submodule_stack_options: SubmoduleStackOptions,
        output_dim: int,
        halting_stack_defaults: SubmoduleStackOptions | None = None,
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
        halting_stack_source = self.layer_controller_options.halting_stack_source
        halting_stack_defaults = self.__submodule_stack_defaults(
            last_layer_bias_option=LastLayerBiasOptions.DISABLED
        )
        resolved_halting_stack_options = resolve_controller_stack_options(
            halting_stack_source, halting_stack_defaults
        )
        halting_gate_config = self.__build_halting_gate_stack(
            resolved_halting_stack_options
        )
        return self.layer_controller_options.halting_option(
            threshold=self.layer_controller_options.halting_threshold,
            dropout_probability=self.layer_controller_options.halting_dropout,
            hidden_state_mode=self.layer_controller_options.halting_hidden_state_mode,
            halting_gate_config=halting_gate_config,
        )

    def build_recurrent_halting_config(self) -> HaltingConfig | None:
        if not self.recurrent_controller_options.recurrent_halting_flag:
            return None
        resolved_halting_stack_defaults = self.__recurrent_halting_stack_defaults()
        recurrent_halting_stack_source = (
            self.recurrent_controller_options.recurrent_halting_stack_source
        )
        resolved_recurrent_halting_stack_options = resolve_controller_stack_options(
            recurrent_halting_stack_source, resolved_halting_stack_defaults
        )
        halting_gate_config = self.__build_halting_gate_stack(
            resolved_recurrent_halting_stack_options
        )
        return self.recurrent_controller_options.recurrent_halting_option(
            threshold=self.recurrent_controller_options.recurrent_halting_threshold,
            dropout_probability=self.recurrent_controller_options.recurrent_halting_dropout,
            hidden_state_mode=self.recurrent_controller_options.recurrent_halting_hidden_state_mode,
            halting_gate_config=halting_gate_config,
        )

    def __build_halting_gate_stack(
        self, options: SubmoduleStackOptions
    ) -> LayerStackConfig:
        halting_hidden_dim = options.hidden_dim or self.output_dim
        return self.__build_controller_stack(
            options,
            hidden_dim=halting_hidden_dim,
            output_dim=STICK_BREAKING_GATE_OUTPUT_DIM,
        )

    def __build_controller_stack(
        self,
        options: SubmoduleStackOptions,
        *,
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
                halting_config=None,
                gate_config=None,
                memory_config=None,
                layer_model_config=LinearLayerConfig(bias_flag=options.bias_flag),
            ),
        )

    def __submodule_stack_defaults(
        self, *, last_layer_bias_option: LastLayerBiasOptions | None = None
    ) -> SubmoduleStackOptions:
        if self.halting_stack_defaults is not None:
            return self.halting_stack_defaults
        if last_layer_bias_option is None:
            return self.submodule_stack_options
        return replace(
            self.submodule_stack_options, last_layer_bias_option=last_layer_bias_option
        )

    def __recurrent_halting_stack_defaults(self) -> SubmoduleStackOptions:
        halting_stack_defaults = self.__submodule_stack_defaults(
            last_layer_bias_option=LastLayerBiasOptions.DISABLED
        )
        if not self.recurrent_stack_inherits_halting_stack:
            return halting_stack_defaults
        return resolve_controller_stack_options(
            self.layer_controller_options.halting_stack_source, halting_stack_defaults
        )


class MemoryConfigFactory:
    def __init__(
        self,
        *,
        hidden_dim: int,
        stack_options: MainLayerStackOptions,
        dynamic_memory_options: DynamicMemoryOptions,
        submodule_stack_options: SubmoduleStackOptions,
    ) -> None:
        self.hidden_dim = hidden_dim
        self.stack_options = stack_options
        self.dynamic_memory_options = dynamic_memory_options
        self.submodule_stack_options = submodule_stack_options

    def build_memory_config(self) -> DynamicMemoryConfig | None:
        if not self.dynamic_memory_options.memory_flag:
            return None
        memory_stack_source = self.dynamic_memory_options.memory_stack_source
        submodule_stack_defaults = self.submodule_stack_options
        resolved_memory_stack_options = resolve_controller_stack_options(
            memory_stack_source, submodule_stack_defaults
        )
        model_config = self.__build_controller_stack(resolved_memory_stack_options)
        return self.dynamic_memory_options.memory_option(
            input_dim=self.hidden_dim,
            output_dim=self.hidden_dim,
            memory_position_option=self.dynamic_memory_options.memory_position_option,
            test_time_training_learning_rate=self.dynamic_memory_options.memory_test_time_training_learning_rate,
            test_time_training_num_inner_steps=self.dynamic_memory_options.memory_test_time_training_num_inner_steps,
            model_config=model_config,
        )

    def __build_controller_stack(
        self,
        options: SubmoduleStackOptions,
        *,
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
                halting_config=None,
                gate_config=None,
                memory_config=None,
                layer_model_config=LinearLayerConfig(bias_flag=options.bias_flag),
            ),
        )


class RecurrentConfigFactory:
    def __init__(
        self,
        *,
        recurrent_controller_options: RecurrentControllerOptions,
        gate_config_factory: GateConfigFactory,
        halting_config_factory: HaltingConfigFactory,
    ) -> None:
        self.recurrent_controller_options = recurrent_controller_options
        self.gate_config_factory = gate_config_factory
        self.halting_config_factory = halting_config_factory

    def build_config(
        self,
        block_config: LayerStackConfig,
        *,
        input_dim: int | None = None,
        output_dim: int | None = None,
    ) -> LayerStackConfig | RecurrentLayerConfig:
        if not self.recurrent_controller_options.recurrent_flag:
            return block_config
        gate_config = self.gate_config_factory.build_recurrent_gate_config()
        halting_config = self.halting_config_factory.build_recurrent_halting_config()
        return RecurrentLayerConfig(
            input_dim=input_dim,
            output_dim=output_dim,
            max_steps=self.recurrent_controller_options.recurrent_max_steps,
            recurrent_layer_norm_position=self.recurrent_controller_options.recurrent_layer_norm_position,
            block_config=block_config,
            gate_config=gate_config,
            residual_config=None,
            halting_config=halting_config,
        )
