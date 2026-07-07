from dataclasses import dataclass

from emperor.base.layer.config import (
    LayerConfig,
    LayerStackConfig,
    RecurrentLayerConfig,
)
from emperor.base.layer.gate import GateConfig
from emperor.memory.config import DynamicMemoryConfig
from emperor.halting.config import StickBreakingConfig
from emperor.linears.core.config import LinearLayerConfig

from models.linears._controller_stack import (
    SubmoduleStackOptions,
    SubmoduleStackSource,
)
from models.linears._gate_config_factory import GateConfigFactory
from models.linears._halting_config_factory import HaltingConfigFactory
from models.linears._memory_config_factory import MemoryConfigFactory
from models.linears._recurrent_config_factory import RecurrentConfigFactory
from models.linears._builder_options import (
    DynamicMemoryOptions,
    LayerControllerOptions,
    MainLayerStackOptions,
    RecurrentControllerOptions,
)

import models.linears.linear.config as config


@dataclass(frozen=True)
class HiddenModelConfigDependencies:
    hidden_dim: int
    stack_options: MainLayerStackOptions | None
    submodule_stack_options: SubmoduleStackOptions | None
    layer_controller_options: LayerControllerOptions | None
    dynamic_memory_options: DynamicMemoryOptions | None
    recurrent_controller_options: RecurrentControllerOptions | None
    output_dim: int


class HiddenModelConfigFactory:
    def __init__(self, dependencies: HiddenModelConfigDependencies) -> None:
        hidden_dim = dependencies.hidden_dim
        stack_options = dependencies.stack_options
        submodule_stack_options = dependencies.submodule_stack_options
        layer_controller_options = dependencies.layer_controller_options
        dynamic_memory_options = dependencies.dynamic_memory_options
        recurrent_controller_options = dependencies.recurrent_controller_options
        output_dim = dependencies.output_dim
        self._hidden_dim = hidden_dim
        self.stack_options = self.__default_stack_options(stack_options)
        self.submodule_stack_options = self.__default_submodule_stack_options(
            submodule_stack_options
        )
        self.layer_controller_options = self.__default_layer_controller_options(
            layer_controller_options
        )
        self.dynamic_memory_options = self.__default_dynamic_memory_options(
            dynamic_memory_options
        )
        self.recurrent_controller_options = self.__default_recurrent_controller_options(
            recurrent_controller_options
        )
        self.gate_config_factory = GateConfigFactory(
            layer_controller_options=self.layer_controller_options,
            recurrent_controller_options=self.recurrent_controller_options,
            submodule_stack_options=self.submodule_stack_options,
        )
        self.halting_config_factory = HaltingConfigFactory(
            layer_controller_options=self.layer_controller_options,
            recurrent_controller_options=self.recurrent_controller_options,
            submodule_stack_options=self.submodule_stack_options,
            output_dim=output_dim,
        )
        self.memory_config_factory = MemoryConfigFactory(
            hidden_dim=self.hidden_dim,
            stack_options=self.stack_options,
            dynamic_memory_options=self.dynamic_memory_options,
            submodule_stack_options=self.submodule_stack_options,
        )
        self.recurrent_config_factory = RecurrentConfigFactory(
            recurrent_controller_options=self.recurrent_controller_options,
            gate_config_factory=self.gate_config_factory,
            halting_config_factory=self.halting_config_factory,
        )

    @property
    def hidden_dim(self) -> int:
        return self._hidden_dim

    def __default_stack_options(
        self,
        stack_options: MainLayerStackOptions | None,
    ) -> MainLayerStackOptions:
        if stack_options is not None:
            return stack_options
        return MainLayerStackOptions(
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
        submodule_stack_options: SubmoduleStackOptions | None,
    ) -> SubmoduleStackOptions:
        if submodule_stack_options is not None:
            return submodule_stack_options
        return SubmoduleStackOptions(
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

    def __default_layer_controller_options(
        self,
        layer_controller_options: LayerControllerOptions | None,
    ) -> LayerControllerOptions:
        if layer_controller_options is not None:
            return layer_controller_options
        gate_stack_source = self.__default_controller_stack_source("GATE_STACK")
        halting_stack_source = self.__default_controller_stack_source("HALTING_STACK")
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
        memory_stack_source = self.__default_controller_stack_source("MEMORY_STACK")
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
            "RECURRENT_GATE_STACK"
        )
        recurrent_halting_stack_source = self.__default_controller_stack_source(
            "RECURRENT_HALTING_STACK"
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
    ) -> SubmoduleStackSource:
        independent_flag = getattr(config, f"{prefix}_INDEPENDENT_FLAG")
        hidden_dim = getattr(config, f"{prefix}_HIDDEN_DIM")
        num_layers = getattr(config, f"{prefix}_NUM_LAYERS")
        last_layer_bias_option = getattr(config, f"{prefix}_LAST_LAYER_BIAS_OPTION")
        apply_output_pipeline_flag = getattr(
            config, f"{prefix}_APPLY_OUTPUT_PIPELINE_FLAG"
        )
        activation = getattr(config, f"{prefix}_ACTIVATION")
        layer_norm_position = getattr(config, f"{prefix}_LAYER_NORM_POSITION")
        residual_connection_option = getattr(
            config, f"{prefix}_RESIDUAL_CONNECTION_OPTION"
        )
        dropout_probability = getattr(config, f"{prefix}_DROPOUT_PROBABILITY")
        bias_flag = getattr(config, f"{prefix}_BIAS_FLAG")

        return SubmoduleStackSource(
            independent_flag=independent_flag,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            last_layer_bias_option=last_layer_bias_option,
            apply_output_pipeline_flag=apply_output_pipeline_flag,
            activation=activation,
            layer_norm_position=layer_norm_position,
            residual_connection_option=residual_connection_option,
            dropout_probability=dropout_probability,
            bias_flag=bias_flag,
        )

    def build_hidden_model_config(self) -> LayerStackConfig | RecurrentLayerConfig:
        gate_config = self.gate_config_factory.build_gate_config()
        halting_config = self.halting_config_factory.build_halting_config()
        memory_config = self.memory_config_factory.build_memory_config()
        layer_config = self.__build_layer_config(
            gate_config=gate_config, halting_config=halting_config
        )
        layer_stack_config = self.__build_stack_config(
            memory_config=memory_config, layer_config=layer_config
        )
        return self.recurrent_config_factory.build_config(layer_stack_config)

    def __build_stack_config(
        self,
        *,
        memory_config: DynamicMemoryConfig | None,
        layer_config: LayerConfig,
    ) -> LayerStackConfig:
        return LayerStackConfig(
            hidden_dim=self.hidden_dim,
            num_layers=self.stack_options.num_layers,
            last_layer_bias_option=self.stack_options.last_layer_bias_option,
            apply_output_pipeline_flag=self.stack_options.apply_output_pipeline_flag,
            shared_gate_config=self.layer_controller_options.shared_gate_config,
            shared_memory_config=memory_config,
            layer_config=layer_config,
        )

    def __build_layer_config(
        self,
        *,
        gate_config: GateConfig | None,
        halting_config: StickBreakingConfig | None,
    ) -> LayerConfig:
        layer_model_config = self.__build_layer_model_config()
        return LayerConfig(
            activation=self.stack_options.activation,
            layer_norm_position=self.stack_options.layer_norm_position,
            residual_connection_option=self.stack_options.residual_connection_option,
            dropout_probability=self.stack_options.dropout_probability,
            gate_config=gate_config,
            halting_config=halting_config,
            layer_model_config=layer_model_config,
        )

    def __build_layer_model_config(self) -> LinearLayerConfig:
        return LinearLayerConfig(
            bias_flag=self.stack_options.bias_flag,
        )
