from dataclasses import dataclass
from typing import TYPE_CHECKING

import models.linears.linear.config as config
from emperor.memory.config import DynamicMemoryConfig
from emperor.memory.options import MemoryPositionOptions
from emperor.linears.core.config import LinearLayerConfig
from emperor.base.layer.gate import GateConfig, LayerGateOptions
from emperor.halting.options import HaltingHiddenStateModeOptions
from emperor.base.layer.residual import ResidualConnectionOptions
from models.linears.linear.experiment_config import ExperimentConfig
from models.linears.linear._control_config_factory import ControlConfigFactory
from emperor.base.options import (
    ActivationOptions,
    LastLayerBiasOptions,
    LayerNormPositionOptions,
)
from emperor.base.layer.config import LayerConfig
from models.linears._controller_stack import (
    ControllerStackOptions,
    ControllerStackSource,
)

if TYPE_CHECKING:
    from emperor.config import ModelConfig


@dataclass(frozen=True)
class LinearStackOptions:
    hidden_dim: int
    bias_flag: bool
    layer_norm_position: LayerNormPositionOptions
    num_layers: int
    activation: ActivationOptions
    residual_connection_option: ResidualConnectionOptions
    dropout_probability: float
    last_layer_bias_option: LastLayerBiasOptions
    apply_output_pipeline_flag: bool


@dataclass(frozen=True)
class LayerControllerOptions:
    stack_gate_flag: bool
    gate_option: LayerGateOptions | None
    gate_activation: ActivationOptions | None
    gate_stack_source: ControllerStackSource
    stack_halting_flag: bool
    halting_threshold: float
    halting_dropout: float
    halting_hidden_state_mode: HaltingHiddenStateModeOptions
    halting_stack_source: ControllerStackSource
    shared_gate_config: GateConfig | None = None


@dataclass(frozen=True)
class DynamicMemoryOptions:
    memory_flag: bool
    memory_option: type[DynamicMemoryConfig]
    memory_position_option: MemoryPositionOptions
    memory_test_time_training_learning_rate: float | None
    memory_test_time_training_num_inner_steps: int | None
    memory_stack_source: ControllerStackSource


@dataclass(frozen=True)
class RecurrentControllerOptions:
    recurrent_flag: bool
    recurrent_max_steps: int
    recurrent_layer_norm_position: LayerNormPositionOptions
    recurrent_gate_flag: bool
    recurrent_gate_option: LayerGateOptions | None
    recurrent_gate_activation: ActivationOptions | None
    recurrent_gate_stack_source: ControllerStackSource
    recurrent_halting_flag: bool
    recurrent_halting_threshold: float
    recurrent_halting_dropout: float
    recurrent_halting_hidden_state_mode: HaltingHiddenStateModeOptions
    recurrent_halting_stack_source: ControllerStackSource


class LinearConfigBuilder:
    def __init__(
        self,
        batch_size: int = config.BATCH_SIZE,
        learning_rate: float = config.LEARNING_RATE,
        input_dim: int = config.INPUT_DIM,
        output_dim: int = config.OUTPUT_DIM,
        layer_norm_position: LayerNormPositionOptions = config.STACK_LAYER_NORM_POSITION,
        stack_hidden_dim: int = config.STACK_HIDDEN_DIM,
        stack_bias_flag: bool = config.STACK_BIAS_FLAG,
        stack_layer_norm_position: LayerNormPositionOptions | None = None,
        stack_num_layers: int = config.STACK_NUM_LAYERS,
        stack_activation: ActivationOptions = config.STACK_ACTIVATION,
        stack_residual_connection_option: ResidualConnectionOptions = config.STACK_RESIDUAL_CONNECTION_OPTION,
        stack_dropout_probability: float = config.STACK_DROPOUT_PROBABILITY,
        stack_last_layer_bias_option: LastLayerBiasOptions = config.STACK_LAST_LAYER_BIAS_OPTION,
        stack_apply_output_pipeline_flag: bool = config.STACK_APPLY_OUTPUT_PIPELINE_FLAG,
        submodule_stack_hidden_dim: int = config.SUBMODULE_STACK_HIDDEN_DIM,
        submodule_stack_layer_norm_position: LayerNormPositionOptions = config.SUBMODULE_STACK_LAYER_NORM_POSITION,
        submodule_stack_num_layers: int = config.SUBMODULE_STACK_NUM_LAYERS,
        submodule_stack_activation: ActivationOptions = config.SUBMODULE_STACK_ACTIVATION,
        submodule_stack_residual_connection_option: ResidualConnectionOptions = config.SUBMODULE_STACK_RESIDUAL_CONNECTION_OPTION,
        submodule_stack_dropout_probability: float = config.SUBMODULE_STACK_DROPOUT_PROBABILITY,
        submodule_stack_last_layer_bias_option: LastLayerBiasOptions = config.SUBMODULE_STACK_LAST_LAYER_BIAS_OPTION,
        submodule_stack_apply_output_pipeline_flag: bool = config.SUBMODULE_STACK_APPLY_OUTPUT_PIPELINE_FLAG,
        submodule_stack_bias_flag: bool = config.SUBMODULE_STACK_BIAS_FLAG,
        stack_gate_flag: bool = config.GATE_FLAG,
        gate_option: LayerGateOptions | None = config.GATE_OPTION,
        gate_activation: ActivationOptions | None = config.GATE_ACTIVATION,
        gate_stack_independent_flag: bool = config.GATE_STACK_INDEPENDENT_FLAG,
        gate_stack_hidden_dim: int | None = config.GATE_STACK_HIDDEN_DIM,
        gate_stack_layer_norm_position: (
            LayerNormPositionOptions | None
        ) = config.GATE_STACK_LAYER_NORM_POSITION,
        gate_stack_num_layers: int | None = config.GATE_STACK_NUM_LAYERS,
        gate_stack_activation: ActivationOptions | None = config.GATE_STACK_ACTIVATION,
        gate_stack_residual_connection_option: (
            ResidualConnectionOptions | None
        ) = config.GATE_STACK_RESIDUAL_CONNECTION_OPTION,
        gate_stack_dropout_probability: (
            float | None
        ) = config.GATE_STACK_DROPOUT_PROBABILITY,
        gate_stack_last_layer_bias_option: (
            LastLayerBiasOptions | None
        ) = config.GATE_STACK_LAST_LAYER_BIAS_OPTION,
        gate_stack_apply_output_pipeline_flag: (
            bool | None
        ) = config.GATE_STACK_APPLY_OUTPUT_PIPELINE_FLAG,
        gate_stack_bias_flag: bool | None = config.GATE_STACK_BIAS_FLAG,
        stack_halting_flag: bool = config.HALTING_FLAG,
        halting_threshold: float = config.HALTING_THRESHOLD,
        halting_dropout: float = config.HALTING_DROPOUT,
        halting_hidden_state_mode: HaltingHiddenStateModeOptions = (
            config.HALTING_HIDDEN_STATE_MODE
        ),
        halting_stack_independent_flag: bool = config.HALTING_STACK_INDEPENDENT_FLAG,
        halting_stack_hidden_dim: int | None = config.HALTING_STACK_HIDDEN_DIM,
        halting_stack_layer_norm_position: (
            LayerNormPositionOptions | None
        ) = config.HALTING_STACK_LAYER_NORM_POSITION,
        halting_stack_num_layers: int | None = config.HALTING_STACK_NUM_LAYERS,
        halting_stack_activation: (
            ActivationOptions | None
        ) = config.HALTING_STACK_ACTIVATION,
        halting_stack_residual_connection_option: (
            ResidualConnectionOptions | None
        ) = config.HALTING_STACK_RESIDUAL_CONNECTION_OPTION,
        halting_stack_dropout_probability: (
            float | None
        ) = config.HALTING_STACK_DROPOUT_PROBABILITY,
        halting_stack_last_layer_bias_option: (
            LastLayerBiasOptions | None
        ) = config.HALTING_STACK_LAST_LAYER_BIAS_OPTION,
        halting_stack_apply_output_pipeline_flag: (
            bool | None
        ) = config.HALTING_STACK_APPLY_OUTPUT_PIPELINE_FLAG,
        halting_stack_bias_flag: bool | None = config.HALTING_STACK_BIAS_FLAG,
        memory_flag: bool = config.MEMORY_FLAG,
        memory_option: type[DynamicMemoryConfig] = config.MEMORY_OPTION,
        memory_position_option: MemoryPositionOptions = config.MEMORY_POSITION_OPTION,
        memory_test_time_training_learning_rate: float | None = (
            config.MEMORY_TEST_TIME_TRAINING_LEARNING_RATE
        ),
        memory_test_time_training_num_inner_steps: int | None = (
            config.MEMORY_TEST_TIME_TRAINING_NUM_INNER_STEPS
        ),
        memory_stack_independent_flag: bool = config.MEMORY_STACK_INDEPENDENT_FLAG,
        memory_stack_hidden_dim: int | None = config.MEMORY_STACK_HIDDEN_DIM,
        memory_stack_layer_norm_position: LayerNormPositionOptions | None = (
            config.MEMORY_STACK_LAYER_NORM_POSITION
        ),
        memory_stack_num_layers: int | None = config.MEMORY_STACK_NUM_LAYERS,
        memory_stack_activation: (
            ActivationOptions | None
        ) = config.MEMORY_STACK_ACTIVATION,
        memory_stack_residual_connection_option: ResidualConnectionOptions | None = (
            config.MEMORY_STACK_RESIDUAL_CONNECTION_OPTION
        ),
        memory_stack_dropout_probability: float | None = (
            config.MEMORY_STACK_DROPOUT_PROBABILITY
        ),
        memory_stack_last_layer_bias_option: LastLayerBiasOptions | None = (
            config.MEMORY_STACK_LAST_LAYER_BIAS_OPTION
        ),
        memory_stack_apply_output_pipeline_flag: bool | None = (
            config.MEMORY_STACK_APPLY_OUTPUT_PIPELINE_FLAG
        ),
        memory_stack_bias_flag: bool | None = config.MEMORY_STACK_BIAS_FLAG,
        recurrent_flag: bool = config.RECURRENT_FLAG,
        recurrent_max_steps: int = config.RECURRENT_MAX_STEPS,
        recurrent_layer_norm_position: LayerNormPositionOptions = config.RECURRENT_LAYER_NORM_POSITION,
        recurrent_gate_flag: bool = config.RECURRENT_GATE_FLAG,
        recurrent_gate_option: LayerGateOptions | None = config.RECURRENT_GATE_OPTION,
        recurrent_gate_activation: (
            ActivationOptions | None
        ) = config.RECURRENT_GATE_ACTIVATION,
        recurrent_gate_stack_independent_flag: bool = config.RECURRENT_GATE_STACK_INDEPENDENT_FLAG,
        recurrent_gate_stack_hidden_dim: int | None = config.RECURRENT_GATE_STACK_HIDDEN_DIM,
        recurrent_gate_stack_layer_norm_position: (
            LayerNormPositionOptions | None
        ) = config.RECURRENT_GATE_STACK_LAYER_NORM_POSITION,
        recurrent_gate_stack_num_layers: (
            int | None
        ) = config.RECURRENT_GATE_STACK_NUM_LAYERS,
        recurrent_gate_stack_activation: (
            ActivationOptions | None
        ) = config.RECURRENT_GATE_STACK_ACTIVATION,
        recurrent_gate_stack_residual_connection_option: (
            ResidualConnectionOptions | None
        ) = config.RECURRENT_GATE_STACK_RESIDUAL_CONNECTION_OPTION,
        recurrent_gate_stack_dropout_probability: (
            float | None
        ) = config.RECURRENT_GATE_STACK_DROPOUT_PROBABILITY,
        recurrent_gate_stack_last_layer_bias_option: (
            LastLayerBiasOptions | None
        ) = config.RECURRENT_GATE_STACK_LAST_LAYER_BIAS_OPTION,
        recurrent_gate_stack_apply_output_pipeline_flag: (
            bool | None
        ) = config.RECURRENT_GATE_STACK_APPLY_OUTPUT_PIPELINE_FLAG,
        recurrent_gate_stack_bias_flag: bool | None = config.RECURRENT_GATE_STACK_BIAS_FLAG,
        recurrent_halting_flag: bool = config.RECURRENT_HALTING_FLAG,
        recurrent_halting_threshold: float = config.RECURRENT_HALTING_THRESHOLD,
        recurrent_halting_dropout: float = config.RECURRENT_HALTING_DROPOUT,
        recurrent_halting_hidden_state_mode: HaltingHiddenStateModeOptions = (
            config.RECURRENT_HALTING_HIDDEN_STATE_MODE
        ),
        recurrent_halting_stack_independent_flag: bool = config.RECURRENT_HALTING_STACK_INDEPENDENT_FLAG,
        recurrent_halting_stack_hidden_dim: (
            int | None
        ) = config.RECURRENT_HALTING_STACK_HIDDEN_DIM,
        recurrent_halting_stack_layer_norm_position: (
            LayerNormPositionOptions | None
        ) = config.RECURRENT_HALTING_STACK_LAYER_NORM_POSITION,
        recurrent_halting_stack_num_layers: (
            int | None
        ) = config.RECURRENT_HALTING_STACK_NUM_LAYERS,
        recurrent_halting_stack_activation: (
            ActivationOptions | None
        ) = config.RECURRENT_HALTING_STACK_ACTIVATION,
        recurrent_halting_stack_residual_connection_option: (
            ResidualConnectionOptions | None
        ) = config.RECURRENT_HALTING_STACK_RESIDUAL_CONNECTION_OPTION,
        recurrent_halting_stack_dropout_probability: (
            float | None
        ) = config.RECURRENT_HALTING_STACK_DROPOUT_PROBABILITY,
        recurrent_halting_stack_last_layer_bias_option: (
            LastLayerBiasOptions | None
        ) = config.RECURRENT_HALTING_STACK_LAST_LAYER_BIAS_OPTION,
        recurrent_halting_stack_apply_output_pipeline_flag: (
            bool | None
        ) = config.RECURRENT_HALTING_STACK_APPLY_OUTPUT_PIPELINE_FLAG,
        recurrent_halting_stack_bias_flag: (
            bool | None
        ) = config.RECURRENT_HALTING_STACK_BIAS_FLAG,
        shared_gate_config: GateConfig | None = None,
        stack_options: LinearStackOptions | None = None,
        submodule_stack_options: ControllerStackOptions | None = None,
        layer_controller_options: LayerControllerOptions | None = None,
        dynamic_memory_options: DynamicMemoryOptions | None = None,
        recurrent_controller_options: RecurrentControllerOptions | None = None,
    ) -> None:
        stack_options = stack_options or LinearStackOptions(
            hidden_dim=stack_hidden_dim,
            bias_flag=stack_bias_flag,
            layer_norm_position=(
                stack_layer_norm_position
                if stack_layer_norm_position is not None
                else layer_norm_position
            ),
            num_layers=stack_num_layers,
            activation=stack_activation,
            residual_connection_option=stack_residual_connection_option,
            dropout_probability=stack_dropout_probability,
            last_layer_bias_option=stack_last_layer_bias_option,
            apply_output_pipeline_flag=stack_apply_output_pipeline_flag,
        )
        submodule_stack_options = submodule_stack_options or ControllerStackOptions(
            hidden_dim=submodule_stack_hidden_dim,
            num_layers=submodule_stack_num_layers,
            last_layer_bias_option=submodule_stack_last_layer_bias_option,
            apply_output_pipeline_flag=submodule_stack_apply_output_pipeline_flag,
            activation=submodule_stack_activation,
            layer_norm_position=submodule_stack_layer_norm_position,
            residual_connection_option=submodule_stack_residual_connection_option,
            dropout_probability=submodule_stack_dropout_probability,
            bias_flag=submodule_stack_bias_flag,
        )
        layer_controller_options = (
            layer_controller_options
            or LayerControllerOptions(
                stack_gate_flag=stack_gate_flag,
                gate_option=gate_option,
                gate_activation=gate_activation,
                gate_stack_source=ControllerStackSource(
                    independent_flag=gate_stack_independent_flag,
                    hidden_dim=gate_stack_hidden_dim,
                    num_layers=gate_stack_num_layers,
                    last_layer_bias_option=gate_stack_last_layer_bias_option,
                    apply_output_pipeline_flag=(
                        gate_stack_apply_output_pipeline_flag
                    ),
                    activation=gate_stack_activation,
                    layer_norm_position=gate_stack_layer_norm_position,
                    residual_connection_option=(
                        gate_stack_residual_connection_option
                    ),
                    dropout_probability=gate_stack_dropout_probability,
                    bias_flag=gate_stack_bias_flag,
                ),
                stack_halting_flag=stack_halting_flag,
                halting_threshold=halting_threshold,
                halting_dropout=halting_dropout,
                halting_hidden_state_mode=halting_hidden_state_mode,
                halting_stack_source=ControllerStackSource(
                    independent_flag=halting_stack_independent_flag,
                    hidden_dim=halting_stack_hidden_dim,
                    num_layers=halting_stack_num_layers,
                    last_layer_bias_option=halting_stack_last_layer_bias_option,
                    apply_output_pipeline_flag=(
                        halting_stack_apply_output_pipeline_flag
                    ),
                    activation=halting_stack_activation,
                    layer_norm_position=halting_stack_layer_norm_position,
                    residual_connection_option=(
                        halting_stack_residual_connection_option
                    ),
                    dropout_probability=halting_stack_dropout_probability,
                    bias_flag=halting_stack_bias_flag,
                ),
                shared_gate_config=shared_gate_config,
            )
        )
        dynamic_memory_options = dynamic_memory_options or DynamicMemoryOptions(
            memory_flag=memory_flag,
            memory_option=memory_option,
            memory_position_option=memory_position_option,
            memory_test_time_training_learning_rate=(
                memory_test_time_training_learning_rate
            ),
            memory_test_time_training_num_inner_steps=(
                memory_test_time_training_num_inner_steps
            ),
            memory_stack_source=ControllerStackSource(
                independent_flag=memory_stack_independent_flag,
                hidden_dim=memory_stack_hidden_dim,
                num_layers=memory_stack_num_layers,
                last_layer_bias_option=memory_stack_last_layer_bias_option,
                apply_output_pipeline_flag=memory_stack_apply_output_pipeline_flag,
                activation=memory_stack_activation,
                layer_norm_position=memory_stack_layer_norm_position,
                residual_connection_option=memory_stack_residual_connection_option,
                dropout_probability=memory_stack_dropout_probability,
                bias_flag=memory_stack_bias_flag,
            ),
        )
        recurrent_controller_options = (
            recurrent_controller_options
            or RecurrentControllerOptions(
                recurrent_flag=recurrent_flag,
                recurrent_max_steps=recurrent_max_steps,
                recurrent_layer_norm_position=recurrent_layer_norm_position,
                recurrent_gate_flag=recurrent_gate_flag,
                recurrent_gate_option=recurrent_gate_option,
                recurrent_gate_activation=recurrent_gate_activation,
                recurrent_gate_stack_source=ControllerStackSource(
                    independent_flag=recurrent_gate_stack_independent_flag,
                    hidden_dim=recurrent_gate_stack_hidden_dim,
                    num_layers=recurrent_gate_stack_num_layers,
                    last_layer_bias_option=(
                        recurrent_gate_stack_last_layer_bias_option
                    ),
                    apply_output_pipeline_flag=(
                        recurrent_gate_stack_apply_output_pipeline_flag
                    ),
                    activation=recurrent_gate_stack_activation,
                    layer_norm_position=recurrent_gate_stack_layer_norm_position,
                    residual_connection_option=(
                        recurrent_gate_stack_residual_connection_option
                    ),
                    dropout_probability=recurrent_gate_stack_dropout_probability,
                    bias_flag=recurrent_gate_stack_bias_flag,
                ),
                recurrent_halting_flag=recurrent_halting_flag,
                recurrent_halting_threshold=recurrent_halting_threshold,
                recurrent_halting_dropout=recurrent_halting_dropout,
                recurrent_halting_hidden_state_mode=(
                    recurrent_halting_hidden_state_mode
                ),
                recurrent_halting_stack_source=ControllerStackSource(
                    independent_flag=recurrent_halting_stack_independent_flag,
                    hidden_dim=recurrent_halting_stack_hidden_dim,
                    num_layers=recurrent_halting_stack_num_layers,
                    last_layer_bias_option=(
                        recurrent_halting_stack_last_layer_bias_option
                    ),
                    apply_output_pipeline_flag=(
                        recurrent_halting_stack_apply_output_pipeline_flag
                    ),
                    activation=recurrent_halting_stack_activation,
                    layer_norm_position=recurrent_halting_stack_layer_norm_position,
                    residual_connection_option=(
                        recurrent_halting_stack_residual_connection_option
                    ),
                    dropout_probability=recurrent_halting_stack_dropout_probability,
                    bias_flag=recurrent_halting_stack_bias_flag,
                ),
            )
        )
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.input_dim = input_dim
        self.stack_options = stack_options
        self.hidden_dim = stack_options.hidden_dim
        self.output_dim = output_dim
        self.bias_flag = stack_options.bias_flag
        self.layer_norm_position = stack_options.layer_norm_position
        self.stack_num_layers = stack_options.num_layers
        self.stack_activation = stack_options.activation
        self.stack_residual_connection_option = (
            stack_options.residual_connection_option
        )
        self.stack_dropout_probability = stack_options.dropout_probability
        self.stack_last_layer_bias_option = stack_options.last_layer_bias_option
        self.stack_apply_output_pipeline_flag = (
            stack_options.apply_output_pipeline_flag
        )
        self.submodule_stack_options = submodule_stack_options
        self.submodule_stack_hidden_dim = submodule_stack_options.hidden_dim
        self.submodule_stack_layer_norm_position = (
            submodule_stack_options.layer_norm_position
        )
        self.submodule_stack_num_layers = submodule_stack_options.num_layers
        self.submodule_stack_activation = submodule_stack_options.activation
        self.submodule_stack_residual_connection_option = (
            submodule_stack_options.residual_connection_option
        )
        self.submodule_stack_dropout_probability = (
            submodule_stack_options.dropout_probability
        )
        self.submodule_stack_last_layer_bias_option = (
            submodule_stack_options.last_layer_bias_option
        )
        self.submodule_stack_apply_output_pipeline_flag = (
            submodule_stack_options.apply_output_pipeline_flag
        )
        self.submodule_stack_bias_flag = submodule_stack_options.bias_flag
        self.layer_controller_options = layer_controller_options
        self.stack_gate_flag = layer_controller_options.stack_gate_flag
        self.gate_option = layer_controller_options.gate_option
        self.gate_activation = layer_controller_options.gate_activation
        self.gate_stack_source = layer_controller_options.gate_stack_source
        self.gate_stack_independent_flag = self.gate_stack_source.independent_flag
        self.gate_stack_hidden_dim = self.gate_stack_source.hidden_dim
        self.gate_stack_layer_norm_position = (
            self.gate_stack_source.layer_norm_position
        )
        self.gate_stack_num_layers = self.gate_stack_source.num_layers
        self.gate_stack_activation = self.gate_stack_source.activation
        self.gate_stack_residual_connection_option = (
            self.gate_stack_source.residual_connection_option
        )
        self.gate_stack_dropout_probability = (
            self.gate_stack_source.dropout_probability
        )
        self.gate_stack_last_layer_bias_option = (
            self.gate_stack_source.last_layer_bias_option
        )
        self.gate_stack_apply_output_pipeline_flag = (
            self.gate_stack_source.apply_output_pipeline_flag
        )
        self.gate_stack_bias_flag = self.gate_stack_source.bias_flag
        self.shared_gate_config = layer_controller_options.shared_gate_config
        self.stack_halting_flag = layer_controller_options.stack_halting_flag
        self.halting_threshold = layer_controller_options.halting_threshold
        self.halting_dropout = layer_controller_options.halting_dropout
        self.halting_hidden_state_mode = (
            layer_controller_options.halting_hidden_state_mode
        )
        self.halting_stack_source = layer_controller_options.halting_stack_source
        self.halting_stack_independent_flag = (
            self.halting_stack_source.independent_flag
        )
        self.halting_stack_hidden_dim = self.halting_stack_source.hidden_dim
        self.halting_stack_layer_norm_position = (
            self.halting_stack_source.layer_norm_position
        )
        self.halting_stack_num_layers = self.halting_stack_source.num_layers
        self.halting_stack_activation = self.halting_stack_source.activation
        self.halting_stack_residual_connection_option = (
            self.halting_stack_source.residual_connection_option
        )
        self.halting_stack_dropout_probability = (
            self.halting_stack_source.dropout_probability
        )
        self.halting_stack_last_layer_bias_option = (
            self.halting_stack_source.last_layer_bias_option
        )
        self.halting_stack_apply_output_pipeline_flag = (
            self.halting_stack_source.apply_output_pipeline_flag
        )
        self.halting_stack_bias_flag = self.halting_stack_source.bias_flag
        self.dynamic_memory_options = dynamic_memory_options
        self.memory_flag = dynamic_memory_options.memory_flag
        self.memory_option = dynamic_memory_options.memory_option
        self.memory_position_option = dynamic_memory_options.memory_position_option
        self.memory_test_time_training_learning_rate = (
            dynamic_memory_options.memory_test_time_training_learning_rate
        )
        self.memory_test_time_training_num_inner_steps = (
            dynamic_memory_options.memory_test_time_training_num_inner_steps
        )
        self.memory_stack_source = dynamic_memory_options.memory_stack_source
        self.memory_stack_independent_flag = self.memory_stack_source.independent_flag
        self.memory_stack_hidden_dim = self.memory_stack_source.hidden_dim
        self.memory_stack_layer_norm_position = (
            self.memory_stack_source.layer_norm_position
        )
        self.memory_stack_num_layers = self.memory_stack_source.num_layers
        self.memory_stack_activation = self.memory_stack_source.activation
        self.memory_stack_residual_connection_option = (
            self.memory_stack_source.residual_connection_option
        )
        self.memory_stack_dropout_probability = (
            self.memory_stack_source.dropout_probability
        )
        self.memory_stack_last_layer_bias_option = (
            self.memory_stack_source.last_layer_bias_option
        )
        self.memory_stack_apply_output_pipeline_flag = (
            self.memory_stack_source.apply_output_pipeline_flag
        )
        self.memory_stack_bias_flag = self.memory_stack_source.bias_flag
        self.recurrent_controller_options = recurrent_controller_options
        self.recurrent_flag = recurrent_controller_options.recurrent_flag
        self.recurrent_max_steps = recurrent_controller_options.recurrent_max_steps
        self.recurrent_layer_norm_position = (
            recurrent_controller_options.recurrent_layer_norm_position
        )
        self.recurrent_gate_flag = recurrent_controller_options.recurrent_gate_flag
        self.recurrent_gate_option = recurrent_controller_options.recurrent_gate_option
        self.recurrent_gate_activation = (
            recurrent_controller_options.recurrent_gate_activation
        )
        self.recurrent_gate_stack_source = (
            recurrent_controller_options.recurrent_gate_stack_source
        )
        self.recurrent_gate_stack_independent_flag = (
            self.recurrent_gate_stack_source.independent_flag
        )
        self.recurrent_gate_stack_hidden_dim = (
            self.recurrent_gate_stack_source.hidden_dim
        )
        self.recurrent_gate_stack_layer_norm_position = (
            self.recurrent_gate_stack_source.layer_norm_position
        )
        self.recurrent_gate_stack_num_layers = (
            self.recurrent_gate_stack_source.num_layers
        )
        self.recurrent_gate_stack_activation = (
            self.recurrent_gate_stack_source.activation
        )
        self.recurrent_gate_stack_residual_connection_option = (
            self.recurrent_gate_stack_source.residual_connection_option
        )
        self.recurrent_gate_stack_dropout_probability = (
            self.recurrent_gate_stack_source.dropout_probability
        )
        self.recurrent_gate_stack_last_layer_bias_option = (
            self.recurrent_gate_stack_source.last_layer_bias_option
        )
        self.recurrent_gate_stack_apply_output_pipeline_flag = (
            self.recurrent_gate_stack_source.apply_output_pipeline_flag
        )
        self.recurrent_gate_stack_bias_flag = self.recurrent_gate_stack_source.bias_flag
        self.recurrent_halting_flag = (
            recurrent_controller_options.recurrent_halting_flag
        )
        self.recurrent_halting_threshold = (
            recurrent_controller_options.recurrent_halting_threshold
        )
        self.recurrent_halting_dropout = (
            recurrent_controller_options.recurrent_halting_dropout
        )
        self.recurrent_halting_hidden_state_mode = (
            recurrent_controller_options.recurrent_halting_hidden_state_mode
        )
        self.recurrent_halting_stack_source = (
            recurrent_controller_options.recurrent_halting_stack_source
        )
        self.recurrent_halting_stack_independent_flag = (
            self.recurrent_halting_stack_source.independent_flag
        )
        self.recurrent_halting_stack_hidden_dim = (
            self.recurrent_halting_stack_source.hidden_dim
        )
        self.recurrent_halting_stack_layer_norm_position = (
            self.recurrent_halting_stack_source.layer_norm_position
        )
        self.recurrent_halting_stack_num_layers = (
            self.recurrent_halting_stack_source.num_layers
        )
        self.recurrent_halting_stack_activation = (
            self.recurrent_halting_stack_source.activation
        )
        self.recurrent_halting_stack_residual_connection_option = (
            self.recurrent_halting_stack_source.residual_connection_option
        )
        self.recurrent_halting_stack_dropout_probability = (
            self.recurrent_halting_stack_source.dropout_probability
        )
        self.recurrent_halting_stack_last_layer_bias_option = (
            self.recurrent_halting_stack_source.last_layer_bias_option
        )
        self.recurrent_halting_stack_apply_output_pipeline_flag = (
            self.recurrent_halting_stack_source.apply_output_pipeline_flag
        )
        self.recurrent_halting_stack_bias_flag = (
            self.recurrent_halting_stack_source.bias_flag
        )

    def build(self) -> "ModelConfig":
        from emperor.config import ModelConfig

        input_model_config = LayerConfig(
            activation=self.stack_activation,
            layer_norm_position=LayerNormPositionOptions.DISABLED,
            residual_connection_option=ResidualConnectionOptions.DISABLED,
            dropout_probability=0.0,
            gate_config=None,
            halting_config=None,
            layer_model_config=LinearLayerConfig(
                bias_flag=True,
            ),
        )

        model_config = ControlConfigFactory(self).build()

        output_model_config = LayerConfig(
            activation=ActivationOptions.DISABLED,
            layer_norm_position=LayerNormPositionOptions.DISABLED,
            residual_connection_option=ResidualConnectionOptions.DISABLED,
            dropout_probability=0.0,
            gate_config=None,
            halting_config=None,
            layer_model_config=LinearLayerConfig(
                bias_flag=True,
            ),
        )

        return ModelConfig(
            learning_rate=self.learning_rate,
            batch_size=self.batch_size,
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
            experiment_config=ExperimentConfig(
                input_model_config=input_model_config,
                model_config=model_config,
                output_model_config=output_model_config,
            ),
        )
