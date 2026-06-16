from emperor.base.layer.residual import ResidualConnectionOptions
import models.linears.linear.config as config

from emperor.base.options import (
    ActivationOptions,
    LastLayerBiasOptions,
    LayerNormPositionOptions,
)
from emperor.base.layer.config import (
    LayerConfig,
    LayerStackConfig,
    RecurrentLayerConfig,
)
from emperor.base.layer.gate import GateConfig, LayerGateOptions
from emperor.halting.config import StickBreakingConfig
from emperor.halting.options import HaltingHiddenStateModeOptions
from emperor.linears.core.config import LinearLayerConfig
from emperor.memory.config import DynamicMemoryConfig
from emperor.memory.options import MemoryPositionOptions
from models.linears.linear.experiment_config import ExperimentConfig

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from emperor.config import ModelConfig


class LinearConfigBuilder:
    def __init__(
        self,
        batch_size: int = config.BATCH_SIZE,
        learning_rate: float = config.LEARNING_RATE,
        input_dim: int = config.INPUT_DIM,
        hidden_dim: int = config.HIDDEN_DIM,
        output_dim: int = config.OUTPUT_DIM,
        bias_flag: bool = config.BIAS_FLAG,
        layer_norm_position: LayerNormPositionOptions = config.LAYER_NORM_POSITION,
        stack_num_layers: int = config.STACK_NUM_LAYERS,
        stack_activation: ActivationOptions = config.STACK_ACTIVATION,
        stack_residual_connection_option: ResidualConnectionOptions = config.STACK_RESIDUAL_CONNECTION_OPTION,
        stack_dropout_probability: float = config.STACK_DROPOUT_PROBABILITY,
        stack_last_layer_bias_option: LastLayerBiasOptions = config.STACK_LAST_LAYER_BIAS_OPTION,
        stack_apply_output_pipeline_flag: bool = config.STACK_APPLY_OUTPUT_PIPELINE_FLAG,
        submodule_hidden_dim: int = config.SUBMODULE_HIDDEN_DIM,
        submodule_layer_norm_position: LayerNormPositionOptions = config.SUBMODULE_LAYER_NORM_POSITION,
        submodule_stack_num_layers: int = config.SUBMODULE_STACK_NUM_LAYERS,
        submodule_stack_activation: ActivationOptions = config.SUBMODULE_STACK_ACTIVATION,
        submodule_stack_residual_connection_option: ResidualConnectionOptions = config.SUBMODULE_STACK_RESIDUAL_CONNECTION_OPTION,
        submodule_stack_dropout_probability: float = config.SUBMODULE_STACK_DROPOUT_PROBABILITY,
        submodule_stack_last_layer_bias_option: LastLayerBiasOptions = config.SUBMODULE_STACK_LAST_LAYER_BIAS_OPTION,
        submodule_stack_apply_output_pipeline_flag: bool = config.SUBMODULE_STACK_APPLY_OUTPUT_PIPELINE_FLAG,
        submodule_bias_flag: bool = config.SUBMODULE_BIAS_FLAG,
        stack_gate_flag: bool = config.GATE_FLAG,
        gate_option: LayerGateOptions | None = config.GATE_OPTION,
        gate_activation: ActivationOptions | None = config.GATE_ACTIVATION,
        gate_hidden_dim: int | None = config.GATE_HIDDEN_DIM,
        gate_layer_norm_position: LayerNormPositionOptions | None = config.GATE_LAYER_NORM_POSITION,
        gate_stack_num_layers: int | None = config.GATE_STACK_NUM_LAYERS,
        gate_stack_activation: ActivationOptions | None = config.GATE_STACK_ACTIVATION,
        gate_stack_residual_connection_option: ResidualConnectionOptions | None = config.GATE_STACK_RESIDUAL_CONNECTION_OPTION,
        gate_stack_dropout_probability: float | None = config.GATE_STACK_DROPOUT_PROBABILITY,
        gate_stack_last_layer_bias_option: LastLayerBiasOptions | None = config.GATE_STACK_LAST_LAYER_BIAS_OPTION,
        gate_stack_apply_output_pipeline_flag: bool | None = config.GATE_STACK_APPLY_OUTPUT_PIPELINE_FLAG,
        gate_bias_flag: bool | None = config.GATE_BIAS_FLAG,
        stack_halting_flag: bool = config.HALTING_FLAG,
        halting_threshold: float = config.HALTING_THRESHOLD,
        halting_dropout: float = config.HALTING_DROPOUT,
        halting_hidden_state_mode: HaltingHiddenStateModeOptions = (
            config.HALTING_HIDDEN_STATE_MODE
        ),
        halting_hidden_dim: int | None = config.HALTING_HIDDEN_DIM,
        halting_output_dim: int = config.HALTING_OUTPUT_DIM,
        halting_layer_norm_position: LayerNormPositionOptions | None = config.HALTING_LAYER_NORM_POSITION,
        halting_stack_num_layers: int | None = config.HALTING_STACK_NUM_LAYERS,
        halting_stack_activation: ActivationOptions | None = config.HALTING_STACK_ACTIVATION,
        halting_stack_residual_connection_option: ResidualConnectionOptions | None = config.HALTING_STACK_RESIDUAL_CONNECTION_OPTION,
        halting_stack_dropout_probability: float | None = config.HALTING_STACK_DROPOUT_PROBABILITY,
        halting_stack_last_layer_bias_option: LastLayerBiasOptions | None = config.HALTING_STACK_LAST_LAYER_BIAS_OPTION,
        halting_stack_apply_output_pipeline_flag: bool | None = config.HALTING_STACK_APPLY_OUTPUT_PIPELINE_FLAG,
        halting_bias_flag: bool | None = config.HALTING_BIAS_FLAG,
        memory_flag: bool = config.MEMORY_FLAG,
        memory_option: type[DynamicMemoryConfig] = config.MEMORY_OPTION,
        memory_position_option: MemoryPositionOptions = config.MEMORY_POSITION_OPTION,
        memory_test_time_training_learning_rate: float | None = (
            config.MEMORY_TEST_TIME_TRAINING_LEARNING_RATE
        ),
        memory_test_time_training_num_inner_steps: int | None = (
            config.MEMORY_TEST_TIME_TRAINING_NUM_INNER_STEPS
        ),
        memory_hidden_dim: int | None = config.MEMORY_HIDDEN_DIM,
        memory_layer_norm_position: LayerNormPositionOptions | None = (
            config.MEMORY_LAYER_NORM_POSITION
        ),
        memory_stack_num_layers: int | None = config.MEMORY_STACK_NUM_LAYERS,
        memory_stack_activation: ActivationOptions | None = config.MEMORY_STACK_ACTIVATION,
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
        memory_bias_flag: bool | None = config.MEMORY_BIAS_FLAG,
        recurrent_flag: bool = config.RECURRENT_FLAG,
        recurrent_max_steps: int = config.RECURRENT_MAX_STEPS,
        recurrent_layer_norm_position: LayerNormPositionOptions = config.RECURRENT_LAYER_NORM_POSITION,
        recurrent_gate_flag: bool = config.RECURRENT_GATE_FLAG,
        recurrent_gate_option: LayerGateOptions | None = config.RECURRENT_GATE_OPTION,
        recurrent_gate_activation: ActivationOptions | None = config.RECURRENT_GATE_ACTIVATION,
        recurrent_gate_hidden_dim: int | None = config.RECURRENT_GATE_HIDDEN_DIM,
        recurrent_gate_layer_norm_position: LayerNormPositionOptions | None = config.RECURRENT_GATE_LAYER_NORM_POSITION,
        recurrent_gate_stack_num_layers: int | None = config.RECURRENT_GATE_STACK_NUM_LAYERS,
        recurrent_gate_stack_activation: ActivationOptions | None = config.RECURRENT_GATE_STACK_ACTIVATION,
        recurrent_gate_stack_residual_connection_option: ResidualConnectionOptions | None = config.RECURRENT_GATE_STACK_RESIDUAL_CONNECTION_OPTION,
        recurrent_gate_stack_dropout_probability: float | None = config.RECURRENT_GATE_STACK_DROPOUT_PROBABILITY,
        recurrent_gate_stack_last_layer_bias_option: LastLayerBiasOptions | None = config.RECURRENT_GATE_STACK_LAST_LAYER_BIAS_OPTION,
        recurrent_gate_stack_apply_output_pipeline_flag: bool | None = config.RECURRENT_GATE_STACK_APPLY_OUTPUT_PIPELINE_FLAG,
        recurrent_gate_bias_flag: bool | None = config.RECURRENT_GATE_BIAS_FLAG,
        recurrent_halting_flag: bool = config.RECURRENT_HALTING_FLAG,
        recurrent_halting_threshold: float = config.RECURRENT_HALTING_THRESHOLD,
        recurrent_halting_dropout: float = config.RECURRENT_HALTING_DROPOUT,
        recurrent_halting_hidden_state_mode: HaltingHiddenStateModeOptions = (
            config.RECURRENT_HALTING_HIDDEN_STATE_MODE
        ),
        recurrent_halting_hidden_dim: int | None = config.RECURRENT_HALTING_HIDDEN_DIM,
        recurrent_halting_output_dim: int = config.RECURRENT_HALTING_OUTPUT_DIM,
        recurrent_halting_layer_norm_position: LayerNormPositionOptions | None = config.RECURRENT_HALTING_LAYER_NORM_POSITION,
        recurrent_halting_stack_num_layers: int | None = config.RECURRENT_HALTING_STACK_NUM_LAYERS,
        recurrent_halting_stack_activation: ActivationOptions | None = config.RECURRENT_HALTING_STACK_ACTIVATION,
        recurrent_halting_stack_residual_connection_option: ResidualConnectionOptions | None = config.RECURRENT_HALTING_STACK_RESIDUAL_CONNECTION_OPTION,
        recurrent_halting_stack_dropout_probability: float | None = config.RECURRENT_HALTING_STACK_DROPOUT_PROBABILITY,
        recurrent_halting_stack_last_layer_bias_option: LastLayerBiasOptions | None = config.RECURRENT_HALTING_STACK_LAST_LAYER_BIAS_OPTION,
        recurrent_halting_stack_apply_output_pipeline_flag: bool | None = config.RECURRENT_HALTING_STACK_APPLY_OUTPUT_PIPELINE_FLAG,
        recurrent_halting_bias_flag: bool | None = config.RECURRENT_HALTING_BIAS_FLAG,
        shared_gate_config: GateConfig | None = None,
    ) -> None:
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.bias_flag = bias_flag
        self.layer_norm_position = layer_norm_position
        self.stack_num_layers = stack_num_layers
        self.stack_activation = stack_activation
        self.stack_residual_connection_option = stack_residual_connection_option
        self.stack_dropout_probability = stack_dropout_probability
        self.stack_last_layer_bias_option = stack_last_layer_bias_option
        self.stack_apply_output_pipeline_flag = stack_apply_output_pipeline_flag
        self.submodule_hidden_dim = submodule_hidden_dim
        self.submodule_layer_norm_position = submodule_layer_norm_position
        self.submodule_stack_num_layers = submodule_stack_num_layers
        self.submodule_stack_activation = submodule_stack_activation
        self.submodule_stack_residual_connection_option = (
            submodule_stack_residual_connection_option
        )
        self.submodule_stack_dropout_probability = (
            submodule_stack_dropout_probability
        )
        self.submodule_stack_last_layer_bias_option = (
            submodule_stack_last_layer_bias_option
        )
        self.submodule_stack_apply_output_pipeline_flag = (
            submodule_stack_apply_output_pipeline_flag
        )
        self.submodule_bias_flag = submodule_bias_flag
        self.stack_gate_flag = stack_gate_flag
        self.gate_option = gate_option
        self.gate_activation = gate_activation
        self.gate_hidden_dim = gate_hidden_dim
        self.gate_layer_norm_position = gate_layer_norm_position
        self.gate_stack_num_layers = gate_stack_num_layers
        self.gate_stack_activation = gate_stack_activation
        self.gate_stack_residual_connection_option = (
            gate_stack_residual_connection_option
        )
        self.gate_stack_dropout_probability = gate_stack_dropout_probability
        self.gate_stack_last_layer_bias_option = gate_stack_last_layer_bias_option
        self.gate_stack_apply_output_pipeline_flag = (
            gate_stack_apply_output_pipeline_flag
        )
        self.gate_bias_flag = gate_bias_flag
        self.shared_gate_config = shared_gate_config
        self.stack_halting_flag = stack_halting_flag
        self.halting_threshold = halting_threshold
        self.halting_dropout = halting_dropout
        self.halting_hidden_state_mode = halting_hidden_state_mode
        self.halting_hidden_dim = halting_hidden_dim
        self.halting_output_dim = halting_output_dim
        self.halting_layer_norm_position = halting_layer_norm_position
        self.halting_stack_num_layers = halting_stack_num_layers
        self.halting_stack_activation = halting_stack_activation
        self.halting_stack_residual_connection_option = (
            halting_stack_residual_connection_option
        )
        self.halting_stack_dropout_probability = halting_stack_dropout_probability
        self.halting_stack_last_layer_bias_option = halting_stack_last_layer_bias_option
        self.halting_stack_apply_output_pipeline_flag = (
            halting_stack_apply_output_pipeline_flag
        )
        self.halting_bias_flag = halting_bias_flag
        self.memory_flag = memory_flag
        self.memory_option = memory_option
        self.memory_position_option = memory_position_option
        self.memory_test_time_training_learning_rate = (
            memory_test_time_training_learning_rate
        )
        self.memory_test_time_training_num_inner_steps = (
            memory_test_time_training_num_inner_steps
        )
        self.memory_hidden_dim = memory_hidden_dim
        self.memory_layer_norm_position = memory_layer_norm_position
        self.memory_stack_num_layers = memory_stack_num_layers
        self.memory_stack_activation = memory_stack_activation
        self.memory_stack_residual_connection_option = (
            memory_stack_residual_connection_option
        )
        self.memory_stack_dropout_probability = memory_stack_dropout_probability
        self.memory_stack_last_layer_bias_option = memory_stack_last_layer_bias_option
        self.memory_stack_apply_output_pipeline_flag = (
            memory_stack_apply_output_pipeline_flag
        )
        self.memory_bias_flag = memory_bias_flag
        self.recurrent_flag = recurrent_flag
        self.recurrent_max_steps = recurrent_max_steps
        self.recurrent_layer_norm_position = recurrent_layer_norm_position
        self.recurrent_gate_flag = recurrent_gate_flag
        self.recurrent_gate_option = recurrent_gate_option
        self.recurrent_gate_activation = recurrent_gate_activation
        self.recurrent_gate_hidden_dim = recurrent_gate_hidden_dim
        self.recurrent_gate_layer_norm_position = recurrent_gate_layer_norm_position
        self.recurrent_gate_stack_num_layers = recurrent_gate_stack_num_layers
        self.recurrent_gate_stack_activation = recurrent_gate_stack_activation
        self.recurrent_gate_stack_residual_connection_option = (
            recurrent_gate_stack_residual_connection_option
        )
        self.recurrent_gate_stack_dropout_probability = (
            recurrent_gate_stack_dropout_probability
        )
        self.recurrent_gate_stack_last_layer_bias_option = (
            recurrent_gate_stack_last_layer_bias_option
        )
        self.recurrent_gate_stack_apply_output_pipeline_flag = (
            recurrent_gate_stack_apply_output_pipeline_flag
        )
        self.recurrent_gate_bias_flag = recurrent_gate_bias_flag
        self.recurrent_halting_flag = recurrent_halting_flag
        self.recurrent_halting_threshold = recurrent_halting_threshold
        self.recurrent_halting_dropout = recurrent_halting_dropout
        self.recurrent_halting_hidden_state_mode = recurrent_halting_hidden_state_mode
        self.recurrent_halting_hidden_dim = recurrent_halting_hidden_dim
        self.recurrent_halting_output_dim = recurrent_halting_output_dim
        self.recurrent_halting_layer_norm_position = (
            recurrent_halting_layer_norm_position
        )
        self.recurrent_halting_stack_num_layers = recurrent_halting_stack_num_layers
        self.recurrent_halting_stack_activation = recurrent_halting_stack_activation
        self.recurrent_halting_stack_residual_connection_option = (
            recurrent_halting_stack_residual_connection_option
        )
        self.recurrent_halting_stack_dropout_probability = (
            recurrent_halting_stack_dropout_probability
        )
        self.recurrent_halting_stack_last_layer_bias_option = (
            recurrent_halting_stack_last_layer_bias_option
        )
        self.recurrent_halting_stack_apply_output_pipeline_flag = (
            recurrent_halting_stack_apply_output_pipeline_flag
        )
        self.recurrent_halting_bias_flag = recurrent_halting_bias_flag

    @staticmethod
    def _resolve_controller_option(override, shared_default):
        return shared_default if override is None else override

    def _resolve_recurrent_controller_option(
        self,
        recurrent_override,
        controller_override,
        shared_default,
    ):
        inherited_default = self._resolve_controller_option(
            controller_override,
            shared_default,
        )
        return self._resolve_controller_option(
            recurrent_override,
            inherited_default,
        )

    def build(self) -> "ModelConfig":
        from emperor.config import ModelConfig

        input_model_config = LayerConfig(
            activation=self.stack_activation,
            layer_norm_position=self.layer_norm_position,
            residual_connection_option=ResidualConnectionOptions.DISABLED,
            dropout_probability=self.stack_dropout_probability,
            gate_config=None,
            halting_config=None,
            layer_model_config=LinearLayerConfig(
                bias_flag=self.bias_flag,
            ),
        )

        gate_config = self._build_gate_config()
        self._validate_shared_gate_config(gate_config)
        halting_config = self._build_halting_config()
        memory_config = self._build_memory_config()
        model_config = LayerStackConfig(
            hidden_dim=self.hidden_dim,
            num_layers=self.stack_num_layers,
            last_layer_bias_option=self.stack_last_layer_bias_option,
            apply_output_pipeline_flag=self.stack_apply_output_pipeline_flag,
            shared_gate_config=self.shared_gate_config,
            shared_memory_config=memory_config,
            layer_config=LayerConfig(
                activation=self.stack_activation,
                layer_norm_position=self.layer_norm_position,
                residual_connection_option=self.stack_residual_connection_option,
                dropout_probability=self.stack_dropout_probability,
                gate_config=gate_config,
                halting_config=halting_config,
                layer_model_config=LinearLayerConfig(
                    bias_flag=self.bias_flag,
                ),
            ),
        )
        model_config = self._maybe_wrap_recurrent(model_config)

        output_model_config = LayerConfig(
            activation=ActivationOptions.DISABLED,
            layer_norm_position=LayerNormPositionOptions.DISABLED,
            residual_connection_option=ResidualConnectionOptions.DISABLED,
            dropout_probability=0.0,
            gate_config=None,
            halting_config=None,
            layer_model_config=LinearLayerConfig(
                bias_flag=self.bias_flag,
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

    def _validate_shared_gate_config(self, gate_config: GateConfig | None) -> None:
        if self._is_active_gate_config(
            self.shared_gate_config
        ) and self._is_active_gate_config(gate_config):
            raise ValueError(
                "shared_gate_config cannot be provided when stack_gate_flag "
                "enables per-layer gate_config."
            )

    @staticmethod
    def _is_active_gate_config(gate_config: GateConfig | None) -> bool:
        return gate_config is not None

    def _maybe_wrap_recurrent(
        self, block_config: LayerStackConfig
    ) -> "LayerStackConfig | RecurrentLayerConfig":
        if not self.recurrent_flag:
            return block_config
        return RecurrentLayerConfig(
            max_steps=self.recurrent_max_steps,
            recurrent_layer_norm_position=self.recurrent_layer_norm_position,
            block_config=block_config,
            gate_config=self._build_recurrent_gate_config(),
            residual_connection_option=ResidualConnectionOptions.DISABLED,
            halting_config=self._build_recurrent_halting_config(),
        )

    def _build_gate_config(
        self, enabled: bool | None = None
    ) -> GateConfig | None:
        if enabled is None:
            enabled = self.stack_gate_flag
        if not enabled:
            return None
        return GateConfig(
            model_config=self._build_gate_config_from_options(
                enabled=enabled,
                hidden_dim=self._resolve_controller_option(
                    self.gate_hidden_dim,
                    self.submodule_hidden_dim,
                ),
                num_layers=self._resolve_controller_option(
                    self.gate_stack_num_layers,
                    self.submodule_stack_num_layers,
                ),
                last_layer_bias_option=self._resolve_controller_option(
                    self.gate_stack_last_layer_bias_option,
                    self.submodule_stack_last_layer_bias_option,
                ),
                apply_output_pipeline_flag=self._resolve_controller_option(
                    self.gate_stack_apply_output_pipeline_flag,
                    self.submodule_stack_apply_output_pipeline_flag,
                ),
                activation=self._resolve_controller_option(
                    self.gate_stack_activation,
                    self.submodule_stack_activation,
                ),
                layer_norm_position=self._resolve_controller_option(
                    self.gate_layer_norm_position,
                    self.submodule_layer_norm_position,
                ),
                residual_connection_option=self._resolve_controller_option(
                    self.gate_stack_residual_connection_option,
                    self.submodule_stack_residual_connection_option,
                ),
                dropout_probability=self._resolve_controller_option(
                    self.gate_stack_dropout_probability,
                    self.submodule_stack_dropout_probability,
                ),
                bias_flag=self._resolve_controller_option(
                    self.gate_bias_flag,
                    self.submodule_bias_flag,
                ),
            ),
            option=self.gate_option,
            activation=self.gate_activation,
        )

    def _build_recurrent_gate_config(self) -> GateConfig | None:
        if not self.recurrent_gate_flag:
            return None
        return GateConfig(
            model_config=self._build_gate_config_from_options(
                enabled=self.recurrent_gate_flag,
                hidden_dim=self._resolve_recurrent_controller_option(
                    self.recurrent_gate_hidden_dim,
                    self.gate_hidden_dim,
                    self.submodule_hidden_dim,
                ),
                num_layers=self._resolve_recurrent_controller_option(
                    self.recurrent_gate_stack_num_layers,
                    self.gate_stack_num_layers,
                    self.submodule_stack_num_layers,
                ),
                last_layer_bias_option=self._resolve_recurrent_controller_option(
                    self.recurrent_gate_stack_last_layer_bias_option,
                    self.gate_stack_last_layer_bias_option,
                    self.submodule_stack_last_layer_bias_option,
                ),
                apply_output_pipeline_flag=self._resolve_recurrent_controller_option(
                    self.recurrent_gate_stack_apply_output_pipeline_flag,
                    self.gate_stack_apply_output_pipeline_flag,
                    self.submodule_stack_apply_output_pipeline_flag,
                ),
                activation=self._resolve_recurrent_controller_option(
                    self.recurrent_gate_stack_activation,
                    self.gate_stack_activation,
                    self.submodule_stack_activation,
                ),
                layer_norm_position=self._resolve_recurrent_controller_option(
                    self.recurrent_gate_layer_norm_position,
                    self.gate_layer_norm_position,
                    self.submodule_layer_norm_position,
                ),
                residual_connection_option=self._resolve_recurrent_controller_option(
                    self.recurrent_gate_stack_residual_connection_option,
                    self.gate_stack_residual_connection_option,
                    self.submodule_stack_residual_connection_option,
                ),
                dropout_probability=self._resolve_recurrent_controller_option(
                    self.recurrent_gate_stack_dropout_probability,
                    self.gate_stack_dropout_probability,
                    self.submodule_stack_dropout_probability,
                ),
                bias_flag=self._resolve_recurrent_controller_option(
                    self.recurrent_gate_bias_flag,
                    self.gate_bias_flag,
                    self.submodule_bias_flag,
                ),
            ),
            option=self.recurrent_gate_option,
            activation=self.recurrent_gate_activation,
        )

    def _build_gate_model_config(
        self, enabled: bool | None = None
    ) -> LayerStackConfig | None:
        if enabled is None:
            enabled = self.stack_gate_flag
        return self._build_gate_config_from_options(
            enabled=enabled,
            hidden_dim=self._resolve_controller_option(
                self.gate_hidden_dim,
                self.submodule_hidden_dim,
            ),
            num_layers=self._resolve_controller_option(
                self.gate_stack_num_layers,
                self.submodule_stack_num_layers,
            ),
            last_layer_bias_option=self._resolve_controller_option(
                self.gate_stack_last_layer_bias_option,
                self.submodule_stack_last_layer_bias_option,
            ),
            apply_output_pipeline_flag=self._resolve_controller_option(
                self.gate_stack_apply_output_pipeline_flag,
                self.submodule_stack_apply_output_pipeline_flag,
            ),
            activation=self._resolve_controller_option(
                self.gate_stack_activation,
                self.submodule_stack_activation,
            ),
            layer_norm_position=self._resolve_controller_option(
                self.gate_layer_norm_position,
                self.submodule_layer_norm_position,
            ),
            residual_connection_option=self._resolve_controller_option(
                self.gate_stack_residual_connection_option,
                self.submodule_stack_residual_connection_option,
            ),
            dropout_probability=self._resolve_controller_option(
                self.gate_stack_dropout_probability,
                self.submodule_stack_dropout_probability,
            ),
            bias_flag=self._resolve_controller_option(
                self.gate_bias_flag,
                self.submodule_bias_flag,
            ),
        )

    def _build_gate_config_from_options(
        self,
        *,
        enabled: bool,
        hidden_dim: int,
        num_layers: int,
        last_layer_bias_option: LastLayerBiasOptions,
        apply_output_pipeline_flag: bool,
        activation: ActivationOptions,
        layer_norm_position: LayerNormPositionOptions,
        residual_connection_option: ResidualConnectionOptions,
        dropout_probability: float,
        bias_flag: bool,
    ) -> LayerStackConfig | None:
        if not enabled:
            return None
        return LayerStackConfig(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            last_layer_bias_option=last_layer_bias_option,
            apply_output_pipeline_flag=apply_output_pipeline_flag,
            layer_config=LayerConfig(
                activation=activation,
                layer_norm_position=layer_norm_position,
                residual_connection_option=residual_connection_option,
                dropout_probability=dropout_probability,
                halting_config=None,
                gate_config=None,
                layer_model_config=LinearLayerConfig(
                    bias_flag=bias_flag,
                ),
            ),
        )

    def _build_halting_config(
        self,
        enabled: bool | None = None,
    ) -> StickBreakingConfig | None:
        if enabled is None:
            enabled = self.stack_halting_flag
        return self._build_halting_config_from_options(
            enabled=enabled,
            threshold=self.halting_threshold,
            halting_dropout=self.halting_dropout,
            hidden_state_mode=self.halting_hidden_state_mode,
            hidden_dim=self._resolve_controller_option(
                self.halting_hidden_dim,
                self.submodule_hidden_dim,
            ),
            output_dim=self.halting_output_dim,
            num_layers=self._resolve_controller_option(
                self.halting_stack_num_layers,
                self.submodule_stack_num_layers,
            ),
            last_layer_bias_option=self._resolve_controller_option(
                self.halting_stack_last_layer_bias_option,
                self.submodule_stack_last_layer_bias_option,
            ),
            apply_output_pipeline_flag=self._resolve_controller_option(
                self.halting_stack_apply_output_pipeline_flag,
                self.submodule_stack_apply_output_pipeline_flag,
            ),
            activation=self._resolve_controller_option(
                self.halting_stack_activation,
                self.submodule_stack_activation,
            ),
            layer_norm_position=self._resolve_controller_option(
                self.halting_layer_norm_position,
                self.submodule_layer_norm_position,
            ),
            residual_connection_option=self._resolve_controller_option(
                self.halting_stack_residual_connection_option,
                self.submodule_stack_residual_connection_option,
            ),
            dropout_probability=self._resolve_controller_option(
                self.halting_stack_dropout_probability,
                self.submodule_stack_dropout_probability,
            ),
            bias_flag=self._resolve_controller_option(
                self.halting_bias_flag,
                self.submodule_bias_flag,
            ),
        )

    def _build_recurrent_halting_config(self) -> StickBreakingConfig | None:
        return self._build_halting_config_from_options(
            enabled=self.recurrent_halting_flag,
            threshold=self.recurrent_halting_threshold,
            halting_dropout=self.recurrent_halting_dropout,
            hidden_state_mode=self.recurrent_halting_hidden_state_mode,
            hidden_dim=self._resolve_recurrent_controller_option(
                self.recurrent_halting_hidden_dim,
                self.halting_hidden_dim,
                self.submodule_hidden_dim,
            ),
            output_dim=self.recurrent_halting_output_dim,
            num_layers=self._resolve_recurrent_controller_option(
                self.recurrent_halting_stack_num_layers,
                self.halting_stack_num_layers,
                self.submodule_stack_num_layers,
            ),
            last_layer_bias_option=self._resolve_recurrent_controller_option(
                self.recurrent_halting_stack_last_layer_bias_option,
                self.halting_stack_last_layer_bias_option,
                self.submodule_stack_last_layer_bias_option,
            ),
            apply_output_pipeline_flag=self._resolve_recurrent_controller_option(
                self.recurrent_halting_stack_apply_output_pipeline_flag,
                self.halting_stack_apply_output_pipeline_flag,
                self.submodule_stack_apply_output_pipeline_flag,
            ),
            activation=self._resolve_recurrent_controller_option(
                self.recurrent_halting_stack_activation,
                self.halting_stack_activation,
                self.submodule_stack_activation,
            ),
            layer_norm_position=self._resolve_recurrent_controller_option(
                self.recurrent_halting_layer_norm_position,
                self.halting_layer_norm_position,
                self.submodule_layer_norm_position,
            ),
            residual_connection_option=self._resolve_recurrent_controller_option(
                self.recurrent_halting_stack_residual_connection_option,
                self.halting_stack_residual_connection_option,
                self.submodule_stack_residual_connection_option,
            ),
            dropout_probability=self._resolve_recurrent_controller_option(
                self.recurrent_halting_stack_dropout_probability,
                self.halting_stack_dropout_probability,
                self.submodule_stack_dropout_probability,
            ),
            bias_flag=self._resolve_recurrent_controller_option(
                self.recurrent_halting_bias_flag,
                self.halting_bias_flag,
                self.submodule_bias_flag,
            ),
        )

    def _build_halting_config_from_options(
        self,
        *,
        enabled: bool,
        threshold: float,
        halting_dropout: float,
        hidden_state_mode: HaltingHiddenStateModeOptions,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        last_layer_bias_option: LastLayerBiasOptions,
        apply_output_pipeline_flag: bool,
        activation: ActivationOptions,
        layer_norm_position: LayerNormPositionOptions,
        residual_connection_option: ResidualConnectionOptions,
        dropout_probability: float,
        bias_flag: bool,
    ) -> StickBreakingConfig | None:
        if not enabled:
            return None
        return StickBreakingConfig(
            threshold=threshold,
            halting_dropout=halting_dropout,
            hidden_state_mode=hidden_state_mode,
            halting_gate_config=LayerStackConfig(
                hidden_dim=hidden_dim or self.output_dim,
                output_dim=output_dim,
                num_layers=num_layers,
                last_layer_bias_option=last_layer_bias_option,
                apply_output_pipeline_flag=apply_output_pipeline_flag,
                layer_config=LayerConfig(
                    activation=activation,
                    layer_norm_position=layer_norm_position,
                    residual_connection_option=residual_connection_option,
                    dropout_probability=dropout_probability,
                    halting_config=None,
                    gate_config=None,
                    layer_model_config=LinearLayerConfig(
                        bias_flag=bias_flag,
                    ),
                ),
            ),
        )

    def _build_memory_config(
        self,
        enabled: bool | None = None,
    ) -> DynamicMemoryConfig | None:
        if enabled is None:
            enabled = self.memory_flag
        return self._build_memory_config_from_options(
            enabled=enabled,
            memory_option=self.memory_option,
            memory_position_option=self.memory_position_option,
            test_time_training_learning_rate=(
                self.memory_test_time_training_learning_rate
            ),
            test_time_training_num_inner_steps=(
                self.memory_test_time_training_num_inner_steps
            ),
            hidden_dim=self._resolve_controller_option(
                self.memory_hidden_dim,
                self.submodule_hidden_dim,
            ),
            num_layers=self._resolve_controller_option(
                self.memory_stack_num_layers,
                self.submodule_stack_num_layers,
            ),
            last_layer_bias_option=self._resolve_controller_option(
                self.memory_stack_last_layer_bias_option,
                self.submodule_stack_last_layer_bias_option,
            ),
            apply_output_pipeline_flag=self._resolve_controller_option(
                self.memory_stack_apply_output_pipeline_flag,
                self.submodule_stack_apply_output_pipeline_flag,
            ),
            activation=self._resolve_controller_option(
                self.memory_stack_activation,
                self.submodule_stack_activation,
            ),
            layer_norm_position=self._resolve_controller_option(
                self.memory_layer_norm_position,
                self.submodule_layer_norm_position,
            ),
            residual_connection_option=self._resolve_controller_option(
                self.memory_stack_residual_connection_option,
                self.submodule_stack_residual_connection_option,
            ),
            dropout_probability=self._resolve_controller_option(
                self.memory_stack_dropout_probability,
                self.submodule_stack_dropout_probability,
            ),
            bias_flag=self._resolve_controller_option(
                self.memory_bias_flag,
                self.submodule_bias_flag,
            ),
        )

    def _build_memory_config_from_options(
        self,
        *,
        enabled: bool,
        memory_option: type[DynamicMemoryConfig],
        memory_position_option: MemoryPositionOptions,
        test_time_training_learning_rate: float | None,
        test_time_training_num_inner_steps: int | None,
        hidden_dim: int,
        num_layers: int,
        last_layer_bias_option: LastLayerBiasOptions,
        apply_output_pipeline_flag: bool,
        activation: ActivationOptions,
        layer_norm_position: LayerNormPositionOptions,
        residual_connection_option: ResidualConnectionOptions,
        dropout_probability: float,
        bias_flag: bool,
    ) -> DynamicMemoryConfig | None:
        if not enabled:
            return None
        return memory_option(
            input_dim=self.hidden_dim,
            output_dim=self.hidden_dim,
            memory_position_option=memory_position_option,
            test_time_training_learning_rate=test_time_training_learning_rate,
            test_time_training_num_inner_steps=test_time_training_num_inner_steps,
            model_config=LayerStackConfig(
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                last_layer_bias_option=last_layer_bias_option,
                apply_output_pipeline_flag=apply_output_pipeline_flag,
                layer_config=LayerConfig(
                    activation=activation,
                    layer_norm_position=layer_norm_position,
                    residual_connection_option=residual_connection_option,
                    dropout_probability=dropout_probability,
                    halting_config=None,
                    gate_config=None,
                    memory_config=None,
                    layer_model_config=LinearLayerConfig(
                        bias_flag=bias_flag,
                    ),
                ),
            ),
        )
