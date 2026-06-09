import models.linears.linear.config as config

from emperor.base.options import (
    ActivationOptions,
    LastLayerBiasOptions,
    LayerNormPositionOptions,
)
from emperor.base.layer import LayerStackConfig
from emperor.base.layer.config import LayerConfig, RecurrentLayerConfig
from emperor.halting.config import StickBreakingConfig
from emperor.halting.options import HaltingHiddenStateModeOptions
from emperor.linears.core.config import LinearLayerConfig
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
        stack_residual_flag: bool = config.STACK_RESIDUAL_FLAG,
        stack_dropout_probability: float = config.STACK_DROPOUT_PROBABILITY,
        stack_last_layer_bias_option: LastLayerBiasOptions = config.STACK_LAST_LAYER_BIAS_OPTION,
        stack_apply_output_pipeline_flag: bool = config.STACK_APPLY_OUTPUT_PIPELINE_FLAG,
        stack_gate_flag: bool = config.GATE_FLAG,
        gate_hidden_dim: int = config.GATE_HIDDEN_DIM,
        gate_layer_norm_position: LayerNormPositionOptions = config.GATE_LAYER_NORM_POSITION,
        gate_stack_num_layers: int = config.GATE_STACK_NUM_LAYERS,
        gate_stack_activation: ActivationOptions = config.GATE_STACK_ACTIVATION,
        gate_stack_residual_flag: bool = config.GATE_STACK_RESIDUAL_FLAG,
        gate_stack_dropout_probability: float = config.GATE_STACK_DROPOUT_PROBABILITY,
        gate_stack_last_layer_bias_option: LastLayerBiasOptions = config.GATE_STACK_LAST_LAYER_BIAS_OPTION,
        gate_stack_apply_output_pipeline_flag: bool = config.GATE_STACK_APPLY_OUTPUT_PIPELINE_FLAG,
        gate_bias_flag: bool = config.GATE_BIAS_FLAG,
        stack_halting_flag: bool = config.HALTING_FLAG,
        halting_threshold: float = config.HALTING_THRESHOLD,
        halting_dropout: float = config.HALTING_DROPOUT,
        halting_hidden_state_mode: HaltingHiddenStateModeOptions = (
            config.HALTING_HIDDEN_STATE_MODE
        ),
        halting_hidden_dim: int = config.HALTING_HIDDEN_DIM,
        halting_output_dim: int = config.HALTING_OUTPUT_DIM,
        halting_layer_norm_position: LayerNormPositionOptions = config.HALTING_LAYER_NORM_POSITION,
        halting_stack_num_layers: int = config.HALTING_STACK_NUM_LAYERS,
        halting_stack_activation: ActivationOptions = config.HALTING_STACK_ACTIVATION,
        halting_stack_residual_flag: bool = config.HALTING_STACK_RESIDUAL_FLAG,
        halting_stack_dropout_probability: float = config.HALTING_STACK_DROPOUT_PROBABILITY,
        halting_stack_last_layer_bias_option: LastLayerBiasOptions = config.HALTING_STACK_LAST_LAYER_BIAS_OPTION,
        halting_stack_apply_output_pipeline_flag: bool = config.HALTING_STACK_APPLY_OUTPUT_PIPELINE_FLAG,
        halting_bias_flag: bool = config.HALTING_BIAS_FLAG,
        recurrent_flag: bool = config.RECURRENT_FLAG,
        recurrent_max_steps: int = config.RECURRENT_MAX_STEPS,
        recurrent_gate_flag: bool = config.RECURRENT_GATE_FLAG,
        recurrent_halting_flag: bool = config.RECURRENT_HALTING_FLAG,
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
        self.stack_residual_flag = stack_residual_flag
        self.stack_dropout_probability = stack_dropout_probability
        self.stack_last_layer_bias_option = stack_last_layer_bias_option
        self.stack_apply_output_pipeline_flag = stack_apply_output_pipeline_flag
        self.stack_gate_flag = stack_gate_flag
        self.gate_hidden_dim = gate_hidden_dim
        self.gate_layer_norm_position = gate_layer_norm_position
        self.gate_stack_num_layers = gate_stack_num_layers
        self.gate_stack_activation = gate_stack_activation
        self.gate_stack_residual_flag = gate_stack_residual_flag
        self.gate_stack_dropout_probability = gate_stack_dropout_probability
        self.gate_stack_last_layer_bias_option = gate_stack_last_layer_bias_option
        self.gate_stack_apply_output_pipeline_flag = (
            gate_stack_apply_output_pipeline_flag
        )
        self.gate_bias_flag = gate_bias_flag
        self.stack_halting_flag = stack_halting_flag
        self.halting_threshold = halting_threshold
        self.halting_dropout = halting_dropout
        self.halting_hidden_state_mode = halting_hidden_state_mode
        self.halting_hidden_dim = halting_hidden_dim
        self.halting_output_dim = halting_output_dim
        self.halting_layer_norm_position = halting_layer_norm_position
        self.halting_stack_num_layers = halting_stack_num_layers
        self.halting_stack_activation = halting_stack_activation
        self.halting_stack_residual_flag = halting_stack_residual_flag
        self.halting_stack_dropout_probability = halting_stack_dropout_probability
        self.halting_stack_last_layer_bias_option = halting_stack_last_layer_bias_option
        self.halting_stack_apply_output_pipeline_flag = (
            halting_stack_apply_output_pipeline_flag
        )
        self.halting_bias_flag = halting_bias_flag
        self.recurrent_flag = recurrent_flag
        self.recurrent_max_steps = recurrent_max_steps
        self.recurrent_gate_flag = recurrent_gate_flag
        self.recurrent_halting_flag = recurrent_halting_flag

    def build(self) -> "ModelConfig":
        from emperor.config import ModelConfig

        input_model_config = LayerConfig(
            activation=self.stack_activation,
            layer_norm_position=self.layer_norm_position,
            residual_flag=False,
            dropout_probability=self.stack_dropout_probability,
            gate_config=None,
            halting_config=None,
            shared_halting_flag=False,
            layer_model_config=LinearLayerConfig(
                bias_flag=self.bias_flag,
            ),
        )

        gate_config = self._build_gate_config()
        halting_config = self._build_halting_config()
        model_config = LayerStackConfig(
            hidden_dim=self.hidden_dim,
            num_layers=self.stack_num_layers,
            last_layer_bias_option=self.stack_last_layer_bias_option,
            apply_output_pipeline_flag=self.stack_apply_output_pipeline_flag,
            layer_config=LayerConfig(
                activation=self.stack_activation,
                layer_norm_position=self.layer_norm_position,
                residual_flag=self.stack_residual_flag,
                dropout_probability=self.stack_dropout_probability,
                gate_config=gate_config,
                halting_config=halting_config,
                shared_halting_flag=False,
                layer_model_config=LinearLayerConfig(
                    bias_flag=self.bias_flag,
                ),
            ),
        )
        model_config = self._maybe_wrap_recurrent(model_config)

        output_model_config = LayerConfig(
            activation=ActivationOptions.DISABLED,
            layer_norm_position=LayerNormPositionOptions.DISABLED,
            residual_flag=False,
            dropout_probability=0.0,
            gate_config=None,
            halting_config=None,
            shared_halting_flag=False,
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

    def _maybe_wrap_recurrent(
        self, block_config: LayerStackConfig
    ) -> "LayerStackConfig | RecurrentLayerConfig":
        if not self.recurrent_flag:
            return block_config
        return RecurrentLayerConfig(
            max_steps=self.recurrent_max_steps,
            block_config=block_config,
            gate_config=self._build_gate_config(self.recurrent_gate_flag),
            halting_config=self._build_halting_config(self.recurrent_halting_flag),
        )

    def _build_gate_config(self, enabled: bool | None = None) -> LayerStackConfig | None:
        if enabled is None:
            enabled = self.stack_gate_flag
        if not enabled:
            return None
        return LayerStackConfig(
            hidden_dim=self.gate_hidden_dim,
            num_layers=self.gate_stack_num_layers,
            last_layer_bias_option=self.gate_stack_last_layer_bias_option,
            apply_output_pipeline_flag=self.gate_stack_apply_output_pipeline_flag,
            layer_config=LayerConfig(
                activation=self.gate_stack_activation,
                layer_norm_position=self.gate_layer_norm_position,
                residual_flag=self.gate_stack_residual_flag,
                dropout_probability=self.gate_stack_dropout_probability,
                halting_config=None,
                shared_halting_flag=False,
                gate_config=None,
                layer_model_config=LinearLayerConfig(
                    bias_flag=self.gate_bias_flag,
                ),
            ),
        )

    def _build_halting_config(
        self,
        enabled: bool | None = None,
    ) -> StickBreakingConfig | None:
        if enabled is None:
            enabled = self.stack_halting_flag
        if not enabled:
            return None
        return StickBreakingConfig(
            threshold=self.halting_threshold,
            halting_dropout=self.halting_dropout,
            hidden_state_mode=self.halting_hidden_state_mode,
            halting_gate_config=LayerStackConfig(
                hidden_dim=self.halting_hidden_dim or self.output_dim,
                output_dim=self.halting_output_dim,
                num_layers=self.halting_stack_num_layers,
                last_layer_bias_option=self.halting_stack_last_layer_bias_option,
                apply_output_pipeline_flag=self.halting_stack_apply_output_pipeline_flag,
                layer_config=LayerConfig(
                    activation=self.halting_stack_activation,
                    layer_norm_position=self.halting_layer_norm_position,
                    residual_flag=self.halting_stack_residual_flag,
                    dropout_probability=self.halting_stack_dropout_probability,
                    halting_config=None,
                    shared_halting_flag=False,
                    gate_config=None,
                    layer_model_config=LinearLayerConfig(
                        bias_flag=self.halting_bias_flag,
                    ),
                ),
            ),
        )
