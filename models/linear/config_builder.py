import models.linear.config as config

from emperor.base.enums import (
    ActivationOptions,
    LastLayerBiasOptions,
    LayerNormPositionOptions,
)
from emperor.base.layer import LayerStackConfig
from emperor.base.layer.config import LayerConfig
from emperor.halting.config import StickBreakingConfig
from emperor.linears.core.config import LinearLayerConfig
from models.linear.experiment_config import ExperimentConfig

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
        stack_halting_flag: bool = config.HALTING_FLAG,
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
        self.stack_halting_flag = stack_halting_flag

    def build(self) -> "ModelConfig":
        from emperor.config import ModelConfig

        gate_config = self._build_gate_config()
        halting_config = self._build_halting_config()
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

        output_model_config = LayerConfig(
            activation=ActivationOptions.DISABLED,
            layer_norm_position=LayerNormPositionOptions.DEFAULT,
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

    def _build_gate_config(self) -> LayerStackConfig | None:
        if not self.stack_gate_flag:
            return None
        return LayerStackConfig(
            hidden_dim=config.GATE_HIDDEN_DIM,
            num_layers=config.GATE_STACK_NUM_LAYERS,
            last_layer_bias_option=config.GATE_STACK_LAST_LAYER_BIAS_OPTION,
            apply_output_pipeline_flag=config.GATE_STACK_APPLY_OUTPUT_PIPELINE_FLAG,
            layer_config=LayerConfig(
                activation=config.GATE_STACK_ACTIVATION,
                layer_norm_position=config.GATE_LAYER_NORM_POSITION,
                residual_flag=config.GATE_STACK_RESIDUAL_FLAG,
                dropout_probability=config.GATE_STACK_DROPOUT_PROBABILITY,
                halting_config=None,
                shared_halting_flag=False,
                gate_config=None,
                layer_model_config=LinearLayerConfig(
                    bias_flag=config.GATE_BIAS_FLAG,
                ),
            ),
        )

    def _build_halting_config(self) -> StickBreakingConfig | None:
        if not self.stack_halting_flag:
            return None
        return StickBreakingConfig(
            threshold=config.HALTING_THRESHOLD,
            halting_dropout=config.HALTING_DROPOUT,
            hidden_state_mode=config.HALTING_HIDDEN_STATE_MODE,
            halting_gate_config=LayerStackConfig(
                hidden_dim=config.HALTING_GATE_HIDDEN_DIM or self.output_dim,
                output_dim=config.HALTING_GATE_OUTPUT_DIM,
                num_layers=config.HALTING_GATE_STACK_NUM_LAYERS,
                last_layer_bias_option=config.HALTING_GATE_STACK_LAST_LAYER_BIAS_OPTION,
                apply_output_pipeline_flag=config.HALTING_GATE_STACK_APPLY_OUTPUT_PIPELINE_FLAG,
                layer_config=LayerConfig(
                    activation=config.HALTING_GATE_STACK_ACTIVATION,
                    layer_norm_position=config.HALTING_GATE_LAYER_NORM_POSITION,
                    residual_flag=config.HALTING_GATE_STACK_RESIDUAL_FLAG,
                    dropout_probability=config.HALTING_GATE_STACK_DROPOUT_PROBABILITY,
                    halting_config=None,
                    shared_halting_flag=False,
                    gate_config=None,
                    layer_model_config=LinearLayerConfig(
                        bias_flag=config.HALTING_GATE_BIAS_FLAG,
                    ),
                ),
            ),
        )
