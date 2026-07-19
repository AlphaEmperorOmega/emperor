from dataclasses import dataclass

import models.neuron.expert_linear.config as config
from emperor.layers import (
    ActivationOptions,
    LayerConfig,
    LayerNormPositionOptions,
)
from emperor.linears import LinearLayerConfig
from models.neuron.expert_linear._hidden.runtime_options import ExpertsStackOptions


@dataclass(frozen=True, slots=True)
class ProjectionConfigDependencies:
    hidden_dim: int
    stack_options: ExpertsStackOptions | None


class ProjectionConfigFactory:
    def __init__(self, dependencies: ProjectionConfigDependencies) -> None:
        self.hidden_dim = dependencies.hidden_dim
        self.stack_options = dependencies.stack_options or ExpertsStackOptions(
            hidden_dim=dependencies.hidden_dim,
            bias_flag=config.STACK_BIAS_FLAG,
            layer_norm_position=config.STACK_LAYER_NORM_POSITION,
            num_layers=config.STACK_NUM_LAYERS,
            activation=config.STACK_ACTIVATION,
            residual_connection_option=config.STACK_RESIDUAL_CONNECTION_OPTION,
            dropout_probability=config.STACK_DROPOUT_PROBABILITY,
            last_layer_bias_option=config.STACK_LAST_LAYER_BIAS_OPTION,
            apply_output_pipeline_flag=config.STACK_APPLY_OUTPUT_PIPELINE_FLAG,
        )

    def build_input_model_config(self) -> LayerConfig:
        return LayerConfig(
            activation=self.stack_options.activation,
            layer_norm_position=self.stack_options.layer_norm_position,
            residual_config=None,
            dropout_probability=self.stack_options.dropout_probability,
            gate_config=None,
            halting_config=None,
            memory_config=None,
            layer_model_config=LinearLayerConfig(bias_flag=True),
        )

    def build_output_model_config(self) -> LayerConfig:
        return LayerConfig(
            activation=ActivationOptions.DISABLED,
            layer_norm_position=LayerNormPositionOptions.DISABLED,
            residual_config=None,
            dropout_probability=0.0,
            gate_config=None,
            halting_config=None,
            memory_config=None,
            layer_model_config=LinearLayerConfig(bias_flag=True),
        )
