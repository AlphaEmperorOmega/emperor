from dataclasses import dataclass

from emperor.base.layer.config import LayerConfig
from emperor.base.layer.residual import ResidualConnectionOptions
from emperor.base.options import ActivationOptions, LayerNormPositionOptions
from emperor.linears.core.config import LinearLayerConfig

import models.experts.linear.config as config
from models.experts._builder_options import ExpertsStackOptions


@dataclass(frozen=True)
class BoundaryConfigDependencies:
    stack_options: ExpertsStackOptions | None


class BoundaryConfigFactory:
    def __init__(self, dependencies: BoundaryConfigDependencies) -> None:
        self.stack_options = self.__default_stack_options(dependencies.stack_options)

    def __default_stack_options(
        self,
        stack_options: ExpertsStackOptions | None,
    ) -> ExpertsStackOptions:
        if stack_options is not None:
            return stack_options
        return ExpertsStackOptions(
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

    def build_input_model_config(self) -> LayerConfig:
        return LayerConfig(
            activation=self.stack_options.activation,
            layer_norm_position=self.stack_options.layer_norm_position,
            residual_connection_option=ResidualConnectionOptions.DISABLED,
            dropout_probability=self.stack_options.dropout_probability,
            gate_config=None,
            halting_config=None,
            memory_config=None,
            layer_model_config=LinearLayerConfig(
                bias_flag=self.stack_options.bias_flag,
            ),
        )

    def build_output_model_config(self) -> LayerConfig:
        return LayerConfig(
            activation=ActivationOptions.DISABLED,
            layer_norm_position=LayerNormPositionOptions.DISABLED,
            residual_connection_option=ResidualConnectionOptions.DISABLED,
            dropout_probability=0.0,
            gate_config=None,
            halting_config=None,
            memory_config=None,
            layer_model_config=LinearLayerConfig(
                bias_flag=self.stack_options.bias_flag,
            ),
        )
