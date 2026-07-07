from dataclasses import dataclass

from emperor.base.layer.config import LayerConfig
from emperor.base.layer.residual import ResidualConnectionOptions
from emperor.base.options import ActivationOptions, LayerNormPositionOptions
from emperor.linears.core.config import LinearLayerConfig
from models.linears._builder_options import MainLayerStackOptions

import models.linears.linear.config as config


@dataclass(frozen=True)
class BoundaryModelConfigDependencies:
    stack_options: MainLayerStackOptions | None


class BoundaryModelConfigFactory:
    def __init__(self, dependencies: BoundaryModelConfigDependencies) -> None:
        stack_options = self.__default_stack_options(dependencies.stack_options)

        self.stack_options = stack_options

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

    def build_input_model_config(self) -> LayerConfig:
        return self.__build_boundary_layer_config(
            activation=self.stack_options.activation,
        )

    def build_output_model_config(self) -> LayerConfig:
        return self.__build_boundary_layer_config(
            activation=ActivationOptions.DISABLED,
        )

    def __build_boundary_layer_config(
        self,
        *,
        activation: ActivationOptions,
    ) -> LayerConfig:
        return LayerConfig(
            activation=activation,
            layer_norm_position=LayerNormPositionOptions.DISABLED,
            residual_connection_option=ResidualConnectionOptions.DISABLED,
            dropout_probability=0.0,
            gate_config=None,
            halting_config=None,
            layer_model_config=LinearLayerConfig(
                bias_flag=True,
            ),
        )
