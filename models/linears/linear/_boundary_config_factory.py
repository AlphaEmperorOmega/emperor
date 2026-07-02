from dataclasses import dataclass

from emperor.base.layer.config import LayerConfig
from emperor.base.layer.residual import ResidualConnectionOptions
from emperor.base.options import ActivationOptions, LayerNormPositionOptions
from emperor.linears.core.config import LinearLayerConfig
from models.linears._builder_options import LinearStackOptions


@dataclass(frozen=True)
class BoundaryConfigDependencies:
    stack_options: LinearStackOptions


class BoundaryConfigFactory:
    def __init__(self, dependencies: BoundaryConfigDependencies) -> None:
        self.stack_options = dependencies.stack_options

    def build_input_model_config(self) -> LayerConfig:
        return self.__build_boundary_layer_config(
            activation=self.stack_options.activation,
        )

    def build_output_model_config(self) -> LayerConfig:
        return self.__build_boundary_layer_config(
            activation=ActivationOptions.DISABLED,
        )

    @staticmethod
    def __build_boundary_layer_config(
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
