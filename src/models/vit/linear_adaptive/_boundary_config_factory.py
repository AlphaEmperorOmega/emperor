from dataclasses import dataclass

import models.vit.linear_adaptive.config as config
from emperor.layers import (
    ActivationOptions,
    LayerConfig,
    LayerNormPositionOptions,
)
from emperor.linears import LinearLayerConfig
from models.vit.linear_adaptive.runtime_options import VitOutputOptions


@dataclass(frozen=True)
class BoundaryConfigDependencies:
    hidden_dim: int
    output_dim: int
    output_options: VitOutputOptions | None


class BoundaryConfigFactory:
    def __init__(self, dependencies: BoundaryConfigDependencies) -> None:
        self.hidden_dim = dependencies.hidden_dim
        self.output_dim = dependencies.output_dim
        self.output_options = self.__default_output_options(dependencies.output_options)

    def __default_output_options(
        self,
        output_options: VitOutputOptions | None,
    ) -> VitOutputOptions:
        if output_options is not None:
            return output_options
        return VitOutputOptions(bias_flag=config.OUTPUT_BIAS_FLAG)

    def build_output_config(self) -> LayerConfig:
        return LayerConfig(
            input_dim=self.hidden_dim,
            output_dim=self.output_dim,
            activation=ActivationOptions.DISABLED,
            layer_norm_position=LayerNormPositionOptions.DISABLED,
            residual_config=None,
            dropout_probability=0.0,
            gate_config=None,
            halting_config=None,
            memory_config=None,
            layer_model_config=LinearLayerConfig(
                bias_flag=self.output_options.bias_flag,
            ),
        )
