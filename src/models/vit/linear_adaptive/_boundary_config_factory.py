from dataclasses import dataclass

import models.vit.linear_adaptive.config as config
from emperor.layers import (
    ActivationOptions,
    LayerConfig,
    LayerNormPositionOptions,
    NormalizationOptions,
)
from emperor.linears import LinearLayerConfig
from models.vit.linear_adaptive import _config_defaults as config_defaults
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
        self.output_options = (
            config_defaults.vit_output_options(config)
            if dependencies.output_options is None
            else dependencies.output_options
        )

    def build_output_config(self) -> LayerConfig:
        return LayerConfig(
            input_dim=self.hidden_dim,
            output_dim=self.output_dim,
            activation=ActivationOptions.DISABLED,
            layer_norm_position=LayerNormPositionOptions.DISABLED,
            normalization=NormalizationOptions.RMS_NORM,
            residual_config=None,
            dropout_probability=0.0,
            gate_config=None,
            halting_config=None,
            memory_config=None,
            layer_model_config=LinearLayerConfig(
                bias_flag=self.output_options.bias_flag,
            ),
        )
