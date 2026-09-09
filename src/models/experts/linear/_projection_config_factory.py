from dataclasses import dataclass

from emperor.layers import (
    ActivationOptions,
    LayerConfig,
    LayerNormPositionOptions,
    NormalizationOptions,
)
from emperor.linears import LinearLayerConfig
from models.experts.linear.runtime_options import ExpertsStackOptions


@dataclass(frozen=True, slots=True)
class ProjectionConfigDependencies:
    stack_options: ExpertsStackOptions


class ProjectionConfigFactory:
    def __init__(self, dependencies: ProjectionConfigDependencies) -> None:
        self.stack_options = dependencies.stack_options

    def build_input_model_config(self) -> LayerConfig:
        return LayerConfig(
            activation=self.stack_options.activation,
            layer_norm_position=self.stack_options.layer_norm_position,
            normalization=self.stack_options.normalization,
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
            normalization=NormalizationOptions.RMS_NORM,
            residual_config=None,
            dropout_probability=0.0,
            gate_config=None,
            halting_config=None,
            memory_config=None,
            layer_model_config=LinearLayerConfig(bias_flag=True),
        )
