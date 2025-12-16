from Emperor.base.enums import ActivationOptions
from Emperor.experiments.utils.factories import Experiments
from Emperor.linears.options import LinearLayerOptions, LinearLayerStackOptions
from Emperor.linears.utils.presets import LinearPresets
from Emperor.behaviours.utils.enums import (
    DynamicDepthOptions,
    DynamicDiagonalOptions,
    DynamicBiasOptions,
    LinearMemoryOptions,
    LinearMemoryPositionOptions,
    LinearMemorySizeOptions,
)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from Emperor.config import ModelConfig


class LinearsExperiments(Experiments):
    def __init__(
        self,
        mini_datasetset_flag: bool = True,
    ) -> None:
        super().__init__(mini_datasetset_flag)

    def train_model(self, layer_type: LinearLayerOptions | LinearLayerStackOptions):
        preset = LinearsBasePreset(layer_type).get_config()
        self._set_model_config(preset)
        self._train_model(layer_type)

    def train_base_model(self):
        self.train_model(LinearLayerOptions.BASE)

    def train_adaptive_model(self):
        self.train_model(LinearLayerOptions.ADAPTIVE)

    def train_base_stack_model(self):
        self.train_model(LinearLayerStackOptions.BASE)

    def train_adaptive_stack_model(self):
        self.train_model(LinearLayerStackOptions.ADAPTIVE)

    def test_all_linear_types(self):
        option_types = [LinearLayerOptions, LinearLayerStackOptions]

        for option_type in option_types:
            for option in option_type:
                self.train_model(option)


class LinearsBasePreset:
    def __init__(
        self,
        linear_layer_options: LinearLayerOptions | LinearLayerStackOptions,
    ) -> None:
        self.linear_layer_options = linear_layer_options

    def get_config(self) -> "ModelConfig":
        match self.linear_layer_options:
            case LinearLayerOptions.BASE:
                return LinearPresets.base_linear_layer_preset(
                    batch_size=64,
                    input_dim=784,
                    output_dim=10,
                    bias_flag=True,
                    data_monitor=None,
                    parameter_monitor=None,
                )
            case LinearLayerOptions.ADAPTIVE:
                return LinearPresets.adaptive_linear_layer_preset(
                    batch_size=64,
                    input_dim=784,
                    output_dim=10,
                    bias_flag=True,
                    generator_depth=DynamicDepthOptions.DEPTH_OF_TWO,
                    diagonal_option=DynamicDiagonalOptions.DISABLED,
                    bias_option=DynamicBiasOptions.DYNAMIC_PARAMETERS,
                    memory_option=LinearMemoryOptions.FUSION,
                    memory_size_option=LinearMemorySizeOptions.LARGE,
                    memory_position_option=LinearMemoryPositionOptions.BEFORE_AFFINE,
                    stack_num_layers=2,
                    stack_hidden_dim=0,
                    stack_activation=ActivationOptions.RELU,
                    stack_residual_flag=False,
                    stack_dropout_probability=0.0,
                )
            case LinearLayerStackOptions.BASE:
                return LinearPresets.base_linear_layer_stack_preset(
                    batch_size=64,
                    input_dim=784,
                    output_dim=10,
                    bias_flag=True,
                    stack_num_layers=2,
                    stack_hidden_dim=0,
                    stack_activation=ActivationOptions.RELU,
                    stack_residual_flag=False,
                    stack_dropout_probability=0.0,
                )
            case LinearLayerStackOptions.ADAPTIVE:
                return LinearPresets.adaptive_stack_preset(
                    batch_size=64,
                    input_dim=784,
                    output_dim=10,
                    bias_flag=True,
                    generator_depth=DynamicDepthOptions.DEPTH_OF_TWO,
                    diagonal_option=DynamicDiagonalOptions.DISABLED,
                    bias_option=DynamicBiasOptions.DYNAMIC_PARAMETERS,
                    memory_option=LinearMemoryOptions.FUSION,
                    memory_size_option=LinearMemorySizeOptions.LARGE,
                    memory_position_option=LinearMemoryPositionOptions.BEFORE_AFFINE,
                    stack_num_layers=2,
                    stack_hidden_dim=0,
                    stack_activation=ActivationOptions.RELU,
                    stack_residual_flag=False,
                    stack_dropout_probability=0.0,
                )
