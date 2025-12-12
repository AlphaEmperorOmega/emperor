from Emperor.experiments.utils.factories import Experiments
from Emperor.linears.options import LinearLayerOptions
from Emperor.linears.utils.config import LinearsConfigs
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

    def train_model(self, layer_type: LinearLayerOptions):
        preset = LinearsBasePreset(layer_type).get_config()
        self._set_model_config(preset)
        self._train_model(layer_type)

    def train_base_model(self):
        self.train_model(LinearLayerOptions.BASE)

    def train_dynamic_model(self):
        self.train_model(LinearLayerOptions.ADAPTIVE)

    def test_all_linear_types(self):
        for layer_type in LinearLayerOptions:
            self.train_model(layer_type)


class LinearsBasePreset:
    def __init__(
        self,
        linear_layer_options: LinearLayerOptions,
    ) -> None:
        self.linear_layer_options = linear_layer_options

    def get_config(self) -> "ModelConfig":
        match self.linear_layer_options:
            case LinearLayerOptions.BASE:
                return LinearsConfigs.base_preset(
                    batch_size=64,
                    input_dim=784,
                    output_dim=10,
                    bias_flag=True,
                )
            case LinearLayerOptions.ADAPTIVE:
                return LinearsConfigs.dynamic_preset(
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
                )
