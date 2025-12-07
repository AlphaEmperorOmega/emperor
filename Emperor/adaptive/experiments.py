from Emperor.experiments.utils.factories import Experiments
from Emperor.adaptive.options import AdaptiveLayerOptions
from Emperor.adaptive.utils.config import ParameterGeneratorConfigs
from Emperor.linears.options import LinearLayerOptions
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

    def train_model(self, layer_type: ParameterGeneratorOptions):
        preset = ParameterGeneratorPresets(layer_type).get_config()
        self._set_model_config(preset)
        self._train_model(layer_type)

    def train_vector_model(self):
        self.train_model(ParameterGeneratorOptions.VECTOR)

    def train_matrix_model(self):
        self.train_model(ParameterGeneratorOptions.MATRIX)

    def train_dynamic_model(self):
        self.train_model(ParameterGeneratorOptions.GENERATOR)

    def test_all_linear_types(self):
        for layer_type in LinearLayerOptions:
            self.train_model(layer_type)


class ParameterGeneratorPresets:
    def __init__(
        self,
        parameter_generator_options: ParameterGeneratorOptions,
    ) -> None:
        self.parameter_generator_options = parameter_generator_options

    def get_config(self) -> "ModelConfig":
        match self.parameter_generator_options:
            case ParameterGeneratorOptions.VECTOR:
                return ParameterGeneratorConfigs.base_preset(
                    batch_size=64,
                    input_dim=784,
                    output_dim=10,
                    bias_flag=True,
                )
            case ParameterGeneratorOptions.MATRIX:
                return ParameterGeneratorConfigs.dynamic_preset(
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
                    stack_depth=2,
                )
            case ParameterGeneratorOptions.GENERATOR:
                return ParameterGeneratorConfigs.dynamic_preset(
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
                    stack_depth=2,
                )
