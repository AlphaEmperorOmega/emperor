from typing import TYPE_CHECKING

from models.linears._builder_options import (
    DynamicMemoryOptions,
    LayerControllerOptions,
    MainLayerStackOptions,
    RecurrentControllerOptions,
)
from models.linears._controller_stack import (
    SubmoduleStackOptions,
)
from models.linears.linear_adaptive._boundary_model_config_factory import (
    AdaptiveBoundaryModelOptions,
    BoundaryModelConfigDependencies,
    BoundaryModelConfigFactory,
)
from models.linears.linear_adaptive._builder_options import (
    AdaptiveGeneratorStackOptions,
    HiddenAdaptiveBiasOptions,
    HiddenAdaptiveDiagonalOptions,
    HiddenAdaptiveMaskOptions,
    HiddenAdaptiveWeightOptions,
)
from models.linears.linear_adaptive._hidden_model_config_factory import (
    HiddenModelConfigDependencies,
    HiddenModelConfigFactory,
)
from models.linears.linear_adaptive.experiment_config import ExperimentConfig

import models.linears.linear_adaptive.config as config

if TYPE_CHECKING:
    from emperor.config import ModelConfig


class LinearAdaptiveConfigBuilder:
    def __init__(
        self,
        batch_size: int = config.BATCH_SIZE,
        learning_rate: float = config.LEARNING_RATE,
        input_dim: int = config.INPUT_DIM,
        hidden_dim: int = config.HIDDEN_DIM,
        output_dim: int = config.OUTPUT_DIM,
        stack_options: MainLayerStackOptions | None = None,
        submodule_stack_options: SubmoduleStackOptions | None = None,
        layer_controller_options: LayerControllerOptions | None = None,
        dynamic_memory_options: DynamicMemoryOptions | None = None,
        recurrent_controller_options: RecurrentControllerOptions | None = None,
        adaptive_generator_stack_options: AdaptiveGeneratorStackOptions | None = None,
        hidden_adaptive_weight_options: HiddenAdaptiveWeightOptions | None = None,
        hidden_adaptive_bias_options: HiddenAdaptiveBiasOptions | None = None,
        hidden_adaptive_diagonal_options: HiddenAdaptiveDiagonalOptions | None = None,
        hidden_adaptive_mask_options: HiddenAdaptiveMaskOptions | None = None,
        input_boundary_options: AdaptiveBoundaryModelOptions | None = None,
        output_boundary_options: AdaptiveBoundaryModelOptions | None = None,
    ) -> None:
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.stack_options = stack_options
        self.submodule_stack_options = submodule_stack_options
        self.layer_controller_options = layer_controller_options
        self.dynamic_memory_options = dynamic_memory_options
        self.recurrent_controller_options = recurrent_controller_options
        self.adaptive_generator_stack_options = adaptive_generator_stack_options
        self.hidden_adaptive_weight_options = hidden_adaptive_weight_options
        self.hidden_adaptive_bias_options = hidden_adaptive_bias_options
        self.hidden_adaptive_diagonal_options = hidden_adaptive_diagonal_options
        self.hidden_adaptive_mask_options = hidden_adaptive_mask_options
        self.input_boundary_options = input_boundary_options
        self.output_boundary_options = output_boundary_options

    def build(self) -> "ModelConfig":
        from emperor.config import ModelConfig

        return ModelConfig(
            learning_rate=self.learning_rate,
            batch_size=self.batch_size,
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
            experiment_config=ExperimentConfig(
                input_model_config=self.__input_model_config(),
                model_config=self.__model_config(),
                output_model_config=self.__output_model_config(),
            ),
        )

    def __input_model_config(self):
        boundary_model_dependencies = self.__boundary_model_config_dependencies()
        boundary_model_factory = BoundaryModelConfigFactory(boundary_model_dependencies)
        return boundary_model_factory.build_input_model_config()

    def __model_config(self):
        hidden_model_dependencies = self.__hidden_model_config_dependencies()
        hidden_model_factory = HiddenModelConfigFactory(hidden_model_dependencies)
        return hidden_model_factory.build_hidden_model_config()

    def __output_model_config(self):
        boundary_model_dependencies = self.__boundary_model_config_dependencies()
        boundary_model_factory = BoundaryModelConfigFactory(boundary_model_dependencies)
        return boundary_model_factory.build_output_model_config()

    def __boundary_model_config_dependencies(self) -> BoundaryModelConfigDependencies:
        return BoundaryModelConfigDependencies(
            stack_options=self.stack_options,
            input_boundary_options=self.input_boundary_options,
            output_boundary_options=self.output_boundary_options,
            adaptive_generator_stack_options=self.adaptive_generator_stack_options,
        )

    def __hidden_model_config_dependencies(self) -> HiddenModelConfigDependencies:
        return HiddenModelConfigDependencies(
            hidden_dim=self.hidden_dim,
            stack_options=self.stack_options,
            submodule_stack_options=self.submodule_stack_options,
            layer_controller_options=self.layer_controller_options,
            dynamic_memory_options=self.dynamic_memory_options,
            recurrent_controller_options=self.recurrent_controller_options,
            hidden_adaptive_weight_options=self.hidden_adaptive_weight_options,
            hidden_adaptive_bias_options=self.hidden_adaptive_bias_options,
            hidden_adaptive_diagonal_options=self.hidden_adaptive_diagonal_options,
            hidden_adaptive_mask_options=self.hidden_adaptive_mask_options,
            adaptive_generator_stack_options=self.adaptive_generator_stack_options,
            output_dim=self.output_dim,
        )
