from typing import TYPE_CHECKING

from models.linears._builder_options import (
    DynamicMemoryOptions,
    LayerControllerOptions,
    LinearStackOptions,
    RecurrentControllerOptions,
)
from models.linears._controller_stack import (
    ControllerStackOptions,
)
from models.linears.linear_adaptive._boundary_config_factory import (
    AdaptiveBoundaryProjectionOptions,
    BoundaryConfigDependencies,
    BoundaryConfigFactory,
)
from models.linears.linear_adaptive._builder_options import (
    AdaptiveGeneratorStackOptions,
    HiddenAdaptiveBiasOptions,
    HiddenAdaptiveDiagonalOptions,
    HiddenAdaptiveMaskOptions,
    HiddenAdaptiveWeightOptions,
)
from models.linears.linear_adaptive._control_config_factory import (
    ControlConfigDependencies,
    ControlConfigFactory,
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
        output_dim: int = config.OUTPUT_DIM,
        stack_options: LinearStackOptions | None = None,
        submodule_stack_options: ControllerStackOptions | None = None,
        layer_controller_options: LayerControllerOptions | None = None,
        dynamic_memory_options: DynamicMemoryOptions | None = None,
        recurrent_controller_options: RecurrentControllerOptions | None = None,
        adaptive_generator_stack_options: AdaptiveGeneratorStackOptions | None = None,
        hidden_adaptive_weight_options: HiddenAdaptiveWeightOptions | None = None,
        hidden_adaptive_bias_options: HiddenAdaptiveBiasOptions | None = None,
        hidden_adaptive_diagonal_options: HiddenAdaptiveDiagonalOptions | None = None,
        hidden_adaptive_mask_options: HiddenAdaptiveMaskOptions | None = None,
        input_boundary_options: AdaptiveBoundaryProjectionOptions | None = None,
        output_boundary_options: AdaptiveBoundaryProjectionOptions | None = None,
    ) -> None:
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.input_dim = input_dim
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

        control_dependencies = self.__control_config_dependencies()
        control_factory = ControlConfigFactory(control_dependencies)
        boundary_dependencies = self.__boundary_config_dependencies()
        boundary_factory = BoundaryConfigFactory(boundary_dependencies)

        return ModelConfig(
            learning_rate=self.learning_rate,
            batch_size=self.batch_size,
            input_dim=self.input_dim,
            hidden_dim=control_factory.stack_options.hidden_dim,
            output_dim=self.output_dim,
            experiment_config=ExperimentConfig(
                input_model_config=boundary_factory.build_input_model_config(),
                model_config=control_factory.build_hidden_model_config(),
                output_model_config=boundary_factory.build_output_model_config(),
            ),
        )

    def __boundary_config_dependencies(self) -> BoundaryConfigDependencies:
        return BoundaryConfigDependencies(
            stack_options=self.stack_options,
            input_boundary_options=self.input_boundary_options,
            output_boundary_options=self.output_boundary_options,
            adaptive_generator_stack_options=self.adaptive_generator_stack_options,
        )

    def __control_config_dependencies(self) -> ControlConfigDependencies:
        return ControlConfigDependencies(
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
