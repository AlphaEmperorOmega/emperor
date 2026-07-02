import models.linears.linear.config as config

from models.linears._controller_stack import ControllerStackOptions
from models.linears.linear._boundary_config_factory import (
    BoundaryConfigDependencies,
    BoundaryConfigFactory,
)
from models.linears.linear._control_config_factory import (
    ControlConfigDependencies,
    ControlConfigFactory,
)
from models.linears.linear.experiment_config import ExperimentConfig
from models.linears._builder_options import (
    DynamicMemoryOptions,
    LayerControllerOptions,
    LinearStackOptions,
    RecurrentControllerOptions,
)
from models.linears._builder_adapter import (
    default_dynamic_memory_options,
    default_layer_controller_options,
    default_linear_stack_options,
    default_recurrent_controller_options,
    default_submodule_stack_options,
)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from emperor.config import ModelConfig


class LinearConfigBuilder:
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
    ) -> None:

        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.stack_options = self.__default_stack_options(stack_options)
        self.submodule_stack_options = self.__default_submodule_stack_options(
            submodule_stack_options
        )
        self.layer_controller_options = self.__default_layer_controller_options(
            layer_controller_options
        )
        self.dynamic_memory_options = self.__default_dynamic_memory_options(
            dynamic_memory_options
        )
        self.recurrent_controller_options = self.__default_recurrent_controller_options(
            recurrent_controller_options
        )

    def build(self) -> "ModelConfig":
        from emperor.config import ModelConfig

        boundary_dependencies = self.__boundary_config_dependencies()
        control_dependencies = self.__control_config_dependencies()
        boundary_factory = BoundaryConfigFactory(boundary_dependencies)
        control_factory = ControlConfigFactory(control_dependencies)

        return ModelConfig(
            learning_rate=self.learning_rate,
            batch_size=self.batch_size,
            input_dim=self.input_dim,
            hidden_dim=self.stack_options.hidden_dim,
            output_dim=self.output_dim,
            experiment_config=ExperimentConfig(
                input_model_config=boundary_factory.build_input_model_config(),
                model_config=control_factory.build(),
                output_model_config=boundary_factory.build_output_model_config(),
            ),
        )

    def __boundary_config_dependencies(self) -> BoundaryConfigDependencies:
        return BoundaryConfigDependencies(
            stack_options=self.stack_options,
        )

    def __control_config_dependencies(self) -> ControlConfigDependencies:
        return ControlConfigDependencies(
            stack_options=self.stack_options,
            submodule_stack_options=self.submodule_stack_options,
            layer_controller_options=self.layer_controller_options,
            dynamic_memory_options=self.dynamic_memory_options,
            recurrent_controller_options=self.recurrent_controller_options,
            output_dim=self.output_dim,
        )

    @staticmethod
    def __default_stack_options(
        stack_options: LinearStackOptions | None,
    ) -> LinearStackOptions:
        return stack_options or default_linear_stack_options(config)

    @staticmethod
    def __default_submodule_stack_options(
        submodule_stack_options: ControllerStackOptions | None,
    ) -> ControllerStackOptions:
        return submodule_stack_options or default_submodule_stack_options(config)

    @staticmethod
    def __default_layer_controller_options(
        layer_controller_options: LayerControllerOptions | None,
    ) -> LayerControllerOptions:
        return layer_controller_options or default_layer_controller_options(config)

    @staticmethod
    def __default_dynamic_memory_options(
        dynamic_memory_options: DynamicMemoryOptions | None,
    ) -> DynamicMemoryOptions:
        return dynamic_memory_options or default_dynamic_memory_options(config)

    @staticmethod
    def __default_recurrent_controller_options(
        recurrent_controller_options: RecurrentControllerOptions | None,
    ) -> RecurrentControllerOptions:
        return recurrent_controller_options or default_recurrent_controller_options(
            config
        )
