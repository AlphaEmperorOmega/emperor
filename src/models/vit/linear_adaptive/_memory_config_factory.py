from emperor.memory import DynamicMemoryConfig
from models.vit.linear_adaptive.runtime_options import (
    DynamicMemoryOptions,
    MainLayerStackOptions,
    SubmoduleStackOptions,
    resolve_controller_stack_options,
)

from ._controller_stack_config import build_controller_stack_config


class MemoryConfigFactory:
    def __init__(
        self,
        *,
        hidden_dim: int,
        stack_options: MainLayerStackOptions,
        dynamic_memory_options: DynamicMemoryOptions,
        submodule_stack_options: SubmoduleStackOptions,
    ) -> None:
        self.hidden_dim = hidden_dim
        self.stack_options = stack_options
        self.dynamic_memory_options = dynamic_memory_options
        self.submodule_stack_options = submodule_stack_options

    def build_memory_config(self) -> DynamicMemoryConfig | None:
        if not self.dynamic_memory_options.memory_flag:
            return None
        memory_stack_source = self.dynamic_memory_options.memory_stack_source
        submodule_stack_defaults = self.submodule_stack_options
        resolved_memory_stack_options = resolve_controller_stack_options(
            memory_stack_source,
            submodule_stack_defaults,
        )
        model_config = build_controller_stack_config(
            resolved_memory_stack_options,
        )
        return self.dynamic_memory_options.memory_option(
            input_dim=self.hidden_dim,
            output_dim=self.hidden_dim,
            memory_position_option=self.dynamic_memory_options.memory_position_option,
            test_time_training_learning_rate=(
                self.dynamic_memory_options.memory_test_time_training_learning_rate
            ),
            test_time_training_num_inner_steps=(
                self.dynamic_memory_options.memory_test_time_training_num_inner_steps
            ),
            model_config=model_config,
        )
