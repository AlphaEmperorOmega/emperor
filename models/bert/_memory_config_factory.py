from emperor.memory.config import DynamicMemoryConfig

from models.experts._controller_stack import build_linear_controller_stack
from models.experts._builder_options import (
    ExpertsDynamicMemoryOptions,
    ExpertsStackOptions,
    ExpertsSubmoduleStackOptions,
    resolve_experts_controller_stack_options,
)


class BertMemoryConfigFactory:
    def __init__(
        self,
        *,
        stack_options: ExpertsStackOptions | ExpertsSubmoduleStackOptions,
        dynamic_memory_options: ExpertsDynamicMemoryOptions,
        submodule_stack_options: ExpertsSubmoduleStackOptions,
    ) -> None:
        self.stack_options = stack_options
        self.dynamic_memory_options = dynamic_memory_options
        self.submodule_stack_options = submodule_stack_options

    def build_memory_config(self) -> DynamicMemoryConfig | None:
        if not self.dynamic_memory_options.memory_flag:
            return None
        memory_stack_options = resolve_experts_controller_stack_options(
            self.dynamic_memory_options.memory_stack_source,
            self.submodule_stack_options,
        )
        model_config = build_linear_controller_stack(memory_stack_options)
        return self.dynamic_memory_options.memory_option(
            input_dim=self.stack_options.hidden_dim,
            output_dim=self.stack_options.hidden_dim,
            memory_position_option=self.dynamic_memory_options.memory_position_option,
            test_time_training_learning_rate=(
                self.dynamic_memory_options.memory_test_time_training_learning_rate
            ),
            test_time_training_num_inner_steps=(
                self.dynamic_memory_options.memory_test_time_training_num_inner_steps
            ),
            model_config=model_config,
        )
