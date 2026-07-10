from emperor.base.layer.config import LayerConfig, LayerStackConfig
from emperor.linears.core.config import LinearLayerConfig
from emperor.memory.config import DynamicMemoryConfig

from models.bert.expert_linear.runtime_options import (
    DynamicMemoryOptions,
    MainLayerStackOptions,
    SubmoduleStackOptions,
    resolve_controller_stack_options,
)


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
        model_config = self.__build_controller_stack(
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

    def __build_controller_stack(
        self,
        options: SubmoduleStackOptions,
        *,
        hidden_dim: int | None = None,
        output_dim: int | None = None,
    ) -> LayerStackConfig:
        return LayerStackConfig(
            hidden_dim=options.hidden_dim if hidden_dim is None else hidden_dim,
            output_dim=output_dim,
            num_layers=options.num_layers,
            last_layer_bias_option=options.last_layer_bias_option,
            apply_output_pipeline_flag=options.apply_output_pipeline_flag,
            layer_config=LayerConfig(
                activation=options.activation,
                layer_norm_position=options.layer_norm_position,
                residual_connection_option=options.residual_connection_option,
                dropout_probability=options.dropout_probability,
                halting_config=None,
                gate_config=None,
                memory_config=None,
                layer_model_config=LinearLayerConfig(
                    bias_flag=options.bias_flag,
                ),
            ),
        )
