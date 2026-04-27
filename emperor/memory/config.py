from dataclasses import dataclass

from emperor.base.layer import LayerStackConfig
from emperor.base.utils import ConfigBase, optional_field
from emperor.memory.options import DynamicMemoryOptions, MemoryPositionOptions


@dataclass
class DynamicMemoryConfig(ConfigBase):
    input_dim: int | None = optional_field(
        "Input dimensionality of the dynamic memory module."
    )
    output_dim: int | None = optional_field(
        "Output dimensionality of the dynamic memory module."
    )
    model_type: DynamicMemoryOptions | None = optional_field(
        "Memory strategy used to blend a learned memory representation with the layer input or output."
    )
    num_memory_slots: int | None = optional_field(
        "Number of memory slots used by attention-style dynamic memory."
    )
    memory_position_option: MemoryPositionOptions | None = optional_field(
        "Specifies whether memory is applied before or after the affine transformation."
    )
    test_time_training_learning_rate: float | None = optional_field(
        "Learning rate for the test-time training inner loop. When None, TTT is disabled."
    )
    test_time_training_num_inner_steps: int | None = optional_field(
        "Number of gradient steps in the TTT inner loop. When None, TTT is disabled."
    )
    model_config: LayerStackConfig | None = optional_field(
        "Configuration for the internal generator network."
    )

    def _registry_owner(self) -> type:
        from emperor.memory.core.base import DynamicMemoryAbstract

        return DynamicMemoryAbstract
