from typing import TYPE_CHECKING

from torch import Tensor

from emperor._validation import ValidatorBase
from emperor.neuron._validation.common import NeuronValidationMixin

if TYPE_CHECKING:
    from emperor.memory import DynamicMemoryConfig
    from emperor.neuron._axons.core import Axons


class AxonsValidator(ValidatorBase, NeuronValidationMixin):
    OPTIONAL_FIELDS = {"memory_config"}

    @classmethod
    def validate(cls, model: "Axons") -> None:
        cls.validate_config(model.cfg)

    @classmethod
    def validate_config(cls, cfg) -> None:
        cls.validate_required_fields(cfg)
        cls.validate_field_types(cfg)
        cls.validate_memory_config(cfg.memory_config)

    @staticmethod
    def validate_memory_config(memory_config: "DynamicMemoryConfig | None") -> None:
        if memory_config is None:
            return
        from emperor.memory import DynamicMemoryConfig

        if not isinstance(memory_config, DynamicMemoryConfig):
            raise TypeError(
                "memory_config must be an instance of DynamicMemoryConfig for "
                f"AxonsConfig, got {type(memory_config).__name__}."
            )

    @classmethod
    def validate_forward_input(cls, input: Tensor) -> None:
        cls.validate_tensor_rank("Axons input", input, 2)
