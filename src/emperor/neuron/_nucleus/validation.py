from typing import TYPE_CHECKING

from torch import Tensor

from emperor._validation import ValidatorBase
from emperor.neuron._validation.common import NeuronValidationMixin

if TYPE_CHECKING:
    from emperor.neuron._nucleus.core import Nucleus


class NucleusValidator(ValidatorBase, NeuronValidationMixin):
    @classmethod
    def validate(cls, model: "Nucleus") -> None:
        cls.validate_required_fields(model.cfg)
        cls.validate_field_types(model.cfg)

    @classmethod
    def validate_forward_input(cls, input: Tensor) -> None:
        cls.validate_tensor_rank("Nucleus input", input, 2)
