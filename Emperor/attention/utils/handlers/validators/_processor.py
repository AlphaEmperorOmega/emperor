from torch import Tensor

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from Emperor.attention.utils.handlers.processor import ProcessorBase


class ProcessorValidator:
    def __init__(self, model: "ProcessorBase"):
        self.model = model

    def is_input_tensor_single_batch(self, tensor: Tensor) -> bool:
        return tensor.size(0) == 1
