from torch import Tensor

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from Emperor.attention.utils.handlers.projector import ProjectorBase


class ProcessorValidator:
    def __init__(self, model: "ProjectorBase"):
        self.model = model

    def get_batched_input_flag(self, tensor: Tensor) -> bool:
        return tensor.size(0) > 1
