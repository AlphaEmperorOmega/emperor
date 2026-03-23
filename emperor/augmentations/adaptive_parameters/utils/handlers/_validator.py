from torch.types import Tensor

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from emperor.augmentations.adaptive_parameters.utils.handlers.bias import BiasHandlerAbstract


class BiasHandlerAbstractValidator:
    def __init__(self, model: "BiasHandlerAbstract"):
        self.model = model

    def ensure_parameters_exist(self, bias_params: Tensor | None) -> None:
        if bias_params is None:
            raise ValueError(
                "The 'bias_params' argument cannot be None. Please provide valid parameters to proceed."
            )
