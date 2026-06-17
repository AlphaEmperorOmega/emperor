from torch import Tensor
from emperor.augmentations.adaptive_parameters.core.diagonal.base import (
    DynamicDiagonalAbstract,
)
from emperor.augmentations.adaptive_parameters.core.diagonal.config import (
    CombinedDynamicDiagonalConfig,
)
from emperor.augmentations.adaptive_parameters.core.diagonal.variants.anti import (
    AntiDynamicDiagonal,
)
from emperor.augmentations.adaptive_parameters.core.diagonal.variants.standard import (
    StandardDynamicDiagonal,
)


class CombinedDynamicDiagonal(DynamicDiagonalAbstract):
    def __init__(
        self,
        cfg: CombinedDynamicDiagonalConfig,
        overrides: CombinedDynamicDiagonalConfig | None = None,
    ):
        super().__init__(cfg, overrides)
        self.diagonal_model = StandardDynamicDiagonal(self.cfg)
        self.anti_diagonal_model = AntiDynamicDiagonal(self.cfg)

    def forward(self, weight_params: Tensor, logits: Tensor) -> Tensor:
        weight_params = self.diagonal_model(weight_params, logits)
        return self.anti_diagonal_model(weight_params, logits)
