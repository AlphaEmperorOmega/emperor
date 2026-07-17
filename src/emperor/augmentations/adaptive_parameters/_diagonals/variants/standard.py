from torch import Tensor

from emperor.augmentations.adaptive_parameters._diagonals.base import (
    DynamicDiagonalAbstract,
)
from emperor.augmentations.adaptive_parameters._diagonals.config import (
    StandardDynamicDiagonalConfig,
)


class StandardDynamicDiagonal(DynamicDiagonalAbstract):
    def __init__(
        self,
        cfg: StandardDynamicDiagonalConfig,
        overrides: StandardDynamicDiagonalConfig | None = None,
    ):
        super().__init__(cfg, overrides)
        self.model = self._init_model()

    def forward(self, weight_params: Tensor, logits: Tensor) -> Tensor:
        diagonal_matrices = self._compute_diagonal_matrix(logits)
        return weight_params + diagonal_matrices
