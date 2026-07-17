from torch import Tensor

from emperor.augmentations.adaptive_parameters._diagonals.base import (
    DynamicDiagonalAbstract,
)
from emperor.augmentations.adaptive_parameters._diagonals.config import (
    AntiDynamicDiagonalConfig,
)


class AntiDynamicDiagonal(DynamicDiagonalAbstract):
    def __init__(
        self,
        cfg: AntiDynamicDiagonalConfig,
        overrides: AntiDynamicDiagonalConfig | None = None,
    ):
        super().__init__(cfg, overrides)
        self.model = self._init_model()

    def forward(self, weight_params: Tensor, logits: Tensor) -> Tensor:
        diagonal_matrices = self._compute_diagonal_matrix(logits)
        anti_diagonal_matrix = diagonal_matrices.flip(dims=[2])
        return weight_params + anti_diagonal_matrix
