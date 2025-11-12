from torch import Tensor
from Emperor.base.utils import Module
from Emperor.base.layer import (
    LayerStackConfig,
    LinearLayerStack,
)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from Emperor.config import ModelConfig


class DefaultDiagonalHandler(Module):
    def __init__(
        self,
        cfg: "ModelConfig",
    ):
        super().__init__()
        self.cfg = cfg
        self.input_dim = cfg.input_dim
        self.output_dim = cfg.output_dim
        self.padding_shape = self.__get_diagonal_padding_shape()

    def __get_diagonal_padding_shape(self) -> tuple | None:
        diagonal_padding_shape = None
        if self.input_dim != self.output_dim:
            padding_size = abs(self.input_dim - self.output_dim)
            diagonal_padding_shape = (0, padding_size, 0, 0)
            if self.input_dim > self.output_dim:
                diagonal_padding_shape = (0, 0, 0, padding_size)
        return diagonal_padding_shape

    def _init_diagonal_model(
        self,
    ) -> LinearLayerStack:
        output_dim = min(self.input_dim, self.output_dim)
        overrides = LayerStackConfig(output_dim=output_dim)
        return LinearLayerStack(self.cfg, overrides)

    def forward(self, weight_params: Tensor) -> Tensor:
        return weight_params

    def _convert_to_diagonal_matrix(
        self,
        vector_matrix: Tensor,
    ) -> Tensor:
        diagonal_matrix = torch.diag_embed(vector_matrix)
        if self.padding_shape is not None:
            diagonal_matrix = F.pad(diagonal_matrix, self.padding_shape)
        return diagonal_matrix


class DiagonalHandler(DefaultDiagonalHandler):
    def __init__(
        self,
        cfg: "ModelConfig",
    ):
        super().__init__(cfg)
        self.input_dim = cfg.input_dim
        self.output_dim = cfg.output_dim
        self.diagonal_generator = self._init_diagonal_model()

    def forward(self, logits: Tensor, weight_params: Tensor) -> Tensor:
        diagonal_matrices = self._compute_diagonal_matrices(logits)
        return weight_params + diagonal_matrices

    def _compute_diagonal_matrix(self, logits: Tensor) -> Tensor:
        vectors = self.diagonal_generator(logits)
        return self._convert_to_diagonal_matrix(vectors)


class AntiDiagonalHandler(DiagonalHandler):
    def __init__(
        self,
        cfg: "ModelConfig",
    ):
        super().__init__(cfg)
        self.cfg = cfg
        self.input_dim = cfg.input_dim
        self.output_dim = cfg.output_dim
        self.diagonal_generator = self._init_diagonal_model()

    def forward(self, logits: Tensor, weight_params: Tensor) -> Tensor:
        diagonal_matrices = self._compute_diagonal_matrices(logits)
        anti_diagonal_matrix = diagonal_matrices.flip(dims=[2])
        return weight_params + anti_diagonal_matrix


class DiagonalAndAntiDiagonalHandler(DefaultDiagonalHandler):
    def __init__(
        self,
        cfg: "ModelConfig",
    ):
        super().__init__(cfg)
        self.diagonal_generator = DiagonalHandler(cfg)
        self.anti_diagonal_generator = AntiDiagonalHandler(cfg)

    def forward(self, logits: Tensor, weight_params: Tensor) -> Tensor:
        weight_params = self.diagonal_generator(logits, weight_params)
        return self.anti_diagonal_generator(logits, weight_params)
