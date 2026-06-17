import torch
import torch.nn.functional as F

from torch import Tensor
from emperor.base.layer import Layer, LayerStack, LayerStackConfig
from emperor.base.utils import Module
from emperor.augmentations.adaptive_parameters.core._validator import (
    DynamicDiagonalValidator,
)
from emperor.augmentations.adaptive_parameters.core.diagonal.config import (
    DynamicDiagonalConfig,
)


class DynamicDiagonalAbstract(Module):
    def __init__(
        self,
        cfg: DynamicDiagonalConfig,
        overrides: DynamicDiagonalConfig | None = None,
    ):
        super().__init__()
        self.cfg: DynamicDiagonalConfig = self._override_config(cfg, overrides)
        DynamicDiagonalValidator.validate(self)
        self.input_dim = self.cfg.input_dim
        self.output_dim = self.cfg.output_dim
        self.padding_shape = self.__get_diagonal_padding_shape()
        self.model_config = self.cfg.model_config

    def __get_diagonal_padding_shape(self) -> tuple | None:
        diagonal_padding_shape = None
        if self.input_dim != self.output_dim:
            padding_size = abs(self.input_dim - self.output_dim)
            diagonal_padding_shape = (0, padding_size, 0, 0)
            if self.input_dim > self.output_dim:
                diagonal_padding_shape = (0, 0, 0, padding_size)
        return diagonal_padding_shape

    def _init_model(
        self,
    ) -> "Layer | LayerStack":
        output_dim = min(self.input_dim, self.output_dim)
        overrides = LayerStackConfig(input_dim=self.input_dim, output_dim=output_dim)
        generator_model = self.model_config.build(overrides)
        DynamicDiagonalValidator.validate_generator_model(generator_model)
        return generator_model

    def forward(self, weight_params: Tensor, logits: Tensor) -> Tensor:
        raise NotImplementedError(f"{type(self).__name__} must implement forward().")

    def __convert_to_diagonal_matrix(
        self,
        vector_matrix: Tensor,
    ) -> Tensor:
        diagonal_matrix = torch.diag_embed(vector_matrix)
        if self.padding_shape is not None:
            diagonal_matrix = F.pad(diagonal_matrix, self.padding_shape)
        return diagonal_matrix

    def _compute_diagonal_matrix(self, logits: Tensor) -> Tensor:
        vectors = Layer.run_model_returning_hidden(self.model, logits)
        return self.__convert_to_diagonal_matrix(vectors)
