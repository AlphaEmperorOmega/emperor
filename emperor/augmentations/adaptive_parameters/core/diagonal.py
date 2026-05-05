import torch
import torch.nn.functional as F

from torch import Tensor
from torch.nn import Sequential
from dataclasses import dataclass
from emperor.base.utils import ConfigBase, Module, optional_field
from emperor.base.layer import Layer, LayerStackConfig
from emperor.augmentations.adaptive_parameters.core._validator import (
    DynamicDiagonalValidator,
)


@dataclass
class DynamicDiagonalConfig(ConfigBase):
    input_dim: int | None = optional_field(
        "Input feature dimension."
    )
    output_dim: int | None = optional_field(
        "Output feature dimension."
    )
    model_config: LayerStackConfig | None = optional_field(
        "Internal generator network config."
    )

    def _registry_owner(self) -> type:
        raise ValueError(
            f"DynamicDiagonalConfig is abstract and has no registered "
            f"DynamicDiagonal class; instantiate a concrete leaf config instead."
        )


@dataclass
class StandardDynamicDiagonalConfig(DynamicDiagonalConfig):
    def _registry_owner(self) -> type:
        return StandardDynamicDiagonal


@dataclass
class AntiDynamicDiagonalConfig(DynamicDiagonalConfig):
    def _registry_owner(self) -> type:
        return AntiDynamicDiagonal


@dataclass
class CombinedDynamicDiagonalConfig(DynamicDiagonalConfig):
    def _registry_owner(self) -> type:
        return CombinedDynamicDiagonal


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
    ) -> "Layer | Sequential":
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
        vectors = Layer.forward_with_state(self.model, logits)
        return self.__convert_to_diagonal_matrix(vectors)


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
