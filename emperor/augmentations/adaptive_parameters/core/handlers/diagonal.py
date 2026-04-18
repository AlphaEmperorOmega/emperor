import torch
import torch.nn.functional as F

from typing import cast
from torch import Tensor
from torch.nn import Sequential
from dataclasses import dataclass
from emperor.base.utils import ConfigBase, Module, optional_field
from emperor.base.registry import subclass_registry
from emperor.base.layer import Layer, LayerStackConfig
from emperor.augmentations.adaptive_parameters.options import DynamicDiagonalOptions


@dataclass
class DiagonalHandlerConfig(ConfigBase):
    input_dim: int | None = optional_field("Input dimension of the diagonal transformation.")
    output_dim: int | None = optional_field("Output dimension of the diagonal transformation.")
    model_type: DynamicDiagonalOptions | None = optional_field(
        "Input-dependent adjustment of the weight matrix diagonal."
    )
    model_config: LayerStackConfig | None = optional_field(
        "Layer stack configuration for the internal generator network."
    )

    def build(
        self, overrides: "ConfigBase | None" = None
    ) -> "DiagonalHandlerAbstract":
        if self.model_type is None:
            raise ValueError("`model_type` must be set before building the handler")
        handler_cls = DiagonalHandlerAbstract.resolve(self.model_type)
        return handler_cls(self, cast("DiagonalHandlerConfig | None", overrides))


@subclass_registry
class DiagonalHandlerAbstract(Module):
    def __init__(
        self,
        cfg: DiagonalHandlerConfig,
        overrides: DiagonalHandlerConfig | None = None,
    ):
        super().__init__()
        self.cfg: DiagonalHandlerConfig = self._override_config(cfg, overrides)
        self.input_dim = self.cfg.input_dim
        self.output_dim = self.cfg.output_dim
        self.padding_shape = self.__get_diagonal_padding_shape()

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
        return self._create_stack(self.cfg.model_config, overrides)

    def _create_stack(self, config, overrides) -> "Layer | Sequential":
        from emperor.linears.core.stack import LinearLayerStack

        return LinearLayerStack(config, overrides).build_model()

    def forward(self, weight_params: Tensor) -> Tensor:
        return weight_params

    def __convert_to_diagonal_matrix(
        self,
        vector_matrix: Tensor,
    ) -> Tensor:
        diagonal_matrix = torch.diag_embed(vector_matrix)
        if self.padding_shape is not None:
            diagonal_matrix = F.pad(diagonal_matrix, self.padding_shape)
        return diagonal_matrix

    def _compute_diagonal_matrix(self, logits: Tensor) -> Tensor:
        vectors = self.diagonal_generator(logits)
        return self.__convert_to_diagonal_matrix(vectors)


@DiagonalHandlerAbstract.register(DynamicDiagonalOptions.DIAGONAL)
class DiagonalHandler(DiagonalHandlerAbstract):
    def __init__(
        self,
        cfg: DiagonalHandlerConfig,
        overrides: DiagonalHandlerConfig | None = None,
    ):
        super().__init__(cfg, overrides)
        self.diagonal_generator = self._init_model()

    def forward(self, weight_params: Tensor, logits: Tensor) -> Tensor:
        diagonal_matrices = self._compute_diagonal_matrix(logits)
        return weight_params + diagonal_matrices


@DiagonalHandlerAbstract.register(DynamicDiagonalOptions.ANTI_DIAGONAL)
class AntiDiagonalHandler(DiagonalHandlerAbstract):
    def __init__(
        self,
        cfg: DiagonalHandlerConfig,
        overrides: DiagonalHandlerConfig | None = None,
    ):
        super().__init__(cfg, overrides)
        self.diagonal_generator = self._init_model()

    def forward(self, weight_params: Tensor, logits: Tensor) -> Tensor:
        diagonal_matrices = self._compute_diagonal_matrix(logits)
        anti_diagonal_matrix = diagonal_matrices.flip(dims=[2])
        return weight_params + anti_diagonal_matrix


@DiagonalHandlerAbstract.register(DynamicDiagonalOptions.DIAGONAL_AND_ANTI_DIAGONAL)
class DiagonalAndAntiDiagonalHandler(DiagonalHandlerAbstract):
    def __init__(
        self,
        cfg: DiagonalHandlerConfig,
        overrides: DiagonalHandlerConfig | None = None,
    ):
        super().__init__(cfg, overrides)
        self.diagonal_generator = DiagonalHandler(self.cfg)
        self.anti_diagonal_generator = AntiDiagonalHandler(self.cfg)

    def forward(self, weight_params: Tensor, logits: Tensor) -> Tensor:
        weight_params = self.diagonal_generator(weight_params, logits)
        return self.anti_diagonal_generator(weight_params, logits)
