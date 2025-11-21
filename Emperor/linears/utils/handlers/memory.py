from torch import Tensor
from Emperor.base.utils import Module
from Emperor.base.layer import (
    LayerStackConfig,
    LinearLayerStack,
)


class DiagonalHandlerAbstract(Module):
    def __init__(
        self,
        cfg: "ModelConfig",
    ):
        super().__init__()
        self.cfg_main = cfg
        self.cfg = getattr(cfg, "linear_layer_config", cfg)
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

    def _init_model(
        self,
    ) -> LinearLayerStack:
        output_dim = min(self.input_dim, self.output_dim)
        overrides = LayerStackConfig(output_dim=output_dim)
        return LinearLayerStack(self.cfg_main, overrides)

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


class DefaultDiagonalHandler(DiagonalHandlerAbstract):
    def __init__(
        self,
        cfg: "ModelConfig",
    ):
        super().__init__(cfg)

    def forward(self, weight_params: Tensor, logits: Tensor) -> Tensor:
        return weight_params
