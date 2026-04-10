from torch import Tensor

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from emperor.linears.core.layers import LinearBase


class LinearBaseValidator:
    @staticmethod
    def validate(model: "LinearBase") -> None:
        LinearBaseValidator.__validate_required_fields(model)
        LinearBaseValidator.__validate_dimensions(model.input_dim, model.output_dim)
        LinearBaseValidator.__validate_bias_flag(model.bias_flag)

    @staticmethod
    def __validate_required_fields(model: "LinearBase") -> None:
        name = model.__class__.__name__
        if model.input_dim is None:
            raise ValueError(f"input_dim is required for {name}")
        if model.output_dim is None:
            raise ValueError(f"output_dim is required for {name}")
        if model.bias_flag is None:
            raise ValueError(f"bias_flag is required for {name}")

    @staticmethod
    def __validate_dimensions(input_dim: int, output_dim: int) -> None:
        if input_dim <= 0:
            raise ValueError(f"input_dim must be greater than 0, received {input_dim}")
        if output_dim <= 0:
            raise ValueError(
                f"output_dim must be greater than 0, received {output_dim}"
            )

    @staticmethod
    def __validate_bias_flag(bias_flag: bool) -> None:
        if not isinstance(bias_flag, bool):
            raise TypeError(f"bias_flag must be a bool, got {type(bias_flag).__name__}")

    @staticmethod
    def validate_input_shape(X: Tensor) -> None:
        if X.dim() != 2:
            raise ValueError(
                f"Input must be a 2D matrix (batch, input_dim), "
                f"got {X.dim()}D tensor with shape {X.shape}"
            )
