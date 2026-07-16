from typing import TYPE_CHECKING

from torch import Tensor

from emperor._validation import ValidatorBase

if TYPE_CHECKING:
    from emperor.linears._layer import LinearAbstract


class LinearValidator(ValidatorBase):
    OPTIONAL_FIELDS = {"override_config"}

    @classmethod
    def validate(cls, model: "LinearAbstract") -> None:
        cls.validate_required_fields(model.cfg)
        cls.validate_field_types(model.cfg)
        cls.validate_dimensions(
            input_dim=model.input_dim,
            output_dim=model.output_dim,
        )

    @staticmethod
    def validate_input_tensor(X: Tensor, input_dim: int) -> None:
        if not isinstance(X, Tensor):
            raise TypeError(f"Input must be a Tensor, got {type(X).__name__}")
        if X.dim() < 2:
            raise ValueError(
                f"Input must have shape (..., input_dim), got {X.dim()}D "
                f"tensor with shape {X.shape}"
            )
        if X.shape[-1] != input_dim:
            raise ValueError(
                f"Input final dimension must be {input_dim}, got {X.shape[-1]} "
                f"for tensor with shape {X.shape}"
            )
