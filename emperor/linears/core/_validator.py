from torch import Tensor

from emperor.base.validator import ValidatorBase

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from emperor.linears.core.layers import LinearAbstract


class LinearValidator(ValidatorBase):
    OPTIONAL_FIELDS = {"override_config"}

    @staticmethod
    def validate(model: "LinearAbstract") -> None:
        LinearValidator.validate_required_fields(model.cfg)
        LinearValidator.validate_field_types(model.cfg)
        LinearValidator.validate_dimensions(
            input_dim=model.input_dim, output_dim=model.output_dim
        )

    @staticmethod
    def validate_input_is_2d(X: Tensor) -> None:
        if X.dim() != 2:
            raise ValueError(
                f"Input must be a 2D matrix (batch, input_dim), "
                f"got {X.dim()}D tensor with shape {X.shape}"
            )
