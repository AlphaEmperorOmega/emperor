from typing import TYPE_CHECKING

from torch import Tensor

from emperor.base.validator import ValidatorBase

if TYPE_CHECKING:
    from emperor.linears.core.layers import LinearAbstract


class LinearValidator(ValidatorBase):
    OPTIONAL_FIELDS = {"override_config"}

    @classmethod
    def validate(cls, model: "LinearAbstract") -> None:
        cls.validate_required_fields(model.cfg)
        cls.validate_field_types(model.cfg)
        cls.validate_dimensions(
            input_dim=model.input_dim, output_dim=model.output_dim
        )
        cls._validate_adaptive_bias_consistency(model)

    @staticmethod
    def _validate_adaptive_bias_consistency(model: "LinearAbstract") -> None:
        adaptive_augmentation_config = getattr(
            model.cfg, "adaptive_augmentation_config", None
        )
        if adaptive_augmentation_config is None:
            return
        if model.bias_flag:
            return
        if adaptive_augmentation_config.bias_config is not None:
            raise ValueError(
                "bias_flag is False but adaptive_augmentation_config.bias_config "
                "is set; cannot apply a dynamic bias to a layer without bias."
            )

    @staticmethod
    def validate_input_is_2d(X: Tensor) -> None:
        if X.dim() != 2:
            raise ValueError(
                f"Input must be a 2D matrix (batch, input_dim), "
                f"got {X.dim()}D tensor with shape {X.shape}"
            )

    @staticmethod
    def validate_input_tensor(X: Tensor, input_dim: int) -> None:
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
