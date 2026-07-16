from typing import TYPE_CHECKING

from torch import Tensor

from emperor._validation import ValidatorBase

if TYPE_CHECKING:
    from emperor.convs._layer import Conv2dLayer


class Conv2dLayerValidator(ValidatorBase):
    OPTIONAL_FIELDS = {"override_config"}

    @classmethod
    def validate(cls, model: "Conv2dLayer") -> None:
        cls.validate_required_fields(model.cfg)
        cls.validate_field_types(model.cfg)
        cls.validate_dimensions(input_dim=model.input_dim, output_dim=model.output_dim)
        cls._validate_kernel_parameters(model)

    @staticmethod
    def _validate_kernel_parameters(model: "Conv2dLayer") -> None:
        if model.kernel_size < 1:
            raise ValueError(f"kernel_size must be >= 1, received {model.kernel_size}")
        if model.stride < 1:
            raise ValueError(f"stride must be >= 1, received {model.stride}")
        if model.padding < 0:
            raise ValueError(f"padding must be >= 0, received {model.padding}")

    @staticmethod
    def validate_forward_inputs(X: Tensor) -> None:
        if not isinstance(X, Tensor):
            raise TypeError(
                f"Input Error: forward input must be a Tensor, "
                f"received {type(X).__name__}."
            )
        if X.dim() != 4:
            raise ValueError(
                f"Input Error: Conv2dLayer expects a 4D input tensor "
                f"(batch, channels, height, width), received a "
                f"{X.dim()}D tensor with shape {tuple(X.shape)}."
            )
