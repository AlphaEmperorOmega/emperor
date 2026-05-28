from torch import Tensor

from emperor.base.validator import ValidatorBase

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from emperor.patch.core.layers import PatchBase


class PatchValidator(ValidatorBase):
    OPTIONAL_FIELDS = {"override_config"}

    @staticmethod
    def validate(model: "PatchBase") -> None:
        PatchValidator.validate_required_fields(model.cfg)
        PatchValidator.validate_field_types(model.cfg)
        PatchValidator.validate_dimensions(
            embedding_dim=model.embedding_dim,
            num_input_channels=model.num_input_channels,
            patch_size=model.patch_size,
        )
        PatchValidator.validate_dropout_probability(model.dropout_probability)

    @staticmethod
    def validate_dropout_probability(value: float) -> None:
        if not 0.0 <= value <= 1.0:
            raise ValueError(
                f"dropout_probability must be in [0.0, 1.0], received {value}"
            )

    @staticmethod
    def validate_forward_inputs(model: "PatchBase", X: Tensor) -> None:
        if not isinstance(X, Tensor):
            raise TypeError(
                f"Input Error: forward input must be a Tensor, "
                f"received {type(X).__name__}."
            )
        if X.dim() != 4:
            raise ValueError(
                f"Input Error: PatchBase expects a 4D input tensor "
                f"(batch, channels, height, width), received a "
                f"{X.dim()}D tensor with shape {tuple(X.shape)}."
            )
        if X.shape[1] != model.num_input_channels:
            raise ValueError(
                f"Input Error: input channel dimension must match "
                f"'num_input_channels', received "
                f"num_input_channels={model.num_input_channels} and input shape "
                f"{tuple(X.shape)}."
            )
