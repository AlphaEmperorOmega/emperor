from typing import TYPE_CHECKING

from torch import Tensor

from emperor._validation import ValidatorBase

if TYPE_CHECKING:
    from emperor.patch._base import PatchBase


class PatchValidator(ValidatorBase):
    OPTIONAL_FIELDS = {"override_config", "class_token_flag"}

    @classmethod
    def validate(cls, model: "PatchBase") -> None:
        cls.validate_required_fields(model.cfg)
        cls.validate_field_types(model.cfg)
        dimensions = {
            "embedding_dim": model.embedding_dim,
            "num_input_channels": model.num_input_channels,
            "patch_size": model.patch_size,
        }
        if hasattr(model.cfg, "stride"):
            dimensions["stride"] = model.cfg.stride
        cls.validate_dimensions(
            **dimensions,
        )
        if hasattr(model.cfg, "padding") and model.cfg.padding < 0:
            raise ValueError(
                "padding must be greater than or equal to 0, "
                f"received {model.cfg.padding}"
            )
        cls._validate_dropout_probability(model.dropout_probability)
        cls._validate_class_token_flag(model.cfg.class_token_flag)

    @staticmethod
    def _validate_dropout_probability(value: float) -> None:
        if not 0.0 <= value <= 1.0:
            raise ValueError(
                f"dropout_probability must be in [0.0, 1.0], received {value}"
            )

    @staticmethod
    def _validate_class_token_flag(value: bool | None) -> None:
        if value is not None and not isinstance(value, bool):
            raise TypeError(
                "class_token_flag must be bool or None for PatchConfig, got "
                f"{type(value).__name__}"
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
