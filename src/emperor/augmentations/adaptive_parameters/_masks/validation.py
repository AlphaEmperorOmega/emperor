from typing import TYPE_CHECKING

from emperor._validation import ValidatorBase
from emperor.augmentations.adaptive_parameters._validation import (
    AdaptiveGeneratorValidatorBase,
)

if TYPE_CHECKING:
    from emperor.augmentations.adaptive_parameters._masks.base import AxisMaskAbstract


class AxisMaskValidator(AdaptiveGeneratorValidatorBase, ValidatorBase):
    OPTIONAL_FIELDS = {"mask_transition_width"}

    @classmethod
    def validate(cls, model: "AxisMaskAbstract") -> None:
        cls.validate_required_fields(model.cfg)
        cls.validate_field_types(model.cfg)
        cls.validate_dimensions(
            input_dim=model.cfg.input_dim,
            output_dim=model.cfg.output_dim,
        )
        cls._validate_mask_threshold(model.cfg.mask_threshold)
        cls._validate_mask_surrogate_scale(model.cfg.mask_surrogate_scale)
        cls._validate_mask_floor(model.cfg.mask_floor)
        mask_transition_width = getattr(model.cfg, "mask_transition_width", None)
        if mask_transition_width is not None:
            cls._validate_mask_transition_width(mask_transition_width)

    @staticmethod
    def _validate_mask_threshold(mask_threshold: float) -> None:
        if not 0.0 <= mask_threshold <= 1.0:
            raise ValueError(
                "mask_threshold must be between 0.0 and 1.0 inclusive, "
                f"received {mask_threshold!r}."
            )

    @staticmethod
    def _validate_mask_surrogate_scale(mask_surrogate_scale: float) -> None:
        if mask_surrogate_scale < 0.0:
            raise ValueError(
                "mask_surrogate_scale must be greater than or equal to 0.0, "
                f"received {mask_surrogate_scale!r}."
            )

    @staticmethod
    def _validate_mask_floor(mask_floor: float) -> None:
        if not 0.0 <= mask_floor < 1.0:
            raise ValueError(
                "mask_floor must be between 0.0 inclusive and 1.0 exclusive, "
                f"received {mask_floor!r}."
            )

    @staticmethod
    def _validate_mask_transition_width(mask_transition_width: float) -> None:
        if mask_transition_width <= 0.0:
            raise ValueError(
                "mask_transition_width must be greater than 0.0, "
                f"received {mask_transition_width!r}."
            )
