from typing import TYPE_CHECKING

from emperor._validation import ValidatorBase
from emperor.augmentations.adaptive_parameters._validation import (
    AdaptiveGeneratorValidatorBase,
)

if TYPE_CHECKING:
    from emperor.augmentations.adaptive_parameters._diagonals.base import (
        DynamicDiagonalAbstract,
    )


class DynamicDiagonalValidator(AdaptiveGeneratorValidatorBase, ValidatorBase):
    @classmethod
    def validate(cls, model: "DynamicDiagonalAbstract") -> None:
        cls.validate_initialization_fields(model)
        cls.validate_model_config(model.cfg)

    @classmethod
    def validate_initialization_fields(cls, model: "DynamicDiagonalAbstract") -> None:
        cls.validate_required_fields(model.cfg)
        cls.validate_field_types(model.cfg)
        cls.validate_dimensions(
            input_dim=model.cfg.input_dim,
            output_dim=model.cfg.output_dim,
        )
