from typing import TYPE_CHECKING

from torch.types import Tensor

from emperor._validation import ValidatorBase
from emperor.augmentations.adaptive_parameters._validation import (
    AdaptiveGeneratorValidatorBase,
)

if TYPE_CHECKING:
    from emperor.augmentations.adaptive_parameters._biases.base import (
        DynamicBiasAbstract,
    )


class DynamicBiasValidator(AdaptiveGeneratorValidatorBase, ValidatorBase):
    OPTIONAL_FIELDS = {"bank_expansion_factor"}

    @classmethod
    def validate(cls, model: "DynamicBiasAbstract") -> None:
        cls.validate_initialization_fields(model)
        cls.validate_variant_config(model)
        cls.validate_model_config(model.cfg)

    @classmethod
    def validate_initialization_fields(cls, model: "DynamicBiasAbstract") -> None:
        cls.validate_required_fields(model.cfg)
        cls.validate_field_types(model.cfg)
        cls.validate_dimensions(
            input_dim=model.cfg.input_dim,
            output_dim=model.cfg.output_dim,
        )
        cls.validate_decay_parameters(model.cfg)

    @classmethod
    def validate_variant_config(cls, model: "DynamicBiasAbstract") -> None:
        from emperor.augmentations.adaptive_parameters._biases.variants.weighted_bank import (
            WeightedBankDynamicBias,
        )

        if isinstance(model, WeightedBankDynamicBias):
            cls.validate_bank_expansion_factor(model)

    @staticmethod
    def validate_bank_expansion_factor(model: "DynamicBiasAbstract") -> None:
        from emperor.augmentations.adaptive_parameters._options import (
            BankExpansionFactorOptions,
        )

        factor = model.cfg.bank_expansion_factor
        if factor is None or not isinstance(factor, BankExpansionFactorOptions):
            raise ValueError(
                f"{type(model).__name__} requires bank_expansion_factor to be a "
                f"BankExpansionFactorOptions value, received {factor!r}."
            )
        if factor == BankExpansionFactorOptions.DISABLED:
            raise ValueError(
                f"{type(model).__name__} requires bank_expansion_factor > 0, "
                f"received {factor}. "
                f"Use FACTOR_OF_ONE, FACTOR_OF_TWO, FACTOR_OF_THREE, or "
                f"FACTOR_OF_FOUR."
            )

    @staticmethod
    def ensure_parameters_exist(bias_params: Tensor | None) -> None:
        if bias_params is None:
            raise ValueError(
                "bias_params must not be None. Provide a valid bias tensor for "
                "this dynamic bias strategy."
            )
