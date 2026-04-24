from torch.types import Tensor

from emperor.base.validator import ValidatorBase

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from emperor.augmentations.adaptive_parameters.core.diagonal import (
        DynamicDiagonalAbstract,
    )
    from emperor.augmentations.adaptive_parameters.core.mask import AxisMaskAbstract
    from emperor.augmentations.adaptive_parameters.core.bias import DynamicBiasAbstract
    from emperor.augmentations.adaptive_parameters.core.depth_mapper import (
        DepthMappingLayer,
        DepthMappingLayerConfig,
    )
    from emperor.augmentations.adaptive_parameters.core.weight import (
        DynamicWeightAbstract,
    )
    from emperor.base.layer import LayerStackConfig


class AdaptiveGeneratorValidatorBase:
    @staticmethod
    def validate_generator_model(generator_model) -> None:
        from torch.nn import Sequential
        from emperor.base.layer import Layer

        if isinstance(generator_model, Layer):
            AdaptiveGeneratorValidatorBase.validate_generator_layer(generator_model)
            return
        if isinstance(generator_model, Sequential):
            AdaptiveGeneratorValidatorBase.validate_generator_sequence(
                generator_model
            )
            return
        raise TypeError(
            "Expected model_config.build(...) to return a Layer or Sequential, "
            f"received {type(generator_model).__name__}."
        )

    @staticmethod
    def validate_generator_sequence(generator_sequence) -> None:
        from emperor.base.layer import Layer

        for generator_layer in generator_sequence:
            if not isinstance(generator_layer, Layer):
                raise TypeError(
                    "Expected each generator sequence item to be a Layer, "
                    f"received {type(generator_layer).__name__}."
                )
            AdaptiveGeneratorValidatorBase.validate_generator_layer(generator_layer)

    @staticmethod
    def validate_generator_layer(generator_layer) -> None:
        from emperor.linears.core.layers import LinearLayer

        if not isinstance(generator_layer.model, LinearLayer):
            raise TypeError(
                "Expected each generator Layer to wrap a LinearLayer, "
                f"received {type(generator_layer.model).__name__}."
            )


class DynamicWeightValidator(ValidatorBase):
    OPTIONAL_FIELDS = {"bank_expansion_factor"}

    BANK_WEIGHT_OPTIONS = None

    @classmethod
    def _get_bank_weight_options(cls) -> set:
        if cls.BANK_WEIGHT_OPTIONS is None:
            from emperor.augmentations.adaptive_parameters.options import (
                DynamicWeightOptions,
            )

            cls.BANK_WEIGHT_OPTIONS = {
                DynamicWeightOptions.LAYERED_WEIGHTED_BANK,
                DynamicWeightOptions.SOFT_WEIGHTED_BANK,
            }
        return cls.BANK_WEIGHT_OPTIONS

    @staticmethod
    def validate(model: "DynamicWeightAbstract") -> None:
        DynamicWeightValidator.validate_required_fields(model.cfg)
        DynamicWeightValidator.validate_field_types(model.cfg)
        DynamicWeightValidator.validate_dimensions(
            input_dim=model.cfg.input_dim, output_dim=model.cfg.output_dim
        )
        DynamicWeightValidator.validate_no_bank_expansion_factor(model)

    @staticmethod
    def validate_no_bank_expansion_factor(model: "DynamicWeightAbstract") -> None:
        from emperor.augmentations.adaptive_parameters.core.weight import (
            LayeredWeightedBankDynamicWeight,
            SoftWeightedBankDynamicWeight,
        )
        from emperor.augmentations.adaptive_parameters.options import (
            BankExpansionFactorOptions,
        )

        if isinstance(
            model, (LayeredWeightedBankDynamicWeight, SoftWeightedBankDynamicWeight)
        ):
            return
        if model.cfg.bank_expansion_factor is not None:
            raise ValueError(
                f"{type(model).__name__} does not support bank_expansion_factor. "
                f"This parameter is only valid for LAYERED_WEIGHTED_BANK and SOFT_WEIGHTED_BANK, "
                f"received {model.cfg.bank_expansion_factor!r}."
            )

    @staticmethod
    def validate_bank_expansion_factor(model: "DynamicWeightAbstract") -> None:
        from emperor.augmentations.adaptive_parameters.options import (
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
                f"Use FACTOR_OF_ONE, FACTOR_OF_TWO, FACTOR_OF_THREE, or FACTOR_OF_FOUR."
            )

    @staticmethod
    def validate_square_dimensions(model: "DynamicWeightAbstract") -> None:
        if model.input_dim != model.output_dim:
            raise ValueError(
                f"{type(model).__name__} requires input_dim == output_dim, "
                f"received input_dim={model.input_dim}, output_dim={model.output_dim}."
            )


class DynamicBiasValidator(AdaptiveGeneratorValidatorBase, ValidatorBase):
    OPTIONAL_FIELDS = {
        "bank_expansion_factor",
        "decay_schedule",
        "decay_rate",
        "decay_warmup_batches",
    }

    @staticmethod
    def validate(model: "DynamicBiasAbstract") -> None:
        DynamicBiasValidator.validate_required_fields(model.cfg)
        DynamicBiasValidator.validate_field_types(model.cfg)
        DynamicBiasValidator.validate_dimensions(
            input_dim=model.cfg.input_dim, output_dim=model.cfg.output_dim
        )
        DynamicBiasValidator.validate_no_bank_expansion_factor(model)

    @staticmethod
    def validate_no_bank_expansion_factor(model: "DynamicBiasAbstract") -> None:
        from emperor.augmentations.adaptive_parameters.core.bias import (
            WeightedBankDynamicBias,
        )

        if isinstance(model, WeightedBankDynamicBias):
            return
        if model.cfg.bank_expansion_factor is not None:
            raise ValueError(
                f"{type(model).__name__} does not support bank_expansion_factor. "
                f"This parameter is only valid for WEIGHTED_BANK, "
                f"received {model.cfg.bank_expansion_factor!r}."
            )

    @staticmethod
    def validate_bank_expansion_factor(model: "DynamicBiasAbstract") -> None:
        factor = model.cfg.bank_expansion_factor
        if factor is None or not isinstance(factor, int):
            raise ValueError(
                f"{type(model).__name__} requires bank_expansion_factor to be an integer value, "
                f"received {factor!r}."
            )
        if factor <= 0:
            raise ValueError(
                f"{type(model).__name__} requires bank_expansion_factor > 0, "
                f"received {factor!r}."
            )

    def ensure_parameters_exist(bias_params: Tensor | None) -> None:
        if bias_params is None:
            raise ValueError(
                "bias_params must not be None. Provide a valid bias tensor for this dynamic bias strategy."
            )


class DynamicDiagonalValidator(AdaptiveGeneratorValidatorBase, ValidatorBase):
    @staticmethod
    def validate(model: "DynamicDiagonalAbstract") -> None:
        DynamicDiagonalValidator.validate_required_fields(model.cfg)
        DynamicDiagonalValidator.validate_field_types(model.cfg)
        DynamicDiagonalValidator.validate_dimensions(
            input_dim=model.cfg.input_dim, output_dim=model.cfg.output_dim
        )


class AxisMaskValidator(AdaptiveGeneratorValidatorBase, ValidatorBase):
    OPTIONAL_FIELDS = {"mask_transition_width"}

    @staticmethod
    def validate(model: "AxisMaskAbstract") -> None:
        AxisMaskValidator.validate_required_fields(model.cfg)
        AxisMaskValidator.validate_field_types(model.cfg)
        AxisMaskValidator.validate_dimensions(
            input_dim=model.cfg.input_dim, output_dim=model.cfg.output_dim
        )
        AxisMaskValidator.validate_mask_threshold(model.cfg.mask_threshold)
        AxisMaskValidator.validate_mask_surrogate_scale(
            model.cfg.mask_surrogate_scale
        )
        AxisMaskValidator.validate_mask_floor(model.cfg.mask_floor)
        if model.cfg.mask_transition_width is not None:
            AxisMaskValidator.validate_mask_transition_width(
                model.cfg.mask_transition_width
            )

    @staticmethod
    def validate_mask_threshold(mask_threshold: float) -> None:
        if not 0.0 <= mask_threshold <= 1.0:
            raise ValueError(
                "mask_threshold must be between 0.0 and 1.0 inclusive, "
                f"received {mask_threshold!r}."
            )

    @staticmethod
    def validate_mask_surrogate_scale(mask_surrogate_scale: float) -> None:
        if mask_surrogate_scale <= 0.0:
            raise ValueError(
                "mask_surrogate_scale must be greater than 0.0, "
                f"received {mask_surrogate_scale!r}."
            )

    @staticmethod
    def validate_mask_floor(mask_floor: float) -> None:
        if not 0.0 <= mask_floor < 1.0:
            raise ValueError(
                "mask_floor must be between 0.0 inclusive and 1.0 exclusive, "
                f"received {mask_floor!r}."
            )

    @staticmethod
    def validate_mask_transition_width(mask_transition_width: float) -> None:
        if mask_transition_width <= 0.0:
            raise ValueError(
                "mask_transition_width must be greater than 0.0, "
                f"received {mask_transition_width!r}."
            )


class DepthMappingValidator(ValidatorBase):
    OPTIONAL_FIELDS = {"override_config", "adaptive_augmentation_config"}

    @staticmethod
    def validate(model: "DepthMappingLayer") -> None:
        DepthMappingValidator.validate_required_fields(model.cfg)
        DepthMappingValidator.validate_field_types(model.cfg)
        DepthMappingValidator.validate_dimensions(
            input_dim=model.cfg.input_dim, output_dim=model.cfg.output_dim
        )
        DepthMappingValidator.validate_generator_depth(model.cfg)

    @staticmethod
    def validate_generator_depth(cfg: "DepthMappingLayerConfig") -> None:
        if cfg.generator_depth is None or cfg.generator_depth.value == 0:
            raise ValueError(
                f"generator_depth must be greater than 0 for DepthMappingLayer, "
                f"received {cfg.generator_depth}. "
                f"Use DEPTH_OF_ONE, DEPTH_OF_TWO, or DEPTH_OF_THREE."
            )

    @staticmethod
    def validate_input_is_2d(input_batch: Tensor) -> None:
        if not input_batch.dim() == 2:
            raise ValueError(
                f"DepthMappingLayerStack expects a 2D input tensor (batch_size, features), "
                f"received a {input_batch.dim()}D tensor with shape {tuple(input_batch.shape)}."
            )

    @staticmethod
    def validate_inner_model_is_linear_layer_config(
        model_config: "LayerStackConfig",
    ) -> None:
        from emperor.linears.core.config import LinearLayerConfig

        inner_config = model_config.layer_config.layer_model_config
        if not isinstance(inner_config, LinearLayerConfig):
            raise TypeError(
                f"model_config.layer_config.layer_model_config must be a LinearLayerConfig, "
                f"received {type(inner_config).__name__}."
            )

    @staticmethod
    def validate_layer_config_has_no_gate_or_halting(
        model_config: "LayerStackConfig",
    ) -> None:
        layer_config = model_config.layer_config
        if layer_config.gate_config is not None:
            raise ValueError(
                "DepthMappingLayerStack does not support gate_config. "
                "Set gate_config to None."
            )
        if layer_config.halting_config is not None:
            raise ValueError(
                "DepthMappingLayerStack does not support halting_config. "
                "Set halting_config to None."
            )
        if layer_config.shared_halting_flag:
            raise ValueError(
                "DepthMappingLayerStack does not support shared_halting_flag. "
                "Set shared_halting_flag to False."
            )


class AdaptiveParameterAugmentationValidator:
    _FIELDS = {
        "input_dim": {
            "type": int,
            "validate": lambda v: v > 0 or "must be > 0",
        },
        "output_dim": {
            "type": int,
            "validate": lambda v: v > 0 or "must be > 0",
        },
        "weight_option": {
            "type": "DynamicWeightOptions",
        },
        "weight_normalization": {
            "type": "WeightNormalizationOptions",
        },
        "generator_depth": {
            "type": "DynamicDepthOptions",
        },
        "diagonal_option": {
            "type": "DynamicDiagonalOptions",
        },
        "bias_option": {
            "type": "DynamicBiasOptions",
        },
        "row_mask_option": {
            "type": "AxisMaskOptions",
        },
        "mask_dimension_option": {
            "type": "MaskDimensionOptions",
            "optional": True,
        },
        "memory_option": {
            "type": "LinearMemoryOptions",
        },
        "memory_size_option": {
            "type": "LinearMemorySizeOptions",
        },
        "memory_position_option": {
            "type": "LinearMemoryPositionOptions",
        },
    }

    def __init__(self, model: "AdaptiveParameterAugmentation"):
        self.model = model
        self._resolve_enum_types()
        self.validate()

    def _resolve_enum_types(self) -> None:
        from emperor.augmentations.adaptive_parameters.options import (
            AxisMaskOptions,
            DynamicBiasOptions,
            DynamicDepthOptions,
            DynamicDiagonalOptions,
            DynamicWeightOptions,
            LinearMemoryOptions,
            LinearMemoryPositionOptions,
            LinearMemorySizeOptions,
            MaskDimensionOptions,
            WeightNormalizationOptions,
        )

        self._TYPES = {
            "DynamicBiasOptions": DynamicBiasOptions,
            "DynamicDepthOptions": DynamicDepthOptions,
            "DynamicDiagonalOptions": DynamicDiagonalOptions,
            "DynamicWeightOptions": DynamicWeightOptions,
            "LinearMemoryOptions": LinearMemoryOptions,
            "LinearMemorySizeOptions": LinearMemorySizeOptions,
            "LinearMemoryPositionOptions": LinearMemoryPositionOptions,
            "MaskDimensionOptions": MaskDimensionOptions,
            "AxisMaskOptions": AxisMaskOptions,
            "WeightNormalizationOptions": WeightNormalizationOptions,
        }

    def validate(self) -> None:
        for name, rules in self._FIELDS.items():
            val = getattr(self.model, name)

            if val is None:
                if rules.get("optional"):
                    continue
                raise ValueError(
                    f"Configuration error: '{name}' is required for "
                    f"{self.model.__class__.__name__}."
                )

            expected = rules.get("type")
            expected_type = self._TYPES.get(expected, expected)
            if not isinstance(val, expected_type):
                raise TypeError(
                    f"Type error: '{name}' on {self.model.__class__.__name__} "
                    f"expected {expected_type.__name__}, got "
                    f"{type(val).__name__} (value={val!r})."
                )

            validator = rules.get("validate")
            if validator is not None:
                result = validator(val)
                if result is not True and result is not None:
                    raise ValueError(
                        f"Configuration error: '{name}' {result} (value={val!r})."
                    )
