from torch.types import Tensor
from emperor.base.validator import ValidatorBase

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from emperor.augmentations.adaptive_parameters.core.diagonal import (
        DynamicDiagonalAbstract,
    )
    from emperor.augmentations.adaptive_parameters.core.mask import AxisMaskAbstract
    from emperor.augmentations.adaptive_parameters.core.bias import DynamicBiasAbstract
    from emperor.augmentations.adaptive_parameters.model import (
        AdaptiveParameterAugmentation,
    )
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
            AdaptiveGeneratorValidatorBase._validate_generator_layer(generator_model)
            return
        if isinstance(generator_model, Sequential):
            AdaptiveGeneratorValidatorBase._validate_generator_sequence(generator_model)
            return
        raise TypeError(
            "Expected model_config.build(...) to return a Layer or Sequential, "
            f"received {type(generator_model).__name__}."
        )

    @staticmethod
    def _validate_generator_sequence(generator_sequence) -> None:
        from emperor.base.layer import Layer

        for generator_layer in generator_sequence:
            if not isinstance(generator_layer, Layer):
                raise TypeError(
                    "Expected each generator sequence item to be a Layer, "
                    f"received {type(generator_layer).__name__}."
                )
            AdaptiveGeneratorValidatorBase._validate_generator_layer(generator_layer)

    @staticmethod
    def _validate_generator_layer(generator_layer) -> None:
        from emperor.linears.core.layers import LinearLayer

        if not isinstance(generator_layer.model, LinearLayer):
            raise TypeError(
                "Expected each generator Layer to wrap a LinearLayer, "
                f"received {type(generator_layer.model).__name__}."
            )

    @staticmethod
    def validate_decay_parameters(cfg) -> None:
        from emperor.augmentations.adaptive_parameters.options import (
            WeightDecayScheduleOptions,
        )

        schedule = cfg.decay_schedule
        if schedule is None or schedule == WeightDecayScheduleOptions.DISABLED:
            return
        decay_rate = cfg.decay_rate
        if decay_rate is None or decay_rate <= 0.0:
            raise ValueError(
                f"decay_rate must be greater than 0.0 when decay_schedule is "
                f"{schedule.name}, received {decay_rate!r}."
            )
        bounded_schedules = {
            WeightDecayScheduleOptions.LINEAR,
            WeightDecayScheduleOptions.MULTIPLICATIVE,
        }
        if schedule in bounded_schedules and decay_rate >= 1.0:
            raise ValueError(
                f"decay_rate must be less than 1.0 for {schedule.name}, "
                f"received {decay_rate!r}."
            )
        decay_warmup_batches = cfg.decay_warmup_batches
        if decay_warmup_batches is not None and decay_warmup_batches < 0:
            raise ValueError(
                f"decay_warmup_batches must be >= 0, "
                f"received {decay_warmup_batches!r}."
            )


class DynamicWeightValidator(AdaptiveGeneratorValidatorBase, ValidatorBase):
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
        AdaptiveGeneratorValidatorBase.validate_decay_parameters(model.cfg)

    @staticmethod
    def validate_no_bank_expansion_factor(model: "DynamicWeightAbstract") -> None:
        from emperor.augmentations.adaptive_parameters.core.weight import (
            LayeredWeightedBankDynamicWeight,
            SoftWeightedBankDynamicWeight,
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
        AdaptiveGeneratorValidatorBase.validate_decay_parameters(model.cfg)

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

    @staticmethod
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
        AxisMaskValidator.validate_mask_surrogate_scale(model.cfg.mask_surrogate_scale)
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
        if mask_surrogate_scale < 0.0:
            raise ValueError(
                "mask_surrogate_scale must be greater than or equal to 0.0, "
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
    OPTIONAL_FIELDS = {
        "model_type",
        "override_config",
        "adaptive_augmentation_config",
    }

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
    @staticmethod
    def validate(model: "AdaptiveParameterAugmentation") -> None:
        AdaptiveParameterAugmentationValidator.validate_dimensions(model)
        AdaptiveParameterAugmentationValidator.validate_sub_configs(model)

    @staticmethod
    def validate_dimensions(model: "AdaptiveParameterAugmentation") -> None:
        if (
            model.input_dim is None
            or not isinstance(model.input_dim, int)
            or model.input_dim <= 0
        ):
            raise ValueError(
                f"input_dim must be a positive integer, received {model.input_dim!r}."
            )
        if (
            model.output_dim is None
            or not isinstance(model.output_dim, int)
            or model.output_dim <= 0
        ):
            raise ValueError(
                f"output_dim must be a positive integer, received {model.output_dim!r}."
            )

    @staticmethod
    def validate_sub_configs(model: "AdaptiveParameterAugmentation") -> None:
        from emperor.base.utils import ConfigBase
        from emperor.base.layer.config import LayerStackConfig

        sub_configs = [
            ("weight_config", model.weight_config),
            ("diagonal_config", model.diagonal_config),
            ("bias_config", model.bias_config),
            ("mask_config", model.mask_config),
        ]
        for name, config in sub_configs:
            if config is None:
                continue
            if not isinstance(config, ConfigBase):
                raise TypeError(
                    f"{name} must be a ConfigBase instance, "
                    f"got {type(config).__name__}."
                )
            if config.model_config is None and model.model_config is None:
                raise ValueError(
                    f"{type(config).__name__} requires a model_config but none was provided "
                    f"on the sub-config or the parent AdaptiveParameterAugmentationConfig."
                )
