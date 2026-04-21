from torch.types import Tensor

from emperor.base.validator import ValidatorBase

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from emperor.augmentations.adaptive_parameters.core.bias import DynamicBiasAbstract
    from emperor.augmentations.adaptive_parameters.core.depth_mapper import (
        DepthMappingLayer,
        DepthMappingLayerConfig,
    )
    from emperor.augmentations.adaptive_parameters.core.weight import (
        DynamicWeightAbstract,
    )
    from emperor.base.layer import LayerStackConfig


class DynamicWeightValidator(ValidatorBase):
    OPTIONAL_FIELDS = {"model_config"}

    @staticmethod
    def validate(model: "DynamicWeightAbstract") -> None:
        DynamicWeightValidator.validate_required_fields(model.cfg)
        DynamicWeightValidator.validate_field_types(model.cfg)
        DynamicWeightValidator.validate_dimensions(
            input_dim=model.cfg.input_dim, output_dim=model.cfg.output_dim
        )

    @staticmethod
    def validate_square_dimensions(model: "DynamicWeightAbstract") -> None:
        if model.input_dim != model.output_dim:
            raise ValueError(
                f"{type(model).__name__} requires input_dim == output_dim, "
                f"received input_dim={model.input_dim}, output_dim={model.output_dim}."
            )


class DynamicBiasAbstractValidator:
    def __init__(self, model: "DynamicBiasAbstract"):
        self.model = model

    def ensure_parameters_exist(self, bias_params: Tensor | None) -> None:
        if bias_params is None:
            raise ValueError(
                "The 'bias_params' argument cannot be None. Please provide valid parameters to proceed."
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
                f"Use DEPTH_OF_ONE, DEPTH_OF_TWO, or DEPTH_OF_THREE"
            )

    @staticmethod
    def validate_input_is_2d(input_batch: Tensor) -> None:
        if not input_batch.dim() == 2:
            raise ValueError(
                f"DepthMappingLayerStack expects a 2D input tensor (batch_size, features), "
                f"received {input_batch.dim()}D tensor with shape {tuple(input_batch.shape)}"
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
                f"received {type(inner_config).__name__}"
            )

    @staticmethod
    def validate_layer_config_has_no_gate_or_halting(
        model_config: "LayerStackConfig",
    ) -> None:
        layer_config = model_config.layer_config
        if layer_config.gate_config is not None:
            raise ValueError(
                "DepthMappingLayerStack does not support gate_config. "
                "gate_config must be None."
            )
        if layer_config.halting_config is not None:
            raise ValueError(
                "DepthMappingLayerStack does not support halting_config. "
                "halting_config must be None."
            )
        if layer_config.shared_halting_flag:
            raise ValueError(
                "DepthMappingLayerStack does not support shared_halting_flag. "
                "shared_halting_flag must be False."
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
            "type": "RowMaskOptions",
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
            DynamicBiasOptions,
            DynamicDepthOptions,
            DynamicDiagonalOptions,
            DynamicWeightOptions,
            LinearMemoryOptions,
            LinearMemoryPositionOptions,
            LinearMemorySizeOptions,
            MaskDimensionOptions,
            RowMaskOptions,
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
            "RowMaskOptions": RowMaskOptions,
            "WeightNormalizationOptions": WeightNormalizationOptions,
        }

    def validate(self) -> None:
        for name, rules in self._FIELDS.items():
            val = getattr(self.model, name)

            if val is None:
                if rules.get("optional"):
                    continue
                raise ValueError(
                    f"Configuration Error: '{name}' is required for "
                    f"{self.model.__class__.__name__}."
                )

            expected = rules.get("type")
            expected_type = self._TYPES.get(expected, expected)
            if not isinstance(val, expected_type):
                raise TypeError(
                    f"Type Error: '{name}' on {self.model.__class__.__name__} "
                    f"expected {expected_type.__name__}, got "
                    f"{type(val).__name__} (value={val!r})."
                )

            validator = rules.get("validate")
            if validator is not None:
                result = validator(val)
                if result is not True and result is not None:
                    raise ValueError(
                        f"Configuration Error: '{name}' {result} (value={val!r})."
                    )
