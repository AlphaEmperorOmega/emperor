from typing import TYPE_CHECKING

from torch.types import Tensor

from emperor._validation import ValidatorBase
from emperor.augmentations.adaptive_parameters._validation import (
    AdaptiveGeneratorValidatorBase,
)

if TYPE_CHECKING:
    from emperor.augmentations.adaptive_parameters._weights.base import (
        DynamicWeightAbstract,
    )
    from emperor.augmentations.adaptive_parameters._weights.depth_mapping import (
        DepthMappingLayer,
        DepthMappingLayerConfig,
    )
    from emperor.layers import LayerStackConfig


class DynamicWeightValidator(AdaptiveGeneratorValidatorBase, ValidatorBase):
    OPTIONAL_FIELDS = {"bank_expansion_factor"}

    @classmethod
    def validate(cls, model: "DynamicWeightAbstract") -> None:
        cls.validate_required_fields(model.cfg)
        cls.validate_field_types(model.cfg)
        cls.validate_dimensions(
            input_dim=model.cfg.input_dim,
            output_dim=model.cfg.output_dim,
        )
        cls.validate_decay_parameters(model.cfg)

    @staticmethod
    def validate_bank_expansion_factor(model: "DynamicWeightAbstract") -> None:
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
    def validate_square_dimensions(model: "DynamicWeightAbstract") -> None:
        if model.input_dim != model.output_dim:
            raise ValueError(
                f"{type(model).__name__} requires input_dim == output_dim, "
                f"received input_dim={model.input_dim}, output_dim={model.output_dim}."
            )


class DepthMappingValidator(ValidatorBase):
    OPTIONAL_FIELDS: set[str] = set()

    @classmethod
    def validate(cls, model: "DepthMappingLayer") -> None:
        cls.validate_required_fields(model.cfg)
        cls.validate_field_types(model.cfg)
        cls.validate_dimensions(
            input_dim=model.cfg.input_dim,
            output_dim=model.cfg.output_dim,
        )
        cls._validate_generator_depth(model.cfg)

    @staticmethod
    def _validate_generator_depth(cfg: "DepthMappingLayerConfig") -> None:
        if cfg.generator_depth is None or cfg.generator_depth.value == 0:
            raise ValueError(
                f"generator_depth must be greater than 0 for DepthMappingLayer, "
                f"received {cfg.generator_depth}. "
                f"Use DEPTH_OF_ONE, DEPTH_OF_TWO, or DEPTH_OF_THREE."
            )

    @staticmethod
    def validate_input_is_2d(
        input_batch: Tensor,
        input_dim: int | None = None,
    ) -> None:
        if not isinstance(input_batch, Tensor):
            raise TypeError(
                "DepthMappingLayerStack input must be a Tensor, "
                f"received {type(input_batch).__name__}."
            )
        if not input_batch.dim() == 2:
            raise ValueError(
                "DepthMappingLayerStack expects a 2D input tensor "
                "(batch_size, features), "
                f"received a {input_batch.dim()}D tensor with shape "
                f"{tuple(input_batch.shape)}."
            )
        if input_dim is not None and input_batch.shape[-1] != input_dim:
            raise ValueError(
                "DepthMappingLayerStack input feature dimension must match "
                "input_dim, "
                f"received input_dim={input_dim} and input shape "
                f"{tuple(input_batch.shape)}."
            )

    @staticmethod
    def validate_inner_model_is_linear_layer_config(
        model_config: "LayerStackConfig",
    ) -> None:
        from emperor.linears import LinearLayerConfig

        inner_config = model_config.layer_config.layer_model_config
        if not isinstance(inner_config, LinearLayerConfig):
            raise TypeError(
                "model_config.layer_config.layer_model_config must be a "
                "LinearLayerConfig, "
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
        if model_config.shared_gate_config is not None:
            raise ValueError(
                "DepthMappingLayerStack does not support shared_gate_config. "
                "Set shared_gate_config to None."
            )
        if layer_config.halting_config is not None:
            raise ValueError(
                "DepthMappingLayerStack does not support halting_config. "
                "Set halting_config to None."
            )
        if model_config.shared_halting_config is not None:
            raise ValueError(
                "DepthMappingLayerStack does not support shared_halting_config. "
                "Set shared_halting_config to None."
            )
        if layer_config.memory_config is not None:
            raise ValueError(
                "DepthMappingLayerStack does not support memory_config. "
                "Set memory_config to None."
            )
        if model_config.shared_memory_config is not None:
            raise ValueError(
                "DepthMappingLayerStack does not support shared_memory_config. "
                "Set shared_memory_config to None."
            )
