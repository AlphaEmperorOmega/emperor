from torch.types import Tensor

from emperor.base.validator import ValidatorBase

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from emperor.augmentations.adaptive_parameters.core.handlers.bias import BiasHandlerAbstract
    from emperor.augmentations.adaptive_parameters.core.handlers.depth_mapper import (
        DepthMappingLayer,
        DepthMappingLayerConfig,
    )
    from emperor.base.layer import LayerStackConfig


class BiasHandlerAbstractValidator:
    def __init__(self, model: "BiasHandlerAbstract"):
        self.model = model

    def ensure_parameters_exist(self, bias_params: Tensor | None) -> None:
        if bias_params is None:
            raise ValueError(
                "The 'bias_params' argument cannot be None. Please provide valid parameters to proceed."
            )


class DepthMappingValidator(ValidatorBase):
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
    def validate_inner_model_is_linear_layer_config(model_config: "LayerStackConfig") -> None:
        from emperor.linears.core.config import LinearLayerConfig

        inner_config = model_config.layer_config.layer_model_config
        if not isinstance(inner_config, LinearLayerConfig):
            raise TypeError(
                f"model_config.layer_config.layer_model_config must be a LinearLayerConfig, "
                f"received {type(inner_config).__name__}"
            )

    @staticmethod
    def validate_layer_config_has_no_gate_or_halting(model_config: "LayerStackConfig") -> None:
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
