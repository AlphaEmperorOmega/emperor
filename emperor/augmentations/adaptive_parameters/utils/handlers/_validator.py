from torch.types import Tensor

from emperor.base.validator import ValidatorBase

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from emperor.augmentations.adaptive_parameters.utils.handlers.bias import BiasHandlerAbstract
    from emperor.augmentations.adaptive_parameters.utils.handlers.depth_mapper import (
        DepthMappingLayer,
        DepthMappingLayerConfig,
    )


class BiasHandlerAbstractValidator:
    def __init__(self, model: "BiasHandlerAbstract"):
        self.model = model

    def ensure_parameters_exist(self, bias_params: Tensor | None) -> None:
        if bias_params is None:
            raise ValueError(
                "The 'bias_params' argument cannot be None. Please provide valid parameters to proceed."
            )


class DepthMappingLayerValidator(ValidatorBase):
    @staticmethod
    def validate(model: "DepthMappingLayer") -> None:
        DepthMappingLayerValidator.validate_required_fields(model.cfg)
        DepthMappingLayerValidator.validate_field_types(model.cfg)
        DepthMappingLayerValidator.validate_dimensions(
            input_dim=model.cfg.input_dim, output_dim=model.cfg.output_dim
        )
        DepthMappingLayerValidator.validate_generator_depth(model.cfg)

    @staticmethod
    def validate_generator_depth(cfg: "DepthMappingLayerConfig") -> None:
        if cfg.generator_depth is None or cfg.generator_depth.value == 0:
            raise ValueError(
                f"generator_depth must be greater than 0 for DepthMappingLayer, "
                f"received {cfg.generator_depth}. "
                f"Use DEPTH_OF_ONE, DEPTH_OF_TWO, or DEPTH_OF_THREE"
            )
