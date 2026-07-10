from typing import TYPE_CHECKING

from emperor.base.layer import LayerStackConfig, RecurrentLayerConfig
from emperor.base.utils import ConfigBase
from emperor.base.validator import ValidatorBase
from emperor.experts.config import MixtureOfExpertsModelConfig

if TYPE_CHECKING:
    from emperor.transformer.feed_forward.core.layers import FeedForward


class FeedForwardValidator(ValidatorBase):
    @staticmethod
    def validate(model: "FeedForward") -> None:
        FeedForwardValidator.validate_required_fields(model.cfg)
        FeedForwardValidator.validate_dimensions(
            input_dim=model.input_dim, output_dim=model.output_dim
        )
        FeedForwardValidator.validate_stack_config_type(model.stack_config)

    @staticmethod
    def validate_stack_config_type(stack_config: ConfigBase) -> None:
        if not isinstance(
            stack_config,
            (LayerStackConfig, MixtureOfExpertsModelConfig, RecurrentLayerConfig),
        ):
            raise TypeError(
                "FeedForward.stack_config must be a LayerStackConfig, "
                "MixtureOfExpertsModelConfig, or RecurrentLayerConfig, got "
                f"{type(stack_config).__name__}"
            )
