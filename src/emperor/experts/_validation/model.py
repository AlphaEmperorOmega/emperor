from typing import TYPE_CHECKING

from emperor._validation import ValidatorBase
from emperor.experts._options import RoutingInitializationMode

if TYPE_CHECKING:
    from emperor.experts._model import MixtureOfExpertsModel


class MixtureOfExpertsModelValidator(ValidatorBase):
    @classmethod
    def validate(cls, model: "MixtureOfExpertsModel") -> None:
        cls.validate_cfg_type(model)
        cls.validate_stack_config_type(model)
        cls.validate_shared_routing_config_when_shared(model)

    @staticmethod
    def validate_cfg_type(model: "MixtureOfExpertsModel") -> None:
        from emperor.experts._config import MixtureOfExpertsModelConfig

        if not isinstance(model.cfg, MixtureOfExpertsModelConfig):
            raise TypeError(
                "Configuration Error: `cfg` must be of type "
                "MixtureOfExpertsModelConfig, received type "
                f"{type(model.cfg).__name__}"
            )

    @staticmethod
    def validate_stack_config_type(model: "MixtureOfExpertsModel") -> None:
        from emperor.layers import LayerStackConfig

        if not isinstance(model.stack_config, LayerStackConfig):
            raise TypeError(
                "Configuration Error: 'stack_config' must be of type "
                "LayerStackConfig, received type "
                f"{type(model.stack_config).__name__}"
            )

    @staticmethod
    def validate_shared_routing_config_when_shared(
        model: "MixtureOfExpertsModel",
    ) -> None:
        from emperor.sampler import RouterConfig, SamplerConfig

        if model.routing_initialization_mode != RoutingInitializationMode.SHARED:
            return
        if not isinstance(model.sampler_config, SamplerConfig):
            raise TypeError(
                "Configuration Error: 'sampler_config' must be of type "
                "SamplerConfig when 'routing_initialization_mode' is SHARED, "
                f"received type {type(model.sampler_config).__name__}"
            )
        if not isinstance(model.sampler_config.router_config, RouterConfig):
            raise TypeError(
                "Configuration Error: 'sampler_config.router_config' must be of "
                "type RouterConfig when 'routing_initialization_mode' is SHARED, "
                f"received type {type(model.sampler_config.router_config).__name__}"
            )
