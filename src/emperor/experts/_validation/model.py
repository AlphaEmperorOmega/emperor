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

    @classmethod
    def validate_cfg_type(cls, model: "MixtureOfExpertsModel") -> None:
        cls.validate_config_type(model.cfg)

    @staticmethod
    def validate_config_type(cfg) -> None:
        from emperor.experts._config import MixtureOfExpertsModelConfig

        if not isinstance(cfg, MixtureOfExpertsModelConfig):
            raise TypeError(
                "Configuration Error: `cfg` must be of type "
                "MixtureOfExpertsModelConfig, received type "
                f"{type(cfg).__name__}"
            )

    @staticmethod
    def validate_overrides_type(overrides) -> None:
        from emperor.experts._config import MixtureOfExpertsModelConfig

        if overrides is not None and not isinstance(
            overrides, MixtureOfExpertsModelConfig
        ):
            raise TypeError(
                "Configuration Error: `overrides` must be of type "
                "MixtureOfExpertsModelConfig or None, received type "
                f"{type(overrides).__name__}"
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
