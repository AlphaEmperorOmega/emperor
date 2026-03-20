from emperor.parametric.utils.mixtures.options import (
    AdaptiveBiasOptions,
    AdaptiveWeightOptions,
)

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from emperor.parametric.utils.layers import AdaptiveParameterLayer
    from emperor.parametric.utils.handlers import ParameterHanlderBase


class _AdaptiveParameterLayerValidator:
    def __init__(self, model: "AdaptiveParameterLayer"):
        self.model = model
        self.__ensure_values_are_not_none()
        self.__ensure_correct_input_types()
        self.__ensure_adaptive_bias_option_is_disabled_for_behaviour_bias()
        self.__ensure_no_parameter_depth_mapping_can_be_used()

    def __ensure_values_are_not_none(self) -> None:
        required_attributes = [
            "input_dim",
            "output_dim",
            "adaptive_weight_option",
            "adaptive_bias_option",
            "init_sampler_model_option",
            "time_tracker_flag",
        ]

        for attr_name in required_attributes:
            if getattr(self.model, attr_name) is None:
                raise ValueError(f"Configuration Error: '{attr_name}' is None.")

    def __ensure_correct_input_types(self) -> None:
        from emperor.parametric.utils.layers import AdaptiveRouterOptions

        required_types = {
            "input_dim": int,
            "output_dim": int,
            "init_sampler_model_option": AdaptiveRouterOptions,
            "adaptive_weight_option": AdaptiveWeightOptions,
            "adaptive_bias_option": AdaptiveBiasOptions,
            "time_tracker_flag": bool,
        }

        for attr_name, expected_type in required_types.items():
            if not isinstance(getattr(self.model, attr_name), expected_type):
                raise TypeError(
                    f"Type Error: '{attr_name}' should be {expected_type.__name__}, but got {type(getattr(self.model, attr_name)).__name__}."
                )

    def __ensure_adaptive_bias_option_is_disabled_for_behaviour_bias(
        self,
    ) -> None:
        from emperor.behaviours.utils.enums import DynamicBiasOptions

        if self.model.adaptive_behaviour_config is not None:
            is_bias_disabled = (
                self.model.adaptive_bias_option != AdaptiveBiasOptions.DISABLED
            )
            is_behaviour_bias_enabled = (
                self.model.adaptive_behaviour_config.bias_option
                != DynamicBiasOptions.DISABLED
            )
            if is_bias_disabled and is_behaviour_bias_enabled:
                raise ValueError(
                    "Configuration Error: 'adaptive_behaviour_config.bias_option' can be used for `AdaptiveParameterLayer` only when 'adaptive_bias_option' is `DISABLED`"
                )

    def __ensure_no_parameter_depth_mapping_can_be_used(
        self,
    ) -> None:
        from emperor.behaviours.utils.enums import DynamicDepthOptions

        if self.model.adaptive_behaviour_config is not None:
            is_generator_depth_disabled = (
                self.model.adaptive_behaviour_config.generator_depth
                != DynamicDepthOptions.DISABLED
            )
            if is_generator_depth_disabled:
                raise ValueError(
                    f"Configuration Error: 'adaptive_behaviour_config.generator_depth' needs to be disabled for `AdaptiveParameterLayer`, got: {self.model.adaptive_behaviour_config.generator_depth}"
                )

    def ensure_indepentent_router_for_vector_option(self) -> None:
        from emperor.parametric.utils.layers import AdaptiveRouterOptions

        is_vector_option = (
            self.model.adaptive_weight_option == AdaptiveWeightOptions.VECTOR
        )
        is_shared_router = (
            self.model.init_sampler_model_option == AdaptiveRouterOptions.SHARED_ROUTER
        )

        if is_vector_option and is_shared_router:
            raise ValueError(
                "When `adaptive_weight_option` is set to `VECTOR`, the `init_sampler_model_option` cannot be `SHARED_ROUTER`. This configuration is not supported."
            )


class _AdaptiveParameterHandlerValidator:
    def __init__(self, model: "ParameterHanlderBase"):
        self.model = model

    def ensure_router_and_sampler_configs_exist(self) -> None:
        required_types = [
            "router_config",
            "sampler_config",
        ]
        for attr_name in required_types:
            if getattr(self.model, attr_name) is None:
                raise ValueError(
                    f"Ensure that both `router_config` and `sampler_config` are provided when `init_sampler_model_flag` is `True` in `AdaptiveParameterLayer`. Current value: {getattr(self.model, attr_name)}."
                )

    def ensure_indepentent_router_for_vector_option(self) -> None:
        from emperor.parametric.utils.layers import AdaptiveRouterOptions

        is_vector_option = (
            self.model.adaptive_weight_option == AdaptiveWeightOptions.VECTOR
        )
        is_shared_router = (
            self.model.init_sampler_model_option == AdaptiveRouterOptions.SHARED_ROUTER
        )

        if is_vector_option and is_shared_router:
            raise ValueError(
                "When `adaptive_weight_option` is set to `VECTOR`, the `init_sampler_model_option` cannot be `SHARED_ROUTER`. This configuration is not supported."
            )

    def ensure_shared_sampler_is_disabled(self) -> None:
        from emperor.parametric.utils.layers import AdaptiveRouterOptions

        is_shared_router = (
            self.model.init_sampler_model_option == AdaptiveRouterOptions.SHARED_ROUTER
        )

        if is_shared_router:
            raise ValueError(
                f"Shared router is not supported for: {self.model.adaptive_weight_option}"
            )
