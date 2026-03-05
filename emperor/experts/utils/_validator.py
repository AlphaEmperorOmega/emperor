from torch import Tensor

from emperor.experts.utils.enums import (
    ExpertWeightingPositionOptions,
    InitSamplerOptions,
)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from emperor.experts.utils.layers import MixtureOfExperts
    from emperor.experts.utils.model import MixtureOfExpertsModel


class _Validator:
    def __init__(self, model: "MixtureOfExperts"):
        self.model = model
        self.__ensure_values_are_not_none()
        self.__ensure_values_have_correct_types()

    def __ensure_values_are_not_none(self):
        if self.model.input_dim is None:
            raise ValueError("Configuration Error: 'input_dim' is None")
        if self.model.output_dim is None:
            raise ValueError("Configuration Error: 'output_dim' is None")
        if self.model.layer_stack_model is None:
            raise ValueError("Configuration Error: 'layer_stack_option' is None")
        if self.model.top_k is None:
            raise ValueError("Configuration Error: 'top_k' is None")
        if self.model.num_experts is None:
            raise ValueError("Configuration Error: 'num_experts' is None")
        if self.model.compute_expert_mixture_flag is None:
            raise ValueError(
                "Configuration Error: 'compute_expert_mixture_flag' is None"
            )
        if self.model.weighted_parameters_flag is None:
            raise ValueError("Configuration Error: 'weighted_parameters_flag' is None")
        if self.model.init_sampler_option is None:
            raise ValueError("Configuration Error: 'init_sampler_option' is None")
        if self.model.weighting_position_option is None:
            raise ValueError("Configuration Error: 'weighting_position_option' is None")
        if self.model.router_model_config is None:
            raise ValueError("Configuration Error: 'router_model_config' is None")
        if self.model.sampler_model_config is None:
            raise ValueError("Configuration Error: 'sampler_model_config' is None")

    def __ensure_values_have_correct_types(self):
        from emperor.linears.options import LinearLayerStackOptions
        from emperor.sampler.utils.samplers import SamplerConfig
        from emperor.sampler.utils.routers import RouterConfig

        if not isinstance(self.model.input_dim, int):
            raise TypeError(
                f"Configuration Error: 'input_dim' must be of type int, received type {type(self.model.input_dim).__name__}"
            )
        if not isinstance(self.model.output_dim, int):
            raise TypeError(
                f"Configuration Error: 'output_dim' must be of type int, received type {type(self.model.output_dim).__name__}"
            )
        if not isinstance(self.model.layer_stack_model, LinearLayerStackOptions):
            raise TypeError(
                f"Configuration Error: 'layer_stack_option' must be of type LinearLayerStackOptions, received type {type(self.model.layer_stack_model).__name__}"
            )
        if not isinstance(self.model.top_k, int):
            raise TypeError(
                f"Configuration Error: 'top_k' must be of type int, received type {type(self.model.top_k).__name__}"
            )
        if not isinstance(self.model.num_experts, int):
            raise TypeError(
                f"Configuration Error: 'num_experts' must be of type int, received type {type(self.model.num_experts).__name__}"
            )
        if not isinstance(self.model.compute_expert_mixture_flag, bool):
            raise TypeError(
                f"Configuration Error: 'compute_expert_mixture_flag' must be of type bool, received type {type(self.model.compute_expert_mixture_flag).__name__}"
            )
        if not isinstance(self.model.weighted_parameters_flag, bool):
            raise TypeError(
                f"Configuration Error: 'weighted_parameters_flag' must be of type bool, received type {type(self.model.weighted_parameters_flag).__name__}"
            )
        if not isinstance(
            self.model.weighting_position_option,
            ExpertWeightingPositionOptions,
        ):
            raise TypeError(
                f"Configuration Error: 'weighting_position_option' must be of type ExpertWeightingPositionOptions, received type {type(self.model.weighting_position_option).__name__}"
            )
        if not isinstance(self.model.init_sampler_option, InitSamplerOptions):
            raise TypeError(
                f"Configuration Error: 'init_sampler_model_flag' must be of type bool, received type {type(self.model.init_sampler_model_flag).__name__}"
            )
        if not isinstance(self.model.router_model_config, RouterConfig):
            raise TypeError(
                f"Configuration Error: 'router_model_config' must be of type RouterConfig, received type {type(self.model.router_model_config).__name__}"
            )
        if not isinstance(self.model.sampler_model_config, SamplerConfig):
            raise TypeError(
                f"Configuration Error: 'sampler_model_config' must be of type SamplerConfig, received type {type(self.model.sampler_model_config).__name__}"
            )
        if self.model.capacity_factor is not None and not isinstance(
            self.model.capacity_factor, float
        ):
            raise TypeError(
                f"Configuration Error: 'capacity_factor' must be of type float or None, received type {type(self.model.capacity_factor).__name__}"
            )

    def ensure_sampler_is_initialized(self) -> None:
        options = [InitSamplerOptions.DISABLED, InitSamplerOptions.LAYER]
        if self.model.init_sampler_option not in options:
            raise ValueError(
                f"The `init_sampler_option` must be set to `InitSamplerOptions.LAYER` to initialize the `RouterModel` and `SamplerModel` when `indices` are not provided. Current option: {self.model.init_sampler_option}"
            )

    def ensure_external_probabilities_are_not_given(
        self,
        probabilities: Tensor | None,
        indices: Tensor | None,
    ) -> None:
        if indices is not None or probabilities is not None:
            raise ValueError(
                "Indices must be None. Providing indices where they are not expected is not allowed."
            )

    def ensure_no_sampler_with_indices(self) -> None:
        if self.model.init_sampler_option != InitSamplerOptions.LAYER:
            raise ValueError(
                f"Invalid configuration: `init_sampler_model_flag` must be set to `False` when `indices` are provided. This prevents creating duplicate `RouterModel` and `SamplerModel` instances in the current layer. Current value: {self.model.init_sampler_option}"
            )

    def ensure_probabilities_exist(self, probabilities: Tensor | None) -> None:
        if probabilities is None:
            raise ValueError(
                "Missing input: `probabilities` must be supplied when `indices` are used to ensure accurate weighting and processing of inputs."
            )

    def ensure_router_config_exists(self) -> None:
        if self.model.router_model_config is None:
            raise ValueError(
                "Configuration Error: `router_model_config` must be defined to properly initialize and utilize the router model in the mixture of experts layer."
            )

    def ensure_sampler_config_exists(self) -> None:
        if self.model.sampler_model_config is None:
            raise ValueError(
                "Configuration Error: `sampler_model_config` must be defined to properly initialize and utilize the sampler model in the mixture of experts layer."
            )

    def ensure_tensor_is_vector_or_matrix(self, X: Tensor | None) -> None:
        if X is not None and X.dim() > 2:
            raise ValueError(
                "Input Error: `X` must be a 1-dimensional or 2-dimensional tensor."
            )


class MixtureOfExpertsModelValidator:
    def __init__(self, model: "MixtureOfExpertsModel"):
        self.model = model
        self.__ensure_propper_main_config()

    def __ensure_propper_main_config(self) -> None:
        from emperor.experts.utils.layers import MixtureOfExpertsConfig

        if self.model.main_cfg == MixtureOfExpertsConfig:
            raise ValueError(
                "Invalid configuration: `main_cfg` must not directly match `MixtureOfExpertsConfig`. Ensure `main_cfg` is correctly derived and properly initialized."
            )

    def ensure_no_sampler_with_indices(self) -> None:
        if self.model.init_sampler_option == InitSamplerOptions.DISABLED:
            raise ValueError(
                "Invalid configuration: `init_sampler_model_flag` must be set to `False` when `indices` are provided. This prevents creating duplicate `RouterModel` and `SamplerModel` instances in the current layer."
            )

    def ensure_tensor_is_vector_or_matrix(self, X: Tensor | None) -> None:
        if X is not None and X.dim() > 2:
            raise ValueError(
                "Input Error: `X` must be a 1-dimensional or 2-dimensional tensor."
            )
