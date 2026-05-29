from torch import Tensor

from emperor.base.layer import LayerState
from emperor.base.validator import ValidatorBase
from emperor.parametric.core.config import (
    AdaptiveRouterOptions,
    ParametricLayerConfig,
    ParametricLayerHandlerConfig,
)
from emperor.parametric.core.mixtures.config import (
    AdaptiveMixtureConfig,
    GeneratorBiasMixtureConfig,
    GeneratorWeightsMixtureConfig,
    MatrixBiasMixtureConfig,
    MatrixWeightsMixtureConfig,
    VectorWeightsMixtureConfig,
)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from emperor.parametric.core.handlers import ParameterHandlerBase
    from emperor.parametric.core.layers import ParametricLayer
    from emperor.parametric.core.handlers import ParametricLayerHandler


class ParametricLayerValidator(ValidatorBase):
    OPTIONAL_FIELDS = {"bias_mixture_config"}

    @staticmethod
    def validate(model: "ParametricLayer") -> None:
        ParametricLayerValidator.validate_required_fields(model.cfg)
        ParametricLayerValidator.validate_field_types(model.cfg)
        ParametricLayerValidator.validate_dimensions(
            input_dim=model.input_dim,
            output_dim=model.output_dim,
        )
        ParametricLayerValidator.__validate_positive_integer(
            "input_dim", model.input_dim
        )
        ParametricLayerValidator.__validate_positive_integer(
            "output_dim", model.output_dim
        )
        ParametricLayerValidator.__validate_weight_mixture_config(
            model.weight_mixture_config
        )
        ParametricLayerValidator.__validate_bias_mixture_config(
            model.bias_mixture_config
        )
        ParametricLayerValidator.__validate_router_and_sampler_configs(model)
        ParametricLayerValidator.__validate_sampler_matches_mixtures(model)
        ParametricLayerValidator.__validate_vector_shared_router(model)
        ParametricLayerValidator.__validate_adaptive_augmentation_config(model)

    @staticmethod
    def validate_forward_inputs(input_batch: Tensor, expected_input_dim: int) -> None:
        if not isinstance(input_batch, Tensor):
            raise TypeError(
                "input_batch must be a Tensor, "
                f"received {type(input_batch).__name__}."
            )
        if input_batch.dim() != 2:
            raise ValueError(
                "Input must be a 2D matrix (batch, input_dim), "
                f"got {input_batch.dim()}D tensor with shape "
                f"{tuple(input_batch.shape)}."
            )
        if input_batch.shape[-1] != expected_input_dim:
            raise ValueError(
                "Input feature dimension must match input_dim, "
                f"received input_dim={expected_input_dim} and input shape "
                f"{tuple(input_batch.shape)}."
            )

    @staticmethod
    def __validate_positive_integer(name: str, value: int) -> None:
        if isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer, received {value!r}.")

    @staticmethod
    def __validate_weight_mixture_config(config: AdaptiveMixtureConfig) -> None:
        weight_configs = (
            VectorWeightsMixtureConfig,
            MatrixWeightsMixtureConfig,
            GeneratorWeightsMixtureConfig,
        )
        if not isinstance(config, weight_configs):
            raise TypeError(
                "weight_mixture_config must be a weight mixture config, "
                f"got {type(config).__name__}."
            )

    @staticmethod
    def __validate_bias_mixture_config(
        config: AdaptiveMixtureConfig | None,
    ) -> None:
        if config is None:
            return
        bias_configs = (MatrixBiasMixtureConfig, GeneratorBiasMixtureConfig)
        if not isinstance(config, bias_configs):
            raise TypeError(
                "bias_mixture_config must be None or a bias mixture config, "
                f"got {type(config).__name__}."
            )

    @staticmethod
    def __validate_router_and_sampler_configs(model: "ParametricLayer") -> None:
        from emperor.sampler.core.config import RouterConfig, SamplerConfig

        if not isinstance(model.router_config, RouterConfig):
            raise TypeError(
                "router_config must be a RouterConfig for ParametricLayer, "
                f"got {type(model.router_config).__name__}."
            )
        if not isinstance(model.sampler_config, SamplerConfig):
            raise TypeError(
                "sampler_config must be a SamplerConfig for ParametricLayer, "
                f"got {type(model.sampler_config).__name__}."
            )

    @staticmethod
    def __validate_sampler_matches_mixtures(model: "ParametricLayer") -> None:
        sampler_config = model.sampler_config
        ParametricLayerValidator.__validate_count_match(
            "sampler_config.top_k",
            sampler_config.top_k,
            "weight_mixture_config.top_k",
            model.weight_mixture_config.top_k,
        )
        ParametricLayerValidator.__validate_count_match(
            "sampler_config.num_experts",
            sampler_config.num_experts,
            "weight_mixture_config.num_experts",
            model.weight_mixture_config.num_experts,
        )
        if model.bias_mixture_config is None:
            return
        ParametricLayerValidator.__validate_count_match(
            "sampler_config.top_k",
            sampler_config.top_k,
            "bias_mixture_config.top_k",
            model.bias_mixture_config.top_k,
        )
        ParametricLayerValidator.__validate_count_match(
            "sampler_config.num_experts",
            sampler_config.num_experts,
            "bias_mixture_config.num_experts",
            model.bias_mixture_config.num_experts,
        )

    @staticmethod
    def __validate_count_match(
        left_name: str,
        left_value: int,
        right_name: str,
        right_value: int,
    ) -> None:
        if left_value != right_value:
            raise ValueError(
                f"{left_name} must match {right_name}, received "
                f"{left_value} and {right_value}."
            )

    @staticmethod
    def __validate_vector_shared_router(model: "ParametricLayer") -> None:
        if not isinstance(model.weight_mixture_config, VectorWeightsMixtureConfig):
            return
        if model.routing_initialization_mode != AdaptiveRouterOptions.SHARED_ROUTER:
            return
        raise ValueError(
            "VectorWeightsMixtureConfig does not support SHARED_ROUTER routing."
        )

    @staticmethod
    def __validate_adaptive_augmentation_config(model: "ParametricLayer") -> None:
        from emperor.augmentations.adaptive_parameters.config import (
            AdaptiveParameterAugmentationConfig,
        )

        if not isinstance(
            model.adaptive_augmentation_config, AdaptiveParameterAugmentationConfig
        ):
            raise TypeError(
                "adaptive_augmentation_config must be an "
                "AdaptiveParameterAugmentationConfig for ParametricLayer, "
                f"got {type(model.adaptive_augmentation_config).__name__}."
            )
        if model.bias_mixture_config is None:
            return
        if model.adaptive_augmentation_config.bias_config is not None:
            raise ValueError(
                "adaptive_augmentation_config.bias_config can only be used when "
                "bias_mixture_config is None."
            )


class ParametricHandlerValidator(ValidatorBase):
    @staticmethod
    def validate(model: "ParameterHandlerBase | ParametricLayerHandler") -> None:
        if hasattr(model, "weight_mixture_config"):
            ParametricHandlerValidator.__validate_parameter_handler(model)
            return
        ParametricHandlerValidator.__validate_layer_handler(model)

    @staticmethod
    def validate_state(state: LayerState) -> None:
        if not isinstance(state, LayerState):
            raise TypeError(
                "state must be a LayerState for ParametricLayerHandler, "
                f"got {type(state).__name__}."
            )
        if not isinstance(state.hidden, Tensor):
            raise TypeError(
                "state.hidden must be a Tensor for ParametricLayerHandler, "
                f"got {type(state.hidden).__name__}."
            )

    @staticmethod
    def __validate_parameter_handler(model: "ParameterHandlerBase") -> None:
        if not isinstance(model.cfg, ParametricLayerConfig):
            raise TypeError(
                "ParameterHandlerBase cfg must be ParametricLayerConfig, "
                f"got {type(model.cfg).__name__}."
            )
        if model.router_config is None or model.sampler_config is None:
            raise ValueError(
                "router_config and sampler_config are required for parametric routing."
            )
        if isinstance(model.weight_mixture_config, VectorWeightsMixtureConfig):
            if model.routing_initialization_mode == AdaptiveRouterOptions.SHARED_ROUTER:
                raise ValueError(
                    "VectorWeightsMixtureConfig does not support SHARED_ROUTER routing."
                )

    @staticmethod
    def __validate_layer_handler(model: "ParametricLayerHandler") -> None:
        if not isinstance(model.cfg, ParametricLayerHandlerConfig):
            raise TypeError(
                "ParametricLayerHandler cfg must be ParametricLayerHandlerConfig, "
                f"got {type(model.cfg).__name__}."
            )
        if not isinstance(model.layer_model_config, ParametricLayerConfig):
            raise TypeError(
                "ParametricLayerHandler.layer_model_config must be "
                f"ParametricLayerConfig, got {type(model.layer_model_config).__name__}."
            )
