import torch

from torch import Tensor

from emperor.base.validator import ValidatorBase

from emperor.experts.core.options import (
    DroppedTokenOptions,
    ExpertWeightingPositionOptions,
    RoutingInitializationMode,
)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from emperor.experts.core.layers import MixtureOfExperts
    from emperor.experts.model import MixtureOfExpertsModel


class MixtureOfExpertsValidator(ValidatorBase):
    OPTIONAL_FIELDS = {"dropped_token_behavior", "sampler_config"}

    @staticmethod
    def validate(model: "MixtureOfExperts") -> None:
        MixtureOfExpertsValidator.validate_required_fields(model.cfg)
        MixtureOfExpertsValidator.validate_field_types(model.cfg)
        MixtureOfExpertsValidator.validate_forward_reference_types(model)
        MixtureOfExpertsValidator.validate_owned_routing_config_types(model)
        MixtureOfExpertsValidator.validate_dimensions(model)
        MixtureOfExpertsValidator.validate_capacity_factor_is_non_negative(model)
        MixtureOfExpertsValidator.validate_capacity_factor_consistent_with_top_k(model)
        MixtureOfExpertsValidator.validate_dims_match_when_capacity_enabled(model)

    @staticmethod
    def validate_forward_reference_types(model: "MixtureOfExperts") -> None:
        from emperor.base.layer import LayerStackConfig

        if not isinstance(model.expert_model_config, LayerStackConfig):
            raise TypeError(
                "Configuration Error: 'expert_model_config' must be of type "
                "LayerStackConfig, received type "
                f"{type(model.expert_model_config).__name__}"
            )
        if not isinstance(
            model.weighting_position_option, ExpertWeightingPositionOptions
        ):
            raise TypeError(
                "Configuration Error: 'weighting_position_option' must be of type "
                "ExpertWeightingPositionOptions, received type "
                f"{type(model.weighting_position_option).__name__}"
            )
        if not isinstance(model.routing_initialization_mode, RoutingInitializationMode):
            raise TypeError(
                "Configuration Error: 'routing_initialization_mode' must be of type "
                "RoutingInitializationMode, received type "
                f"{type(model.routing_initialization_mode).__name__}"
            )
        if model.cfg.dropped_token_behavior is None:
            return
        if not isinstance(model.cfg.dropped_token_behavior, DroppedTokenOptions):
            raise TypeError(
                "Configuration Error: 'dropped_token_behavior' must be of type "
                "DroppedTokenOptions, received type "
                f"{type(model.cfg.dropped_token_behavior).__name__}"
            )

    @staticmethod
    def validate_owned_routing_config_types(model: "MixtureOfExperts") -> None:
        from emperor.sampler.core.config import RouterConfig, SamplerConfig

        if model.routing_initialization_mode != RoutingInitializationMode.LAYER:
            return
        if not isinstance(model.sampler_config, SamplerConfig):
            raise TypeError(
                "Configuration Error: 'sampler_config' must be of type "
                "SamplerConfig when 'routing_initialization_mode' is LAYER, "
                f"received type {type(model.sampler_config).__name__}"
            )
        if not isinstance(model.sampler_config.router_config, RouterConfig):
            raise TypeError(
                "Configuration Error: 'sampler_config.router_config' must be of "
                "type RouterConfig when 'routing_initialization_mode' is LAYER, "
                f"received type {type(model.sampler_config.router_config).__name__}"
            )

    @staticmethod
    def validate_dimensions(model: "MixtureOfExperts") -> None:
        MixtureOfExpertsValidator.validate_positive_integer(
            "input_dim", model.input_dim
        )
        MixtureOfExpertsValidator.validate_positive_integer(
            "output_dim", model.output_dim
        )
        MixtureOfExpertsValidator.validate_positive_integer("top_k", model.top_k)
        MixtureOfExpertsValidator.validate_positive_integer(
            "num_experts", model.num_experts
        )
        if model.top_k > model.num_experts:
            raise ValueError(
                "Configuration Error: 'top_k' cannot exceed 'num_experts' for "
                "MixtureOfExperts, received "
                f"top_k={model.top_k}, num_experts={model.num_experts}."
            )

    @staticmethod
    def validate_positive_integer(name: str, value: int) -> None:
        if isinstance(value, bool) or value <= 0:
            raise ValueError(
                f"Configuration Error: '{name}' must be a positive integer, "
                f"received {value!r}."
            )

    @staticmethod
    def validate_capacity_factor_is_non_negative(model: "MixtureOfExperts") -> None:
        if model.capacity_factor < 0.0:
            raise ValueError(
                "Configuration Error: 'capacity_factor' must be >= 0.0, received "
                f"{model.capacity_factor}"
            )

    @staticmethod
    def validate_capacity_factor_consistent_with_top_k(
        model: "MixtureOfExperts",
    ) -> None:
        if model.capacity_factor > 0.0 and model.top_k == model.num_experts:
            raise ValueError(
                "Configuration Error: 'capacity_factor' cannot be > 0.0 when "
                "'top_k' equals 'num_experts'. When top_k == num_experts all tokens "
                "pass through all experts unconditionally, so capacity limiting has "
                "no effect and dropped tokens cannot occur."
            )

    @staticmethod
    def validate_dims_match_when_capacity_enabled(model: "MixtureOfExperts") -> None:
        if model.capacity_factor > 0.0 and model.input_dim != model.output_dim:
            raise ValueError(
                "Configuration Error: 'input_dim' must equal 'output_dim' when "
                "'capacity_factor' > 0.0, because dropped tokens pass through as "
                "identity and must match the expert output shape. Got "
                f"input_dim={model.input_dim}, output_dim={model.output_dim}"
            )

    @staticmethod
    def validate_sampler_is_initialized(model: "MixtureOfExperts") -> None:
        if model.routing_initialization_mode != RoutingInitializationMode.LAYER:
            raise ValueError(
                "Invalid configuration: `routing_initialization_mode` must be "
                "`RoutingInitializationMode.LAYER` to initialize the `RouterModel` and "
                "`SamplerModel` when `indices` are not provided. Current option: "
                f"{model.routing_initialization_mode}"
            )

    @staticmethod
    def validate_external_probabilities_are_not_given(
        probabilities: Tensor | None,
        indices: Tensor | None,
    ) -> None:
        if indices is not None or probabilities is not None:
            raise ValueError(
                "`probabilities` and `indices` must both be None when the "
                "MixtureOfExperts layer owns routing. Providing external routing "
                "inputs where they are not expected is not allowed."
            )

    @staticmethod
    def validate_probabilities_exist(probabilities: Tensor | None) -> None:
        if probabilities is None:
            raise ValueError(
                "Missing input: `probabilities` must be supplied when `indices` "
                "are used to ensure accurate weighting and processing of inputs."
            )

    @staticmethod
    def validate_router_config_exists(model: "MixtureOfExperts") -> None:
        if model.sampler_config.router_config is None:
            raise ValueError(
                "Configuration Error: `sampler_config.router_config` must be "
                "defined to properly initialize and utilize the router model in "
                "the mixture of experts layer."
            )

    @staticmethod
    def validate_sampler_config_exists(model: "MixtureOfExperts") -> None:
        if model.sampler_config is None:
            raise ValueError(
                "Configuration Error: `sampler_config` must be defined to "
                "properly initialize and utilize the sampler model in the mixture "
                "of experts layer."
            )

    @staticmethod
    def validate_forward_inputs(
        model: "MixtureOfExperts",
        input_batch: Tensor,
        probabilities: Tensor | None,
        indices: Tensor | None,
    ) -> None:
        MixtureOfExpertsValidator.validate_input_batch(model, input_batch)
        MixtureOfExpertsValidator.validate_probabilities(
            model, input_batch, probabilities
        )
        MixtureOfExpertsValidator.validate_indices(model, input_batch, indices)
        MixtureOfExpertsValidator.validate_external_routing_inputs(
            model, probabilities, indices
        )

    @staticmethod
    def validate_reduce_forward_inputs(
        model: "MixtureOfExperts",
        input_batch: Tensor,
        probabilities: Tensor | None,
        indices: Tensor | None,
    ) -> None:
        MixtureOfExpertsValidator.validate_input_batch(model, input_batch)
        if probabilities is not None:
            MixtureOfExpertsValidator.validate_tensor_is_vector_or_matrix(
                "probabilities", probabilities
            )
            MixtureOfExpertsValidator.validate_routing_width(
                "probabilities", probabilities, model.top_k
            )
        if indices is not None:
            MixtureOfExpertsValidator.validate_tensor_is_vector_or_matrix(
                "indices", indices
            )
            MixtureOfExpertsValidator.validate_routing_width(
                "indices", indices, model.top_k
            )
            MixtureOfExpertsValidator.validate_indices_dtype_and_range(model, indices)
        MixtureOfExpertsValidator.validate_external_routing_inputs(
            model, probabilities, indices
        )
        if probabilities is not None and probabilities.numel() != input_batch.shape[0]:
            raise ValueError(
                "Input Error: 'probabilities' must contain one routing weight per "
                "flattened reduce input sample, received probabilities shape "
                f"{tuple(probabilities.shape)} and input_batch shape "
                f"{tuple(input_batch.shape)}."
            )
        if indices is not None and indices.numel() != input_batch.shape[0]:
            raise ValueError(
                "Input Error: 'indices' must contain one expert id per flattened "
                "reduce input sample, received indices shape "
                f"{tuple(indices.shape)} and input_batch shape "
                f"{tuple(input_batch.shape)}."
            )

    @staticmethod
    def validate_input_batch(model: "MixtureOfExperts", input_batch) -> None:
        if not isinstance(input_batch, Tensor):
            raise TypeError(
                "Input Error: 'input_batch' must be a Tensor for MixtureOfExperts, "
                f"received {type(input_batch).__name__}."
            )
        if input_batch.dim() != 2:
            raise ValueError(
                "Input Error: MixtureOfExperts expects a 2D input tensor "
                "(batch_size, input_dim), received a "
                f"{input_batch.dim()}D tensor with shape {tuple(input_batch.shape)}."
            )
        if input_batch.shape[-1] != model.input_dim:
            raise ValueError(
                "Input Error: input feature dimension must match 'input_dim' for "
                "MixtureOfExperts, received "
                f"input_dim={model.input_dim} and input shape {tuple(input_batch.shape)}."
            )

    @staticmethod
    def validate_probabilities(
        model: "MixtureOfExperts",
        input_batch: Tensor,
        probabilities,
    ) -> None:
        if probabilities is None:
            return
        MixtureOfExpertsValidator.validate_tensor_is_vector_or_matrix(
            "probabilities", probabilities
        )
        MixtureOfExpertsValidator.validate_batch_dimension(
            "probabilities", probabilities, input_batch
        )
        MixtureOfExpertsValidator.validate_routing_width(
            "probabilities", probabilities, model.top_k
        )

    @staticmethod
    def validate_indices(
        model: "MixtureOfExperts",
        input_batch: Tensor,
        indices,
    ) -> None:
        if indices is None:
            return
        MixtureOfExpertsValidator.validate_tensor_is_vector_or_matrix(
            "indices", indices
        )
        MixtureOfExpertsValidator.validate_batch_dimension(
            "indices", indices, input_batch
        )
        MixtureOfExpertsValidator.validate_routing_width(
            "indices", indices, model.top_k
        )
        MixtureOfExpertsValidator.validate_indices_dtype_and_range(model, indices)

    @staticmethod
    def validate_indices_dtype_and_range(
        model: "MixtureOfExperts",
        indices: Tensor,
    ) -> None:
        integer_dtypes = {
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        }
        if indices.dtype not in integer_dtypes:
            raise TypeError(
                "Input Error: 'indices' must contain integer expert ids for "
                f"MixtureOfExperts, received dtype {indices.dtype}."
            )
        if indices.numel() > 0 and (
            indices.min() < 0 or indices.max() >= model.num_experts
        ):
            raise ValueError(
                "Input Error: 'indices' values must be in [0, num_experts), "
                f"received num_experts={model.num_experts} and indices range "
                f"[{indices.min().item()}, {indices.max().item()}]."
            )

    @staticmethod
    def validate_external_routing_inputs(
        model: "MixtureOfExperts",
        probabilities: Tensor | None,
        indices: Tensor | None,
    ) -> None:
        if model.routing_initialization_mode == RoutingInitializationMode.LAYER:
            MixtureOfExpertsValidator.validate_external_probabilities_are_not_given(
                probabilities, indices
            )
            return
        if probabilities is None:
            raise ValueError(
                "Missing input: 'probabilities' must be supplied when external "
                "routing is used by MixtureOfExperts."
            )
        if model.top_k != model.num_experts and indices is None:
            raise ValueError(
                "Missing input: 'indices' must be supplied when external sparse "
                "routing is used by MixtureOfExperts."
            )
        if model.top_k == model.num_experts and indices is not None:
            raise ValueError(
                "Input Error: 'indices' must be None when 'top_k' equals "
                "'num_experts' for dense MixtureOfExperts routing."
            )

    @staticmethod
    def validate_tensor_is_vector_or_matrix(name: str | Tensor, tensor=None) -> None:
        if tensor is None:
            tensor = name
            name = "tensor"
        if not isinstance(tensor, Tensor):
            raise TypeError(
                f"Input Error: '{name}' must be a Tensor for MixtureOfExperts, "
                f"received {type(tensor).__name__}."
            )
        if tensor.dim() not in (1, 2):
            raise ValueError(
                f"Input Error: '{name}' must be a 1D or 2D tensor for "
                f"MixtureOfExperts, received a {tensor.dim()}D tensor with shape "
                f"{tuple(tensor.shape)}."
            )

    @staticmethod
    def validate_batch_dimension(
        name: str, tensor: Tensor, input_batch: Tensor
    ) -> None:
        if tensor.shape[0] != input_batch.shape[0]:
            raise ValueError(
                f"Input Error: '{name}' batch dimension must match input_batch, "
                f"received {name} shape {tuple(tensor.shape)} and input_batch shape "
                f"{tuple(input_batch.shape)}."
            )

    @staticmethod
    def validate_routing_width(name: str, tensor: Tensor, top_k: int) -> None:
        if top_k == 1:
            valid_shape = tensor.dim() == 1 or tensor.shape[-1] == 1
        else:
            valid_shape = tensor.dim() == 2 and tensor.shape[-1] == top_k
        if not valid_shape:
            raise ValueError(
                f"Input Error: '{name}' routing dimension must match top_k, "
                f"received top_k={top_k} and {name} shape {tuple(tensor.shape)}."
            )


class MixtureOfExpertsModelValidator(ValidatorBase):
    @staticmethod
    def validate(model: "MixtureOfExpertsModel") -> None:
        MixtureOfExpertsModelValidator.validate_main_config_is_derived(model)

    @staticmethod
    def validate_main_config_is_derived(model: "MixtureOfExpertsModel") -> None:
        from emperor.experts.core.config import MixtureOfExpertsConfig

        if not isinstance(model.main_cfg, MixtureOfExpertsConfig):
            raise ValueError(
                "Invalid configuration: `main_cfg` must be a "
                "`MixtureOfExpertsConfig` instance."
            )

