import math
from dataclasses import replace
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import torch
from torch import Tensor

from emperor._validation import ValidatorBase
from emperor.experts._options import (
    DroppedTokenOptions,
    ExpertWeightingPositionOptions,
    RoutingInitializationMode,
)

if TYPE_CHECKING:
    from emperor.experts._layers.mixture import MixtureOfExperts


class MixtureOfExpertsValidator(ValidatorBase):
    OPTIONAL_FIELDS = {"dropped_token_behavior", "sampler_config"}

    @classmethod
    def validate(cls, model: "MixtureOfExperts") -> None:
        cls.validate_config_type(model.cfg)
        cls.validate_required_fields(model.cfg)
        cls.validate_field_types(model.cfg)
        cls.validate_forward_reference_types(model)
        cls.validate_owned_routing_config_types(model)
        cls.validate_owned_routing_config_coherence(model)
        cls.validate_dimensions(model)
        cls.validate_capacity_factor_is_non_negative(model)
        cls.validate_capacity_factor_consistent_with_top_k(model)
        cls.validate_dims_match_when_capacity_enabled(model)

    @classmethod
    def validate_config(
        cls,
        cfg,
        *,
        input_dim: int,
        output_dim: int,
    ) -> None:
        """Validate a mixture config with its effective layer dimensions."""

        cls.validate_config_type(cfg)
        resolved_config = replace(
            cfg,
            input_dim=input_dim,
            output_dim=output_dim,
        )
        cls.validate(
            SimpleNamespace(
                cfg=resolved_config,
                input_dim=resolved_config.input_dim,
                output_dim=resolved_config.output_dim,
                expert_model_config=resolved_config.expert_model_config,
                top_k=resolved_config.top_k,
                num_experts=resolved_config.num_experts,
                capacity_factor=resolved_config.capacity_factor,
                weighting_position_option=(resolved_config.weighting_position_option),
                routing_initialization_mode=(
                    resolved_config.routing_initialization_mode
                ),
                sampler_config=resolved_config.sampler_config,
            )
        )

    @staticmethod
    def validate_config_type(cfg) -> None:
        from emperor.experts._config import MixtureOfExpertsConfig

        if not isinstance(cfg, MixtureOfExpertsConfig):
            raise TypeError(
                "Configuration Error: `cfg` must be of type "
                "MixtureOfExpertsConfig, received type "
                f"{type(cfg).__name__}"
            )

    @staticmethod
    def validate_overrides_type(overrides) -> None:
        from emperor.experts._config import MixtureOfExpertsConfig

        if overrides is not None and not isinstance(overrides, MixtureOfExpertsConfig):
            raise TypeError(
                "Configuration Error: `overrides` must be of type "
                "MixtureOfExpertsConfig or None, received type "
                f"{type(overrides).__name__}"
            )

    @staticmethod
    def validate_forward_reference_types(model: "MixtureOfExperts") -> None:
        from emperor.layers import LayerStackConfig, RecurrentLayerConfig

        if not isinstance(
            model.expert_model_config,
            (LayerStackConfig, RecurrentLayerConfig),
        ):
            raise TypeError(
                "Configuration Error: 'expert_model_config' must be of type "
                "LayerStackConfig or RecurrentLayerConfig, received type "
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
        from emperor.sampler import RouterConfig, SamplerConfig

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
    def validate_owned_routing_config_coherence(model: "MixtureOfExperts") -> None:
        if model.routing_initialization_mode != RoutingInitializationMode.LAYER:
            return
        if model.top_k != model.sampler_config.top_k:
            raise ValueError(
                "Configuration Error: mixture top_k must match "
                "sampler_config.top_k, received "
                f"top_k={model.top_k} and "
                f"sampler_config.top_k={model.sampler_config.top_k}"
            )
        if model.num_experts != model.sampler_config.num_experts:
            raise ValueError(
                "Configuration Error: mixture num_experts must match "
                "sampler_config.num_experts, received "
                f"num_experts={model.num_experts} and "
                f"sampler_config.num_experts={model.sampler_config.num_experts}"
            )
        model.sampler_config.validate_for_router_input_dim(model.input_dim)

    @classmethod
    def validate_dimensions(cls, model: "MixtureOfExperts") -> None:
        cls.validate_positive_integer("input_dim", model.input_dim)
        cls.validate_positive_integer("output_dim", model.output_dim)
        cls.validate_positive_integer("top_k", model.top_k)
        cls.validate_positive_integer("num_experts", model.num_experts)
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
        if not math.isfinite(model.capacity_factor):
            raise ValueError(
                "Configuration Error: 'capacity_factor' must be finite, received "
                f"{model.capacity_factor}"
            )
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

    @classmethod
    def validate_forward_inputs(
        cls,
        model: "MixtureOfExperts",
        input_batch: Tensor,
        probabilities: Tensor | None,
        indices: Tensor | None,
        skip_mask: Tensor | None = None,
    ) -> None:
        cls.validate_input_batch(model, input_batch)
        cls.validate_probabilities(model, input_batch, probabilities)
        cls.validate_indices(model, input_batch, indices)
        cls.validate_external_routing_inputs(model, probabilities, indices)
        cls.validate_skip_mask(input_batch, skip_mask, input_batch.shape[0])

    @classmethod
    def validate_reduce_forward_inputs(
        cls,
        model: "MixtureOfExperts",
        input_batch: Tensor,
        probabilities: Tensor | None,
        indices: Tensor | None,
        skip_mask: Tensor | None = None,
    ) -> None:
        cls.validate_input_batch(model, input_batch)
        cls.validate_external_routing_inputs(model, probabilities, indices)
        validated_probabilities = cast(Tensor, probabilities)
        cls.validate_tensor_is_vector_or_matrix(
            "probabilities", validated_probabilities
        )
        cls.validate_routing_width(
            "probabilities", validated_probabilities, model.top_k
        )
        cls.validate_probabilities_floating(validated_probabilities)
        cls.validate_probabilities_dtype(input_batch, validated_probabilities)
        cls.validate_probabilities_device(input_batch, validated_probabilities)
        cls.validate_probabilities_finite(validated_probabilities)
        cls.validate_probabilities_range(validated_probabilities)
        if indices is not None:
            cls.validate_tensor_is_vector_or_matrix("indices", indices)
            cls.validate_routing_width("indices", indices, model.top_k)
            cls.validate_indices_device(input_batch, indices)
            cls.validate_indices_dtype_and_range(model, indices)
            cls.validate_unique_expert_indices(model, indices)
        if validated_probabilities.numel() != input_batch.shape[0]:
            raise ValueError(
                "Input Error: 'probabilities' must contain one routing weight per "
                "flattened reduce input sample, received probabilities shape "
                f"{tuple(validated_probabilities.shape)} and input_batch shape "
                f"{tuple(input_batch.shape)}."
            )
        if indices is not None and indices.numel() != input_batch.shape[0]:
            raise ValueError(
                "Input Error: 'indices' must contain one expert id per flattened "
                "reduce input sample, received indices shape "
                f"{tuple(indices.shape)} and input_batch shape "
                f"{tuple(input_batch.shape)}."
            )
        cls.validate_skip_mask(
            input_batch,
            skip_mask,
            validated_probabilities.shape[0],
        )

    @staticmethod
    def validate_skip_mask(
        input_batch: Tensor,
        skip_mask,
        expected_batch_size: int,
    ) -> None:
        if skip_mask is None:
            return
        if not isinstance(skip_mask, Tensor):
            raise TypeError(
                "Input Error: 'skip_mask' must be a Tensor or None for "
                f"MixtureOfExperts, received {type(skip_mask).__name__}."
            )
        if skip_mask.dim() != 2:
            raise ValueError(
                "Input Error: 'skip_mask' must have shape (batch_size, 1) for "
                "MixtureOfExperts, received a "
                f"{skip_mask.dim()}D tensor with shape {tuple(skip_mask.shape)}."
            )
        if skip_mask.shape[1] != 1:
            raise ValueError(
                "Input Error: 'skip_mask' feature dimension must be 1 for "
                "MixtureOfExperts, received shape "
                f"{tuple(skip_mask.shape)}."
            )
        if skip_mask.shape[0] != expected_batch_size:
            raise ValueError(
                "Input Error: 'skip_mask' batch dimension must match the expected "
                "routing batch size for MixtureOfExperts, received skip_mask shape "
                f"{tuple(skip_mask.shape)} and expected batch size "
                f"{expected_batch_size}."
            )
        if skip_mask.device != input_batch.device:
            raise ValueError(
                "Input Error: 'skip_mask' must be on the same device as "
                "input_batch for MixtureOfExperts, received "
                f"skip_mask device {skip_mask.device} and input_batch device "
                f"{input_batch.device}."
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
                f"input_dim={model.input_dim} and input shape "
                f"{tuple(input_batch.shape)}."
            )
        if input_batch.shape[0] == 0:
            raise ValueError(
                "Input Error: MixtureOfExperts requires at least one input sample, "
                f"received input shape {tuple(input_batch.shape)}."
            )

    @classmethod
    def validate_probabilities(
        cls,
        model: "MixtureOfExperts",
        input_batch: Tensor,
        probabilities,
    ) -> None:
        if probabilities is None:
            return
        cls.validate_tensor_is_vector_or_matrix("probabilities", probabilities)
        cls.validate_batch_dimension("probabilities", probabilities, input_batch)
        cls.validate_routing_width("probabilities", probabilities, model.top_k)
        cls.validate_probabilities_floating(probabilities)
        cls.validate_probabilities_dtype(input_batch, probabilities)
        cls.validate_probabilities_device(input_batch, probabilities)
        cls.validate_probabilities_finite(probabilities)
        cls.validate_probabilities_range(probabilities)

    @staticmethod
    def validate_probabilities_floating(probabilities: Tensor) -> None:
        if not torch.is_floating_point(probabilities):
            raise TypeError(
                "Input Error: 'probabilities' must have a floating-point dtype "
                f"for MixtureOfExperts, received dtype {probabilities.dtype}."
            )

    @staticmethod
    def validate_probabilities_dtype(
        input_batch: Tensor,
        probabilities: Tensor,
    ) -> None:
        if probabilities.dtype != input_batch.dtype:
            raise ValueError(
                "Input Error: 'probabilities' dtype must match input_batch dtype "
                f"for MixtureOfExperts, received probabilities dtype "
                f"{probabilities.dtype} and input_batch dtype {input_batch.dtype}."
            )

    @staticmethod
    def validate_probabilities_device(
        input_batch: Tensor,
        probabilities: Tensor,
    ) -> None:
        if probabilities.device != input_batch.device:
            raise ValueError(
                "Input Error: 'probabilities' device must match input_batch device "
                f"for MixtureOfExperts, received probabilities device "
                f"{probabilities.device} and input_batch device {input_batch.device}."
            )

    @staticmethod
    def validate_probabilities_finite(probabilities: Tensor) -> None:
        if not torch.isfinite(probabilities).all().item():
            raise ValueError(
                "Input Error: 'probabilities' values must all be finite for "
                "MixtureOfExperts."
            )

    @staticmethod
    def validate_probabilities_range(probabilities: Tensor) -> None:
        values_are_out_of_range = torch.logical_or(
            probabilities < 0.0,
            probabilities > 1.0,
        ).any()
        if values_are_out_of_range.item():
            raise ValueError(
                "Input Error: 'probabilities' values must be in the closed interval "
                "[0, 1] for MixtureOfExperts."
            )

    @classmethod
    def validate_indices(
        cls,
        model: "MixtureOfExperts",
        input_batch: Tensor,
        indices,
    ) -> None:
        if indices is None:
            return
        cls.validate_tensor_is_vector_or_matrix("indices", indices)
        cls.validate_batch_dimension("indices", indices, input_batch)
        cls.validate_routing_width("indices", indices, model.top_k)
        cls.validate_indices_device(input_batch, indices)
        cls.validate_indices_dtype_and_range(model, indices)
        cls.validate_unique_expert_indices(model, indices)

    @staticmethod
    def validate_indices_device(input_batch: Tensor, indices: Tensor) -> None:
        if indices.device != input_batch.device:
            raise ValueError(
                "Input Error: 'indices' device must match input_batch device for "
                "MixtureOfExperts, received indices device "
                f"{indices.device} and input_batch device {input_batch.device}."
            )

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
    def validate_unique_expert_indices(
        model: "MixtureOfExperts",
        indices: Tensor,
    ) -> None:
        if model.top_k == 1 or model.top_k == model.num_experts:
            return
        sorted_indices = indices.sort(dim=-1).values
        duplicate_rows = (
            (sorted_indices[:, 1:] == sorted_indices[:, :-1])
            .any(dim=-1)
            .nonzero()
            .flatten()
        )
        if duplicate_rows.numel() > 0:
            raise ValueError(
                "Input Error: 'indices' must contain distinct expert ids for each "
                "input sample in sparse MixtureOfExperts routing, received duplicate "
                f"expert ids in sample rows {duplicate_rows.tolist()}."
            )

    @classmethod
    def validate_external_routing_inputs(
        cls,
        model: "MixtureOfExperts",
        probabilities: Tensor | None,
        indices: Tensor | None,
    ) -> None:
        if model.routing_initialization_mode == RoutingInitializationMode.LAYER:
            cls.validate_external_probabilities_are_not_given(probabilities, indices)
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
