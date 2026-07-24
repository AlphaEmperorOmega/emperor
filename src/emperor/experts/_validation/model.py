from typing import TYPE_CHECKING

from emperor._validation import ValidatorBase
from emperor.experts._options import RoutingInitializationMode

if TYPE_CHECKING:
    from emperor.experts._model import MixtureOfExpertsModel


class MixtureOfExpertsModelValidator(ValidatorBase):
    OPTIONAL_FIELDS = {"sampler_config"}

    @classmethod
    def validate(cls, model: "MixtureOfExpertsModel") -> None:
        cls.validate_cfg_type(model)
        cls.validate_required_fields(model.cfg)
        cls.validate_field_types(model.cfg)
        cls.validate_stack_config_type(model)
        cls.validate_stack_configuration(model)
        cls.validate_dimensions(model)
        cls.validate_stack_boundary_dimensions(model)
        cls.validate_expert_layer_config_type(model)
        cls.validate_expert_leaf_config_type(model)
        cls.validate_expert_leaf_configuration(model)
        cls.validate_top_k_coherence(model)
        cls.validate_routing_mode_coherence(model)
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
    def validate_stack_configuration(model: "MixtureOfExpertsModel") -> None:
        from types import SimpleNamespace

        from emperor.layers import LayerStack

        LayerStack.VALIDATOR.validate(SimpleNamespace(cfg=model.stack_config))

    @classmethod
    def validate_dimensions(cls, model: "MixtureOfExpertsModel") -> None:
        for name in ("input_dim", "output_dim", "top_k"):
            cls.validate_positive_integer(name, getattr(model, name))

    @staticmethod
    def validate_positive_integer(name: str, value: int) -> None:
        if isinstance(value, bool) or value <= 0:
            raise ValueError(
                f"Configuration Error: '{name}' must be a positive integer, "
                f"received {value!r}."
            )

    @staticmethod
    def validate_stack_boundary_dimensions(model: "MixtureOfExpertsModel") -> None:
        for dimension_name in ("input_dim", "output_dim"):
            model_dimension = getattr(model, dimension_name)
            stack_dimension = getattr(model.stack_config, dimension_name)
            if model_dimension != stack_dimension:
                raise ValueError(
                    f"Configuration Error: model {dimension_name} must match "
                    f"stack_config.{dimension_name}, received {dimension_name}="
                    f"{model_dimension} and stack_config.{dimension_name}="
                    f"{stack_dimension}"
                )

    @staticmethod
    def validate_expert_layer_config_type(model: "MixtureOfExpertsModel") -> None:
        from emperor.experts._config import MixtureOfExpertsLayerConfig

        layer_config = model.stack_config.layer_config
        if not isinstance(layer_config, MixtureOfExpertsLayerConfig):
            raise TypeError(
                "Configuration Error: 'stack_config.layer_config' must be of type "
                "MixtureOfExpertsLayerConfig, received type "
                f"{type(layer_config).__name__}"
            )

    @staticmethod
    def validate_expert_leaf_config_type(model: "MixtureOfExpertsModel") -> None:
        from emperor.experts._config import MixtureOfExpertsConfig

        leaf_config = model.stack_config.layer_config.layer_model_config
        if not isinstance(leaf_config, MixtureOfExpertsConfig):
            raise TypeError(
                "Configuration Error: "
                "'stack_config.layer_config.layer_model_config' must be of type "
                "MixtureOfExpertsConfig, received type "
                f"{type(leaf_config).__name__}"
            )

    @classmethod
    def validate_expert_leaf_configuration(
        cls,
        model: "MixtureOfExpertsModel",
    ) -> None:
        leaf_config = model.stack_config.layer_config.layer_model_config
        leaf_owner = leaf_config._registry_owner()
        for (
            input_dim,
            output_dim,
        ) in cls._resolve_layer_dimensions(model.stack_config):
            leaf_owner.VALIDATOR.validate_config(
                leaf_config,
                input_dim=input_dim,
                output_dim=output_dim,
            )

    @staticmethod
    def _resolve_layer_dimensions(stack_config) -> tuple[tuple[int, int], ...]:
        from emperor.layers import MirroredLayerStackConfig

        input_dim = stack_config.input_dim
        hidden_dim = stack_config.hidden_dim
        output_dim = stack_config.output_dim
        num_layers = stack_config.num_layers
        if isinstance(stack_config, MirroredLayerStackConfig):
            hidden_pairs = ((hidden_dim, hidden_dim),) * (num_layers - 1)
            return (
                (input_dim, hidden_dim),
                *hidden_pairs,
                *hidden_pairs,
                (hidden_dim, output_dim),
            )

        dimensions: list[tuple[int, int]] = []
        requires_input_projection = input_dim != hidden_dim and num_layers > 1
        boundary_layer_count = 1
        if requires_input_projection:
            dimensions.append((input_dim, hidden_dim))
            boundary_layer_count = 2
        hidden_layer_count = num_layers - boundary_layer_count
        dimensions.extend(((hidden_dim, hidden_dim),) * hidden_layer_count)
        output_layer_input_dim = hidden_dim if num_layers > 1 else input_dim
        dimensions.append((output_layer_input_dim, output_dim))
        return tuple(dimensions)

    @staticmethod
    def validate_top_k_coherence(model: "MixtureOfExpertsModel") -> None:
        leaf_config = model.stack_config.layer_config.layer_model_config
        if model.top_k != leaf_config.top_k:
            raise ValueError(
                "Configuration Error: model top_k must match the expert leaf top_k, "
                f"received top_k={model.top_k} and leaf top_k={leaf_config.top_k}"
            )

    @staticmethod
    def validate_routing_mode_coherence(model: "MixtureOfExpertsModel") -> None:
        leaf_config = model.stack_config.layer_config.layer_model_config
        model_owns_layer_routing = (
            model.routing_initialization_mode == RoutingInitializationMode.LAYER
        )
        leaf_owns_layer_routing = (
            leaf_config.routing_initialization_mode == RoutingInitializationMode.LAYER
        )
        if model_owns_layer_routing != leaf_owns_layer_routing:
            raise ValueError(
                "Configuration Error: model routing_initialization_mode must match "
                "the expert leaf routing_initialization_mode for LAYER ownership, "
                "received model mode "
                f"{model.routing_initialization_mode} and leaf mode "
                f"{leaf_config.routing_initialization_mode}"
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
        if model.top_k != model.sampler_config.top_k:
            raise ValueError(
                "Configuration Error: model top_k must match sampler_config.top_k, "
                f"received top_k={model.top_k} and "
                f"sampler_config.top_k={model.sampler_config.top_k}"
            )
        leaf_config = model.stack_config.layer_config.layer_model_config
        if leaf_config.num_experts != model.sampler_config.num_experts:
            raise ValueError(
                "Configuration Error: expert leaf num_experts must match "
                "sampler_config.num_experts, received leaf num_experts="
                f"{leaf_config.num_experts} and sampler_config.num_experts="
                f"{model.sampler_config.num_experts}"
            )
        model.sampler_config.validate_for_router_input_dim(model.input_dim)
