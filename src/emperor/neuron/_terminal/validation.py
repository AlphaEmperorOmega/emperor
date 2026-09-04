from types import SimpleNamespace
from typing import TYPE_CHECKING

from torch import Tensor

from emperor._validation import ValidatorBase
from emperor.neuron._validation.common import NeuronValidationMixin

if TYPE_CHECKING:
    from emperor.neuron._config import TerminalRoutingTreeConfig
    from emperor.neuron._terminal.core import Terminal
    from emperor.neuron._terminal.routing import RoutingTreeDelegate
    from emperor.neuron._terminal.routing_tree_topology import RoutingTreePlan
    from emperor.sampler import SamplerConfig


class RoutingTreeDelegateValidator(ValidatorBase):
    @staticmethod
    def validate_routing_tree_config(
        routing_tree_config: "TerminalRoutingTreeConfig | None",
    ) -> None:
        """Validate the configuration required to construct a routing tree."""
        if routing_tree_config is None:
            raise ValueError("RoutingTreeDelegate requires routing_tree_config.")

    @staticmethod
    def validate_forward_inputs(
        model: "RoutingTreeDelegate",
        input_matrix: object,
        skip_mask: Tensor | None,
    ) -> None:
        """Validate one routing-tree sampling request."""
        if not isinstance(input_matrix, Tensor):
            raise TypeError(
                "input_matrix must be a Tensor, "
                f"received {type(input_matrix).__name__}."
            )

        if input_matrix.dim() != 2 or input_matrix.shape[-1] != model.input_dim:
            raise ValueError(
                "Terminal routing tree input must have shape "
                f"(batch_size, {model.input_dim}), received "
                f"{tuple(input_matrix.shape)}."
            )

        if skip_mask is not None:
            raise ValueError(
                "Terminal routing trees do not accept a shared skip_mask; each "
                "conditionally executed node manages its own sampler state."
            )

    @classmethod
    def validate_routing_tree_plan(
        cls,
        model: "RoutingTreeDelegate",
        routing_tree_plan: "RoutingTreePlan",
    ) -> None:
        cls.validate_plan_sampler_configs(
            input_dim=model.input_dim,
            leaf_sampler_config=model.leaf_sampler_config,
            direction_sampler_config=model.direction_sampler_config,
            routing_tree_plan=routing_tree_plan,
        )

    @classmethod
    def validate_plan_sampler_configs(
        cls,
        *,
        input_dim: int,
        leaf_sampler_config: "SamplerConfig",
        direction_sampler_config: "SamplerConfig",
        routing_tree_plan: "RoutingTreePlan",
    ) -> None:
        from emperor.neuron._terminal.routing.node import RoutingTreeNode
        from emperor.sampler import RouterConfig

        for template_name, sampler_template in (
            ("sampler_config", leaf_sampler_config),
            (
                "routing_tree_config.direction_sampler_config",
                direction_sampler_config,
            ),
        ):
            if not isinstance(sampler_template.router_config, RouterConfig):
                raise ValueError(
                    f"{template_name}.router_config must be a RouterConfig for "
                    "Terminal routing trees; routerless direct-logit sampling is "
                    "available only in flat mode."
                )

        for node in routing_tree_plan.walk():
            if node.is_leaf:
                if len(node.connection_indices) < routing_tree_plan.leaf_top_k:
                    raise ValueError(
                        "Terminal routing tree leaf "
                        f"{cls._format_tree_path(node.path)} contains "
                        f"{len(node.connection_indices)} connections, fewer than "
                        "sampler_config.top_k="
                        f"{routing_tree_plan.leaf_top_k}."
                    )
                template = leaf_sampler_config
                num_experts = len(node.connection_indices)
                top_k = routing_tree_plan.leaf_top_k
            else:
                num_children = len(node.children)
                level_top_k = routing_tree_plan.direction_top_k[node.level]
                if num_children < 2:
                    raise ValueError(
                        "Terminal routing tree internal node "
                        f"{cls._format_tree_path(node.path)} must contain at least "
                        f"two nonempty spatial regions, received {num_children}."
                    )
                if num_children < level_top_k:
                    raise ValueError(
                        "Terminal routing tree internal node "
                        f"{cls._format_tree_path(node.path)} contains "
                        f"{num_children} nonempty regions, fewer than "
                        f"direction_top_k[{node.level}]={level_top_k}."
                    )
                template = direction_sampler_config
                num_experts = num_children
                top_k = level_top_k

            derived_config = RoutingTreeNode.derive_sampler_config(
                template,
                input_dim=input_dim,
                num_experts=num_experts,
                top_k=top_k,
            )
            derived_config.validate_for_router_input_dim(input_dim)

    @staticmethod
    def _format_tree_path(path: tuple[int, ...]) -> str:
        if not path:
            return "<root>"
        return ".".join(str(branch) for branch in path)


class Validator(ValidatorBase, NeuronValidationMixin):
    OPTIONAL_FIELDS = {"routing_tree_config"}

    @classmethod
    def validate_config_fields(cls, cfg) -> None:
        cls.validate_required_fields(cfg)
        cls.validate_field_types(cfg)
        cls.validate_connection_shape(cfg)
        cls.validate_routing_tree_config_fields(cfg)

    @classmethod
    def validate_routing_tree_config_fields(cls, cfg) -> None:
        from emperor.neuron._config import TerminalRoutingTreeConfig
        from emperor.neuron._options import TerminalRoutingTreeDepthOptions
        from emperor.sampler import SamplerConfig

        routing_tree_config = cfg.routing_tree_config
        if routing_tree_config is None:
            return
        if not isinstance(routing_tree_config, TerminalRoutingTreeConfig):
            raise TypeError(
                "routing_tree_config must be a TerminalRoutingTreeConfig for "
                f"TerminalConfig, got {type(routing_tree_config).__name__}."
            )
        if not isinstance(
            routing_tree_config.depth,
            TerminalRoutingTreeDepthOptions,
        ):
            raise TypeError(
                "routing_tree_config.depth must be a "
                "TerminalRoutingTreeDepthOptions, got "
                f"{type(routing_tree_config.depth).__name__}."
            )

        expected_direction_levels = routing_tree_config.depth.value - 1
        cls._validate_direction_tuple(
            "routing_tree_config.direction_branch_counts",
            routing_tree_config.direction_branch_counts,
            expected_direction_levels,
        )
        cls._validate_direction_tuple(
            "routing_tree_config.direction_top_k",
            routing_tree_config.direction_top_k,
            expected_direction_levels,
        )
        direction_sampler_config = routing_tree_config.direction_sampler_config
        if direction_sampler_config is not None and not isinstance(
            direction_sampler_config,
            SamplerConfig,
        ):
            raise TypeError(
                "routing_tree_config.direction_sampler_config must be a "
                f"SamplerConfig or None, got {type(direction_sampler_config).__name__}."
            )

        leaf_top_k = cfg.sampler_config.top_k
        if (
            not isinstance(leaf_top_k, int)
            or isinstance(leaf_top_k, bool)
            or leaf_top_k <= 0
        ):
            raise ValueError(
                "sampler_config.top_k must be a positive integer for a Terminal "
                f"routing tree, received {leaf_top_k!r}."
            )

    @staticmethod
    def _validate_direction_tuple(
        field_name: str,
        value,
        expected_length: int,
    ) -> None:
        if not isinstance(value, tuple):
            raise TypeError(
                f"{field_name} must be a tuple, got {type(value).__name__}."
            )
        if len(value) != expected_length:
            raise ValueError(
                f"{field_name} must contain depth - 1 ({expected_length}) entries, "
                f"received {len(value)}."
            )
        for level, item in enumerate(value, start=1):
            if not isinstance(item, int) or isinstance(item, bool) or item <= 0:
                raise ValueError(
                    f"{field_name}[{level - 1}] must be a positive integer, "
                    f"received {item!r}."
                )

    @staticmethod
    def validate_connection_shape(cfg) -> None:
        from emperor.neuron._options import TerminalConnectionShapeOptions

        if not isinstance(cfg.connection_shape, TerminalConnectionShapeOptions):
            raise TypeError(
                "connection_shape must be a TerminalConnectionShapeOptions "
                f"for TerminalConfig, got {type(cfg.connection_shape).__name__}."
            )

    @classmethod
    def validate(cls, model: "Terminal") -> None:
        cls.validate_config_fields(model.cfg)
        cls.validate_positive_integer("input_dim", model.input_dim)
        cls.validate_integer("x_axis_position", model.x_axis_position)
        cls.validate_integer("y_axis_position", model.y_axis_position)
        cls.validate_integer("z_axis_position", model.z_axis_position)
        cls.validate_sampler_config(model)

    @classmethod
    def validate_config_composition(cls, cfg) -> None:
        """Validate Terminal composition without constructing trainable modules."""

        from emperor.neuron._terminal.connection_topology import (
            TargetCoordinateBuilder,
        )

        cls.validate_config_fields(cfg)
        neuron_connections = TargetCoordinateBuilder(cfg).build()
        terminal_validation_target = SimpleNamespace(
            cfg=cfg,
            input_dim=cfg.input_dim,
            x_axis_position=cfg.x_axis_position,
            y_axis_position=cfg.y_axis_position,
            z_axis_position=cfg.z_axis_position,
            xy_axis_range=cfg.xy_axis_range.value,
            z_axis_range=cfg.z_axis_range.value,
            sampler_config=cfg.sampler_config,
            routing_tree_config=cfg.routing_tree_config,
            total_neuron_connections=int(neuron_connections.shape[0]),
        )
        cls.validate(terminal_validation_target)
        if cfg.routing_tree_config is not None:
            cls.validate_routing_tree_composition(cfg, neuron_connections)
            return
        cfg.sampler_config.validate_for_router_input_dim(cfg.input_dim)

    @staticmethod
    def validate_routing_tree_composition(cfg, neuron_connections: Tensor) -> None:
        from emperor.neuron._terminal.routing_tree_topology import RoutingTreeCompiler

        routing_tree_config = cfg.routing_tree_config
        leaf_sampler_config = cfg.sampler_config
        direction_sampler_config = (
            routing_tree_config.direction_sampler_config or leaf_sampler_config
        )
        routing_tree_plan = RoutingTreeCompiler(
            neuron_connections=neuron_connections,
            routing_tree_config=routing_tree_config,
            leaf_top_k=leaf_sampler_config.top_k,
        ).compile()
        RoutingTreeDelegateValidator.validate_plan_sampler_configs(
            input_dim=cfg.input_dim,
            leaf_sampler_config=leaf_sampler_config,
            direction_sampler_config=direction_sampler_config,
            routing_tree_plan=routing_tree_plan,
        )

    @classmethod
    def validate_sampler_config(cls, model: "Terminal") -> None:
        from emperor.sampler import RouterConfig

        sampler_config = model.sampler_config
        if model.routing_tree_config is not None:
            return

        cls.validate_positive_integer(
            "sampler_config.num_experts",
            sampler_config.num_experts,
        )
        if sampler_config.num_experts != model.total_neuron_connections:
            raise ValueError(
                "sampler_config.num_experts must equal Terminal "
                "total_neuron_connections, received "
                f"num_experts={sampler_config.num_experts} and "
                f"total_neuron_connections={model.total_neuron_connections}."
            )

        router_config = sampler_config.router_config
        if router_config is None:
            cls.validate_logits_only_input_dim(model)
            return
        if not isinstance(router_config, RouterConfig):
            raise TypeError(
                "sampler_config.router_config must be a RouterConfig for Terminal, "
                f"got {type(router_config).__name__}."
            )
        cls.validate_positive_integer(
            "sampler_config.router_config.num_experts",
            router_config.num_experts,
        )
        if router_config.num_experts != model.total_neuron_connections:
            raise ValueError(
                "sampler_config.router_config.num_experts must equal Terminal "
                "total_neuron_connections, received "
                f"num_experts={router_config.num_experts} and "
                f"total_neuron_connections={model.total_neuron_connections}."
            )

    @staticmethod
    def validate_logits_only_input_dim(model: "Terminal") -> None:
        if model.input_dim == model.total_neuron_connections:
            return
        raise ValueError(
            "sampler_config.router_config is required when Terminal input_dim "
            "does not equal total_neuron_connections, received "
            f"input_dim={model.input_dim} and "
            f"total_neuron_connections={model.total_neuron_connections}."
        )

    @classmethod
    def validate_forward_input(cls, model: "Terminal", input: Tensor) -> None:
        cls.validate_tensor_rank("Terminal input", input, 2)
        if input.shape[-1] != model.input_dim:
            raise ValueError(
                "Terminal input feature dimension must match input_dim, "
                f"received input_dim={model.input_dim} and input shape "
                f"{tuple(input.shape)}."
            )
