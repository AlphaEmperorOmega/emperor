from __future__ import annotations

from dataclasses import fields

from emperor.config import ConfigBase


def _config_classes():
    from emperor.layers._config import LayerConfig, LayerStackConfig

    return LayerConfig, LayerStackConfig


def _gate_config_class():
    from emperor.layers._config import GateConfig

    return GateConfig


def _residual_config_class():
    from emperor.layers._config import ResidualConfig

    return ResidualConfig


def _linear_layer_config_class():
    from emperor.linears import LinearLayerConfig

    return LinearLayerConfig


def _gate_option_field_path(owner_name: str | None = None) -> str:
    return f"{owner_name}.option" if owner_name is not None else "gate_config.option"


_HALTING_CONFIG_FIELDS = (
    "input_dim",
    "threshold",
    "dropout_probability",
    "hidden_state_mode",
    "halting_gate_config",
)
_MEMORY_CONFIG_FIELDS = (
    "input_dim",
    "output_dim",
    "memory_position_option",
    "test_time_training_learning_rate",
    "test_time_training_num_inner_steps",
    "model_config",
)


def _validate_halting_lifecycle_owner(
    halting_config,
    *,
    field_name: str,
    owner_name: str,
) -> None:
    try:
        owner = halting_config._registry_owner()
    except NotImplementedError as exc:
        raise ValueError(
            f"{field_name} must be a concrete halting config for {owner_name}"
        ) from exc

    if isinstance(owner, type):
        supports_interface = getattr(owner, "implements_halting_interface", None)
        if callable(supports_interface) and supports_interface():
            return
    built_owner_name = getattr(owner, "__name__", type(owner).__name__)
    raise ValueError(
        f"{field_name} {type(halting_config).__name__} builds "
        f"{built_owner_name}, which does not implement the HaltingInterface "
        f"required by {owner_name}"
    )


def _matches_config_contract(config: object, field_names: tuple[str, ...]) -> bool:
    return isinstance(config, ConfigBase) and all(
        hasattr(config, field_name) for field_name in field_names
    )


def _adaptive_grouping_paths(
    config: ConfigBase,
    *,
    root: str,
) -> tuple[str, ...]:
    matches: list[str] = []
    _collect_adaptive_grouping_paths(config, root, set(), matches)
    return tuple(matches)


def _collect_adaptive_grouping_paths(
    value: object,
    path: str,
    visited: set[int],
    matches: list[str],
) -> None:
    if isinstance(value, ConfigBase):
        identity = id(value)
        if identity in visited:
            return
        visited.add(identity)
        try:
            config_validator = value.registry_owner().VALIDATOR
        except (AttributeError, NotImplementedError):
            config_validator = None
        grouping_is_enabled = getattr(
            config_validator,
            "grouping_is_enabled",
            None,
        )
        if callable(grouping_is_enabled) and grouping_is_enabled(value):
            matches.append(path)
        for config_field in fields(value):
            field_value = getattr(value, config_field.name)
            field_path = f"{path}.{config_field.name}"
            _collect_adaptive_grouping_paths(
                field_value,
                field_path,
                visited,
                matches,
            )
        return

    if isinstance(value, dict):
        identity = id(value)
        if identity in visited:
            return
        visited.add(identity)
        for key, item in value.items():
            _collect_adaptive_grouping_paths(
                item,
                f"{path}[{key!r}]",
                visited,
                matches,
            )
        return

    if isinstance(value, (list, tuple)):
        identity = id(value)
        if identity in visited:
            return
        visited.add(identity)
        for index, item in enumerate(value):
            _collect_adaptive_grouping_paths(
                item,
                f"{path}[{index}]",
                visited,
                matches,
            )


def _validate_no_grouping_with_context_controllers(
    config: ConfigBase,
    *,
    owner_name: str,
    controllers: tuple[tuple[str, object | None], ...],
) -> None:
    active_controller_names = tuple(
        name for name, controller in controllers if controller is not None
    )
    if not active_controller_names:
        return

    grouping_paths = _adaptive_grouping_paths(
        config,
        root=owner_name,
    )
    if not grouping_paths:
        return
    controller_list = ", ".join(active_controller_names)
    raise ValueError(
        f"{owner_name} cannot combine enabled adaptive parameter grouping with "
        f"{controller_list}: context sharing is restricted inside halting or "
        f"memory owners. Found grouping at {grouping_paths[0]}."
    )
