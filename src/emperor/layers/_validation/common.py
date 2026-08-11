from __future__ import annotations

from numbers import Integral

from emperor._validation import _adaptive_grouping_paths
from emperor.config import ConfigBase


def _config_classes():
    from emperor.layers._config import LayerConfig, LayerStackConfig

    return LayerConfig, LayerStackConfig


def _gate_config_class():
    from emperor.layers._config import GateConfig

    return GateConfig


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
            validator = getattr(owner, "VALIDATOR", None)
            validate_minimum_step_capability = getattr(
                validator,
                "validate_minimum_step_capability",
                None,
            )
            if callable(validate_minimum_step_capability):
                validate_minimum_step_capability(
                    halting_config,
                    owner_name=owner_name,
                )
            else:
                configured_min_steps = getattr(halting_config, "min_steps", None)
                has_default_minimum = configured_min_steps is None or (
                    not isinstance(configured_min_steps, bool)
                    and isinstance(configured_min_steps, Integral)
                    and configured_min_steps == 1
                )
                if (
                    not has_default_minimum
                    and getattr(owner, "supports_minimum_step_delay", False) is not True
                ):
                    raise ValueError(
                        f"halting_config.min_steps={configured_min_steps} requires "
                        f"minimum-step delay support for {owner_name}; "
                        f"{owner.__name__} implements only the legacy lifecycle."
                    )
            return
    built_owner_name = getattr(owner, "__name__", type(owner).__name__)
    raise ValueError(
        f"{field_name} {type(halting_config).__name__} builds "
        f"{built_owner_name}, which does not implement the HaltingInterface "
        f"required by {owner_name}"
    )


def _validate_halting_owner_step_contract(
    halting_config,
    *,
    owner_step_limit: int | None,
    owner_name: str,
) -> None:
    validator = getattr(halting_config._registry_owner(), "VALIDATOR", None)
    validate_owner_step_contract = getattr(
        validator,
        "validate_owner_step_contract",
        None,
    )
    if callable(validate_owner_step_contract):
        validate_owner_step_contract(
            halting_config,
            owner_step_limit=owner_step_limit,
            owner_name=owner_name,
        )


def _validate_halting_required_update_count(
    halting_config,
    *,
    required_update_count: int,
    owner_name: str,
) -> None:
    validator = getattr(halting_config._registry_owner(), "VALIDATOR", None)
    validate_required_update_count = getattr(
        validator,
        "validate_required_update_count",
        None,
    )
    if callable(validate_required_update_count):
        validate_required_update_count(
            halting_config,
            required_update_count=required_update_count,
            owner_name=owner_name,
        )
        return

    configured_min_steps = getattr(halting_config, "min_steps", None)
    min_steps = 1 if configured_min_steps is None else configured_min_steps
    if min_steps < required_update_count:
        raise ValueError(
            "halting_config.min_steps must be greater than or equal to the "
            f"required update count for {owner_name}; received "
            f"min_steps={min_steps} and "
            f"required_update_count={required_update_count}."
        )


def _matches_config_contract(config: object, field_names: tuple[str, ...]) -> bool:
    return isinstance(config, ConfigBase) and all(
        hasattr(config, field_name) for field_name in field_names
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
