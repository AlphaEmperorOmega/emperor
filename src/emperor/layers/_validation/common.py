from __future__ import annotations

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
