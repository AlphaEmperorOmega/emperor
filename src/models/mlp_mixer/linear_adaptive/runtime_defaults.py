from __future__ import annotations

from dataclasses import fields, replace
from types import ModuleType
from typing import Final

import models.mlp_mixer.linear_adaptive.config as config
from model_runtime.packages.runtime_values import validate_runtime_default_values
from models.mlp_mixer.linear_adaptive.runtime_options import RuntimeOptions

_PACKAGE = "models.mlp_mixer.linear_adaptive"
_RUNTIME_FIELDS = {field.name for field in fields(RuntimeOptions)}
_CONTROLLER_STACK_FIELDS = (
    "hidden_dim",
    "layer_norm_position",
    "num_layers",
    "activation",
    "residual_connection_option",
    "dropout_probability",
    "last_layer_bias_option",
    "apply_output_pipeline_flag",
    "bias_flag",
)
_INHERITED_RUNTIME_FIELDS = {
    "stack_activation": (
        "token_mixer_stack_activation",
        "channel_mixer_stack_activation",
    ),
    "stack_bias_flag": ("token_mixer_bias_flag", "channel_mixer_bias_flag"),
}


def _runtime_updates(values: dict[str, object]) -> dict[str, object]:
    updates = {key: value for key, value in values.items() if key in _RUNTIME_FIELDS}
    for field in _CONTROLLER_STACK_FIELDS:
        canonical = f"submodule_stack_{field}"
        compatibility = f"controller_stack_{field}"
        if canonical in values:
            updates[compatibility] = values[canonical]
        elif compatibility in values:
            updates[canonical] = values[compatibility]

    for source, targets in _INHERITED_RUNTIME_FIELDS.items():
        if source not in values:
            continue
        for target in targets:
            if target not in values:
                updates[target] = values[source]
    return updates


def runtime_from_config(config_module: ModuleType = config) -> RuntimeOptions:
    return RuntimeOptions(
        **{
            field.name: getattr(config_module, field.name.upper())
            for field in fields(RuntimeOptions)
        }
    )


def runtime_from_flat(
    flat_kwargs: dict | None = None,
    config_module: ModuleType = config,
) -> RuntimeOptions:
    values = validate_runtime_default_values(
        flat_kwargs,
        package=_PACKAGE,
        config_module=config_module,
    )
    runtime_updates = _runtime_updates(values)
    return replace(runtime_from_config(config_module), **runtime_updates)


DEFAULT_RUNTIME: Final[RuntimeOptions] = runtime_from_flat()

__all__ = ["DEFAULT_RUNTIME", "runtime_from_config", "runtime_from_flat"]
