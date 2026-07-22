from __future__ import annotations

from dataclasses import fields, replace
from types import ModuleType
from typing import Any, Final

import models.vit.linear.config as config
from model_runtime.packages.runtime_values import validate_runtime_default_values
from models.vit.linear._builder_adapter import linear_builder_kwargs_from_flat
from models.vit.linear.runtime_options import RuntimeOptions

_RUNTIME_FIELDS = {field.name for field in fields(RuntimeOptions)}


def runtime_from_config(config_module: ModuleType = config) -> RuntimeOptions:
    return RuntimeOptions(
        batch_size=config_module.BATCH_SIZE,
        learning_rate=config_module.LEARNING_RATE,
        input_dim=config_module.INPUT_DIM,
        output_dim=config_module.OUTPUT_DIM,
        patch_options=None,
        encoder_options=None,
        positional_embedding_options=None,
        attention_options=None,
        feed_forward_options=None,
        output_options=None,
        attention_projection_stack_options=None,
        attention_projection_layer_controller_options=None,
        attention_projection_dynamic_memory_options=None,
        attention_projection_recurrent_controller_options=None,
        feed_forward_stack_options=None,
        feed_forward_layer_controller_options=None,
        feed_forward_dynamic_memory_options=None,
        feed_forward_recurrent_controller_options=None,
        stack_options=None,
        submodule_stack_options=None,
        layer_controller_options=None,
        dynamic_memory_options=None,
        recurrent_controller_options=None,
    )


def _runtime_with_fields(
    runtime: RuntimeOptions,
    builder_options: dict[str, Any],
) -> RuntimeOptions:
    unknown = set(builder_options) - _RUNTIME_FIELDS
    if unknown:
        name = sorted(unknown)[0]
        raise TypeError(
            "VitLinearConfigBuilder.__init__() got an unexpected keyword "
            f"argument {name!r}"
        )
    return replace(runtime, **builder_options)


def runtime_from_flat(
    flat_kwargs: dict[str, Any] | None = None,
    config_module: ModuleType = config,
) -> RuntimeOptions:
    runtime = runtime_from_config(config_module)
    builder_options = linear_builder_kwargs_from_flat(
        validate_runtime_default_values(
            flat_kwargs,
            package="models.vit.linear",
            config_module=config_module,
        ),
        config_module,
    )
    return _runtime_with_fields(runtime, builder_options)


DEFAULT_RUNTIME: Final[RuntimeOptions] = runtime_from_flat()


__all__ = [
    "DEFAULT_RUNTIME",
    "runtime_from_config",
    "runtime_from_flat",
]
