from __future__ import annotations

from collections.abc import Mapping
from typing import Final

from model_runtime.packages.runtime_values import validate_runtime_default_values

from . import config
from ._builder_adapter import linear_adaptive_builder_kwargs_from_flat
from .runtime_options import RuntimeOptions


def runtime_from_flat(values: Mapping[str, object] | None = None) -> RuntimeOptions:
    return RuntimeOptions(
        linear_adaptive_builder_kwargs_from_flat(
            validate_runtime_default_values(
                values,
                package="models.bert.linear_adaptive",
                config_module=config,
            ),
            config,
        )
    )


DEFAULT_RUNTIME: Final = runtime_from_flat()

__all__ = ["DEFAULT_RUNTIME", "runtime_from_flat"]
