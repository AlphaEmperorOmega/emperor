"""Public Interface for dynamic-memory modules."""

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from emperor.memory._base import DynamicMemoryAbstract
    from emperor.memory._config import (
        AttentionDynamicMemoryConfig,
        DynamicMemoryConfig,
        ElementWiseWeightedDynamicMemoryConfig,
        GatedResidualDynamicMemoryConfig,
        MemoryPositionOptions,
        WeightedDynamicMemoryConfig,
    )
    from emperor.memory._monitoring import MemoryMonitorCallback
    from emperor.memory._variants.attention import AttentionDynamicMemory
    from emperor.memory._variants.element_wise_weighted import (
        ElementWiseWeightedDynamicMemory,
    )
    from emperor.memory._variants.gated_residual import (
        GatedResidualDynamicMemory,
    )
    from emperor.memory._variants.weighted import WeightedDynamicMemory

__all__ = (
    "AttentionDynamicMemory",
    "AttentionDynamicMemoryConfig",
    "DynamicMemoryConfig",
    "DynamicMemoryAbstract",
    "ElementWiseWeightedDynamicMemory",
    "ElementWiseWeightedDynamicMemoryConfig",
    "GatedResidualDynamicMemory",
    "GatedResidualDynamicMemoryConfig",
    "MemoryMonitorCallback",
    "MemoryPositionOptions",
    "WeightedDynamicMemory",
    "WeightedDynamicMemoryConfig",
)

_LAZY_EXPORTS = {
    "AttentionDynamicMemory": (
        "emperor.memory._variants.attention",
        "AttentionDynamicMemory",
    ),
    "AttentionDynamicMemoryConfig": (
        "emperor.memory._config",
        "AttentionDynamicMemoryConfig",
    ),
    "DynamicMemoryConfig": (
        "emperor.memory._config",
        "DynamicMemoryConfig",
    ),
    "DynamicMemoryAbstract": (
        "emperor.memory._base",
        "DynamicMemoryAbstract",
    ),
    "ElementWiseWeightedDynamicMemory": (
        "emperor.memory._variants.element_wise_weighted",
        "ElementWiseWeightedDynamicMemory",
    ),
    "ElementWiseWeightedDynamicMemoryConfig": (
        "emperor.memory._config",
        "ElementWiseWeightedDynamicMemoryConfig",
    ),
    "GatedResidualDynamicMemory": (
        "emperor.memory._variants.gated_residual",
        "GatedResidualDynamicMemory",
    ),
    "GatedResidualDynamicMemoryConfig": (
        "emperor.memory._config",
        "GatedResidualDynamicMemoryConfig",
    ),
    "MemoryMonitorCallback": (
        "emperor.memory._monitoring",
        "MemoryMonitorCallback",
    ),
    "MemoryPositionOptions": (
        "emperor.memory._config",
        "MemoryPositionOptions",
    ),
    "WeightedDynamicMemory": (
        "emperor.memory._variants.weighted",
        "WeightedDynamicMemory",
    ),
    "WeightedDynamicMemoryConfig": (
        "emperor.memory._config",
        "WeightedDynamicMemoryConfig",
    ),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute_name = _LAZY_EXPORTS[name]
    except KeyError as error:
        message = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(message) from error

    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value
