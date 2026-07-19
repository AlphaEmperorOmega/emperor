"""Public Interface for dynamic-memory configuration."""

from emperor.memory._config import (
    AttentionDynamicMemoryConfig,
    DynamicMemoryConfig,
    ElementWiseWeightedDynamicMemoryConfig,
    GatedResidualDynamicMemoryConfig,
    MemoryPositionOptions,
    WeightedDynamicMemoryConfig,
)
from emperor.memory._interface import MemoryInterface

__all__ = (
    "AttentionDynamicMemoryConfig",
    "DynamicMemoryConfig",
    "ElementWiseWeightedDynamicMemoryConfig",
    "GatedResidualDynamicMemoryConfig",
    "MemoryInterface",
    "MemoryPositionOptions",
    "WeightedDynamicMemoryConfig",
)
