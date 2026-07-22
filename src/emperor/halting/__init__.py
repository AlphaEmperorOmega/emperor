"""Public Interface for adaptive-computation halting."""

from emperor.halting._base import HaltingBase, HaltingStateBase
from emperor.halting._config import (
    HaltingConfig,
    HaltingHiddenStateModeOptions,
    SoftHaltingConfig,
    StickBreakingConfig,
)
from emperor.halting._interface import HaltingInterface
from emperor.halting._monitoring.callback import HaltingMonitorCallback
from emperor.halting._monitoring.tracking import (
    HaltingUsageTracker,
    HaltingUsageTrackerManager,
)
from emperor.halting._strategies.soft import SoftHalting, SoftHaltingState
from emperor.halting._strategies.stick_breaking import (
    StickBreaking,
    StickBreakingState,
)

__all__ = (
    "HaltingConfig",
    "SoftHaltingConfig",
    "StickBreakingConfig",
    "HaltingHiddenStateModeOptions",
    "HaltingInterface",
    "HaltingBase",
    "HaltingStateBase",
    "SoftHalting",
    "SoftHaltingState",
    "StickBreaking",
    "StickBreakingState",
    "HaltingMonitorCallback",
    "HaltingUsageTracker",
    "HaltingUsageTrackerManager",
)
