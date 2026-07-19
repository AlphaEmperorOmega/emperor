"""Public Interface for adaptive-computation halting."""

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
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

_LAZY_EXPORTS = {
    "HaltingConfig": ("emperor.halting._config", "HaltingConfig"),
    "SoftHaltingConfig": ("emperor.halting._config", "SoftHaltingConfig"),
    "StickBreakingConfig": (
        "emperor.halting._config",
        "StickBreakingConfig",
    ),
    "HaltingHiddenStateModeOptions": (
        "emperor.halting._config",
        "HaltingHiddenStateModeOptions",
    ),
    "HaltingInterface": ("emperor.halting._interface", "HaltingInterface"),
    "HaltingBase": ("emperor.halting._base", "HaltingBase"),
    "HaltingStateBase": ("emperor.halting._base", "HaltingStateBase"),
    "SoftHalting": ("emperor.halting._strategies.soft", "SoftHalting"),
    "SoftHaltingState": (
        "emperor.halting._strategies.soft",
        "SoftHaltingState",
    ),
    "StickBreaking": (
        "emperor.halting._strategies.stick_breaking",
        "StickBreaking",
    ),
    "StickBreakingState": (
        "emperor.halting._strategies.stick_breaking",
        "StickBreakingState",
    ),
    "HaltingMonitorCallback": (
        "emperor.halting._monitoring.callback",
        "HaltingMonitorCallback",
    ),
    "HaltingUsageTracker": (
        "emperor.halting._monitoring.tracking",
        "HaltingUsageTracker",
    ),
    "HaltingUsageTrackerManager": (
        "emperor.halting._monitoring.tracking",
        "HaltingUsageTrackerManager",
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
