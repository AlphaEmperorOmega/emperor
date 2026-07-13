from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from workbench.backend.main import WorkbenchApiSettings, app, create_app

COMPATIBILITY_STATUS = "stable"
REPLACEMENT_IMPORT = "workbench.backend.main"

__all__ = ["WorkbenchApiSettings", "app", "create_app"]


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from workbench.backend.main import WorkbenchApiSettings, app, create_app

    exports = {
        "WorkbenchApiSettings": WorkbenchApiSettings,
        "app": app,
        "create_app": create_app,
    }
    globals().update(exports)
    return exports[name]
