"""Stable compatibility exports for the public Viewer API ASGI target.

The application factory lives in :mod:`viewer.backend.main`, but existing
commands use ``viewer.backend.api:app``. Keep that import stable.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from viewer.backend.main import ViewerApiSettings, app, create_app

COMPATIBILITY_STATUS = "stable"
REPLACEMENT_IMPORT = "viewer.backend.main"

__all__ = ["ViewerApiSettings", "app", "create_app"]


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from viewer.backend.main import ViewerApiSettings, app, create_app

    exports = {
        "ViewerApiSettings": ViewerApiSettings,
        "app": app,
        "create_app": create_app,
    }
    globals().update(exports)
    return exports[name]
