"""Stable compatibility exports for the public Viewer API ASGI target.

The application factory lives in :mod:`viewer.backend.main`, but existing
commands use ``viewer.backend.api:app``. Keep that import stable.
"""

from viewer.backend.main import ViewerApiSettings, app, create_app

COMPATIBILITY_STATUS = "stable"
REPLACEMENT_IMPORT = "viewer.backend.main"

__all__ = ["ViewerApiSettings", "app", "create_app"]
