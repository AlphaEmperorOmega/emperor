"""Stable compatibility exports for Viewer backend settings.

The canonical settings interface lives in :mod:`viewer.backend.core.config`.
"""

from __future__ import annotations

from viewer.backend.core.config import (
    LOCAL_FRONTEND_ORIGINS,
    ViewerApiSettings,
    get_viewer_api_settings,
)

COMPATIBILITY_STATUS = "stable"
REPLACEMENT_IMPORT = "viewer.backend.core.config"

__all__ = [
    "LOCAL_FRONTEND_ORIGINS",
    "ViewerApiSettings",
    "get_viewer_api_settings",
]
