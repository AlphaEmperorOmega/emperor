"""Stable compatibility exports for Workbench backend settings.

The canonical settings interface lives in :mod:`workbench.backend.core.config`.
"""

from __future__ import annotations

from workbench.backend.core.config import (
    LOCAL_FRONTEND_ORIGINS,
    WorkbenchApiSettings,
    get_workbench_api_settings,
)

COMPATIBILITY_STATUS = "stable"
REPLACEMENT_IMPORT = "workbench.backend.core.config"

__all__ = [
    "LOCAL_FRONTEND_ORIGINS",
    "WorkbenchApiSettings",
    "get_workbench_api_settings",
]
