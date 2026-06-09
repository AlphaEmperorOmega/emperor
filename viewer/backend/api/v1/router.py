"""Aggregate router for the root-mounted Viewer API routes.

The ``api.v1`` package is an internal organization namespace. The public HTTP
contract is currently mounted at root paths such as ``/health`` and ``/models``.
"""

from __future__ import annotations

from fastapi import APIRouter

from viewer.backend.api.v1.routers import (
    capabilities,
    config_snapshots,
    health,
    inspection,
    logs,
    models,
    training,
)

INTERNAL_API_VERSION_NAMESPACE = "v1"
PUBLIC_API_PREFIX = ""

router = APIRouter()
router.include_router(capabilities.router)
router.include_router(health.router)
router.include_router(models.router)
router.include_router(inspection.router)
router.include_router(logs.router)
router.include_router(training.router)
router.include_router(config_snapshots.router)

__all__ = ["INTERNAL_API_VERSION_NAMESPACE", "PUBLIC_API_PREFIX", "router"]
