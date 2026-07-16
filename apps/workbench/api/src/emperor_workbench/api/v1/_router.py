from __future__ import annotations

from fastapi import APIRouter

from emperor_workbench.api._probes import router as probes_router
from emperor_workbench.api.v1 import (
    config_snapshots,
    inspection,
    model_packages,
    run_history,
    run_plans,
    training_jobs,
)
from emperor_workbench.api.v1._capabilities import router as capabilities_router

INTERNAL_API_VERSION_NAMESPACE = "v1"
PUBLIC_API_PREFIX = ""

router = APIRouter()
router.include_router(capabilities_router)
router.include_router(probes_router)
router.include_router(model_packages.router)
router.include_router(inspection.router)
router.include_router(run_history.router)
router.include_router(training_jobs.router)
router.include_router(run_plans.router)
router.include_router(config_snapshots.router)

__all__ = ["INTERNAL_API_VERSION_NAMESPACE", "PUBLIC_API_PREFIX", "router"]
