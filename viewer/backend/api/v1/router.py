"""Aggregate router for the Viewer API routes."""

from __future__ import annotations

from fastapi import APIRouter

from viewer.backend.api.v1.routers import health, inspection, logs, models, training

router = APIRouter()
router.include_router(health.router)
router.include_router(models.router)
router.include_router(inspection.router)
router.include_router(logs.router)
router.include_router(training.router)
