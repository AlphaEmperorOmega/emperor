from fastapi import APIRouter, Depends

from emperor_workbench.api._security import require_bearer_auth
from emperor_workbench.api.v1.run_history import (
    _experiments,
    _imports,
    _monitoring,
    _runs,
)

router = APIRouter(
    prefix="/logs",
    tags=["logs"],
    dependencies=[Depends(require_bearer_auth)],
)
router.include_router(_runs.router)
router.include_router(_experiments.router)
router.include_router(_imports.router)
router.include_router(_monitoring.router)

__all__ = ["router"]
