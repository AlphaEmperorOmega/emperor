"""Deprecated compatibility import for the training job router.

Use :mod:`workbench.backend.api.v1.routers.training` instead.
Remove this shim only after a repository import audit and migration note.
"""

from workbench.backend.api.v1.routers.training import router

COMPATIBILITY_STATUS = "deprecated"
REPLACEMENT_IMPORT = "workbench.backend.api.v1.routers.training"
REMOVAL_CONDITION = "No non-test imports remain and the migration is documented."

__all__ = ["router"]
