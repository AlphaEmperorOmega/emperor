"""Deprecated compatibility import for the historical log-run router.

Use :mod:`viewer.backend.api.v1.routers.logs` instead.
Remove this shim only after a repository import audit and migration note.
"""

from viewer.backend.api.v1.routers.logs import router

COMPATIBILITY_STATUS = "deprecated"
REPLACEMENT_IMPORT = "viewer.backend.api.v1.routers.logs"
REMOVAL_CONDITION = "No non-test imports remain and the migration is documented."

__all__ = ["router"]
