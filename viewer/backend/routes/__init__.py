"""Deprecated compatibility package for legacy route imports.

Active route implementations live under :mod:`viewer.backend.api.v1.routers`.
Remove these shims only after a repository import audit and migration note.
"""

COMPATIBILITY_STATUS = "deprecated"
REPLACEMENT_IMPORT = "viewer.backend.api.v1.routers"
REMOVAL_CONDITION = "No non-test imports remain and the migration is documented."
