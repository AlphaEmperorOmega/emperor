from __future__ import annotations

from enum import StrEnum

from emperor_workbench.failures import DomainFailure


class ConfigSnapshotFailure(DomainFailure):
    """A Config Snapshot request cannot be completed."""


class ConfigSnapshotConflictReason(StrEnum):
    ID = "id"
    NAME = "name"
    RUNTIME_DEFAULTS = "runtime-defaults"
    STALE = "stale"


class ConfigSnapshotConflictError(Exception):
    def __init__(self, reason: ConfigSnapshotConflictReason) -> None:
        self.reason = reason
        super().__init__(reason.value)


def config_snapshot_conflict_failure(
    exc: ConfigSnapshotConflictError,
) -> ConfigSnapshotFailure:
    messages = {
        ConfigSnapshotConflictReason.ID: (
            "A config snapshot with this id already exists."
        ),
        ConfigSnapshotConflictReason.NAME: "A snapshot with this name already exists.",
        ConfigSnapshotConflictReason.RUNTIME_DEFAULTS: (
            "A snapshot with these config values already exists."
        ),
        ConfigSnapshotConflictReason.STALE: (
            "The config snapshot changed concurrently. Retry the update."
        ),
    }
    return ConfigSnapshotFailure(messages[exc.reason])


__all__ = [
    "ConfigSnapshotConflictError",
    "ConfigSnapshotConflictReason",
    "ConfigSnapshotFailure",
    "config_snapshot_conflict_failure",
]
