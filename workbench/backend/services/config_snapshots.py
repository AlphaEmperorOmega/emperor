"""Deprecated import shim for the deep Config Snapshot Interface."""

from workbench.backend.config_snapshots import (
    ConfigSnapshotService,
    config_snapshot_schema,
)

config_schema = config_snapshot_schema

__all__ = ["ConfigSnapshotService"]
