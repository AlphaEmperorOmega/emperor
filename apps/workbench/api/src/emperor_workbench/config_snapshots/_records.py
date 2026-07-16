from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from types import MappingProxyType


def current_timestamp() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True, slots=True)
class ConfigSnapshotRecord:
    id: str
    model: str
    preset: str
    name: str
    overrides: Mapping[str, str]
    created_at: str = field(default_factory=current_timestamp)
    updated_at: str = field(default_factory=current_timestamp)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "overrides",
            MappingProxyType(dict(self.overrides)),
        )


@dataclass(frozen=True, slots=True)
class ConfigSnapshotDeletion:
    model: str
    snapshots: tuple[ConfigSnapshotRecord, ...]


def snapshot_sort_key(
    snapshot: ConfigSnapshotRecord,
) -> tuple[str, str, str, str]:
    return (snapshot.model, snapshot.preset, snapshot.created_at, snapshot.id)


__all__ = [
    "ConfigSnapshotDeletion",
    "ConfigSnapshotRecord",
    "current_timestamp",
    "snapshot_sort_key",
]
