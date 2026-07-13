from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


class ActiveLogWriter(Protocol):
    """Structural view of a process currently writing a Log Experiment."""

    id: str
    status: str
    log_folder: str


ActiveLogWriterSource = Callable[[], Iterable[ActiveLogWriter]]


@dataclass(frozen=True, slots=True)
class HistoricalCheckpointCandidate:
    """One contained checkpoint frozen for a historical Inspection read."""

    path: Path
    size_bytes: int
    modified_at_ns: int


@dataclass(frozen=True, slots=True)
class HistoricalInspectionContext:
    """Minimum historical Run state needed by Workbench Inspection."""

    run_id: str
    model: str
    preset: str
    dataset: str
    params: Mapping[str, Any]
    checkpoint_candidates: tuple[HistoricalCheckpointCandidate, ...]

    @property
    def checkpoint_paths(self) -> tuple[Path, ...]:
        """Compatibility projection; loading uses the frozen candidates."""

        return tuple(candidate.path for candidate in self.checkpoint_candidates)


class HistoricalInspectionSource(Protocol):
    """Narrow consumer Interface implemented by the Run History capability."""

    def inspection_context(self, run_id: str) -> HistoricalInspectionContext: ...


__all__ = [
    "ActiveLogWriter",
    "ActiveLogWriterSource",
    "HistoricalCheckpointCandidate",
    "HistoricalInspectionContext",
    "HistoricalInspectionSource",
]
