from __future__ import annotations

import threading
import time
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field

from emperor_workbench.failures import FailureKind
from emperor_workbench.log_experiments._errors import LogExperimentFailure

DEFAULT_LOG_EXPERIMENT_MUTATION_TIMEOUT_SECONDS = 30.0
LOG_EXPERIMENT_MUTATION_TIMEOUT_MESSAGE = (
    "Timed out waiting for another mutation of this log experiment."
)


@dataclass(slots=True)
class _MutationEntry:
    lock: threading.Lock = field(default_factory=threading.Lock)
    users: int = 0


class LogExperimentMutationCoordinator:
    """Serialize mutations that target the same Log Experiment identity."""

    def __init__(
        self,
        *,
        acquire_timeout_seconds: float = (
            DEFAULT_LOG_EXPERIMENT_MUTATION_TIMEOUT_SECONDS
        ),
    ) -> None:
        if acquire_timeout_seconds <= 0:
            raise ValueError("Mutation lock timeout must be greater than zero")
        self._acquire_timeout_seconds = acquire_timeout_seconds
        self._registry_lock = threading.Lock()
        self._entries: dict[str, _MutationEntry] = {}

    @contextmanager
    def coordinate(self, experiments: Iterable[str]) -> Iterator[None]:
        names = tuple(sorted({str(name) for name in experiments if str(name)}))
        entries = self._reserve(names)
        acquired: list[_MutationEntry] = []
        deadline = time.monotonic() + self._acquire_timeout_seconds
        try:
            for entry in entries:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not entry.lock.acquire(timeout=remaining):
                    raise LogExperimentFailure(
                        LOG_EXPERIMENT_MUTATION_TIMEOUT_MESSAGE,
                        kind=FailureKind.UNAVAILABLE,
                    )
                acquired.append(entry)
            yield
        finally:
            for entry in reversed(acquired):
                entry.lock.release()
            self._release_reservations(names, entries)

    def _reserve(self, names: tuple[str, ...]) -> tuple[_MutationEntry, ...]:
        with self._registry_lock:
            entries = []
            for name in names:
                entry = self._entries.get(name)
                if entry is None:
                    entry = _MutationEntry()
                    self._entries[name] = entry
                entry.users += 1
                entries.append(entry)
            return tuple(entries)

    def _release_reservations(
        self,
        names: tuple[str, ...],
        entries: tuple[_MutationEntry, ...],
    ) -> None:
        with self._registry_lock:
            for name, entry in zip(names, entries, strict=True):
                entry.users -= 1
                if entry.users == 0 and self._entries.get(name) is entry:
                    self._entries.pop(name)


__all__ = [
    "DEFAULT_LOG_EXPERIMENT_MUTATION_TIMEOUT_SECONDS",
    "LOG_EXPERIMENT_MUTATION_TIMEOUT_MESSAGE",
    "LogExperimentMutationCoordinator",
]
