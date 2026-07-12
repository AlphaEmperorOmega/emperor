"""Transport-neutral failure contract shared by Workbench capabilities."""

from __future__ import annotations

from enum import StrEnum


class FailureKind(StrEnum):
    INVALID = "invalid"
    CONFLICT = "conflict"
    TIMEOUT = "timeout"
    UNAVAILABLE = "unavailable"
    TOO_LARGE = "too-large"


class DomainFailure(Exception):
    """A caller-facing capability failure without HTTP semantics."""

    def __init__(
        self,
        detail: str,
        *,
        kind: FailureKind = FailureKind.INVALID,
    ) -> None:
        super().__init__(detail)
        self.detail = detail
        self.kind = kind


__all__ = ["DomainFailure", "FailureKind"]
