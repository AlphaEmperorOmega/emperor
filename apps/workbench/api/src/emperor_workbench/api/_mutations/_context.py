"""Track the deterministic identity of the active HTTP mutation."""

from __future__ import annotations

import contextvars
import hashlib
from collections.abc import Iterator
from contextlib import contextmanager

_MUTATION_IDENTITY: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "workbench_mutation_identity",
    default=None,
)


def current_mutation_identity() -> str | None:
    return _MUTATION_IDENTITY.get()


def deterministic_mutation_resource_id(namespace: str) -> str | None:
    identity = current_mutation_identity()
    if identity is None:
        return None
    return hashlib.sha256(f"{namespace}\0{identity}".encode()).hexdigest()[:32]


@contextmanager
def activate_mutation_identity(identity: str) -> Iterator[None]:
    token = _MUTATION_IDENTITY.set(identity)
    try:
        yield
    finally:
        _MUTATION_IDENTITY.reset(token)


__all__ = [
    "activate_mutation_identity",
    "current_mutation_identity",
    "deterministic_mutation_resource_id",
]
