"""Semantic checkpoint selection policy for historical Inspection."""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path
from typing import Protocol, TypeVar

CHECKPOINT_EPOCH_RE = re.compile(r"(?:^|[-_])epoch=(?P<value>\d+)(?:[-_]|$)")
CHECKPOINT_STEP_RE = re.compile(r"(?:^|[-_])step=(?P<value>\d+)(?:[-_]|$)")


class RankedCheckpoint(Protocol):
    path: Path
    modified_at_ns: int


CheckpointT = TypeVar("CheckpointT", bound=RankedCheckpoint)


def _parse_checkpoint_field(pattern: re.Pattern[str], filename: str) -> int | None:
    match = pattern.search(Path(filename).stem)
    if match is None:
        return None
    return int(match.group("value"))


def parse_checkpoint_epoch(filename: str) -> int | None:
    return _parse_checkpoint_field(CHECKPOINT_EPOCH_RE, filename)


def parse_checkpoint_step(filename: str) -> int | None:
    return _parse_checkpoint_field(CHECKPOINT_STEP_RE, filename)


def rank_historical_checkpoints(
    candidates: Iterable[CheckpointT],
) -> tuple[CheckpointT, ...]:
    """Rank candidates from preferred to fallback, deterministically.

    A loadable ``last.ckpt`` wins by being attempted first. The checkpoint
    reader continues after corruption, so every remaining candidate stays in
    descending semantic step/epoch/mtime order with a stable path tie-break.
    """

    def key(candidate: CheckpointT) -> tuple[int, int, int, int, str]:
        filename = candidate.path.name
        step = parse_checkpoint_step(filename)
        epoch = parse_checkpoint_epoch(filename)
        return (
            0 if filename == "last.ckpt" else 1,
            -(step if step is not None else -1),
            -(epoch if epoch is not None else -1),
            -candidate.modified_at_ns,
            candidate.path.as_posix(),
        )

    return tuple(sorted(candidates, key=key))


__all__ = [
    "parse_checkpoint_epoch",
    "parse_checkpoint_step",
    "rank_historical_checkpoints",
]
