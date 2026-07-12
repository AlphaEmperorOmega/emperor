"""Low-level TensorBoard event-file access shared by the monitor readers.

Both :class:`~workbench.backend.tensorboard.readers.TensorBoardMonitorReader` and the
Run History query implementation read the same event files; these
helpers are the single implementation of that access so the two stay in step.
"""

from __future__ import annotations

import base64
import math
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Any

from tensorboard.backend.event_processing import event_accumulator

DEFAULT_TENSORBOARD_SIZE_GUIDANCE = {
    event_accumulator.SCALARS: 500,
    event_accumulator.HISTOGRAMS: 1,
    event_accumulator.IMAGES: 1,
    event_accumulator.TENSORS: 1,
}
TENSORBOARD_TAG_SIZE_GUIDANCE = {
    event_accumulator.SCALARS: 1,
    event_accumulator.HISTOGRAMS: 1,
    event_accumulator.IMAGES: 1,
    event_accumulator.TENSORS: 1,
}
MAX_TENSORBOARD_TEXT_SUMMARY_CHARS = 20_000
MAX_TENSORBOARD_IMAGE_SUMMARY_BYTES = 1_000_000
EventFileFingerprint = tuple[tuple[str, int, int], ...]


@dataclass(frozen=True, slots=True)
class EventFileIndex:
    root: Path
    dirs: tuple[Path, ...]
    files: tuple[Path, ...]
    fingerprint: EventFileFingerprint
    total_size: int

    def cache_key(self, *parts: Any) -> tuple[Any, ...]:
        return (self.root.as_posix(), self.fingerprint, *parts)

    def exceeds(self, byte_limit: int) -> bool:
        return byte_limit > 0 and self.total_size > byte_limit

    def load_accumulator(
        self,
        event_dir: Path,
        *,
        size_guidance: dict[int, int] | None = None,
    ) -> Any | None:
        """Load one directory from this contained observation only."""
        if event_dir not in self.dirs:
            return None
        if size_guidance is None:
            return load_event_accumulator(event_dir)
        return load_event_accumulator(event_dir, size_guidance=size_guidance)


class TensorBoardEventCache:
    """Generation-safe root-keyed LRU cache group for event projections."""

    def __init__(self, limits: Mapping[str, int]) -> None:
        if not limits or any(limit < 1 for limit in limits.values()):
            raise ValueError("TensorBoard event cache limits must be positive.")
        self._limits = dict(limits)
        self._caches: dict[str, OrderedDict[tuple[Any, ...], Any]] = {
            name: OrderedDict() for name in limits
        }
        self._generation = 0
        self._lock = RLock()

    def token(self) -> int:
        with self._lock:
            return self._generation

    def get(self, cache_name: str, key: tuple[Any, ...]) -> Any | None:
        with self._lock:
            cache = self._caches[cache_name]
            if key not in cache:
                return None
            cache.move_to_end(key)
            return cache[key]

    def publish(
        self,
        cache_name: str,
        key: tuple[Any, ...],
        value: Any,
        *,
        generation: int,
    ) -> None:
        with self._lock:
            if generation != self._generation:
                return
            cache = self._caches[cache_name]
            cache[key] = value
            cache.move_to_end(key)
            while len(cache) > self._limits[cache_name]:
                cache.popitem(last=False)

    def clear_roots(self, roots: set[str]) -> None:
        if not roots:
            return
        with self._lock:
            self._generation += 1
            for cache in self._caches.values():
                for key in list(cache):
                    cached_root = str(key[0]) if key else ""
                    if any(
                        cached_root == root
                        or cached_root.startswith(f"{root}/")
                        for root in roots
                    ):
                        cache.pop(key, None)

    def clear(self) -> None:
        with self._lock:
            self._generation += 1
            for cache in self._caches.values():
                cache.clear()


def finite_float(value: Any) -> float:
    """Coerce ``value`` to a float, mapping non-finite values to ``0.0``."""
    number = float(value)
    if math.isfinite(number):
        return number
    return 0.0


def scalar_points(
    accumulator,
    tag: str,
    limit: int | None,
) -> list[dict[str, Any]]:
    """Read scalar events for ``tag`` as frontend-compatible point payloads."""
    events = accumulator.Scalars(tag)
    if limit is not None:
        events = events[-limit:]
    return [
        {
            "step": int(event.step),
            "wallTime": finite_float(event.wall_time),
            "value": finite_float(event.value),
        }
        for event in events
    ]


def event_file_index(
    root: Path,
    *,
    candidates: tuple[Path, ...] | None = None,
) -> EventFileIndex:
    """Return a contained event-file index in one tree walk.

    ``EventAccumulator`` reads every matching event file in a directory. If any
    matching member resolves outside the requested root, the entire directory is
    ignored so a safe sibling cannot make the accumulator open the escaping file.
    """
    try:
        trusted_root = root.resolve(strict=True)
    except OSError:
        return EventFileIndex(
            root=root,
            dirs=(),
            files=(),
            fingerprint=(),
            total_size=0,
        )

    files_by_dir: dict[Path, list[tuple[Path, Path, int, int]]] = {}
    unsafe_dirs: set[Path] = set()
    paths = candidates if candidates is not None else tuple(
        root.rglob("events.out.tfevents.*")
    )
    for path in paths:
        if not path.name.startswith("events.out.tfevents."):
            continue
        event_dir = path.parent
        try:
            resolved = path.resolve(strict=True)
            resolved.relative_to(trusted_root)
        except (OSError, ValueError):
            unsafe_dirs.add(event_dir)
            continue
        if not resolved.is_file():
            continue
        try:
            stat = resolved.stat()
        except OSError:
            continue
        files_by_dir.setdefault(event_dir, []).append(
            (path, resolved, int(stat.st_size), int(stat.st_mtime_ns))
        )

    dirs: list[Path] = []
    files: list[Path] = []
    fingerprint: list[tuple[str, int, int]] = []
    total = 0
    for event_dir in sorted(files_by_dir):
        if event_dir in unsafe_dirs:
            continue
        dirs.append(event_dir)
        for path, resolved, size, modified_at in sorted(
            files_by_dir[event_dir],
            key=lambda item: item[0],
        ):
            files.append(resolved)
            total += size
            fingerprint.append((path.as_posix(), size, modified_at))
    return EventFileIndex(
        root=root,
        dirs=tuple(dirs),
        files=tuple(files),
        fingerprint=tuple(sorted(fingerprint)),
        total_size=total,
    )


def load_event_accumulator(
    run_dir: Path,
    *,
    size_guidance: dict[int, int] | None = None,
):
    """Load and reload an ``EventAccumulator``, or ``None`` if it cannot be read."""
    try:
        accumulator = event_accumulator.EventAccumulator(
            str(run_dir),
            size_guidance=dict(size_guidance or DEFAULT_TENSORBOARD_SIZE_GUIDANCE),
        )
        accumulator.Reload()
    except Exception:
        return None
    return accumulator


def _truncated_payload_metadata(
    *,
    raw_bytes: int,
    limit: int,
    reason: str,
) -> dict[str, Any]:
    return {
        "eventBytes": raw_bytes,
        "truncated": True,
        "truncationReason": f"{reason}: {raw_bytes} bytes exceeds {limit} byte cap",
        "sourceItemCount": 1,
        "returnedItemCount": 0,
    }


def image_summary(accumulator, tag: str) -> dict[str, Any] | None:
    """Read the latest image summary for ``tag`` as a data URL payload."""
    events = accumulator.Images(tag)
    if not events:
        return None
    event = events[-1]
    encoded = event.encoded_image_string
    if isinstance(encoded, str):
        encoded = encoded.encode("latin1")
    raw_bytes = len(encoded)
    if raw_bytes > MAX_TENSORBOARD_IMAGE_SUMMARY_BYTES:
        return {
            "tag": tag,
            "step": int(event.step),
            "wallTime": finite_float(event.wall_time),
            "mimeType": "image/png",
            "dataUrl": "",
            **_truncated_payload_metadata(
                raw_bytes=raw_bytes,
                limit=MAX_TENSORBOARD_IMAGE_SUMMARY_BYTES,
                reason="image payload omitted",
            ),
        }
    data = base64.b64encode(encoded).decode("ascii")
    return {
        "tag": tag,
        "step": int(event.step),
        "wallTime": finite_float(event.wall_time),
        "mimeType": "image/png",
        "dataUrl": f"data:image/png;base64,{data}",
        "eventBytes": raw_bytes,
        "truncated": False,
        "sourceItemCount": 1,
        "returnedItemCount": 1,
    }


def text_summary(accumulator, tag: str) -> dict[str, Any] | None:
    """Read the latest TensorBoard text summary for ``tag``."""
    events = accumulator.Tensors(tag)
    if not events:
        return None
    event = events[-1]
    values = list(getattr(event.tensor_proto, "string_val", []))
    if not values:
        return None
    value = values[0]
    if isinstance(value, bytes):
        text = value.decode("utf-8", errors="replace")
    else:
        text = str(value)
    source_chars = len(text)
    source_bytes = len(text.encode("utf-8", errors="replace"))
    truncated = source_chars > MAX_TENSORBOARD_TEXT_SUMMARY_CHARS
    if truncated:
        text = text[:MAX_TENSORBOARD_TEXT_SUMMARY_CHARS]
    return {
        "tag": tag,
        "step": int(event.step),
        "wallTime": finite_float(event.wall_time),
        "text": text,
        "eventBytes": source_bytes,
        "truncated": truncated,
        "truncationReason": (
            "text payload truncated: "
            f"{source_chars} chars exceeds "
            f"{MAX_TENSORBOARD_TEXT_SUMMARY_CHARS} char cap"
            if truncated
            else None
        ),
        "sourceItemCount": source_chars,
        "returnedItemCount": len(text),
    }
