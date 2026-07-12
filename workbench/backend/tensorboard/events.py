"""Low-level TensorBoard event-file access shared by the monitor readers.

Both :class:`~workbench.backend.tensorboard.readers.TensorBoardMonitorReader` and the
Run History query implementation read the same event files; these
helpers are the single implementation of that access so the two stay in step.
"""

from __future__ import annotations

import base64
import math
import os
import shutil
import stat
import sys
import tempfile
from collections import OrderedDict, deque
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Any

from tensorboard.backend.event_processing import event_accumulator, event_file_loader

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
DEFAULT_TENSORBOARD_EVENT_READ_BUDGET = 64 * 1024 * 1024
DEFAULT_TENSORBOARD_CACHE_MAX_BYTES = 128 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class EventFileIndex:
    root: Path
    dirs: tuple[Path, ...]
    files: tuple[Path, ...]
    fingerprint: EventFileFingerprint
    total_size: int
    complete: bool = True
    file_metadata: tuple[tuple[int, int], ...] = ()

    def cache_key(self, *parts: Any) -> tuple[Any, ...]:
        return (self.root.as_posix(), self.complete, self.fingerprint, *parts)

    def exceeds(self, byte_limit: int) -> bool:
        return byte_limit > 0 and self.total_size > byte_limit

    def load_accumulator(
        self,
        event_dir: Path,
        *,
        size_guidance: dict[int, int] | None = None,
    ) -> Any | None:
        """Load one directory from this contained observation only."""
        if not self.complete or event_dir not in self.dirs:
            return None
        selected = tuple(
            (path, metadata)
            for path, metadata in zip(
                self.files,
                self.file_metadata,
                strict=True,
            )
            if path.parent == event_dir
        )
        if not selected:
            return None
        for path, (expected_size, expected_modified_at) in selected:
            try:
                if path.resolve(strict=True) != path:
                    return None
                path.relative_to(self.root)
                observed = path.stat()
            except (OSError, ValueError):
                return None
            if (
                not stat.S_ISREG(observed.st_mode)
                or int(observed.st_size) != expected_size
                or int(observed.st_mtime_ns) != expected_modified_at
            ):
                return None
        explicit_files = tuple(path for path, _metadata in selected)
        explicit_metadata = tuple(metadata for _path, metadata in selected)
        if size_guidance is None:
            return load_event_accumulator(
                event_dir,
                event_files=explicit_files,
                event_metadata=explicit_metadata,
                trusted_root=self.root,
            )
        return load_event_accumulator(
            event_dir,
            size_guidance=size_guidance,
            event_files=explicit_files,
            event_metadata=explicit_metadata,
            trusted_root=self.root,
        )


class TensorBoardEventCache:
    """Generation-safe root-keyed LRU cache group for event projections."""

    def __init__(
        self,
        limits: Mapping[str, int],
        *,
        max_bytes: int = DEFAULT_TENSORBOARD_CACHE_MAX_BYTES,
    ) -> None:
        if not limits or any(limit < 1 for limit in limits.values()):
            raise ValueError("TensorBoard event cache limits must be positive.")
        self._limits = dict(limits)
        self._caches: dict[str, OrderedDict[tuple[Any, ...], Any]] = {
            name: OrderedDict() for name in limits
        }
        self._max_bytes = max(1, int(max_bytes))
        self._current_weight_bytes = 0
        self._weights: dict[tuple[str, tuple[Any, ...]], int] = {}
        self._global_lru: OrderedDict[
            tuple[str, tuple[Any, ...]],
            None,
        ] = OrderedDict()
        self._generation = 0
        self._lock = RLock()

    @property
    def current_weight_bytes(self) -> int:
        with self._lock:
            return self._current_weight_bytes

    def token(self) -> int:
        with self._lock:
            return self._generation

    def get(self, cache_name: str, key: tuple[Any, ...]) -> Any | None:
        with self._lock:
            cache = self._caches[cache_name]
            if key not in cache:
                return None
            cache.move_to_end(key)
            global_key = (cache_name, key)
            if global_key in self._global_lru:
                self._global_lru.move_to_end(global_key)
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
            self._remove(cache_name, key)
            weight = _cache_entry_weight(key, value)
            if weight > self._max_bytes:
                return
            cache[key] = value
            cache.move_to_end(key)
            global_key = (cache_name, key)
            self._weights[global_key] = weight
            self._global_lru[global_key] = None
            self._current_weight_bytes += weight
            while len(cache) > self._limits[cache_name]:
                oldest_key = next(iter(cache))
                self._remove(cache_name, oldest_key)
            while self._current_weight_bytes > self._max_bytes and self._global_lru:
                oldest_cache_name, oldest_key = next(iter(self._global_lru))
                self._remove(oldest_cache_name, oldest_key)

    def _remove(self, cache_name: str, key: tuple[Any, ...]) -> None:
        cache = self._caches[cache_name]
        cache.pop(key, None)
        global_key = (cache_name, key)
        weight = self._weights.pop(global_key, 0)
        self._current_weight_bytes = max(
            0,
            self._current_weight_bytes - weight,
        )
        self._global_lru.pop(global_key, None)

    def clear_roots(self, roots: set[str]) -> None:
        if not roots:
            return
        with self._lock:
            self._generation += 1
            for cache_name, cache in self._caches.items():
                for key in list(cache):
                    cached_root = str(key[0]) if key else ""
                    if any(
                        cached_root == root or cached_root.startswith(f"{root}/")
                        for root in roots
                    ):
                        self._remove(cache_name, key)

    def clear(self) -> None:
        with self._lock:
            self._generation += 1
            for cache in self._caches.values():
                cache.clear()
            self._weights.clear()
            self._global_lru.clear()
            self._current_weight_bytes = 0


def _cache_entry_weight(key: tuple[Any, ...], value: Any) -> int:
    seen: set[int] = set()
    return max(
        1,
        _estimated_object_bytes(key, seen) + _estimated_object_bytes(value, seen),
        _event_fingerprint_bytes(key),
    )


def _estimated_object_bytes(value: Any, seen: set[int]) -> int:
    identity = id(value)
    if identity in seen:
        return 0
    seen.add(identity)
    if isinstance(value, str):
        return len(value.encode("utf-8"))
    if isinstance(value, (bytes, bytearray, memoryview)):
        return len(value)
    if value is None or isinstance(value, (bool, int, float)):
        return 8
    if isinstance(value, Mapping):
        return 2 + sum(
            _estimated_object_bytes(key, seen) + _estimated_object_bytes(item, seen)
            for key, item in value.items()
        )
    if isinstance(value, (tuple, list, set, frozenset, deque)):
        return 2 + sum(_estimated_object_bytes(item, seen) for item in value)
    return sys.getsizeof(value)


def _event_fingerprint_bytes(value: Any) -> int:
    if (
        isinstance(value, tuple)
        and value
        and all(
            isinstance(item, tuple)
            and len(item) == 3
            and isinstance(item[0], str)
            and isinstance(item[1], int)
            and isinstance(item[2], int)
            for item in value
        )
    ):
        return sum(item[1] for item in value)
    if isinstance(value, (tuple, list)):
        return max(
            (_event_fingerprint_bytes(item) for item in value),
            default=0,
        )
    return 0


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
    complete: bool = True,
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
            complete=complete,
        )

    files_by_dir: dict[Path, list[tuple[Path, Path, int, int]]] = {}
    unsafe_dirs: set[Path] = set()
    paths = (
        candidates
        if candidates is not None
        else tuple(root.rglob("events.out.tfevents.*"))
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
    file_metadata: list[tuple[int, int]] = []
    for event_dir in sorted(files_by_dir):
        if event_dir in unsafe_dirs:
            continue
        dirs.append(event_dir)
        for path, resolved, size, modified_at in sorted(
            files_by_dir[event_dir],
            key=lambda item: item[0],
        ):
            files.append(resolved)
            file_metadata.append((size, modified_at))
            total += size
            fingerprint.append((path.as_posix(), size, modified_at))
    return EventFileIndex(
        root=root,
        dirs=tuple(dirs),
        files=tuple(files),
        fingerprint=tuple(sorted(fingerprint)),
        total_size=total,
        complete=complete,
        file_metadata=tuple(file_metadata),
    )


class _ExplicitEventAccumulator:
    def __init__(self, accumulators: list[Any]) -> None:
        self._accumulators = accumulators

    def Tags(self) -> dict[str, Any]:
        tags_by_key: dict[str, Any] = {}
        for accumulator in self._accumulators:
            for key, values in accumulator.Tags().items():
                if isinstance(values, list):
                    current = tags_by_key.setdefault(key, set())
                    current.update(values)
                elif isinstance(values, bool):
                    tags_by_key[key] = bool(tags_by_key.get(key)) or values
                elif key not in tags_by_key:
                    tags_by_key[key] = values
        return {
            key: sorted(value) if isinstance(value, set) else value
            for key, value in tags_by_key.items()
        }

    def _events(self, method: str, tag: str) -> list[Any]:
        events: list[Any] = []
        for accumulator in self._accumulators:
            try:
                events.extend(getattr(accumulator, method)(tag))
            except (KeyError, ValueError):
                continue
        events.sort(
            key=lambda event: (
                float(getattr(event, "wall_time", 0.0)),
                int(getattr(event, "step", 0)),
            )
        )
        return events

    def Scalars(self, tag: str) -> list[Any]:
        return self._events("Scalars", tag)

    def Histograms(self, tag: str) -> list[Any]:
        return self._events("Histograms", tag)

    def Images(self, tag: str) -> list[Any]:
        return self._events("Images", tag)

    def Tensors(self, tag: str) -> list[Any]:
        return self._events("Tensors", tag)


def _copy_observed_event_file(
    path: Path,
    *,
    trusted_root: Path,
    expected_size: int,
    expected_modified_at: int,
    destination: Path,
) -> bool:
    try:
        relative = path.relative_to(trusted_root)
    except ValueError:
        return False
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    file_flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC
    opened: list[int] = []
    source_fd: int | None = None
    try:
        directory_fd = os.open(trusted_root, directory_flags)
        opened.append(directory_fd)
        for part in relative.parts[:-1]:
            directory_fd = os.open(part, directory_flags, dir_fd=directory_fd)
            opened.append(directory_fd)
        source_fd = os.open(relative.parts[-1], file_flags, dir_fd=directory_fd)
        observed = os.fstat(source_fd)
        if (
            not stat.S_ISREG(observed.st_mode)
            or int(observed.st_size) != expected_size
            or int(observed.st_mtime_ns) != expected_modified_at
        ):
            return False
        with (
            os.fdopen(os.dup(source_fd), "rb") as source,
            destination.open("xb") as out,
        ):
            shutil.copyfileobj(source, out, length=1024 * 1024)
        return True
    except OSError:
        return False
    finally:
        if source_fd is not None:
            os.close(source_fd)
        for directory_fd in reversed(opened):
            os.close(directory_fd)


def exact_scalar_tails(
    index: EventFileIndex,
    tags: list[str],
    *,
    max_points: int,
    byte_budget: int = DEFAULT_TENSORBOARD_EVENT_READ_BUDGET,
) -> dict[str, dict[str, Any]]:
    """Stream explicit event files and retain each tag's exact final points."""

    requested_tags = list(dict.fromkeys(tags))
    point_limit = max(1, int(max_points))
    if not index.complete:
        return {
            tag: {"points": [], "sourcePointCount": 0, "truncated": False}
            for tag in requested_tags
        }
    if byte_budget > 0 and index.total_size > byte_budget:
        raise ValueError(
            "TensorBoard scalar event files exceed the shared "
            f"{byte_budget} byte read budget."
        )

    tails = {tag: deque(maxlen=point_limit) for tag in requested_tags}
    counts = {tag: 0 for tag in requested_tags}
    requested = set(requested_tags)
    observed_files = sorted(
        zip(index.files, index.file_metadata, strict=True),
        key=lambda item: (item[1][1], item[0].as_posix()),
    )
    with tempfile.TemporaryDirectory(prefix="workbench-scalar-events-") as temporary:
        temporary_root = Path(temporary)
        for file_index, (path, (size, modified_at)) in enumerate(observed_files):
            copy_path = temporary_root / f"events.out.tfevents.{file_index:08d}"
            if not _copy_observed_event_file(
                path,
                trusted_root=index.root,
                expected_size=size,
                expected_modified_at=modified_at,
                destination=copy_path,
            ):
                return {
                    tag: {
                        "points": [],
                        "sourcePointCount": 0,
                        "truncated": False,
                    }
                    for tag in requested_tags
                }
            loader = event_file_loader.LegacyEventFileLoader(str(copy_path))
            for event in loader.Load():
                if not event.HasField("summary"):
                    continue
                for value in event.summary.value:
                    if value.tag not in requested or not value.HasField("simple_value"):
                        continue
                    counts[value.tag] += 1
                    tails[value.tag].append(
                        {
                            "step": int(event.step),
                            "wallTime": finite_float(event.wall_time),
                            "value": finite_float(value.simple_value),
                        }
                    )

    return {
        tag: {
            "points": list(tails[tag]),
            "sourcePointCount": counts[tag],
            "truncated": counts[tag] > len(tails[tag]),
        }
        for tag in requested_tags
    }


def load_event_accumulator(
    run_dir: Path,
    *,
    size_guidance: dict[int, int] | None = None,
    event_files: tuple[Path, ...] | None = None,
    event_metadata: tuple[tuple[int, int], ...] | None = None,
    trusted_root: Path | None = None,
):
    """Load and reload an ``EventAccumulator``, or ``None`` if it cannot be read."""
    guidance = dict(size_guidance or DEFAULT_TENSORBOARD_SIZE_GUIDANCE)
    try:
        if event_files is None:
            accumulator = event_accumulator.EventAccumulator(
                str(run_dir),
                size_guidance=guidance,
            )
            accumulator.Reload()
            return accumulator
        if event_metadata is None or trusted_root is None:
            return None
        accumulators: list[Any] = []
        with tempfile.TemporaryDirectory(prefix="workbench-events-") as temporary:
            temporary_root = Path(temporary)
            for index, (path, (size, modified_at)) in enumerate(
                zip(event_files, event_metadata, strict=True)
            ):
                copy_path = temporary_root / f"events.out.tfevents.{index:08d}"
                if not _copy_observed_event_file(
                    path,
                    trusted_root=trusted_root,
                    expected_size=size,
                    expected_modified_at=modified_at,
                    destination=copy_path,
                ):
                    return None
                accumulator = event_accumulator.EventAccumulator(
                    str(copy_path),
                    size_guidance=guidance,
                )
                accumulator.Reload()
                accumulators.append(accumulator)
        return _ExplicitEventAccumulator(accumulators)
    except Exception:
        return None


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
