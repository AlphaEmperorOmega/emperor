"""Low-level TensorBoard event-file access shared by the monitor readers.

Both :class:`~workbench.backend.tensorboard.readers.TensorBoardMonitorReader` and the
Run History query implementation read the same event files; these
helpers are the single implementation of that access so the two stay in step.
"""

from __future__ import annotations

import base64
import math
from dataclasses import dataclass
from pathlib import Path
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


@dataclass(frozen=True)
class EventFileIndex:
    dirs: tuple[Path, ...]
    fingerprint: EventFileFingerprint
    total_size: int


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


def event_dirs(root: Path) -> list[Path]:
    """Return the sorted, de-duplicated directories under ``root`` holding events."""
    return list(event_file_index(root).dirs)


def event_file_total_size(root: Path) -> int:
    """Return total bytes for TensorBoard event files below ``root``."""
    return event_file_index(root).total_size


def event_file_fingerprint(root: Path) -> EventFileFingerprint:
    """Return event-file identity data for cache invalidation."""
    return event_file_index(root).fingerprint


def event_file_index(root: Path) -> EventFileIndex:
    """Return a contained event-file index in one tree walk.

    ``EventAccumulator`` reads every matching event file in a directory. If any
    matching member resolves outside the requested root, the entire directory is
    ignored so a safe sibling cannot make the accumulator open the escaping file.
    """
    try:
        trusted_root = root.resolve(strict=True)
    except OSError:
        return EventFileIndex(dirs=(), fingerprint=(), total_size=0)

    files_by_dir: dict[Path, list[tuple[Path, int, int]]] = {}
    unsafe_dirs: set[Path] = set()
    for path in root.rglob("events.out.tfevents.*"):
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
            (path, int(stat.st_size), int(stat.st_mtime_ns))
        )

    dirs: list[Path] = []
    fingerprint: list[tuple[str, int, int]] = []
    total = 0
    for event_dir in sorted(files_by_dir):
        if event_dir in unsafe_dirs:
            continue
        dirs.append(event_dir)
        for path, size, modified_at in files_by_dir[event_dir]:
            total += size
            fingerprint.append((path.as_posix(), size, modified_at))
    return EventFileIndex(
        dirs=tuple(dirs),
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
