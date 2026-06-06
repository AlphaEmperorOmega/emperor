"""Low-level TensorBoard event-file access shared by the monitor readers.

Both :class:`~viewer.backend.monitor_data.TensorBoardMonitorReader` and
:class:`~viewer.backend.log_runs.LogRunIndex` read the same event files; these
helpers are the single implementation of that access so the two stay in step.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from tensorboard.backend.event_processing import event_accumulator


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
    event_files = list(root.rglob("events.out.tfevents.*"))
    if not event_files:
        return []
    return sorted({path.parent for path in event_files})


def load_event_accumulator(run_dir: Path):
    """Load and reload an ``EventAccumulator``, or ``None`` if it cannot be read."""
    try:
        accumulator = event_accumulator.EventAccumulator(
            str(run_dir),
            size_guidance={
                event_accumulator.SCALARS: 0,
                event_accumulator.HISTOGRAMS: 0,
                event_accumulator.IMAGES: 0,
            },
        )
        accumulator.Reload()
    except Exception:
        return None
    return accumulator
