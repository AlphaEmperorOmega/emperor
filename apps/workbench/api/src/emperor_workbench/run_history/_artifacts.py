from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from threading import RLock
from typing import Any

from model_runtime.runs import NonFiniteJsonValue, replace_non_finite_json

import emperor_workbench.tensorboard as tensorboard
from emperor_workbench.run_history._paths import (
    read_regular_file_beneath,
    resolved_under_root,
)
from emperor_workbench.tensorboard import EventFileIndex

HPARAM_INT_RE = re.compile(r"^[+-]?\d+$")
HPARAM_FLOAT_RE = re.compile(
    r"^[+-]?(?:(?:\d+\.\d*)|(?:\.\d+)|(?:\d+))(?:[eE][+-]?\d+)$"
    r"|^[+-]?(?:(?:\d+\.\d*)|(?:\.\d+))$"
)
DEFAULT_RUN_ARTIFACT_MAX_FILES = 10_000
DEFAULT_RUN_ARTIFACT_MAX_DEPTH = 16
DEFAULT_RUN_METADATA_FILE_MAX_BYTES = 4 * 1024 * 1024
LOGGER = logging.getLogger(__name__)
_CHECKPOINT_EPOCH_RE = re.compile(r"(?:^|[-_])epoch=(?P<value>\d+)(?:[-_]|$)")
_CHECKPOINT_STEP_RE = re.compile(r"(?:^|[-_])step=(?P<value>\d+)(?:[-_]|$)")


@dataclass(frozen=True, slots=True)
class RunArtifactBudgets:
    max_files: int = DEFAULT_RUN_ARTIFACT_MAX_FILES
    max_depth: int = DEFAULT_RUN_ARTIFACT_MAX_DEPTH
    max_metadata_file_bytes: int = DEFAULT_RUN_METADATA_FILE_MAX_BYTES

    def __post_init__(self) -> None:
        if self.max_files < 1:
            raise ValueError("Run Artifact max_files must be positive.")
        if self.max_depth < 0:
            raise ValueError("Run Artifact max_depth cannot be negative.")
        if self.max_metadata_file_bytes < 1:
            raise ValueError("Run Artifact max_metadata_file_bytes must be positive.")


DEFAULT_RUN_ARTIFACT_BUDGETS = RunArtifactBudgets()


@dataclass(frozen=True, slots=True)
class ObservedRunArtifact:
    path: Path
    relative_path: str
    size: int
    modified_at_ns: int

    @property
    def modified_at(self) -> str:
        return (
            datetime.fromtimestamp(self.modified_at_ns / 1_000_000_000, UTC)
            .isoformat()
            .replace("+00:00", "Z")
        )


class RunArtifactObservation:
    """One bounded, freshness-keyed view of a Run's readable artifacts."""

    def __init__(
        self,
        *,
        root: Path,
        run_dir: Path,
        budgets: RunArtifactBudgets,
        event_files: EventFileIndex,
        event_artifacts: tuple[ObservedRunArtifact, ...],
        result: ObservedRunArtifact | None,
        hparams: ObservedRunArtifact | None,
        checkpoints: tuple[ObservedRunArtifact, ...],
        fingerprint: tuple[tuple[Any, ...], ...],
        observed_entry_count: int,
        truncation_reasons: tuple[str, ...],
    ) -> None:
        self.root = root
        self.run_dir = run_dir
        self.budgets = budgets
        self.event_files = event_files
        self.event_artifacts = event_artifacts
        self.result = result
        self.hparams = hparams
        self.checkpoints = checkpoints
        self.fingerprint = fingerprint
        self.observed_entry_count = observed_entry_count
        self.truncation_reasons = truncation_reasons
        self._result_payload: dict[str, Any] | None = None
        self._result_loaded = False
        self._hparams_payload: dict[str, Any] | None = None
        self._hparams_loaded = False
        self._metadata_lock = RLock()

    @property
    def truncated(self) -> bool:
        return bool(self.truncation_reasons)

    @property
    def artifact_count(self) -> int:
        return (
            len(self.event_artifacts)
            + len(self.checkpoints)
            + int(self.result is not None)
            + int(self.hparams is not None)
        )

    def experiment_task(self) -> str | None:
        value = self._result().get("experimentTask")
        return value if isinstance(value, str) and value else None

    def metrics(self) -> dict[str, Any]:
        return self._result_object("metrics")

    def params(self) -> dict[str, Any]:
        return self._result_object("params")

    def hparams_values(self) -> dict[str, Any]:
        with self._metadata_lock:
            if not self._hparams_loaded:
                self._hparams_payload = (
                    _read_hparams_flat(
                        self.hparams.path,
                        max_bytes=self.budgets.max_metadata_file_bytes,
                        run_root=self.run_dir,
                        anchor_root=self.root,
                    )
                    if self.hparams is not None
                    else {}
                )
                self._hparams_loaded = True
            return copy.deepcopy(self._hparams_payload or {})

    def _result(self) -> dict[str, Any]:
        with self._metadata_lock:
            if not self._result_loaded:
                self._result_payload = (
                    _read_result_payload(
                        self.result.path,
                        max_bytes=self.budgets.max_metadata_file_bytes,
                        run_label=self.run_dir.relative_to(self.root).as_posix(),
                        run_root=self.run_dir,
                        anchor_root=self.root,
                    )
                    if self.result is not None
                    else {}
                )
                self._result_loaded = True
            return self._result_payload or {}

    def _result_object(self, key: str) -> dict[str, Any]:
        value = self._result().get(key)
        return copy.deepcopy(value) if isinstance(value, dict) else {}


def _read_bounded_text(
    path: Path,
    *,
    max_bytes: int,
    run_root: Path | None = None,
    anchor_root: Path | None = None,
) -> str | None:
    boundary = run_root or path.parent
    anchor = anchor_root or boundary
    raw = read_regular_file_beneath(
        path,
        boundary=boundary,
        anchor=anchor,
        max_bytes=max_bytes,
    )
    if raw is None:
        return None
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return None


def _read_result_payload(
    result_path: Path,
    *,
    max_bytes: int = DEFAULT_RUN_METADATA_FILE_MAX_BYTES,
    run_label: str | None = None,
    run_root: Path | None = None,
    anchor_root: Path | None = None,
) -> dict[str, Any]:
    try:
        text = _read_bounded_text(
            result_path,
            max_bytes=max_bytes,
            run_root=run_root,
            anchor_root=anchor_root,
        )
        payload = json.loads(text) if text is not None else None
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}

    def diagnose(invalid: NonFiniteJsonValue) -> None:
        LOGGER.warning(
            "Run Artifact contains a non-finite JSON number: %s",
            json.dumps(
                {
                    "code": "non_finite_run_artifact_value",
                    "run": run_label or result_path.parent.as_posix(),
                    "fieldPath": invalid.path,
                },
                sort_keys=True,
            ),
        )

    return replace_non_finite_json(payload, on_replace=diagnose)


def _parse_hparam_value(raw_value: str) -> bool | int | float | str | None:
    value = raw_value.strip()
    if not value:
        return ""
    if " #" in value:
        value = value.split(" #", 1)[0].strip()
    normalized = value.lower()
    if normalized in {"null", "none", "~"}:
        return None
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    if HPARAM_INT_RE.fullmatch(value):
        try:
            return int(value)
        except ValueError:
            return value
    if HPARAM_FLOAT_RE.fullmatch(value):
        try:
            return float(value)
        except ValueError:
            return value
    return value


def _read_hparams_flat(
    hparams_path: Path,
    *,
    max_bytes: int = DEFAULT_RUN_METADATA_FILE_MAX_BYTES,
    run_root: Path | None = None,
    anchor_root: Path | None = None,
) -> dict[str, Any]:
    text = _read_bounded_text(
        hparams_path,
        max_bytes=max_bytes,
        run_root=run_root,
        anchor_root=anchor_root,
    )
    if text is None:
        return {}
    lines = text.splitlines()

    values: dict[str, Any] = {}
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, raw_value = line.split(":", 1)
        key = key.strip()
        value = raw_value.strip()
        if (
            not key
            or not value
            or value in {"|", ">"}
            or value.startswith(("[", "{", "- "))
        ):
            continue
        values[key] = _parse_hparam_value(value)
    return values


def _run_relative_file_label(run_dir: Path, path: Path) -> str:
    try:
        return path.relative_to(run_dir).as_posix()
    except ValueError:
        return path.name


def _observed_artifact(
    path: Path,
    *,
    root: Path,
    run_root: Path,
) -> ObservedRunArtifact | None:
    resolved = resolved_under_root(path, run_root)
    if resolved is None or not resolved.is_file():
        return None
    try:
        stat = resolved.stat()
        relative_path = resolved.relative_to(root).as_posix()
    except (OSError, ValueError):
        return None
    return ObservedRunArtifact(
        path=resolved,
        relative_path=relative_path,
        size=int(stat.st_size),
        modified_at_ns=int(stat.st_mtime_ns),
    )


def observe_run_artifacts(
    run_dir: Path,
    root: Path,
    *,
    budgets: RunArtifactBudgets = DEFAULT_RUN_ARTIFACT_BUDGETS,
) -> RunArtifactObservation:
    """Observe every readable Run Artifact through one bounded tree walk."""
    resolved_root = root.resolve()
    resolved_run = resolved_under_root(run_dir, resolved_root)
    if resolved_run is None or not resolved_run.is_dir():
        raise ValueError(f"Run Artifact root is not contained: {run_dir}")

    event_candidates: list[Path] = []
    checkpoints: list[ObservedRunArtifact] = []
    result: ObservedRunArtifact | None = None
    hparams: ObservedRunArtifact | None = None
    fingerprint: list[tuple[Any, ...]] = []
    truncation_reasons: list[str] = []
    observed_entry_count = 0
    pending: list[tuple[Path, int]] = [(resolved_run, 0)]
    stop = False

    while pending and not stop:
        current, depth = pending.pop()
        try:
            current_stat = current.stat()
            current_relative = current.relative_to(resolved_run).as_posix()
            with os.scandir(current) as scanned:
                entries = sorted(scanned, key=lambda entry: entry.name)
        except (OSError, ValueError):
            truncation_reasons.append(
                "Run Artifact traversal encountered an unreadable directory."
            )
            continue
        fingerprint.append(
            (
                "dir",
                current_relative,
                int(current_stat.st_size),
                int(current_stat.st_mtime_ns),
            )
        )
        child_dirs: list[Path] = []
        for entry in entries:
            if observed_entry_count >= budgets.max_files:
                truncation_reasons.append(
                    "Run Artifact observation reached its filesystem item cap: "
                    f"{budgets.max_files}."
                )
                stop = True
                break
            observed_entry_count += 1
            candidate = Path(entry.path)
            if entry.name.startswith("events.out.tfevents."):
                event_candidates.append(candidate)
                continue
            try:
                is_directory = entry.is_dir(follow_symlinks=False)
                is_file = entry.is_file(follow_symlinks=False)
                is_symlink = entry.is_symlink()
            except OSError:
                continue
            if is_directory:
                if depth >= budgets.max_depth:
                    truncation_reasons.append(
                        "Run Artifact observation reached its recursion cap: "
                        f"{budgets.max_depth}."
                    )
                else:
                    child_dirs.append(candidate)
                continue
            if not is_file and not is_symlink:
                continue
            artifact = _observed_artifact(
                candidate,
                root=resolved_root,
                run_root=resolved_run,
            )
            if artifact is None:
                continue
            try:
                run_relative = candidate.relative_to(resolved_run)
            except ValueError:
                continue
            if len(run_relative.parts) == 1 and candidate.name == "result.json":
                result = artifact
            elif len(run_relative.parts) == 1 and candidate.name == "hparams.yaml":
                hparams = artifact
            elif candidate.suffix == ".ckpt":
                checkpoints.append(artifact)
            else:
                continue
            fingerprint.append(
                (
                    "file",
                    artifact.relative_path,
                    artifact.size,
                    artifact.modified_at_ns,
                )
            )
        pending.extend((path, depth + 1) for path in reversed(child_dirs))

    event_files = tensorboard.event_file_index(
        resolved_run,
        candidates=tuple(event_candidates),
        complete=not truncation_reasons,
    )
    event_artifacts = tuple(
        ObservedRunArtifact(
            path=path,
            relative_path=path.relative_to(resolved_root).as_posix(),
            size=size,
            modified_at_ns=modified_at_ns,
        )
        for path, (_fingerprint_path, size, modified_at_ns) in zip(
            event_files.files,
            event_files.fingerprint,
            strict=True,
        )
    )
    fingerprint.extend(
        (
            "event",
            artifact.relative_path,
            artifact.size,
            artifact.modified_at_ns,
        )
        for artifact in event_artifacts
    )
    for label, artifact in (("result.json", result), ("hparams.yaml", hparams)):
        if artifact is not None and artifact.size > budgets.max_metadata_file_bytes:
            truncation_reasons.append(
                f"{label} exceeds the Run metadata byte cap: "
                f"{artifact.size} > {budgets.max_metadata_file_bytes}."
            )

    return RunArtifactObservation(
        root=resolved_root,
        run_dir=resolved_run,
        budgets=budgets,
        event_files=event_files,
        event_artifacts=event_artifacts,
        result=result,
        hparams=hparams,
        checkpoints=tuple(
            sorted(checkpoints, key=lambda artifact: artifact.relative_path)
        ),
        fingerprint=tuple(sorted(fingerprint)),
        observed_entry_count=observed_entry_count,
        truncation_reasons=tuple(dict.fromkeys(truncation_reasons)),
    )


def _file_id(run_id: str, relative_path: str) -> str:
    return hashlib.sha256(f"{run_id}:{relative_path}".encode()).hexdigest()[:16]


def _parse_checkpoint_epoch(filename: str) -> int | None:
    return _parse_checkpoint_field(_CHECKPOINT_EPOCH_RE, filename)


def _parse_checkpoint_step(filename: str) -> int | None:
    return _parse_checkpoint_field(_CHECKPOINT_STEP_RE, filename)


def _parse_checkpoint_field(pattern: re.Pattern[str], filename: str) -> int | None:
    match = pattern.search(Path(filename).stem)
    return int(match.group("value")) if match is not None else None


__all__ = [
    "DEFAULT_RUN_ARTIFACT_BUDGETS",
    "DEFAULT_RUN_ARTIFACT_MAX_DEPTH",
    "DEFAULT_RUN_ARTIFACT_MAX_FILES",
    "DEFAULT_RUN_METADATA_FILE_MAX_BYTES",
    "ObservedRunArtifact",
    "RunArtifactBudgets",
    "RunArtifactObservation",
    "_file_id",
    "_parse_checkpoint_epoch",
    "_parse_checkpoint_step",
    "_run_relative_file_label",
    "observe_run_artifacts",
]
