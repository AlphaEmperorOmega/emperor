from __future__ import annotations

import unicodedata
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Protocol, cast

import torch

from model_runtime.runs._checkpoint_archive import checkpoint_archive_receipt
from model_runtime.runs._checkpoint_isolation import (
    decode_checkpoint_isolated,
    isolated_decode_available,
)
from model_runtime.runs._checkpoint_payload import (
    ValidatedCheckpointPayload,
    validate_checkpoint_payload,
)
from model_runtime.runs._checkpoint_receipt import CheckpointPayloadReceipt
from model_runtime.runs._checkpoint_snapshot import (
    CheckpointSnapshot,
    admit_checkpoint_snapshot,
)
from model_runtime.runs.checkpoint_admission import (
    DEFAULT_CHECKPOINT_ADMISSION_POLICY,
    CheckpointAdmissionPolicy,
)
from model_runtime.runs.errors import InvalidCheckpointContinuation

_DIAGNOSTIC_SAMPLE_SIZE = 8
_DIAGNOSTIC_TEXT_BYTES = 512


class _RunPlanView(Protocol):
    @property
    def runs(self) -> Sequence[object]: ...


class _TrainingRunView(Protocol):
    @property
    def num_epochs(self) -> int: ...


@dataclass(frozen=True, slots=True)
class CheckpointContinuation:
    checkpoint_path: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "checkpoint_path", Path(self.checkpoint_path))


@dataclass(slots=True)
class _LoadedCheckpoint:
    request: CheckpointContinuation
    snapshot: CheckpointSnapshot
    state_dict: Mapping[str, Any] | None
    epoch: int
    completed_epochs: int
    global_step: int

    def release_payload(self) -> None:
        self.state_dict = None


@dataclass(frozen=True, slots=True)
class CheckpointExecution:
    checkpoint_path: Path | None
    provenance: Mapping[str, object] | None
    strict_model_preloader: Callable[[object], None] | None


class CheckpointContinuationLifecycle:
    __slots__ = ("_loaded",)

    def __init__(self, loaded: _LoadedCheckpoint | None) -> None:
        self._loaded = loaded

    @classmethod
    def admit(
        cls,
        continuation: CheckpointContinuation | None,
        plan: _RunPlanView,
        *,
        admission_policy: CheckpointAdmissionPolicy = (
            DEFAULT_CHECKPOINT_ADMISSION_POLICY
        ),
    ) -> CheckpointContinuationLifecycle:
        if continuation is None:
            return cls(None)
        if len(plan.runs) != 1:
            raise InvalidCheckpointContinuation(
                "Checkpoint continuation requires a Run Plan containing exactly "
                "one Run."
            )
        return cls(_load_checkpoint(continuation, admission_policy))

    def __enter__(self) -> CheckpointContinuationLifecycle:
        return self

    def __exit__(
        self,
        _exc_type: object,
        exception: BaseException | None,
        _traceback: object,
    ) -> None:
        try:
            self.close()
        except Exception as cleanup_error:
            if exception is None:
                raise
            exception.add_note(
                "Checkpoint snapshot cleanup also failed: "
                f"{type(cleanup_error).__name__}."
            )

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def close(self) -> None:
        loaded = self._loaded
        if loaded is None:
            return
        loaded.release_payload()
        loaded.snapshot.close()
        self._loaded = None

    def bind_training_runs(
        self,
        training_runs: Sequence[_TrainingRunView],
    ) -> CheckpointExecution:
        loaded = self._loaded
        if loaded is None:
            return CheckpointExecution(None, None, None)
        _validate_target_epochs(loaded, training_runs[0].num_epochs)
        return CheckpointExecution(
            checkpoint_path=loaded.snapshot.path,
            provenance=_resumed_from_payload(loaded),
            strict_model_preloader=self._strict_preload_model,
        )

    def _strict_preload_model(self, model: object) -> None:
        loaded = self._loaded
        if loaded is None:
            raise RuntimeError("Checkpoint continuation lifecycle is closed.")
        state_dict = loaded.state_dict
        if state_dict is None:
            raise RuntimeError("Checkpoint strict model preload was already consumed.")
        _strict_preload_model_state(state_dict, model)
        loaded.release_payload()


def _checkpoint_snapshot(
    continuation: CheckpointContinuation,
    admission_policy: CheckpointAdmissionPolicy,
) -> CheckpointSnapshot:
    path = continuation.checkpoint_path
    return admit_checkpoint_snapshot(
        path,
        admission_policy.max_file_bytes,
        _display_path(path),
    )


def _load_checkpoint(
    continuation: CheckpointContinuation,
    admission_policy: CheckpointAdmissionPolicy,
) -> _LoadedCheckpoint:
    snapshot = _checkpoint_snapshot(continuation, admission_policy)
    path = continuation.checkpoint_path
    try:
        worker_receipt = _isolated_payload_receipt(snapshot, path, admission_policy)
        validated = _parent_payload(snapshot, path, admission_policy)
        if worker_receipt is not None and validated.receipt != worker_receipt:
            raise InvalidCheckpointContinuation(
                f"Checkpoint '{_display_path(path)}' decoder receipt did not match "
                "parent validation."
            )
        return _LoadedCheckpoint(
            request=continuation,
            snapshot=snapshot,
            state_dict=validated.state_dict,
            epoch=validated.receipt.epoch,
            completed_epochs=validated.receipt.completed_epochs,
            global_step=validated.receipt.global_step,
        )
    except BaseException as exception:
        try:
            snapshot.close()
        except Exception as cleanup_error:
            exception.add_note(
                "Checkpoint snapshot cleanup also failed: "
                f"{type(cleanup_error).__name__}."
            )
        if not isinstance(exception, Exception):
            raise
        if isinstance(exception, InvalidCheckpointContinuation):
            raise
        raise InvalidCheckpointContinuation(
            f"Checkpoint '{_display_path(path)}' continuation admission failed."
        ) from exception


def _isolated_payload_receipt(
    snapshot: CheckpointSnapshot,
    path: Path,
    admission_policy: CheckpointAdmissionPolicy,
) -> CheckpointPayloadReceipt | None:
    if isolated_decode_available():
        return decode_checkpoint_isolated(
            snapshot.path,
            path,
            snapshot.sha256,
            snapshot.size_bytes,
            admission_policy,
        )
    if admission_policy.require_isolated_decode:
        raise InvalidCheckpointContinuation(
            "Checkpoint continuation requires isolated decode, but hard decoder "
            "controls are unavailable on this platform."
        )
    return None


def _load_snapshot_payload(snapshot_file: BinaryIO, path: Path) -> object:
    try:
        return cast(Any, torch).load(
            snapshot_file,
            map_location="cpu",
            weights_only=True,
        )
    except Exception as exc:
        raise InvalidCheckpointContinuation(
            f"Checkpoint '{_display_path(path)}' could not be loaded as a "
            "Lightning checkpoint."
        ) from exc


def _parent_payload(
    snapshot: CheckpointSnapshot,
    path: Path,
    admission_policy: CheckpointAdmissionPolicy,
) -> ValidatedCheckpointPayload:
    snapshot_file = snapshot.open_verified(
        admission_policy.max_file_bytes,
        _display_path(path),
    )
    with snapshot_file:
        archive = checkpoint_archive_receipt(snapshot_file, path, admission_policy)
        snapshot_file.seek(0)
        payload = _load_snapshot_payload(snapshot_file, path)
    return validate_checkpoint_payload(payload, path, admission_policy, archive)


def _validate_target_epochs(
    continuation: _LoadedCheckpoint,
    target_epochs: int,
) -> None:
    completed_epochs = continuation.completed_epochs
    if target_epochs <= completed_epochs:
        raise InvalidCheckpointContinuation(
            f"Target NUM_EPOCHS ({target_epochs}) must be greater than the "
            f"checkpoint's completed epochs ({completed_epochs}; saved epoch "
            f"{continuation.epoch})."
        )


def _strict_preload_model_state(
    checkpoint_state: Mapping[str, Any],
    model: object,
) -> None:
    if isinstance(model, Mapping):
        _preflight_model_state(
            checkpoint_state,
            cast(Mapping[str, Any], model),
        )
        return
    load_state_dict = getattr(model, "load_state_dict", None)
    if not callable(load_state_dict):
        raise TypeError("Checkpoint model validation requires a torch module.")
    try:
        load_state_dict(checkpoint_state, strict=True)
    except RuntimeError as exc:
        _classify_model_load_failure(checkpoint_state, model, exc)
        detail = _bounded_text(str(exc), _DIAGNOSTIC_TEXT_BYTES)
        raise InvalidCheckpointContinuation(
            "Checkpoint model state could not be loaded strictly into the "
            f"selected Model Package: {detail}"
        ) from exc


def _classify_model_load_failure(
    checkpoint_state: Mapping[str, Any],
    model: object,
    load_error: RuntimeError,
) -> None:
    state_dict = getattr(model, "state_dict", None)
    if not callable(state_dict):
        return
    current_state = state_dict()
    if not isinstance(current_state, Mapping):
        return
    try:
        _preflight_model_state(
            checkpoint_state,
            cast(Mapping[str, Any], current_state),
        )
    except InvalidCheckpointContinuation as mismatch:
        raise mismatch from load_error


def _preflight_model_state(
    checkpoint_state: Mapping[str, Any],
    model_state: Mapping[str, Any],
) -> None:
    _preflight_model_state_keys(checkpoint_state, model_state)
    _preflight_model_state_shapes(checkpoint_state, model_state)


def _preflight_model_state_keys(
    checkpoint_state: Mapping[str, Any],
    model_state: Mapping[str, Any],
) -> None:
    missing_count = 0
    unexpected_count = 0
    missing_sample: list[str] = []
    unexpected_sample: list[str] = []
    for key in model_state:
        if key not in checkpoint_state:
            missing_count += 1
            if len(missing_sample) < _DIAGNOSTIC_SAMPLE_SIZE:
                missing_sample.append(_bounded_text(str(key), 64))
    for key in checkpoint_state:
        if key not in model_state:
            unexpected_count += 1
            if len(unexpected_sample) < _DIAGNOSTIC_SAMPLE_SIZE:
                unexpected_sample.append(_bounded_text(str(key), 64))
    if missing_count or unexpected_count:
        raise InvalidCheckpointContinuation(
            "Checkpoint model state keys do not exactly match the selected Model "
            f"Package (missing_count={missing_count}, "
            f"missing_sample={missing_sample}, unexpected_count={unexpected_count}, "
            f"unexpected_sample={unexpected_sample})."
        )


def _preflight_model_state_shapes(
    checkpoint_state: Mapping[str, Any],
    model_state: Mapping[str, Any],
) -> None:
    mismatch_count = 0
    mismatch_sample: list[str] = []
    for key, model_value in model_state.items():
        checkpoint_value = checkpoint_state[key]
        checkpoint_shape = getattr(checkpoint_value, "shape", None)
        model_shape = getattr(model_value, "shape", None)
        if checkpoint_shape is None or model_shape is None or checkpoint_shape != model_shape:
            mismatch_count += 1
            if len(mismatch_sample) < _DIAGNOSTIC_SAMPLE_SIZE:
                mismatch_sample.append(
                    f"{_bounded_text(str(key), 64)}: "
                    f"checkpoint={_shape_text(checkpoint_shape)}, "
                    f"model={_shape_text(model_shape)}"
                )
    if mismatch_count:
        raise InvalidCheckpointContinuation(
            "Checkpoint tensor shapes do not exactly match the selected Model "
            f"Package (mismatch_count={mismatch_count}, "
            f"mismatch_sample={mismatch_sample})."
        )


def _shape_text(shape: object) -> str:
    if shape is None:
        return "None"
    try:
        dimensions: list[str] = []
        for index, value in enumerate(cast(Sequence[object], shape)):
            if index == _DIAGNOSTIC_SAMPLE_SIZE:
                dimensions.append("...")
                break
            dimensions.append(_bounded_text(str(value), 16))
        return f"({', '.join(dimensions)})"
    except TypeError:
        return _bounded_text(str(shape), 64)


def _bounded_text(value: str, maximum_bytes: int) -> str:
    single_line = " ".join(
        "".join(
            "?" if unicodedata.category(character).startswith("C") else character
            for character in value
        ).split()
    )
    return single_line.encode("utf-8", errors="replace")[:maximum_bytes].decode(
        "utf-8",
        errors="ignore",
    )


def _display_path(path: Path) -> str:
    return _bounded_text(str(path), 4096)


def _resumed_from_payload(
    continuation: _LoadedCheckpoint,
) -> dict[str, str | int]:
    return {
        "checkpoint": continuation.request.checkpoint_path.name,
        "epoch": continuation.epoch,
        "globalStep": continuation.global_step,
        "sha256": continuation.snapshot.sha256,
    }


__all__ = ["CheckpointContinuation"]
