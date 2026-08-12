from __future__ import annotations

import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

import torch

from model_runtime.runs.errors import InvalidCheckpointContinuation


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


@dataclass(frozen=True, slots=True)
class _LoadedCheckpoint:
    request: CheckpointContinuation
    state_dict: Mapping[str, Any]
    epoch: int
    global_step: int


@dataclass(frozen=True, slots=True)
class CheckpointExecution:
    checkpoint_path: Path | None
    provenance: Mapping[str, object] | None
    model_validator: Callable[[object], None] | None


@dataclass(frozen=True, slots=True)
class CheckpointContinuationLifecycle:
    _loaded: _LoadedCheckpoint | None

    @classmethod
    def admit(
        cls,
        continuation: CheckpointContinuation | None,
        plan: _RunPlanView,
    ) -> CheckpointContinuationLifecycle:
        if continuation is None:
            return cls(None)
        if len(plan.runs) != 1:
            raise InvalidCheckpointContinuation(
                "Checkpoint continuation requires a Run Plan containing exactly "
                "one Run."
            )
        return cls(_load_checkpoint(continuation))

    def bind_training_runs(
        self,
        training_runs: Sequence[_TrainingRunView],
    ) -> CheckpointExecution:
        loaded = self._loaded
        if loaded is None:
            return CheckpointExecution(None, None, None)
        _validate_target_epochs(loaded, training_runs[0].num_epochs)
        return CheckpointExecution(
            checkpoint_path=loaded.request.checkpoint_path,
            provenance=_resumed_from_payload(loaded),
            model_validator=self._validate_model,
        )

    def _validate_model(self, model: object) -> None:
        loaded = self._loaded
        if loaded is None:
            return
        _validate_model_state(loaded, model)


def _validate_checkpoint_file(
    continuation: CheckpointContinuation,
) -> CheckpointContinuation:
    path = continuation.checkpoint_path
    if not path.is_file() or not os.access(path, os.R_OK):
        raise InvalidCheckpointContinuation(
            f"Checkpoint '{path}' must be a readable regular file."
        )
    try:
        with path.open("rb"):
            pass
    except OSError as exc:
        raise InvalidCheckpointContinuation(
            f"Checkpoint '{path}' must be a readable regular file: {exc}"
        ) from exc
    return continuation


def _checkpoint_counter(
    checkpoint_payload: Mapping[object, object],
    path: Path,
    field: str,
) -> int:
    value = checkpoint_payload.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise InvalidCheckpointContinuation(
            f"Checkpoint '{path}' must contain a nonnegative {field}."
        )
    return value


def _load_checkpoint(
    continuation: CheckpointContinuation,
) -> _LoadedCheckpoint:
    _validate_checkpoint_file(continuation)
    path = continuation.checkpoint_path
    try:
        payload: object = cast(Any, torch).load(
            path,
            map_location="cpu",
            weights_only=True,
        )
    except Exception as exc:
        raise InvalidCheckpointContinuation(
            f"Checkpoint '{path}' could not be loaded as a Lightning checkpoint: {exc}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise InvalidCheckpointContinuation(
            f"Checkpoint '{path}' must contain a mapping payload."
        )
    checkpoint_payload = cast(Mapping[object, object], payload)
    version = checkpoint_payload.get("pytorch-lightning_version")
    if not isinstance(version, str) or not version.strip():
        raise InvalidCheckpointContinuation(
            f"Checkpoint '{path}' must contain a nonempty Lightning version."
        )
    state_dict = checkpoint_payload.get("state_dict")
    if not isinstance(state_dict, Mapping) or not state_dict:
        raise InvalidCheckpointContinuation(
            f"Checkpoint '{path}' must contain a nonempty state_dict."
        )
    state_mapping = cast(Mapping[object, Any], state_dict)
    if any(not isinstance(key, str) for key in state_mapping):
        raise InvalidCheckpointContinuation(
            f"Checkpoint '{path}' state_dict keys must be strings."
        )
    typed_state_dict = {cast(str, key): value for key, value in state_mapping.items()}
    epoch = _checkpoint_counter(checkpoint_payload, path, "epoch")
    global_step = _checkpoint_counter(checkpoint_payload, path, "global_step")
    optimizer_states = checkpoint_payload.get("optimizer_states")
    if not isinstance(optimizer_states, (list, tuple)) or not optimizer_states:
        raise InvalidCheckpointContinuation(
            f"Checkpoint '{path}' must contain nonempty optimizer_states."
        )
    return _LoadedCheckpoint(
        request=continuation,
        state_dict=typed_state_dict,
        epoch=epoch,
        global_step=global_step,
    )


def _validate_target_epochs(
    continuation: _LoadedCheckpoint,
    target_epochs: int,
) -> None:
    completed_epochs = continuation.epoch + 1
    if target_epochs <= completed_epochs:
        raise InvalidCheckpointContinuation(
            f"Target NUM_EPOCHS ({target_epochs}) must be greater than the "
            f"checkpoint's completed epochs ({completed_epochs}; saved epoch "
            f"{continuation.epoch})."
        )


def _validate_model_state(
    continuation: _LoadedCheckpoint,
    model: Any,
) -> None:
    if isinstance(model, Mapping):
        _validate_model_state_mapping(
            continuation,
            cast(Mapping[str, Any], model),
        )
        return

    load_state_dict = getattr(model, "load_state_dict", None)
    state_dict = getattr(model, "state_dict", None)
    if not callable(load_state_dict) or not callable(state_dict):
        raise TypeError("Checkpoint model validation requires a torch module.")
    try:
        load_state_dict(continuation.state_dict, strict=True)
    except RuntimeError as exc:
        current_state = state_dict()
        if not isinstance(current_state, Mapping):
            raise TypeError(
                "Checkpoint model state_dict must return a mapping."
            ) from exc
        _validate_model_state_mapping(
            continuation,
            cast(Mapping[str, Any], current_state),
        )
        raise InvalidCheckpointContinuation(
            "Checkpoint model state could not be loaded strictly into the "
            f"selected Model Package: {exc}"
        ) from exc


def _validate_model_state_mapping(
    continuation: _LoadedCheckpoint,
    model_state_dict: Mapping[str, Any],
) -> None:
    checkpoint_keys = set(continuation.state_dict)
    model_keys = set(model_state_dict)
    if checkpoint_keys != model_keys:
        missing = sorted(str(key) for key in model_keys - checkpoint_keys)
        unexpected = sorted(str(key) for key in checkpoint_keys - model_keys)
        raise InvalidCheckpointContinuation(
            "Checkpoint model state keys do not exactly match the selected Model "
            f"Package (missing={missing}, unexpected={unexpected})."
        )
    mismatches: list[str] = []
    for key, model_value in model_state_dict.items():
        checkpoint_value = continuation.state_dict[key]
        checkpoint_shape = getattr(checkpoint_value, "shape", None)
        model_shape = getattr(model_value, "shape", None)
        if (
            checkpoint_shape is None
            or model_shape is None
            or tuple(cast(Sequence[object], checkpoint_shape))
            != tuple(cast(Sequence[object], model_shape))
        ):
            mismatches.append(
                f"{key}: checkpoint={checkpoint_shape}, model={model_shape}"
            )
    if mismatches:
        raise InvalidCheckpointContinuation(
            "Checkpoint tensor shapes do not exactly match the selected Model "
            f"Package ({'; '.join(mismatches)})."
        )


def _resumed_from_payload(
    continuation: _LoadedCheckpoint,
) -> dict[str, str | int]:
    return {
        "checkpoint": continuation.request.checkpoint_path.name,
        "epoch": continuation.epoch,
        "globalStep": continuation.global_step,
    }


__all__ = ["CheckpointContinuation"]
