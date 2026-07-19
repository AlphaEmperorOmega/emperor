from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from model_runtime.runs.errors import InvalidCheckpointContinuation


@dataclass(frozen=True, slots=True)
class CheckpointContinuation:
    checkpoint_path: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "checkpoint_path", Path(self.checkpoint_path))


@dataclass(frozen=True, slots=True)
class _LoadedCheckpointContinuation:
    request: CheckpointContinuation
    state_dict: Mapping[str, Any]
    epoch: int
    global_step: int


def validate_checkpoint_file(
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


def load_checkpoint_continuation(
    continuation: CheckpointContinuation,
) -> _LoadedCheckpointContinuation:
    validate_checkpoint_file(continuation)
    path = continuation.checkpoint_path
    try:
        payload = torch.load(
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
    version = payload.get("pytorch-lightning_version")
    if not isinstance(version, str) or not version.strip():
        raise InvalidCheckpointContinuation(
            f"Checkpoint '{path}' must contain a nonempty Lightning version."
        )
    state_dict = payload.get("state_dict")
    if not isinstance(state_dict, Mapping) or not state_dict:
        raise InvalidCheckpointContinuation(
            f"Checkpoint '{path}' must contain a nonempty state_dict."
        )
    epoch = payload.get("epoch")
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
        raise InvalidCheckpointContinuation(
            f"Checkpoint '{path}' must contain a nonnegative epoch."
        )
    global_step = payload.get("global_step")
    if (
        isinstance(global_step, bool)
        or not isinstance(global_step, int)
        or global_step < 0
    ):
        raise InvalidCheckpointContinuation(
            f"Checkpoint '{path}' must contain a nonnegative global_step."
        )
    optimizer_states = payload.get("optimizer_states")
    if not isinstance(optimizer_states, (list, tuple)) or not optimizer_states:
        raise InvalidCheckpointContinuation(
            f"Checkpoint '{path}' must contain nonempty optimizer_states."
        )
    return _LoadedCheckpointContinuation(
        request=continuation,
        state_dict=state_dict,
        epoch=epoch,
        global_step=global_step,
    )


def validate_target_epochs(
    continuation: _LoadedCheckpointContinuation,
    target_epochs: int,
) -> None:
    completed_epochs = continuation.epoch + 1
    if target_epochs <= completed_epochs:
        raise InvalidCheckpointContinuation(
            f"Target NUM_EPOCHS ({target_epochs}) must be greater than the "
            f"checkpoint's completed epochs ({completed_epochs}; saved epoch "
            f"{continuation.epoch})."
        )


def validate_model_state(
    continuation: _LoadedCheckpointContinuation,
    model: Any,
) -> None:
    if isinstance(model, Mapping):
        _validate_model_state_mapping(continuation, model)
        return

    load_state_dict = getattr(model, "load_state_dict", None)
    state_dict = getattr(model, "state_dict", None)
    if not callable(load_state_dict) or not callable(state_dict):
        raise TypeError("Checkpoint model validation requires a torch module.")
    try:
        load_state_dict(continuation.state_dict, strict=True)
    except RuntimeError as exc:
        _validate_model_state_mapping(continuation, state_dict())
        raise InvalidCheckpointContinuation(
            "Checkpoint model state could not be loaded strictly into the "
            f"selected Model Package: {exc}"
        ) from exc


def _validate_model_state_mapping(
    continuation: _LoadedCheckpointContinuation,
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
    mismatches = []
    for key, model_value in model_state_dict.items():
        checkpoint_value = continuation.state_dict[key]
        checkpoint_shape = getattr(checkpoint_value, "shape", None)
        model_shape = getattr(model_value, "shape", None)
        if checkpoint_shape is None or tuple(checkpoint_shape) != tuple(model_shape):
            mismatches.append(
                f"{key}: checkpoint={checkpoint_shape}, model={model_shape}"
            )
    if mismatches:
        raise InvalidCheckpointContinuation(
            "Checkpoint tensor shapes do not exactly match the selected Model "
            f"Package ({'; '.join(mismatches)})."
        )


def resumed_from_payload(
    continuation: _LoadedCheckpointContinuation,
) -> dict[str, str | int]:
    return {
        "checkpoint": continuation.request.checkpoint_path.name,
        "epoch": continuation.epoch,
        "globalStep": continuation.global_step,
    }


__all__ = ["CheckpointContinuation"]
