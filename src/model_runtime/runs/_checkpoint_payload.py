from __future__ import annotations

import hashlib
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NoReturn, cast

import torch

from model_runtime.runs._checkpoint_receipt import (
    CheckpointArchiveReceipt,
    CheckpointPayloadReceipt,
)
from model_runtime.runs.checkpoint_admission import CheckpointAdmissionPolicy
from model_runtime.runs.errors import InvalidCheckpointContinuation

_MAX_PROTOCOL_INTEGER = 2**63 - 1


@dataclass(frozen=True, slots=True)
class ValidatedCheckpointPayload:
    state_dict: Mapping[str, Any]
    receipt: CheckpointPayloadReceipt


@dataclass(slots=True)
class _ResourceTotals:
    payload_nodes: int = 0
    maximum_container_depth: int = 0
    tensor_count: int = 0
    tensor_dimensions: int = 0
    tensor_elements: int = 0
    tensor_bytes: int = 0
    unique_storage_bytes: int = 0
    scalar_bytes: int = 0


@dataclass(frozen=True, slots=True)
class _CheckpointSchema:
    version_bytes: bytes
    state_dict: Mapping[str, Any]
    epoch: int
    completed_epochs: int
    global_step: int
    optimizer_states: int


def _integer_set() -> set[int]:
    return set()


def _storage_identity_set() -> set[tuple[str, int, int]]:
    return set()


def _checkpoint_counter(
    checkpoint_payload: Mapping[object, object],
    path: Path,
    field: str,
) -> int:
    value = checkpoint_payload.get(field)
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
        or value > _MAX_PROTOCOL_INTEGER
    ):
        raise InvalidCheckpointContinuation(
            f"Checkpoint '{path}' must contain a bounded nonnegative {field}."
        )
    return value


def _completed_epochs(
    checkpoint_payload: Mapping[object, object],
    path: Path,
    saved_epoch: int,
) -> int:
    loops = checkpoint_payload.get("loops")
    if loops is None:
        return _next_completed_epoch(path, saved_epoch)
    if not isinstance(loops, Mapping):
        raise InvalidCheckpointContinuation(
            f"Checkpoint '{path}' loops must be a mapping."
        )
    loops_mapping = cast(Mapping[object, object], loops)
    fit_loop = loops_mapping.get("fit_loop")
    if not isinstance(fit_loop, Mapping):
        return _next_completed_epoch(path, saved_epoch)
    fit_loop_mapping = cast(Mapping[object, object], fit_loop)
    epoch_progress = fit_loop_mapping.get("epoch_progress")
    if not isinstance(epoch_progress, Mapping):
        return _next_completed_epoch(path, saved_epoch)
    epoch_progress_mapping = cast(Mapping[object, object], epoch_progress)
    current = epoch_progress_mapping.get("current")
    if not isinstance(current, Mapping) or "completed" not in current:
        return _next_completed_epoch(path, saved_epoch)
    current_mapping = cast(Mapping[object, object], current)
    completed = current_mapping["completed"]
    if (
        isinstance(completed, bool)
        or not isinstance(completed, int)
        or completed < 0
        or completed > _MAX_PROTOCOL_INTEGER
    ):
        raise InvalidCheckpointContinuation(
            f"Checkpoint '{path}' must contain bounded nonnegative completed epochs."
        )
    return completed


def _next_completed_epoch(path: Path, saved_epoch: int) -> int:
    if saved_epoch == _MAX_PROTOCOL_INTEGER:
        raise InvalidCheckpointContinuation(
            f"Checkpoint '{path}' completed epochs exceed the supported limit."
        )
    return saved_epoch + 1


def _raise_limit(path: Path, detail: str) -> NoReturn:
    raise InvalidCheckpointContinuation(f"Checkpoint '{path}' {detail}")


def _validate_tensor_format(
    value: torch.Tensor,
    path: Path,
    policy: CheckpointAdmissionPolicy,
) -> None:
    if value.device.type != "cpu":
        _raise_limit(path, f"contains a tensor on unsupported device {value.device}.")
    if value.layout is not torch.strided:
        _raise_limit(path, f"contains a tensor with unsupported layout {value.layout}.")
    if value.is_quantized:
        _raise_limit(path, "contains an unsupported quantized tensor.")
    dtype_name = str(value.dtype)
    if dtype_name not in policy.allowed_dtypes:
        _raise_limit(path, f"contains a tensor with unsupported dtype {dtype_name}.")


def _record_tensor_dimensions(
    value: torch.Tensor,
    path: Path,
    policy: CheckpointAdmissionPolicy,
    totals: _ResourceTotals,
) -> None:
    dimensions = int(value.dim())
    if dimensions > policy.max_tensor_dimensions:
        _raise_limit(
            path,
            f"contains a tensor with {dimensions} dimensions, exceeding the "
            f"limit of {policy.max_tensor_dimensions}.",
        )
    totals.tensor_dimensions += dimensions
    if totals.tensor_dimensions > policy.max_aggregate_tensor_dimensions:
        _raise_limit(
            path,
            f"contains {totals.tensor_dimensions} aggregate tensor dimensions, "
            f"exceeding the limit of {policy.max_aggregate_tensor_dimensions}.",
        )


def _record_tensor_logical_size(
    value: torch.Tensor,
    path: Path,
    policy: CheckpointAdmissionPolicy,
    totals: _ResourceTotals,
) -> None:
    elements = int(value.numel())
    logical_bytes = elements * int(value.element_size())
    if elements > policy.max_tensor_elements:
        _raise_limit(
            path,
            f"contains a tensor with {elements} elements, exceeding the per-tensor "
            f"limit of {policy.max_tensor_elements}.",
        )
    if logical_bytes > policy.max_tensor_bytes:
        _raise_limit(
            path,
            f"contains a tensor with {logical_bytes} logical bytes, exceeding the "
            f"per-tensor limit of {policy.max_tensor_bytes} bytes.",
        )
    totals.tensor_elements += elements
    totals.tensor_bytes += logical_bytes
    if totals.tensor_elements > policy.max_aggregate_tensor_elements:
        _raise_limit(
            path,
            f"contains {totals.tensor_elements} aggregate tensor elements, "
            f"exceeding the limit of {policy.max_aggregate_tensor_elements}.",
        )
    if totals.tensor_bytes > policy.max_aggregate_tensor_bytes:
        _raise_limit(
            path,
            f"contains {totals.tensor_bytes} aggregate logical tensor bytes, "
            f"exceeding the limit of {policy.max_aggregate_tensor_bytes} bytes.",
        )


def _record_tensor_storage(
    value: torch.Tensor,
    path: Path,
    policy: CheckpointAdmissionPolicy,
    totals: _ResourceTotals,
    seen_storages: set[tuple[str, int, int]],
) -> None:
    storage = value.untyped_storage()
    storage_bytes = int(storage.nbytes())
    identity = (str(value.device), int(storage.data_ptr()), storage_bytes)
    if identity in seen_storages:
        return
    seen_storages.add(identity)
    totals.unique_storage_bytes += storage_bytes
    if totals.unique_storage_bytes > policy.max_unique_storage_bytes:
        _raise_limit(
            path,
            f"uses {totals.unique_storage_bytes} unique tensor storage bytes, "
            f"exceeding the limit of {policy.max_unique_storage_bytes} bytes.",
        )


def _mapping_children(value: Mapping[object, object]) -> Iterator[object]:
    for key, item in value.items():
        yield key
        yield item


def _sequence_children(value: list[object] | tuple[object, ...] | set[object] | frozenset[object]) -> Iterator[object]:
    yield from value


def _is_raw_storage(value: object) -> bool:
    return isinstance(value, torch.UntypedStorage) or type(value).__name__.endswith(
        "Storage"
    )


def _numeric_scalar_size(value: object) -> int | None:
    if value is None:
        return 0
    if isinstance(value, bool):
        return 1
    if isinstance(value, int):
        return max(1, (value.bit_length() + 8) // 8)
    if isinstance(value, float):
        return 8
    if isinstance(value, complex):
        return 16
    return None


def _text_scalar_size(
    value: object,
    path: Path,
    policy: CheckpointAdmissionPolicy,
) -> int | None:
    if isinstance(value, str):
        if len(value) > policy.max_scalar_bytes:
            _raise_limit(
                path,
                f"contains a scalar value longer than the per-value limit of "
                f"{policy.max_scalar_bytes} bytes.",
            )
        try:
            return len(value.encode("utf-8"))
        except UnicodeEncodeError as exc:
            raise InvalidCheckpointContinuation(
                f"Checkpoint '{path}' contains a string that is not valid UTF-8."
            ) from exc
    return None


def _buffer_scalar_size(value: object) -> int | None:
    if isinstance(value, (bytes, bytearray)):
        return len(value)
    if isinstance(value, memoryview):
        return value.nbytes
    return None


def _torch_scalar_size(value: object) -> int | None:
    safe_torch_types = (
        torch.device,
        torch.dtype,
        torch.layout,
        torch.memory_format,
        torch.qscheme,
    )
    if isinstance(value, safe_torch_types):
        return len(str(value).encode("utf-8"))
    return None


def _scalar_size_bytes(
    value: object,
    path: Path,
    policy: CheckpointAdmissionPolicy,
) -> int | None:
    numeric_bytes = _numeric_scalar_size(value)
    if numeric_bytes is not None:
        return numeric_bytes
    text_bytes = _text_scalar_size(value, path, policy)
    if text_bytes is not None:
        return text_bytes
    buffer_bytes = _buffer_scalar_size(value)
    if buffer_bytes is not None:
        return buffer_bytes
    return _torch_scalar_size(value)


@dataclass(slots=True)
class _PayloadInventory:
    path: Path
    policy: CheckpointAdmissionPolicy
    totals: _ResourceTotals = field(default_factory=_ResourceTotals)
    seen_containers: set[int] = field(default_factory=_integer_set)
    seen_tensors: set[int] = field(default_factory=_integer_set)
    seen_storages: set[tuple[str, int, int]] = field(
        default_factory=_storage_identity_set
    )
    seen_attribute_owners: set[int] = field(default_factory=_integer_set)

    def collect(self, payload: object) -> _ResourceTotals:
        iterators: list[Iterator[tuple[object, int]]] = [iter(((payload, 0),))]
        while iterators:
            try:
                value, depth = next(iterators[-1])
            except StopIteration:
                iterators.pop()
                continue
            self._record_value(value, depth, iterators)
        return self.totals

    def _record_value(
        self,
        value: object,
        depth: int,
        iterators: list[Iterator[tuple[object, int]]],
    ) -> None:
        self.totals.payload_nodes += 1
        if self.totals.payload_nodes > self.policy.max_payload_nodes:
            _raise_limit(
                self.path,
                f"contains more than {self.policy.max_payload_nodes} payload values.",
            )
        if torch.is_tensor(value):
            self._record_tensor(value, depth, iterators)
        elif _is_raw_storage(value):
            _raise_limit(self.path, "contains unsupported raw tensor storage.")
        elif isinstance(value, Mapping):
            mapping = cast(Mapping[object, object], value)
            self._record_container(mapping, depth, _mapping_children(mapping), iterators)
        elif isinstance(value, (list, tuple, set, frozenset)):
            sequence = cast(
                list[object] | tuple[object, ...] | set[object] | frozenset[object],
                value,
            )
            self._record_container(
                sequence,
                depth,
                _sequence_children(sequence),
                iterators,
            )
        else:
            self._record_scalar(value)
        self._record_attributes(cast(object, value), depth, iterators)

    def _record_attributes(
        self,
        value: object,
        depth: int,
        iterators: list[Iterator[tuple[object, int]]],
    ) -> None:
        identity = id(value)
        if identity in self.seen_attribute_owners:
            return
        self.seen_attribute_owners.add(identity)
        try:
            attributes = vars(value)
        except TypeError:
            return
        if attributes:
            attribute_mapping = cast(Mapping[object, object], attributes)
            iterators.append(iter(((attribute_mapping, depth + 1),)))

    def _record_container(
        self,
        container: object,
        depth: int,
        children: Iterator[object],
        iterators: list[Iterator[tuple[object, int]]],
    ) -> None:
        if depth > self.policy.max_container_depth:
            _raise_limit(
                self.path,
                "contains containers deeper than the limit of "
                f"{self.policy.max_container_depth}.",
            )
        self.totals.maximum_container_depth = max(
            self.totals.maximum_container_depth,
            depth,
        )
        identity = id(container)
        if identity in self.seen_containers:
            return
        self.seen_containers.add(identity)
        child_depth = depth + 1
        iterators.append((child, child_depth) for child in children)

    def _record_tensor(
        self,
        value: torch.Tensor,
        depth: int,
        iterators: list[Iterator[tuple[object, int]]],
    ) -> None:
        identity = id(value)
        if identity in self.seen_tensors:
            return
        self.seen_tensors.add(identity)
        self.totals.tensor_count += 1
        if self.totals.tensor_count > self.policy.max_tensor_count:
            _raise_limit(
                self.path,
                f"contains {self.totals.tensor_count} tensors, exceeding the "
                f"limit of {self.policy.max_tensor_count}.",
            )
        _record_tensor_dimensions(value, self.path, self.policy, self.totals)
        _validate_tensor_format(value, self.path, self.policy)
        _record_tensor_logical_size(value, self.path, self.policy, self.totals)
        _record_tensor_storage(
            value,
            self.path,
            self.policy,
            self.totals,
            self.seen_storages,
        )

    def _record_scalar(self, value: object) -> None:
        scalar_bytes = _scalar_size_bytes(value, self.path, self.policy)
        if scalar_bytes is None:
            value_type = type(value).__name__[:64]
            _raise_limit(
                self.path,
                f"contains an unsupported payload value of type {value_type}.",
            )
        if scalar_bytes > self.policy.max_scalar_bytes:
            _raise_limit(
                self.path,
                f"contains a scalar value of {scalar_bytes} bytes, exceeding "
                f"the per-value limit of {self.policy.max_scalar_bytes} bytes.",
            )
        self.totals.scalar_bytes += scalar_bytes
        if self.totals.scalar_bytes > self.policy.max_aggregate_scalar_bytes:
            _raise_limit(
                self.path,
                f"contains {self.totals.scalar_bytes} aggregate scalar bytes, "
                "exceeding the limit of "
                f"{self.policy.max_aggregate_scalar_bytes} bytes.",
            )


def _lightning_version_bytes(
    checkpoint_payload: Mapping[object, object],
    path: Path,
) -> bytes:
    version = checkpoint_payload.get("pytorch-lightning_version")
    if not isinstance(version, str) or len(version) > 256 or not version.strip():
        raise InvalidCheckpointContinuation(
            f"Checkpoint '{path}' must contain a nonempty Lightning version."
        )
    try:
        version_bytes = version.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise InvalidCheckpointContinuation(
            f"Checkpoint '{path}' Lightning version must be valid UTF-8."
        ) from exc
    if len(version_bytes) > 256:
        raise InvalidCheckpointContinuation(
            f"Checkpoint '{path}' Lightning version exceeds 256 UTF-8 bytes."
        )
    return version_bytes


def _state_dictionary(
    checkpoint_payload: Mapping[object, object],
    path: Path,
    policy: CheckpointAdmissionPolicy,
) -> Mapping[str, Any]:
    state_dict = checkpoint_payload.get("state_dict")
    if not isinstance(state_dict, Mapping) or not state_dict:
        raise InvalidCheckpointContinuation(
            f"Checkpoint '{path}' must contain a nonempty state_dict."
        )
    state_mapping = cast(Mapping[object, object], state_dict)
    inspected_state_keys = 0
    for key in state_mapping:
        inspected_state_keys += 1
        if inspected_state_keys > policy.max_payload_nodes:
            _raise_limit(
                path,
                "state_dict alone exceeds the payload value limit of "
                f"{policy.max_payload_nodes}.",
            )
        if not isinstance(key, str):
            raise InvalidCheckpointContinuation(
                f"Checkpoint '{path}' state_dict keys must be strings."
            )
    return cast(Mapping[str, Any], state_mapping)


def _checkpoint_schema(
    checkpoint_payload: Mapping[object, object],
    path: Path,
    policy: CheckpointAdmissionPolicy,
) -> _CheckpointSchema:
    version_bytes = _lightning_version_bytes(checkpoint_payload, path)
    state_dict = _state_dictionary(checkpoint_payload, path, policy)
    epoch = _checkpoint_counter(checkpoint_payload, path, "epoch")
    completed_epochs = _completed_epochs(checkpoint_payload, path, epoch)
    global_step = _checkpoint_counter(checkpoint_payload, path, "global_step")
    optimizer_states = checkpoint_payload.get("optimizer_states")
    if not isinstance(optimizer_states, (list, tuple)) or not optimizer_states:
        raise InvalidCheckpointContinuation(
            f"Checkpoint '{path}' must contain nonempty optimizer_states."
        )
    optimizer_sequence = cast(list[object] | tuple[object, ...], optimizer_states)
    return _CheckpointSchema(
        version_bytes=version_bytes,
        state_dict=state_dict,
        epoch=epoch,
        completed_epochs=completed_epochs,
        global_step=global_step,
        optimizer_states=len(optimizer_sequence),
    )


def _payload_receipt(
    schema: _CheckpointSchema,
    totals: _ResourceTotals,
    archive: CheckpointArchiveReceipt,
) -> CheckpointPayloadReceipt:
    return CheckpointPayloadReceipt(
        lightning_version_sha256=hashlib.sha256(schema.version_bytes).hexdigest(),
        epoch=schema.epoch,
        completed_epochs=schema.completed_epochs,
        global_step=schema.global_step,
        state_dict_keys=len(schema.state_dict),
        optimizer_states=schema.optimizer_states,
        archive_records=archive.records,
        archive_uncompressed_bytes=archive.uncompressed_bytes,
        payload_nodes=totals.payload_nodes,
        maximum_container_depth=totals.maximum_container_depth,
        tensor_count=totals.tensor_count,
        tensor_dimensions=totals.tensor_dimensions,
        tensor_elements=totals.tensor_elements,
        tensor_bytes=totals.tensor_bytes,
        unique_storage_bytes=totals.unique_storage_bytes,
        scalar_bytes=totals.scalar_bytes,
    )


def validate_checkpoint_payload(
    payload: object,
    path: Path,
    policy: CheckpointAdmissionPolicy,
    archive: CheckpointArchiveReceipt | None = None,
) -> ValidatedCheckpointPayload:
    """Validate Lightning schema and account every reachable payload value."""

    if not isinstance(payload, Mapping):
        raise InvalidCheckpointContinuation(
            f"Checkpoint '{path}' must contain a mapping payload."
        )
    checkpoint_payload = cast(Mapping[object, object], payload)
    schema = _checkpoint_schema(checkpoint_payload, path, policy)
    totals = _PayloadInventory(path, policy).collect(checkpoint_payload)
    archive_receipt = archive or CheckpointArchiveReceipt(0, 0)
    return ValidatedCheckpointPayload(
        state_dict=schema.state_dict,
        receipt=_payload_receipt(schema, totals, archive_receipt),
    )


__all__: list[str] = []
