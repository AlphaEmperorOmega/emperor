from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CheckpointArchiveReceipt:
    records: int
    uncompressed_bytes: int


@dataclass(frozen=True, slots=True)
class CheckpointPayloadReceipt:
    lightning_version_sha256: str
    epoch: int
    completed_epochs: int
    global_step: int
    state_dict_keys: int
    optimizer_states: int
    archive_records: int
    archive_uncompressed_bytes: int
    payload_nodes: int
    maximum_container_depth: int
    tensor_count: int
    tensor_dimensions: int
    tensor_elements: int
    tensor_bytes: int
    unique_storage_bytes: int
    scalar_bytes: int


__all__: list[str] = []
