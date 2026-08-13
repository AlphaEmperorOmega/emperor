from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import NoReturn, cast

_MAX_PROTOCOL_INTEGER = 2**63 - 1
_MAX_WORKER_SECONDS = 24 * 60 * 60
_DTYPE_NAME = re.compile(r"torch\.[a-z0-9_]+")

_DEFAULT_ALLOWED_DTYPES = frozenset(
    {
        "torch.bfloat16",
        "torch.bool",
        "torch.complex128",
        "torch.complex64",
        "torch.float16",
        "torch.float32",
        "torch.float64",
        "torch.int16",
        "torch.int32",
        "torch.int64",
        "torch.int8",
        "torch.uint8",
    }
)


@dataclass(frozen=True, slots=True)
class CheckpointAdmissionPolicy:
    """Deployment-owned limits for checkpoint continuation admission.

    Requiring isolated decode fails closed unless every hard worker control is
    available. The portable fallback is intended only for trusted local files;
    services still need deployment-owned whole-job containment.
    """

    max_file_bytes: int = 4 * 1024**3
    max_payload_nodes: int = 2_000_000
    max_container_depth: int = 128
    max_scalar_bytes: int = 16 * 1024 * 1024
    max_aggregate_scalar_bytes: int = 256 * 1024 * 1024
    max_archive_record_bytes: int = 4 * 1024**3
    max_archive_uncompressed_bytes: int = 8 * 1024**3
    max_archive_records: int = 1_000_000
    max_tensor_count: int = 1_000_000
    max_tensor_dimensions: int = 64
    max_aggregate_tensor_dimensions: int = 1_000_000
    max_tensor_elements: int = 1_000_000_000
    max_tensor_bytes: int = 4 * 1024**3
    max_aggregate_tensor_elements: int = 2_000_000_000
    max_aggregate_tensor_bytes: int = 8 * 1024**3
    max_unique_storage_bytes: int = 8 * 1024**3
    worker_memory_bytes: int = 16 * 1024**3
    worker_cpu_seconds: int = 120
    worker_wall_timeout_seconds: float = 120.0
    allowed_dtypes: frozenset[str] = _DEFAULT_ALLOWED_DTYPES
    require_isolated_decode: bool = False

    def __post_init__(self) -> None:
        _validate_integer_fields(self)
        _validate_worker_timeouts(self)
        _validate_allowed_dtypes(self)
        _validate_aggregate_limits(self)
        require_isolated_decode = _runtime_value(self.require_isolated_decode)
        if not isinstance(require_isolated_decode, bool):
            raise TypeError(
                "Checkpoint continuation require_isolated_decode must be a boolean."
            )


_INTEGER_FIELDS = (
    "max_file_bytes",
    "max_payload_nodes",
    "max_container_depth",
    "max_scalar_bytes",
    "max_aggregate_scalar_bytes",
    "max_archive_record_bytes",
    "max_archive_uncompressed_bytes",
    "max_archive_records",
    "max_tensor_count",
    "max_tensor_dimensions",
    "max_aggregate_tensor_dimensions",
    "max_tensor_elements",
    "max_tensor_bytes",
    "max_aggregate_tensor_elements",
    "max_aggregate_tensor_bytes",
    "max_unique_storage_bytes",
    "worker_memory_bytes",
    "worker_cpu_seconds",
)


def _runtime_value(value: object) -> object:
    """Hide a typed field behind its runtime admission Interface."""

    return value


def _validate_integer_fields(policy: CheckpointAdmissionPolicy) -> None:
    for field_name in _INTEGER_FIELDS:
        value: object = getattr(policy, field_name)
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 1
            or value > _MAX_PROTOCOL_INTEGER
        ):
            raise ValueError(
                f"Checkpoint continuation {field_name} must be a positive signed "
                "63-bit integer."
            )


def _validate_worker_timeouts(policy: CheckpointAdmissionPolicy) -> None:
    if policy.worker_cpu_seconds > _MAX_WORKER_SECONDS:
        raise ValueError(
            "Checkpoint continuation worker_cpu_seconds must not exceed "
            f"{_MAX_WORKER_SECONDS}."
        )
    wall_timeout = _runtime_value(policy.worker_wall_timeout_seconds)
    if (
        isinstance(wall_timeout, bool)
        or not isinstance(wall_timeout, (int, float))
        or not math.isfinite(wall_timeout)
        or wall_timeout <= 0
        or wall_timeout > _MAX_WORKER_SECONDS
    ):
        raise ValueError(
            "Checkpoint continuation worker_wall_timeout_seconds must be positive, "
            f"finite, and at most {_MAX_WORKER_SECONDS}."
        )


def _validate_allowed_dtypes(policy: CheckpointAdmissionPolicy) -> None:
    allowed_dtypes = _runtime_value(policy.allowed_dtypes)
    if not isinstance(allowed_dtypes, frozenset):
        raise TypeError(
            "Checkpoint continuation allowed_dtypes must be a frozenset of dtype "
            "names."
        )
    candidates = cast(frozenset[object], allowed_dtypes)
    if len(candidates) > 64 or not candidates:
        _raise_invalid_dtypes()
    for candidate in candidates:
        if not isinstance(candidate, str):
            _raise_invalid_dtypes()
        try:
            encoded_dtype = candidate.encode("utf-8")
        except UnicodeEncodeError as exc:
            _raise_invalid_dtypes(exc)
        if len(encoded_dtype) > 64 or _DTYPE_NAME.fullmatch(candidate) is None:
            _raise_invalid_dtypes()


def _raise_invalid_dtypes(cause: Exception | None = None) -> NoReturn:
    error = ValueError(
        "Checkpoint continuation allowed_dtypes must contain at most 64 bounded "
        "torch dtype names."
    )
    if cause is None:
        raise error
    raise error from cause


def _validate_aggregate_limits(policy: CheckpointAdmissionPolicy) -> None:
    if policy.max_aggregate_tensor_elements < policy.max_tensor_elements:
        raise ValueError(
            "Checkpoint continuation aggregate tensor elements limit must not be "
            "smaller than the per-tensor limit."
        )
    if policy.max_aggregate_tensor_bytes < policy.max_tensor_bytes:
        raise ValueError(
            "Checkpoint continuation aggregate tensor byte limit must not be "
            "smaller than the per-tensor limit."
        )
    if policy.max_aggregate_tensor_dimensions < policy.max_tensor_dimensions:
        raise ValueError(
            "Checkpoint continuation aggregate tensor dimensions limit must not "
            "be smaller than the per-tensor limit."
        )
    if policy.max_aggregate_scalar_bytes < policy.max_scalar_bytes:
        raise ValueError(
            "Checkpoint continuation aggregate scalar byte limit must not be "
            "smaller than the per-value limit."
        )
    if policy.max_archive_uncompressed_bytes < policy.max_archive_record_bytes:
        raise ValueError(
            "Checkpoint continuation archive uncompressed byte limit must not be "
            "smaller than the per-record limit."
        )


DEFAULT_CHECKPOINT_ADMISSION_POLICY = CheckpointAdmissionPolicy()


__all__ = [
    "CheckpointAdmissionPolicy",
    "DEFAULT_CHECKPOINT_ADMISSION_POLICY",
]
