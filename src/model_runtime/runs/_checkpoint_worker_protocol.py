from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, cast

from model_runtime.runs._checkpoint_receipt import CheckpointPayloadReceipt
from model_runtime.runs.checkpoint_admission import CheckpointAdmissionPolicy

MAX_CHECKPOINT_WORKER_RECEIPT_BYTES = 4096
MAX_CHECKPOINT_WORKER_REQUEST_BYTES = 16 * 1024
_MAX_PROTOCOL_INTEGER = 2**63 - 1
_HASH = re.compile(r"[0-9a-f]{64}")
_REQUEST_KEYS = frozenset(
    {
        "snapshotPath",
        "receiptPath",
        "sourceName",
        "sha256",
        "sizeBytes",
        "policy",
    }
)
_SUCCESS_KEYS = frozenset(
    {"ok", "sha256", "sizeBytes", "controls", "receipt"}
)
_ERROR_KEYS = frozenset({"ok", "category", "detail"})
_CONTROL_KEYS = frozenset(
    {"addressSpaceBytes", "coreBytes", "cpuSeconds", "fileSizeBytes"}
)
_ERROR_CATEGORIES = frozenset(
    {
        "controls-unavailable",
        "internal-error",
        "invalid-payload",
        "invalid-request",
        "load-failed",
    }
)
_POLICY_KEYS = frozenset(field.name for field in fields(CheckpointAdmissionPolicy))
_RECEIPT_KEYS = frozenset(field.name for field in fields(CheckpointPayloadReceipt))


@dataclass(frozen=True, slots=True)
class DecodedCheckpointWorkerReceipt:
    sha256: str
    size_bytes: int
    controls: Mapping[str, int]
    receipt: CheckpointPayloadReceipt


@dataclass(frozen=True, slots=True)
class CheckpointWorkerRequest:
    snapshot_path: Path
    receipt_path: Path
    source_name: str
    sha256: str
    size_bytes: int
    policy: CheckpointAdmissionPolicy


class CheckpointWorkerRejected(ValueError):
    def __init__(self, category: str, detail: str) -> None:
        super().__init__(detail)
        self.category = category
        self.detail = detail


def _is_bounded_integer(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, int)
        and 0 <= value <= _MAX_PROTOCOL_INTEGER
    )


def _is_hash(value: object) -> bool:
    return isinstance(value, str) and _HASH.fullmatch(value) is not None


def _bounded_utf8(value: object, maximum_bytes: int) -> bool:
    if not isinstance(value, str):
        return False
    try:
        return len(value.encode("utf-8")) <= maximum_bytes
    except UnicodeEncodeError:
        return False


def _bounded_detail(detail: str) -> str:
    normalized = "".join(
        " " if unicodedata.category(character).startswith("C") else character
        for character in detail
    )
    single_line = " ".join(normalized.split())
    encoded = single_line.encode("utf-8", errors="replace")[:768]
    return encoded.decode("utf-8", errors="ignore")


def _object_mapping(
    value: object,
    expected_keys: frozenset[str],
    error_message: str,
) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise ValueError(error_message)
    object_mapping = cast(dict[object, object], value)
    if any(not isinstance(key, str) for key in object_mapping):
        raise ValueError(error_message)
    mapping = cast(dict[str, object], object_mapping)
    if frozenset(mapping) != expected_keys:
        raise ValueError(error_message)
    return mapping


def _construct_policy(values: Mapping[str, object]) -> CheckpointAdmissionPolicy:
    constructor = cast(Any, CheckpointAdmissionPolicy)
    return cast(CheckpointAdmissionPolicy, constructor(**dict(values)))


def _construct_receipt(values: Mapping[str, object]) -> CheckpointPayloadReceipt:
    constructor = cast(Any, CheckpointPayloadReceipt)
    return cast(CheckpointPayloadReceipt, constructor(**dict(values)))


def encode_worker_request(
    request: CheckpointWorkerRequest,
) -> bytes:
    payload = {
        "snapshotPath": str(request.snapshot_path),
        "receiptPath": str(request.receipt_path),
        "sourceName": request.source_name,
        "sha256": request.sha256,
        "sizeBytes": request.size_bytes,
        "policy": {
            **asdict(request.policy),
            "allowed_dtypes": sorted(request.policy.allowed_dtypes),
        },
    }
    encoded = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    if len(encoded) > MAX_CHECKPOINT_WORKER_REQUEST_BYTES:
        raise ValueError("Checkpoint decoder request exceeds its size limit.")
    return encoded


def decode_worker_request(
    raw: bytes,
) -> CheckpointWorkerRequest:
    payload: object = json.loads(raw)
    request = _object_mapping(
        payload,
        _REQUEST_KEYS,
        "Invalid checkpoint decoder request.",
    )
    snapshot_path = request["snapshotPath"]
    receipt_path = request["receiptPath"]
    source_name = request["sourceName"]
    sha256 = request["sha256"]
    size_bytes = request["sizeBytes"]
    policy_value = request["policy"]
    if (
        not _bounded_utf8(snapshot_path, 4096)
        or not _bounded_utf8(receipt_path, 4096)
        or not _bounded_utf8(source_name, 255)
        or not _is_hash(sha256)
        or not _is_bounded_integer(size_bytes)
    ):
        raise ValueError("Invalid checkpoint decoder request.")
    policy_values = dict(
        _object_mapping(
            policy_value,
            _POLICY_KEYS,
            "Invalid checkpoint decoder request.",
        )
    )
    allowed_dtypes = policy_values.get("allowed_dtypes")
    if not isinstance(allowed_dtypes, list):
        raise ValueError("Invalid checkpoint decoder request.")
    dtype_values = cast(list[object], allowed_dtypes)
    if any(not isinstance(value, str) for value in dtype_values):
        raise ValueError("Invalid checkpoint decoder request.")
    policy_values["allowed_dtypes"] = frozenset(cast(list[str], dtype_values))
    return CheckpointWorkerRequest(
        snapshot_path=Path(cast(str, snapshot_path)),
        receipt_path=Path(cast(str, receipt_path)),
        source_name=cast(str, source_name),
        sha256=cast(str, sha256),
        size_bytes=cast(int, size_bytes),
        policy=_construct_policy(policy_values),
    )


def encode_worker_receipt(
    sha256: str,
    size_bytes: int,
    receipt: CheckpointPayloadReceipt,
    controls: Mapping[str, int],
) -> bytes:
    encoded = json.dumps(
        {
            "ok": True,
            "sha256": sha256,
            "sizeBytes": size_bytes,
            "controls": dict(controls),
            "receipt": asdict(receipt),
        },
        separators=(",", ":"),
    ).encode("utf-8")
    if len(encoded) > MAX_CHECKPOINT_WORKER_RECEIPT_BYTES:
        raise ValueError("Checkpoint decoder receipt exceeds its size limit.")
    return encoded


def encode_worker_error(category: str, detail: str) -> bytes:
    if category not in _ERROR_CATEGORIES:
        category = "internal-error"
    encoded = json.dumps(
        {
            "ok": False,
            "category": category,
            "detail": _bounded_detail(detail),
        },
        separators=(",", ":"),
    ).encode("utf-8")
    if len(encoded) > MAX_CHECKPOINT_WORKER_RECEIPT_BYTES:
        raise ValueError("Checkpoint decoder receipt exceeds its size limit.")
    return encoded


def _validated_controls(value: object) -> dict[str, int]:
    controls = _object_mapping(
        value,
        _CONTROL_KEYS,
        "Invalid checkpoint decoder receipt.",
    )
    validated: dict[str, int] = {}
    for name, item in controls.items():
        if not _is_bounded_integer(item):
            raise ValueError("Invalid checkpoint decoder receipt.")
        validated[name] = cast(int, item)
    return validated


def _validated_receipt(value: object) -> CheckpointPayloadReceipt:
    receipt = _object_mapping(
        value,
        _RECEIPT_KEYS,
        "Invalid checkpoint decoder receipt.",
    )
    version_hash = receipt.get("lightning_version_sha256")
    if not _is_hash(version_hash):
        raise ValueError("Invalid checkpoint decoder receipt.")
    if any(
        not _is_bounded_integer(field_value)
        for field_name, field_value in receipt.items()
        if field_name != "lightning_version_sha256"
    ):
        raise ValueError("Invalid checkpoint decoder receipt.")
    return _construct_receipt(receipt)


def decode_worker_response(raw: bytes) -> DecodedCheckpointWorkerReceipt:
    if len(raw) > MAX_CHECKPOINT_WORKER_RECEIPT_BYTES:
        raise ValueError("Checkpoint decoder receipt exceeds its size limit.")
    payload: object = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("Invalid checkpoint decoder receipt.")
    raw_mapping = cast(dict[object, object], payload)
    if raw_mapping.get("ok") is not True:
        error = _object_mapping(
            cast(object, payload),
            _ERROR_KEYS,
            "Invalid checkpoint decoder receipt.",
        )
        category = error.get("category")
        detail = error.get("detail")
        if (
            not isinstance(category, str)
            or category not in _ERROR_CATEGORIES
            or not isinstance(detail, str)
            or not _bounded_utf8(detail, 768)
            or _bounded_detail(detail) != detail
        ):
            raise ValueError("Invalid checkpoint decoder receipt.")
        raise CheckpointWorkerRejected(category, detail)
    success = _object_mapping(
        cast(object, payload),
        _SUCCESS_KEYS,
        "Invalid checkpoint decoder receipt.",
    )
    sha256 = success["sha256"]
    size_bytes = success["sizeBytes"]
    if not _is_hash(sha256) or not _is_bounded_integer(size_bytes):
        raise ValueError("Invalid checkpoint decoder receipt.")
    return DecodedCheckpointWorkerReceipt(
        sha256=cast(str, sha256),
        size_bytes=cast(int, size_bytes),
        controls=_validated_controls(success["controls"]),
        receipt=_validated_receipt(success["receipt"]),
    )


__all__: list[str] = []
