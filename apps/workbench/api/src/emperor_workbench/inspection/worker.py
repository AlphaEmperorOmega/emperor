from __future__ import annotations

import contextlib
import json
import os
import sys
from collections.abc import Mapping
from typing import Any

from emperor_workbench.inspection._errors import (
    InspectionFailure,
    inspection_failure,
)
from emperor_workbench.inspection._worker_protocol import (
    InspectionWorkerRequest,
    decode_worker_request,
    domain_failure_envelope,
    internal_failure_envelope,
    success_envelope,
)

try:
    import resource
except ImportError:  # pragma: no cover - Windows standard library
    resource = None  # type: ignore[assignment]


def _apply_worker_limits(request: InspectionWorkerRequest) -> None:
    if resource is not None:
        resource.setrlimit(
            resource.RLIMIT_AS,
            (request.memory_bytes, request.memory_bytes),
        )
    if hasattr(os, "sched_getaffinity") and hasattr(os, "sched_setaffinity"):
        available_cpus = sorted(os.sched_getaffinity(0))
        selected_cpus = available_cpus[: min(request.cpu_count, len(available_cpus))]
        if selected_cpus:
            os.sched_setaffinity(0, selected_cpus)


def _run_worker(request: InspectionWorkerRequest) -> dict[str, Any]:
    _apply_worker_limits(request)

    from model_runtime.inspection import InspectionRequest

    from emperor_workbench.model_packages import (
        ModelPackageCatalog,
        ModelPackageFailure,
    )
    from emperor_workbench.project_adapter import ProjectAdapterClient

    with ProjectAdapterClient(timeout_seconds=None) as project_adapter:
        try:
            selected_model_package = ModelPackageCatalog(project_adapter).select_parts(
                request.model_type,
                request.model,
            )
            parsed_overrides = selected_model_package.parse_overrides(request.overrides)
            inspection_result = selected_model_package.inspect(
                InspectionRequest(
                    preset=request.preset,
                    overrides=parsed_overrides,
                    dataset=request.dataset,
                    experiment_task=request.experiment_task,
                )
            )
        except ModelPackageFailure as exc:
            raise inspection_failure(exc) from exc
    return success_envelope(inspection_result)


def main() -> int:
    protocol_stdout = sys.stdout
    try:
        payload = json.loads(sys.stdin.buffer.read())
        if not isinstance(payload, Mapping):
            raise ValueError("Inspection worker request must be a mapping.")
        request = decode_worker_request(payload)
        with contextlib.redirect_stdout(sys.stderr):
            envelope = _run_worker(request)
    except InspectionFailure as exc:
        envelope = domain_failure_envelope(exc)
    except Exception:
        envelope = internal_failure_envelope()
    try:
        json.dump(
            envelope,
            protocol_stdout,
            allow_nan=False,
            separators=(",", ":"),
        )
    except Exception:
        return 70
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main"]
