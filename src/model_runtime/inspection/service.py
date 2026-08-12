from __future__ import annotations

from model_runtime.inspection.capture_limits import InspectionCapture
from model_runtime.inspection.materialization import (
    materialize_configuration,
    materialize_inspection,
)
from model_runtime.inspection.records import InspectionRequest, InspectionResult
from model_runtime.packages import ModelPackage


def inspect_model(
    package: ModelPackage,
    request: InspectionRequest,
) -> InspectionResult:
    from model_runtime.inspection.model_graph import inspect_model_graph

    materialized = materialize_inspection(package, request)
    capture = InspectionCapture(request.capture_limits)
    graph = inspect_model_graph(
        materialized.model,
        limits=request.capture_limits,
        _capture=capture,
    )
    result = materialized.result(graph)
    capture.ensure_total_output(result)
    return result


def validate_configuration(
    package: ModelPackage,
    request: InspectionRequest,
) -> None:
    materialize_configuration(package, request)


__all__ = ["inspect_model", "validate_configuration"]
