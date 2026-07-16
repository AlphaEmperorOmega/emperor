from __future__ import annotations

from model_runtime.inspection import InspectionRequest, InspectionResult

from emperor_workbench.inspection._errors import (
    InspectionFailure,
    inspection_failure,
)
from emperor_workbench.model_packages import (
    ModelPackageFailure,
    SelectedModelPackage,
)


class InProcessInspectionExecutor:
    """Execute semantic Inspection inside the current process."""

    def inspect(
        self,
        selected: SelectedModelPackage,
        request: InspectionRequest,
    ) -> InspectionResult:
        try:
            return selected.inspect(request)
        except ModelPackageFailure as exc:
            raise inspection_failure(exc) from exc


__all__ = ["InProcessInspectionExecutor", "InspectionFailure"]
