from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from model_runtime.inspection import InspectionRequest, InspectionResult

    from emperor_workbench.model_packages import SelectedModelPackage


class InspectionExecutor(Protocol):
    """Execution seam for one already-selected Model Package."""

    def inspect(
        self,
        selected: SelectedModelPackage,
        request: InspectionRequest,
    ) -> InspectionResult: ...


__all__ = ["InspectionExecutor"]
