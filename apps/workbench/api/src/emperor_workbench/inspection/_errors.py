from __future__ import annotations

from typing import TYPE_CHECKING

from emperor_workbench.failures import DomainFailure

if TYPE_CHECKING:
    from emperor_workbench.model_packages import ModelPackageFailure


class InspectionFailure(DomainFailure):
    """A semantic Inspection request cannot be completed."""


def inspection_failure(exc: ModelPackageFailure) -> InspectionFailure:
    return InspectionFailure(exc.detail, kind=exc.kind)


__all__ = ["InspectionFailure"]
