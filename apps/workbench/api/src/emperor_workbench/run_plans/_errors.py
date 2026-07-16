from emperor_workbench.failures import DomainFailure


class RunPlanFailure(DomainFailure):
    """A Run Plan request cannot be completed."""


__all__ = ["RunPlanFailure"]
