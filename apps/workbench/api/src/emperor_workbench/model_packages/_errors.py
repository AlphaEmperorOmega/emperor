from emperor_workbench.failures import DomainFailure


class ModelPackageFailure(DomainFailure):
    """A Model Package cannot be discovered, selected, or interpreted."""


__all__ = ["ModelPackageFailure"]
