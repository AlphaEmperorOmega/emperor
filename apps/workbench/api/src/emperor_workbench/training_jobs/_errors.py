from emperor_workbench.failures import DomainFailure


class TrainingJobFailure(DomainFailure):
    """A Training Job request cannot be completed."""


__all__ = ["TrainingJobFailure"]
