"""Training Job capability failures."""

from workbench.backend.failures import DomainFailure


class TrainingJobFailure(DomainFailure):
    """A Training Job request cannot be completed."""


__all__ = ["TrainingJobFailure"]
