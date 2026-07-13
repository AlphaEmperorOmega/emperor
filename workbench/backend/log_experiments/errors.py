from workbench.backend.failures import DomainFailure


class LogExperimentFailure(DomainFailure):
    """A Log Experiment request or coordination attempt failed."""


__all__ = ["LogExperimentFailure"]
