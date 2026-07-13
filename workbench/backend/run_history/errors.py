from workbench.backend.failures import DomainFailure


class RunHistoryFailure(DomainFailure):
    """A Run History request cannot be completed."""


__all__ = ["RunHistoryFailure"]
