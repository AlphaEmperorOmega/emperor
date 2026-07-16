from emperor_workbench.failures import DomainFailure, FailureKind


class ProjectAdapterFailure(DomainFailure):
    """The configured model project Adapter rejected or failed a request."""

    def __init__(
        self,
        detail: str,
        *,
        kind: FailureKind = FailureKind.INVALID,
        remote_type: str | None = None,
        remote_cause_detail: str | None = None,
    ) -> None:
        super().__init__(detail, kind=kind)
        self.remote_type = remote_type
        self.remote_cause_detail = remote_cause_detail
