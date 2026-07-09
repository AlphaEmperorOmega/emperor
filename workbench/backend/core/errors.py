"""Application error types that can be translated at the API seam."""

from __future__ import annotations


class ApiError(Exception):
    """Base error for user-facing Workbench API failures."""

    status_code = 400

    def __init__(self, detail: str, *, status_code: int | None = None) -> None:
        super().__init__(detail)
        self.detail = detail
        if status_code is not None:
            self.status_code = status_code
