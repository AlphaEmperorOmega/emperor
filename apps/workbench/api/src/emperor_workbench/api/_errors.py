from __future__ import annotations

from fastapi import Request, Response
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from model_runtime.runs import replace_non_finite_json
from starlette.responses import JSONResponse

from emperor_workbench.failures import DomainFailure, FailureKind

_DOMAIN_FAILURE_STATUS = {
    FailureKind.INVALID: 400,
    FailureKind.CONFLICT: 409,
    FailureKind.TIMEOUT: 408,
    FailureKind.UNAVAILABLE: 503,
    FailureKind.TOO_LARGE: 413,
}


class ApiError(Exception):
    """A finite, user-facing HTTP failure raised by the transport layer."""

    status_code = 400

    def __init__(self, detail: str, *, status_code: int | None = None) -> None:
        super().__init__(detail)
        self.detail = detail
        if status_code is not None:
            self.status_code = status_code


def known_failure_response(exc: ApiError | DomainFailure) -> JSONResponse:
    if isinstance(exc, ApiError):
        status_code = exc.status_code
    else:
        status_code = _DOMAIN_FAILURE_STATUS[exc.kind]
    return JSONResponse(
        status_code=status_code,
        content=replace_non_finite_json({"detail": exc.detail}),
    )


async def api_error_handler(request: Request, exc: ApiError) -> Response:
    del request
    return known_failure_response(exc)


async def domain_failure_handler(
    request: Request,
    exc: DomainFailure,
) -> Response:
    del request
    return known_failure_response(exc)


async def request_validation_error_handler(
    request: Request,
    exc: RequestValidationError,
) -> Response:
    del request
    return JSONResponse(
        status_code=422,
        content=replace_non_finite_json({"detail": jsonable_encoder(exc.errors())}),
    )


__all__ = [
    "ApiError",
    "api_error_handler",
    "domain_failure_handler",
    "known_failure_response",
    "request_validation_error_handler",
]
