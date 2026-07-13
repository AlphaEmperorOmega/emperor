"""Shared FastAPI exception handlers."""

from __future__ import annotations

from emperor.runs import replace_non_finite_json
from fastapi import HTTPException, Request, Response
from fastapi.encoders import jsonable_encoder
from fastapi.exception_handlers import http_exception_handler
from fastapi.exceptions import RequestValidationError
from starlette.responses import JSONResponse

from workbench.backend.core.errors import ApiError
from workbench.backend.failures import DomainFailure, FailureKind

_DOMAIN_FAILURE_STATUS = {
    FailureKind.INVALID: 400,
    FailureKind.CONFLICT: 409,
    FailureKind.TIMEOUT: 408,
    FailureKind.UNAVAILABLE: 503,
    FailureKind.TOO_LARGE: 413,
}


async def api_error_handler(request: Request, exc: ApiError) -> Response:
    http_error = HTTPException(status_code=exc.status_code, detail=exc.detail)
    return await http_exception_handler(request, http_error)


async def domain_failure_handler(
    request: Request,
    exc: DomainFailure,
) -> Response:
    http_error = HTTPException(
        status_code=_DOMAIN_FAILURE_STATUS[exc.kind],
        detail=exc.detail,
    )
    return await http_exception_handler(request, http_error)


async def request_validation_error_handler(
    request: Request,
    exc: RequestValidationError,
) -> Response:
    del request
    return JSONResponse(
        status_code=422,
        content=replace_non_finite_json({"detail": jsonable_encoder(exc.errors())}),
    )
