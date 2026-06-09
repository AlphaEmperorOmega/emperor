"""Shared FastAPI exception handlers."""

from __future__ import annotations

from fastapi import HTTPException, Request, Response
from fastapi.exception_handlers import http_exception_handler

from viewer.backend.core.errors import ApiError


async def api_error_handler(request: Request, exc: ApiError) -> Response:
    http_error = HTTPException(status_code=exc.status_code, detail=exc.detail)
    return await http_exception_handler(request, http_error)
