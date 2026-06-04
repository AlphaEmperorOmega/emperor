"""Shared FastAPI exception handlers."""

from __future__ import annotations

from fastapi import Request
from fastapi.responses import JSONResponse

from viewer.backend.inspector.errors import InspectorError


async def inspector_error_handler(_request: Request, exc: InspectorError) -> JSONResponse:
    return JSONResponse(status_code=400, content={"detail": str(exc)})
