from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI


@asynccontextmanager
async def lifespan_client(
    application: FastAPI,
    *,
    base_url: str = "http://localhost",
    headers: Mapping[str, str] | None = None,
    **client_options: Any,
) -> AsyncIterator[httpx.AsyncClient]:
    """Serve one AsyncClient while the real application lifespan is active."""

    async with application.router.lifespan_context(application):
        transport = httpx.ASGITransport(app=application)
        async with httpx.AsyncClient(
            transport=transport,
            base_url=base_url,
            headers=headers,
            **client_options,
        ) as client:
            yield client


__all__ = ["lifespan_client"]
