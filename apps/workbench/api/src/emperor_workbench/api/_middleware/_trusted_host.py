from __future__ import annotations

from starlette.datastructures import Headers
from starlette.responses import PlainTextResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from emperor_workbench.settings import WorkbenchApiSettings


def _authority_host(authority: str) -> str:
    value = authority.strip().casefold()
    if value.startswith("["):
        closing_bracket = value.find("]")
        return value[: closing_bracket + 1] if closing_bracket >= 0 else value
    return value.split(":", 1)[0].rstrip(".")


def _host_matches_pattern(host: str, pattern: str) -> bool:
    normalized_pattern = _authority_host(pattern)
    if normalized_pattern == "*":
        return True
    if normalized_pattern.startswith("*."):
        suffix = normalized_pattern[1:]
        return host.endswith(suffix) and host != suffix[1:]
    return host == normalized_pattern


def host_is_trusted(headers: Headers, settings: WorkbenchApiSettings) -> bool:
    host = _authority_host(headers.get("host", ""))
    return bool(host) and any(
        _host_matches_pattern(host, pattern) for pattern in settings.trusted_hosts
    )


class WorkbenchTrustedHostMiddleware:
    """Reject unconfigured authorities before any route policy executes."""

    def __init__(
        self,
        app: ASGIApp,
        *,
        settings: WorkbenchApiSettings,
    ) -> None:
        self.app = app
        self.settings = settings

    async def __call__(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
    ) -> None:
        if scope["type"] not in {"http", "websocket"}:
            await self.app(scope, receive, send)
            return
        if not host_is_trusted(Headers(scope=scope), self.settings):
            response = PlainTextResponse("Invalid host header", status_code=400)
            await response(scope, receive, send)
            return
        await self.app(scope, receive, send)


__all__ = ["WorkbenchTrustedHostMiddleware", "host_is_trusted"]
