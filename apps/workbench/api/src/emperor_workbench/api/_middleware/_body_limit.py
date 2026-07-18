from __future__ import annotations

from starlette.datastructures import Headers
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send


def _is_json_content_type(headers: Headers) -> bool:
    media_type = headers.get("content-type", "").split(";", 1)[0].strip().lower()
    return media_type == "application/json" or media_type.endswith("+json")


def _declared_body_exceeds_limit(headers: Headers, max_bytes: int) -> bool:
    content_length = headers.get("content-length")
    if content_length is None:
        return False
    try:
        declared_bytes = int(content_length)
    except ValueError:
        return False
    return declared_bytes > max_bytes


class JsonBodyLimitMiddleware:
    """Stream JSON request bodies into a bounded replay buffer."""

    def __init__(self, app: ASGIApp, *, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max(1, int(max_bytes))

    async def __call__(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
    ) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = Headers(scope=scope)
        if not _is_json_content_type(headers):
            await self.app(scope, receive, send)
            return

        if _declared_body_exceeds_limit(headers, self.max_bytes):
            await self._reject(scope, receive, send)
            return

        buffered_messages: list[Message] = []
        received_bytes = 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                buffered_messages.append(message)
                break
            body = message.get("body", b"")
            if isinstance(body, bytes):
                received_bytes += len(body)
            if received_bytes > self.max_bytes:
                await self._reject(scope, receive, send)
                return
            buffered_messages.append(message)
            if not message.get("more_body", False):
                break

        async def replay_receive() -> Message:
            if buffered_messages:
                return buffered_messages.pop(0)
            return {"type": "http.request", "body": b"", "more_body": False}

        await self.app(scope, replay_receive, send)

    async def _reject(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
    ) -> None:
        response = JSONResponse(
            {"detail": (f"JSON request body exceeds the {self.max_bytes} byte limit.")},
            status_code=413,
        )
        await response(scope, receive, send)


__all__ = ["JsonBodyLimitMiddleware"]
