from __future__ import annotations

import asyncio
import contextvars
import hashlib
import secrets
import sqlite3
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from typing import Any, TypeVar

from starlette.datastructures import Headers
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from emperor_workbench.api._container import current_container
from emperor_workbench.api._errors import ApiError
from emperor_workbench.api._mutations._context import (
    activate_mutation_identity,
    current_mutation_identity,
    deterministic_mutation_resource_id,
)
from emperor_workbench.api._mutations._journal import (
    MutationJournal,
    StoredMutationResponse,
)
from emperor_workbench.api._mutations._policy import (
    HttpOperationCatalog,
    operation_policy_enabled,
)
from emperor_workbench.api._security import (
    LOCAL_MUTATION_DISABLED_DETAIL,
    UNAUTHORIZED_DETAIL,
    WWW_AUTHENTICATE_HEADER,
)
from emperor_workbench.settings import WorkbenchApiSettings

ResultT = TypeVar("ResultT")

_ACTIVE_MUTATION_RUNTIME: contextvars.ContextVar[MutationExecutionRuntime | None] = (
    contextvars.ContextVar("workbench_mutation_runtime", default=None)
)

IDEMPOTENCY_HEADER_NAME = "Idempotency-Key"
IDEMPOTENCY_KEY_REQUIRED_DETAIL = "Idempotency-Key is required for mutation requests."
IDEMPOTENCY_KEY_INVALID_DETAIL = (
    "Idempotency-Key must contain 1 to 128 printable ASCII characters."
)
IDEMPOTENCY_KEY_CONFLICT_DETAIL = (
    "Idempotency-Key was already used for a different request."
)
IDEMPOTENCY_KEY_IN_PROGRESS_DETAIL = (
    "A mutation with this Idempotency-Key is still in progress."
)
MUTATION_ADMISSION_UNAVAILABLE_DETAIL = (
    "Mutation execution is temporarily at capacity; retry later."
)


class MutationExecutionRuntime:
    """App-scoped mutation journal, executor, limiters, and tracked tasks."""

    def __init__(
        self,
        state_root: Path | None,
        *,
        max_workers: int = 4,
    ) -> None:
        self.journal = MutationJournal(state_root) if state_root is not None else None
        self._executor = ThreadPoolExecutor(
            max_workers=max(1, int(max_workers)),
            thread_name_prefix="workbench-mutation",
        )
        self._limiters: dict[str, asyncio.Semaphore] = {}
        self._background: set[asyncio.Task[object]] = set()
        self._closed = False

    def _limiter(self, name: str, concurrency: int) -> asyncio.Semaphore:
        limiter = self._limiters.get(name)
        if limiter is None:
            limiter = asyncio.Semaphore(max(1, concurrency))
            self._limiters[name] = limiter
        return limiter

    def track(self, task: asyncio.Task[ResultT]) -> asyncio.Task[ResultT]:
        self._background.add(task)
        task.add_done_callback(self._background.discard)
        return task

    async def run(
        self,
        callable_object: Callable[..., ResultT],
        *args: Any,
        admission_timeout_seconds: float = 1.0,
        concurrency: int = 4,
        limiter_name: str = "default-mutations",
        limiter: asyncio.Semaphore | None = None,
        limiter_already_acquired: bool = False,
        **kwargs: Any,
    ) -> ResultT:
        if self._closed:
            raise RuntimeError("Mutation execution runtime is closed.")
        if limiter_already_acquired and limiter is None:
            raise ValueError("An acquired mutation limiter must be provided")
        work_limiter = limiter or self._limiter(limiter_name, concurrency)
        acquired = limiter_already_acquired
        if not acquired:
            try:
                await asyncio.wait_for(
                    work_limiter.acquire(),
                    timeout=max(0.001, admission_timeout_seconds),
                )
            except TimeoutError as exc:
                raise ApiError(
                    MUTATION_ADMISSION_UNAVAILABLE_DETAIL,
                    status_code=503,
                ) from exc
            acquired = True

        context = contextvars.copy_context()
        call = partial(callable_object, *args, **kwargs)
        future = self._executor.submit(context.run, call)

        async def await_result() -> ResultT:
            while not future.done():
                await asyncio.sleep(0.01)
            return future.result()

        task = self.track(asyncio.create_task(await_result()))
        release_deferred = False
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            release_deferred = True
            task.add_done_callback(
                lambda completed: _release_limiter_when_done(
                    completed,
                    work_limiter,
                )
            )
            raise
        finally:
            if acquired and not release_deferred:
                work_limiter.release()

    async def close(self) -> None:
        if self._closed:
            return
        while self._background:
            pending = tuple(self._background)
            await asyncio.gather(*pending, return_exceptions=True)
            self._background.difference_update(pending)
        self._closed = True
        self._executor.shutdown(wait=True, cancel_futures=False)
        self._limiters.clear()

    def close_executor(self) -> None:
        """Close a not-yet-published runtime after bootstrap failure."""

        if self._closed:
            return
        self._closed = True
        self._executor.shutdown(wait=True, cancel_futures=True)
        self._limiters.clear()


@contextmanager
def _activate_mutation_runtime(
    runtime: MutationExecutionRuntime,
) -> Iterator[None]:
    token = _ACTIVE_MUTATION_RUNTIME.set(runtime)
    try:
        yield
    finally:
        _ACTIVE_MUTATION_RUNTIME.reset(token)


def _release_limiter_when_done(
    task: asyncio.Task[object],
    limiter: asyncio.Semaphore,
) -> None:
    limiter.release()
    if not task.cancelled():
        task.exception()


async def run_mutation_io(
    callable_object: Callable[..., ResultT],
    *args: Any,
    admission_timeout_seconds: float = 1.0,
    concurrency: int = 4,
    limiter_name: str = "default-mutations",
    limiter: asyncio.Semaphore | None = None,
    limiter_already_acquired: bool = False,
    **kwargs: Any,
) -> ResultT:
    """Run mutation work using the active app-scoped execution runtime."""

    runtime = _ACTIVE_MUTATION_RUNTIME.get()
    if runtime is None:
        try:
            runtime = current_container().mutation_execution
        except RuntimeError:
            runtime = None
    if runtime is None:
        runtime = MutationExecutionRuntime(None, max_workers=1)
        try:
            return await runtime.run(
                callable_object,
                *args,
                admission_timeout_seconds=admission_timeout_seconds,
                concurrency=concurrency,
                limiter_name=limiter_name,
                limiter=limiter,
                limiter_already_acquired=limiter_already_acquired,
                **kwargs,
            )
        finally:
            await runtime.close()
    return await runtime.run(
        callable_object,
        *args,
        admission_timeout_seconds=admission_timeout_seconds,
        concurrency=concurrency,
        limiter_name=limiter_name,
        limiter=limiter,
        limiter_already_acquired=limiter_already_acquired,
        **kwargs,
    )


def _valid_idempotency_key(key: str) -> bool:
    return 1 <= len(key) <= 128 and all(0x20 <= ord(char) <= 0x7E for char in key)


def _caller(headers: Headers) -> str:
    authorization = headers.get("authorization")
    if not authorization:
        return "local"
    return "authorization:" + hashlib.sha256(authorization.encode()).hexdigest()


async def _consume_request_hash(scope: Scope, receive: Receive) -> str:
    digest = _request_digest(scope)
    while True:
        message = await receive()
        if message["type"] == "http.disconnect":
            break
        if message["type"] != "http.request":
            continue
        digest.update(message.get("body", b""))
        if not message.get("more_body", False):
            break
    return digest.hexdigest()


def _request_digest(scope: Scope) -> Any:
    digest = hashlib.sha256()
    digest.update(scope.get("raw_path", scope.get("path", "").encode()))
    digest.update(b"\0")
    digest.update(scope.get("query_string", b""))
    digest.update(b"\0")
    return digest


def _body_expected(headers: Headers) -> bool:
    if headers.get("transfer-encoding"):
        return True
    try:
        return int(headers.get("content-length", "0")) > 0
    except ValueError:
        return True


async def _send_stored(response: StoredMutationResponse, send: Send) -> None:
    await send(
        {
            "type": "http.response.start",
            "status": response.status_code,
            "headers": list(response.headers),
        }
    )
    await send(
        {
            "type": "http.response.body",
            "body": response.body,
            "more_body": False,
        }
    )


async def _run_journal_io(
    runtime: MutationExecutionRuntime,
    callable_object: Callable[..., ResultT],
    *args: Any,
    **kwargs: Any,
) -> ResultT:
    try:
        return await runtime.run(
            callable_object,
            *args,
            limiter_name="mutation-journal",
            concurrency=1,
            **kwargs,
        )
    except (OSError, sqlite3.Error) as exc:
        raise ApiError(
            MUTATION_ADMISSION_UNAVAILABLE_DETAIL,
            status_code=503,
        ) from exc


class MutationExecutionMiddleware:
    """Journal and execute mutation routes behind their origin/host checks."""

    def __init__(
        self,
        app: ASGIApp,
        *,
        settings: WorkbenchApiSettings,
        operation_catalog: HttpOperationCatalog,
    ) -> None:
        self.app = app
        self.settings = settings
        self.operation_catalog = operation_catalog

    async def __call__(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
    ) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        operation = self.operation_catalog.mutation_for_scope(scope)
        if operation is None:
            await self.app(scope, receive, send)
            return

        headers = Headers(scope=scope)
        if self.settings.auth_mode == "bearer":
            authorization = headers.get("authorization", "")
            scheme, _, credential = authorization.partition(" ")
            configured = self.settings.token or ""
            if (
                scheme.casefold() != "bearer"
                or not credential
                or not configured
                or not secrets.compare_digest(credential, configured)
            ):
                await JSONResponse(
                    {"detail": UNAUTHORIZED_DETAIL},
                    status_code=401,
                    headers={"WWW-Authenticate": WWW_AUTHENTICATE_HEADER},
                )(scope, receive, send)
                return
        if not operation_policy_enabled(operation.policy, self.settings):
            await JSONResponse(
                {"detail": LOCAL_MUTATION_DISABLED_DETAIL},
                status_code=403,
            )(scope, receive, send)
            return

        key = headers.get(IDEMPOTENCY_HEADER_NAME)
        if not key:
            await JSONResponse(
                {"detail": IDEMPOTENCY_KEY_REQUIRED_DETAIL},
                status_code=428,
            )(scope, receive, send)
            return
        if not _valid_idempotency_key(key):
            await JSONResponse(
                {"detail": IDEMPOTENCY_KEY_INVALID_DETAIL},
                status_code=422,
            )(scope, receive, send)
            return

        runtime = current_container().mutation_execution
        journal = runtime.journal
        if journal is None:
            raise RuntimeError("Mutation journal is unavailable.")

        caller = _caller(headers)
        method = str(scope.get("method", "")).upper()
        route = operation.route.path
        admission = await _run_journal_io(
            runtime,
            journal.admit,
            caller=caller,
            method=method,
            route=route,
            key=key,
        )

        if admission.action == "in_progress":
            await JSONResponse(
                {"detail": IDEMPOTENCY_KEY_IN_PROGRESS_DETAIL},
                status_code=409,
                headers={"Retry-After": "1"},
            )(scope, receive, send)
            return
        if admission.action == "replay":
            assert admission.response is not None
            request_hash = await _consume_request_hash(scope, receive)
            if request_hash != admission.request_hash:
                await JSONResponse(
                    {"detail": IDEMPOTENCY_KEY_CONFLICT_DETAIL},
                    status_code=409,
                )(scope, receive, send)
                return
            await _send_stored(admission.response, send)
            return

        digest = _request_digest(scope)
        body_complete = not _body_expected(headers)
        hash_recorded = False

        async def record_request_hash() -> bool:
            nonlocal hash_recorded
            if hash_recorded:
                return True
            hash_recorded = await _run_journal_io(
                runtime,
                journal.record_request_hash,
                caller=caller,
                method=method,
                route=route,
                key=key,
                request_hash=digest.hexdigest(),
            )
            return hash_recorded

        if body_complete and not await record_request_hash():
            await JSONResponse(
                {"detail": IDEMPOTENCY_KEY_CONFLICT_DETAIL},
                status_code=409,
            )(scope, receive, send)
            return

        async def observe_receive() -> Message:
            nonlocal body_complete
            message = await receive()
            if message["type"] == "http.request":
                digest.update(message.get("body", b""))
                if not message.get("more_body", False):
                    body_complete = True
                    if not await record_request_hash():
                        raise ApiError(
                            IDEMPOTENCY_KEY_CONFLICT_DETAIL,
                            status_code=409,
                        )
            return message

        identity = hashlib.sha256(
            f"{caller}\0{method}\0{route}\0{key}".encode()
        ).hexdigest()

        async def execute() -> StoredMutationResponse:
            messages: list[Message] = []

            async def capture(message: Message) -> None:
                messages.append(message)

            try:
                with (
                    _activate_mutation_runtime(runtime),
                    activate_mutation_identity(identity),
                ):
                    await self.app(scope, observe_receive, capture)
                    start = next(
                        message
                        for message in messages
                        if message["type"] == "http.response.start"
                    )
                    response = StoredMutationResponse(
                        status_code=int(start["status"]),
                        headers=tuple(start.get("headers", ())),
                        body=b"".join(
                            message.get("body", b"")
                            for message in messages
                            if message["type"] == "http.response.body"
                        ),
                    )
                    if body_complete and await record_request_hash():
                        await _run_journal_io(
                            runtime,
                            journal.complete,
                            caller=caller,
                            method=method,
                            route=route,
                            key=key,
                            response=response,
                        )
                    else:
                        await _run_journal_io(
                            runtime,
                            journal.abandon,
                            caller=caller,
                            method=method,
                            route=route,
                            key=key,
                        )
                    return response
            except BaseException:
                try:
                    await _run_journal_io(
                        runtime,
                        journal.abandon,
                        caller=caller,
                        method=method,
                        route=route,
                        key=key,
                    )
                except ApiError:
                    pass
                raise

        task = runtime.track(asyncio.create_task(execute()))
        response = await asyncio.shield(task)
        await _send_stored(response, send)


__all__ = [
    "IDEMPOTENCY_HEADER_NAME",
    "MutationExecutionMiddleware",
    "MutationExecutionRuntime",
    "current_mutation_identity",
    "deterministic_mutation_resource_id",
    "run_mutation_io",
]
