"""Durable, idempotent execution for HTTP mutations.

Read work keeps its bounded timeout in :mod:`workbench.backend.blocking`. Mutation
work has a bounded admission wait, then runs to a definitive result so the API
never reports a timeout while a side effect is still executing.
"""

from __future__ import annotations

import asyncio
import contextvars
import hashlib
import json
import os
import secrets
import sqlite3
import time
import uuid
import weakref
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, TypeVar

from starlette.datastructures import Headers
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from workbench.backend.api.mutation_policy import (
    HttpOperationCatalog,
    operation_policy_enabled,
)
from workbench.backend.core.config import WorkbenchApiSettings
from workbench.backend.core.errors import ApiError
from workbench.backend.core.security import (
    LOCAL_MUTATION_DISABLED_DETAIL,
    UNAUTHORIZED_DETAIL,
    WWW_AUTHENTICATE_HEADER,
)
from workbench.backend.mutation_context import (
    activate_mutation_identity,
    current_mutation_identity,
    deterministic_mutation_resource_id,
)

ResultT = TypeVar("ResultT")

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

_PROCESS_TOKEN = uuid.uuid4().hex
_mutation_limiters: weakref.WeakKeyDictionary[
    asyncio.AbstractEventLoop,
    dict[str, asyncio.Semaphore],
] = weakref.WeakKeyDictionary()


def _mutation_limiter(name: str, concurrency: int) -> asyncio.Semaphore:
    loop = asyncio.get_running_loop()
    limiters = _mutation_limiters.get(loop)
    if limiters is None:
        limiters = {}
        _mutation_limiters[loop] = limiters
    limiter = limiters.get(name)
    if limiter is None:
        limiter = asyncio.Semaphore(max(1, concurrency))
        limiters[name] = limiter
    return limiter


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
    """Admit bounded mutation work, then await it without an execution timeout."""

    if limiter_already_acquired and limiter is None:
        raise ValueError("An acquired mutation limiter must be provided")
    work_limiter = limiter or _mutation_limiter(limiter_name, concurrency)
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
    executor = ThreadPoolExecutor(
        max_workers=1,
        thread_name_prefix="workbench-mutation",
    )
    future = executor.submit(context.run, call)

    async def await_result() -> ResultT:
        try:
            while not future.done():
                await asyncio.sleep(0.01)
            return future.result()
        finally:
            executor.shutdown(wait=True, cancel_futures=False)

    task: asyncio.Task[ResultT] = asyncio.create_task(await_result())
    release_deferred = False
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        release_deferred = True
        task.add_done_callback(
            lambda completed: _release_limiter_when_done(completed, work_limiter)
        )
        raise
    finally:
        if acquired and not release_deferred:
            work_limiter.release()


@dataclass(frozen=True, slots=True)
class StoredMutationResponse:
    status_code: int
    headers: tuple[tuple[bytes, bytes], ...]
    body: bytes


@dataclass(frozen=True, slots=True)
class MutationAdmission:
    action: str
    response: StoredMutationResponse | None = None
    request_hash: str | None = None


class MutationJournal:
    """SQLite-backed journal with one atomic admission decision per scoped key."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.path = self.root / "mutation-journal.sqlite3"

    def _connect(self) -> sqlite3.Connection:
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.root, 0o700)
        connection = sqlite3.connect(self.path, timeout=1.0)
        os.chmod(self.path, 0o600)
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS mutation_journal (
                caller TEXT NOT NULL,
                method TEXT NOT NULL,
                route TEXT NOT NULL,
                idempotency_key TEXT NOT NULL,
                request_hash TEXT NOT NULL,
                state TEXT NOT NULL,
                owner TEXT NOT NULL,
                status_code INTEGER,
                headers_json TEXT,
                response_body BLOB,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                PRIMARY KEY (caller, method, route, idempotency_key)
            )
            """
        )
        return connection

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def admit(
        self,
        *,
        caller: str,
        method: str,
        route: str,
        key: str,
    ) -> MutationAdmission:
        now = time.time()
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT request_hash, state, owner, status_code,
                       headers_json, response_body
                  FROM mutation_journal
                 WHERE caller = ? AND method = ? AND route = ?
                   AND idempotency_key = ?
                """,
                (caller, method, route, key),
            ).fetchone()
            if row is None:
                connection.execute(
                    """
                    INSERT INTO mutation_journal (
                        caller, method, route, idempotency_key, request_hash,
                        state, owner, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, 'in_progress', ?, ?, ?)
                    """,
                    (
                        caller,
                        method,
                        route,
                        key,
                        "",
                        _PROCESS_TOKEN,
                        now,
                        now,
                    ),
                )
                return MutationAdmission("execute")

            stored_hash, state, owner, status_code, headers_json, body = row
            if state == "completed":
                decoded_headers = tuple(
                    (name.encode("latin-1"), value.encode("latin-1"))
                    for name, value in json.loads(headers_json or "[]")
                )
                return MutationAdmission(
                    "replay",
                    StoredMutationResponse(
                        status_code=int(status_code),
                        headers=decoded_headers,
                        body=bytes(body or b""),
                    ),
                    str(stored_hash),
                )
            if owner == _PROCESS_TOKEN:
                return MutationAdmission("in_progress")

            # A different process owned the unfinished row. Re-execute through
            # deterministic resource reconciliation and retain the original key.
            connection.execute(
                """
                UPDATE mutation_journal
                   SET owner = ?, updated_at = ?
                 WHERE caller = ? AND method = ? AND route = ?
                   AND idempotency_key = ?
                """,
                (_PROCESS_TOKEN, now, caller, method, route, key),
            )
            return MutationAdmission("execute")

    def record_request_hash(
        self,
        *,
        caller: str,
        method: str,
        route: str,
        key: str,
        request_hash: str,
    ) -> bool:
        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT request_hash FROM mutation_journal
                 WHERE caller = ? AND method = ? AND route = ?
                   AND idempotency_key = ? AND state = 'in_progress'
                   AND owner = ?
                """,
                (caller, method, route, key, _PROCESS_TOKEN),
            ).fetchone()
            if row is None:
                return False
            stored_hash = str(row[0] or "")
            if stored_hash and stored_hash != request_hash:
                return False
            connection.execute(
                """
                UPDATE mutation_journal
                   SET request_hash = ?, updated_at = ?
                 WHERE caller = ? AND method = ? AND route = ?
                   AND idempotency_key = ? AND owner = ?
                """,
                (
                    request_hash,
                    time.time(),
                    caller,
                    method,
                    route,
                    key,
                    _PROCESS_TOKEN,
                ),
            )
            return True

    def complete(
        self,
        *,
        caller: str,
        method: str,
        route: str,
        key: str,
        response: StoredMutationResponse,
    ) -> None:
        headers_json = json.dumps(
            [
                [name.decode("latin-1"), value.decode("latin-1")]
                for name, value in response.headers
            ],
            separators=(",", ":"),
        )
        with self._connection() as connection:
            connection.execute(
                """
                UPDATE mutation_journal
                   SET state = 'completed', status_code = ?, headers_json = ?,
                       response_body = ?, updated_at = ?
                 WHERE caller = ? AND method = ? AND route = ?
                   AND idempotency_key = ? AND owner = ?
                """,
                (
                    response.status_code,
                    headers_json,
                    response.body,
                    time.time(),
                    caller,
                    method,
                    route,
                    key,
                    _PROCESS_TOKEN,
                ),
            )

    def abandon(
        self,
        *,
        caller: str,
        method: str,
        route: str,
        key: str,
    ) -> None:
        with self._connection() as connection:
            connection.execute(
                """
                DELETE FROM mutation_journal
                 WHERE caller = ? AND method = ? AND route = ?
                   AND idempotency_key = ? AND state = 'in_progress'
                   AND owner = ?
                """,
                (caller, method, route, key, _PROCESS_TOKEN),
            )


def _valid_idempotency_key(key: str) -> bool:
    return 1 <= len(key) <= 128 and all(0x20 <= ord(char) <= 0x7E for char in key)


def _caller(headers: Headers) -> str:
    authorization = headers.get("authorization")
    if not authorization:
        return "local"
    return "authorization:" + hashlib.sha256(authorization.encode()).hexdigest()


async def _request_body(receive: Receive) -> tuple[bytes, Receive]:
    body = bytearray()
    while True:
        message = await receive()
        if message["type"] == "http.disconnect":
            break
        if message["type"] != "http.request":
            continue
        body.extend(message.get("body", b""))
        if not message.get("more_body", False):
            break

    sent = False

    async def replay_receive() -> Message:
        nonlocal sent
        if not sent:
            sent = True
            return {
                "type": "http.request",
                "body": bytes(body),
                "more_body": False,
            }
        # The complete request was replayed above. Returning disconnect for any
        # subsequent probe avoids waiting on transports that only publish their
        # disconnect after the response has completed.
        return {"type": "http.disconnect"}

    return bytes(body), replay_receive


def _request_hash(scope: Scope, body: bytes) -> str:
    digest = _request_digest(scope)
    digest.update(body)
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
        self.journal = MutationJournal(Path(settings.state_root))
        self._background: set[asyncio.Task[StoredMutationResponse]] = set()

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

        caller = _caller(headers)
        method = str(scope.get("method", "")).upper()
        route = operation.route.path
        try:
            admission = self.journal.admit(
                caller=caller,
                method=method,
                route=route,
                key=key,
            )
        except (OSError, sqlite3.Error) as exc:
            raise ApiError(
                MUTATION_ADMISSION_UNAVAILABLE_DETAIL,
                status_code=503,
            ) from exc

        if admission.action == "in_progress":
            await JSONResponse(
                {"detail": IDEMPOTENCY_KEY_IN_PROGRESS_DETAIL},
                status_code=409,
                headers={"Retry-After": "1"},
            )(scope, receive, send)
            return
        if admission.action == "replay":
            assert admission.response is not None
            body, _replay_receive = await _request_body(receive)
            if _request_hash(scope, body) != admission.request_hash:
                await JSONResponse(
                    {"detail": IDEMPOTENCY_KEY_CONFLICT_DETAIL},
                    status_code=409,
                )(scope, _replay_receive, send)
                return
            await _send_stored(admission.response, send)
            return

        digest = _request_digest(scope)
        body_complete = not _body_expected(headers)
        hash_recorded = False

        def record_request_hash() -> bool:
            nonlocal hash_recorded
            if hash_recorded:
                return True
            hash_recorded = self.journal.record_request_hash(
                caller=caller,
                method=method,
                route=route,
                key=key,
                request_hash=digest.hexdigest(),
            )
            return hash_recorded

        if body_complete and not record_request_hash():
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
                    if not record_request_hash():
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
                with activate_mutation_identity(identity):
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
                    if body_complete and record_request_hash():
                        self.journal.complete(
                            caller=caller,
                            method=method,
                            route=route,
                            key=key,
                            response=response,
                        )
                    else:
                        self.journal.abandon(
                            caller=caller,
                            method=method,
                            route=route,
                            key=key,
                        )
                    return response
            except BaseException:
                self.journal.abandon(
                    caller=caller,
                    method=method,
                    route=route,
                    key=key,
                )
                raise

        task = asyncio.create_task(execute())
        self._background.add(task)
        task.add_done_callback(self._background.discard)
        response = await asyncio.shield(task)
        await _send_stored(response, send)


__all__ = [
    "IDEMPOTENCY_HEADER_NAME",
    "MutationExecutionMiddleware",
    "MutationJournal",
    "current_mutation_identity",
    "deterministic_mutation_resource_id",
    "run_mutation_io",
]
