from __future__ import annotations

import asyncio
import sqlite3
import tempfile
import threading
import unittest
from pathlib import Path
from typing import Annotated
from unittest.mock import patch

import httpx
from fastapi import Depends, FastAPI, Request

from emperor_workbench.api._container import WorkbenchContainerSlot
from emperor_workbench.api._errors import ApiError
from emperor_workbench.api._lifespan import create_lifespan
from emperor_workbench.api._middleware import configure_middleware
from emperor_workbench.api._mutations import (
    HttpOperationPolicy,
    build_http_operation_catalog,
    declare_http_operation,
)
from emperor_workbench.api._mutations._journal import MutationJournal
from emperor_workbench.settings import WorkbenchApiSettings


async def _settings_dependency() -> WorkbenchApiSettings:
    raise AssertionError("test dependency must be overridden")


def _counter_app(
    state_root: Path,
    calls: list[str],
    *,
    identities: list[str | None] | None = None,
    started: threading.Event | None = None,
    release: threading.Event | None = None,
    handler_error: BaseException | None = None,
) -> FastAPI:
    from emperor_workbench.api._mutations import (
        current_mutation_identity,
        run_mutation_io,
    )

    settings = WorkbenchApiSettings(
        allow_unsafe_local_mutations=True,
        logs_root=str(state_root / "logs"),
        snapshots_root=str(state_root / "snapshots"),
        state_root=str(state_root),
        training_cancellation_mode="process-group",
    )
    container_slot = WorkbenchContainerSlot()
    app = FastAPI(
        lifespan=create_lifespan(
            settings,
            container_slot=container_slot,
            project_adapter=None,
        )
    )

    async def settings_override() -> WorkbenchApiSettings:
        return settings

    app.dependency_overrides[_settings_dependency] = settings_override

    @app.post("/items/{item_id}")
    @declare_http_operation(HttpOperationPolicy.LOCAL_MUTATION)
    async def create_item(
        item_id: str,
        payload: dict[str, object],
        settings: Annotated[WorkbenchApiSettings, Depends(_settings_dependency)],
    ) -> dict[str, object]:
        if handler_error is not None:
            raise handler_error

        def mutate() -> dict[str, object]:
            calls.append(item_id)
            if identities is not None:
                identities.append(current_mutation_identity())
            if started is not None:
                started.set()
            if release is not None:
                release.wait(timeout=5)
            return {"item": item_id, "call": len(calls), "payload": payload}

        return await run_mutation_io(mutate)

    catalog = build_http_operation_catalog(app.routes)
    configure_middleware(
        app,
        settings,
        catalog,
        container_slot=container_slot,
    )
    return app


def _streaming_app(state_root: Path, calls: list[int]) -> FastAPI:
    settings = WorkbenchApiSettings(
        allow_unsafe_local_mutations=True,
        logs_root=str(state_root / "logs"),
        snapshots_root=str(state_root / "snapshots"),
        state_root=str(state_root),
        training_cancellation_mode="process-group",
    )
    container_slot = WorkbenchContainerSlot()
    app = FastAPI(
        lifespan=create_lifespan(
            settings,
            container_slot=container_slot,
            project_adapter=None,
        )
    )

    async def settings_override() -> WorkbenchApiSettings:
        return settings

    app.dependency_overrides[_settings_dependency] = settings_override

    @app.post("/upload")
    @declare_http_operation(HttpOperationPolicy.LOCAL_MUTATION)
    async def upload(
        request: Request,
        settings: Annotated[WorkbenchApiSettings, Depends(_settings_dependency)],
    ) -> dict[str, int]:
        del settings
        size = 0
        async for chunk in request.stream():
            size += len(chunk)
        calls.append(size)
        return {"size": size}

    configure_middleware(
        app,
        settings,
        build_http_operation_catalog(app.routes),
        container_slot=container_slot,
    )
    return app


class MutationExecutionTests(unittest.TestCase):
    @staticmethod
    async def _post(
        app: FastAPI,
        item_id: str,
        payload: dict[str, object],
        *,
        key: str | None,
    ) -> httpx.Response:
        headers = {"X-Workbench-Mutation": "true"}
        if key is not None:
            headers["Idempotency-Key"] = key
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://localhost",
        ) as client:
            return await client.post(
                f"/items/{item_id}",
                json=payload,
                headers=headers,
            )

    def test_mutations_require_a_valid_idempotency_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            app = _counter_app(Path(tmp), [])

            async def scenario() -> tuple[httpx.Response, ...]:
                async with app.router.lifespan_context(app):
                    return (
                        await self._post(app, "one", {}, key=None),
                        await self._post(app, "one", {}, key=""),
                        await self._post(app, "one", {}, key="x" * 129),
                    )

            missing, empty, too_long = asyncio.run(scenario())

        self.assertEqual(missing.status_code, 428)
        self.assertEqual(empty.status_code, 428)
        self.assertEqual(too_long.status_code, 422)

    def test_middleware_known_failure_uses_finite_json_and_cors(self) -> None:
        from emperor_workbench.api import create_app

        origin = "https://workbench.example"

        async def scenario(root: Path) -> httpx.Response:
            app = create_app(
                WorkbenchApiSettings(
                    logs_root=str(root / "logs"),
                    snapshots_root=str(root / "snapshots"),
                    state_root=str(root / "state"),
                    cors_origins=[origin],
                    allow_unsafe_local_mutations=True,
                    training_cancellation_mode="process-group",
                )
            )
            async with app.router.lifespan_context(app):
                with patch.object(
                    MutationJournal,
                    "admit",
                    side_effect=sqlite3.OperationalError("journal unavailable"),
                ):
                    transport = httpx.ASGITransport(
                        app=app,
                        raise_app_exceptions=False,
                    )
                    async with httpx.AsyncClient(
                        transport=transport,
                        base_url="http://localhost",
                    ) as client:
                        return await client.post(
                            "/config-snapshots",
                            headers={
                                "Origin": origin,
                                "X-Workbench-Mutation": "true",
                                "Idempotency-Key": "known-failure",
                            },
                            json={
                                "modelType": "linears",
                                "model": "linear",
                                "preset": "baseline",
                                "name": "known-failure",
                                "overrides": {},
                            },
                        )

        with tempfile.TemporaryDirectory() as tmp:
            response = asyncio.run(scenario(Path(tmp)))

        self.assertEqual(response.status_code, 503, response.text)
        self.assertEqual(
            response.json(),
            {"detail": ("Mutation execution is temporarily at capacity; retry later.")},
        )
        self.assertEqual(
            response.headers["access-control-allow-origin"],
            origin,
        )

    def test_all_journal_writes_use_finite_failure_translation(self) -> None:
        async def scenario(
            root: Path,
            method_name: str,
        ) -> httpx.Response:
            app = _counter_app(root, [])
            async with app.router.lifespan_context(app):
                with patch.object(
                    MutationJournal,
                    method_name,
                    side_effect=sqlite3.OperationalError(f"{method_name} unavailable"),
                ):
                    return await self._post(
                        app,
                        "one",
                        {"value": 1},
                        key=f"{method_name}-failure",
                    )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            responses = {
                method_name: asyncio.run(scenario(root, method_name))
                for method_name in ("record_request_hash", "complete")
            }

        for method_name, response in responses.items():
            with self.subTest(method_name=method_name):
                self.assertEqual(response.status_code, 503, response.text)
                self.assertEqual(
                    response.json(),
                    {
                        "detail": (
                            "Mutation execution is temporarily at capacity; "
                            "retry later."
                        )
                    },
                )

    def test_abandon_failure_does_not_mask_primary_known_failure(self) -> None:
        async def scenario(root: Path) -> httpx.Response:
            app = _counter_app(
                root,
                [],
                handler_error=ApiError("primary failure", status_code=409),
            )
            async with app.router.lifespan_context(app):
                with patch.object(
                    MutationJournal,
                    "abandon",
                    side_effect=sqlite3.OperationalError("abandon unavailable"),
                ):
                    return await self._post(
                        app,
                        "one",
                        {"value": 1},
                        key="abandon-failure",
                    )

        with tempfile.TemporaryDirectory() as tmp:
            response = asyncio.run(scenario(Path(tmp)))

        self.assertEqual(response.status_code, 409, response.text)
        self.assertEqual(response.json(), {"detail": "primary failure"})

    def test_journal_io_runs_on_the_app_scoped_mutation_executor(self) -> None:
        thread_names: list[str] = []

        def unavailable(*args: object, **kwargs: object) -> None:
            del args, kwargs
            thread_names.append(threading.current_thread().name)
            raise sqlite3.OperationalError("journal unavailable")

        async def scenario(root: Path) -> httpx.Response:
            app = _counter_app(root, [])
            async with app.router.lifespan_context(app):
                with patch.object(
                    MutationJournal,
                    "admit",
                    side_effect=unavailable,
                ):
                    return await self._post(
                        app,
                        "one",
                        {"value": 1},
                        key="thread-probe",
                    )

        with tempfile.TemporaryDirectory() as tmp:
            response = asyncio.run(scenario(Path(tmp)))

        self.assertEqual(response.status_code, 503, response.text)
        self.assertEqual(len(thread_names), 1)
        self.assertTrue(thread_names[0].startswith("workbench-mutation"))

    def test_chunked_multipart_replay_hashes_without_reexecuting_handler(
        self,
    ) -> None:
        chunks = [
            b"--stream-boundary\r\n",
            (
                b'Content-Disposition: form-data; name="archive"; '
                b'filename="logs.zip"\r\n'
            ),
            b"Content-Type: application/zip\r\n\r\n",
            *(b"x" * (128 * 1024) for _ in range(8)),
            b"\r\n--stream-boundary--\r\n",
        ]

        async def scenario(
            root: Path,
        ) -> tuple[httpx.Response, httpx.Response, int]:
            calls: list[int] = []
            app = _streaming_app(root, calls)
            yielded = 0

            async def content_stream():
                nonlocal yielded
                for chunk in chunks:
                    yielded += 1
                    yield chunk

            headers = {
                "content-type": "multipart/form-data; boundary=stream-boundary",
                "X-Workbench-Mutation": "true",
                "Idempotency-Key": "streaming-replay",
            }
            async with app.router.lifespan_context(app):
                transport = httpx.ASGITransport(app=app)
                async with httpx.AsyncClient(
                    transport=transport,
                    base_url="http://localhost",
                ) as client:
                    first = await asyncio.wait_for(
                        client.post(
                            "/upload",
                            content=content_stream(),
                            headers=headers,
                        ),
                        timeout=5,
                    )
                    replay = await asyncio.wait_for(
                        client.post(
                            "/upload",
                            content=content_stream(),
                            headers=headers,
                        ),
                        timeout=5,
                    )
            return first, replay, yielded

        with tempfile.TemporaryDirectory() as tmp:
            first, replay, yielded = asyncio.run(scenario(Path(tmp)))

        self.assertEqual(first.status_code, 200, first.text)
        self.assertEqual(replay.status_code, 200, replay.text)
        self.assertEqual(replay.json(), first.json())
        self.assertEqual(yielded, len(chunks) * 2)

    def test_mutation_identity_is_unique_propagated_and_request_scoped(self) -> None:
        from emperor_workbench.api._mutations import current_mutation_identity

        with tempfile.TemporaryDirectory() as tmp:
            identities: list[str | None] = []
            app = _counter_app(Path(tmp), [], identities=identities)

            async def scenario() -> tuple[httpx.Response, httpx.Response]:
                async with app.router.lifespan_context(app):
                    first = await self._post(app, "one", {}, key="first-key")
                    second = await self._post(app, "two", {}, key="second-key")
                    return first, second

            self.assertIsNone(current_mutation_identity())
            first, second = asyncio.run(scenario())
            self.assertIsNone(current_mutation_identity())

        self.assertEqual(first.status_code, 200, first.text)
        self.assertEqual(second.status_code, 200, second.text)
        self.assertEqual(len(identities), 2)
        self.assertTrue(all(identity is not None for identity in identities))
        self.assertNotEqual(identities[0], identities[1])

    def test_same_key_and_request_replays_persisted_result_after_restart(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state_root = Path(tmp)
            first_calls: list[str] = []
            first_app = _counter_app(state_root, first_calls)

            restarted_calls: list[str] = []
            restarted_app = _counter_app(state_root, restarted_calls)

            async def scenario() -> tuple[httpx.Response, httpx.Response]:
                async with first_app.router.lifespan_context(first_app):
                    first = await self._post(
                        first_app,
                        "one",
                        {"name": "same"},
                        key="replay-key",
                    )
                async with restarted_app.router.lifespan_context(restarted_app):
                    replay = await self._post(
                        restarted_app,
                        "one",
                        {"name": "same"},
                        key="replay-key",
                    )
                return first, replay

            first, replay = asyncio.run(scenario())

        self.assertEqual(first.status_code, 200)
        self.assertEqual(replay.status_code, 200)
        self.assertEqual(replay.json(), first.json())
        self.assertEqual(first_calls, ["one"])
        self.assertEqual(restarted_calls, [])

    def test_same_scoped_key_with_a_different_request_conflicts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            calls: list[str] = []
            app = _counter_app(Path(tmp), calls)

            async def scenario() -> tuple[httpx.Response, httpx.Response]:
                async with app.router.lifespan_context(app):
                    first = await self._post(
                        app,
                        "one",
                        {"value": 1},
                        key="same",
                    )
                    conflict = await self._post(
                        app,
                        "one",
                        {"value": 2},
                        key="same",
                    )
                    return first, conflict

            first, conflict = asyncio.run(scenario())

        self.assertEqual(first.status_code, 200)
        self.assertEqual(conflict.status_code, 409)
        self.assertEqual(calls, ["one"])

    def test_in_progress_retry_conflicts_and_disconnected_work_finishes_once(
        self,
    ) -> None:
        async def scenario(state_root: Path):
            calls: list[str] = []
            started = threading.Event()
            release = threading.Event()
            app = _counter_app(
                state_root,
                calls,
                started=started,
                release=release,
            )

            async with app.router.lifespan_context(app):
                first_task = asyncio.create_task(
                    self._post(app, "one", {"value": 1}, key="disconnect-key")
                )
                for _ in range(200):
                    if started.is_set():
                        break
                    await asyncio.sleep(0.01)
                self.assertTrue(started.is_set())
                first_task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await first_task

                in_progress = await self._post(
                    app,
                    "one",
                    {"value": 1},
                    key="disconnect-key",
                )
                release.set()

                replay: httpx.Response | None = None
                for _ in range(100):
                    replay = await self._post(
                        app,
                        "one",
                        {"value": 1},
                        key="disconnect-key",
                    )
                    if replay.status_code == 200:
                        break
                    await asyncio.sleep(0.01)
                return calls, in_progress, replay

        with tempfile.TemporaryDirectory() as tmp:
            calls, in_progress, replay = asyncio.run(scenario(Path(tmp)))

        self.assertEqual(in_progress.status_code, 409)
        self.assertEqual(in_progress.headers.get("Retry-After"), "1")
        self.assertIsNotNone(replay)
        self.assertEqual(replay.status_code, 200)
        self.assertEqual(calls, ["one"])


if __name__ == "__main__":
    unittest.main()
