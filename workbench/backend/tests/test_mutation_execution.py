from __future__ import annotations

import asyncio
import tempfile
import threading
import unittest
from pathlib import Path
from typing import Annotated

import httpx
from fastapi import Depends, FastAPI

from workbench.backend.api.mutation_policy import (
    HttpOperationPolicy,
    build_http_operation_catalog,
    declare_http_operation,
)
from workbench.backend.core.config import WorkbenchApiSettings
from workbench.backend.middleware import configure_middleware


async def _settings_dependency() -> WorkbenchApiSettings:
    raise AssertionError("test dependency must be overridden")


def _counter_app(
    state_root: Path,
    calls: list[str],
    *,
    identities: list[str | None] | None = None,
    started: threading.Event | None = None,
    release: threading.Event | None = None,
) -> FastAPI:
    from workbench.backend.mutation_execution import (
        current_mutation_identity,
        run_mutation_io,
    )

    settings = WorkbenchApiSettings(
        allow_unsafe_local_mutations=True,
        state_root=str(state_root),
    )
    app = FastAPI()

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
    configure_middleware(app, settings, catalog)
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

            missing = asyncio.run(self._post(app, "one", {}, key=None))
            empty = asyncio.run(self._post(app, "one", {}, key=""))
            too_long = asyncio.run(self._post(app, "one", {}, key="x" * 129))

        self.assertEqual(missing.status_code, 428)
        self.assertEqual(empty.status_code, 428)
        self.assertEqual(too_long.status_code, 422)

    def test_mutation_identity_is_unique_propagated_and_request_scoped(self) -> None:
        from workbench.backend.mutation_context import current_mutation_identity

        with tempfile.TemporaryDirectory() as tmp:
            identities: list[str | None] = []
            app = _counter_app(Path(tmp), [], identities=identities)

            self.assertIsNone(current_mutation_identity())
            first = asyncio.run(self._post(app, "one", {}, key="first-key"))
            self.assertIsNone(current_mutation_identity())
            second = asyncio.run(self._post(app, "two", {}, key="second-key"))
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
            first = asyncio.run(
                self._post(first_app, "one", {"name": "same"}, key="replay-key")
            )

            restarted_calls: list[str] = []
            restarted_app = _counter_app(state_root, restarted_calls)
            replay = asyncio.run(
                self._post(
                    restarted_app,
                    "one",
                    {"name": "same"},
                    key="replay-key",
                )
            )

        self.assertEqual(first.status_code, 200)
        self.assertEqual(replay.status_code, 200)
        self.assertEqual(replay.json(), first.json())
        self.assertEqual(first_calls, ["one"])
        self.assertEqual(restarted_calls, [])

    def test_same_scoped_key_with_a_different_request_conflicts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            calls: list[str] = []
            app = _counter_app(Path(tmp), calls)
            first = asyncio.run(self._post(app, "one", {"value": 1}, key="same"))
            conflict = asyncio.run(self._post(app, "one", {"value": 2}, key="same"))

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
