from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path

import httpx
from fastapi import Request

from workbench.backend.core.config import WorkbenchApiSettings
from workbench.backend.main import create_app
from workbench.backend.schemas import (
    ConfigSnapshotCreateRequest,
    TrainingSearchRequest,
)
from workbench.backend.training_jobs.limits import (
    MAX_TRAINING_SEARCH_AXIS_VALUES,
)


class JsonRequestBudgetTests(unittest.TestCase):
    def test_chunked_json_is_rejected_before_the_complete_body_is_read(self) -> None:
        yielded_chunks: list[int] = []

        async def body():  # type: ignore[no-untyped-def]
            for index in range(4):
                yielded_chunks.append(index)
                yield b"x" * (400 * 1024)

        async def request() -> httpx.Response:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                app = create_app(
                    WorkbenchApiSettings(
                        trusted_hosts=["testserver"],
                        logs_root=str(root / "logs"),
                        snapshots_root=str(root / "snapshots"),
                        state_root=str(root / "state"),
                    )
                )

                @app.post("/body-probe")
                async def body_probe(request: Request) -> dict[str, int]:
                    return {"bytes": len(await request.body())}

                transport = httpx.ASGITransport(app=app)
                async with httpx.AsyncClient(
                    transport=transport,
                    base_url="http://testserver",
                ) as client:
                    return await client.post(
                        "/body-probe",
                        content=body(),
                        headers={"Content-Type": "application/json"},
                    )

        response = asyncio.run(request())

        self.assertEqual(response.status_code, 413)
        self.assertEqual(
            response.json(),
            {"detail": "JSON request body exceeds the 1048576 byte limit."},
        )
        self.assertEqual(yielded_chunks, [0, 1, 2])

    def test_unbounded_request_collections_and_strings_have_schema_caps(self) -> None:
        with self.assertRaisesRegex(ValueError, "at most 256 characters"):
            ConfigSnapshotCreateRequest(
                modelType="linears",
                model="linear",
                preset="baseline",
                name="n" * 257,
                overrides={"HIDDEN_DIM": "128"},
            )
        with self.assertRaisesRegex(ValueError, "at most 50 items"):
            TrainingSearchRequest(
                mode="grid",
                values={"HIDDEN_DIM": list(range(MAX_TRAINING_SEARCH_AXIS_VALUES + 1))},
            )


if __name__ == "__main__":
    unittest.main()
