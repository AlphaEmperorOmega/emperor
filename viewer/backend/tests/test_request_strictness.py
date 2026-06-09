from __future__ import annotations

import asyncio
import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import httpx

from viewer.backend.api import ViewerApiSettings, create_app
from viewer.backend.training_jobs import TrainingJobManager


class FakeProcess:
    pid = 1234

    def poll(self) -> int | None:
        return None

    def terminate(self) -> None:
        pass


class FakeRunner:
    def start(self, command, *, cwd, env, log_path):
        log_path.write_text("fake training log\n", encoding="utf-8")
        return FakeProcess()


REQUEST_BODY_ENDPOINT_CASES = (
    (
        "/inspect",
        {
            "model": "linears/linear",
            "preset": "baseline",
            "overrides": {},
        },
    ),
    (
        "/logs/runs/delete-plan",
        {
            "experiments": [],
            "datasets": [],
            "models": [],
            "presets": [],
            "runIds": [],
        },
    ),
    (
        "/logs/runs/delete",
        {
            "experiments": [],
            "datasets": [],
            "models": [],
            "presets": [],
            "runIds": [],
        },
    ),
    (
        "/logs/tags",
        {
            "runIds": [],
        },
    ),
    (
        "/logs/scalars",
        {
            "runIds": [],
            "tags": [],
        },
    ),
    (
        "/logs/parameter-status",
        {
            "runIds": [],
        },
    ),
    (
        "/training/jobs",
        {
            "model": "linears/linear",
            "preset": "baseline",
            "presets": ["baseline"],
            "datasets": ["Mnist"],
            "overrides": {},
            "logFolder": "extra_field_rejection",
            "monitors": [],
        },
    ),
    (
        "/training/run-plan",
        {
            "model": "linears/linear",
            "preset": "baseline",
            "presets": ["baseline"],
            "datasets": ["Mnist"],
            "overrides": {},
            "logFolder": "extra_field_rejection",
        },
    ),
)


class RequestStrictnessTests(unittest.TestCase):
    def test_body_endpoints_reject_extra_request_fields(self) -> None:
        unexpected_field = "unexpectedField"

        with tempfile.TemporaryDirectory() as tmp:
            logs_root = Path(tmp) / "logs"
            manager = TrainingJobManager(
                root=Path(tmp) / "jobs",
                logs_root=logs_root,
                runner=FakeRunner(),
            )
            test_app = create_app(
                ViewerApiSettings(logs_root=str(logs_root)),
                training_manager=manager,
            )

            for path, payload in REQUEST_BODY_ENDPOINT_CASES:
                with self.subTest(path=path):
                    response = asyncio.run(
                        self._post_with_extra_field(
                            test_app,
                            path,
                            payload,
                            unexpected_field,
                        )
                    )

                    self.assertEqual(response.status_code, 422)
                    self.assert_extra_forbidden(response, unexpected_field)

    async def _post_with_extra_field(
        self,
        test_app,
        path: str,
        payload: dict[str, object],
        unexpected_field: str,
    ) -> httpx.Response:
        transport = httpx.ASGITransport(app=test_app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            return await client.post(
                path,
                json={**payload, unexpected_field: "unexpected"},
            )

    def assert_extra_forbidden(
        self,
        response: httpx.Response,
        unexpected_field: str,
    ) -> None:
        errors = response.json().get("detail", [])
        self.assertTrue(
            any(
                isinstance(error, dict)
                and error.get("type") == "extra_forbidden"
                and unexpected_field
                in {str(location) for location in error.get("loc", [])}
                for error in errors
            ),
            response.text,
        )


if __name__ == "__main__":
    unittest.main()
