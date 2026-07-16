from __future__ import annotations

import asyncio
import json
import os
import tempfile
import unittest
import uuid
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import httpx

from emperor_workbench.api import create_app
from emperor_workbench.settings import WorkbenchApiSettings
from tests.support import lifespan_client
from tests.support.training_jobs import (
    TrainingJobServiceHarness,
    create_app_with_training_service,
)


class FakeProcess:
    pid = 1234

    def poll(self) -> int | None:
        return None

    def terminate(self) -> None:
        pass

    def wait(self, timeout: float | None = None) -> int:
        return -15

    def kill(self) -> None:
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
        "/logs/checkpoints",
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
    def test_json_body_requires_json_content_type_and_accepts_json_media_types(
        self,
    ) -> None:
        payload = {
            "experiments": [],
            "datasets": [],
            "models": [],
            "presets": [],
            "runIds": [],
        }

        with tempfile.TemporaryDirectory() as tmp:
            test_app = create_app(
                WorkbenchApiSettings(logs_root=str(Path(tmp) / "logs"))
            )

            async def call_api() -> tuple[
                httpx.Response,
                httpx.Response,
                httpx.Response,
            ]:
                encoded = json.dumps(payload).encode()
                async with lifespan_client(test_app) as client:
                    missing_content_type = await client.post(
                        "/logs/runs/delete-plan",
                        content=encoded,
                    )
                    application_json = await client.post(
                        "/logs/runs/delete-plan",
                        json=payload,
                    )
                    vendor_json = await client.post(
                        "/logs/runs/delete-plan",
                        content=encoded,
                        headers={"Content-Type": "application/vnd.emperor+json"},
                    )
                    return missing_content_type, application_json, vendor_json

            missing_content_type, application_json, vendor_json = asyncio.run(
                call_api()
            )

        self.assertIs(test_app.router.strict_content_type, True)
        self.assertEqual(missing_content_type.status_code, 422)
        self.assertEqual(application_json.status_code, 200, application_json.text)
        self.assertEqual(vendor_json.status_code, 200, vendor_json.text)

    def test_body_endpoints_reject_extra_request_fields(self) -> None:
        unexpected_field = "unexpectedField"

        with tempfile.TemporaryDirectory() as tmp:
            logs_root = Path(tmp) / "logs"
            manager = TrainingJobServiceHarness(
                root=Path(tmp) / "jobs",
                logs_root=logs_root,
                runner=FakeRunner(),
            )
            test_app = create_app_with_training_service(
                WorkbenchApiSettings(
                    logs_root=str(logs_root),
                    allow_unsafe_local_mutations=True,
                ),
                manager,
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

    def test_log_fanout_request_limits_reject_overlarge_lists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logs_root = Path(tmp) / "logs"
            test_app = create_app(
                WorkbenchApiSettings(
                    logs_root=str(logs_root),
                    allow_unsafe_local_mutations=True,
                )
            )

            async def call_api() -> tuple[httpx.Response, httpx.Response]:
                async with lifespan_client(
                    test_app,
                    headers={
                        "X-Workbench-Mutation": "true",
                        "Idempotency-Key": uuid.uuid4().hex,
                    },
                ) as client:
                    run_ids = [f"run-{index}" for index in range(51)]
                    parameter_status_response = await client.post(
                        "/logs/parameter-status",
                        json={"runIds": run_ids},
                    )
                    delete_filter_response = await client.post(
                        "/logs/runs/delete-plan",
                        json={
                            "experiments": [],
                            "datasets": [],
                            "models": [],
                            "presets": [],
                            "runIds": run_ids,
                        },
                    )
                    return parameter_status_response, delete_filter_response

            parameter_status_response, delete_filter_response = asyncio.run(call_api())

        self.assertEqual(parameter_status_response.status_code, 422)
        self.assertEqual(delete_filter_response.status_code, 422)

    async def _post_with_extra_field(
        self,
        test_app,
        path: str,
        payload: dict[str, object],
        unexpected_field: str,
    ) -> httpx.Response:
        async with lifespan_client(
            test_app,
            headers={
                "X-Workbench-Mutation": "true",
                "Idempotency-Key": uuid.uuid4().hex,
            },
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
