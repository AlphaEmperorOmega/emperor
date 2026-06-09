from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

import httpx

from viewer.backend.api import create_app
from viewer.backend.settings import ViewerApiSettings

# Synthetic config schema so the API tests stay fast and model-independent while
# still exercising the real validation (non-default, locked, dedupe).
FAKE_FIELDS: dict[str, Any] = {
    "model": "linears/linear",
    "fields": [
        {
            "key": "learning_rate",
            "type": "float",
            "default": 0.001,
            "nullable": False,
            "locked": False,
            "label": "learning rate",
        },
        {
            "key": "batch_size",
            "type": "int",
            "default": 64,
            "nullable": False,
            "locked": False,
            "label": "batch size",
        },
        {
            "key": "seed",
            "type": "int",
            "default": 42,
            "nullable": False,
            "locked": True,
            "label": "seed",
        },
    ],
}


def fake_config_schema(model: str, preset: str | None = None) -> dict[str, Any]:
    return FAKE_FIELDS


@mock.patch(
    "viewer.backend.services.config_snapshots.config_schema",
    fake_config_schema,
)
class ConfigSnapshotApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        root = Path(self._tmp.name)
        app = create_app(
            ViewerApiSettings(
                logs_root=str(root / "logs"),
                snapshots_root=str(root / "snapshots"),
                auth_mode="none",
            )
        )
        self.app = app

    def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        async def request() -> httpx.Response:
            transport = httpx.ASGITransport(app=self.app)
            async with httpx.AsyncClient(
                transport=transport,
                base_url="http://testserver",
            ) as client:
                return await client.request(method, path, **kwargs)

        return asyncio.run(request())

    def _create(self, **overrides: str):
        return self._request(
            "POST",
            "/config-snapshots",
            json={
                "model": "linears/linear",
                "preset": "baseline",
                "name": "",
                "overrides": overrides,
            },
        )

    def test_create_then_list_persists_snapshot(self) -> None:
        response = self._create(learning_rate="0.01", batch_size="128")
        self.assertEqual(response.status_code, 200, response.text)
        snapshot = response.json()
        self.assertEqual(
            snapshot["overrides"], {"learning_rate": "0.01", "batch_size": "128"}
        )
        self.assertTrue(snapshot["name"])

        listed = self._request(
            "GET",
            "/config-snapshots",
            params={"model": "linears/linear"},
        )
        self.assertEqual(listed.status_code, 200)
        body = listed.json()
        self.assertEqual(body["model"], "linears/linear")
        self.assertEqual([s["id"] for s in body["snapshots"]], [snapshot["id"]])

    def test_rejects_unsafe_storage_paths(self) -> None:
        listed = self._request(
            "GET",
            "/config-snapshots",
            params={"model": "../outside"},
        )
        created = self._request(
            "POST",
            "/config-snapshots",
            json={
                "model": "../outside",
                "preset": "baseline",
                "name": "",
                "overrides": {"learning_rate": "0.01"},
            },
        )

        self.assertEqual(listed.status_code, 400)
        self.assertIn("Invalid config snapshot", listed.json()["detail"])
        self.assertEqual(created.status_code, 400)
        self.assertIn("Invalid config snapshot", created.json()["detail"])

    def test_create_rejects_default_only_override(self) -> None:
        response = self._create(learning_rate="0.001")
        self.assertEqual(response.status_code, 400)
        self.assertIn("non-default", response.json()["detail"])

    def test_create_rejects_locked_field(self) -> None:
        response = self._create(seed="7")
        self.assertEqual(response.status_code, 400)
        self.assertIn("preset-locked", response.json()["detail"])

    def test_create_rejects_duplicate(self) -> None:
        self.assertEqual(self._create(learning_rate="0.01").status_code, 200)
        duplicate = self._create(learning_rate="0.01")
        self.assertEqual(duplicate.status_code, 400)
        self.assertIn("already exists", duplicate.json()["detail"])

    def test_rename_updates_name(self) -> None:
        snapshot_id = self._create(learning_rate="0.01").json()["id"]
        renamed = self._request(
            "PATCH", f"/config-snapshots/{snapshot_id}", json={"name": "tuned lr"}
        )
        self.assertEqual(renamed.status_code, 200)
        self.assertEqual(renamed.json()["name"], "tuned lr")

    def test_rename_rejects_empty_name(self) -> None:
        snapshot_id = self._create(learning_rate="0.01").json()["id"]
        renamed = self._request(
            "PATCH", f"/config-snapshots/{snapshot_id}", json={"name": "   "}
        )
        self.assertEqual(renamed.status_code, 400)

    def test_delete_returns_remaining_snapshots(self) -> None:
        snapshot_id = self._create(learning_rate="0.01").json()["id"]
        deleted = self._request("DELETE", f"/config-snapshots/{snapshot_id}")
        self.assertEqual(deleted.status_code, 200)
        self.assertEqual(deleted.json(), {"model": "linears/linear", "snapshots": []})

    def test_delete_unknown_snapshot_is_rejected(self) -> None:
        response = self._request("DELETE", "/config-snapshots/missing")
        self.assertEqual(response.status_code, 400)
        self.assertIn("Unknown config snapshot", response.json()["detail"])


if __name__ == "__main__":
    unittest.main()
