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
    "modelType": "linears",
    "model": "linear",
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


def split_test_model(model: str) -> tuple[str, str]:
    if "/" not in model:
        return "linears", model
    model_type, model_name = model.split("/", 1)
    return model_type, model_name


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

    def _create(
        self,
        *,
        model: str = "linears/linear",
        preset: str = "baseline",
        name: str | None = None,
        **overrides: str,
    ):
        snapshot_name = name
        if snapshot_name is None:
            override_name = "_".join(
                f"{key}_{value}" for key, value in sorted(overrides.items())
            )
            snapshot_name = f"{preset}_{override_name or 'snapshot'}"
        model_type, model_name = split_test_model(model)
        return self._request(
            "POST",
            "/config-snapshots",
            json={
                "modelType": model_type,
                "model": model_name,
                "preset": preset,
                "name": snapshot_name,
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
            params={"modelType": "linears", "model": "linear"},
        )
        self.assertEqual(listed.status_code, 200)
        body = listed.json()
        self.assertEqual(body["modelType"], "linears")
        self.assertEqual(body["model"], "linear")
        self.assertEqual([s["id"] for s in body["snapshots"]], [snapshot["id"]])

    def test_library_lists_all_snapshots_while_scoped_list_stays_filtered(self) -> None:
        linear_snapshot = self._create(
            model="linears/linear",
            learning_rate="0.01",
        ).json()
        adaptive_snapshot = self._create(
            model="linears/linear_adaptive",
            learning_rate="0.02",
        ).json()

        library = self._request("GET", "/config-snapshots/library")
        self.assertEqual(library.status_code, 200, library.text)
        self.assertEqual(
            [
                (snapshot["modelType"], snapshot["model"], snapshot["id"])
                for snapshot in library.json()["snapshots"]
            ],
            [
                ("linears", "linear", linear_snapshot["id"]),
                ("linears", "linear_adaptive", adaptive_snapshot["id"]),
            ],
        )

        scoped = self._request(
            "GET",
            "/config-snapshots",
            params={"modelType": "linears", "model": "linear"},
        )
        self.assertEqual(
            [
                (snapshot["modelType"], snapshot["model"], snapshot["id"])
                for snapshot in scoped.json()["snapshots"]
            ],
            [("linears", "linear", linear_snapshot["id"])],
        )

    def test_rejects_unsafe_storage_paths(self) -> None:
        listed = self._request(
            "GET",
            "/config-snapshots",
            params={"modelType": "linears", "model": "../outside"},
        )
        created = self._request(
            "POST",
            "/config-snapshots",
            json={
                "modelType": "linears",
                "model": "../outside",
                "preset": "baseline",
                "name": "unsafe",
                "overrides": {"learning_rate": "0.01"},
            },
        )

        self.assertEqual(listed.status_code, 400)
        self.assertIn("Unknown model", listed.json()["detail"])
        self.assertEqual(created.status_code, 400)
        self.assertIn("Unknown model", created.json()["detail"])

    def test_create_rejects_default_only_override(self) -> None:
        response = self._create(learning_rate="0.001")
        self.assertEqual(response.status_code, 400)
        self.assertIn("non-default", response.json()["detail"])

    def test_create_rejects_empty_name(self) -> None:
        response = self._create(name="   ", learning_rate="0.01")
        self.assertEqual(response.status_code, 400)
        self.assertIn("name cannot be empty", response.json()["detail"])

    def test_create_rejects_locked_field(self) -> None:
        response = self._create(seed="7")
        self.assertEqual(response.status_code, 400)
        self.assertIn("preset-locked", response.json()["detail"])

    def test_create_rejects_duplicate(self) -> None:
        self.assertEqual(self._create(learning_rate="0.01").status_code, 200)
        duplicate = self._create(name="same values", learning_rate="0.01")
        self.assertEqual(duplicate.status_code, 400)
        self.assertIn("already exists", duplicate.json()["detail"])

    def test_create_rejects_duplicate_name(self) -> None:
        self.assertEqual(
            self._create(name="Tuned LR", learning_rate="0.01").status_code,
            200,
        )
        duplicate = self._create(name=" tuned lr ", batch_size="128")
        self.assertEqual(duplicate.status_code, 400)
        self.assertIn("name already exists", duplicate.json()["detail"])

    def test_rename_updates_name(self) -> None:
        snapshot_id = self._create(learning_rate="0.01").json()["id"]
        renamed = self._request(
            "PATCH", f"/config-snapshots/{snapshot_id}", json={"name": "tuned lr"}
        )
        self.assertEqual(renamed.status_code, 200)
        self.assertEqual(renamed.json()["name"], "tuned lr")

    def test_update_snapshot_changes_name_and_overrides_in_place(self) -> None:
        snapshot = self._create(learning_rate="0.01").json()
        with mock.patch(
            "viewer.backend.services.config_snapshots._now",
            return_value="2026-06-02T00:00:00+00:00",
        ):
            updated = self._request(
                "PATCH",
                f"/config-snapshots/{snapshot['id']}",
                json={
                    "name": "larger batch",
                    "overrides": {
                        "learning_rate": "0.001",
                        "batch_size": "128",
                    },
                },
            )

        self.assertEqual(updated.status_code, 200, updated.text)
        body = updated.json()
        self.assertEqual(body["id"], snapshot["id"])
        self.assertEqual(body["modelType"], snapshot["modelType"])
        self.assertEqual(body["model"], snapshot["model"])
        self.assertEqual(body["preset"], snapshot["preset"])
        self.assertEqual(body["createdAt"], snapshot["createdAt"])
        self.assertEqual(body["updatedAt"], "2026-06-02T00:00:00+00:00")
        self.assertEqual(body["name"], "larger batch")
        self.assertEqual(body["overrides"], {"batch_size": "128"})

    def test_rename_rejects_empty_name(self) -> None:
        snapshot_id = self._create(learning_rate="0.01").json()["id"]
        renamed = self._request(
            "PATCH", f"/config-snapshots/{snapshot_id}", json={"name": "   "}
        )
        self.assertEqual(renamed.status_code, 400)

    def test_rename_rejects_duplicate_name(self) -> None:
        first = self._create(name="Tuned LR", learning_rate="0.01").json()
        second = self._create(name="Large Batch", batch_size="128").json()
        renamed = self._request(
            "PATCH",
            f"/config-snapshots/{second['id']}",
            json={"name": first["name"].lower()},
        )
        self.assertEqual(renamed.status_code, 400)
        self.assertIn("name already exists", renamed.json()["detail"])

    def test_update_rejects_default_only_override(self) -> None:
        snapshot_id = self._create(learning_rate="0.01").json()["id"]
        updated = self._request(
            "PATCH",
            f"/config-snapshots/{snapshot_id}",
            json={"overrides": {"learning_rate": "0.001"}},
        )

        self.assertEqual(updated.status_code, 400)
        self.assertIn("non-default", updated.json()["detail"])

    def test_update_rejects_locked_field(self) -> None:
        snapshot_id = self._create(learning_rate="0.01").json()["id"]
        updated = self._request(
            "PATCH",
            f"/config-snapshots/{snapshot_id}",
            json={"overrides": {"seed": "7"}},
        )

        self.assertEqual(updated.status_code, 400)
        self.assertIn("preset-locked", updated.json()["detail"])

    def test_update_rejects_duplicate_other_snapshot(self) -> None:
        first = self._create(learning_rate="0.01").json()
        second = self._create(batch_size="128").json()

        unchanged_self = self._request(
            "PATCH",
            f"/config-snapshots/{first['id']}",
            json={"name": "same values", "overrides": {"learning_rate": "0.01"}},
        )
        duplicate = self._request(
            "PATCH",
            f"/config-snapshots/{second['id']}",
            json={"overrides": {"learning_rate": "0.01"}},
        )

        self.assertEqual(unchanged_self.status_code, 200, unchanged_self.text)
        self.assertEqual(duplicate.status_code, 400)
        self.assertIn("already exists", duplicate.json()["detail"])

    def test_update_unknown_snapshot_is_rejected(self) -> None:
        response = self._request(
            "PATCH",
            "/config-snapshots/missing",
            json={"name": "missing", "overrides": {"learning_rate": "0.01"}},
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("Unknown config snapshot", response.json()["detail"])

    def test_delete_returns_remaining_snapshots(self) -> None:
        snapshot_id = self._create(learning_rate="0.01").json()["id"]
        deleted = self._request("DELETE", f"/config-snapshots/{snapshot_id}")
        self.assertEqual(deleted.status_code, 200)
        self.assertEqual(
            deleted.json(),
            {"modelType": "linears", "model": "linear", "snapshots": []},
        )

    def test_delete_unknown_snapshot_is_rejected(self) -> None:
        response = self._request("DELETE", "/config-snapshots/missing")
        self.assertEqual(response.status_code, 400)
        self.assertIn("Unknown config snapshot", response.json()["detail"])


if __name__ == "__main__":
    unittest.main()
