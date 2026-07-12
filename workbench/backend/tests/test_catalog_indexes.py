from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from workbench.backend.config_snapshots import (
    ConfigSnapshotRecord,
    FileSystemConfigSnapshotStore,
)
from workbench.backend.run_history.scanner import LogRunScanner


def _write_run(logs_root: Path, experiment: str, run_name: str) -> None:
    logs_root.joinpath(
        experiment,
        "linear",
        "BASELINE",
        "Mnist",
        run_name,
        "version_0",
    ).mkdir(parents=True)


class RunCatalogIndexTests(unittest.TestCase):
    def test_new_scanner_loads_persistent_catalog_without_recursive_rescan(
        self,
    ) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            logs_root = root / "logs"
            state_root = root / "state"
            _write_run(logs_root, "experiment", "run_20260712_010203")
            first = LogRunScanner(
                logs_root=logs_root,
                state_root=state_root,
                cache_ttl_seconds=3600,
            )
            self.assertEqual(len(first.list_runs(result_projection="none")), 1)

            restarted = LogRunScanner(
                logs_root=logs_root,
                state_root=state_root,
                cache_ttl_seconds=3600,
            )
            with patch.object(
                restarted,
                "_version_dirs_and_fingerprint",
                side_effect=AssertionError("persistent catalog must be reused"),
            ):
                runs = restarted.list_runs(result_projection="none")

            self.assertEqual(len(runs), 1)
            self.assertTrue((state_root / "catalogs" / "run-history.json").is_file())

    def test_due_reconciliation_discovers_external_run_additions(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            logs_root = root / "logs"
            scanner = LogRunScanner(
                logs_root=logs_root,
                state_root=root / "state",
                cache_ttl_seconds=0,
            )
            _write_run(logs_root, "experiment", "run_20260712_010203")
            self.assertEqual(len(scanner.list_runs(result_projection="none")), 1)

            _write_run(logs_root, "experiment", "run_20260712_020304")

            self.assertEqual(len(scanner.list_runs(result_projection="none")), 2)


class ConfigSnapshotCatalogIndexTests(unittest.TestCase):
    def test_snapshot_mutations_publish_a_restart_safe_catalog(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshots_root = root / "snapshots"
            state_root = root / "state"
            first = FileSystemConfigSnapshotStore(
                snapshots_root,
                state_root=state_root,
                reconciliation_interval_seconds=3600,
            )
            first.create(
                ConfigSnapshotRecord(
                    id="snapshot-1",
                    model="linears/linear",
                    preset="baseline",
                    name="First",
                    overrides={"hidden_dim": "128"},
                )
            )

            restarted = FileSystemConfigSnapshotStore(
                snapshots_root,
                state_root=state_root,
                reconciliation_interval_seconds=3600,
            )
            with patch.object(
                restarted,
                "_scan_all",
                side_effect=AssertionError("persistent catalog must be reused"),
            ):
                snapshots = restarted.list_all()
            self.assertEqual([snapshot.id for snapshot in snapshots], ["snapshot-1"])

            self.assertTrue(restarted.delete("snapshot-1"))
            third = FileSystemConfigSnapshotStore(
                snapshots_root,
                state_root=state_root,
                reconciliation_interval_seconds=3600,
            )
            with patch.object(
                third,
                "_scan_all",
                side_effect=AssertionError("delete must update the catalog"),
            ):
                self.assertEqual(third.list_all(), [])

    def test_due_reconciliation_discovers_external_snapshot_additions(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshots_root = root / "snapshots"
            store = FileSystemConfigSnapshotStore(
                snapshots_root,
                state_root=root / "state",
                reconciliation_interval_seconds=0,
            )
            self.assertEqual(store.list_all(), [])
            external = snapshots_root / "linears" / "linear" / "external.json"
            external.parent.mkdir(parents=True)
            external.write_text(
                json.dumps(
                    {
                        "id": "external",
                        "modelType": "linears",
                        "model": "linear",
                        "preset": "baseline",
                        "name": "External",
                        "overrides": {},
                        "created_at": "2026-07-12T00:00:00+00:00",
                        "updated_at": "2026-07-12T00:00:00+00:00",
                    }
                ),
                encoding="utf-8",
            )

            self.assertEqual(
                [snapshot.id for snapshot in store.list_all()],
                ["external"],
            )


if __name__ == "__main__":
    unittest.main()
