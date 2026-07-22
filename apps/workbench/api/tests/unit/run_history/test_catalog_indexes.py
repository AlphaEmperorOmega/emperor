from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from tests.unit.run_history._support import log_run_scanner


def _write_run(logs_root: Path, experiment: str, run_name: str) -> None:
    logs_root.joinpath(
        experiment,
        "linears",
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
            first = log_run_scanner(
                logs_root=logs_root,
                state_root=state_root,
                cache_ttl_seconds=3600,
            )
            self.assertEqual(len(first.list_runs(result_projection="none")), 1)
            catalog_payload = json.loads(
                (state_root / "catalogs" / "run-history.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                set(catalog_payload["entries"][0]),
                {
                    "id",
                    "group",
                    "experiment",
                    "model",
                    "preset",
                    "dataset",
                    "runName",
                    "timestamp",
                    "version",
                    "relativePath",
                    "hasResult",
                    "eventFileCount",
                    "checkpointCount",
                    "hasHparams",
                },
            )

            restarted = log_run_scanner(
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

    def test_scanner_rejects_flat_model_identity_in_persistent_catalog(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            logs_root = root / "logs"
            state_root = root / "state"
            catalog_path = state_root / "catalogs" / "run-history.json"
            _write_run(logs_root, "experiment", "run_20260712_010203")
            first = log_run_scanner(
                logs_root=logs_root,
                state_root=state_root,
                cache_ttl_seconds=3600,
            )
            self.assertEqual(len(first.list_runs(result_projection="none")), 1)

            catalog_payload = json.loads(catalog_path.read_text(encoding="utf-8"))
            catalog_payload["entries"][0]["model"] = "linear"
            catalog_path.write_text(json.dumps(catalog_payload), encoding="utf-8")

            restarted = log_run_scanner(
                logs_root=logs_root,
                state_root=state_root,
                cache_ttl_seconds=3600,
            )
            with patch.object(
                restarted,
                "_version_dirs_and_fingerprint",
                wraps=restarted._version_dirs_and_fingerprint,
            ) as rescan:
                runs = restarted.list_runs(result_projection="none")

            self.assertEqual([run.model for run in runs], ["linears/linear"])
            rescan.assert_called_once()
            rewritten = json.loads(catalog_path.read_text(encoding="utf-8"))
            self.assertEqual(rewritten["entries"][0]["model"], "linears/linear")

    def test_due_reconciliation_discovers_external_run_additions(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            logs_root = root / "logs"
            scanner = log_run_scanner(
                logs_root=logs_root,
                state_root=root / "state",
                cache_ttl_seconds=0,
            )
            _write_run(logs_root, "experiment", "run_20260712_010203")
            self.assertEqual(len(scanner.list_runs(result_projection="none")), 1)

            _write_run(logs_root, "experiment", "run_20260712_020304")

            self.assertEqual(len(scanner.list_runs(result_projection="none")), 2)


if __name__ == "__main__":
    unittest.main()
