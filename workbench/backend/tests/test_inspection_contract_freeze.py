from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import unittest
from collections.abc import Callable
from pathlib import Path
from typing import Any

from workbench.backend.inspector.discovery import (
    discover_models,
    list_model_datasets,
    list_model_monitors,
    list_model_presets,
)
from workbench.backend.inspector.schema import config_schema, search_space_schema
from workbench.backend.inspector.service import inspect_model
from workbench.backend.schemas import InspectResponse

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "inspection_contract_v1.json"


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=False,
    ).encode()


def _digest(value: Any) -> tuple[str, int]:
    payload = _canonical_json(value)
    return hashlib.sha256(payload).hexdigest(), len(payload)


class InspectionContractFreezeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))

    def test_graph_http_payloads_match_ordered_goldens(self) -> None:
        for name, expected in self.fixture["graphs"].items():
            with self.subTest(case=name):
                result = inspect_model(
                    expected["model"],
                    expected["preset"],
                    dataset=expected["dataset"],
                    experiment_task=expected["experimentTask"],
                )
                payload = InspectResponse.model_validate(result).model_dump(mode="json")

                self.assertEqual(
                    _digest(payload),
                    (expected["sha256"], expected["bytes"]),
                )
                self.assertEqual(len(payload["nodes"]), expected["nodeCount"])
                self.assertEqual(len(payload["edges"]), expected["edgeCount"])
                self.assertEqual(
                    list(payload),
                    [
                        "modelType",
                        "model",
                        "preset",
                        "parameterCount",
                        "parameterSizeBytes",
                        "nodes",
                        "edges",
                    ],
                )

    def test_configuration_and_search_payloads_match_ordered_goldens(self) -> None:
        for name, expected in self.fixture["schemas"].items():
            with self.subTest(case=name):
                schema = config_schema(expected["model"], expected["preset"])
                search = search_space_schema(
                    expected["model"],
                    expected["preset"],
                    expected["presets"],
                )

                self.assertEqual(
                    _digest(schema),
                    (expected["configSha256"], expected["configBytes"]),
                )
                self.assertEqual(len(schema["fields"]), expected["fieldCount"])
                self.assertEqual(
                    _digest(search),
                    (expected["searchSha256"], expected["searchBytes"]),
                )
                self.assertEqual(len(search["axes"]), expected["axisCount"])

    def test_catalog_and_model_metadata_payloads_match_goldens(self) -> None:
        catalog_digest, catalog_bytes, model_count = self.fixture["catalog"]
        models = discover_models()
        self.assertEqual(_digest(models), (catalog_digest, catalog_bytes))
        self.assertEqual(len(models), model_count)

        for model_name, expected in self.fixture["discovery"].items():
            with self.subTest(model=model_name):
                payload = {
                    "presets": list_model_presets(model_name),
                    "datasets": list_model_datasets(model_name),
                    "monitors": list_model_monitors(model_name),
                }
                digest, byte_count, presets, groups, monitors = expected
                self.assertEqual(_digest(payload), (digest, byte_count))
                self.assertEqual(len(payload["presets"]), presets)
                self.assertEqual(len(payload["datasets"]["datasetGroups"]), groups)
                self.assertEqual(len(payload["monitors"]), monitors)

    def test_inspection_failures_match_class_detail_and_status_goldens(self) -> None:
        cases: dict[str, Callable[[], Any]] = {
            "unknown-model": lambda: inspect_model("unknown/model", "baseline"),
            "unknown-override": lambda: inspect_model(
                "linears/linear",
                "baseline",
                overrides={"NO_SUCH_FIELD": "1"},
            ),
            "malformed-override": lambda: inspect_model(
                "linears/linear",
                "baseline",
                overrides={"HIDDEN_DIM": "not-an-int"},
            ),
            "locked-override": lambda: inspect_model(
                "linears/linear",
                "gating",
                overrides={"GATE_FLAG": "false"},
            ),
            "incompatible-dataset": lambda: inspect_model(
                "linears/linear",
                "baseline",
                dataset="WikiText2",
            ),
            "incompatible-task": lambda: inspect_model(
                "linears/linear",
                "baseline",
                experiment_task="causal-language-modeling",
            ),
            "path-dataset": lambda: inspect_model(
                "linears/linear",
                "baseline",
                dataset="../Mnist",
            ),
        }

        for name, call in cases.items():
            with self.subTest(case=name):
                with self.assertRaises(Exception) as raised:
                    call()
                error = raised.exception
                actual = [
                    f"{type(error).__module__}.{type(error).__qualname__}",
                    getattr(error, "detail", str(error)),
                    getattr(error, "status_code", None),
                ]
                self.assertEqual(actual, self.fixture["errors"][name])

    def test_cli_json_and_workbench_graph_are_equivalent(self) -> None:
        completed = self._run_cli(
            "--model-type",
            "linears",
            "--model",
            "linear",
            "--preset",
            "baseline",
            "--datasets",
            "Mnist",
            "--format",
            "json",
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(
            json.loads(completed.stdout),
            inspect_model(
                "linears/linear",
                "baseline",
                dataset="Mnist",
            ),
        )

    def test_cli_error_and_exit_contracts_are_stable(self) -> None:
        cases = {
            "unknown-model": (
                "--model-type",
                "unknown",
                "--model",
                "model",
                "--preset",
                "baseline",
            ),
            "invalid-preset": (
                "--model-type",
                "linears",
                "--model",
                "linear",
                "--preset",
                "nope",
                "--datasets",
                "Mnist",
            ),
            "locked-override": (
                "--model-type",
                "linears",
                "--model",
                "linear",
                "--preset",
                "gating",
                "--datasets",
                "Mnist",
                "--config",
                "--gate-flag",
                "false",
            ),
            "pre-inspection-malformed-override": (
                "--model-type",
                "linears",
                "--model",
                "linear",
                "--preset",
                "baseline",
                "--datasets",
                "Mnist",
                "--config",
                "--hidden-dim",
                "not-an-int",
            ),
        }

        for name, arguments in cases.items():
            with self.subTest(case=name):
                completed = self._run_cli(*arguments)
                expected_code, expected_last_line, expected_traceback = self.fixture[
                    "cliErrors"
                ][name]
                self.assertEqual(completed.returncode, expected_code)
                self.assertEqual(completed.stdout, "")
                self.assertEqual(
                    completed.stderr.strip().splitlines()[-1],
                    expected_last_line,
                )
                self.assertEqual(
                    "Traceback (most recent call last):" in completed.stderr,
                    expected_traceback,
                )

    @staticmethod
    def _run_cli(*arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-m", "workbench.backend.cli", *arguments],
            cwd=Path(__file__).resolve().parents[3],
            env={**os.environ, "MPLCONFIGDIR": "/tmp/matplotlib"},
            check=False,
            capture_output=True,
            text=True,
            timeout=90,
        )


if __name__ == "__main__":
    unittest.main()
