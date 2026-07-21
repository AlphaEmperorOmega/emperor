from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from emperor.experiments import ExperimentTask
from model_runtime.packages.identity import ModelIdentity
from model_runtime.runs.artifacts import (
    FilesystemRunArtifacts,
    result_metrics_payload,
    write_run_result,
)


class RunsArtifactsTests(unittest.TestCase):
    def test_relative_run_name_and_result_name_are_stable(self) -> None:
        artifacts = FilesystemRunArtifacts(
            namespace="runs_fixture",
            clock=lambda: datetime(2026, 6, 1, 1, 2, 3),
        )
        identity = ModelIdentity("linears", "linear")
        parameters = {
            "batch_size": 128,
            "input_dim": 784,
            "output_dim": 10,
        }

        run_name = artifacts.run_name(
            identity,
            "BASELINE",
            "Mnist",
            parameters,
        )

        self.assertEqual(
            run_name,
            "runs_fixture/linears/linear/BASELINE/Mnist/408b10c0_20260601_010203",
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = write_run_result(
                Path(tmp) / run_name / "version_0",
                {"metrics": {"validation_accuracy": 0.75}},
            )
            self.assertEqual(path.name, "result.json")
            self.assertEqual(
                json.loads(path.read_text(encoding="utf-8")),
                {"metrics": {"validation_accuracy": 0.75}},
            )

    def test_best_results_ranking_is_task_aware_and_atomic(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            artifacts = FilesystemRunArtifacts(root=Path(tmp), namespace="fixture")
            identity = ModelIdentity("gpt", "linear")
            high_loss = {
                "dataset": "WikiText2",
                "metrics": {"validation/loss": 2.0},
            }
            low_loss = {
                "dataset": "WikiText2",
                "metrics": {"validation/loss": 1.0},
            }

            artifacts.update_best_results(
                identity,
                ExperimentTask.CAUSAL_LANGUAGE_MODELING,
                high_loss,
            )
            merged = artifacts.update_best_results(
                identity,
                ExperimentTask.CAUSAL_LANGUAGE_MODELING,
                low_loss,
            )

            self.assertEqual(
                [run["metrics"]["validation/loss"] for run in merged["WikiText2"]],
                [1.0, 2.0],
            )
            self.assertEqual(
                [run["rank"] for run in merged["WikiText2"]],
                [1, 2],
            )
            self.assertTrue(artifacts.best_results_path(identity).is_file())

    def test_result_metrics_preserve_limits_and_drop_large_structures(self) -> None:
        payload = result_metrics_payload(
            {
                "validation/accuracy": 0.75,
                "validation/per_class": {"a": 1.0},
            }
        )

        self.assertEqual(payload["metrics"], {"validation/accuracy": 0.75})
        self.assertEqual(payload["metricsOriginalCount"], 2)
        self.assertEqual(payload["metricsDroppedCount"], 1)


if __name__ == "__main__":
    unittest.main()
