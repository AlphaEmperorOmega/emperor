from __future__ import annotations

import json
import math
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

from emperor.experiments import ExperimentTask
from model_runtime.packages import ModelIdentity
from model_runtime.runs.artifacts import FilesystemRunArtifacts, RunArtifacts


class RunsArtifactsTests(unittest.TestCase):
    def test_relative_run_name_and_result_name_are_stable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            artifacts = FilesystemRunArtifacts(
                root=Path(tmp) / "logs",
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
            path = artifacts.write_result(
                artifacts.root / run_name / "version_0",
                {"metrics": {"validation_accuracy": 0.75}},
            )
            self.assertEqual(path.name, "result.json")
            self.assertEqual(
                json.loads(path.read_text(encoding="utf-8")),
                {"metrics": {"validation_accuracy": 0.75}},
            )

    def test_concurrent_best_result_updates_are_locked_and_keep_top_five(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            artifacts = FilesystemRunArtifacts(root=Path(tmp), namespace="fixture")
            identity = ModelIdentity("linears", "linear")

            def update(score: int) -> None:
                artifacts.update_best_results(
                    identity,
                    ExperimentTask.IMAGE_CLASSIFICATION,
                    {
                        "dataset": "Mnist",
                        "params": {"score": score},
                        "metrics": {"validation_accuracy": score},
                    },
                )

            with ThreadPoolExecutor(max_workers=8) as executor:
                list(executor.map(update, range(12)))

            merged = artifacts.read_best_results(identity)
            self.assertEqual(
                [run["metrics"]["validation_accuracy"] for run in merged["Mnist"]],
                [11, 10, 9, 8, 7],
            )
            self.assertEqual(
                [run["rank"] for run in merged["Mnist"]],
                [1, 2, 3, 4, 5],
            )
            summary_path = artifacts.best_results_path(identity)
            self.assertEqual(
                json.loads(summary_path.read_text(encoding="utf-8")),
                merged,
            )
            self.assertEqual(list(summary_path.parent.glob(".*.tmp")), [])

    def test_language_model_results_rank_lowest_validation_loss_first(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            artifacts = FilesystemRunArtifacts(root=Path(tmp))
            identity = ModelIdentity("gpt", "linear")
            for loss in (2.0, 1.0):
                artifacts.update_best_results(
                    identity,
                    ExperimentTask.CAUSAL_LANGUAGE_MODELING,
                    {
                        "dataset": "WikiText2",
                        "metrics": {"validation/loss": loss},
                    },
                )

            merged = artifacts.read_best_results(identity)
            self.assertEqual(
                [run["metrics"]["validation/loss"] for run in merged["WikiText2"]],
                [1.0, 2.0],
            )

    def test_update_merges_existing_datasets_before_replacing_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            artifacts = FilesystemRunArtifacts(root=Path(tmp))
            identity = ModelIdentity("linears", "linear")
            for dataset, score in (("Mnist", 0.8), ("FashionMnist", 0.7)):
                artifacts.update_best_results(
                    identity,
                    ExperimentTask.IMAGE_CLASSIFICATION,
                    {
                        "dataset": dataset,
                        "params": {"score": score},
                        "metrics": {"validation_accuracy": score},
                    },
                )

            merged = artifacts.update_best_results(
                identity,
                ExperimentTask.IMAGE_CLASSIFICATION,
                {
                    "dataset": "Mnist",
                    "params": {"score": 0.9},
                    "metrics": {"validation_accuracy": 0.9},
                },
            )

            self.assertEqual(
                [run["params"]["score"] for run in merged["Mnist"]],
                [0.9, 0.8],
            )
            self.assertEqual(merged["FashionMnist"][0]["params"]["score"], 0.7)

    def test_result_metrics_preserve_limits_and_drop_large_structures(self) -> None:
        payload = FilesystemRunArtifacts().result_metrics_payload(
            {
                "validation/accuracy": 0.75,
                "validation/per_class": {"a": 1.0},
            }
        )

        self.assertEqual(payload["metrics"], {"validation/accuracy": 0.75})
        self.assertEqual(payload["metricsOriginalCount"], 2)
        self.assertEqual(payload["metricsDroppedCount"], 1)

    def test_namespace_must_be_one_relative_folder(self) -> None:
        for namespace in ("", ".", "..", "../escape", "nested/folder", "/logs", "a\\b"):
            with self.subTest(namespace=namespace), self.assertRaises(ValueError):
                FilesystemRunArtifacts(namespace=namespace)

    def test_failed_atomic_replacement_preserves_existing_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            artifacts = FilesystemRunArtifacts(root=Path(tmp))
            run_dir = Path(tmp) / "run" / "version_0"
            result_path = artifacts.write_result(run_dir, {"status": "original"})

            with (
                patch(
                    "model_runtime.runs.artifacts.os.replace",
                    side_effect=OSError("replace failed"),
                ),
                self.assertRaisesRegex(OSError, "replace failed"),
            ):
                artifacts.write_result(run_dir, {"status": "replacement"})

            self.assertEqual(
                json.loads(result_path.read_text(encoding="utf-8")),
                {"status": "original"},
            )
            self.assertEqual(list(run_dir.glob(".result.json.*.tmp")), [])

    def test_non_finite_results_are_rejected_before_persistence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            artifacts = FilesystemRunArtifacts(root=Path(tmp))
            with self.assertRaises(ValueError):
                artifacts.write_result(
                    Path(tmp) / "run",
                    {"metrics": {"loss": math.nan}},
                )
            self.assertFalse((Path(tmp) / "run" / "result.json").exists())

    def test_filesystem_implementation_satisfies_run_artifact_interface(self) -> None:
        self.assertIsInstance(FilesystemRunArtifacts(), RunArtifacts)


if __name__ == "__main__":
    unittest.main()
