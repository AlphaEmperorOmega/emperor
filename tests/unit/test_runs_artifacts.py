from __future__ import annotations

import json
import math
import multiprocessing
import tempfile
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import patch

from filelock import FileLock

from emperor.experiments import ExperimentTask
from model_runtime.packages import ModelIdentity
from model_runtime.runs.artifacts import (
    DEFAULT_RESULT_METRIC_KEY_LIMIT,
    FilesystemRunArtifacts,
    RunArtifacts,
)
from model_runtime.task_behavior import (
    CORE_RESULT_METRIC_KEYS,
    EXPERIMENT_TASK_BEHAVIORS,
)

if TYPE_CHECKING:
    from multiprocessing.queues import Queue
    from multiprocessing.synchronize import Barrier


def _reserve_run_artifacts_in_process(
    artifact_root: str,
    start: Barrier,
    results: Queue[tuple[str, int, str]],
) -> None:
    artifacts = FilesystemRunArtifacts(
        root=Path(artifact_root),
        namespace="runs_fixture",
        clock=lambda: datetime(2026, 6, 1, 1, 2, 3),
    )
    start.wait()
    allocation = artifacts.reserve_run(
        ModelIdentity("linears", "linear"),
        "BASELINE",
        "Mnist",
        {"batch_size": 128},
    )
    results.put((allocation.name, allocation.version, str(allocation.log_dir)))


class RunsArtifactsTests(unittest.TestCase):
    def test_filesystem_allocations_are_atomic_across_spawned_processes(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            context = multiprocessing.get_context("spawn")
            process_count = 4
            start = context.Barrier(process_count)
            results = context.Queue()
            processes = [
                context.Process(
                    target=_reserve_run_artifacts_in_process,
                    args=(tmp, start, results),
                )
                for _ in range(process_count)
            ]

            try:
                for process in processes:
                    process.start()
                for process in processes:
                    process.join(timeout=20)
                for process in processes:
                    if process.is_alive():
                        process.terminate()
                        process.join(timeout=5)

                self.assertEqual(
                    [process.exitcode for process in processes],
                    [0] * process_count,
                )
                allocations = [results.get(timeout=5) for _ in processes]
            finally:
                results.close()
                results.join_thread()

            names = {name for name, _, _ in allocations}
            versions = {version for _, version, _ in allocations}
            log_dirs = {Path(log_dir) for _, _, log_dir in allocations}
            self.assertEqual(
                names,
                {"runs_fixture/linears/linear/BASELINE/Mnist/e10f8fea_20260601_010203"},
            )
            self.assertEqual(versions, {0, 1, 2, 3})
            self.assertEqual(len(log_dirs), process_count)
            for log_dir in log_dirs:
                self.assertTrue(log_dir.is_dir())
                self.assertTrue(log_dir.resolve().is_relative_to(Path(tmp).resolve()))

    def test_filesystem_reservation_never_reuses_an_existing_attempt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            artifacts = FilesystemRunArtifacts(
                root=Path(tmp),
                clock=lambda: datetime(2026, 6, 1, 1, 2, 3),
            )
            identity = ModelIdentity("linears", "linear")
            run_name = artifacts.run_name(identity, "BASELINE", "Mnist", {})
            abandoned_attempt = artifacts.root / run_name / "version_0"
            abandoned_attempt.mkdir(parents=True)

            reservation = artifacts.reserve_run(
                identity,
                "BASELINE",
                "Mnist",
                {},
            )

            self.assertEqual(reservation.name, run_name)
            self.assertEqual(reservation.version, 1)
            self.assertEqual(
                reservation.log_dir, (abandoned_attempt.parent / "version_1").resolve()
            )
            self.assertTrue(abandoned_attempt.is_dir())
            self.assertTrue(reservation.log_dir.is_dir())

    def test_artifact_paths_require_typed_model_identity(self) -> None:
        artifacts = FilesystemRunArtifacts(root=Path("/safe/root"))

        for raw_identity in (
            "linears/linear",
            "/tmp/escape",
            "../../escape",
            "linears\\linear",
            " linears/linear ",
        ):
            with (
                self.subTest(identity=raw_identity),
                self.assertRaisesRegex(
                    TypeError,
                    "ModelIdentity",
                ),
            ):
                artifacts.best_results_path(raw_identity)

    def test_artifact_identity_ignores_overridden_catalog_key(self) -> None:
        class HostileIdentity(ModelIdentity):
            @property
            def catalog_key(self) -> str:
                return "../../escape"

        artifacts = FilesystemRunArtifacts()
        identity = HostileIdentity("linears", "linear")

        self.assertTrue(
            artifacts.run_name(identity, "baseline", "Mnist", {}).startswith(
                "linears/linear/baseline/Mnist/"
            )
        )

    def test_run_name_rejects_path_like_preset_and_dataset_segments(self) -> None:
        artifacts = FilesystemRunArtifacts()
        identity = ModelIdentity("linears", "linear")

        for field_name, preset, dataset in (
            ("preset_key", "../escape", "Mnist"),
            ("preset_key", "/tmp", "Mnist"),
            ("preset_key", "nested/folder", "Mnist"),
            ("preset_key", "nested\\folder", "Mnist"),
            ("dataset", "baseline", "../escape"),
            ("dataset", "baseline", "/tmp"),
            ("dataset", "baseline", "nested/folder"),
            ("dataset", "baseline", " nested "),
        ):
            with self.subTest(field=field_name, value=(preset, dataset)):
                with self.assertRaisesRegex(ValueError, field_name):
                    artifacts.run_name(identity, preset, dataset, {})

    def test_result_write_rejects_log_directory_outside_artifact_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "logs"
            outside = Path(tmp) / "outside"
            artifacts = FilesystemRunArtifacts(root=root)

            with self.assertRaisesRegex(ValueError, "outside artifact root"):
                artifacts.write_result(outside, {"status": "escaped"})

            self.assertFalse(outside.exists())

    def test_model_root_rejects_existing_symlink_escape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "logs"
            outside = Path(tmp) / "outside"
            root.mkdir()
            outside.mkdir()
            (root / "linears").symlink_to(outside, target_is_directory=True)
            artifacts = FilesystemRunArtifacts(root=root)

            with self.assertRaisesRegex(ValueError, "outside artifact root"):
                artifacts.best_results_path(ModelIdentity("linears", "linear"))

            self.assertEqual(list(outside.iterdir()), [])

    def test_final_artifact_file_symlinks_cannot_escape_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "logs"
            model_root = root / "linears" / "linear"
            log_dir = root / "run" / "version_0"
            model_root.mkdir(parents=True)
            log_dir.mkdir(parents=True)
            outside_summary = Path(tmp) / "outside-summary.json"
            outside_result = Path(tmp) / "outside-result.json"
            outside_summary.write_text('{"secret": true}', encoding="utf-8")
            outside_result.write_text('{"status": "unchanged"}', encoding="utf-8")
            (model_root / "best_results.json").symlink_to(outside_summary)
            (log_dir / "result.json").symlink_to(outside_result)
            artifacts = FilesystemRunArtifacts(root=root)
            identity = ModelIdentity("linears", "linear")

            with self.assertRaisesRegex(ValueError, "outside artifact root"):
                artifacts.read_best_results(identity)
            with self.assertRaisesRegex(ValueError, "outside artifact root"):
                artifacts.write_result(log_dir, {"status": "escaped"})

            self.assertEqual(
                json.loads(outside_summary.read_text(encoding="utf-8")),
                {"secret": True},
            )
            self.assertEqual(
                json.loads(outside_result.read_text(encoding="utf-8")),
                {"status": "unchanged"},
            )

    def test_final_lock_file_symlink_cannot_escape_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "logs"
            model_root = root / "linears" / "linear"
            model_root.mkdir(parents=True)
            outside_lock = Path(tmp) / "outside.lock"
            outside_lock.write_text("unchanged", encoding="utf-8")
            (model_root / "best_results.json.lock").symlink_to(outside_lock)
            artifacts = FilesystemRunArtifacts(root=root)

            with self.assertRaisesRegex(ValueError, "outside artifact root"):
                artifacts.update_best_results(
                    ModelIdentity("linears", "linear"),
                    None,
                    {"dataset": "Mnist", "metrics": {}},
                )

            self.assertEqual(outside_lock.read_text(encoding="utf-8"), "unchanged")
            self.assertFalse((model_root / "best_results.json").exists())

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

    def test_best_result_replay_is_idempotent_per_artifact_attempt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            artifacts = FilesystemRunArtifacts(root=Path(tmp))
            identity = ModelIdentity("linears", "linear")
            result = {
                "status": "completed",
                "artifactId": "attempt-a",
                "runId": "run-0001",
                "dataset": "Mnist",
                "params": {},
                "metrics": {"validation_accuracy": 0.8},
            }

            artifacts.update_best_results(
                identity,
                ExperimentTask.IMAGE_CLASSIFICATION,
                result,
            )
            replayed = artifacts.update_best_results(
                identity,
                ExperimentTask.IMAGE_CLASSIFICATION,
                {**result, "rank": 99},
            )
            with self.assertRaisesRegex(ValueError, "artifactId.*attempt-a"):
                artifacts.update_best_results(
                    identity,
                    ExperimentTask.IMAGE_CLASSIFICATION,
                    {**result, "dataset": "FashionMNIST"},
                )
            distinct_attempt = artifacts.update_best_results(
                identity,
                ExperimentTask.IMAGE_CLASSIFICATION,
                {**result, "artifactId": "attempt-b"},
            )

            self.assertEqual(len(replayed["Mnist"]), 1)
            self.assertNotIn("FashionMNIST", replayed)
            self.assertEqual(
                [entry["artifactId"] for entry in distinct_attempt["Mnist"]],
                ["attempt-a", "attempt-b"],
            )
            with self.assertRaisesRegex(ValueError, "artifactId.*attempt-a"):
                artifacts.update_best_results(
                    identity,
                    ExperimentTask.IMAGE_CLASSIFICATION,
                    {
                        **result,
                        "metrics": {"validation_accuracy": 0.1},
                    },
                )

    def test_best_results_lock_timeout_is_finite_and_recovers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            artifacts = FilesystemRunArtifacts(
                Path(tmp),
                None,
                datetime.now,
                best_results_lock_timeout_seconds=0.02,
            )
            identity = ModelIdentity("linears", "linear")
            summary_path = artifacts.best_results_path(identity)
            lock_path = summary_path.with_suffix(summary_path.suffix + ".lock")
            lock_path.parent.mkdir(parents=True)
            held_lock = FileLock(str(lock_path))
            held_lock.acquire()
            started = time.monotonic()
            try:
                with self.assertRaisesRegex(TimeoutError, "best-results lock"):
                    artifacts.update_best_results(
                        identity,
                        None,
                        {"dataset": "Mnist", "metrics": {}},
                    )
            finally:
                held_lock.release()

            self.assertLess(time.monotonic() - started, 1.0)
            merged = artifacts.update_best_results(
                identity,
                None,
                {"dataset": "Mnist", "metrics": {}},
            )
            self.assertEqual(len(merged["Mnist"]), 1)

    def test_best_results_lock_timeout_policy_is_explicit_and_validated(self) -> None:
        default = FilesystemRunArtifacts()
        explicit = FilesystemRunArtifacts(best_results_lock_timeout_seconds=2.5)

        self.assertEqual(default.best_results_lock_timeout_seconds, 30.0)
        self.assertIn("best_results_lock_timeout_seconds=30.0", repr(default))
        self.assertNotEqual(default, explicit)
        with self.assertRaises(TypeError):
            FilesystemRunArtifacts(Path("logs"), None, datetime.now, 2.5)
        for value in (True, 0, -1, math.inf, math.nan, "1"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                FilesystemRunArtifacts(best_results_lock_timeout_seconds=value)

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

    def test_result_metrics_keep_core_metrics_after_optional_metric_limit(self) -> None:
        optional_metrics = {
            f"adaptive/module_{index:04d}/mean": index
            for index in range(DEFAULT_RESULT_METRIC_KEY_LIMIT + 200)
        }

        payload = FilesystemRunArtifacts().result_metrics_payload(
            {
                **optional_metrics,
                "validation/perplexity": 42.0,
            }
        )

        self.assertEqual(payload["metrics"]["validation/perplexity"], 42.0)

    def test_core_result_metrics_are_independent_of_insertion_order(self) -> None:
        optional_metrics = {
            f"monitor/module_{index:04d}/mean": index
            for index in range(DEFAULT_RESULT_METRIC_KEY_LIMIT + 1)
        }
        core_metrics = {
            "validation/loss": 3.5,
            "validation/perplexity": 33.1,
        }

        core_first = FilesystemRunArtifacts().result_metrics_payload(
            {**core_metrics, **optional_metrics}
        )
        core_last = FilesystemRunArtifacts().result_metrics_payload(
            {**optional_metrics, **core_metrics}
        )

        for payload in (core_first, core_last):
            self.assertEqual(
                {key: payload["metrics"][key] for key in core_metrics},
                core_metrics,
            )

    def test_result_metric_limit_applies_only_to_optional_metrics(self) -> None:
        metrics = {
            **{
                f"monitor/module_{index:04d}/mean": index
                for index in range(DEFAULT_RESULT_METRIC_KEY_LIMIT + 100)
            },
            "validation/loss": 3.5,
            "validation/perplexity": 33.1,
        }

        payload = FilesystemRunArtifacts().result_metrics_payload(metrics)

        self.assertEqual(
            len(payload["metrics"]),
            DEFAULT_RESULT_METRIC_KEY_LIMIT + 2,
        )
        self.assertLessEqual(
            len(payload["metrics"]),
            DEFAULT_RESULT_METRIC_KEY_LIMIT + len(CORE_RESULT_METRIC_KEYS),
        )
        self.assertEqual(payload["metricsOriginalCount"], len(metrics))
        self.assertEqual(payload["metricsDroppedCount"], 100)

    def test_truncated_optional_metric_selection_and_order_are_deterministic(
        self,
    ) -> None:
        optional_keys = [
            f"monitor/module_{index:04d}/mean"
            for index in range(DEFAULT_RESULT_METRIC_KEY_LIMIT + 100)
        ]
        core_metrics = {"validation/loss": 3.5}
        ascending = FilesystemRunArtifacts().result_metrics_payload(
            {
                **{key: key for key in optional_keys},
                **core_metrics,
            }
        )
        descending = FilesystemRunArtifacts().result_metrics_payload(
            {
                **{key: key for key in reversed(optional_keys)},
                **core_metrics,
            }
        )

        expected_optional_keys = optional_keys[:DEFAULT_RESULT_METRIC_KEY_LIMIT]
        self.assertEqual(
            [key for key in ascending["metrics"] if key not in core_metrics],
            expected_optional_keys,
        )
        self.assertEqual(
            list(ascending["metrics"]),
            list(descending["metrics"]),
        )
        self.assertEqual(ascending, descending)

    def test_low_cardinality_result_metrics_remain_unchanged(self) -> None:
        metrics = {
            "monitor/z_metric": 1.0,
            "validation/perplexity": 42.0,
            "monitor/a_metric": 2.0,
        }

        payload = FilesystemRunArtifacts().result_metrics_payload(metrics)

        self.assertEqual(payload, {"metrics": metrics})
        self.assertEqual(list(payload["metrics"]), list(metrics))

    def test_every_task_core_result_contract_survives_optional_metric_pressure(
        self,
    ) -> None:
        optional_metrics = {
            f"monitor/module_{index:04d}/mean": index
            for index in range(DEFAULT_RESULT_METRIC_KEY_LIMIT + 1)
        }

        for task, behavior in EXPERIMENT_TASK_BEHAVIORS.items():
            ranking_keys = {
                key
                for ranking_metric in behavior.ranking_metrics
                for key in ranking_metric.keys
            }
            expected_core_keys = {
                *behavior.core_result_metric_keys,
                *ranking_keys,
            }
            with self.subTest(task=task.name):
                payload = FilesystemRunArtifacts().result_metrics_payload(
                    {
                        **optional_metrics,
                        **{key: 1.0 for key in expected_core_keys},
                    }
                )

                self.assertTrue(
                    expected_core_keys.issubset(payload["metrics"]),
                    expected_core_keys.difference(payload["metrics"]),
                )

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
