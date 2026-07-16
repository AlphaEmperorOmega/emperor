from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import emperor_workbench.tensorboard as tensorboard_events
from emperor_workbench.run_history import _artifacts as run_artifacts
from emperor_workbench.run_history._artifacts import (
    RunArtifactBudgets,
    observe_run_artifacts,
)
from emperor_workbench.run_history._query import LogRunQueryService
from tests.support.training_jobs import (
    write_tensorboard_run,
)
from tests.unit.run_history._support import (
    FakeTensorBoardAccumulator,
    log_run_scanner,
)
from tests.unit.run_history._support import (
    run_history as _run_history,
)
from tests.unit.tensorboard._support import patch_event_accumulator_loader


class RunHistoryCatalogObservationTests(unittest.TestCase):
    def test_log_run_scanner_reuses_recent_run_parse_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logs_root = Path(tmp) / "logs"
            write_tensorboard_run(
                logs_root,
                ["linear", "BASELINE", "Mnist", "aaa_20260601_010203", "version_0"],
            )
            write_tensorboard_run(
                logs_root,
                ["linear", "BASELINE", "Mnist", "bbb_20260601_020304", "version_0"],
            )
            scanner = log_run_scanner(logs_root=logs_root, cache_ttl_seconds=60)

            with patch.object(scanner, "parse_run", wraps=scanner.parse_run) as parse:
                first = scanner.list_runs()
                second = scanner.list_runs()
                scanner.clear_cache()
                third = scanner.list_runs()

        self.assertEqual([run.id for run in first], [run.id for run in second])
        self.assertEqual([run.id for run in first], [run.id for run in third])
        self.assertEqual(parse.call_count, 4)

    def test_log_run_scanner_reuses_expired_cache_when_catalog_is_unchanged(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logs_root = Path(tmp) / "logs"
            run_dir = write_tensorboard_run(
                logs_root,
                ["linear", "BASELINE", "Mnist", "aaa_20260601_010203", "version_0"],
            )
            scanner = log_run_scanner(logs_root=logs_root, cache_ttl_seconds=0)

            with patch.object(scanner, "parse_run", wraps=scanner.parse_run) as parse:
                first = scanner.list_runs()
                second = scanner.list_runs()
                (run_dir / "result.json").write_text(
                    json.dumps({"metrics": {"accuracy": 0.95}}),
                    encoding="utf-8",
                )
                third = scanner.list_runs()

        self.assertEqual([run.id for run in first], [run.id for run in second])
        self.assertEqual([run.id for run in first], [run.id for run in third])
        self.assertEqual(parse.call_count, 2)

    def test_summary_listing_parses_result_once_without_projecting_metrics(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logs_root = Path(tmp) / "logs"
            run_dir = write_tensorboard_run(
                logs_root,
                ["linear", "BASELINE", "Mnist", "aaa_20260601_010203", "version_0"],
            )
            run_dir.joinpath("result.json").write_text(
                json.dumps(
                    {
                        "experimentTask": "image_classification",
                        "metrics": {"test/accuracy": 0.9},
                    }
                ),
                encoding="utf-8",
            )
            service = _run_history(logs_root)

            with (
                patch.object(
                    run_artifacts,
                    "_read_result_payload",
                    wraps=run_artifacts._read_result_payload,
                ) as read_result,
                patch.object(
                    run_artifacts.RunArtifactObservation,
                    "metrics",
                    side_effect=AssertionError(
                        "summary projection must not materialize metrics"
                    ),
                ) as project_metrics,
            ):
                run = service.list_runs(
                    limit=10,
                    offset=0,
                    projection="summary",
                ).runs[0]

        self.assertEqual(run.experiment_task, "image_classification")
        self.assertEqual(run.metrics, {})
        self.assertEqual(read_result.call_count, 1)
        project_metrics.assert_not_called()

    def test_run_artifact_observation_reuses_one_bounded_metadata_parse(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "version_0"
            run_dir.mkdir()
            run_dir.joinpath("result.json").write_text(
                json.dumps(
                    {
                        "experimentTask": "image_classification",
                        "metrics": {"accuracy": 0.9},
                        "params": {"batch_size": 8},
                    }
                ),
                encoding="utf-8",
            )
            observation = observe_run_artifacts(run_dir, root)

            with patch.object(
                run_artifacts,
                "_read_result_payload",
                wraps=run_artifacts._read_result_payload,
            ) as read_result:
                experiment_task = observation.experiment_task()
                metrics = observation.metrics()
                params = observation.params()

        self.assertEqual(experiment_task, "image_classification")
        self.assertEqual(metrics, {"accuracy": 0.9})
        self.assertEqual(params, {"batch_size": 8})
        read_result.assert_called_once()

    def test_incomplete_event_observation_refuses_tensorboard_loading(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "logs"
            run_dir = root / "experiment" / "run"
            run_dir.mkdir(parents=True)
            run_dir.joinpath("events.out.tfevents.safe").write_bytes(b"safe")
            omitted = run_dir / "z-omitted"
            omitted.mkdir()
            omitted.joinpath("events.out.tfevents.hidden").write_bytes(b"hidden")

            observation = observe_run_artifacts(
                run_dir,
                root,
                budgets=RunArtifactBudgets(
                    max_files=1,
                    max_depth=16,
                    max_metadata_file_bytes=1024,
                ),
            )
            with patch_event_accumulator_loader(
                return_value=FakeTensorBoardAccumulator(),
            ) as load:
                accumulator = observation.event_files.load_accumulator(run_dir)

        self.assertTrue(observation.truncated)
        self.assertFalse(observation.event_files.complete)
        self.assertIsNone(accumulator)
        load.assert_not_called()

    def test_event_file_replacement_after_observation_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "logs"
            run_dir = root / "experiment" / "run"
            run_dir.mkdir(parents=True)
            event_path = run_dir / "events.out.tfevents.safe"
            event_path.write_bytes(b"safe")
            observation = observe_run_artifacts(run_dir, root)
            outside = Path(tmp) / "events.out.tfevents.outside"
            outside.write_bytes(b"secret")
            event_path.unlink()
            event_path.symlink_to(outside)

            with patch_event_accumulator_loader(
                return_value=FakeTensorBoardAccumulator(),
            ) as load:
                accumulator = observation.event_files.load_accumulator(run_dir)

        self.assertIsNone(accumulator)
        load.assert_not_called()

    def test_run_artifact_observation_composes_shared_event_index(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "version_0"
            nested = run_dir / "nested"
            nested.mkdir(parents=True)
            event_file = nested / "events.out.tfevents.test"
            event_file.write_bytes(b"event")

            with patch.object(
                tensorboard_events,
                "event_file_index",
                wraps=tensorboard_events.event_file_index,
            ) as observe_events:
                observation = observe_run_artifacts(run_dir, root)

        observe_events.assert_called_once_with(
            run_dir,
            candidates=(event_file,),
            complete=True,
        )
        self.assertEqual(observation.event_files.files, (event_file,))
        self.assertEqual(
            tuple(artifact.path for artifact in observation.event_artifacts),
            (event_file,),
        )

    def test_run_history_tensorboard_query_reuses_catalog_event_index(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logs_root = Path(tmp) / "logs"
            run_dir = logs_root.joinpath(
                "linear",
                "BASELINE",
                "Mnist",
                "aaa_20260601_010203",
                "version_0",
            )
            run_dir.mkdir(parents=True)
            run_dir.joinpath("events.out.tfevents.test").write_bytes(b"event")
            scanner = log_run_scanner(logs_root=logs_root)
            query = LogRunQueryService(scanner=scanner)

            with (
                patch.object(
                    tensorboard_events,
                    "event_file_index",
                    wraps=tensorboard_events.event_file_index,
                ) as observe_events,
                patch_event_accumulator_loader(
                    return_value=FakeTensorBoardAccumulator(),
                ),
            ):
                run_id = scanner.list_runs(result_projection="none")[0].id
                tags = query.tags_for_runs([run_id])

        self.assertEqual(observe_events.call_count, 1)
        self.assertEqual(tags[0].scalar_tags, ("train/loss",))


if __name__ == "__main__":
    unittest.main()
