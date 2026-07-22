from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tests.support.training_jobs import (
    write_tensorboard_run,
)
from tests.unit.run_history._support import log_run_scanner
from tests.unit.run_history._support import (
    run_history as _run_history,
)


class RunHistoryCatalogListingTests(unittest.TestCase):
    def test_catalog_generation_exposes_external_run_before_cache_ttl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logs_root = Path(tmp) / "logs"
            service = _run_history(logs_root)
            self.assertEqual(service.list_runs(limit=10, offset=0).total, 0)

            write_tensorboard_run(
                logs_root,
                [
                    "experiment",
                    "linears",
                    "linear",
                    "BASELINE",
                    "Mnist",
                    "completed_20260601_010203",
                    "version_0",
                ],
            )

            refreshed = service.list_runs(limit=10, offset=0)

        self.assertEqual(refreshed.total, 1)
        self.assertEqual(refreshed.runs[0].experiment, "experiment")

    def test_run_history_parses_supported_log_shapes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logs_root = Path(tmp) / "logs"
            default_run = write_tensorboard_run(
                logs_root,
                [
                    "linears",
                    "linear",
                    "BASELINE",
                    "Mnist",
                    "aaa_20260601_010203",
                    "version_0",
                ],
                metrics={"test/accuracy": 0.9},
            )
            categorized_run = write_tensorboard_run(
                logs_root,
                [
                    "linears",
                    "linear",
                    "BASELINE",
                    "Mnist",
                    "categorized_20260601_010204",
                    "version_0",
                ],
                metrics={"test/accuracy": 0.91},
            )
            custom_run = write_tensorboard_run(
                logs_root,
                [
                    "test_model",
                    "linears",
                    "linear_adaptive",
                    "DUAL_MODEL_WEIGHT",
                    "Cifar10",
                    "bbb_20260601_020304",
                    "version_1",
                ],
                metrics=None,
                hparams=False,
                checkpoint=False,
            )
            workbench_run = write_tensorboard_run(
                logs_root,
                [
                    "workbench-training",
                    "job-123",
                    "linears",
                    "linear",
                    "BASELINE",
                    "FashionMNIST",
                    "ccc_20260601_030405",
                    "version_0",
                ],
                metrics={"validation/accuracy": 0.8},
            )
            no_event_run = logs_root.joinpath(
                "linears",
                "linear",
                "BASELINE",
                "Mnist",
                "ddd_20260601_040506",
                "version_2",
            )
            no_event_run.mkdir(parents=True)
            malformed_result_run = write_tensorboard_run(
                logs_root,
                [
                    "linears",
                    "linear",
                    "BASELINE",
                    "Mnist",
                    "eee_20260601_050607",
                    "version_0",
                ],
                metrics=None,
            )
            (malformed_result_run / "result.json").write_text(
                "{not valid json",
                encoding="utf-8",
            )
            outside_run = write_tensorboard_run(
                Path(tmp) / "outside",
                [
                    "linears",
                    "linear",
                    "BASELINE",
                    "Mnist",
                    "fff_20260601_060708",
                    "version_0",
                ],
                metrics={"test/accuracy": 1.0},
            )
            escaped_run_parent = logs_root.joinpath(
                "linears",
                "linear",
                "BASELINE",
                "Mnist",
                "escaped_20260601_070809",
            )
            escaped_run_parent.mkdir(parents=True)
            escaped_run = escaped_run_parent / "version_99"
            escaped_run.symlink_to(outside_run, target_is_directory=True)

            runs = log_run_scanner(logs_root=logs_root).list_runs()

        by_path = {run.relative_path: run for run in runs}
        default_summary = by_path[default_run.relative_to(logs_root).as_posix()]
        categorized_summary = by_path[categorized_run.relative_to(logs_root).as_posix()]
        custom_summary = by_path[custom_run.relative_to(logs_root).as_posix()]
        workbench_summary = by_path[workbench_run.relative_to(logs_root).as_posix()]
        no_event_summary = by_path[no_event_run.relative_to(logs_root).as_posix()]
        malformed_result_summary = by_path[
            malformed_result_run.relative_to(logs_root).as_posix()
        ]

        self.assertIsNone(default_summary.group)
        self.assertEqual(default_summary.experiment, "linears")
        self.assertEqual(default_summary.model, "linears/linear")
        self.assertEqual(default_summary.preset, "BASELINE")
        self.assertEqual(default_summary.dataset, "Mnist")
        self.assertEqual(default_summary.timestamp, "2026-06-01 01:02:03")
        self.assertTrue(default_summary.has_result)
        self.assertGreater(default_summary.event_file_count, 0)
        self.assertEqual(default_summary.checkpoint_count, 1)
        self.assertTrue(default_summary.has_hparams)
        self.assertEqual(default_summary.metrics["test/accuracy"], 0.9)

        self.assertIsNone(categorized_summary.group)
        self.assertEqual(categorized_summary.experiment, "linears")
        self.assertEqual(categorized_summary.model, "linears/linear")
        self.assertEqual(categorized_summary.preset, "BASELINE")
        self.assertEqual(categorized_summary.dataset, "Mnist")
        self.assertEqual(categorized_summary.metrics["test/accuracy"], 0.91)

        self.assertEqual(custom_summary.group, "test_model")
        self.assertEqual(custom_summary.experiment, "test_model")
        self.assertEqual(custom_summary.model, "linears/linear_adaptive")
        self.assertFalse(custom_summary.has_result)
        self.assertFalse(custom_summary.has_hparams)
        self.assertEqual(custom_summary.checkpoint_count, 0)

        self.assertEqual(workbench_summary.group, "workbench-training/job-123")
        self.assertEqual(workbench_summary.experiment, "workbench-training")
        self.assertEqual(workbench_summary.model, "linears/linear")
        self.assertEqual(workbench_summary.dataset, "FashionMNIST")

        self.assertFalse(no_event_summary.has_result)
        self.assertEqual(no_event_summary.event_file_count, 0)
        self.assertEqual(no_event_summary.checkpoint_count, 0)
        self.assertFalse(no_event_summary.has_hparams)
        self.assertEqual(no_event_summary.metrics, {})

        self.assertTrue(malformed_result_summary.has_result)
        self.assertGreater(malformed_result_summary.event_file_count, 0)
        self.assertEqual(malformed_result_summary.metrics, {})
        self.assertNotIn(
            "linears/linear/BASELINE/Mnist/escaped_20260601_070809/version_99",
            by_path,
        )

    def test_expired_catalog_detects_new_nested_event_and_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logs_root = Path(tmp) / "logs"
            run_dir = logs_root.joinpath(
                "linears",
                "linear",
                "BASELINE",
                "Mnist",
                "nested_20260601_010203",
                "version_0",
            )
            nested = run_dir / "already-present"
            nested.mkdir(parents=True)
            scanner = log_run_scanner(logs_root=logs_root, cache_ttl_seconds=0)

            initial = scanner.list_runs()[0]
            nested.joinpath("events.out.tfevents.new").write_bytes(b"event")
            nested.joinpath("epoch=1-step=2.ckpt").write_bytes(b"checkpoint")
            changed = scanner.list_runs()[0]

        self.assertEqual(initial.event_file_count, 0)
        self.assertEqual(initial.checkpoint_count, 0)
        self.assertEqual(changed.event_file_count, 1)
        self.assertEqual(changed.checkpoint_count, 1)

    def test_log_run_listing_filters_facets_by_experiment_task(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logs_root = Path(tmp) / "logs"
            for task, experiment in (
                ("image-classification", "image_exp"),
                ("language-modeling", "language_exp"),
            ):
                run_dir = write_tensorboard_run(
                    logs_root,
                    [
                        experiment,
                        "linears",
                        "linear",
                        "BASELINE",
                        "Mnist",
                        f"run_20260711_01010{len(experiment)}",
                        "version_0",
                    ],
                )
                (run_dir / "result.json").write_text(
                    json.dumps({"experimentTask": task, "metrics": {}}),
                    encoding="utf-8",
                )

            payload = _run_history(logs_root).list_runs(
                limit=10,
                offset=0,
                model=["linears/linear"],
                experiment_task="image-classification",
                projection="summary",
            )

            self.assertEqual(payload.total, 1)
            self.assertEqual(payload.runs[0].experiment, "image_exp")
            self.assertEqual(
                [facet.experiment for facet in payload.facets.experiments],
                ["image_exp"],
            )

    def test_run_history_lists_safe_top_level_experiments(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            logs_root = root / "logs"
            logs_root.mkdir()
            logs_root.joinpath("empty_experiment").mkdir()
            logs_root.joinpath("bad-name").mkdir()
            logs_root.joinpath("_bad_name").mkdir()
            outside_experiment = root / "outside_experiment"
            outside_experiment.mkdir()
            logs_root.joinpath("linked_experiment").symlink_to(
                outside_experiment,
                target_is_directory=True,
            )
            write_tensorboard_run(
                logs_root,
                [
                    "test_model",
                    "linears",
                    "linear",
                    "BASELINE",
                    "Mnist",
                    "aaa_20260601_010203",
                    "version_0",
                ],
            )
            write_tensorboard_run(
                logs_root,
                [
                    "workbench-training",
                    "job-123",
                    "linears",
                    "linear",
                    "BASELINE",
                    "Mnist",
                    "bbb_20260601_020304",
                    "version_0",
                ],
            )

            experiments = log_run_scanner(logs_root=logs_root).list_experiments()

        self.assertEqual(
            [experiment.experiment for experiment in experiments],
            ["empty_experiment", "test_model"],
        )
        by_name = {experiment.experiment: experiment for experiment in experiments}
        self.assertEqual(by_name["empty_experiment"].run_count, 0)
        self.assertEqual(by_name["test_model"].run_count, 1)
        self.assertEqual(by_name["test_model"].relative_path, "test_model")

    def test_qualified_model_filter_does_not_match_a_shared_leaf_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logs_root = Path(tmp) / "logs"
            for model_type in ("linears", "experts"):
                write_tensorboard_run(
                    logs_root,
                    [
                        "experiment",
                        model_type,
                        "linear",
                        "BASELINE",
                        "Mnist",
                        f"{model_type}_20260601_010203",
                        "version_0",
                    ],
                )
            service = _run_history(logs_root)

            qualified = service.list_runs(
                limit=10,
                offset=0,
                model=["linears/linear"],
            )
            flat_identity = service.list_runs(
                limit=10,
                offset=0,
                model=["linear"],
            )

        self.assertEqual(qualified.total, 1)
        self.assertEqual(qualified.runs[0].model, "linears/linear")
        self.assertEqual(flat_identity.total, 0)

    def test_scanner_rejects_flat_legacy_model_layout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logs_root = Path(tmp) / "logs"
            write_tensorboard_run(
                logs_root,
                [
                    "linear",
                    "BASELINE",
                    "Mnist",
                    "legacy_20260601_010203",
                    "version_0",
                ],
            )

            runs = log_run_scanner(logs_root=logs_root).list_runs()

        self.assertEqual(runs, [])


if __name__ == "__main__":
    unittest.main()
