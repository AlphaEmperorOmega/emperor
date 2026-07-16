from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import emperor_workbench.tensorboard as tensorboard_events
from emperor_workbench.failures import FailureKind
from emperor_workbench.run_history import (
    RunHistoryFailure,
)
from emperor_workbench.run_history._query import LogRunQueryService
from emperor_workbench.tensorboard import ScalarPoint, ScalarTail
from tests.unit.run_history._support import (
    FakeTensorBoardAccumulator,
    log_run_scanner,
)
from tests.unit.tensorboard._support import patch_event_accumulator_loader


def write_fake_event_run(
    logs_root: Path,
    run_name: str,
    payload: bytes,
) -> Path:
    run_dir = logs_root.joinpath(
        "linear",
        "BASELINE",
        "Mnist",
        run_name,
        "version_0",
    )
    run_dir.mkdir(parents=True)
    (run_dir / "events.out.tfevents.fake").write_bytes(payload)
    return run_dir


class RunHistoryTensorBoardQueryTests(unittest.TestCase):
    def test_run_history_reads_tensorboard_media_summaries(self) -> None:
        import torch
        from torch.utils.tensorboard import SummaryWriter

        with tempfile.TemporaryDirectory() as tmp:
            logs_root = Path(tmp) / "logs"
            run_dir = logs_root.joinpath(
                "linear",
                "BASELINE",
                "Cifar10",
                "diagnostics_20260613_120000",
                "version_0",
            )
            writer = SummaryWriter(log_dir=str(run_dir))
            writer.add_scalar("gap/accuracy", 0.12, 1)
            writer.add_image(
                "validation/examples/predictions",
                torch.zeros(3, 8, 8),
                1,
            )
            writer.add_text(
                "validation/examples/predictions",
                "true=cat predicted=dog confidence=0.91",
                1,
            )
            writer.flush()
            writer.close()

            scanner = log_run_scanner(logs_root=logs_root)
            query = LogRunQueryService(scanner=scanner)
            run = scanner.list_runs()[0]
            tags = query.tags_for_runs([run.id])[0]
            media = query.media_for_runs(
                run_ids=[run.id],
                image_tags=["validation/examples/predictions"],
                text_tags=["validation/examples/predictions/text_summary"],
            )

        self.assertEqual(tags.scalar_tags, ("gap/accuracy",))
        self.assertEqual(tags.image_tags, ("validation/examples/predictions",))
        self.assertEqual(
            tags.text_tags,
            ("validation/examples/predictions/text_summary",),
        )
        self.assertEqual(media.images[0].run_id, run.id)
        self.assertTrue(media.images[0].data_url.startswith("data:image/png;base64,"))
        self.assertEqual(media.texts[0].run_id, run.id)
        self.assertIn("true=cat", media.texts[0].text)

    def test_log_run_query_service_reads_latest_summary_across_event_dirs(self) -> None:
        from torch.utils.tensorboard import SummaryWriter

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            early_writer = SummaryWriter(log_dir=str(run_dir / "early"))
            early_writer.add_text("validation/examples/predictions", "early", 1)
            early_writer.flush()
            early_writer.close()
            late_writer = SummaryWriter(log_dir=str(run_dir / "late"))
            late_writer.add_text("validation/examples/predictions", "late", 2)
            late_writer.flush()
            late_writer.close()

            query = LogRunQueryService(scanner=log_run_scanner(logs_root=Path(tmp)))
            summary = query.read_text_summary(
                run_dir,
                "validation/examples/predictions/text_summary",
            )

        self.assertIsNotNone(summary)
        assert summary is not None
        self.assertEqual(summary.step, 2)
        self.assertEqual(summary.text, "late")

    def test_log_run_query_service_streams_exact_bounded_scalar_tails(self) -> None:
        from torch.utils.tensorboard import SummaryWriter

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for point_count in (501, 1_000, 2_000):
                with self.subTest(point_count=point_count):
                    run_dir = root / f"run-{point_count}"
                    writer = SummaryWriter(log_dir=str(run_dir))
                    for step in range(point_count):
                        writer.add_scalar("train/loss", float(step), step)
                    writer.close()
                    query = LogRunQueryService(scanner=log_run_scanner(logs_root=root))

                    series = query.read_scalar_series(
                        run_dir,
                        "train/loss",
                        max_points=500,
                    )

                    self.assertEqual(series.source_point_count, point_count)
                    self.assertTrue(series.truncated)
                    self.assertEqual(len(series.points), 500)
                    self.assertEqual(
                        [point.step for point in series.points],
                        list(range(point_count - 500, point_count)),
                    )

    def test_log_run_query_service_caches_tags_and_scalar_tails_until_event_changes(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            run_dir.mkdir()
            event_file = run_dir / "events.out.tfevents.cache"
            event_file.write_text("first", encoding="utf-8")
            service = LogRunQueryService(
                scanner=log_run_scanner(logs_root=Path(tmp)),
            )
            load_calls: list[Path] = []

            def load_accumulator(
                event_dir: Path,
                **_kwargs,
            ) -> FakeTensorBoardAccumulator:
                load_calls.append(event_dir)
                return FakeTensorBoardAccumulator()

            scalar_tail = {
                "train/loss": ScalarTail(
                    points=(
                        ScalarPoint(step=2, wall_time=2.0, value=0.25),
                        ScalarPoint(step=3, wall_time=3.0, value=0.125),
                    ),
                    source_point_count=3,
                    truncated=True,
                )
            }

            with (
                patch_event_accumulator_loader(load_accumulator),
                patch.object(
                    tensorboard_events,
                    "exact_scalar_tails",
                    return_value=scalar_tail,
                ) as stream_tails,
            ):
                first_tags = service.read_tags(run_dir)
                second_tags = service.read_tags(run_dir)
                first_scalars = service.read_scalar_series(
                    run_dir,
                    "train/loss",
                    max_points=2,
                )
                second_scalars = service.read_scalar_series(
                    run_dir,
                    "train/loss",
                    max_points=2,
                )
                event_file.write_text("first-second", encoding="utf-8")
                changed_tags = service.read_tags(run_dir)

        self.assertEqual(first_tags, second_tags)
        self.assertEqual(first_tags, changed_tags)
        self.assertEqual(first_scalars, second_scalars)
        self.assertEqual(first_scalars.source_point_count, 3)
        self.assertTrue(first_scalars.truncated)
        self.assertEqual(
            [point.step for point in first_scalars.points],
            [2, 3],
        )
        self.assertEqual(len(load_calls), 2)
        stream_tails.assert_called_once()

    def test_log_run_query_service_streams_each_scalar_tag_batch_once(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            run_dir.mkdir()
            (run_dir / "events.out.tfevents.cache").write_text(
                "events",
                encoding="utf-8",
            )
            service = LogRunQueryService(
                scanner=log_run_scanner(logs_root=Path(tmp)),
            )

            streamed = {
                tag: ScalarTail(
                    points=(
                        ScalarPoint(
                            step=1,
                            wall_time=1.0,
                            value=0.5 if tag == "loss" else 0.8,
                        ),
                    ),
                    source_point_count=1,
                    truncated=False,
                )
                for tag in ("loss", "accuracy")
            }
            with patch.object(
                tensorboard_events,
                "exact_scalar_tails",
                return_value=streamed,
            ) as stream_tails:
                result = service.read_scalar_series_batch(
                    run_dir,
                    ["loss", "accuracy"],
                    max_points=2,
                )

        stream_tails.assert_called_once()
        self.assertEqual(result["loss"].points[0].value, 0.5)
        self.assertEqual(result["accuracy"].points[0].value, 0.8)

    def test_log_run_query_service_skips_tag_scan_for_oversized_event_files(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            run_dir.mkdir()
            event_file = run_dir / "events.out.tfevents.large"
            event_file.write_text("large-event-payload", encoding="utf-8")
            service = LogRunQueryService(
                scanner=log_run_scanner(logs_root=Path(tmp)),
                max_tag_event_bytes=4,
            )

            with patch_event_accumulator_loader() as load:
                tags = service.read_tags(run_dir)

        self.assertEqual(tags.scalar_tags, ())
        self.assertEqual(tags.histogram_tags, ())
        self.assertEqual(tags.image_tags, ())
        self.assertEqual(tags.text_tags, ())
        self.assertEqual(tags.event_bytes, len("large-event-payload"))
        self.assertEqual(tags.skipped_event_files, 1)
        self.assertEqual(tags.source_item_count, 1)
        self.assertEqual(tags.returned_item_count, 0)
        self.assertTrue(tags.truncated)
        self.assertIn("event files skipped", tags.truncation_reason or "")
        load.assert_not_called()

    def test_log_run_query_service_rejects_uncached_tags_past_batch_budget(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logs_root = Path(tmp)
            write_fake_event_run(
                logs_root,
                "first_20260601_010203",
                b"1111",
            )
            write_fake_event_run(
                logs_root,
                "second_20260601_020304",
                b"2222",
            )
            write_fake_event_run(
                logs_root,
                "third_20260601_030405",
                b"3333",
            )
            scanner = log_run_scanner(logs_root=logs_root)
            run_ids = {run.run_name: run.id for run in scanner.list_runs()}
            service = LogRunQueryService(
                scanner=scanner,
                max_tag_event_bytes=100,
                max_tag_batch_event_bytes=8,
            )
            loaded_event_dirs: list[Path] = []

            def load_accumulator(
                event_dir: Path,
                **_kwargs,
            ) -> FakeTensorBoardAccumulator:
                loaded_event_dirs.append(event_dir)
                return FakeTensorBoardAccumulator()

            with patch_event_accumulator_loader(load_accumulator):
                with self.assertRaises(RunHistoryFailure) as raised:
                    service.tags_for_runs(
                        [
                            run_ids["first_20260601_010203"],
                            run_ids["second_20260601_020304"],
                            run_ids["third_20260601_030405"],
                        ]
                    )

        self.assertEqual(len(loaded_event_dirs), 2)
        self.assertEqual(raised.exception.kind, FailureKind.TOO_LARGE)
        self.assertIn("8 byte read budget", raised.exception.detail)

    def test_log_run_query_service_does_not_cache_batch_budget_skips(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logs_root = Path(tmp)
            write_fake_event_run(
                logs_root,
                "first_20260601_010203",
                b"1111",
            )
            write_fake_event_run(
                logs_root,
                "second_20260601_020304",
                b"2222",
            )
            write_fake_event_run(
                logs_root,
                "third_20260601_030405",
                b"3333",
            )
            scanner = log_run_scanner(logs_root=logs_root)
            run_ids = {run.run_name: run.id for run in scanner.list_runs()}
            service = LogRunQueryService(
                scanner=scanner,
                max_tag_event_bytes=100,
                max_tag_batch_event_bytes=8,
            )

            with patch_event_accumulator_loader(
                return_value=FakeTensorBoardAccumulator(),
            ) as load:
                with self.assertRaises(RunHistoryFailure):
                    service.tags_for_runs(
                        [
                            run_ids["first_20260601_010203"],
                            run_ids["second_20260601_020304"],
                            run_ids["third_20260601_030405"],
                        ]
                    )
                second_payloads = service.tags_for_runs(
                    [run_ids["third_20260601_030405"]]
                )

        self.assertEqual(load.call_count, 3)
        self.assertFalse(second_payloads[0].truncated)
        self.assertEqual(second_payloads[0].scalar_tags, ("train/loss",))

    def test_log_run_query_service_cached_tags_bypass_batch_budget(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logs_root = Path(tmp)
            write_fake_event_run(
                logs_root,
                "first_20260601_010203",
                b"11111111",
            )
            write_fake_event_run(
                logs_root,
                "second_20260601_020304",
                b"2222",
            )
            scanner = log_run_scanner(logs_root=logs_root)
            run_ids = {run.run_name: run.id for run in scanner.list_runs()}
            service = LogRunQueryService(
                scanner=scanner,
                max_tag_event_bytes=100,
                max_tag_batch_event_bytes=8,
            )

            with patch_event_accumulator_loader(
                return_value=FakeTensorBoardAccumulator(),
            ) as load:
                service.tags_for_runs([run_ids["first_20260601_010203"]])
                payloads = service.tags_for_runs(
                    [
                        run_ids["first_20260601_010203"],
                        run_ids["second_20260601_020304"],
                    ]
                )

        self.assertEqual(load.call_count, 2)
        self.assertFalse(payloads[0].truncated)
        self.assertFalse(payloads[1].truncated)
        self.assertEqual(payloads[0].scalar_tags, ("train/loss",))
        self.assertEqual(payloads[1].scalar_tags, ("train/loss",))


if __name__ == "__main__":
    unittest.main()
