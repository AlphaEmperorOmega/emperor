from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from torch.utils.tensorboard import SummaryWriter

from emperor_workbench.tensorboard import MonitorData, TensorBoardMonitorReader
from tests.unit.tensorboard._monitoring_support import (
    NoMatchingMonitorAccumulator,
    ReadFailureAccumulator,
    TagsFailureAccumulator,
)


class TensorBoardMonitorReaderFailureTests(unittest.TestCase):
    def read_with_accumulators(
        self,
        log_dir: Path,
        accumulators: list[object | None],
    ) -> MonitorData:
        run_dirs = [log_dir / f"run-{index}" for index, _ in enumerate(accumulators)]
        for index, run_dir in enumerate(run_dirs):
            run_dir.mkdir()
            run_dir.joinpath(f"events.out.tfevents.{index}").write_bytes(b"events")
        with patch(
            "emperor_workbench.tensorboard._events.load_event_accumulator",
            side_effect=accumulators,
        ):
            return TensorBoardMonitorReader().read(
                job_id="job-1",
                node_path="main_model.0.model",
                dataset="Mnist",
                log_dir=str(log_dir),
            )

    def assert_empty_monitor_payload(
        self,
        data: MonitorData,
        log_dir: Path | None,
    ) -> None:
        self.assertEqual(
            data,
            MonitorData(
                job_id="job-1",
                node_path="main_model.0.model",
                preset=None,
                dataset="Mnist",
                log_dir=str(log_dir) if log_dir is not None else None,
                scalar_series=(),
                histograms=(),
                images=(),
            ),
        )

    def test_missing_or_nonexistent_log_dir_returns_empty_payload(self) -> None:
        reader = TensorBoardMonitorReader()

        missing_data = reader.read(
            job_id="job-1",
            node_path="main_model.0.model",
            dataset="Mnist",
            log_dir=None,
        )
        self.assert_empty_monitor_payload(missing_data, None)

        with tempfile.TemporaryDirectory() as tmp:
            nonexistent_log_dir = Path(tmp) / "missing-run"
            nonexistent_data = reader.read(
                job_id="job-1",
                node_path="main_model.0.model",
                dataset="Mnist",
                log_dir=str(nonexistent_log_dir),
            )

        self.assert_empty_monitor_payload(nonexistent_data, nonexistent_log_dir)

    def test_accumulator_load_failure_returns_empty_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp)
            data = self.read_with_accumulators(log_dir, [None])

        self.assert_empty_monitor_payload(data, log_dir)

    def test_oversized_event_files_skip_monitor_tensorboard_load(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp)
            event_file = log_dir / "events.out.tfevents.large"
            event_file.write_text("large-event-payload", encoding="utf-8")
            reader = TensorBoardMonitorReader(max_event_bytes=4)

            with patch(
                "emperor_workbench.tensorboard._events.load_event_accumulator"
            ) as load:
                data = reader.read(
                    job_id="job-1",
                    node_path="main_model.0.model",
                    dataset="Mnist",
                    log_dir=str(log_dir),
                )

        self.assertEqual(
            data,
            MonitorData(
                job_id="job-1",
                node_path="main_model.0.model",
                preset=None,
                dataset="Mnist",
                log_dir=str(log_dir),
                scalar_series=(),
                histograms=(),
                images=(),
                event_bytes=len("large-event-payload"),
                skipped_event_files=1,
                truncated=True,
                truncation_reason=(
                    "event files skipped: 19 bytes exceeds 4 byte read cap"
                ),
                source_item_count=1,
                returned_item_count=0,
            ),
        )
        load.assert_not_called()

    def test_tags_failure_returns_empty_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp)
            data = self.read_with_accumulators(log_dir, [TagsFailureAccumulator()])

        self.assert_empty_monitor_payload(data, log_dir)

    def test_scalar_histogram_and_image_failures_are_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp)
            data = self.read_with_accumulators(log_dir, [ReadFailureAccumulator()])

        self.assert_empty_monitor_payload(data, log_dir)

    def test_monitor_reader_matches_legacy_layer_path_aliases(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp)
            writer = SummaryWriter(log_dir=str(log_dir))
            writer.add_scalar("main_model.0.model/weights/mean", 0.25, 1)
            writer.flush()
            writer.close()

            data = TensorBoardMonitorReader().read(
                job_id="job-1",
                node_path="main_model.layers.0.model",
                dataset="Mnist",
                log_dir=str(log_dir),
            )

        self.assertEqual(data.node_path, "main_model.layers.0.model")
        self.assertEqual(
            [(series.tag, series.label) for series in data.scalar_series],
            [("main_model.0.model/weights/mean", "weights/mean")],
        )

    def test_negative_monitor_results_are_cached_until_event_files_change(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp)
            event_file = log_dir / "events.out.tfevents.cache"
            event_file.write_text("first", encoding="utf-8")
            reader = TensorBoardMonitorReader()

            with patch(
                "emperor_workbench.tensorboard._events.load_event_accumulator",
                return_value=NoMatchingMonitorAccumulator(),
            ) as load:
                first = reader.read(
                    job_id="job-1",
                    node_path="main_model.0.model",
                    dataset="Mnist",
                    log_dir=str(log_dir),
                )
                second = reader.read(
                    job_id="job-1",
                    node_path="main_model.0.model",
                    dataset="Mnist",
                    log_dir=str(log_dir),
                )
                event_file.write_text("first-second", encoding="utf-8")
                changed = reader.read(
                    job_id="job-1",
                    node_path="main_model.0.model",
                    dataset="Mnist",
                    log_dir=str(log_dir),
                )

        self.assertEqual(first, second)
        self.assertEqual(first, changed)
        self.assertEqual(load.call_count, 2)


if __name__ == "__main__":
    unittest.main()
