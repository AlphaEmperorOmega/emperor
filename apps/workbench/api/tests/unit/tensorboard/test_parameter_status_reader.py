from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from tensorboard.backend.event_processing import event_accumulator
from torch.utils.tensorboard import SummaryWriter

from emperor_workbench.tensorboard import (
    ParameterStatus,
    TensorBoardParameterStatusReader,
)
from tests.unit.tensorboard._monitoring_support import (
    LargeParameterStatusAccumulator,
    ParameterStatusAccumulator,
)


class TensorBoardParameterStatusReaderTests(unittest.TestCase):
    def write_scalars(
        self,
        log_dir: Path,
        scalars: dict[str, list[tuple[int, float]]],
    ) -> None:
        writer = SummaryWriter(log_dir=str(log_dir))
        for tag, points in scalars.items():
            for step, value in points:
                writer.add_scalar(tag, value, step)
        writer.flush()
        writer.close()

    def test_classifies_delta_statuses_and_missing_channels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp) / "run"
            self.write_scalars(
                log_dir,
                {
                    "main_model.layers.0.model/weights/relative_delta_norm": [
                        (2, 0.0),
                        (3, 1e-6),
                    ],
                    "main_model.layers.0.model/bias/delta_norm": [(2, 0.0), (3, 0.0)],
                    "main_model.layers.1.model/weights/delta_norm": [
                        (2, 0.0),
                        (3, 0.0),
                    ],
                    "main_model.layers.2.model/weights/l2_norm": [(1, 4.0)],
                    "main_model.layers.3.model/weights/delta_norm": [
                        (2, 0.0),
                        (3, 0.25),
                    ],
                },
            )

            data = TensorBoardParameterStatusReader().read(
                source_id="run-1",
                preset="baseline",
                dataset="Mnist",
                log_dir=str(log_dir),
            )

        nodes = {node.node_path: node for node in data.nodes}
        self.assertEqual(data.source_id, "run-1")
        self.assertEqual(data.preset, "baseline")
        self.assertEqual(data.dataset, "Mnist")
        self.assertEqual(nodes["main_model.layers.0.model"].weights.status, "updated")
        self.assertEqual(
            nodes["main_model.layers.0.model"].weights.metric,
            "main_model.layers.0.model/weights/relative_delta_norm",
        )
        self.assertEqual(nodes["main_model.layers.0.model"].weights.last_step, 3)
        self.assertEqual(nodes["main_model.layers.0.model"].weights.observed_points, 2)
        self.assertEqual(nodes["main_model.layers.0.model"].bias.status, "unchanged")
        self.assertEqual(nodes["main_model.layers.0.model"].bias.observed_points, 2)
        self.assertEqual(nodes["main_model.layers.1.model"].weights.status, "unchanged")
        self.assertEqual(nodes["main_model.layers.1.model"].bias.status, "missing")
        self.assertEqual(nodes["main_model.layers.2.model"].weights.status, "unknown")
        self.assertEqual(nodes["main_model.layers.3.model"].weights.status, "updated")
        self.assertEqual(
            nodes["main_model.layers.3.model"].weights.metric,
            "main_model.layers.3.model/weights/delta_norm",
        )

    def test_single_zero_delta_point_is_unknown(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp) / "run"
            self.write_scalars(
                log_dir,
                {
                    "main_model.layers.0.model/weights/delta_norm": [(2, 0.0)],
                },
            )

            data = TensorBoardParameterStatusReader().read(
                source_id="run-1",
                preset="baseline",
                dataset="Mnist",
                log_dir=str(log_dir),
            )

        node = data.nodes[0]
        self.assertEqual(node.node_path, "main_model.layers.0.model")
        self.assertEqual(node.weights.status, "unknown")
        self.assertEqual(
            node.weights.metric, "main_model.layers.0.model/weights/delta_norm"
        )
        self.assertEqual(node.weights.last_step, 2)
        self.assertEqual(node.weights.observed_points, 1)
        self.assertEqual(node.bias.status, "missing")

    def test_old_logs_without_delta_metrics_use_value_stat_changes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp) / "run"
            self.write_scalars(
                log_dir,
                {
                    "main_model.layers.0.model/weights/l2_norm": [(1, 4.0), (2, 4.25)],
                    "main_model.layers.0.model/bias/mean": [(1, 0.5), (2, 0.5)],
                },
            )

            data = TensorBoardParameterStatusReader().read(
                source_id="run-1",
                preset="baseline",
                dataset="Mnist",
                log_dir=str(log_dir),
            )

        node = data.nodes[0]
        self.assertEqual(node.node_path, "main_model.layers.0.model")
        self.assertEqual(node.weights.status, "updated")
        self.assertEqual(
            node.weights.metric, "main_model.layers.0.model/weights/l2_norm"
        )
        self.assertEqual(node.bias.status, "unchanged")
        self.assertEqual(node.bias.metric, "main_model.layers.0.model/bias/mean")

    def test_missing_or_nonexistent_log_dir_returns_empty_status_payload(self) -> None:
        reader = TensorBoardParameterStatusReader()

        missing_data = reader.read(
            source_id="job-1",
            preset=None,
            dataset="Mnist",
            log_dir=None,
        )

        with tempfile.TemporaryDirectory() as tmp:
            nonexistent_log_dir = Path(tmp) / "missing-run"
            nonexistent_data = reader.read(
                source_id="job-1",
                preset=None,
                dataset="Mnist",
                log_dir=str(nonexistent_log_dir),
            )

        self.assertEqual(
            missing_data,
            ParameterStatus(
                source_id="job-1",
                preset=None,
                dataset="Mnist",
                log_dir=None,
                nodes=(),
            ),
        )
        self.assertEqual(
            nonexistent_data,
            ParameterStatus(
                source_id="job-1",
                preset=None,
                dataset="Mnist",
                log_dir=str(nonexistent_log_dir),
                nodes=(),
            ),
        )

    def test_oversized_event_files_skip_parameter_status_tensorboard_load(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp)
            event_file = log_dir / "events.out.tfevents.large"
            event_file.write_text("large-event-payload", encoding="utf-8")
            reader = TensorBoardParameterStatusReader(max_event_bytes=4)

            with patch(
                "emperor_workbench.tensorboard._events.load_event_accumulator"
            ) as load:
                data = reader.read(
                    source_id="job-1",
                    preset=None,
                    dataset="Mnist",
                    log_dir=str(log_dir),
                )

        self.assertEqual(
            data,
            ParameterStatus(
                source_id="job-1",
                preset=None,
                dataset="Mnist",
                log_dir=str(log_dir),
                nodes=(),
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

    def test_parameter_status_is_cached_until_event_files_change(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp)
            event_file = log_dir / "events.out.tfevents.cache"
            event_file.write_text("first", encoding="utf-8")
            reader = TensorBoardParameterStatusReader()

            with patch(
                "emperor_workbench.tensorboard._events.load_event_accumulator",
                return_value=ParameterStatusAccumulator(),
            ) as load:
                first = reader.read(
                    source_id="job-1",
                    preset=None,
                    dataset="Mnist",
                    log_dir=str(log_dir),
                )
                second = reader.read(
                    source_id="job-1",
                    preset=None,
                    dataset="Mnist",
                    log_dir=str(log_dir),
                )
                event_file.write_text("first-second", encoding="utf-8")
                changed = reader.read(
                    source_id="job-1",
                    preset=None,
                    dataset="Mnist",
                    log_dir=str(log_dir),
                )

        self.assertEqual(first, second)
        self.assertEqual(first, changed)
        self.assertEqual(load.call_count, 2)
        node = first.nodes[0]
        self.assertEqual(node.node_path, "main_model.layers.0.model")
        self.assertEqual(node.weights.status, "updated")

    def test_parameter_status_uses_custom_scalar_size_guidance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp)
            (log_dir / "events.out.tfevents.cache").write_text(
                "events",
                encoding="utf-8",
            )
            reader = TensorBoardParameterStatusReader(scalar_point_limit=7)

            with patch(
                "emperor_workbench.tensorboard._events.load_event_accumulator",
                return_value=ParameterStatusAccumulator(),
            ) as load:
                reader.read(
                    source_id="job-1",
                    preset=None,
                    dataset="Mnist",
                    log_dir=str(log_dir),
                )

        size_guidance = load.call_args.kwargs["size_guidance"]
        self.assertEqual(size_guidance[event_accumulator.SCALARS], 7)

    def test_parameter_status_classification_uses_bounded_scalar_tail(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp)
            (log_dir / "events.out.tfevents.cache").write_text(
                "events",
                encoding="utf-8",
            )
            reader = TensorBoardParameterStatusReader(scalar_point_limit=3)

            with patch(
                "emperor_workbench.tensorboard._events.load_event_accumulator",
                return_value=LargeParameterStatusAccumulator(),
            ):
                data = reader.read(
                    source_id="job-1",
                    preset=None,
                    dataset="Mnist",
                    log_dir=str(log_dir),
                )

        node = data.nodes[0]
        self.assertEqual(node.weights.status, "unchanged")
        self.assertEqual(node.weights.observed_points, 3)
        self.assertEqual(node.weights.last_step, 5)
        self.assertEqual(node.bias.status, "unchanged")
        self.assertEqual(node.bias.observed_points, 3)
        self.assertEqual(node.bias.last_step, 5)


if __name__ == "__main__":
    unittest.main()
