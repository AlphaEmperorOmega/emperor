from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from model_runtime.runs import JsonlRunProgress
from model_runtime.runs._lightning_progress import lightning_progress_adapter
from model_runtime.runs._metrics import sanitize_metric_payload
from model_runtime.runs.progress import ContextualRunProgress, RunProgressContext


def _context() -> RunProgressContext:
    return RunProgressContext(
        experiment_task="image-classification",
        dataset="Mnist",
        preset="baseline",
        preset_key="BASELINE",
        log_dir="logs/run/version_0",
        run_id="run-0001",
        run_index=1,
        run_total=1,
        total_epochs=30,
    )


class RunsProgressTests(unittest.TestCase):
    def test_jsonl_writer_rejects_non_finite_metrics_before_persisting(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "progress.jsonl"
            progress = JsonlRunProgress(path)

            with self.assertRaises(ValueError):
                progress.write_event({"type": "step", "metrics": {"loss": math.inf}})

            self.assertFalse(path.exists())

    def test_growth_callback_reports_a_neuron_regrown_after_pruning(self) -> None:
        events: list[dict[str, object]] = []
        cluster = SimpleNamespace(
            cluster={"neuron_1_1_1": object(), "neuron_2_1_1": object()},
            x_axis_total_neurons=2,
            y_axis_total_neurons=1,
            z_axis_total_neurons=1,
        )
        progress = type(
            "Progress",
            (),
            {"write_event": lambda _self, event: events.append(dict(event))},
        )()
        callback = lightning_progress_adapter(
            ContextualRunProgress(progress, _context()),
            step_interval=10,
        )
        callback._clusters = [("cluster", cluster)]
        callback._known_names = {"cluster": set(cluster.cluster)}
        trainer = SimpleNamespace(current_epoch=0, global_step=1)

        del cluster.cluster["neuron_2_1_1"]
        callback.on_train_batch_end(trainer, None, None, None, 0)
        cluster.cluster["neuron_2_1_1"] = object()
        callback.on_train_batch_end(trainer, None, None, None, 1)

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["type"], "neuron_added")
        self.assertEqual(events[0]["coord"], [2, 1, 1])
        self.assertEqual(events[0]["runId"], "run-0001")
        self.assertEqual(events[0]["experimentTask"], "image-classification")

    def test_metric_sanitization_and_jsonl_shape_remain_portable(self) -> None:
        sanitized, original_count, dropped_count = sanitize_metric_payload(
            {
                "validation/accuracy": 0.75,
                "validation/confusion_matrix": [[1, 0], [0, 1]],
            },
            metric_key_limit=512,
            string_value_limit=20_000,
        )
        self.assertEqual(sanitized, {"validation/accuracy": 0.75})
        self.assertEqual(original_count, 2)
        self.assertEqual(dropped_count, 1)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "progress.jsonl"
            progress = ContextualRunProgress(
                JsonlRunProgress(path),
                _context(),
            )
            progress.write_event(
                {
                    "type": "dataset_started",
                    "status": "running",
                }
            )
            payload = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(payload["dataset"], "Mnist")
        self.assertEqual(payload["preset"], "baseline")
        self.assertEqual(payload["presetKey"], "BASELINE")
        self.assertEqual(payload["runId"], "run-0001")
        self.assertEqual(payload["runIndex"], 1)
        self.assertEqual(payload["runTotal"], 1)
        self.assertEqual(payload["totalEpochs"], 30)
        self.assertEqual(payload["type"], "dataset_started")
        self.assertEqual(payload["status"], "running")


if __name__ == "__main__":
    unittest.main()
