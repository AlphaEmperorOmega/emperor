from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from model_runtime.runs import JsonlTrainingProgressCallback
from model_runtime.runs.progress import (
    NeuronClusterGrowthCallback,
    sanitize_metric_payload,
)


class RunsProgressTests(unittest.TestCase):
    def test_growth_callback_reports_a_neuron_regrown_after_pruning(self) -> None:
        events: list[dict[str, object]] = []
        cluster = SimpleNamespace(
            cluster={"neuron_1_1_1": object(), "neuron_2_1_1": object()},
            x_axis_total_neurons=2,
            y_axis_total_neurons=1,
            z_axis_total_neurons=1,
        )
        callback = NeuronClusterGrowthCallback(events.append)
        callback._clusters = [("cluster", cluster)]
        callback._known_names = {"cluster": set(cluster.cluster)}
        trainer = SimpleNamespace(current_epoch=0, global_step=1)

        del cluster.cluster["neuron_2_1_1"]
        callback.on_train_batch_end(trainer, None, None, None, 0)
        cluster.cluster["neuron_2_1_1"] = object()
        callback.on_train_batch_end(trainer, None, None, None, 1)

        self.assertEqual(
            events,
            [
                {
                    "type": "neuron_added",
                    "node": "cluster",
                    "coord": [2, 1, 1],
                    "count": 2,
                    "capacity": [2, 1, 1],
                    "epoch": 0,
                    "step": 1,
                }
            ],
        )

    def test_metric_sanitization_and_jsonl_shape_remain_portable(self) -> None:
        sanitized, original_count, dropped_count = sanitize_metric_payload(
            {
                "validation/accuracy": 0.75,
                "validation/confusion_matrix": [[1, 0], [0, 1]],
            }
        )
        self.assertEqual(sanitized, {"validation/accuracy": 0.75})
        self.assertEqual(original_count, 2)
        self.assertEqual(dropped_count, 1)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "progress.jsonl"
            callback = JsonlTrainingProgressCallback(path)
            callback.set_run_context(
                "Mnist",
                "logs/run/version_0",
                "baseline",
                "BASELINE",
                run_id="run-0001",
                run_index=1,
                run_total=1,
                total_epochs=30,
            )
            callback.write_event(
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
