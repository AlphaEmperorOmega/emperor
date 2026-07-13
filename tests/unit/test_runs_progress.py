from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from model_runtime.runs import JsonlTrainingProgressCallback
from model_runtime.runs.progress import (
    sanitize_metric_payload,
)


class RunsProgressTests(unittest.TestCase):
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
