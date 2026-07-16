from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from emperor_workbench.run_plans import RunPlanPersistenceCodec
from emperor_workbench.training_jobs._filesystem_store import (
    FileSystemTrainingJobStore,
)
from tests.unit.training_jobs._support import make_record


class TrainingJobRunPlanPersistenceTests(unittest.TestCase):
    def test_training_job_store_preserves_the_run_plan_json_shape(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            record = make_record("job-1")
            record.root = root / record.id

            FileSystemTrainingJobStore(root).save(record)
            persisted = json.loads(
                (record.root / "metadata.json").read_text(encoding="utf-8")
            )
            recovered = FileSystemTrainingJobStore(root).get(record.id)

        self.assertEqual(
            persisted["run_plan"],
            RunPlanPersistenceCodec.encode(record.run_plan),
        )
        self.assertIsNotNone(recovered)
        assert recovered is not None
        self.assertEqual(recovered.run_plan, record.run_plan)


if __name__ == "__main__":
    unittest.main()
