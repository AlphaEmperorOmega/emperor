from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from emperor.experiments.tasks import ExperimentTask

from support.dataset_metadata import DATASET_TASKS

RUN_DOWNLOAD_TESTS = os.environ.get("EMPEROR_RUN_DATASET_DOWNLOAD_TESTS") == "1"


@unittest.skipUnless(
    RUN_DOWNLOAD_TESTS,
    "set EMPEROR_RUN_DATASET_DOWNLOAD_TESTS=1 for real dataset downloads",
)
class DatasetDownloadIntegrationTests(unittest.TestCase):
    def test_every_declared_dataset_prepares_from_its_real_source(self) -> None:
        with tempfile.TemporaryDirectory(prefix="emperor-datasets-") as temporary:
            root = Path(temporary)
            for dataset_type, task in DATASET_TASKS.items():
                with self.subTest(dataset=dataset_type.__name__, task=task.name):
                    dataset = self._dataset(dataset_type, task, root)
                    dataset.prepare_data()

    def _dataset(self, dataset_type: type, task: ExperimentTask, root: Path):
        kwargs = {"batch_size": 2, "seed": 17}
        if task in {
            ExperimentTask.BERT_PRETRAINING,
            ExperimentTask.CAUSAL_LANGUAGE_MODELING,
        }:
            kwargs.update(root=str(root), num_workers=0)
        elif task == ExperimentTask.TEXT_TRANSLATION:
            kwargs.update(root=root, num_workers=0)
        dataset = dataset_type(**kwargs)
        if task == ExperimentTask.IMAGE_CLASSIFICATION:
            dataset.root = str(root)
            dataset.num_workers = 0
        return dataset


if __name__ == "__main__":
    unittest.main()
