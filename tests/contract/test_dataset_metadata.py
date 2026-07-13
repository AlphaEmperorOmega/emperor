from __future__ import annotations

import tempfile
import unittest
from collections import defaultdict
from inspect import signature
from pathlib import Path

import torch
from emperor.experiments.tasks import ExperimentTask
from models.catalog import discover_model_packages

from support.dataset_metadata import DATASET_TASKS, offline_dataset_metadata

SEED = 17
BATCH_SIZE = 2


def declared_dataset_tasks() -> dict[type, set[ExperimentTask]]:
    tasks_by_dataset: dict[type, set[ExperimentTask]] = defaultdict(set)
    for package in discover_model_packages():
        for task, dataset_types in package.dataset_metadata.items():
            for dataset_type in dataset_types:
                tasks_by_dataset[dataset_type].add(task)
    return dict(tasks_by_dataset)


class DatasetMetadataContractTests(unittest.TestCase):
    def test_every_declared_dataset_seed_is_optional_by_default(self) -> None:
        for dataset_type in DATASET_TASKS:
            with self.subTest(dataset=dataset_type.__name__):
                seed_parameter = signature(dataset_type).parameters["seed"]
                self.assertIsNone(seed_parameter.default)

                with tempfile.TemporaryDirectory() as temporary_directory:
                    with offline_dataset_metadata(
                        dataset_type,
                        Path(temporary_directory),
                        seed=None,
                    ) as dataset:
                        self.assertIsNone(dataset.seed)
                        dataset.prepare_data()
                        dataset.setup("fit")
                        self.assertIsNone(dataset.train_dataloader().generator)

    def test_every_declared_dataset_treats_zero_as_an_explicit_seed(self) -> None:
        for dataset_type in DATASET_TASKS:
            with self.subTest(dataset=dataset_type.__name__):
                with tempfile.TemporaryDirectory() as temporary_directory:
                    with offline_dataset_metadata(
                        dataset_type,
                        Path(temporary_directory),
                        seed=0,
                    ) as dataset:
                        self.assertEqual(dataset.seed, 0)
                        dataset.prepare_data()
                        dataset.setup("fit")
                        generator = dataset.train_dataloader().generator
                        self.assertIsNotNone(generator)
                        self.assertEqual(generator.initial_seed(), 0)

    def test_every_declared_dataset_has_one_compatible_task_and_offline_fixture(
        self,
    ) -> None:
        actual = declared_dataset_tasks()
        expected = {
            dataset_type: {task} for dataset_type, task in DATASET_TASKS.items()
        }

        self.assertEqual(actual, expected)

    def test_every_declared_dataset_has_deterministic_one_batch_semantics(
        self,
    ) -> None:
        for dataset_type, task in DATASET_TASKS.items():
            with self.subTest(dataset=dataset_type.__name__, task=task.name):
                batches = []
                loader_seeds = []
                with tempfile.TemporaryDirectory() as temporary_directory:
                    root = Path(temporary_directory)
                    for run_index in range(2):
                        with offline_dataset_metadata(
                            dataset_type,
                            root / f"run-{run_index}",
                            seed=SEED,
                        ) as dataset:
                            self.assertEqual(dataset.seed, SEED)
                            dataset.prepare_data()
                            dataset.setup("fit")
                            loader = dataset.train_dataloader()
                            self.assertIsNotNone(loader.generator)
                            loader_seeds.append(loader.generator.initial_seed())
                            batches.append(next(iter(loader)))

                self.assertEqual(loader_seeds, [SEED, SEED])
                self.assertEqual(len(batches[0]), len(batches[1]))
                for first, second in zip(*batches, strict=True):
                    torch.testing.assert_close(first, second)

                self._assert_batch_semantics(task, dataset_type, batches[0])

    def _assert_batch_semantics(
        self,
        task: ExperimentTask,
        dataset_type: type,
        batch,
    ) -> None:
        if task == ExperimentTask.IMAGE_CLASSIFICATION:
            inputs, labels = batch
            self.assertEqual(inputs.dtype, torch.float32)
            self.assertEqual(labels.dtype, torch.long)
            self.assertEqual(inputs.ndim, 4)
            self.assertEqual(inputs.shape[0], BATCH_SIZE)
            self.assertEqual(inputs.shape[1], dataset_type.num_channels)
            self.assertEqual(labels.shape, torch.Size([BATCH_SIZE]))
            self.assertTrue(torch.isfinite(inputs).all())
            self.assertTrue(
                torch.all((labels >= 0) & (labels < dataset_type.num_classes))
            )
            return

        if task == ExperimentTask.CAUSAL_LANGUAGE_MODELING:
            input_ids, labels = batch
            self.assertEqual(input_ids.dtype, torch.long)
            self.assertEqual(labels.dtype, torch.long)
            self.assertEqual(input_ids.shape, torch.Size([BATCH_SIZE, 4]))
            self.assertEqual(labels.shape, input_ids.shape)
            torch.testing.assert_close(input_ids[:, 1:], labels[:, :-1])
            return

        if task == ExperimentTask.BERT_PRETRAINING:
            (
                input_ids,
                mlm_labels,
                attention_mask,
                token_type_ids,
                next_sentence_labels,
            ) = batch
            for tensor in batch:
                self.assertEqual(tensor.dtype, torch.long)
            expected_sequence_shape = torch.Size([BATCH_SIZE, 10])
            self.assertEqual(input_ids.shape, expected_sequence_shape)
            self.assertEqual(mlm_labels.shape, expected_sequence_shape)
            self.assertEqual(attention_mask.shape, expected_sequence_shape)
            self.assertEqual(token_type_ids.shape, expected_sequence_shape)
            self.assertEqual(next_sentence_labels.shape, torch.Size([BATCH_SIZE]))
            self.assertTrue(torch.all((attention_mask == 0) | (attention_mask == 1)))
            self.assertTrue(torch.all((token_type_ids == 0) | (token_type_ids == 1)))
            self.assertTrue(
                torch.all(
                    (next_sentence_labels == 0) | (next_sentence_labels == 1)
                )
            )
            return

        if task == ExperimentTask.TEXT_TRANSLATION:
            source_ids, target_ids = batch
            self.assertEqual(source_ids.dtype, torch.long)
            self.assertEqual(target_ids.dtype, torch.long)
            self.assertEqual(source_ids.shape, torch.Size([BATCH_SIZE, 7]))
            self.assertEqual(target_ids.shape, torch.Size([BATCH_SIZE, 6]))
            self.assertTrue(torch.all(source_ids[:, 0] == 2))
            self.assertTrue(torch.all(target_ids[:, 0] == 2))
            self.assertTrue(torch.all((source_ids == 3).any(dim=1)))
            self.assertTrue(torch.all((target_ids == 3).any(dim=1)))
            return

        self.fail(f"Unhandled Experiment Task: {task!r}")


if __name__ == "__main__":
    unittest.main()
