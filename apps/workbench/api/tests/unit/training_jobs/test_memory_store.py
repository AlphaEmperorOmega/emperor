from __future__ import annotations

import unittest

from emperor_workbench.training_jobs._memory_store import InMemoryTrainingJobStore
from tests.unit.training_jobs._support import make_record


class InMemoryTrainingJobStoreTests(unittest.TestCase):
    def test_save_get_and_list_records(self) -> None:
        store = InMemoryTrainingJobStore()
        record = make_record()

        store.save(record)

        self.assertIs(store.get("job-1"), record)
        self.assertEqual(store.list(), [record])

    def test_get_missing_record_returns_none(self) -> None:
        store = InMemoryTrainingJobStore()

        self.assertIsNone(store.get("missing"))

    def test_in_memory_store_has_no_cross_instance_persistence(self) -> None:
        first_store = InMemoryTrainingJobStore()
        second_store = InMemoryTrainingJobStore()
        first_store.save(make_record())

        self.assertIsNone(second_store.get("job-1"))
        self.assertEqual(second_store.list(), [])


if __name__ == "__main__":
    unittest.main()
