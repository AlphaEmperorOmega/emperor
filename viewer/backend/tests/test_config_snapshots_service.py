from __future__ import annotations

import re
import unittest

from viewer.backend.config_snapshots import InMemoryConfigSnapshotStore
from viewer.backend.inspector.errors import InspectorError
from viewer.backend.repositories.config_snapshots import ConfigSnapshotRepository
from viewer.backend.services.config_snapshots import ConfigSnapshotService


class ConfigSnapshotServiceAdaptiveValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store = InMemoryConfigSnapshotStore()
        self.service = ConfigSnapshotService(ConfigSnapshotRepository(self.store))

    def test_create_rejects_adaptive_flag_without_matching_option(self) -> None:
        with self.assertRaisesRegex(
            InspectorError,
            re.escape(
                "Invalid config snapshot overrides: weight_option must be set "
                "when weight_option_flag is True."
            ),
        ):
            self.service.create_snapshot(
                model="linears/linear_adaptive",
                preset="baseline",
                name="flag only",
                overrides={"WEIGHT_OPTION_FLAG": "true"},
            )

    def test_update_rejects_adaptive_flag_without_matching_option(self) -> None:
        snapshot = self.service.create_snapshot(
            model="linears/linear_adaptive",
            preset="baseline",
            name="wider stack",
            overrides={"HIDDEN_DIM": "384"},
        )

        with self.assertRaisesRegex(
            InspectorError,
            re.escape(
                "Invalid config snapshot overrides: weight_option must be set "
                "when weight_option_flag is True."
            ),
        ):
            self.service.update_snapshot(
                snapshot["id"],
                overrides={"WEIGHT_OPTION_FLAG": "true"},
            )

        stored = self.store.get(snapshot["id"])
        self.assertIsNotNone(stored)
        assert stored is not None
        self.assertEqual(stored.overrides, {"HIDDEN_DIM": "384"})

    def test_create_accepts_adaptive_flag_with_matching_option(self) -> None:
        snapshot = self.service.create_snapshot(
            model="linears/linear_adaptive",
            preset="baseline",
            name="single weight",
            overrides={
                "WEIGHT_OPTION_FLAG": "true",
                "WEIGHT_OPTION": "SingleModelDynamicWeightConfig",
            },
        )

        self.assertEqual(
            snapshot["overrides"],
            {
                "WEIGHT_OPTION_FLAG": "true",
                "WEIGHT_OPTION": "SingleModelDynamicWeightConfig",
            },
        )


if __name__ == "__main__":
    unittest.main()
