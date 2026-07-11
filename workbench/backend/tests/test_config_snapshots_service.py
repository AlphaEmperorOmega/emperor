from __future__ import annotations

import re
import unittest

from workbench.backend.config_snapshots import InMemoryConfigSnapshotStore
from workbench.backend.inspector.errors import InspectorError
from workbench.backend.services.config_snapshots import ConfigSnapshotService


class ConfigSnapshotServiceAdaptiveValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store = InMemoryConfigSnapshotStore()
        self.service = ConfigSnapshotService(self.store)

    def test_create_rejects_adaptive_flag_without_matching_option(self) -> None:
        with self.assertRaisesRegex(
            InspectorError,
            re.escape(
                "Invalid config snapshot overrides: models.linears.linear_adaptive: "
                "runtime key 'weight_option' must be set when "
                "'weight_option_flag' is True"
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
                "Invalid config snapshot overrides: models.linears.linear_adaptive: "
                "runtime key 'weight_option' must be set when "
                "'weight_option_flag' is True"
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

    def test_real_schema_create_and_update_freeze_canonical_serialization(
        self,
    ) -> None:
        snapshot = self.service.create_snapshot(
            model="linears/linear",
            preset="baseline",
            name="canonical values",
            overrides={
                "hidden-dim": "128.0",
                "stack-bias-flag": "FALSE",
                "stack_dropout_probability": "0.250",
            },
        )

        self.assertEqual(
            snapshot["overrides"],
            {
                "HIDDEN_DIM": "128",
                "STACK_DROPOUT_PROBABILITY": "0.25",
                "STACK_BIAS_FLAG": "false",
            },
        )

        updated = self.service.update_snapshot(
            snapshot["id"],
            overrides={
                "hidden_dim": "64.0",
                "memory_stack_hidden_dim": "48",
            },
        )
        self.assertEqual(
            updated["overrides"],
            {"HIDDEN_DIM": "64", "MEMORY_STACK_HIDDEN_DIM": "48"},
        )

    def test_real_schema_rejects_preset_locked_create_without_persisting(self) -> None:
        with self.assertRaisesRegex(InspectorError, "preset-locked fields"):
            self.service.create_snapshot(
                model="linears/linear",
                preset="gating",
                name="disable locked gate",
                overrides={"gate_flag": "false"},
            )

        self.assertEqual(self.store.list("linears/linear"), [])

    def test_failed_real_construction_does_not_persist_create(self) -> None:
        with self.assertRaisesRegex(
            InspectorError,
            "runtime key 'weight_option' must be set",
        ):
            self.service.create_snapshot(
                model="linears/linear_adaptive",
                preset="baseline",
                name="invalid adaptive config",
                overrides={"weight_option_flag": "true"},
            )

        self.assertEqual(self.store.list("linears/linear_adaptive"), [])


if __name__ == "__main__":
    unittest.main()
