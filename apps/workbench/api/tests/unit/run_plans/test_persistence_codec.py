from __future__ import annotations

import copy
import math
import unittest
from dataclasses import replace

from emperor_workbench.run_plans import RunPlanPersistenceCodec, TrainingSearch
from tests.unit.training_jobs._support import make_record


class RunPlanPersistenceCodecTests(unittest.TestCase):
    def test_custom_value_authorization_survives_persistence(self) -> None:
        custom_search = TrainingSearch(
            mode="grid",
            values={"HIDDEN_DIM": [65]},
            custom_value_axes=("HIDDEN_DIM",),
        )
        source = make_record().run_plan
        plan = replace(
            source,
            search=custom_search,
            preset_searches={"baseline": custom_search},
        )

        payload = RunPlanPersistenceCodec.encode(plan)
        self.assertEqual(payload["search"]["customValueAxes"], ["HIDDEN_DIM"])
        self.assertEqual(
            payload["presetSearches"]["baseline"]["customValueAxes"],
            ["HIDDEN_DIM"],
        )

        restored = RunPlanPersistenceCodec.decode(payload)
        self.assertEqual(restored.search, custom_search)
        self.assertEqual(restored.preset_searches["baseline"], custom_search)

        legacy_payload = copy.deepcopy(payload)
        del legacy_payload["search"]["customValueAxes"]
        del legacy_payload["presetSearches"]["baseline"]["customValueAxes"]
        legacy = RunPlanPersistenceCodec.decode(legacy_payload)
        self.assertEqual(legacy.search.custom_value_axes, ())
        self.assertEqual(legacy.preset_searches["baseline"].custom_value_axes, ())

    def test_run_plan_codec_rejects_retired_command_field(self) -> None:
        payload = RunPlanPersistenceCodec.encode(make_record().run_plan)
        payload["runs"].append(
            {
                "id": "run-1",
                "index": 1,
                "status": "Pending",
                "preset": "baseline",
                "dataset": "Mnist",
                "experimentTask": "classification",
                "changes": [],
                "overrides": {},
                "commandArgv": ["mise", "run", "experiment", "--"],
                "commands": {
                    "posix": "mise run experiment --",
                    "powershell": "mise run experiment --",
                },
                "command": "python retired.py",
                "totalEpochs": 1,
            }
        )

        with self.assertRaisesRegex(ValueError, "command is retired"):
            RunPlanPersistenceCodec.decode(payload)

    def test_run_plan_codec_rejects_malformed_values_without_coercion(self) -> None:
        payload = RunPlanPersistenceCodec.encode(make_record().run_plan)
        malformed_payloads = []

        string_summary = copy.deepcopy(payload)
        string_summary["summary"]["totalRuns"] = "0"
        malformed_payloads.append(string_summary)

        boolean_marker = copy.deepcopy(payload)
        boolean_marker["isRandomSearch"] = 0
        malformed_payloads.append(boolean_marker)

        nonfinite_override = copy.deepcopy(payload)
        nonfinite_override["overrides"] = {"HIDDEN_DIM": math.inf}
        malformed_payloads.append(nonfinite_override)

        non_string_preset = copy.deepcopy(payload)
        non_string_preset["presets"] = [1]
        malformed_payloads.append(non_string_preset)

        duplicate_dataset = copy.deepcopy(payload)
        duplicate_dataset["datasets"] = ["Mnist", "Mnist"]
        malformed_payloads.append(duplicate_dataset)

        for malformed in malformed_payloads:
            with self.subTest(payload=malformed), self.assertRaises(ValueError):
                RunPlanPersistenceCodec.decode(malformed)

        for malformed_search in (
            {"mode": "shuffle", "values": {"HIDDEN_DIM": [64]}},
            {
                "mode": "random",
                "values": {"HIDDEN_DIM": [64]},
                "randomSamples": "1",
            },
            {
                "mode": "grid",
                "values": {
                    "hidden_dim": [64],
                    "HIDDEN_DIM": [128],
                },
            },
            {"mode": "grid", "values": {"HIDDEN_DIM": [math.nan]}},
            {
                "mode": "grid",
                "values": {"HIDDEN_DIM": [64]},
                "customValueAxes": ["UNKNOWN"],
            },
            {
                "mode": "grid",
                "values": {"HIDDEN_DIM": [64]},
                "customValueAxes": ["hidden_dim", "HIDDEN_DIM"],
            },
        ):
            with (
                self.subTest(search=malformed_search),
                self.assertRaises(ValueError),
            ):
                RunPlanPersistenceCodec.decode_search(malformed_search)


if __name__ == "__main__":
    unittest.main()
