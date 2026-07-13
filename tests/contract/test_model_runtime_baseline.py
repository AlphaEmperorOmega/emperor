from __future__ import annotations

import json
import unittest
from pathlib import Path

from models.catalog import (
    discover_model_ids,
    discover_model_types,
    model_package,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASELINE_PATH = (
    PROJECT_ROOT
    / "docs"
    / "architecture"
    / "model-runtime-baseline-2026-07-13.json"
)


class ModelRuntimeBaselineTests(unittest.TestCase):
    def test_catalog_identity_order_and_implementations_match_baseline(self) -> None:
        baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
        catalog = baseline["catalog"]

        self.assertEqual([entry["id"] for entry in catalog], discover_model_ids())
        self.assertEqual(
            [entry["module"] for entry in catalog],
            [model_package(model_id).module_path for model_id in discover_model_ids()],
        )
        self.assertEqual(baseline["cli_model_types"], discover_model_types())


if __name__ == "__main__":
    unittest.main()
