from __future__ import annotations

import os
import unittest

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from emperor.inspection import (
    InspectionRequest,
    configuration_schema,
    parse_overrides,
    search_space_schema,
)
from emperor.inspection import (
    inspect_model as inspect_model_semantically,
)
from emperor.model_packages import ModelPackage, model_package

from workbench.backend.inspection_errors import call_inspection, call_model_package
from workbench.backend.inspection_serialization import (
    configuration_schema_payload,
    inspection_result_payload,
    model_presets_payload,
    search_space_payload,
)
from workbench.backend.inspector.errors import InspectorError
from workbench.backend.inspector.schema import (
    config_schema as legacy_config_schema,
)
from workbench.backend.inspector.schema import (
    search_space_schema as legacy_search_space_schema,
)
from workbench.backend.inspector.service import inspect_model as legacy_inspect_model


class InspectionAdapterEquivalenceTests(unittest.TestCase):
    def test_broken_package_failures_map_to_stable_workbench_errors(self) -> None:
        package = ModelPackage(
            "broken",
            "missing",
            "models.__inspection_missing__",
        )

        calls = (
            lambda: call_inspection(configuration_schema, package),
            lambda: call_inspection(
                parse_overrides,
                package,
                {"HIDDEN_DIM": "1"},
            ),
            lambda: call_model_package(
                package,
                model_presets_payload,
                package,
            ),
        )
        for call in calls:
            with self.subTest(call=call):
                with self.assertRaises(InspectorError) as raised:
                    call()
                self.assertEqual(raised.exception.status_code, 400)
                self.assertIn(
                    "Failed to import model package 'broken/missing'",
                    raised.exception.detail,
                )

    def test_graph_adapter_matches_legacy_path_for_each_experiment_task(self) -> None:
        cases = (
            ("linears/linear", "baseline", "Mnist", "image-classification"),
            (
                "linears/linear_adaptive",
                "full-stack",
                "Cifar10",
                "image-classification",
            ),
            (
                "bert/linear",
                "baseline",
                "PennTreebankBertPretraining",
                "bert-pretraining",
            ),
            (
                "gpt/linear",
                "baseline",
                "WikiText2",
                "causal-language-modeling",
            ),
            (
                "transformer/linear",
                "baseline",
                "Multi30kDeEn",
                "text-translation",
            ),
        )
        for model_id, preset, dataset, experiment_task in cases:
            with self.subTest(model=model_id):
                package = model_package(model_id)
                assert package is not None
                semantic = inspect_model_semantically(
                    package,
                    InspectionRequest(
                        preset=preset,
                        dataset=dataset,
                        experiment_task=experiment_task,
                    ),
                )
                self.assertEqual(
                    inspection_result_payload(semantic),
                    legacy_inspect_model(
                        model_id,
                        preset,
                        dataset=dataset,
                        experiment_task=experiment_task,
                    ),
                )

    def test_schema_adapters_match_legacy_simple_and_adaptive_paths(self) -> None:
        cases = (
            ("linears/linear", "baseline", None),
            ("linears/linear", "gating", None),
            ("linears/linear_adaptive", "full-stack", None),
            (
                "linears/linear_adaptive",
                "baseline",
                ["full-stack", "dual-weight-gating"],
            ),
        )
        for model_id, preset, presets in cases:
            with self.subTest(model=model_id, preset=preset, presets=presets):
                package = model_package(model_id)
                assert package is not None
                self.assertEqual(
                    configuration_schema_payload(
                        configuration_schema(package, preset=preset)
                    ),
                    legacy_config_schema(model_id, preset),
                )
                self.assertEqual(
                    search_space_payload(
                        search_space_schema(
                            package,
                            preset=preset,
                            presets=presets,
                        )
                    ),
                    legacy_search_space_schema(model_id, preset, presets),
                )


if __name__ == "__main__":
    unittest.main()
