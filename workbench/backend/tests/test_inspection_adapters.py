from __future__ import annotations

import os
import unittest
from unittest.mock import patch

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from emperor.inspection import InspectionRequest, InspectionResult
from emperor.model_packages import ModelPackage

from workbench.backend.api.v1.routers.models import (
    _config_schema as http_config_schema,
)
from workbench.backend.api.v1.routers.models import (
    _search_space as http_search_space,
)
from workbench.backend.inspection_adapter import WorkbenchInspectionAdapter
from workbench.backend.inspection_errors import InspectionFailure
from workbench.backend.inspection_serialization import (
    configuration_schema_payload,
    inspection_result_payload,
    search_space_payload,
)
from workbench.backend.inspector.schema import (
    config_schema as legacy_config_schema,
)
from workbench.backend.inspector.schema import (
    search_space_schema as legacy_search_space_schema,
)
from workbench.backend.inspector.service import inspect_model as legacy_inspect_model
from workbench.backend.services.inspection import InspectionService


class InspectionAdapterEquivalenceTests(unittest.TestCase):
    def test_inspection_service_returns_transport_neutral_result(self) -> None:
        result = InspectionService().inspect(
            model_type="linears",
            model="linear",
            preset="baseline",
            overrides={},
            dataset="Mnist",
        )

        self.assertIsInstance(result, InspectionResult)

    def test_broken_package_failures_map_to_stable_workbench_errors(self) -> None:
        package = ModelPackage(
            "broken",
            "missing",
            "models.__inspection_missing__",
        )
        adapter = WorkbenchInspectionAdapter.from_package(package)

        calls = (
            adapter.configuration,
            lambda: adapter.parse_overrides(
                {"HIDDEN_DIM": "1"},
            ),
            adapter.presets_payload,
        )
        for call in calls:
            with self.subTest(call=call):
                with self.assertRaises(InspectionFailure) as raised:
                    call()
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
                adapter = WorkbenchInspectionAdapter.select(model_id)
                expected = adapter.inspect_payload(
                    InspectionRequest(
                        preset=preset,
                        dataset=dataset,
                        experiment_task=experiment_task,
                    ),
                )
                self.assertEqual(
                    expected,
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
                adapter = WorkbenchInspectionAdapter.select(model_id)
                self.assertEqual(
                    adapter.configuration_payload(preset),
                    legacy_config_schema(model_id, preset),
                )
                self.assertEqual(
                    adapter.search_space_payload(preset, presets),
                    legacy_search_space_schema(model_id, preset, presets),
                )

    def test_canonical_adapter_owns_selected_inspection_and_payloads(self) -> None:
        adapter = WorkbenchInspectionAdapter.select("linears/linear")
        request = InspectionRequest(
            preset="baseline",
            dataset="Mnist",
            experiment_task="image-classification",
        )

        self.assertEqual(
            adapter.inspect_payload(request),
            inspection_result_payload(adapter.inspect(request)),
        )
        self.assertEqual(
            adapter.configuration_payload("baseline"),
            configuration_schema_payload(adapter.configuration("baseline")),
        )
        self.assertEqual(
            adapter.search_space_payload("baseline"),
            search_space_payload(adapter.search_space("baseline")),
        )

    def test_legacy_graph_and_schema_paths_delegate_to_canonical_adapter(
        self,
    ) -> None:
        graph_calls: list[str] = []
        semantic_graph_calls: list[str] = []
        configuration_calls: list[str] = []
        search_calls: list[str] = []
        original_graph = WorkbenchInspectionAdapter.inspect_payload
        original_semantic_graph = WorkbenchInspectionAdapter.inspect
        original_configuration = WorkbenchInspectionAdapter.configuration_payload
        original_search = WorkbenchInspectionAdapter.search_space_payload

        def inspect_payload(adapter, request):  # type: ignore[no-untyped-def]
            graph_calls.append(adapter.package.catalog_key)
            return original_graph(adapter, request)

        def inspect(adapter, request):  # type: ignore[no-untyped-def]
            semantic_graph_calls.append(adapter.package.catalog_key)
            return original_semantic_graph(adapter, request)

        def configuration_payload(  # type: ignore[no-untyped-def]
            adapter,
            preset,
        ):
            configuration_calls.append(adapter.package.catalog_key)
            return original_configuration(adapter, preset)

        def search_payload(adapter, preset, presets=None):  # type: ignore[no-untyped-def]
            search_calls.append(adapter.package.catalog_key)
            return original_search(adapter, preset, presets)

        with (
            patch.object(
                WorkbenchInspectionAdapter,
                "inspect_payload",
                inspect_payload,
            ),
            patch.object(
                WorkbenchInspectionAdapter,
                "inspect",
                inspect,
            ),
            patch.object(
                WorkbenchInspectionAdapter,
                "configuration_payload",
                configuration_payload,
            ),
            patch.object(
                WorkbenchInspectionAdapter,
                "search_space_payload",
                search_payload,
            ),
        ):
            legacy_inspect_model(
                "linears/linear",
                "baseline",
                dataset="Mnist",
            )
            legacy_config_schema("linears/linear", "baseline")
            legacy_search_space_schema("linears/linear", "baseline")
            InspectionService().inspect(
                model_type="linears",
                model="linear",
                preset="baseline",
                overrides={},
                dataset="Mnist",
            )
            http_config_schema("linears", "linear", "baseline")
            http_search_space("linears", "linear", "baseline", None)

        self.assertEqual(graph_calls, ["linears/linear"])
        self.assertEqual(
            semantic_graph_calls,
            ["linears/linear", "linears/linear"],
        )
        self.assertEqual(
            configuration_calls,
            ["linears/linear", "linears/linear"],
        )
        self.assertEqual(search_calls, ["linears/linear", "linears/linear"])


if __name__ == "__main__":
    unittest.main()
