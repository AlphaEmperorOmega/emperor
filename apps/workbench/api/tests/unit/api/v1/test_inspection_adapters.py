from __future__ import annotations

import os
import unittest
from dataclasses import FrozenInstanceError
from unittest.mock import Mock

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from model_runtime.inspection import InspectionResult

from emperor_workbench.api.v1.inspection._mapping import inspection_response
from emperor_workbench.api.v1.model_packages._mapping import (
    config_schema_response,
    search_space_response,
)
from emperor_workbench.api.v1.model_packages._routes import (
    _config_schema as http_config_schema,
)
from emperor_workbench.api.v1.model_packages._routes import (
    _search_space as http_search_space,
)
from emperor_workbench.inspection import (
    InProcessInspectionExecutor,
    InspectionService,
)
from emperor_workbench.model_packages import (
    ModelPackageCatalog,
    ModelPackageFailure,
    SelectedModelPackage,
)
from emperor_workbench.project_adapter import (
    ModelPackageReference,
    ProjectAdapterClient,
    ProjectAdapterFailure,
)
from tests.support.inspection import config_schema as support_config_schema
from tests.support.inspection import inspect_model as support_inspect_model
from tests.support.inspection import (
    search_space_schema as support_search_space_schema,
)
from tests.support.model_packages import project_adapter_client


def _metadata_payload() -> dict[str, object]:
    return {
        "default_experiment_task": "image-classification",
        "presets": [
            {
                "name": "baseline",
                "label": "BASELINE",
                "description": "Baseline preset.",
            }
        ],
        "dataset_groups": [
            {
                "experiment_task": "image-classification",
                "label": "Image classification",
                "datasets": [
                    {
                        "name": "Mnist",
                        "label": "MNIST",
                        "input_dim": 784,
                        "output_dim": 10,
                    }
                ],
            }
        ],
        "monitors": [
            {
                "name": "activation",
                "label": "Activation",
                "description": "Activation statistics.",
                "kinds": ["scalar"],
                "defaultEnabled": False,
            }
        ],
        "runtime_defaults": {
            "HIDDEN_DIM": 128,
            "NESTED": {
                "axes": [
                    1,
                    {
                        "enabled": True,
                    },
                ]
            },
        },
    }


def _selected_with_metadata(payload: object) -> SelectedModelPackage:
    client = Mock(spec=ProjectAdapterClient)
    client._package_metadata.return_value = payload
    return SelectedModelPackage(ModelPackageReference("linears", "linear", client))


class InspectionAdapterEquivalenceTests(unittest.TestCase):
    def catalog(self) -> ModelPackageCatalog:
        return ModelPackageCatalog(project_adapter_client())

    def test_inspection_service_returns_transport_neutral_result(self) -> None:
        selected = self.catalog().select("linears/linear")
        result = InspectionService(InProcessInspectionExecutor()).inspect(
            selected,
            preset="baseline",
            overrides={},
            dataset="Mnist",
        )

        self.assertIsInstance(result, InspectionResult)

    def test_broken_package_failures_map_to_model_package_failure(self) -> None:
        client = Mock(spec=ProjectAdapterClient)
        client.command = ("broken-adapter",)
        failure = ProjectAdapterFailure(
            "Failed to import model package 'broken/missing': missing module"
        )
        client.configuration.side_effect = failure
        client._package_metadata.side_effect = failure
        selected = SelectedModelPackage(
            ModelPackageReference("broken", "missing", client)
        )

        for call in (selected.configuration, selected.metadata):
            with self.subTest(call=call):
                with self.assertRaises(ModelPackageFailure) as raised:
                    call()
                self.assertIn(
                    "Failed to import model package 'broken/missing'",
                    raised.exception.detail,
                )

    def test_semantic_inspection_matches_the_frozen_http_projection(self) -> None:
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
                result = InspectionService(InProcessInspectionExecutor()).inspect(
                    self.catalog().select(model_id),
                    preset=preset,
                    overrides={},
                    dataset=dataset,
                    experiment_task=experiment_task,
                )
                expected = inspection_response(result).model_dump(mode="json")
                self.assertEqual(
                    expected,
                    support_inspect_model(
                        model_id,
                        preset,
                        dataset=dataset,
                        experiment_task=experiment_task,
                    ),
                )

    def test_model_package_schema_and_search_match_frozen_helpers(self) -> None:
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
                selected = self.catalog().select(model_id)
                self.assertEqual(
                    config_schema_response(selected.configuration(preset)).model_dump(),
                    support_config_schema(model_id, preset),
                )
                self.assertEqual(
                    search_space_response(
                        selected.search_space(preset, presets)
                    ).model_dump(),
                    support_search_space_schema(model_id, preset, presets),
                )

    def test_model_package_metadata_is_frozen_and_semantic(self) -> None:
        selected = self.catalog().select("linears/linear")
        metadata = selected.metadata()

        self.assertTrue(metadata.presets)
        self.assertTrue(metadata.dataset_groups)
        self.assertTrue(metadata.monitors)
        self.assertEqual(
            metadata.default_experiment_task,
            "image-classification",
        )
        with self.assertRaises(FrozenInstanceError):
            metadata.default_experiment_task = "changed"  # type: ignore[misc]
        with self.assertRaises(TypeError):
            metadata.runtime_defaults["HIDDEN_DIM"] = 64  # type: ignore[index]

    def test_model_package_runtime_defaults_are_recursively_immutable(self) -> None:
        payload = _metadata_payload()
        metadata = _selected_with_metadata(payload).metadata()
        nested = metadata.runtime_defaults["NESTED"]
        axes = nested["axes"]
        enabled = axes[1]

        with self.assertRaises(TypeError):
            nested["axes"] = ()  # type: ignore[index]
        with self.assertRaises(AttributeError):
            axes.append(2)
        with self.assertRaises(TypeError):
            enabled["enabled"] = False  # type: ignore[index]

        payload_defaults = payload["runtime_defaults"]
        assert isinstance(payload_defaults, dict)
        payload_nested = payload_defaults["NESTED"]
        assert isinstance(payload_nested, dict)
        payload_axes = payload_nested["axes"]
        assert isinstance(payload_axes, list)
        payload_enabled = payload_axes[1]
        assert isinstance(payload_enabled, dict)
        payload_enabled["enabled"] = False
        self.assertTrue(metadata.runtime_defaults["NESTED"]["axes"][1]["enabled"])

    def test_malformed_model_package_metadata_maps_to_domain_failure(self) -> None:
        cases: list[tuple[str, object]] = []

        payload = _metadata_payload()
        payload["presets"] = ["not-an-object"]
        cases.append(("preset entry", payload))

        payload = _metadata_payload()
        presets = payload["presets"]
        assert isinstance(presets, list)
        preset = presets[0]
        assert isinstance(preset, dict)
        preset["name"] = 123
        cases.append(("preset name", payload))

        payload = _metadata_payload()
        groups = payload["dataset_groups"]
        assert isinstance(groups, list)
        group = groups[0]
        assert isinstance(group, dict)
        datasets = group["datasets"]
        assert isinstance(datasets, list)
        dataset = datasets[0]
        assert isinstance(dataset, dict)
        dataset["input_dim"] = True
        cases.append(("dataset dimension", payload))

        payload = _metadata_payload()
        monitors = payload["monitors"]
        assert isinstance(monitors, list)
        monitor = monitors[0]
        assert isinstance(monitor, dict)
        monitor["defaultEnabled"] = "false"
        cases.append(("monitor default", payload))

        payload = _metadata_payload()
        del payload["default_experiment_task"]
        cases.append(("missing task", payload))

        payload = _metadata_payload()
        runtime_defaults = payload["runtime_defaults"]
        assert isinstance(runtime_defaults, dict)
        runtime_defaults["INVALID"] = object()
        cases.append(("runtime default", payload))

        for name, malformed in cases:
            with self.subTest(name=name):
                with self.assertRaises(ModelPackageFailure) as raised:
                    _selected_with_metadata(malformed).metadata()
                self.assertIn(
                    "invalid Model Package metadata",
                    raised.exception.detail,
                )

    def test_malformed_model_package_mapping_results_map_to_domain_failure(
        self,
    ) -> None:
        client = Mock(spec=ProjectAdapterClient)
        selected = SelectedModelPackage(
            ModelPackageReference("linears", "linear", client)
        )

        for name, call in (
            ("preset locks", lambda: selected.preset_locks("baseline")),
            (
                "serialized overrides",
                lambda: selected.serialize_overrides({"HIDDEN_DIM": 64}),
            ),
        ):
            for malformed in (None, [], [("HIDDEN_DIM", 64)], {1: 64}):
                with self.subTest(name=name, malformed=malformed):
                    client.call.return_value = malformed
                    with self.assertRaises(ModelPackageFailure) as raised:
                        call()
                    self.assertIn(
                        f"invalid {name}",
                        raised.exception.detail,
                    )

    def test_http_helpers_consume_the_model_packages_interface(self) -> None:
        catalog = self.catalog()
        configuration = http_config_schema(
            catalog,
            "linears",
            "linear",
            "baseline",
        )
        search = http_search_space(
            catalog,
            "linears",
            "linear",
            "baseline",
            None,
        )

        self.assertEqual(
            config_schema_response(configuration).model_dump(),
            support_config_schema("linears/linear", "baseline"),
        )
        self.assertEqual(
            search_space_response(search).model_dump(),
            support_search_space_schema("linears/linear", "baseline"),
        )


if __name__ == "__main__":
    unittest.main()
