from __future__ import annotations

import copy
import math
import random
import unittest
from contextlib import redirect_stderr
from io import StringIO

from model_runtime.cli import (
    PROTOCOL_VERSION,
    WireCodecError,
    configuration_schema_from_wire,
    configuration_schema_to_wire,
    inspection_result_from_wire,
    inspection_result_to_wire,
    json_value_to_wire,
    package_metadata_from_wire,
    package_metadata_to_wire,
    planning_budget_from_wire,
    planning_budget_to_wire,
    random_state_from_wire,
    random_state_to_wire,
    run_plan_from_wire,
    run_plan_to_wire,
    run_request_from_wire,
    run_request_to_wire,
    run_result_from_wire,
    run_result_to_wire,
    search_space_from_wire,
    search_space_to_wire,
    submitted_run_from_wire,
    submitted_run_to_wire,
    to_wire,
)
from model_runtime.inspection import (
    ConfigurationField,
    ConfigurationSchema,
    GraphConfiguration,
    GraphConfigurationField,
    GraphEdge,
    GraphNode,
    InspectionResult,
    SearchAxis,
    SearchSpace,
)
from model_runtime.packages import ModelIdentity
from model_runtime.runs import (
    PlanningBudget,
    RunParameter,
    RunPlan,
    RunRequest,
    RunResult,
    RunSpec,
    SearchAxisSelection,
    SearchSpec,
    SubmittedRun,
)
from models.adapter_cli import AdapterProtocolError, _response, process_request
from models.catalog import model_package


def _identity() -> ModelIdentity:
    return ModelIdentity("linears", "linear")


def _configuration_schema() -> ConfigurationSchema:
    return ConfigurationSchema(
        identity=_identity(),
        fields=(
            ConfigurationField(
                key="HIDDEN_DIM",
                flag="--hidden-dim",
                section_path=("model", "shape"),
                description="Hidden width.",
                value_type="int",
                default=128,
                nullable=False,
                choices=(64, 128),
                maximum=4096,
                locked=True,
                locked_value=128,
                locked_reason="Baseline lock.",
            ),
        ),
    )


def _search_space() -> SearchSpace:
    return SearchSpace(
        identity=_identity(),
        preset="baseline",
        axes=(
            SearchAxis(
                key="hidden_dim",
                search_key="SEARCH_SPACE_HIDDEN_DIM",
                section="model",
                value_type="int",
                values=(64, 128),
                locked=False,
                locked_by_presets=("gating",),
                lock_reasons=("Gating lock.",),
            ),
        ),
    )


def _inspection_result() -> InspectionResult:
    return InspectionResult(
        identity=_identity(),
        preset="baseline",
        parameter_count=42,
        parameter_size_bytes=168,
        nodes=(
            GraphNode(
                id="model.layer",
                type_name="Linear",
                description="Projection.",
                path="layer",
                graph_role="architecture",
                parameter_count=42,
                parameter_size_bytes=168,
                details={"shape": [6, 7], "trainable": True},
                configuration=GraphConfiguration(
                    type_name="LinearConfig",
                    fields=(
                        GraphConfigurationField(
                            key="activation",
                            value={"name": "ReLU"},
                            description=None,
                        ),
                    ),
                ),
            ),
        ),
        edges=(GraphEdge(id="root-layer", source="model", target="model.layer"),),
    )


def _run_request() -> RunRequest:
    return RunRequest(
        presets=("baseline",),
        datasets=("Mnist",),
        experiment_task="image-classification",
        overrides={"NUM_EPOCHS": 3},
        search=SearchSpec(
            mode="random",
            axes=(
                SearchAxisSelection(
                    key="hidden_dim",
                    values=(64, 128),
                    allow_custom_values=True,
                ),
            ),
            random_samples=2,
        ),
    )


def _run_plan() -> RunPlan:
    request = _run_request()
    return RunPlan(
        identity=_identity(),
        presets=request.presets,
        experiment_task="image-classification",
        datasets=request.datasets,
        overrides=request.overrides,
        search=request.search,
        runs=(
            RunSpec(
                id="run-0001",
                experiment_task="image-classification",
                preset="baseline",
                dataset="Mnist",
                parameters=(
                    RunParameter(key="HIDDEN_DIM", value=128, source="search"),
                ),
            ),
        ),
    )


class CliWireRoundTripTests(unittest.TestCase):
    def test_adapter_envelope_requires_exact_version_and_payload_types(self) -> None:
        for request in (
            {"version": True, "operation": "catalog", "payload": {}},
            {"version": PROTOCOL_VERSION, "operation": "catalog"},
            {"version": PROTOCOL_VERSION, "operation": "catalog", "payload": []},
        ):
            with self.subTest(request=request), self.assertRaises(AdapterProtocolError):
                process_request(request)

        with redirect_stderr(StringIO()):
            response = _response(
                b'{"version":1,"operation":"catalog","payload":{"value":NaN}}'
            )
        self.assertIs(response["ok"], False)
        self.assertEqual(response["error"]["kind"], "invalid")

        parsed = process_request(
            {
                "version": PROTOCOL_VERSION,
                "operation": "parse_search_value",
                "payload": {
                    "model_id": "linears/linear",
                    "search_key": "SEARCH_SPACE_HIDDEN_DIM",
                    "value": 64,
                },
            }
        )
        self.assertEqual(parsed["result"], 64)

    def test_protocol_version_and_package_metadata_roundtrip(self) -> None:
        self.assertEqual(PROTOCOL_VERSION, 1)
        package = model_package("linears/linear")
        self.assertIsNotNone(package)

        payload = package_metadata_to_wire(package)

        self.assertEqual(package_metadata_from_wire(payload), payload)
        self.assertEqual(payload["catalog_key"], "linears/linear")

    def test_inspection_records_roundtrip_without_field_name_changes(self) -> None:
        schema = _configuration_schema()
        search_space = _search_space()
        result = _inspection_result()

        schema_payload = configuration_schema_to_wire(schema)
        search_payload = search_space_to_wire(search_space)
        result_payload = inspection_result_to_wire(result)

        self.assertEqual(configuration_schema_from_wire(schema_payload), schema)
        self.assertEqual(search_space_from_wire(search_payload), search_space)
        self.assertEqual(inspection_result_from_wire(result_payload), result)
        self.assertEqual(
            set(schema_payload["fields"][0]),
            {
                "key",
                "flag",
                "section_path",
                "description",
                "value_type",
                "default",
                "nullable",
                "choices",
                "maximum",
                "locked",
                "locked_value",
                "locked_reason",
            },
        )

    def test_runs_records_budgets_and_random_state_roundtrip(self) -> None:
        request = _run_request()
        budget = PlanningBudget(
            max_axes=3,
            max_values_per_axis=8,
            max_materialized_runs=20,
        )
        submitted = SubmittedRun(
            id="run-0001",
            preset="baseline",
            dataset="Mnist",
            overrides={"HIDDEN_DIM": 128},
        )
        plan = _run_plan()
        result = RunResult(
            run_id="run-0001",
            experiment_task="image-classification",
            preset="baseline",
            dataset="Mnist",
            log_dir="logs/run/version_0",
            payload={"metrics": {"validation_accuracy": 0.75}},
        )
        random_source = random.Random(7)

        self.assertEqual(run_request_from_wire(run_request_to_wire(request)), request)
        self.assertEqual(
            planning_budget_from_wire(planning_budget_to_wire(budget)),
            budget,
        )
        self.assertEqual(
            submitted_run_from_wire(submitted_run_to_wire(submitted)),
            submitted,
        )
        self.assertEqual(run_plan_from_wire(run_plan_to_wire(plan)), plan)
        self.assertEqual(run_result_from_wire(run_result_to_wire(result)), result)
        self.assertEqual(
            random_state_from_wire(random_state_to_wire(random_source.getstate())),
            random_source.getstate(),
        )

    def test_malformed_record_fields_fail_at_the_codec_seam(self) -> None:
        schema_payload = configuration_schema_to_wire(_configuration_schema())
        result_payload = inspection_result_to_wire(_inspection_result())
        request_payload = run_request_to_wire(_run_request())
        plan_payload = run_plan_to_wire(_run_plan())
        budget_payload = planning_budget_to_wire(PlanningBudget(max_axes=2))
        package_payload = package_metadata_to_wire(model_package("linears/linear"))

        malformed: tuple[tuple[object, object, str], ...] = (
            (
                configuration_schema_from_wire,
                {**schema_payload, "fields": "not-a-list"},
                "fields must be a list",
            ),
            (
                configuration_schema_from_wire,
                {
                    **schema_payload,
                    "fields": [{**schema_payload["fields"][0], "nullable": 1}],
                },
                "nullable must be a boolean",
            ),
            (
                inspection_result_from_wire,
                {
                    **result_payload,
                    "nodes": [{**result_payload["nodes"][0], "graph_role": "display"}],
                },
                "graph_role must be one of",
            ),
            (
                inspection_result_from_wire,
                {**result_payload, "parameter_count": True},
                "parameter_count must be an integer",
            ),
            (
                run_request_from_wire,
                {**request_payload, "presets": "baseline"},
                "presets must be a list",
            ),
            (
                run_request_from_wire,
                {
                    **request_payload,
                    "search": {**request_payload["search"], "mode": "exhaustive"},
                },
                "search.mode must be one of",
            ),
            (
                run_plan_from_wire,
                {
                    **plan_payload,
                    "runs": [
                        {
                            **plan_payload["runs"][0],
                            "parameters": [
                                {
                                    **plan_payload["runs"][0]["parameters"][0],
                                    "source": "manual",
                                }
                            ],
                        }
                    ],
                },
                "source must be one of",
            ),
            (
                planning_budget_from_wire,
                {**budget_payload, "max_axes": True},
                "max_axes must be an integer",
            ),
            (
                package_metadata_from_wire,
                {**package_payload, "default_experiment_task": "classification"},
                "not a supported Experiment Task",
            ),
            (
                package_metadata_from_wire,
                {**package_payload, "runtime_defaults": {"VALUE": [1]}},
                "runtime_defaults.VALUE must be a JSON scalar",
            ),
        )
        for decoder, payload, message in malformed:
            with (
                self.subTest(message=message),
                self.assertRaisesRegex(
                    WireCodecError,
                    message,
                ),
            ):
                decoder(payload)

    def test_json_projection_rejects_non_finite_and_arbitrary_objects(self) -> None:
        for value in (
            math.nan,
            {"loss": math.inf},
            {1: "non-string-key"},
            {None: "non-string-key"},
            {"unsupported"},
            object(),
        ):
            with (
                self.subTest(value=type(value).__name__),
                self.assertRaises(WireCodecError),
            ):
                json_value_to_wire(value)

        with self.assertRaises(WireCodecError):
            to_wire(object())

    def test_unknown_and_missing_fields_are_rejected_deterministically(self) -> None:
        payload = run_plan_to_wire(_run_plan())
        unknown = {**payload, "extra": True}
        missing = copy.deepcopy(payload)
        del missing["runs"]

        with self.assertRaisesRegex(WireCodecError, "unknown field 'extra'"):
            run_plan_from_wire(unknown)
        with self.assertRaisesRegex(WireCodecError, "missing required field 'runs'"):
            run_plan_from_wire(missing)


if __name__ == "__main__":
    unittest.main()
