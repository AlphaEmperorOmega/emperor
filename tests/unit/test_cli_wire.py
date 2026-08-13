from __future__ import annotations

import copy
import json
import math
import random
import unittest
from contextlib import redirect_stderr
from dataclasses import replace
from io import StringIO
from unittest.mock import patch

import model_runtime.cli as cli_facade
import model_runtime.cli.wire as wire_facade
from model_runtime.cli import (
    PROTOCOL_VERSION,
    WireCodecError,
    configuration_schema_from_wire,
    configuration_schema_to_wire,
    identity_from_wire,
    inspection_result_from_wire,
    inspection_result_to_wire,
    json_value_from_wire,
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
    search_spec_from_wire,
    search_spec_to_wire,
    submitted_run_from_wire,
    submitted_run_to_wire,
)
from model_runtime.cli._wire_graph import InspectionWireLimits
from model_runtime.inspection import (
    ConfigurationField,
    ConfigurationFieldCondition,
    ConfigurationSchema,
    GraphConfiguration,
    GraphConfigurationField,
    GraphEdge,
    GraphNode,
    InspectionCaptureLimits,
    InspectionResult,
    SearchAxis,
    SearchSpace,
)
from model_runtime.packages import ModelIdentity
from model_runtime.runs import (
    PlanningBudget,
    PresetSearch,
    RunParameter,
    RunPlan,
    RunPlanExecutionError,
    RunRequest,
    RunResult,
    RunSpec,
    SearchAxisSelection,
    SearchSpec,
    SubmittedRun,
)
from models.adapter_cli import AdapterProtocolError, _response, process_request
from models.catalog import MODEL_CATALOG, model_package


def _identity() -> ModelIdentity:
    return ModelIdentity("linears", "linear")


def _nested_json_list(container_depth: int):
    value = 0
    for _ in range(container_depth):
        value = [value]
    return value


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
                applicable_when=(
                    ConfigurationFieldCondition(
                        key="MODEL_OPTION",
                        values=("StandardConfig", "AlternateConfig"),
                    ),
                ),
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
                id="__root__",
                type_name="Model",
                description="Root model.",
                path="model",
                graph_role="architecture",
                parameter_count=42,
                parameter_size_bytes=168,
                details={},
                configuration=None,
            ),
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
        edges=(
            GraphEdge(
                id="root-layer",
                source="__root__",
                target="model.layer",
            ),
        ),
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

    def test_adapter_error_preserves_structured_partial_plan_outcome(self) -> None:
        completed = RunResult(
            run_id="run-0001",
            experiment_task="image-classification",
            preset="baseline",
            dataset="Mnist",
            log_dir="logs/run/version_0",
            payload={"status": "completed", "artifactId": "attempt-a"},
        )
        error = RunPlanExecutionError(
            completed_results=(completed,),
            affected_run_id="run-" + "x" * 1_024,
            phase="training",
            execution_id="execution-a",
        )
        error.__cause__ = RuntimeError("fit failed")
        request = json.dumps(
            {"version": PROTOCOL_VERSION, "operation": "catalog", "payload": {}}
        ).encode()

        with (
            patch("models.adapter_cli._handle", side_effect=error),
            redirect_stderr(StringIO()),
        ):
            response = _response(request)

        self.assertIs(response["ok"], False)
        self.assertEqual(response["error"]["kind"], "unavailable")
        self.assertEqual(response["error"]["phase"], "training")
        self.assertEqual(
            response["error"]["affected_run_id"],
            "run-" + "x" * 1_024,
        )
        self.assertEqual(response["error"]["execution_id"], "execution-a")
        self.assertEqual(
            response["error"]["completed_results"],
            [run_result_to_wire(completed)],
        )

    def test_run_plan_wire_preserves_per_preset_search_provenance(self) -> None:
        baseline_search = SearchSpec(
            mode="random",
            axes=(SearchAxisSelection("HIDDEN_DIM", (64, 128)),),
            random_samples=1,
        )
        post_norm_search = SearchSpec(
            mode="random",
            axes=(SearchAxisSelection("STACK_ACTIVATION", ("RELU",)),),
            random_samples=1,
        )
        plan = RunPlan(
            identity=_identity(),
            presets=("baseline", "post-norm"),
            experiment_task="image-classification",
            datasets=("Mnist",),
            overrides={},
            search=None,
            runs=(),
            preset_searches=(
                PresetSearch("baseline", baseline_search),
                PresetSearch("post-norm", post_norm_search),
            ),
        )

        payload = run_plan_to_wire(plan)
        restored = run_plan_from_wire(payload)

        self.assertEqual(restored, plan)
        self.assertEqual(
            [entry["preset"] for entry in payload["preset_searches"]],
            ["baseline", "post-norm"],
        )
        self.assertEqual(
            restored.search_for_preset("post-norm"),
            post_norm_search,
        )

        legacy_payload = dict(payload)
        del legacy_payload["preset_searches"]
        legacy = run_plan_from_wire(legacy_payload)
        self.assertEqual(
            legacy.preset_searches,
            (
                PresetSearch("baseline", baseline_search),
                PresetSearch("post-norm", baseline_search),
            ),
        )

    def test_adapter_forwards_memory_limit_to_inspection_and_validation(self) -> None:
        captured_requests = []

        def capture_inspection(_package, request):
            captured_requests.append(request)
            return _inspection_result()

        def capture_validation(_package, request):
            captured_requests.append(request)

        payload = {
            "model_id": "linears/linear",
            "preset": "baseline",
            "overrides": {},
            "dataset": "Mnist",
            "experiment_task": "image-classification",
            "memory_limit_bytes": 768 * 1024**2,
        }
        with (
            patch("models.adapter_cli.inspect_model", side_effect=capture_inspection),
            patch(
                "models.adapter_cli.validate_configuration",
                side_effect=capture_validation,
            ),
        ):
            process_request(
                {
                    "version": PROTOCOL_VERSION,
                    "operation": "inspect",
                    "payload": payload,
                }
            )
            process_request(
                {
                    "version": PROTOCOL_VERSION,
                    "operation": "validate",
                    "payload": payload,
                }
            )

        self.assertEqual(
            [request.memory_limit_bytes for request in captured_requests],
            [768 * 1024**2, 768 * 1024**2],
        )
        for invalid in (True, 0, -1):
            with (
                self.subTest(invalid=invalid),
                self.assertRaisesRegex(
                    AdapterProtocolError,
                    "memory_limit_bytes must be a positive integer",
                ),
            ):
                process_request(
                    {
                        "version": PROTOCOL_VERSION,
                        "operation": "validate",
                        "payload": {
                            **payload,
                            "memory_limit_bytes": invalid,
                        },
                    }
                )

    def test_protocol_version_and_package_metadata_roundtrip(self) -> None:
        self.assertEqual(PROTOCOL_VERSION, 1)
        package = model_package("linears/linear")
        self.assertIsNotNone(package)

        payload = package_metadata_to_wire(package)

        self.assertEqual(package_metadata_from_wire(payload), payload)
        self.assertEqual(payload["catalog_key"], "linears/linear")

    def test_package_metadata_uses_authoritative_runtime_defaults_fields(self) -> None:
        for catalog_key, package in MODEL_CATALOG.items():
            with self.subTest(model=catalog_key):
                payload = package_metadata_to_wire(package)
                expected_keys = tuple(
                    key for key, _value in package.runtime_defaults_spec.default_items()
                )

                self.assertEqual(
                    tuple(payload["runtime_defaults"]),
                    expected_keys,
                )

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
                "applicableWhen",
                "maximum",
                "locked",
                "locked_value",
                "locked_reason",
            },
        )
        self.assertEqual(
            schema_payload["fields"][0]["applicableWhen"],
            [
                {
                    "key": "MODEL_OPTION",
                    "values": ["StandardConfig", "AlternateConfig"],
                }
            ],
        )

    def test_cli_facades_expose_only_explicit_wire_operations(self) -> None:
        for facade in (cli_facade, wire_facade):
            with self.subTest(facade=facade.__name__):
                self.assertNotIn("to_wire", facade.__all__)
                self.assertFalse(hasattr(facade, "to_wire"))
                for operation in (
                    "identity_to_wire",
                    "inspection_result_to_wire",
                    "planning_budget_to_wire",
                    "run_plan_to_wire",
                    "run_results_to_wire",
                    "submitted_runs_to_wire",
                    "json_value_to_wire",
                ):
                    self.assertTrue(callable(getattr(facade, operation)))

    def test_schema_and_search_decoder_validation_precedence_is_stable(self) -> None:
        schema_payload = configuration_schema_to_wire(_configuration_schema())
        schema_payload["identity"]["model_type"] = 1
        schema_payload["fields"][0]["key"] = 1
        schema_payload["fields"][0]["applicableWhen"][0]["key"] = 1
        with self.assertRaisesRegex(
            WireCodecError,
            r"\$\.fields\[0\]\.applicableWhen\[0\]\.key must be a string",
        ):
            configuration_schema_from_wire(schema_payload)

        search_payload = search_space_to_wire(_search_space())
        search_payload["identity"]["model_type"] = 1
        search_payload["preset"] = 1
        search_payload["axes"][0]["key"] = 1
        search_payload["axes"][0]["search_key"] = 1
        with self.assertRaisesRegex(
            WireCodecError,
            r"\$\.axes\[0\]\.key must be a string",
        ):
            search_space_from_wire(search_payload)

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

    def test_omitted_wire_budget_fields_keep_safe_defaults(self) -> None:
        self.assertEqual(planning_budget_from_wire({}), PlanningBudget())

    def test_wire_budget_decode_is_complete_transport_acceptance(self) -> None:
        self.assertEqual(
            planning_budget_from_wire(
                {
                    "max_axes": None,
                    "max_values_per_axis": None,
                    "max_materialized_runs": 2_000,
                }
            ),
            PlanningBudget(
                max_axes=None,
                max_values_per_axis=None,
                max_materialized_runs=2_000,
            ),
        )

        for payload in (
            {
                "max_axes": None,
                "max_values_per_axis": None,
                "max_materialized_runs": None,
            },
            {"max_materialized_runs": 2_001},
        ):
            with (
                self.subTest(payload=payload),
                self.assertRaisesRegex(
                    WireCodecError,
                    "transport requires max_materialized_runs at most 2000",
                ),
            ):
                planning_budget_from_wire(payload)

        with self.assertRaisesRegex(
            WireCodecError,
            "transport requires max_materialized_runs at most 2000",
        ):
            planning_budget_to_wire(PlanningBudget.unlimited())

    def test_wire_budget_shape_errors_precede_transport_limits(self) -> None:
        with self.assertRaisesRegex(
            WireCodecError,
            r"\$\.max_axes must be an integer",
        ):
            planning_budget_from_wire(
                {
                    "max_axes": True,
                    "max_materialized_runs": None,
                }
            )

    def test_identity_decoder_translates_domain_syntax_errors(self) -> None:
        with self.assertRaisesRegex(
            WireCodecError,
            "Invalid model identity",
        ) as caught:
            identity_from_wire({"model_type": "../linears", "model": "linear"})
        self.assertIsInstance(caught.exception.__cause__, ValueError)

        with self.assertRaisesRegex(
            WireCodecError,
            r"\$\.identity\.model must be a string",
        ):
            identity_from_wire({"model_type": "../linears", "model": 1})

    def test_adapter_rejects_untransportable_budget_before_planning(self) -> None:
        request_payload = run_request_to_wire(_run_request())

        for operation, extra_payload in (
            ("plan_runs", {}),
            (
                "accept_run_plan",
                {
                    "runs": [
                        {
                            "id": "run-1",
                            "preset": "baseline",
                            "dataset": "Mnist",
                            "overrides": {},
                        }
                    ]
                },
            ),
        ):
            with (
                self.subTest(operation=operation),
                patch(f"models.adapter_cli.{operation}") as planner,
                self.assertRaisesRegex(
                    WireCodecError,
                    "transport requires max_materialized_runs at most 2000",
                ),
            ):
                process_request(
                    {
                        "version": PROTOCOL_VERSION,
                        "operation": operation,
                        "payload": {
                            "model_id": "linears/linear",
                            "request": request_payload,
                            "budget": {
                                "max_axes": None,
                                "max_values_per_axis": None,
                                "max_materialized_runs": None,
                            },
                            **extra_payload,
                        },
                    }
                )
            planner.assert_not_called()

    def test_run_plan_wire_rejects_oversized_collections_before_item_decode(
        self,
    ) -> None:
        payload = run_plan_to_wire(_run_plan())
        with self.assertRaisesRegex(
            WireCodecError,
            r"\$\.runs must be a list",
        ):
            run_plan_from_wire({**payload, "runs": (None,) * 2_001})

        with self.assertRaisesRegex(
            WireCodecError,
            r"\$\.runs must contain at most 2000 items",
        ):
            run_plan_from_wire({**payload, "runs": [None] * 2_001})

        oversized_parameters = copy.deepcopy(payload)
        oversized_parameters["runs"][0]["parameters"] = [None] * 1_025
        with self.assertRaisesRegex(
            WireCodecError,
            r"\$\.runs\[0\]\.parameters must contain at most 1024 items",
        ):
            run_plan_from_wire(oversized_parameters)

    def test_encode_sequence_limits_preserve_exact_paths(self) -> None:
        request = replace(_run_request(), presets=("baseline",) * 2_001)
        search = SearchSpec(
            mode="grid",
            axes=(SearchAxisSelection("HIDDEN_DIM", (64,)),) * 17,
        )
        inspection = _inspection_result()
        oversized_graph = replace(
            inspection,
            nodes=(inspection.nodes[0],) * 8_193,
        )
        cases = (
            (
                run_request_to_wire,
                request,
                r"\$\.presets must contain at most 2000 items",
            ),
            (
                search_spec_to_wire,
                search,
                r"\$\.search\.axes must contain at most 16 items",
            ),
            (
                inspection_result_to_wire,
                oversized_graph,
                r"\$\.nodes must contain at most 8192 items",
            ),
        )

        for encoder, value, message in cases:
            with (
                self.subTest(encoder=encoder.__name__),
                self.assertRaisesRegex(WireCodecError, message),
            ):
                encoder(value)

    def test_experiment_task_wire_errors_preserve_consumer_paths(self) -> None:
        request_payload = run_request_to_wire(_run_request())
        package_payload = package_metadata_to_wire(model_package("linears/linear"))
        cases = (
            (
                run_request_from_wire,
                {**request_payload, "experiment_task": "unsupported"},
                r"\$\.experiment_task is not a supported Experiment Task",
            ),
            (
                package_metadata_from_wire,
                {**package_payload, "default_experiment_task": "unsupported"},
                r"\$\.default_experiment_task is not a supported Experiment Task",
            ),
        )

        for decoder, payload, message in cases:
            with (
                self.subTest(decoder=decoder.__name__),
                self.assertRaisesRegex(WireCodecError, message),
            ):
                decoder(payload)

    def test_package_metadata_decoder_validation_precedence_is_stable(
        self,
    ) -> None:
        payload = package_metadata_to_wire(model_package("linears/linear"))

        invalid_preset_and_dataset = copy.deepcopy(payload)
        invalid_preset_and_dataset["presets"][0]["name"] = 1
        invalid_preset_and_dataset["dataset_groups"][0]["datasets"][0]["name"] = 1
        with self.assertRaisesRegex(
            WireCodecError,
            r"\$\.presets\[0\]\.name must be a string",
        ):
            package_metadata_from_wire(invalid_preset_and_dataset)

        invalid_dataset_and_group = copy.deepcopy(payload)
        invalid_dataset_and_group["dataset_groups"][0]["datasets"][0]["input_dim"] = (
            True
        )
        invalid_dataset_and_group["dataset_groups"][0]["experiment_task"] = (
            "unsupported"
        )
        with self.assertRaisesRegex(
            WireCodecError,
            r"\$\.dataset_groups\[0\]\.datasets\[0\]\.input_dim must be an integer",
        ):
            package_metadata_from_wire(invalid_dataset_and_group)

        invalid_monitor_and_default = copy.deepcopy(payload)
        invalid_monitor_and_default["monitors"][0]["defaultEnabled"] = 1
        invalid_monitor_and_default["default_experiment_task"] = "unsupported"
        with self.assertRaisesRegex(
            WireCodecError,
            r"\$\.monitors\[0\]\.defaultEnabled must be a boolean",
        ):
            package_metadata_from_wire(invalid_monitor_and_default)

        invalid_default_and_runtime_value = copy.deepcopy(payload)
        invalid_default_and_runtime_value["default_experiment_task"] = "unsupported"
        invalid_default_and_runtime_value["runtime_defaults"] = {"VALUE": [1]}
        with self.assertRaisesRegex(
            WireCodecError,
            r"\$\.default_experiment_task is not a supported Experiment Task",
        ):
            package_metadata_from_wire(invalid_default_and_runtime_value)

    def test_run_plan_decoder_validation_precedence_is_stable(self) -> None:
        payload = run_plan_to_wire(_run_plan())

        invalid_parameter_value_and_source = copy.deepcopy(payload)
        parameter = invalid_parameter_value_and_source["runs"][0]["parameters"][0]
        parameter["value"] = {1: "invalid-key"}
        parameter["source"] = "unsupported"
        with self.assertRaisesRegex(
            WireCodecError,
            r"\$\.runs\[0\]\.parameters\[0\]\.value object keys must be strings",
        ):
            run_plan_from_wire(invalid_parameter_value_and_source)

        invalid_parameter_and_run = copy.deepcopy(payload)
        invalid_parameter_and_run["runs"][0]["parameters"][0]["key"] = 1
        invalid_parameter_and_run["runs"][0]["id"] = 1
        with self.assertRaisesRegex(
            WireCodecError,
            r"\$\.runs\[0\]\.parameters\[0\]\.key must be a string",
        ):
            run_plan_from_wire(invalid_parameter_and_run)

        invalid_run_and_identity = copy.deepcopy(payload)
        invalid_run_and_identity["runs"][0]["dataset"] = 1
        invalid_run_and_identity["identity"]["model_type"] = 1
        with self.assertRaisesRegex(
            WireCodecError,
            r"\$\.runs\[0\]\.dataset must be a string",
        ):
            run_plan_from_wire(invalid_run_and_identity)

        invalid_identity_and_presets = copy.deepcopy(payload)
        invalid_identity_and_presets["identity"]["model_type"] = 1
        invalid_identity_and_presets["presets"] = [1]
        with self.assertRaisesRegex(
            WireCodecError,
            r"\$\.identity\.model_type must be a string",
        ):
            run_plan_from_wire(invalid_identity_and_presets)

    def test_search_spec_decoder_validation_precedence_is_stable(self) -> None:
        payload = search_spec_to_wire(_run_request().search)
        assert payload is not None

        invalid_value_and_custom_flag = copy.deepcopy(payload)
        invalid_value_and_custom_flag["axes"][0]["values"][0] = {1: "invalid-key"}
        invalid_value_and_custom_flag["axes"][0]["allow_custom_values"] = 1
        with self.assertRaisesRegex(
            WireCodecError,
            r"\$\.search\.axes\[0\]\.values\[0\] object keys must be strings",
        ):
            search_spec_from_wire(invalid_value_and_custom_flag)

        invalid_axis_and_mode = copy.deepcopy(payload)
        invalid_axis_and_mode["axes"][0]["key"] = 1
        invalid_axis_and_mode["mode"] = "unsupported"
        with self.assertRaisesRegex(
            WireCodecError,
            r"\$\.search\.axes\[0\]\.key must be a string",
        ):
            search_spec_from_wire(invalid_axis_and_mode)

        invalid_mode_and_samples = copy.deepcopy(payload)
        invalid_mode_and_samples["mode"] = "unsupported"
        invalid_mode_and_samples["random_samples"] = True
        with self.assertRaisesRegex(
            WireCodecError,
            r"\$\.search\.mode must be one of",
        ):
            search_spec_from_wire(invalid_mode_and_samples)

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
                configuration_schema_from_wire,
                {
                    **schema_payload,
                    "fields": [
                        {
                            key: value
                            for key, value in schema_payload["fields"][0].items()
                            if key != "applicableWhen"
                        }
                    ],
                },
                "missing required field 'applicableWhen'",
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

    def test_inspection_wire_enforces_graph_and_output_limits_before_decode(
        self,
    ) -> None:
        default_limits = InspectionWireLimits()
        self.assertEqual(default_limits.maximum_graph_nodes, 8_192)
        self.assertEqual(default_limits.maximum_graph_edges, 8_192)
        self.assertEqual(default_limits.maximum_output_bytes, 16 * 1024**2)

        result_payload = inspection_result_to_wire(_inspection_result())
        with self.assertRaisesRegex(
            WireCodecError,
            r"\$\.nodes must contain at most 8192 items",
        ):
            inspection_result_from_wire({**result_payload, "nodes": [None] * 8_193})

        with (
            patch(
                "model_runtime.cli._wire_inspection._INSPECTION_WIRE_LIMITS",
                InspectionCaptureLimits(maximum_output_bytes=256),
            ),
            self.assertRaisesRegex(
                WireCodecError,
                "output byte limit of 256 exceeded",
            ),
        ):
            inspection_result_from_wire(result_payload)

    def test_inspection_wire_output_limit_bounds_actual_json_bytes(self) -> None:
        result = _inspection_result()
        object.__setattr__(result, "preset", "😀" * 1_000)

        payload = inspection_result_to_wire(result)
        serialized_size = len(
            json.dumps(
                payload,
                ensure_ascii=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        self.assertGreater(serialized_size, 6_000)

        with (
            patch(
                "model_runtime.cli._wire_inspection._INSPECTION_WIRE_LIMITS",
                InspectionCaptureLimits(maximum_output_bytes=6_000),
            ),
            self.assertRaisesRegex(
                WireCodecError,
                "output byte limit of 6000 exceeded",
            ),
        ):
            inspection_result_to_wire(result)

    def test_inspection_wire_rejects_relationally_invalid_graphs(self) -> None:
        valid_payload = inspection_result_to_wire(_inspection_result())
        duplicate_node = copy.deepcopy(valid_payload)
        duplicate_node["nodes"].append(copy.deepcopy(duplicate_node["nodes"][0]))
        missing_endpoint = copy.deepcopy(valid_payload)
        missing_endpoint["edges"][0]["source"] = "missing"
        duplicate_edge = copy.deepcopy(valid_payload)
        duplicate_edge["edges"].append(copy.deepcopy(duplicate_edge["edges"][0]))
        cyclic_graph = copy.deepcopy(valid_payload)
        cyclic_graph["edges"].append(
            {
                "id": "layer-root",
                "source": "model.layer",
                "target": "__root__",
            }
        )

        for payload, message in (
            (duplicate_node, "duplicate graph node id"),
            (missing_endpoint, "unknown graph node"),
            (duplicate_edge, "duplicate graph edge id"),
            (cyclic_graph, "graph must be acyclic"),
        ):
            with (
                self.subTest(message=message),
                self.assertRaisesRegex(WireCodecError, message),
            ):
                inspection_result_from_wire(payload)

    def test_inspection_wire_validation_precedence_is_stable(self) -> None:
        valid_payload = inspection_result_to_wire(_inspection_result())

        duplicate_node = copy.deepcopy(valid_payload)
        duplicate_node["nodes"].append(copy.deepcopy(duplicate_node["nodes"][0]))
        duplicate_node["edges"] = [None]
        with self.assertRaisesRegex(WireCodecError, "duplicate graph node id"):
            inspection_result_from_wire(duplicate_node)

        duplicate_edge = copy.deepcopy(valid_payload)
        repeated_edge = copy.deepcopy(duplicate_edge["edges"][0])
        repeated_edge["source"] = "missing-source"
        duplicate_edge["edges"].append(repeated_edge)
        with self.assertRaisesRegex(WireCodecError, "duplicate graph edge id"):
            inspection_result_from_wire(duplicate_edge)

        missing_endpoints = copy.deepcopy(valid_payload)
        missing_endpoints["edges"][0]["source"] = "missing-source"
        missing_endpoints["edges"][0]["target"] = "missing-target"
        with self.assertRaisesRegex(
            WireCodecError,
            r"\$\.edges\[0\]\.source references unknown graph node",
        ):
            inspection_result_from_wire(missing_endpoints)

        cyclic_graph = copy.deepcopy(valid_payload)
        cyclic_graph["edges"].append(
            {
                "id": "layer-root",
                "source": "model.layer",
                "target": "__root__",
            }
        )
        cyclic_graph["parameter_count"] = True
        with self.assertRaisesRegex(WireCodecError, "graph must be acyclic"):
            inspection_result_from_wire(cyclic_graph)

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

    def test_json_encoder_rejects_cyclic_containers_with_the_value_path(self) -> None:
        cyclic_list = []
        cyclic_list.append(cyclic_list)
        cyclic_mapping = {}
        cyclic_mapping["self"] = cyclic_mapping

        for value, path in (
            (cyclic_list, r"\$\[0\]"),
            (cyclic_mapping, r"\$\.self"),
        ):
            with (
                self.subTest(value=type(value).__name__),
                self.assertRaisesRegex(
                    WireCodecError,
                    f"{path} contains a cyclic JSON container reference",
                ),
            ):
                json_value_to_wire(value)

    def test_json_codecs_enforce_the_exact_shared_nesting_limit(self) -> None:
        at_limit = _nested_json_list(64)
        over_limit = _nested_json_list(65)

        self.assertEqual(json_value_to_wire(at_limit), at_limit)
        self.assertEqual(json_value_from_wire(at_limit), at_limit)
        for codec in (json_value_to_wire, json_value_from_wire):
            with (
                self.subTest(codec=codec.__name__),
                self.assertRaisesRegex(
                    WireCodecError,
                    "maximum JSON nesting depth of 64 exceeded",
                ),
            ):
                codec(over_limit)

    def test_record_decoders_share_the_json_nesting_limit(self) -> None:
        over_limit = _nested_json_list(65)

        search_payload = search_spec_to_wire(_run_request().search)
        search_payload["axes"][0]["values"][0] = over_limit

        plan_payload = run_plan_to_wire(_run_plan())
        plan_payload["runs"][0]["parameters"][0]["value"] = over_limit

        result_payload = run_result_to_wire(
            RunResult(
                run_id="run-0001",
                experiment_task="image-classification",
                preset="baseline",
                dataset="Mnist",
                log_dir="logs/run/version_0",
                payload={"metrics": {"validation_accuracy": 0.75}},
            )
        )
        result_payload["payload"]["metrics"]["validation_accuracy"] = over_limit

        graph_details_payload = inspection_result_to_wire(_inspection_result())
        graph_details_payload["nodes"][1]["details"]["shape"] = over_limit

        graph_configuration_payload = inspection_result_to_wire(_inspection_result())
        graph_configuration_payload["nodes"][1]["configuration"]["fields"][0][
            "value"
        ] = over_limit

        for decoder, payload in (
            (search_spec_from_wire, search_payload),
            (run_plan_from_wire, plan_payload),
            (run_result_from_wire, result_payload),
            (inspection_result_from_wire, graph_details_payload),
            (inspection_result_from_wire, graph_configuration_payload),
            (random_state_from_wire, over_limit),
        ):
            with (
                self.subTest(decoder=decoder.__name__),
                self.assertRaisesRegex(
                    WireCodecError,
                    "maximum JSON nesting depth of 64 exceeded",
                ),
            ):
                decoder(payload)

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
