from __future__ import annotations

import math
import unittest

from pydantic import TypeAdapter, ValidationError

from workbench.backend.schemas import (
    GraphConfigFieldResponse,
    GraphNodeResponse,
    JsonObject,
    JsonValue,
    LogMediaRequest,
    LogRunArtifactsResponse,
    LogRunResponse,
    TrainingRunResponse,
)


class JsonApiSchemaTests(unittest.TestCase):
    def test_response_models_serialize_non_finite_floats_as_strict_json(self) -> None:
        from workbench.backend.schemas._base import ApiResponseModel

        class FloatResponse(ApiResponseModel):
            value: float

        for value in (math.nan, math.inf, -math.inf):
            with self.subTest(value=value):
                self.assertEqual(
                    FloatResponse(value=value).model_dump_json(),
                    '{"value":null}',
                )

    def test_json_value_accepts_nested_json_payloads(self) -> None:
        value = {
            "nested": [1, "two", True, None, {"score": 0.75}],
        }

        self.assertEqual(TypeAdapter(JsonValue).validate_python(value), value)
        self.assertEqual(TypeAdapter(JsonObject).validate_python(value), value)

    def test_json_value_rejects_non_json_python_objects(self) -> None:
        with self.assertRaises(ValidationError):
            TypeAdapter(JsonValue).validate_python(object())
        with self.assertRaises(ValidationError):
            TypeAdapter(JsonObject).validate_python({"bad": object()})

    def test_json_value_rejects_non_finite_numbers_at_any_depth(self) -> None:
        for value in (math.nan, math.inf, -math.inf):
            with self.subTest(value=value), self.assertRaises(ValidationError):
                TypeAdapter(JsonObject).validate_python(
                    {"metrics": {"nested": [0.5, value]}}
                )

    def test_opaque_api_json_fields_reject_non_json_values(self) -> None:
        with self.assertRaises(ValidationError):
            GraphConfigFieldResponse.model_validate({"key": "bad", "value": object()})
        with self.assertRaises(ValidationError):
            GraphNodeResponse.model_validate(
                {
                    "id": "node-1",
                    "label": "Node",
                    "typeName": "Layer",
                    "path": "main.node",
                    "graphRole": "architecture",
                    "parameterCount": 0,
                    "parameterSizeBytes": 0,
                    "details": {"bad": object()},
                    "config": None,
                }
            )
        with self.assertRaises(ValidationError):
            LogRunResponse.model_validate(
                {
                    "id": "run-1",
                    "group": None,
                    "experiment": "exp",
                    "model": "linears/linear",
                    "preset": "baseline",
                    "dataset": "Mnist",
                    "runName": "run-1",
                    "timestamp": None,
                    "version": "version_0",
                    "relativePath": "exp/linears/linear/baseline/Mnist/run-1",
                    "hasResult": True,
                    "eventFileCount": 1,
                    "checkpointCount": 0,
                    "hasHparams": True,
                    "metrics": {"bad": object()},
                }
            )
        with self.assertRaises(ValidationError):
            LogRunArtifactsResponse.model_validate(
                {
                    "runId": "run-1",
                    "params": {"bad": object()},
                    "metrics": {},
                    "artifacts": [],
                    "checkpoints": [],
                }
            )
        with self.assertRaises(ValidationError):
            TrainingRunResponse.model_validate(
                {
                    "id": "run-1",
                    "index": 0,
                    "preset": "baseline",
                    "dataset": "Mnist",
                    "changes": [],
                    "overrides": {},
                    "command": "python train.py",
                    "totalEpochs": 1,
                    "currentEpoch": 0,
                    "metrics": {"bad": object()},
                    "logDir": None,
                    "error": None,
                }
            )

    def test_log_media_request_caps_requested_tag_counts(self) -> None:
        valid_payload = {
            "runIds": ["run-1"],
            "imageTags": [f"image/{index}" for index in range(20)],
            "textTags": [f"text/{index}" for index in range(20)],
        }
        self.assertEqual(
            LogMediaRequest.model_validate(valid_payload).imageTags,
            valid_payload["imageTags"],
        )

        with self.assertRaises(ValidationError):
            LogMediaRequest.model_validate(
                {
                    **valid_payload,
                    "imageTags": [f"image/{index}" for index in range(21)],
                }
            )
        with self.assertRaises(ValidationError):
            LogMediaRequest.model_validate(
                {
                    **valid_payload,
                    "textTags": [f"text/{index}" for index in range(21)],
                }
            )
