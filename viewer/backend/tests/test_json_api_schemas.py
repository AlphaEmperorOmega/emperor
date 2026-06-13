from __future__ import annotations

import unittest

from pydantic import TypeAdapter, ValidationError

from viewer.backend.schemas import (
    GraphConfigFieldResponse,
    GraphNodeResponse,
    JsonObject,
    JsonValue,
    LogRunArtifactsResponse,
    LogRunResponse,
    TrainingRunResponse,
)


class JsonApiSchemaTests(unittest.TestCase):
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

    def test_opaque_api_json_fields_reject_non_json_values(self) -> None:
        with self.assertRaises(ValidationError):
            GraphConfigFieldResponse.model_validate(
                {"key": "bad", "value": object()}
            )
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
