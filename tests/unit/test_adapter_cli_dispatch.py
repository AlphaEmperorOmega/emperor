from __future__ import annotations

import re
import unittest
from unittest.mock import patch

from model_runtime.cli import PROTOCOL_VERSION
from models.adapter_cli import (
    _OPERATION_HANDLERS,
    AdapterProtocolError,
    process_request,
)

_PACKAGE_OPERATIONS = {
    "package_metadata",
    "resolve",
    "configuration",
    "search_space",
    "parse_overrides",
    "serialize_overrides",
    "preset_locks",
    "reject_locked_overrides",
    "validate",
    "inspect",
    "parse_search_value",
    "checkpoint_config_overrides",
    "plan_runs",
    "accept_run_plan",
    "execute_run_plan",
}


def _request(operation: str, payload: dict[str, object]) -> dict[str, object]:
    return {
        "version": PROTOCOL_VERSION,
        "operation": operation,
        "payload": payload,
    }


class AdapterCliDispatchContracts(unittest.TestCase):
    def test_dispatch_table_is_immutable_and_declares_every_package_operation(
        self,
    ) -> None:
        self.assertEqual(set(_OPERATION_HANDLERS), _PACKAGE_OPERATIONS)
        with self.assertRaises(TypeError):
            _OPERATION_HANDLERS["replacement"] = lambda _package, _payload: None

    def test_unknown_operation_preserves_package_validation_precedence(self) -> None:
        cases = (
            (
                {},
                AdapterProtocolError,
                "Adapter request requires model_id.",
            ),
            (
                {"model_id": "missing/model"},
                ValueError,
                "Unknown model: missing/model",
            ),
            (
                {"model_id": "linears/linear"},
                AdapterProtocolError,
                "Unknown Adapter operation: missing_operation",
            ),
        )
        for payload, error_type, message in cases:
            with (
                self.subTest(payload=payload),
                self.assertRaisesRegex(error_type, f"^{re.escape(message)}$"),
            ):
                process_request(_request("missing_operation", payload))

        catalog = process_request(_request("catalog", {}))
        self.assertTrue(catalog["result"])

    def test_handlers_keep_same_module_patch_seams(self) -> None:
        with patch(
            "models.adapter_cli._resolve",
            return_value={"resolved": "patched"},
        ) as resolve:
            response = process_request(
                _request("resolve", {"model_id": "linears/linear"})
            )

        self.assertEqual(response["result"], {"resolved": "patched"})
        resolve.assert_called_once_with({"model_id": "linears/linear"})

        schema_sentinel = object()
        with (
            patch(
                "models.adapter_cli.configuration_schema",
                return_value=schema_sentinel,
            ) as configuration_schema,
            patch(
                "models.adapter_cli.configuration_schema_to_wire",
                return_value={"schema": "patched"},
            ) as to_wire,
        ):
            response = process_request(
                _request("configuration", {"model_id": "linears/linear"})
            )

        self.assertEqual(response["result"], {"schema": "patched"})
        configuration_schema.assert_called_once()
        to_wire.assert_called_once_with(schema_sentinel)


if __name__ == "__main__":
    unittest.main()
