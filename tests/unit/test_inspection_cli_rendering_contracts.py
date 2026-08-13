from __future__ import annotations

import json
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from types import SimpleNamespace
from unittest.mock import patch

from model_runtime.inspection import InspectionCaptureLimits, InspectionRequest
from models.inspection_cli import _parse_args, _render_json_inspection, run_inspection


class InspectionCliRenderingContracts(unittest.TestCase):
    def test_shape_trace_help_identifies_trusted_local_execution(self) -> None:
        stdout = StringIO()

        with redirect_stdout(stdout), self.assertRaises(SystemExit) as raised:
            _parse_args(
                [
                    "--model-type",
                    "linears",
                    "--model",
                    "linear",
                    "--help",
                ]
            )

        self.assertEqual(raised.exception.code, 0)
        help_text = stdout.getvalue().lower()
        self.assertIn("trusted local model", help_text)
        self.assertIn("caller process", help_text)
        self.assertIn("no deadline or memory isolation", help_text)

    def test_json_output_is_compact_and_keeps_payload_order(self) -> None:
        stdout = StringIO()

        with redirect_stdout(stdout):
            _render_json_inspection(
                {"second": [1, 2], "first": {"enabled": True}},
                InspectionRequest(preset="baseline"),
            )

        self.assertEqual(
            stdout.getvalue(),
            '{"second":[1,2],"first":{"enabled":true}}\n',
        )

    def test_json_output_limit_counts_the_trailing_newline(self) -> None:
        payload = {"value": "é"}
        encoded = json.dumps(payload, separators=(",", ":"))
        encoded_bytes = len(encoded.encode("utf-8"))
        request = InspectionRequest(
            preset="baseline",
            capture_limits=InspectionCaptureLimits(
                maximum_output_bytes=encoded_bytes + 1
            ),
        )
        stdout = StringIO()

        with redirect_stdout(stdout):
            _render_json_inspection(payload, request)

        self.assertEqual(stdout.getvalue(), f"{encoded}\n")

    def test_run_inspection_reports_json_byte_limit_errors_on_stderr(self) -> None:
        payload = {"value": "é"}
        encoded = json.dumps(payload, separators=(",", ":"))
        maximum_output_bytes = len(encoded.encode("utf-8"))
        request = InspectionRequest(
            preset="baseline",
            capture_limits=InspectionCaptureLimits(
                maximum_output_bytes=maximum_output_bytes
            ),
        )
        resolved = SimpleNamespace(output_format="json", request=request)
        stdout = StringIO()
        stderr = StringIO()

        with (
            patch(
                "models.inspection_cli._resolve_inspection_request",
                return_value=resolved,
            ),
            patch("models.inspection_cli._execute_inspection", return_value=object()),
            patch("models.inspection_cli._execution_payload", return_value=payload),
            redirect_stdout(stdout),
            redirect_stderr(stderr),
        ):
            return_code = run_inspection([])

        self.assertEqual(return_code, 1)
        self.assertEqual(stdout.getvalue(), "")
        self.assertEqual(
            stderr.getvalue(),
            f"Inspection output byte limit of {maximum_output_bytes} exceeded.\n",
        )


if __name__ == "__main__":
    unittest.main()
