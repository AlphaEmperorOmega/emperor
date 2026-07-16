from __future__ import annotations

import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import psutil
from model_runtime.inspection import InspectionRequest, InspectionResult

from emperor_workbench.inspection import (
    InspectionFailure,
    InspectionWorkerLimits,
    SubprocessInspectionExecutor,
)
from emperor_workbench.model_packages import ModelPackageCatalog
from emperor_workbench.settings import WorkbenchApiSettings
from tests.support.model_packages import project_adapter_client


class InspectionWorkerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.package = ModelPackageCatalog(project_adapter_client()).select(
            "linears/linear"
        )
        self.request = InspectionRequest(preset="baseline", dataset="Mnist")

    def test_balanced_limits_are_the_configuration_defaults(self) -> None:
        settings = WorkbenchApiSettings()

        self.assertEqual(settings.inspection_memory_limit_bytes, 4 * 1024**3)
        self.assertEqual(settings.inspection_cpu_limit, 4)
        self.assertEqual(settings.inspection_timeout_seconds, 60.0)

    def test_worker_limits_accept_environment_overrides(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "WORKBENCH_API_INSPECTION_MEMORY_LIMIT_BYTES": "1073741824",
                "WORKBENCH_API_INSPECTION_CPU_LIMIT": "2",
                "WORKBENCH_API_INSPECTION_TIMEOUT_SECONDS": "7.5",
            },
        ):
            settings = WorkbenchApiSettings()

        self.assertEqual(settings.inspection_memory_limit_bytes, 1024**3)
        self.assertEqual(settings.inspection_cpu_limit, 2)
        self.assertEqual(settings.inspection_timeout_seconds, 7.5)

    def test_worker_returns_a_semantic_inspection_result(self) -> None:
        executor = SubprocessInspectionExecutor(
            InspectionWorkerLimits(
                memory_bytes=4 * 1024**3,
                cpu_count=1,
                timeout_seconds=20,
            )
        )

        result = executor.inspect(self.package, self.request)

        self.assertIsInstance(result, InspectionResult)
        self.assertEqual(result.identity.catalog_key, "linears/linear")
        self.assertGreater(result.parameter_count, 0)

    def test_timeout_kills_the_worker_and_subsequent_inspection_works(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            child_pid_path = Path(tmp) / "child.pid"
            worker_code = (
                "import pathlib, subprocess, sys, time; "
                "child=subprocess.Popen([sys.executable, '-c', "
                "'import time; time.sleep(60)']); "
                f"pathlib.Path({str(child_pid_path)!r}).write_text(str(child.pid)); "
                "time.sleep(60)"
            )
            executor = SubprocessInspectionExecutor(
                InspectionWorkerLimits(
                    memory_bytes=4 * 1024**3,
                    cpu_count=1,
                    timeout_seconds=0.5,
                ),
                command=(sys.executable, "-c", worker_code),
            )

            started_at = time.monotonic()
            with self.assertRaisesRegex(
                InspectionFailure,
                "exceeded the 0.5 second",
            ):
                executor.inspect(self.package, self.request)
            self.assertLess(time.monotonic() - started_at, 2)
            child_pid = int(child_pid_path.read_text(encoding="utf-8"))
            for _attempt in range(40):
                if not psutil.pid_exists(child_pid):
                    break
                time.sleep(0.05)
            self.assertFalse(psutil.pid_exists(child_pid))

        healthy_result = SubprocessInspectionExecutor(
            InspectionWorkerLimits(
                memory_bytes=4 * 1024**3,
                cpu_count=1,
                timeout_seconds=20,
            )
        ).inspect(self.package, self.request)
        self.assertGreater(healthy_result.parameter_count, 0)

    def test_memory_breach_cannot_take_down_the_backend_process(self) -> None:
        resource_setup = (
            "import resource; resource.setrlimit(resource.RLIMIT_AS, (limit, limit)); "
            if os.name == "posix"
            else ""
        )
        worker_code = (
            "import json, sys; "
            "payload=json.loads(sys.stdin.read()); "
            "limit=payload['limits']['memoryBytes']; "
            f"{resource_setup}"
            "bytearray(limit * 2)"
        )
        executor = SubprocessInspectionExecutor(
            InspectionWorkerLimits(
                memory_bytes=128 * 1024**2,
                cpu_count=1,
                timeout_seconds=5,
            ),
            command=(sys.executable, "-c", worker_code),
        )

        with self.assertRaisesRegex(
            InspectionFailure,
            "Inspection worker crashed",
        ):
            executor.inspect(self.package, self.request)

        healthy_result = SubprocessInspectionExecutor(
            InspectionWorkerLimits(
                memory_bytes=4 * 1024**3,
                cpu_count=1,
                timeout_seconds=20,
            )
        ).inspect(self.package, self.request)
        self.assertGreater(healthy_result.parameter_count, 0)

    def test_worker_crash_and_malformed_output_are_stable_failures(self) -> None:
        cases = (
            (
                (sys.executable, "-c", "raise SystemExit(23)"),
                "Inspection worker crashed.",
            ),
            (
                (sys.executable, "-c", "print('not-json')"),
                "Inspection worker produced an invalid result.",
            ),
        )
        for command, detail in cases:
            with self.subTest(detail=detail):
                executor = SubprocessInspectionExecutor(
                    InspectionWorkerLimits(
                        memory_bytes=4 * 1024**3,
                        cpu_count=1,
                        timeout_seconds=5,
                    ),
                    command=command,
                )
                with self.assertRaisesRegex(InspectionFailure, detail):
                    executor.inspect(self.package, self.request)


if __name__ == "__main__":
    unittest.main()
