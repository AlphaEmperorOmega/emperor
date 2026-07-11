from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import unittest


class ModelMetadataConcurrencyTests(unittest.TestCase):
    def test_cold_preset_dataset_and_monitor_requests_complete_concurrently(
        self,
    ) -> None:
        script = textwrap.dedent(
            """
            from concurrent.futures import ThreadPoolExecutor
            from threading import Barrier

            from workbench.backend.api.v1.routers.models import (
                _list_datasets,
                _list_monitors,
                _list_presets,
            )

            start = Barrier(4)

            def request(call):
                start.wait()
                return call("bert", "linear")

            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = [
                    executor.submit(request, _list_presets),
                    executor.submit(request, _list_datasets),
                    executor.submit(request, _list_monitors),
                ]
                start.wait()
                presets, datasets, monitors = [
                    future.result(timeout=30) for future in futures
                ]

            assert presets
            assert datasets["datasetGroups"]
            assert monitors
            """
        )
        env = {
            **os.environ,
            "MPLCONFIGDIR": os.environ.get("MPLCONFIGDIR", "/tmp/matplotlib"),
        }

        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            check=False,
            env=env,
            text=True,
            timeout=60,
        )

        self.assertEqual(
            result.returncode,
            0,
            msg=f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}",
        )


if __name__ == "__main__":
    unittest.main()
