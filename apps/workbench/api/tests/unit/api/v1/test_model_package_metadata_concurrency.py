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

            from emperor_workbench.api.v1.model_packages._routes import _list_metadata
            from emperor_workbench.model_packages import ModelPackageCatalog
            from emperor_workbench.project_adapter import ProjectAdapterClient

            start = Barrier(4)

            def request(catalog):
                start.wait()
                return _list_metadata(catalog, "bert", "linear")

            with ProjectAdapterClient() as client:
                catalog = ModelPackageCatalog(client)
                with ThreadPoolExecutor(max_workers=3) as executor:
                    futures = [
                        executor.submit(request, catalog)
                        for _index in range(3)
                    ]
                    start.wait()
                    results = [future.result(timeout=30) for future in futures]

            assert all(identity.catalog_key == "bert/linear" for identity, _ in results)
            assert all(metadata.presets for _, metadata in results)
            assert all(metadata.dataset_groups for _, metadata in results)
            assert all(metadata.monitors for _, metadata in results)
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
