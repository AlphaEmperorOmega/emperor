from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from emperor_workbench.tensorboard import MonitorData, TensorBoardMonitorReader
from tests.support import lifespan_client
from tests.support.model_packages import list_log_runs


class HistoricalMonitorDataFailureTests(unittest.TestCase):
    def write_historical_run(self, logs_root: Path) -> tuple[str, Path]:
        run_dir = logs_root.joinpath(
            "test_model",
            "linears",
            "linear",
            "baseline",
            "Mnist",
            "historical_20260601_010203",
            "version_0",
        )
        run_dir.mkdir(parents=True)
        (run_dir / "events.out.tfevents.test").write_text("broken", encoding="utf-8")
        run = list_log_runs(logs_root=logs_root)[0]
        return run.id, run_dir

    def test_historical_monitor_endpoint_projects_empty_reader_result(self) -> None:
        import httpx

        from emperor_workbench.api import create_app
        from emperor_workbench.settings import WorkbenchApiSettings

        async def call_api(logs_root: Path, run_id: str) -> httpx.Response:
            app = create_app(WorkbenchApiSettings(logs_root=str(logs_root)))
            async with lifespan_client(app) as client:
                return await client.get(
                    f"/logs/runs/{run_id}/monitor-data",
                    params={"nodePath": "main_model.0.model"},
                )

        with tempfile.TemporaryDirectory() as tmp:
            logs_root = Path(tmp) / "logs"
            run_id, run_dir = self.write_historical_run(logs_root)

            with patch.object(
                TensorBoardMonitorReader,
                "read",
                return_value=MonitorData(
                    job_id=run_id,
                    node_path="main_model.0.model",
                    preset=None,
                    dataset="Mnist",
                    log_dir=str(run_dir),
                    scalar_series=(),
                    histograms=(),
                    images=(),
                ),
            ) as read_monitor:
                response = asyncio.run(call_api(logs_root, run_id))

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(
            response.json(),
            {
                "jobId": run_id,
                "nodePath": "main_model.0.model",
                "preset": None,
                "dataset": "Mnist",
                "logDir": str(run_dir),
                "scalarSeries": [],
                "histograms": [],
                "images": [],
            },
        )
        read_monitor.assert_called_once()


if __name__ == "__main__":
    unittest.main()
