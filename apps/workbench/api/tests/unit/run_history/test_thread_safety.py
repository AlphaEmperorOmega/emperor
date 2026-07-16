from __future__ import annotations

import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory

from emperor_workbench.run_history._query import LogRunQueryService
from emperor_workbench.tensorboard import (
    TensorBoardMonitorReader,
    TensorBoardParameterStatusReader,
)
from tests.unit.run_history._support import log_run_scanner


class RunHistoryThreadSafetyTests(unittest.TestCase):
    def test_log_query_and_monitor_caches_handle_concurrent_clears(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "run-1"
            root.mkdir()
            (root / "events.out.tfevents.invalid").write_bytes(b"invalid")
            query_service = LogRunQueryService(scanner=log_run_scanner())
            monitor_reader = TensorBoardMonitorReader()
            parameter_reader = TensorBoardParameterStatusReader()

            def read_or_clear(index: int) -> None:
                if index % 2 == 0:
                    query_service.read_tags(root)
                    monitor_reader.read(
                        job_id="job-1",
                        node_path="main",
                        dataset="mnist",
                        log_dir=root.as_posix(),
                    )
                    parameter_reader.read(
                        source_id="job-1",
                        preset="baseline",
                        dataset="mnist",
                        log_dir=root.as_posix(),
                    )
                elif index % 4 == 1:
                    query_service.clear_run_caches([root])
                    monitor_reader.clear_roots({root.as_posix()})
                    parameter_reader.clear_roots({root.as_posix()})
                else:
                    query_service.clear_cache()
                    monitor_reader.clear_cache()
                    parameter_reader.clear_cache()

            with ThreadPoolExecutor(max_workers=8) as executor:
                list(executor.map(read_or_clear, range(80)))

            query_service.clear_cache()
            monitor_reader.clear_cache()
            parameter_reader.clear_cache()


if __name__ == "__main__":
    unittest.main()
