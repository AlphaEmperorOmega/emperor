from __future__ import annotations

import json
import os
import sys
import tempfile
import time
import unittest
import uuid
from pathlib import Path

import psutil

from emperor_workbench.training_jobs._containment._windows_job import (
    PersistedWindowsJobProcessHandle,
    WindowsJob,
    WindowsJobLimits,
    training_job_object_name,
)


@unittest.skipUnless(os.name == "nt", "Windows Job Objects require Windows")
class WindowsJobObjectTests(unittest.TestCase):
    def test_named_job_recovers_and_kills_worker_child_and_grandchild(self) -> None:
        job_id = f"test-{uuid.uuid4().hex}"
        name = training_job_object_name(job_id)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            marker = root / "descendants.json"
            log_path = root / "worker.log"
            child_code = (
                "import json, os, pathlib, subprocess, sys, time; "
                "grandchild=subprocess.Popen([sys.executable, '-c', "
                "'import time; time.sleep(60)']); "
                f"pathlib.Path({str(marker)!r}).write_text("
                "json.dumps([os.getpid(), grandchild.pid])); "
                "time.sleep(60)"
            )
            worker_code = (
                "import subprocess, sys, time; "
                f"subprocess.Popen([sys.executable, '-c', {child_code!r}]); "
                "time.sleep(60)"
            )
            job = WindowsJob.create(
                name=name,
                limits=WindowsJobLimits(
                    memory_bytes=1024**3,
                    cpu_count=max(1, os.cpu_count() or 1),
                    process_count=8,
                ),
            )
            with log_path.open("wb") as log:
                process = job.start_suspended(
                    [sys.executable, "-c", worker_code],
                    cwd=root,
                    env=dict(os.environ),
                    stdout=log,
                    stderr=log,
                )
            deadline = time.monotonic() + 10
            while not marker.is_file() and time.monotonic() < deadline:
                time.sleep(0.05)
            self.assertTrue(marker.is_file())
            descendants = json.loads(marker.read_text(encoding="utf-8"))
            pids = {process.pid, *(int(pid) for pid in descendants)}

            # Simulate backend loss: the inherited named handle keeps the Job
            # alive so the restarted backend can reopen and reconcile it.
            job.close()
            recovered = WindowsJob.open(name)
            self.assertIsNotNone(recovered)
            assert recovered is not None
            self.assertTrue(pids.issubset(set(recovered.process_ids())))
            handle = PersistedWindowsJobProcessHandle(
                pid=process.pid,
                job=recovered,
            )
            handle.kill()
            handle.wait(timeout=10)
            recovered.close()

            for _attempt in range(100):
                if all(not psutil.pid_exists(pid) for pid in pids):
                    break
                time.sleep(0.05)
            self.assertTrue(all(not psutil.pid_exists(pid) for pid in pids))


if __name__ == "__main__":
    unittest.main()
