from __future__ import annotations

import multiprocessing
import tempfile
import unittest
from pathlib import Path

from filelock import Timeout

from model_runtime.runs.locking import exclusive_file_lock


def _hold_lock(path: str, ready, release) -> None:
    with exclusive_file_lock(path):
        ready.set()
        release.wait(10)


class PortableLockingTests(unittest.TestCase):
    def test_interprocess_contention_times_out_without_fcntl(self) -> None:
        context = multiprocessing.get_context("spawn")
        ready = context.Event()
        release = context.Event()
        with tempfile.TemporaryDirectory() as temporary:
            path = str(Path(temporary) / "shared.lock")
            process = context.Process(
                target=_hold_lock,
                args=(path, ready, release),
            )
            process.start()
            try:
                self.assertTrue(ready.wait(10), "child did not acquire lock")
                with self.assertRaises(Timeout):
                    with exclusive_file_lock(path, timeout=0.1):
                        pass
            finally:
                release.set()
                process.join(10)
                if process.is_alive():
                    process.terminate()
                    process.join(5)
            self.assertEqual(process.exitcode, 0)


if __name__ == "__main__":
    unittest.main()
