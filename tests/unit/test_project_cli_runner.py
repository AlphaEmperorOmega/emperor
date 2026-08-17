from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from models.project_cli.runner import run_tests


class ProjectCliRunnerTests(unittest.TestCase):
    def test_test_process_can_import_repository_and_test_packages(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository_root = Path(directory).resolve()
            (repository_root / "tests").mkdir()

            with (
                patch("models.project_cli.runner.distribution"),
                patch("models.project_cli.runner.subprocess.run") as run,
            ):
                run.return_value.returncode = 0

                return_code = run_tests([], repository_root=repository_root)

        self.assertEqual(return_code, 0)
        environment = run.call_args.kwargs["env"]
        self.assertEqual(environment["PYTHONSAFEPATH"], "1")
        self.assertEqual(
            environment["PYTHONPATH"].split(os.pathsep),
            [str(repository_root), str(repository_root / "tests")],
        )


if __name__ == "__main__":
    unittest.main()
