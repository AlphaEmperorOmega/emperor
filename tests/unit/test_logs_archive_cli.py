from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
import zipfile
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from models.project_cli.logs_archive import USAGE, archive_logs


class _FixedDatetime:
    @classmethod
    def now(cls) -> _FixedDatetime:
        return cls()

    def strftime(self, date_format: str) -> str:
        if date_format != "%Y%m%d_%H%M%S":
            raise AssertionError(date_format)
        return "20240102_030405"


class PortableLogArchiveCliTests(unittest.TestCase):
    @staticmethod
    def _project(root: Path) -> None:
        (root / "pyproject.toml").write_text("[project]\nname='probe'\n")
        (root / "src" / "models").mkdir(parents=True)

    def test_archive_supports_spaces_and_unicode_paths(self) -> None:
        with tempfile.TemporaryDirectory(prefix="Emperor archive Ω ") as temporary:
            root = Path(temporary)
            self._project(root)
            event = root / "logs" / "experiment space Ω" / "version_0" / "event"
            event.parent.mkdir(parents=True)
            event.write_text("payload", encoding="utf-8")
            output = root / "archive space Ω.zip"

            result = archive_logs(
                ["experiment space Ω", str(output)],
                repository_root=root,
            )

            self.assertEqual(result, 0)
            with zipfile.ZipFile(output) as archive:
                self.assertIn(
                    "experiment space Ω/version_0/event",
                    archive.namelist(),
                )

    def test_help_and_planning_failure_keep_exact_streams_and_return_codes(
        self,
    ) -> None:
        stdout = StringIO()
        stderr = StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            result = archive_logs(["--help"])

        self.assertEqual(result, 0)
        self.assertEqual(stdout.getvalue(), f"{USAGE}\n")
        self.assertEqual(stderr.getvalue(), "")

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._project(root)
            stdout = StringIO()
            stderr = StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                result = archive_logs([], repository_root=root)

        self.assertEqual(result, 1)
        self.assertEqual(stdout.getvalue(), "")
        self.assertEqual(
            stderr.getvalue(),
            "Error: ./logs not found. Run this command from the project "
            f"directory.\n\n{USAGE}\n",
        )

    def test_default_archive_plan_keeps_name_and_success_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._project(root)
            (root / "logs").mkdir()
            expected_output = root / "logs_20240102_030405.zip"
            stdout = StringIO()
            stderr = StringIO()

            with (
                patch("models.project_cli.logs_archive.datetime", _FixedDatetime),
                redirect_stdout(stdout),
                redirect_stderr(stderr),
            ):
                result = archive_logs([], repository_root=root)

            self.assertEqual(result, 0)
            self.assertTrue(expected_output.is_file())
            self.assertEqual(stderr.getvalue(), "")
            self.assertEqual(
                stdout.getvalue(),
                f"Created archive: {expected_output}\n"
                "Included files: 0\n"
                "Archive size: 0.00 MiB\n",
            )

    def test_archive_rejects_symlink_escape(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._project(root)
            logs = root / "logs"
            logs.mkdir()
            outside = root / "outside"
            outside.mkdir()
            link = logs / "escape"
            try:
                link.symlink_to(outside, target_is_directory=True)
            except OSError as exc:
                self.skipTest(f"symlinks unavailable: {exc}")

            with self.assertRaisesRegex(SystemExit, "refusing symlink"):
                archive_logs([], repository_root=root)

    @unittest.skipUnless(sys.platform == "win32", "junctions require Windows")
    def test_archive_rejects_windows_junction_escape(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._project(root)
            logs = root / "logs"
            logs.mkdir()
            outside = root / "outside"
            outside.mkdir()
            junction = logs / "escape"
            completed = subprocess.run(
                ["cmd.exe", "/d", "/c", "mklink", "/J", junction, outside],
                check=False,
                capture_output=True,
                text=True,
            )
            if completed.returncode != 0:
                self.skipTest(f"junction creation unavailable: {completed.stderr}")

            with self.assertRaisesRegex(SystemExit, "reparse point"):
                archive_logs([], repository_root=root)


if __name__ == "__main__":
    unittest.main()
