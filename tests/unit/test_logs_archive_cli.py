from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

from models.project_cli.logs_archive import archive_logs


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
