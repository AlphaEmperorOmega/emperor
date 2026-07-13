from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO_ROOT / "download_logs.sh"


def create_minimal_project(root: Path) -> None:
    (root / "experiment.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    (root / "pyproject.toml").write_text("[project]\nname = 'test'\n", encoding="utf-8")
    (root / "emperor").mkdir()


@unittest.skipUnless(os.name == "posix", "download_logs.sh is a Unix wrapper")
class DownloadLogsScriptTests(unittest.TestCase):
    def test_selected_log_folder_archive_preserves_folder_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            create_minimal_project(root)
            event_file = root / "logs" / "my_experiment" / "version_0" / "events.out"
            event_file.parent.mkdir(parents=True)
            event_file.write_text("events", encoding="utf-8")
            output = root / "selected.zip"

            subprocess.run(
                ["bash", str(SCRIPT_PATH), "my_experiment", str(output)],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            )

            with zipfile.ZipFile(output) as archive:
                names = set(archive.namelist())
            self.assertIn("my_experiment/version_0/events.out", names)
            self.assertNotIn("version_0/events.out", names)

    def test_full_logs_archive_keeps_logs_contents_at_archive_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            create_minimal_project(root)
            event_file = root / "logs" / "my_experiment" / "version_0" / "events.out"
            event_file.parent.mkdir(parents=True)
            event_file.write_text("events", encoding="utf-8")
            output = root / "all.zip"

            subprocess.run(
                ["bash", str(SCRIPT_PATH), "logs", str(output)],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            )

            with zipfile.ZipFile(output) as archive:
                names = set(archive.namelist())
            self.assertIn("my_experiment/version_0/events.out", names)
            self.assertNotIn("logs/my_experiment/version_0/events.out", names)


if __name__ == "__main__":
    unittest.main()
