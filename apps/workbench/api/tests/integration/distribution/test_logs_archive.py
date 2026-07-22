from __future__ import annotations

import tempfile
import unittest
import zipfile
from pathlib import Path

from models.project_cli.logs_archive import archive_logs


def create_minimal_project(root: Path) -> None:
    (root / "pyproject.toml").write_text("[project]\nname = 'test'\n", encoding="utf-8")
    (root / "src" / "models").mkdir(parents=True)


class LogsArchiveTests(unittest.TestCase):
    def test_selected_log_folder_archive_preserves_folder_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            create_minimal_project(root)
            event_file = root / "logs" / "my_experiment" / "version_0" / "events.out"
            event_file.parent.mkdir(parents=True)
            event_file.write_text("events", encoding="utf-8")
            output = root / "selected.zip"

            result = archive_logs(
                ["my_experiment", str(output)],
                repository_root=root,
            )
            self.assertEqual(result, 0)

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

            result = archive_logs(
                ["logs", str(output)],
                repository_root=root,
            )
            self.assertEqual(result, 0)

            with zipfile.ZipFile(output) as archive:
                names = set(archive.namelist())
            self.assertIn("my_experiment/version_0/events.out", names)
            self.assertNotIn("logs/my_experiment/version_0/events.out", names)


if __name__ == "__main__":
    unittest.main()
