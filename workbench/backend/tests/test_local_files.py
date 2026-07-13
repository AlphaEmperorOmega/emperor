from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from workbench.backend.storage.local_files import (
    apply_owner_only_permissions,
    read_json_object,
    reject_symlink,
    require_safe_name,
    resolve_under_root,
    safe_child_path,
    write_json_atomic,
)


class LocalFilePathTests(unittest.TestCase):
    def test_safe_child_path_accepts_nested_relative_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            path = safe_child_path(root, "models/linear/config.json")

            self.assertEqual(path, root.resolve() / "models" / "linear" / "config.json")

    def test_safe_child_path_rejects_absolute_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            with self.assertRaises(ValueError):
                safe_child_path(root, root / "outside.json")

    def test_safe_child_path_rejects_parent_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            with self.assertRaises(ValueError):
                safe_child_path(root, "../outside.json")

    def test_safe_child_path_rejects_empty_or_dot_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            for value in ("", "."):
                with self.subTest(value=value):
                    with self.assertRaises(ValueError):
                        safe_child_path(root, value)

    def test_safe_child_path_rejects_windows_separators(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            with self.assertRaises(ValueError):
                safe_child_path(root, "models\\linear")

    def test_resolve_under_root_rejects_symlink_escape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "root"
            outside = Path(tmp) / "outside"
            root.mkdir()
            outside.mkdir()
            link = root / "link"
            try:
                link.symlink_to(outside, target_is_directory=True)
            except OSError as exc:
                self.skipTest(f"symlinks unavailable: {exc}")

            with self.assertRaises(ValueError):
                resolve_under_root(root, link / "file.txt")

    @unittest.skipUnless(sys.platform == "win32", "junctions require Windows")
    def test_resolve_under_root_rejects_junction_escape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "root"
            outside = Path(tmp) / "outside"
            root.mkdir()
            outside.mkdir()
            junction = root / "junction"
            completed = subprocess.run(
                ["cmd.exe", "/d", "/c", "mklink", "/J", junction, outside],
                check=False,
                capture_output=True,
                text=True,
            )
            if completed.returncode != 0:
                self.skipTest(f"junction creation unavailable: {completed.stderr}")

            with self.assertRaises(ValueError):
                resolve_under_root(root, junction / "escaped.txt")

    def test_reject_symlink_rejects_direct_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "target.txt"
            target.write_text("ok", encoding="utf-8")
            link = root / "link.txt"
            try:
                link.symlink_to(target)
            except OSError as exc:
                self.skipTest(f"symlinks unavailable: {exc}")

            with self.assertRaises(ValueError):
                reject_symlink(link, "test path")

    @unittest.skipUnless(sys.platform == "win32", "DACLs require Windows")
    def test_owner_only_permissions_replace_inherited_windows_dacl(self) -> None:
        import win32security

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "private.json"
            path.write_text("{}", encoding="utf-8")

            apply_owner_only_permissions(path)

            security = win32security.GetFileSecurity(
                str(path),
                win32security.OWNER_SECURITY_INFORMATION
                | win32security.DACL_SECURITY_INFORMATION,
            )
            owner = security.GetSecurityDescriptorOwner()
            dacl = security.GetSecurityDescriptorDacl()
            self.assertIsNotNone(dacl)
            assert dacl is not None
            self.assertEqual(dacl.GetAceCount(), 1)
            ace = dacl.GetAce(0)
            self.assertTrue(win32security.EqualSid(ace[2], owner))

    def test_require_safe_name_accepts_single_component(self) -> None:
        self.assertEqual(require_safe_name("linear", "model"), "linear")

    def test_require_safe_name_rejects_path_components(self) -> None:
        for value in ("", ".", "..", "linears/linear", "linears\\linear"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    require_safe_name(value, "model")


class LocalFileJsonTests(unittest.TestCase):
    def test_write_json_atomic_and_read_json_object_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nested" / "metadata.json"

            write_json_atomic(path, {"b": 2, "a": 1})

            self.assertEqual(read_json_object(path), {"a": 1, "b": 2})

    def test_read_json_object_returns_none_for_missing_or_invalid_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            missing = root / "missing.json"
            invalid = root / "invalid.json"
            invalid_utf8 = root / "invalid-utf8.json"
            non_object = root / "list.json"
            invalid.write_text("{not json", encoding="utf-8")
            invalid_utf8.write_bytes(bytes((255, 254)))
            non_object.write_text("[1, 2]", encoding="utf-8")

            self.assertIsNone(read_json_object(missing))
            self.assertIsNone(read_json_object(invalid))
            self.assertIsNone(read_json_object(invalid_utf8))
            self.assertIsNone(read_json_object(non_object))


if __name__ == "__main__":
    unittest.main()
