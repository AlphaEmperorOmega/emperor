from __future__ import annotations

import unittest

from tests.architecture._support import (
    PACKAGE_NAME,
    compatibility_marker_modules,
    facade_modules,
    legacy_public_modules,
    literal_all,
    load_manifest,
    module_path,
    obsolete_paths,
    public_modules,
    sys_modules_aliases,
    top_level_functions,
    wildcard_imports,
)


class WorkbenchInterfaceArchitectureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = load_manifest()

    def test_manifest_defines_exact_target_interface_sets(self) -> None:
        public_interfaces = set(self.manifest["public_interfaces"])
        pending_interfaces = set(self.manifest["pending_interfaces"])
        protected_interfaces = set(self.manifest["protected_interfaces"])
        self.assertEqual(1, self.manifest["schema_version"])
        self.assertTrue(self.manifest["migration_complete"])
        self.assertEqual(
            public_interfaces,
            pending_interfaces | protected_interfaces,
        )
        self.assertFalse(pending_interfaces & protected_interfaces)

        process_interfaces = set(self.manifest["process_interfaces"])
        pending_process_interfaces = set(self.manifest["pending_process_interfaces"])
        protected_process_interfaces = process_interfaces - pending_process_interfaces
        self.assertEqual(
            {
                "emperor_workbench.inspection.worker",
                "emperor_workbench.training_jobs.cgroup_worker",
                "emperor_workbench.training_jobs.worker",
            },
            process_interfaces,
        )
        self.assertEqual(
            {
                "emperor_workbench.inspection.worker",
                "emperor_workbench.training_jobs.cgroup_worker",
                "emperor_workbench.training_jobs.worker",
            },
            protected_process_interfaces,
        )

    def test_protected_interfaces_have_exact_literal_exports(self) -> None:
        for module_name, contract in self.manifest["protected_interfaces"].items():
            with self.subTest(module=module_name):
                path = module_path(module_name)
                self.assertIsNotNone(path)
                assert path is not None
                self.assertEqual(tuple(contract["exports"]), literal_all(path))

    def test_current_migration_interfaces_remain_exact(self) -> None:
        for module_name, contract in self.manifest["legacy_interface_exports"].items():
            with self.subTest(module=module_name):
                path = module_path(module_name)
                self.assertIsNotNone(path)
                assert path is not None
                self.assertEqual(tuple(contract["exports"]), literal_all(path))

    def test_every_existing_pending_interface_is_in_the_exact_ledger(self) -> None:
        pending = set(self.manifest["pending_interfaces"])
        present = {
            module_name
            for module_name in pending
            if module_path(module_name) is not None
        }
        classified = {
            *self.manifest["legacy_interface_exports"],
            *self.manifest["legacy_missing_all_interfaces"],
        }
        self.assertEqual(present, classified)
        for module_name in self.manifest["legacy_missing_all_interfaces"]:
            with self.subTest(module=module_name):
                path = module_path(module_name)
                self.assertIsNotNone(path)
                assert path is not None
                self.assertIsNone(literal_all(path))

    def test_process_interfaces_are_real_process_entry_points(self) -> None:
        pending = set(self.manifest["pending_process_interfaces"])
        for module_name in self.manifest["process_interfaces"]:
            if module_name in pending:
                continue
            with self.subTest(module=module_name):
                path = module_path(module_name)
                self.assertIsNotNone(path)
                assert path is not None
                self.assertIn("main", top_level_functions(path))

    def test_pending_process_interfaces_do_not_exist_unclassified(self) -> None:
        present = [
            module_name
            for module_name in self.manifest["pending_process_interfaces"]
            if module_path(module_name) is not None
        ]
        self.assertEqual([], present)

    def test_legacy_public_module_ledger_is_exact(self) -> None:
        self.assertEqual(
            tuple(self.manifest["legacy_public_modules"]),
            legacy_public_modules(self.manifest),
        )

    def test_all_public_modules_are_classified(self) -> None:
        actual = set(public_modules())
        classified = {
            *self.manifest["public_interfaces"],
            *self.manifest["process_interfaces"],
            *self.manifest["legacy_public_modules"],
        }
        self.assertEqual(set(), actual - classified)
        self.assertNotIn(PACKAGE_NAME, classified)

    def test_obsolete_path_ledger_is_exact(self) -> None:
        self.assertEqual(
            tuple(self.manifest["legacy_obsolete_paths"]),
            obsolete_paths(),
        )

    def test_facade_and_compatibility_ledgers_are_exact(self) -> None:
        self.assertEqual(
            tuple(self.manifest["legacy_lazy_interfaces"]),
            facade_modules(),
        )
        self.assertEqual(
            tuple(self.manifest["legacy_compatibility_marker_modules"]),
            compatibility_marker_modules(),
        )

    def test_alias_mechanisms_are_forbidden(self) -> None:
        self.assertEqual((), wildcard_imports())
        self.assertEqual((), sys_modules_aliases())

    def test_completed_migration_cannot_retain_a_legacy_ledger(self) -> None:
        if not self.manifest["migration_complete"]:
            self.skipTest("migration ledger remains active")
        self.assertEqual([], self.manifest["pending_interfaces"])
        self.assertEqual([], self.manifest["pending_process_interfaces"])
        for key in (
            "legacy_public_modules",
            "legacy_obsolete_paths",
            "legacy_lazy_interfaces",
            "legacy_compatibility_marker_modules",
            "legacy_missing_all_interfaces",
            "legacy_cross_owner_private_imports",
            "legacy_disallowed_dependencies",
            "legacy_dependency_cycles",
            "legacy_framework_imports_outside_api",
            "legacy_http_contract_modules_outside_api",
            "legacy_app_state_violations",
        ):
            with self.subTest(ledger=key):
                self.assertEqual([], self.manifest[key])
        self.assertEqual({}, self.manifest["legacy_interface_exports"])


if __name__ == "__main__":
    unittest.main()
