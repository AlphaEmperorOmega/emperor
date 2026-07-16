from __future__ import annotations

import unittest

from tests.architecture._support import (
    dependency_graph,
    dependency_violations,
    load_manifest,
    manifest_records,
    owner_prefixes,
    strongly_connected_components,
)


class WorkbenchDependencyDirectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = load_manifest()

    def test_allowed_dependency_table_covers_exact_owner_set(self) -> None:
        owners = set(owner_prefixes(self.manifest))
        self.assertEqual(owners, set(self.manifest["allowed_dependencies"]))
        unknown_dependencies = {
            dependency
            for dependencies in self.manifest["allowed_dependencies"].values()
            for dependency in dependencies
            if dependency not in owners
        }
        self.assertEqual(set(), unknown_dependencies)

    def test_disallowed_dependency_ledger_is_exact(self) -> None:
        actual = tuple(
            violation.as_dict() for violation in dependency_violations(self.manifest)
        )
        expected = manifest_records(
            self.manifest,
            "legacy_disallowed_dependencies",
        )
        self.assertEqual(expected, actual)

    def test_runtime_owner_cycle_ledger_is_exact(self) -> None:
        actual = strongly_connected_components(dependency_graph(self.manifest))
        expected = tuple(
            tuple(component) for component in self.manifest["legacy_dependency_cycles"]
        )
        self.assertEqual(expected, actual)


if __name__ == "__main__":
    unittest.main()
