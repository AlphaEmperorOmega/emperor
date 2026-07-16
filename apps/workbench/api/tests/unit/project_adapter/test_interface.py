from __future__ import annotations

import unittest

import emperor_workbench.project_adapter as project_adapter_interface


class ProjectAdapterInterfaceTests(unittest.TestCase):
    def test_interface_is_curated_without_a_global_accessor(self) -> None:
        self.assertEqual(
            project_adapter_interface.__all__,
            [
                "DatasetReference",
                "ModelPackageReference",
                "MonitorReference",
                "PROJECT_ADAPTER_COMMAND_ENV",
                "PresetReference",
                "ProjectAdapterClient",
                "ProjectAdapterFailure",
            ],
        )
        self.assertFalse(hasattr(project_adapter_interface, "project_adapter"))
