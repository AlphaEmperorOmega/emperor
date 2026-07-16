from __future__ import annotations

import asyncio
import unittest
from unittest.mock import patch

import httpx

from emperor_workbench.api import create_app
from emperor_workbench.settings import WorkbenchApiSettings
from emperor_workbench.training_jobs import TrainingResourceLimits
from emperor_workbench.training_jobs._containment._cgroup_v2 import (
    CgroupV2Manager,
    StrictCancellationUnavailable,
)
from tests.support import lifespan_client


class TrainingCapabilityHttpProbeTests(unittest.TestCase):
    def test_capabilities_endpoint_degrades_unavailable_strict_probe(
        self,
    ) -> None:
        with patch.object(
            CgroupV2Manager,
            "__init__",
            side_effect=StrictCancellationUnavailable("missing /proc"),
        ) as construct_cgroups:
            test_app = create_app(
                WorkbenchApiSettings(
                    training_cancellation_mode="strict-cgroup",
                )
            )

            async def call_api() -> httpx.Response:
                async with lifespan_client(test_app) as client:
                    return await client.get("/capabilities")

            response = asyncio.run(call_api())

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json()["trainingCancellationCapability"],
            "unsupported",
        )
        construct_cgroups.assert_called_once_with(
            resource_limits=TrainingResourceLimits()
        )


if __name__ == "__main__":
    unittest.main()
