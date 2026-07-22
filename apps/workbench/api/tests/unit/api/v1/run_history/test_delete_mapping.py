from __future__ import annotations

import unittest

from emperor_workbench.api.v1.run_history._mapping import (
    LOG_METADATA_RESPONSE_LIMIT,
    log_run_artifacts_to_payload,
    log_run_delete_plan_to_payload,
    log_run_delete_result_to_payload,
)
from emperor_workbench.run_history import (
    ActiveLogRunDeleteBlocker,
    LogRunArtifact,
    LogRunArtifacts,
    LogRunDeleteCandidate,
    LogRunDeletePlan,
    LogRunDeleteResult,
)


class LogRunDeleteHttpMappingTests(unittest.TestCase):
    def test_artifact_response_caps_metadata_rows(self) -> None:
        artifacts = tuple(
            LogRunArtifact(
                id=f"artifact-{index}",
                kind="checkpoint",
                label=f"epoch=0-step={index}.ckpt",
                relative_path=f"checkpoints/epoch=0-step={index}.ckpt",
                size_bytes=10,
                modified_at="2026-07-16T00:00:00Z",
            )
            for index in range(LOG_METADATA_RESPONSE_LIMIT + 10)
        )

        payload = log_run_artifacts_to_payload(
            LogRunArtifacts(
                run_id="run-1",
                params={},
                metrics={},
                artifacts=artifacts,
                checkpoints=(),
            )
        )

        returned_count = len(payload["artifacts"]) + len(payload["checkpoints"])
        self.assertEqual(returned_count, LOG_METADATA_RESPONSE_LIMIT)
        self.assertGreater(payload["sourceItemCount"], payload["returnedItemCount"])
        self.assertEqual(payload["returnedItemCount"], LOG_METADATA_RESPONSE_LIMIT)
        self.assertTrue(payload["truncated"])
        self.assertIn("capped", payload["truncationReason"])

    def test_delete_plan_and_result_response_payloads_are_stable(self) -> None:
        candidate = LogRunDeleteCandidate(
            id="run-1",
            experiment="test_model",
            model="linears/linear",
            preset="BASELINE",
            dataset="Mnist",
            run_name="aaa_20260601_010203",
            version="version_0",
            relative_path=(
                "test_model/linears/linear/BASELINE/Mnist/aaa_20260601_010203/version_0"
            ),
        )
        blocker = ActiveLogRunDeleteBlocker(
            id="job-1",
            log_folder="test_model",
            status="running",
        )
        expected_common = {
            "candidateCount": 1,
            "sourceItemCount": 1,
            "returnedItemCount": 1,
            "truncated": False,
            "truncationReason": None,
            "counts": {
                "runs": 1,
                "experiments": 1,
                "datasets": 1,
                "models": 1,
                "presets": 1,
            },
            "affected": {
                "experiments": ["test_model"],
                "datasets": ["Mnist"],
                "models": [{"modelType": "linears", "model": "linear"}],
                "presets": ["BASELINE"],
                "runIds": ["run-1"],
            },
            "candidates": [
                {
                    "id": "run-1",
                    "experiment": "test_model",
                    "modelType": "linears",
                    "model": "linear",
                    "preset": "BASELINE",
                    "dataset": "Mnist",
                    "runName": "aaa_20260601_010203",
                    "version": "version_0",
                    "relativePath": (
                        "test_model/linears/linear/BASELINE/Mnist/"
                        "aaa_20260601_010203/version_0"
                    ),
                }
            ],
        }

        plan_payload = log_run_delete_plan_to_payload(
            LogRunDeletePlan(
                candidates=(candidate,),
                blocked_by_active_jobs=(blocker,),
            )
        )
        result_payload = log_run_delete_result_to_payload(
            LogRunDeleteResult(
                candidates=(candidate,),
                deleted_run_ids=("run-1",),
                deleted_relative_paths=(candidate.relative_path,),
            )
        )

        self.assertEqual(
            plan_payload,
            {
                **expected_common,
                "blockedByActiveJobs": [
                    {
                        "id": "job-1",
                        "logFolder": "test_model",
                        "status": "running",
                    }
                ],
                "canDelete": False,
            },
        )
        self.assertEqual(
            result_payload,
            {
                "deletedRunIds": ["run-1"],
                "deletedRunCount": 1,
                "deletedRelativePaths": [candidate.relative_path],
                **expected_common,
                "blockedByActiveJobs": [],
                "canDelete": True,
            },
        )

    def test_delete_plan_response_caps_candidate_preview(self) -> None:
        candidates = [
            LogRunDeleteCandidate(
                id=f"run-{index}",
                experiment="test_model",
                model="linears/linear",
                preset="BASELINE",
                dataset="Mnist",
                run_name=f"run_{index:06d}_20260601_010203",
                version="version_0",
                relative_path=(
                    "test_model/linears/linear/BASELINE/Mnist/"
                    f"run_{index:06d}_20260601_010203/version_0"
                ),
            )
            for index in range(LOG_METADATA_RESPONSE_LIMIT + 3)
        ]

        payload = log_run_delete_plan_to_payload(
            LogRunDeletePlan(candidates=tuple(candidates))
        )

        self.assertEqual(payload["candidateCount"], LOG_METADATA_RESPONSE_LIMIT + 3)
        self.assertEqual(payload["sourceItemCount"], LOG_METADATA_RESPONSE_LIMIT + 3)
        self.assertEqual(payload["returnedItemCount"], LOG_METADATA_RESPONSE_LIMIT)
        self.assertEqual(len(payload["candidates"]), LOG_METADATA_RESPONSE_LIMIT)
        self.assertTrue(payload["truncated"])
        self.assertIn("capped", payload["truncationReason"])


if __name__ == "__main__":
    unittest.main()
