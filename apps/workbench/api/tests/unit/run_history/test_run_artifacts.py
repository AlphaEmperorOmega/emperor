from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from emperor_workbench.run_history import (
    RunHistoryFailure,
)
from emperor_workbench.run_history._artifacts import (
    RunArtifactBudgets,
    observe_run_artifacts,
)
from emperor_workbench.run_history._query import LogRunQueryService
from tests.support.training_jobs import (
    write_tensorboard_run,
)
from tests.unit.run_history._support import log_run_scanner


class RunHistoryArtifactTests(unittest.TestCase):
    def test_run_artifacts_cannot_resolve_into_a_sibling_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "logs"
            first_run = root / "experiment" / "first"
            second_run = root / "experiment" / "second"
            first_run.mkdir(parents=True)
            second_run.mkdir(parents=True)
            second_run.joinpath("result.json").write_text(
                json.dumps({"metrics": {"secret": 1.0}}),
                encoding="utf-8",
            )
            second_run.joinpath("secret.ckpt").write_bytes(b"secret")
            first_run.joinpath("result.json").symlink_to(second_run / "result.json")
            first_run.joinpath("stolen.ckpt").symlink_to(second_run / "secret.ckpt")

            observation = observe_run_artifacts(first_run, root)

        self.assertEqual(observation.metrics(), {})
        self.assertIsNone(observation.result)
        self.assertEqual(observation.checkpoints, ())

    def test_run_artifact_in_run_symlink_is_allowed_but_replacement_fails_closed(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "logs"
            run_dir = root / "experiment" / "run"
            run_dir.mkdir(parents=True)
            target = run_dir / "contained-result.json"
            target.write_text(
                json.dumps({"metrics": {"accuracy": 0.9}}),
                encoding="utf-8",
            )
            result = run_dir / "result.json"
            result.symlink_to(target)

            allowed = observe_run_artifacts(run_dir, root)
            self.assertEqual(allowed.metrics(), {"accuracy": 0.9})

            result.unlink()
            result.write_text(
                json.dumps({"metrics": {"beforeSwap": 1.0}}),
                encoding="utf-8",
            )
            replaced = observe_run_artifacts(run_dir, root)
            outside = Path(tmp) / "outside-result.json"
            outside.write_text(
                json.dumps({"metrics": {"secret": 1.0}}),
                encoding="utf-8",
            )
            result.unlink()
            result.symlink_to(outside)

            self.assertEqual(replaced.metrics(), {})

    def test_run_artifact_observation_enforces_file_depth_and_byte_budgets(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "version_0"
            deep = run_dir / "level-1" / "level-2"
            deep.mkdir(parents=True)
            run_dir.joinpath("result.json").write_text(
                json.dumps({"metrics": {"accuracy": 0.9}}),
                encoding="utf-8",
            )
            for name in ("a.ckpt", "b.ckpt", "c.ckpt"):
                run_dir.joinpath(name).write_bytes(b"checkpoint")
            deep.joinpath("deep.ckpt").write_bytes(b"checkpoint")

            observation = observe_run_artifacts(
                run_dir,
                root,
                budgets=RunArtifactBudgets(
                    max_files=5,
                    max_depth=1,
                    max_metadata_file_bytes=8,
                ),
            )
            depth_limited = observe_run_artifacts(
                run_dir,
                root,
                budgets=RunArtifactBudgets(
                    max_files=100,
                    max_depth=1,
                    max_metadata_file_bytes=1024,
                ),
            )

        self.assertLessEqual(observation.observed_entry_count, 5)
        self.assertLessEqual(len(observation.checkpoints), 3)
        self.assertEqual(observation.metrics(), {})
        self.assertTrue(observation.truncated)
        self.assertTrue(
            any("item cap" in reason for reason in observation.truncation_reasons)
        )
        self.assertTrue(
            any("byte cap" in reason for reason in observation.truncation_reasons)
        )
        self.assertNotIn(
            "deep.ckpt",
            {artifact.path.name for artifact in depth_limited.checkpoints},
        )
        self.assertTrue(
            any(
                "recursion cap" in reason for reason in depth_limited.truncation_reasons
            )
        )

    def test_run_history_reads_checkpoints_and_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logs_root = Path(tmp) / "logs"
            run_dir = write_tensorboard_run(
                logs_root,
                ["linear", "BASELINE", "Mnist", "aaa_20260601_010203", "version_0"],
                metrics={"test/accuracy": 0.9},
            )
            (run_dir / "result.json").write_text(
                json.dumps(
                    {
                        "params": {"learning_rate": 0.01, "optimizer": "adam"},
                        "metrics": {"test/accuracy": 0.9},
                    }
                ),
                encoding="utf-8",
            )
            (run_dir / "hparams.yaml").write_text(
                "\n".join(
                    [
                        "batch_size: 4",
                        "use_bias: true",
                        "description: 'baseline run'",
                        "nested:",
                        "ignored_list: [1, 2]",
                    ]
                ),
                encoding="utf-8",
            )
            (run_dir / "checkpoints" / "last.ckpt").write_text(
                "checkpoint",
                encoding="utf-8",
            )
            (run_dir / "checkpoints" / "epoch=2-step=300.ckpt").write_text(
                "checkpoint",
                encoding="utf-8",
            )
            malformed_run = write_tensorboard_run(
                logs_root,
                [
                    "linear",
                    "BASELINE",
                    "Mnist",
                    "malformed_20260601_050607",
                    "version_0",
                ],
                metrics=None,
            )
            (malformed_run / "result.json").write_text(
                "{not valid json",
                encoding="utf-8",
            )
            (malformed_run / "hparams.yaml").write_text(
                "nested:\nignored_list: [1, 2]\n",
                encoding="utf-8",
            )

            scanner = log_run_scanner(logs_root=logs_root)
            query = LogRunQueryService(scanner=scanner)
            runs_by_path = {run.relative_path: run for run in scanner.list_runs()}
            run = runs_by_path["linear/BASELINE/Mnist/aaa_20260601_010203/version_0"]
            malformed = runs_by_path[
                "linear/BASELINE/Mnist/malformed_20260601_050607/version_0"
            ]

            checkpoints = query.checkpoints_for_runs([run.id])
            artifacts = query.artifacts_for_run(run.id)
            malformed_artifacts = query.artifacts_for_run(malformed.id)
            with self.assertRaises(RunHistoryFailure):
                query.checkpoints_for_runs(["not-a-run"])
            with self.assertRaises(RunHistoryFailure):
                query.artifacts_for_run("not-a-run")

        self.assertEqual(
            [
                (
                    checkpoint.filename,
                    checkpoint.epoch,
                    checkpoint.step,
                )
                for checkpoint in checkpoints
            ],
            [
                ("epoch=0-step=1.ckpt", 0, 1),
                ("epoch=2-step=300.ckpt", 2, 300),
                ("last.ckpt", None, None),
            ],
        )
        self.assertTrue(
            checkpoints[0].relative_path.endswith(
                "linear/BASELINE/Mnist/aaa_20260601_010203/version_0/"
                "checkpoints/epoch=0-step=1.ckpt"
            )
        )
        self.assertGreater(checkpoints[0].size_bytes, 0)
        self.assertTrue(checkpoints[0].modified_at.endswith("Z"))
        self.assertEqual(artifacts.run_id, run.id)
        self.assertEqual(
            artifacts.params,
            {
                "batch_size": 4,
                "use_bias": True,
                "description": "baseline run",
                "learning_rate": 0.01,
                "optimizer": "adam",
            },
        )
        self.assertEqual(artifacts.metrics, {"test/accuracy": 0.9})
        self.assertEqual(
            sorted({artifact.kind for artifact in artifacts.artifacts}),
            ["checkpoint", "event_file", "hparams", "result"],
        )
        self.assertEqual(
            len(
                [
                    artifact
                    for artifact in artifacts.artifacts
                    if artifact.kind == "checkpoint"
                ]
            ),
            3,
        )
        self.assertEqual(malformed_artifacts.params, {})
        self.assertEqual(malformed_artifacts.metrics, {})


if __name__ == "__main__":
    unittest.main()
