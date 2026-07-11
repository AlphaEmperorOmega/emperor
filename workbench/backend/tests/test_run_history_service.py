from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

from workbench.backend.inspector.errors import InspectorError
from workbench.backend.log_experiments import (
    LogExperimentMutationCoordinator,
)
from workbench.backend.run_history import (
    HistoricalInspectionSource,
    RunHistoryService,
)
from workbench.backend.tests.helpers import write_tensorboard_run


@dataclass(frozen=True, slots=True)
class _ActiveWriter:
    id: str
    status: str
    log_folder: str


def _service(
    logs_root: Path,
    *,
    writers: list[_ActiveWriter] | None = None,
) -> RunHistoryService:
    active_writers = writers or []
    return RunHistoryService(
        logs_root=logs_root,
        mutation_coordinator=LogExperimentMutationCoordinator(),
        active_log_writers=lambda: list(active_writers),
    )


class RunHistoryServiceContractTests(unittest.TestCase):
    def test_historical_inspection_uses_one_narrow_immutable_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logs_root = Path(tmp) / "logs"
            run_dir = write_tensorboard_run(
                logs_root,
                [
                    "test_model",
                    "linear",
                    "BASELINE",
                    "Mnist",
                    "run_20260711_010203",
                    "version_0",
                ],
                metrics={"test/accuracy": 0.9},
            )
            (run_dir / "result.json").write_text(
                json.dumps(
                    {
                        "metrics": {"test/accuracy": 0.9},
                        "params": {"batch_size": 8, "dropout": 0.1},
                    }
                ),
                encoding="utf-8",
            )
            service = _service(logs_root)
            source: HistoricalInspectionSource = service
            run_id = service.list_runs(limit=1, offset=0)["runs"][0]["id"]

            context = source.inspection_context(str(run_id))

            self.assertEqual(context.run_id, run_id)
            self.assertEqual(context.model, "linears/linear")
            self.assertEqual(context.preset, "BASELINE")
            self.assertEqual(context.dataset, "Mnist")
            self.assertEqual(
                dict(context.params),
                {"batch_size": 8, "dropout": 0.1},
            )
            self.assertEqual(
                context.checkpoint_paths,
                (run_dir / "checkpoints" / "epoch=0-step=1.ckpt",),
            )
            with self.assertRaises(TypeError):
                context.params["batch_size"] = 16  # type: ignore[index]

    def test_mutation_reads_typed_active_writer_source_inside_capability(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logs_root = Path(tmp) / "logs"
            experiment = logs_root / "test_model"
            experiment.mkdir(parents=True)
            service = _service(
                logs_root,
                writers=[
                    _ActiveWriter(
                        id="job-1",
                        status="Running",
                        log_folder="test_model",
                    )
                ],
            )

            with self.assertRaisesRegex(
                InspectorError,
                "A training job is still writing to this log folder",
            ):
                service.delete_experiment("test_model")

            self.assertTrue(experiment.is_dir())

    def test_inspection_context_revalidates_replaced_checkpoint_containment(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            logs_root = root / "logs"
            run_dir = write_tensorboard_run(
                logs_root,
                [
                    "test_model",
                    "linear",
                    "BASELINE",
                    "Mnist",
                    "run_20260711_050607",
                    "version_0",
                ],
            )
            service = _service(logs_root)
            run_id = str(service.list_runs(limit=1, offset=0)["runs"][0]["id"])
            checkpoint = run_dir / "checkpoints" / "epoch=0-step=1.ckpt"
            checkpoint.unlink()
            outside_checkpoint = root / "outside.ckpt"
            outside_checkpoint.write_text("outside", encoding="utf-8")
            checkpoint.symlink_to(outside_checkpoint)

            context = service.inspection_context(run_id)

            self.assertEqual(context.checkpoint_paths, ())
            self.assertEqual(outside_checkpoint.read_text(encoding="utf-8"), "outside")


if __name__ == "__main__":
    unittest.main()
