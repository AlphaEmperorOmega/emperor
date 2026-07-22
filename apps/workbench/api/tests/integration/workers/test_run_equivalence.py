from __future__ import annotations

import contextlib
import io
import json
import os
import random
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from emperor.experiments.tasks import ExperimentTask
from emperor.layers import ActivationOptions
from model_runtime.packages import RandomSearch
from models.linears.linear.presets import ExperimentPreset
from models.package_cli import run_model_package_cli

from emperor_workbench.model_packages import ModelPackageCatalog
from emperor_workbench.run_plans import (
    CreateTrainingRunPlanCommand,
    RunPlanPersistenceCodec,
    RunPlanService,
    RunPlanWorkerAcceptance,
    TrainingSearch,
)
from emperor_workbench.training_jobs import worker as training_worker
from tests.support.model_packages import project_adapter_client


def _semantic_rows(plan) -> list[tuple[str, str, str, dict]]:
    return [(run.id, run.preset, run.dataset, dict(run.overrides)) for run in plan.runs]


class RunsCliEquivalenceTests(unittest.TestCase):
    def test_seeded_cli_and_worker_execute_equivalent_plan_and_artifact_scope(
        self,
    ) -> None:
        args = SimpleNamespace(
            datasets=["mnist", "cifar10"],
            logdir="runs_equivalence",
        )
        mode = SimpleNamespace(
            experiment_task=ExperimentTask.IMAGE_CLASSIFICATION,
            preset=ExperimentPreset.BASELINE,
            selected_presets=[
                ExperimentPreset.BASELINE,
                ExperimentPreset.GATING,
            ],
            search_mode=RandomSearch(num_samples=3),
            search_keys=None,
            config_overrides={"stack_num_layers": 4},
            search_overrides={
                "hidden_dim": [64, 128],
                "stack_activation": [
                    ActivationOptions.RELU,
                    ActivationOptions.GELU,
                ],
            },
            monitor_names=["linear"],
        )
        parser = SimpleNamespace(parse_args=lambda _argv=None: args)
        random_state = random.getstate()
        try:
            random.seed(13)
            with (
                patch(
                    "models.package_cli.get_experiment_parser",
                    return_value=parser,
                ),
                patch(
                    "models.package_cli.resolve_cli_selection",
                    return_value=mode,
                ),
                patch(
                    "models.package_cli.execute_runs",
                    return_value=(),
                ) as cli_execute,
            ):
                run_model_package_cli("linears/linear")
        finally:
            random.setstate(random_state)

        service = RunPlanService(
            random_source=random.Random(13),
            model_packages=ModelPackageCatalog(project_adapter_client()),
        )
        plan = service.preview(
            CreateTrainingRunPlanCommand(
                model="linears/linear",
                preset="baseline",
                presets=["baseline", "gating"],
                datasets=["Mnist", "Cifar10"],
                overrides={"stack_num_layers": 4},
                search=TrainingSearch(
                    mode="random",
                    random_samples=3,
                    values={
                        "hidden_dim": [64, 128],
                        "stack_activation": ["RELU", "GELU"],
                    },
                ),
                log_folder="runs_equivalence",
                monitors=["linear"],
            )
        )
        serialized_plan = RunPlanPersistenceCodec.encode(plan)
        worker_payload = {
            "id": "equivalence-job",
            "monitors": ["linear"],
            "plannedRunCount": len(serialized_plan["runs"]),
            "runPlan": serialized_plan,
        }
        with tempfile.TemporaryDirectory() as tmp:
            payload_path = Path(tmp) / "payload.json"
            progress_path = Path(tmp) / "progress.jsonl"
            payload_path.write_text(json.dumps(worker_payload), encoding="utf-8")
            with (
                patch.object(
                    sys,
                    "argv",
                    [
                        "training_worker",
                        "--payload",
                        str(payload_path),
                        "--progress",
                        str(progress_path),
                    ],
                ),
                patch.object(
                    RunPlanWorkerAcceptance,
                    "execute",
                    return_value=plan,
                ) as worker_execute,
                contextlib.redirect_stderr(io.StringIO()),
            ):
                training_worker.main()

        cli_package, cli_plan = cli_execute.call_args.args
        (executed_payload,) = worker_execute.call_args.args
        worker_package = project_adapter_client().package("linears/linear")
        worker_plan = RunPlanWorkerAcceptance.accept(
            worker_package,
            executed_payload,
        )
        self.assertEqual(cli_package.identity, worker_package.identity)
        self.assertEqual(cli_plan.identity, worker_plan.identity)
        self.assertEqual(cli_plan.presets, worker_plan.presets)
        self.assertEqual(cli_plan.experiment_task, worker_plan.experiment_task)
        self.assertEqual(cli_plan.datasets, worker_plan.datasets)
        self.assertEqual(dict(cli_plan.overrides), dict(worker_plan.overrides))
        self.assertEqual(cli_plan.search, worker_plan.search)
        self.assertEqual(_semantic_rows(cli_plan), _semantic_rows(worker_plan))
        self.assertEqual(
            cli_execute.call_args.kwargs["artifacts"].root,
            worker_execute.call_args.kwargs["logs_root"],
        )
        self.assertEqual(
            cli_execute.call_args.kwargs["artifacts"].namespace,
            executed_payload["runPlan"]["logFolder"],
        )
        self.assertEqual(
            cli_execute.call_args.kwargs["monitors"],
            tuple(executed_payload["monitors"]),
        )
        self.assertNotIn("progress", cli_execute.call_args.kwargs)
        self.assertEqual(
            worker_execute.call_args.kwargs["progress_path"],
            progress_path,
        )


if __name__ == "__main__":
    unittest.main()
