from __future__ import annotations

import contextlib
import copy
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from emperor.runs.progress import (
    JsonlTrainingProgressCallback,
    NeuronClusterGrowthCallback,
)

from workbench.backend.training_jobs import run_plan_adapter
from workbench.backend.training_jobs import worker as training_worker
from workbench.backend.training_jobs.launcher import TRAINING_LOGS_ROOT_ENV
from workbench.backend.training_jobs.limits import (
    MAX_TRAINING_DATASETS,
    MAX_TRAINING_MONITORS,
)
from workbench.backend.training_jobs.run_plan_adapter import (
    WorkbenchRunPlanAdapter,
    encode_persisted_run_plan,
)

COMMON_PROGRESS_EVENT_KEYS = {
    "timestamp",
    "dataset",
    "preset",
    "presetKey",
    "logDir",
    "runId",
    "runIndex",
    "runTotal",
    "totalEpochs",
}


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def worker_payload(
    *,
    preset: str = "baseline",
    presets: list[str] | None = None,
    datasets: list[str] | None = None,
    overrides: dict | None = None,
    search: dict | None = None,
    monitors: list[str] | None = None,
) -> dict[str, object]:
    builder = WorkbenchRunPlanAdapter()
    selected = builder.resolve_inputs(
        model="linears/linear",
        preset=preset,
        presets=presets,
        datasets=datasets or ["Mnist"],
        overrides=overrides or {},
        search=search,
    )
    plan = builder.create(
        model="linears/linear",
        selected=selected,
        log_folder="unit_logs",
        monitors=monitors or [],
    )
    serialized_plan = encode_persisted_run_plan(plan)
    return {
        "id": "job-123",
        "monitors": monitors or [],
        "plannedRunCount": len(plan.runs),
        "runPlan": serialized_plan,
    }


class TrainingWorkerPlanAcceptanceTests(unittest.TestCase):
    def test_worker_accepts_exact_materialized_rows_without_sampling(self) -> None:
        payload = worker_payload(
            search={
                "mode": "random",
                "values": {"hidden_dim": [64, 128]},
                "randomSamples": 1,
            }
        )
        package = training_worker.load_model_parts("linears/linear")

        with patch("emperor.runs.planning.plan_runs") as plan_runs:
            plan = run_plan_adapter.accept_worker_run_plan(package, payload)

        plan_runs.assert_not_called()
        self.assertEqual(len(plan.runs), 1)
        self.assertEqual(plan.runs[0].id, payload["runPlan"]["runs"][0]["id"])
        self.assertEqual(plan.runs[0].preset, "baseline")
        self.assertEqual(plan.runs[0].dataset, "Mnist")

    def test_worker_requires_a_nonempty_materialized_plan(self) -> None:
        package = training_worker.load_model_parts("linears/linear")
        base = worker_payload()
        cases = (
            None,
            "not-a-plan",
            {},
            {"runs": None},
            {"runs": []},
        )
        for run_plan in cases:
            with self.subTest(run_plan=run_plan):
                payload = {**base, "runPlan": run_plan}
                with self.assertRaisesRegex(
                    ValueError,
                    "non-empty materialized run plan",
                ):
                    run_plan_adapter.accept_worker_run_plan(package, payload)

    def test_worker_rejects_non_object_and_foreign_rows(self) -> None:
        package = training_worker.load_model_parts("linears/linear")
        base = worker_payload()
        non_object = {
            **base,
            "runPlan": {**base["runPlan"], "runs": ["not-a-row"]},
        }
        with self.assertRaisesRegex(ValueError, "row 1 must be an object"):
            run_plan_adapter.accept_worker_run_plan(package, non_object)

        foreign = {
            **base,
            "runPlan": {
                **base["runPlan"],
                "modelType": "gpt",
                "model": "linear",
            },
        }
        with self.assertRaisesRegex(ValueError, "does not match selected model"):
            run_plan_adapter.accept_worker_run_plan(package, foreign)

    def test_worker_revalidates_locked_row_overrides(self) -> None:
        payload = worker_payload(preset="gating", presets=["gating"])
        payload["runPlan"]["runs"][0]["overrides"]["gate_flag"] = "false"
        package = training_worker.load_model_parts("linears/linear")

        with self.assertRaisesRegex(ValueError, "locked fields"):
            run_plan_adapter.accept_worker_run_plan(package, payload)

    def test_worker_rejects_tampered_derived_epoch_total(self) -> None:
        payload = worker_payload(overrides={"num_epochs": 7})
        payload["runPlan"]["runs"][0]["totalEpochs"] = 999
        package = training_worker.load_model_parts("linears/linear")

        with self.assertRaisesRegex(ValueError, "total epochs"):
            run_plan_adapter.accept_worker_run_plan(package, payload)

    def test_worker_rejects_stale_or_tampered_plan_envelopes(self) -> None:
        package = training_worker.load_model_parts("linears/linear")
        cases = (
            ("preset", "gating", "primary preset"),
            ("presets", ["gating"], "presets"),
            ("datasets", ["Cifar10"], "datasets"),
            ("overrides", {"HIDDEN_DIM": 64}, "overrides"),
            (
                "search",
                {
                    "mode": "grid",
                    "values": {"HIDDEN_DIM": [64]},
                },
                "search",
            ),
            ("logFolder", "other_logs", "log folder"),
        )
        for field, value, message in cases:
            with self.subTest(field=field):
                payload = copy.deepcopy(worker_payload())
                payload["runPlan"][field] = value
                with self.assertRaisesRegex(ValueError, message):
                    run_plan_adapter.accept_worker_run_plan(package, payload)

        for field, limit in (
            ("datasets", MAX_TRAINING_DATASETS),
            ("monitors", MAX_TRAINING_MONITORS),
        ):
            with self.subTest(field=field):
                payload = copy.deepcopy(worker_payload())
                target = payload if field == "monitors" else payload["runPlan"]
                target[field] = [f"value-{index}" for index in range(limit + 1)]
                with self.assertRaisesRegex(
                    ValueError,
                    f"at most {limit} {field}",
                ):
                    run_plan_adapter.accept_worker_run_plan(package, payload)

        payload = copy.deepcopy(worker_payload())
        del payload["runPlan"]["modelType"]
        del payload["runPlan"]["model"]
        with self.assertRaisesRegex(ValueError, "valid model identity"):
            run_plan_adapter.accept_worker_run_plan(package, payload)

    def test_worker_rejects_structurally_erased_plan_state(self) -> None:
        package = training_worker.load_model_parts("linears/linear")
        for field in ("id", "preset", "dataset", "experimentTask", "overrides"):
            with self.subTest(row_field=field):
                payload = copy.deepcopy(worker_payload())
                del payload["runPlan"]["runs"][0][field]
                with self.assertRaisesRegex(ValueError, field):
                    run_plan_adapter.accept_worker_run_plan(package, payload)

        for field in (
            "preset",
            "presets",
            "experimentTask",
            "datasets",
            "overrides",
            "search",
            "logFolder",
        ):
            with self.subTest(envelope_field=field):
                payload = copy.deepcopy(worker_payload())
                del payload["runPlan"][field]
                with self.assertRaisesRegex(ValueError, "requires"):
                    run_plan_adapter.accept_worker_run_plan(package, payload)

    def test_worker_rejects_limits_before_copying_private_plan_or_search(self) -> None:
        package = training_worker.load_model_parts("linears/linear")
        payload = copy.deepcopy(worker_payload())
        payload["runPlan"]["runs"] = [
            copy.deepcopy(payload["runPlan"]["runs"][0]) for _index in range(2001)
        ]
        with (
            patch.object(run_plan_adapter, "SubmittedRun") as submitted_run,
            self.assertRaisesRegex(ValueError, "submitted runs exceeds 2000"),
        ):
            run_plan_adapter.accept_worker_run_plan(package, payload)
        submitted_run.assert_not_called()

        for search, message in (
            (
                {
                    "mode": "grid",
                    "values": {f"axis_{index}": [1] for index in range(17)},
                },
                "at most 16 selected axes",
            ),
            (
                {
                    "mode": "grid",
                    "values": {"hidden_dim": list(range(51))},
                },
                "at most 50 selected values",
            ),
            (
                {
                    "mode": "random",
                    "values": {"hidden_dim": [64]},
                    "randomSamples": 2001,
                },
                "between 1 and 2000",
            ),
        ):
            with self.subTest(message=message):
                payload = copy.deepcopy(worker_payload())
                payload["runPlan"]["search"] = copy.deepcopy(search)
                with (
                    patch.object(run_plan_adapter, "SubmittedRun") as submitted_run,
                    self.assertRaisesRegex(ValueError, message),
                ):
                    run_plan_adapter.accept_worker_run_plan(package, payload)
                submitted_run.assert_not_called()


class TrainingWorkerProgressTests(unittest.TestCase):
    def run_worker(
        self,
        payload: dict[str, object],
        progress_path: Path,
        *,
        execute_side_effect: Exception | None = None,
    ):
        payload_path = progress_path.parent / "payload.json"
        payload_path.write_text(json.dumps(payload), encoding="utf-8")
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
            patch(
                "workbench.backend.training_jobs.worker.execute_runs",
                side_effect=execute_side_effect,
                return_value=(),
            ) as execute,
        ):
            training_worker.main()
        return execute

    def test_worker_executes_accepted_plan_and_writes_lifecycle_events(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            progress_path = Path(tmp) / "progress.jsonl"
            payload = worker_payload(
                presets=["baseline", "gating"],
                datasets=["Mnist", "Cifar10"],
            )

            execute = self.run_worker(payload, progress_path)

            execute.assert_called_once()
            package, plan = execute.call_args.args
            self.assertEqual(package.catalog_key, "linears/linear")
            self.assertEqual(plan.presets, ("baseline", "gating"))
            self.assertEqual(plan.datasets, ("Mnist", "Cifar10"))
            self.assertEqual(len(plan.runs), 4)
            self.assertIsInstance(
                execute.call_args.kwargs["progress"],
                JsonlTrainingProgressCallback,
            )
            events = read_jsonl(progress_path)

        self.assertEqual([event["type"] for event in events], ["started", "completed"])
        self.assertEqual(events[0]["status"], "running")
        self.assertEqual(events[0]["jobId"], "job-123")
        self.assertEqual(events[0]["modelType"], "linears")
        self.assertEqual(events[0]["model"], "linear")
        self.assertEqual(events[1]["status"], "completed")
        self.assertEqual(events[1]["preset"], "gating")
        self.assertEqual(events[1]["presets"], ["baseline", "gating"])

    def test_worker_uses_process_owned_artifact_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            progress_path = root / "progress.jsonl"
            logs_root = root / "custom-logs"
            with patch.dict(
                os.environ,
                {TRAINING_LOGS_ROOT_ENV: str(logs_root)},
            ):
                execute = self.run_worker(worker_payload(), progress_path)

        artifacts = execute.call_args.kwargs["artifacts"]
        self.assertEqual(artifacts.root, logs_root)

    def test_worker_rejects_oversized_collections_before_started_event(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            progress_path = Path(tmp) / "progress.jsonl"
            payload = worker_payload()
            payload["monitors"] = [
                f"monitor-{index}" for index in range(MAX_TRAINING_MONITORS + 1)
            ]
            stderr = io.StringIO()
            with (
                contextlib.redirect_stderr(stderr),
                self.assertRaises(SystemExit),
            ):
                self.run_worker(payload, progress_path)
            events = read_jsonl(progress_path)

        self.assertEqual([event["type"] for event in events], ["error"])
        self.assertIn("at most 100 monitors", events[0]["error"])

    def test_worker_writes_started_then_error_when_plan_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            progress_path = Path(tmp) / "progress.jsonl"
            payload = {**worker_payload(), "runPlan": None}
            stderr = io.StringIO()
            with (
                contextlib.redirect_stderr(stderr),
                self.assertRaises(SystemExit) as raised,
                patch("workbench.backend.training_jobs.worker.execute_runs") as execute,
            ):
                self.run_worker(payload, progress_path)

            events = read_jsonl(progress_path)

        self.assertEqual(raised.exception.code, 1)
        execute.assert_not_called()
        self.assertEqual([event["type"] for event in events], ["started", "error"])
        self.assertIn("non-empty materialized run plan", events[-1]["error"])

    def test_worker_writes_error_event_when_execution_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            progress_path = Path(tmp) / "progress.jsonl"
            stderr = io.StringIO()
            with (
                contextlib.redirect_stderr(stderr),
                self.assertRaises(SystemExit) as raised,
            ):
                self.run_worker(
                    worker_payload(),
                    progress_path,
                    execute_side_effect=RuntimeError("training failed"),
                )
            events = read_jsonl(progress_path)

        self.assertEqual(raised.exception.code, 1)
        self.assertEqual([event["type"] for event in events], ["started", "error"])
        self.assertEqual(events[-1]["status"], "failed")
        self.assertEqual(events[-1]["error"], "training failed")
        self.assertEqual(events[-1]["experimentTask"], "image-classification")
        self.assertIn("RuntimeError: training failed", events[-1]["traceback"])

    def test_neuron_cluster_growth_callback_caps_coordinate_payloads(self) -> None:
        events: list[dict[str, object]] = []
        callback = NeuronClusterGrowthCallback(events.append)
        initial_names = {"neuron_1_1_1"}
        all_names = {
            *initial_names,
            *(f"neuron_{index}_1_1" for index in range(2, 128)),
        }
        cluster = type(
            "Cluster",
            (),
            {
                "cluster": {name: object() for name in all_names},
                "x_axis_total_neurons": 200,
                "y_axis_total_neurons": 1,
                "z_axis_total_neurons": 1,
            },
        )()
        callback._clusters = [("cluster", cluster)]
        callback._known_names = {"cluster": set(initial_names)}
        trainer = type("Trainer", (), {"current_epoch": 3, "global_step": 42})()

        callback.on_train_batch_end(trainer, object(), None, None, 0)

        self.assertEqual(len(events), 1)
        event = events[0]
        self.assertEqual(event["type"], "neurons_added")
        self.assertEqual(event["coordinateCount"], len(all_names) - 1)
        self.assertEqual(len(event["coordinates"]), 100)
        self.assertTrue(event["coordinatesTruncated"])


if __name__ == "__main__":
    unittest.main()
