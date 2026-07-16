from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from emperor_workbench.run_plans import (
    RunPlanProgressProjector,
    TrainingRunPlanView,
    TrainingRunView,
)
from emperor_workbench.training_jobs._progress_projection import (
    TrainingLiveProjectionCache,
)
from emperor_workbench.training_jobs._progress_store import (
    TrainingProgressSnapshot,
    TrainingProgressStore,
)
from emperor_workbench.training_jobs._records import TrainingJobRecord
from emperor_workbench.training_jobs._snapshot import TrainingJobProjector
from tests.support.training_jobs import training_job_payload


def _run(run_id: str, index: int, dataset: str) -> TrainingRunView:
    return TrainingRunView(
        id=run_id,
        index=index,
        status="Pending",
        preset="baseline",
        dataset=dataset,
        experiment_task="classification",
        changes=[],
        overrides={},
        command="train",
        total_epochs=2,
    )


def _job(root: Path) -> TrainingJobRecord:
    runs = [_run("run-1", 1, "Mnist"), _run("run-2", 2, "Cifar10")]
    return TrainingJobRecord(
        id="job-1",
        model="linears/linear",
        preset="baseline",
        presets=["baseline"],
        experiment_task="classification",
        datasets=["Mnist", "Cifar10"],
        overrides={},
        search=None,
        planned_run_count=2,
        run_plan=TrainingRunPlanView(
            model="linears/linear",
            preset="baseline",
            presets=["baseline"],
            experiment_task="classification",
            datasets=["Mnist", "Cifar10"],
            overrides={},
            search=None,
            log_folder="projection-test",
            is_random_search=False,
            runs=runs,
            summary=RunPlanProgressProjector.summarize(runs),
        ),
        monitors=["linear"],
        log_folder="projection-test",
        observed_command=["train"],
        root=root,
        pid=123,
    )


def _events() -> list[dict[str, Any]]:
    return [
        {"type": "job_started", "status": "running"},
        {
            "type": "dataset_started",
            "status": "running",
            "runId": "run-1",
            "dataset": "Mnist",
            "preset": "baseline",
            "logDir": "/logs/projection-test/run-1",
        },
        {
            "type": "epoch_started",
            "status": "running",
            "runId": "run-1",
            "dataset": "Mnist",
            "preset": "baseline",
            "epoch": 0,
            "metrics": {"train/loss": 0.5},
        },
        {
            "type": "cluster_initialized",
            "node": "main_model.0",
            "count": 2,
            "capacity": [2, 3, 4],
        },
        {
            "type": "neuron_added",
            "node": "main_model.0",
            "count": 3,
            "coord": [1, 2, 3],
            "step": 4,
            "epoch": 0,
        },
        {
            "type": "neurons_added",
            "node": "main_model.0",
            "count": 5,
            "coordinates": [[2, 3, 4], ["invalid", 0, 0]],
            "coordinateCount": 2,
            "step": 5,
            "epoch": 1,
        },
        {
            "type": "dataset_completed",
            "status": "completed",
            "runId": "run-1",
            "dataset": "Mnist",
            "preset": "baseline",
            "epoch": 1,
            "metrics": {"test/accuracy": 0.9},
            "logDir": "/logs/projection-test/run-1",
        },
        {
            "type": "dataset_started",
            "status": "running",
            "runId": "run-2",
            "dataset": "Cifar10",
            "preset": "baseline",
            "logDir": "/logs/projection-test/run-2",
        },
        {
            "type": "error",
            "status": "failed",
            "runId": "run-2",
            "dataset": "Cifar10",
            "preset": "baseline",
            "error": "boom",
            "traceback": "trace",
        },
    ]


def _snapshot(
    events: list[dict[str, Any]],
    *,
    new_events: list[dict[str, Any]] | None = None,
    reset: bool = False,
) -> TrainingProgressSnapshot:
    return TrainingProgressSnapshot(
        events=events,
        new_events=events if new_events is None else new_events,
        total_count=len(events),
        reset=reset,
    )


class TrainingProjectionParityTests(unittest.TestCase):
    def test_projection_cache_bounds_retained_job_reducers(self) -> None:
        with TemporaryDirectory() as tmp:
            cache = TrainingLiveProjectionCache(max_cached_jobs=2)
            for index in range(3):
                job = _job(Path(tmp) / f"job-{index}")
                job.id = f"job-{index}"
                cache.project(job, _snapshot([], reset=True))

        self.assertEqual(cache.cached_job_count, 2)
        self.assertNotIn("job-0", cache._cache)

    def _full_payload(
        self,
        projector: TrainingJobProjector,
        job: TrainingJobRecord,
        events: list[dict[str, Any]],
    ) -> dict[str, Any]:
        replay = TrainingLiveProjectionCache().project(
            job,
            _snapshot(events, reset=True),
        )
        return training_job_payload(projector.project_snapshot(job, replay))

    def _live_payload(
        self,
        projector: TrainingJobProjector,
        cache: TrainingLiveProjectionCache,
        job: TrainingJobRecord,
        snapshot: TrainingProgressSnapshot,
    ) -> dict[str, Any]:
        projection = cache.project(job, snapshot)
        return training_job_payload(projector.project_snapshot(job, projection))

    def test_replay_and_incremental_fold_match_after_every_event_prefix(self) -> None:
        with TemporaryDirectory() as tmp:
            job = _job(Path(tmp) / "job-1")
            projector = TrainingJobProjector()
            cache = TrainingLiveProjectionCache()
            events = _events()

            for event_count in range(len(events) + 1):
                with self.subTest(event_count=event_count):
                    prefix = events[:event_count]
                    live = self._live_payload(
                        projector,
                        cache,
                        job,
                        _snapshot(
                            prefix,
                            new_events=(prefix[-1:] if prefix else []),
                            reset=event_count == 0,
                        ),
                    )
                    self.assertEqual(
                        live,
                        self._full_payload(projector, job, prefix),
                    )

    def test_replay_and_incremental_finalization_match_for_every_job_status(
        self,
    ) -> None:
        with TemporaryDirectory() as tmp:
            events = _events()
            for status in (
                "queued",
                "running",
                "unknown",
                "completed",
                "failed",
                "cancelled",
            ):
                with self.subTest(status=status):
                    job = _job(Path(tmp) / status)
                    job.status = status
                    projector = TrainingJobProjector()
                    live = self._live_payload(
                        projector,
                        TrainingLiveProjectionCache(),
                        job,
                        _snapshot(events, reset=True),
                    )
                    self.assertEqual(
                        live,
                        self._full_payload(projector, job, events),
                    )

    def test_live_projection_preserves_full_counts_with_last_100_event_tail(
        self,
    ) -> None:
        with TemporaryDirectory() as tmp:
            job = _job(Path(tmp) / "job-1")
            projector = TrainingJobProjector()
            cache = TrainingLiveProjectionCache()
            events = [
                {
                    "type": "step",
                    "runId": "run-1",
                    "dataset": "Mnist",
                    "preset": "baseline",
                    "step": index,
                    "metrics": {"train/loss": float(index)},
                }
                for index in range(125)
            ]

            full = self._full_payload(projector, job, events)
            live = self._live_payload(
                projector,
                cache,
                job,
                _snapshot(events, reset=True),
            )

        self.assertEqual(live["events"], events[-100:])
        self.assertEqual(live["eventCount"], 125)
        self.assertEqual(live["eventCounts"], {"step": 125})
        self.assertTrue(live["eventsTruncated"])
        for key in full.keys() - {"events", "eventsTruncated"}:
            with self.subTest(key=key):
                self.assertEqual(live[key], full[key])

    def test_reset_rewrite_count_regression_eviction_and_fresh_replay_rebuild(
        self,
    ) -> None:
        with TemporaryDirectory() as tmp:
            job = _job(Path(tmp) / "job-1")
            projector = TrainingJobProjector()
            cache = TrainingLiveProjectionCache()
            initial = _events()[:5]
            rewritten = [copy.deepcopy(event) for event in _events()[6:9]]

            self._live_payload(
                projector,
                cache,
                job,
                _snapshot(initial, reset=True),
            )

            for label, snapshot in (
                ("same-count rewrite", _snapshot(rewritten, reset=True)),
                ("count regression", _snapshot(rewritten[:1], reset=False)),
            ):
                with self.subTest(label=label):
                    live = self._live_payload(projector, cache, job, snapshot)
                    self.assertEqual(
                        live,
                        self._full_payload(projector, job, snapshot.events),
                    )

            cache.evict(job.id)
            replay = self._live_payload(
                projector,
                cache,
                job,
                _snapshot(rewritten, new_events=[], reset=False),
            )
            fresh_replay = self._live_payload(
                projector,
                TrainingLiveProjectionCache(),
                job,
                _snapshot(rewritten, new_events=[], reset=False),
            )

        expected = self._full_payload(projector, job, rewritten)
        self.assertEqual(replay, expected)
        self.assertEqual(fresh_replay, expected)

    def test_projection_uses_full_snapshot_when_new_events_cache_is_behind(
        self,
    ) -> None:
        with TemporaryDirectory() as tmp:
            job = _job(Path(tmp) / "job-1")
            projector = TrainingJobProjector()
            cache = TrainingLiveProjectionCache()
            events = _events()[:2]

            self._live_payload(
                projector,
                cache,
                job,
                _snapshot(events[:1], reset=True),
            )
            live = self._live_payload(
                projector,
                cache,
                job,
                _snapshot(events, new_events=[], reset=False),
            )

        self.assertEqual(live, self._full_payload(projector, job, events))

    def test_progress_file_same_count_rewrite_resets_projection_state(self) -> None:
        with TemporaryDirectory() as tmp:
            job = _job(Path(tmp) / "job-1")
            job.root.mkdir(parents=True)
            store = TrainingProgressStore()
            cache = TrainingLiveProjectionCache()
            initial = _events()[3:5]
            rewritten = _events()[6:8]
            job.progress_path.write_text(
                "\n".join(json.dumps(event) for event in initial) + "\n",
                encoding="utf-8",
            )
            first = store.read_snapshot(job)
            cache.project(job, first)
            cursor = cache.cursor(job.id)

            job.progress_path.write_text(
                "\n".join(json.dumps(event) for event in rewritten) + "\n",
                encoding="utf-8",
            )
            second = store.read_snapshot(job, cursor=cursor)
            rebuilt = cache.project(job, second)
            fresh = TrainingLiveProjectionCache().project(
                job,
                second,
            )

        self.assertTrue(second.reset)
        self.assertEqual(rebuilt, fresh)
        self.assertNotIn("cluster_initialized", rebuilt.event_counts)
        self.assertEqual(
            rebuilt.event_counts,
            {"dataset_completed": 1, "dataset_started": 1},
        )

    def test_projection_catches_up_after_another_reader_consumes_new_events(
        self,
    ) -> None:
        with TemporaryDirectory() as tmp:
            job = _job(Path(tmp) / "job-1")
            job.root.mkdir(parents=True)
            store = TrainingProgressStore()
            cache = TrainingLiveProjectionCache()
            events = _events()[:2]
            job.progress_path.write_text(
                json.dumps(events[0]) + "\n",
                encoding="utf-8",
            )
            cache.project(job, store.read_snapshot(job))
            projection_cursor = cache.cursor(job.id)

            with job.progress_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(events[1]) + "\n")
            consumed_elsewhere = store.read_summary(job)
            caught_up_snapshot = store.read_snapshot(
                job,
                cursor=projection_cursor,
            )
            projection = cache.project(
                job,
                caught_up_snapshot,
            )

        self.assertEqual(consumed_elsewhere.new_events, [])
        self.assertEqual(caught_up_snapshot.new_events, [events[1]])
        self.assertEqual(projection.event_count, 2)
        self.assertEqual(projection.latest_event, events[1])
        self.assertEqual(
            projection.event_counts,
            {"job_started": 1, "dataset_started": 1},
        )
        self.assertEqual(projection.run_plan.runs[0].status, "Running")


if __name__ == "__main__":
    unittest.main()
