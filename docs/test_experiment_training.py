import json
import os
import tempfile
import unittest
from enum import Enum
from pathlib import Path

import emperor.experiments.base as experiments_base
from emperor.experiments.base import (
    ExperimentBase,
    ExperimentPresetsBase,
    PresetDefinition,
    _result_metrics_payload,
)
from emperor.experiments.monitors import MonitorOption
from emperor.experiments.progress import JsonlTrainingProgressCallback
from lightning.pytorch.callbacks import Callback
from models.parser import resolve_dataset_names


class FakeMetric:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value


class FakeConfig:
    batch_size = 2

    def get_custom_parameters(self):
        return {}


class FakeDatasetA:
    flattened_input_dim = 4
    num_classes = 2

    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.num_workers = 4


class FakeDatasetB:
    flattened_input_dim = 8
    num_classes = 3

    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.num_workers = 4


class FakeOption(Enum):
    BASELINE = "baseline"
    GATING = "gating"
    HALTING = "halting"


class FakePresetGenerator(ExperimentPresetsBase):
    def __init__(self):
        super().__init__(
            {
                preset: PresetDefinition(
                    preset_values={},
                    description=preset.name.lower(),
                )
                for preset in FakeOption
            }
        )
        self.seen_datasets = []
        self.seen_presets = []

    def get_config(
        self,
        model_config_preset,
        dataset,
        search_mode=None,
        log_folder=None,
        search_keys=None,
        config_overrides=None,
        search_overrides=None,
    ):
        self.seen_presets.append(model_config_preset.name)
        self.seen_datasets.append(dataset.__name__)
        return [FakeConfig()]


class FakeModel:
    def __init__(self, config):
        self.config = config


class FakeLogger:
    def __init__(self, save_dir, name):
        self.log_dir = str(Path(save_dir) / name)


class FakeTrainer:
    instances = []

    def __init__(self, max_epochs, logger, callbacks, **kwargs):
        self.max_epochs = max_epochs
        self.logger = logger
        self.callbacks = callbacks
        self.callback_metrics = {"validation_accuracy": FakeMetric(0.75)}
        self.current_epoch = 1
        self.global_step = 2
        type(self).instances.append(self)

    def fit(self, model, datamodule):
        self.model = model
        self.fit_datamodule = datamodule

    def test(self, model, datamodule):
        self.test_datamodule = datamodule


class FakeMonitorCallback(Callback):
    pass


class CaptureTrainingCallback(Callback):
    def __init__(self):
        self.contexts = []
        self.events = []

    def set_run_context(
        self,
        dataset,
        log_dir=None,
        preset=None,
        preset_key=None,
        run_id=None,
        run_index=None,
        run_total=None,
        total_epochs=None,
    ):
        self.contexts.append(
            {
                "dataset": dataset,
                "logDir": log_dir,
                "preset": preset,
                "presetKey": preset_key,
                "runId": run_id,
                "runIndex": run_index,
                "runTotal": run_total,
                "totalEpochs": total_epochs,
            }
        )

    def write_event(self, event):
        self.events.append(dict(event))


class FakeExperiment(ExperimentBase):
    def _num_epochs(self):
        return 1

    def _dataset_options(self):
        return [FakeDatasetA, FakeDatasetB]

    def _model_type(self):
        return FakeModel

    def _preset_generator_instance(self):
        return FakePresetGenerator()

    def _experiment_preset_enum(self):
        return FakeOption

    def _load_trainer_config(self, config_overrides=None):
        return {"trainer_args": {}, "callbacks": []}


class TestExperimentTraining(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.original_cwd = os.getcwd()
        os.chdir(self.tempdir.name)
        FakeTrainer.instances.clear()
        self.original_trainer = experiments_base.Trainer
        self.original_logger = experiments_base.TensorBoardLogger
        experiments_base.Trainer = FakeTrainer
        experiments_base.TensorBoardLogger = FakeLogger

    def tearDown(self):
        experiments_base.Trainer = self.original_trainer
        experiments_base.TensorBoardLogger = self.original_logger
        os.chdir(self.original_cwd)
        self.tempdir.cleanup()

    def test_selected_datasets_limit_training_loop(self):
        experiment = FakeExperiment(FakeOption.BASELINE)

        experiment.train_model(selected_datasets=[FakeDatasetB])

        self.assertEqual(
            experiment.preset_generator.seen_datasets,
            ["FakeDatasetB"],
        )

    def test_selected_presets_limit_training_loop_in_order(self):
        experiment = FakeExperiment()

        experiment.train_model(
            selected_datasets=[FakeDatasetA],
            selected_presets=[FakeOption.GATING, FakeOption.BASELINE],
        )

        self.assertEqual(
            experiment.preset_generator.seen_presets,
            ["GATING", "BASELINE"],
        )

    def test_data_num_workers_override_updates_datamodule(self):
        experiment = FakeExperiment(FakeOption.BASELINE)

        experiment.train_model(
            selected_datasets=[FakeDatasetA],
            config_overrides={"data_num_workers": 0},
        )

        self.assertEqual(FakeTrainer.instances[0].fit_datamodule.num_workers, 0)

    def test_run_test_after_fit_override_skips_test_phase(self):
        experiment = FakeExperiment(FakeOption.BASELINE)

        experiment.train_model(
            selected_datasets=[FakeDatasetA],
            config_overrides={"run_test_after_fit": False},
        )

        self.assertFalse(hasattr(FakeTrainer.instances[0], "test_datamodule"))

    def test_run_test_after_fit_defaults_to_enabled(self):
        experiment = FakeExperiment(FakeOption.BASELINE)

        experiment.train_model(selected_datasets=[FakeDatasetA])

        self.assertIs(
            FakeTrainer.instances[0].test_datamodule,
            FakeTrainer.instances[0].fit_datamodule,
        )

    def test_train_model_instantiates_model_with_generated_config(self):
        experiment = FakeExperiment(FakeOption.BASELINE)

        experiment.train_model(selected_datasets=[FakeDatasetA])

        self.assertIsInstance(FakeTrainer.instances[0].model.config, FakeConfig)

    def test_materialized_run_sets_progress_context_and_events(self):
        experiment = FakeExperiment()
        callback = CaptureTrainingCallback()

        experiment.train_model(
            callbacks=[callback],
            materialized_runs=[
                {
                    "id": "run-from-plan",
                    "index": 7,
                    "run_total": 9,
                    "preset": FakeOption.HALTING,
                    "dataset_type": FakeDatasetB,
                    "config_overrides": {"num_epochs": 3},
                }
            ],
        )

        self.assertEqual(experiment.preset_generator.seen_presets, ["HALTING"])
        self.assertEqual(experiment.preset_generator.seen_datasets, ["FakeDatasetB"])
        self.assertEqual(len(callback.contexts), 1)
        self.assertEqual(
            callback.contexts[0],
            {
                "dataset": "FakeDatasetB",
                "logDir": callback.contexts[0]["logDir"],
                "preset": "halting",
                "presetKey": "HALTING",
                "runId": "run-from-plan",
                "runIndex": 7,
                "runTotal": 9,
                "totalEpochs": 3,
            },
        )
        self.assertTrue(
            callback.contexts[0]["logDir"].startswith(
                f"logs/{experiment._public_model_id()}/HALTING/FakeDatasetB/default_"
            )
        )
        self.assertEqual(
            [event["type"] for event in callback.events],
            ["dataset_started", "dataset_completed"],
        )

        started, completed = callback.events
        self.assertEqual(started["status"], "running")
        self.assertEqual(started["dataset"], "FakeDatasetB")
        self.assertEqual(started["preset"], "halting")
        self.assertEqual(started["presetKey"], "HALTING")
        self.assertEqual(started["runId"], "run-from-plan")
        self.assertEqual(started["runIndex"], 7)
        self.assertEqual(started["runTotal"], 9)
        self.assertEqual(started["totalEpochs"], 3)
        self.assertEqual(started["params"], {})
        self.assertEqual(completed["metrics"], {"validation_accuracy": 0.75})

    def test_train_model_rejects_path_like_log_folder(self):
        experiment = FakeExperiment(FakeOption.BASELINE)

        with self.assertRaises(ValueError):
            experiment.train_model(
                log_folder="../escape",
                selected_datasets=[FakeDatasetA],
            )

    def test_update_best_results_merges_existing_summary_before_writing(self):
        experiment = FakeExperiment(FakeOption.BASELINE)
        summary_path = (
            Path("logs")
            / "unit_results"
            / experiment._public_model_id()
            / "best_results.json"
        )
        summary_path.parent.mkdir(parents=True)
        existing = {
            "FakeDatasetA": [
                {
                    "dataset": "FakeDatasetA",
                    "params": {"hidden_dim": 32},
                    "metrics": {"validation_accuracy": 0.8},
                    "rank": 1,
                }
            ],
            "FakeDatasetB": [
                {
                    "dataset": "FakeDatasetB",
                    "params": {"hidden_dim": 64},
                    "metrics": {"validation_accuracy": 0.7},
                    "rank": 1,
                }
            ],
        }
        summary_path.write_text(json.dumps(existing), encoding="utf-8")
        top5 = {}
        result = {
            "dataset": "FakeDatasetA",
            "params": {"hidden_dim": 128},
            "metrics": {"validation_accuracy": 0.9},
        }

        experiment._update_best_results(result, top5, "unit_results")

        written = json.loads(summary_path.read_text(encoding="utf-8"))
        self.assertEqual(written["FakeDatasetA"][0]["params"], {"hidden_dim": 128})
        self.assertEqual(written["FakeDatasetA"][0]["rank"], 1)
        self.assertEqual(written["FakeDatasetA"][1]["params"], {"hidden_dim": 32})
        self.assertEqual(written["FakeDatasetA"][1]["rank"], 2)
        self.assertEqual(written["FakeDatasetB"], existing["FakeDatasetB"])
        self.assertEqual(top5, written)

    def test_progress_callback_emits_jsonl_events(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "progress.jsonl"
            callback = JsonlTrainingProgressCallback(path)
            trainer = FakeTrainer(
                max_epochs=1,
                logger=FakeLogger("logs", "test"),
                callbacks=[],
            )
            callback.set_run_context(
                "FakeDatasetB",
                "logs/test",
                "baseline",
                "BASELINE",
            )

            callback.on_train_batch_end(trainer, None, None, None, 3)

            event = json.loads(path.read_text().splitlines()[0])
            self.assertEqual(event["status"], "running")
            self.assertEqual(event["dataset"], "FakeDatasetB")
            self.assertEqual(event["preset"], "baseline")
            self.assertEqual(event["presetKey"], "BASELINE")
            self.assertEqual(event["epoch"], 1)
            self.assertEqual(event["step"], 2)
            self.assertEqual(event["metrics"]["validation_accuracy"], 0.75)

    def test_progress_callback_can_throttle_step_events(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "progress.jsonl"
            callback = JsonlTrainingProgressCallback(path, step_interval=5)
            trainer = FakeTrainer(
                max_epochs=1,
                logger=FakeLogger("logs", "test"),
                callbacks=[],
            )
            callback.set_run_context(
                "FakeDatasetB",
                "logs/test",
                "baseline",
                "BASELINE",
            )

            callback.on_train_batch_end(trainer, None, None, None, 3)
            self.assertFalse(path.exists())

            trainer.global_step = 5
            callback.on_train_batch_end(trainer, None, None, None, 4)

            events = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual(len(events), 1)
            self.assertEqual(events[0]["type"], "step")
            self.assertEqual(events[0]["step"], 5)

    def test_progress_callback_filters_high_cardinality_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "progress.jsonl"
            callback = JsonlTrainingProgressCallback(
                path,
                metric_key_limit=3,
            )
            trainer = FakeTrainer(
                max_epochs=1,
                logger=FakeLogger("logs", "test"),
                callbacks=[],
            )
            trainer.callback_metrics = {
                "validation_accuracy": FakeMetric(0.75),
                "train/confusion_matrix/0/0": FakeMetric(12),
                "validation/per_class/0/accuracy": FakeMetric(0.5),
                "train/loss": FakeMetric(0.25),
                "grad_norm": FakeMetric(1.2),
                "extra_metric": FakeMetric(99),
            }

            callback.on_train_batch_end(trainer, None, None, None, 3)

            event = json.loads(path.read_text().splitlines()[0])
            self.assertEqual(
                event["metrics"],
                {
                    "validation_accuracy": 0.75,
                    "train/loss": 0.25,
                    "grad_norm": 1.2,
                },
            )
            self.assertEqual(event["metricsOriginalCount"], 6)
            self.assertEqual(event["metricsDroppedCount"], 3)

    def test_result_metrics_filter_high_cardinality_metrics(self):
        payload = _result_metrics_payload(
            {
                "validation_accuracy": FakeMetric(0.75),
                "train/confusion_matrix/0/0": FakeMetric(12),
                "validation/per_class/0/accuracy": FakeMetric(0.5),
                "train/loss": FakeMetric(0.25),
            }
        )

        self.assertEqual(
            payload,
            {
                "metrics": {
                    "validation_accuracy": 0.75,
                    "train/loss": 0.25,
                },
                "metricsOriginalCount": 4,
                "metricsDroppedCount": 2,
            },
        )

    def test_progress_callback_enforces_event_byte_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "progress.jsonl"
            callback = JsonlTrainingProgressCallback(
                path,
                metric_key_limit=100,
                event_byte_limit=800,
            )
            trainer = FakeTrainer(
                max_epochs=1,
                logger=FakeLogger("logs", "test"),
                callbacks=[],
            )
            trainer.callback_metrics = {
                "validation_accuracy": FakeMetric(0.75),
                **{f"metric_{index}": "x" * 120 for index in range(20)},
            }

            callback.on_train_batch_end(trainer, None, None, None, 3)

            raw_line = path.read_text().splitlines()[0]
            event = json.loads(raw_line)
            self.assertLessEqual(len(raw_line.encode("utf-8")), 800)
            self.assertEqual(event["metrics"]["validation_accuracy"], 0.75)
            self.assertGreater(event["metricsDroppedCount"], 0)

    def test_monitor_option_build_callback_returns_fresh_instances(self):
        option = MonitorOption(
            name="fake",
            label="Fake monitor",
            description="Test monitor.",
            kinds=["scalar"],
            callback_factory=FakeMonitorCallback,
        )

        first = option.build_callback()
        second = option.build_callback()

        self.assertIsInstance(first, FakeMonitorCallback)
        self.assertIsInstance(second, FakeMonitorCallback)
        self.assertIsNot(first, second)

    def test_resolve_dataset_names_handles_known_unknown_and_multiple(self):
        resolved = resolve_dataset_names(
            [FakeDatasetA, FakeDatasetB],
            ["fake-dataset-b", "FakeDatasetB", "FakeDatasetA", "fake-dataset-a"],
        )
        self.assertEqual(resolved, [FakeDatasetB, FakeDatasetA])

        with self.assertRaises(ValueError) as error:
            resolve_dataset_names([FakeDatasetA], ["UnknownDataset"])
        self.assertEqual(
            str(error.exception),
            "Unknown --datasets: ['UnknownDataset']. Valid datasets: fake-dataset-a",
        )
