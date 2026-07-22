import json
import os
import tempfile
import unittest
from enum import Enum
from pathlib import Path
from types import ModuleType

from lightning.pytorch.callbacks import Callback

import model_runtime.runs.experiment as experiments_base
from emperor.experiments import ExperimentTask
from emperor.monitoring import MonitorOption
from model_runtime.packages import (
    ExperimentPresetsBase,
    ModelIdentity,
    ModelMetadata,
    ModelPackage,
    PresetDefinition,
)
from model_runtime.runs import ExperimentBase, JsonlTrainingProgressCallback
from model_runtime.runs.experiment import _result_metrics_payload


class FakeMetric:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value


class FakeConfig:
    batch_size = 2


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
        *,
        config_overrides=None,
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


class FakePackageAdapter:
    def __init__(self) -> None:
        identity = ModelIdentity("test", "fake")
        runtime_defaults = ModuleType("tests.fake_runtime_defaults")
        runtime_defaults.DATA_NUM_WORKERS = 4
        runtime_defaults.RUN_TEST_AFTER_FIT = True
        runtime_defaults.SEED = None
        dataset_metadata = ModuleType("tests.fake_dataset_metadata")
        dataset_metadata.DEFAULT_EXPERIMENT_TASK = ExperimentTask.IMAGE_CLASSIFICATION
        dataset_metadata.DATASET_OPTIONS_BY_TASK = {
            ExperimentTask.IMAGE_CLASSIFICATION: [FakeDatasetA, FakeDatasetB]
        }
        monitor_metadata = ModuleType("tests.fake_monitor_metadata")
        monitor_metadata.MONITOR_OPTIONS = []
        search_metadata = ModuleType("tests.fake_search_metadata")
        self.metadata = ModelMetadata(
            identity=identity,
            runtime_defaults=runtime_defaults,
            dataset_options=dataset_metadata,
            monitor_options_source=monitor_metadata,
            search_space=search_metadata,
        )

    def load_metadata(self):
        return self.metadata

    def load_runtime_options_type(self):
        return object

    def bind_runtime_defaults(self, values):
        return object()

    def load_preset_type(self):
        return FakeOption

    def load_presets(self):
        return FakePresetGenerator()

    def build_configuration(self, presets, preset, dataset, **kwargs):
        return presets.get_config(preset, dataset, **kwargs)[0]

    def build_model(self, configuration):
        return FakeModel(configuration)

    def build_experiment(self, preset, **kwargs):
        raise AssertionError("The fake package does not construct experiments.")


def fake_model_package() -> ModelPackage:
    adapter = FakePackageAdapter()
    return ModelPackage(adapter.metadata.identity, adapter)


class FakeExperiment(ExperimentBase):
    def _num_epochs(self):
        return 1

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
        self.model_package = fake_model_package()

    def tearDown(self):
        experiments_base.Trainer = self.original_trainer
        experiments_base.TensorBoardLogger = self.original_logger
        os.chdir(self.original_cwd)
        self.tempdir.cleanup()

    def _execute_run(
        self,
        experiment,
        *,
        dataset_type=FakeDatasetA,
        preset=FakeOption.BASELINE,
        config_overrides=None,
        parameters=None,
        callbacks=None,
        log_folder=None,
        run_id="run-0001",
        run_index=1,
        run_total=1,
    ):
        training_run = experiment.materialize_training_runs(
            [
                {
                    "id": run_id,
                    "index": run_index,
                    "run_total": run_total,
                    "preset": preset,
                    "dataset_type": dataset_type,
                    "parameters": parameters or {},
                    "config_overrides": config_overrides or {},
                }
            ],
            log_folder,
        )[0]
        return experiment.execute_training_run(
            training_run,
            log_folder=log_folder,
            callbacks=callbacks or [],
            best_results=experiment.load_best_results(log_folder),
        )

    def test_data_num_workers_override_updates_datamodule(self):
        experiment = FakeExperiment(
            FakeOption.BASELINE, model_package=self.model_package
        )

        self._execute_run(
            experiment,
            config_overrides={"data_num_workers": 0},
        )

        self.assertEqual(FakeTrainer.instances[0].fit_datamodule.num_workers, 0)

    def test_run_test_after_fit_override_skips_test_phase(self):
        experiment = FakeExperiment(
            FakeOption.BASELINE, model_package=self.model_package
        )

        self._execute_run(
            experiment,
            config_overrides={"run_test_after_fit": False},
        )

        self.assertFalse(hasattr(FakeTrainer.instances[0], "test_datamodule"))

    def test_run_test_after_fit_defaults_to_enabled(self):
        experiment = FakeExperiment(
            FakeOption.BASELINE, model_package=self.model_package
        )

        self._execute_run(experiment)

        self.assertIs(
            FakeTrainer.instances[0].test_datamodule,
            FakeTrainer.instances[0].fit_datamodule,
        )

    def test_run_execution_instantiates_model_with_materialized_config(self):
        experiment = FakeExperiment(
            FakeOption.BASELINE, model_package=self.model_package
        )

        self._execute_run(experiment)

        self.assertIsInstance(FakeTrainer.instances[0].model.config, FakeConfig)

    def test_materialized_run_sets_progress_context_and_events(self):
        experiment = FakeExperiment(model_package=self.model_package)
        callback = CaptureTrainingCallback()

        self._execute_run(
            experiment,
            callbacks=[callback],
            dataset_type=FakeDatasetB,
            preset=FakeOption.HALTING,
            parameters={"NUM_EPOCHS": 3},
            config_overrides={"num_epochs": 3},
            run_id="run-from-plan",
            run_index=7,
            run_total=9,
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
                f"logs/{experiment.model_package.catalog_key}/HALTING/FakeDatasetB/"
            )
        )
        self.assertNotIn("/default_", callback.contexts[0]["logDir"])
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
        self.assertEqual(started["params"], {"NUM_EPOCHS": 3})
        self.assertEqual(completed["metrics"], {"validation_accuracy": 0.75})

    def test_run_execution_rejects_path_like_log_folder(self):
        experiment = FakeExperiment(
            FakeOption.BASELINE, model_package=self.model_package
        )

        with self.assertRaises(ValueError):
            self._execute_run(experiment, log_folder="../escape")

    def test_update_best_results_merges_existing_summary_before_writing(self):
        experiment = FakeExperiment(
            FakeOption.BASELINE, model_package=self.model_package
        )
        summary_path = (
            Path("logs")
            / "unit_results"
            / experiment.model_package.catalog_key
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

    def test_causal_language_model_results_rank_lowest_validation_loss_first(self):
        experiment = FakeExperiment(
            FakeOption.BASELINE, model_package=self.model_package
        )
        experiment.experiment_task = ExperimentTask.CAUSAL_LANGUAGE_MODELING
        low_loss = {"metrics": {"validation/loss": 0.25}}
        high_loss = {"metrics": {"validation/loss": 0.75}}

        self.assertGreater(
            experiment._result_ranking_score(low_loss),
            experiment._result_ranking_score(high_loss),
        )

    def test_causal_language_model_dataset_receives_configured_sequence_length(self):
        experiment = FakeExperiment(
            FakeOption.BASELINE, model_package=self.model_package
        )
        training_run = type(
            "TrainingRun",
            (),
            {
                "experiment_task": ExperimentTask.CAUSAL_LANGUAGE_MODELING,
                "config": type(
                    "Config",
                    (),
                    {"batch_size": 7, "sequence_length": 19},
                )(),
            },
        )()

        self.assertEqual(
            experiment._dataset_constructor_kwargs(training_run),
            {"batch_size": 7, "sequence_length": 19},
        )

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
            callback_factory=lambda _settings: FakeMonitorCallback(),
        )

        first = option.build_callback()
        second = option.build_callback()

        self.assertIsInstance(first, FakeMonitorCallback)
        self.assertIsInstance(second, FakeMonitorCallback)
        self.assertIsNot(first, second)
